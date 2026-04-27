"""
Engine for efficient inference.

KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API:
- Tensors are (B, T, H, D) not (B, H, T, D)
- FA3 updates the cache in-place during flash_attn_with_kvcache
- Position tracked per batch element via cache_seqlens tensor

Engine supports batched generation with variable-length prompts:
- Each prompt is prefilled individually (batch=1), avoiding padding/masking
- After prefill, KV caches are combined and decode proceeds in a single batch
- For equal-length prompts, batched prefill is used (single forward pass)
"""

import torch
import torch.nn.functional as F

from nanoproof.common import COMPUTE_DTYPE, maybe_dump_memory_snapshot


class KVCache:
    """
    KV Cache for Flash Attention 3 (and SDPA fallback).
    Pre-allocated tensors with per-batch-element position tracking.
    """

    def __init__(
        self,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        num_layers,
        device="cpu",
        dtype=None,
    ):
        if dtype is None:
            dtype = COMPUTE_DTYPE
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(
            num_layers,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.zeros(
            num_layers,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for smear (set by model forward pass)
        self.prev_embedding = None

    def reset(self):
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert (
            self.n_layers == other.n_layers
            and self.n_heads == other.n_heads
            and self.head_dim == other.head_dim
        )
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(
                self.batch_size, -1, -1
            ).clone()


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


# -----------------------------------------------------------------------------


class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.completed = False


class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _kv_model_kwargs(self):
        m = self.model.config
        return {
            "num_heads": m.n_kv_head,
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer,
        }

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        num_samples=1,
        max_tokens=None,
        min_tokens=None,
        temperature=1.0,
        top_k=None,
        seed=42,
        return_logits=False,
    ):
        """
        Generate tokens from prompt(s). Accepts either list[int] (single prompt) or
        list[list[int]] (batched prompts).

        For variable-length batched prompts, each prompt is prefilled individually
        (no padding/masking needed), then decode proceeds in a single batch.

        Yields:
            If return_logits=False:
                (token_column, token_masks) tuples
            If return_logits=True:
                (token_column, token_masks, logits_column) triples
        """
        assert isinstance(tokens, list), "tokens must be a list"

        # Normalize input
        is_batched = len(tokens) > 0 and isinstance(tokens[0], list)
        if is_batched:
            prompts = tokens
        else:
            assert isinstance(tokens[0], int), (
                "expecting list of ints or list of lists of ints"
            )
            prompts = [tokens]

        device = self.model.get_device()
        dtype = COMPUTE_DTYPE
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        num_prompts = len(prompts)
        total_rows = num_prompts * num_samples

        eos = self.tokenizer.get_eos_token_id()
        bos = self.tokenizer.get_bos_token_id()

        kv_kwargs = self._kv_model_kwargs()

        # 1) Prefill: process each prompt individually, collect logits
        prompt_lengths = [len(p) for p in prompts]
        max_prompt_len = max(prompt_lengths)

        # Determine decode cache size
        kv_length_hint = (
            (max_prompt_len + max_tokens)
            if max_tokens is not None
            else self.model.config.sequence_len
        )

        kv_cache_decode = None
        try:
            # Prefill each prompt individually, then copy into the shared
            # decode cache.  This avoids allocating a separate prefill KV
            # cache for the whole batch (which would coexist with the decode
            # cache and double peak memory).
            kv_cache_decode = KVCache(
                batch_size=total_rows,
                seq_len=kv_length_hint,
                device=device,
                dtype=dtype,
                **kv_kwargs,
            )
            all_logits = []
            for i, prompt in enumerate(prompts):
                ids = torch.tensor([prompt], dtype=torch.long, device=device)
                kv_single = KVCache(
                    batch_size=1,
                    seq_len=len(prompt),
                    device=device,
                    dtype=dtype,
                    **kv_kwargs,
                )
                prompt_logits = self.model.forward(ids, kv_cache=kv_single)
                # .clone() to release the full (1, prompt_len, vocab) tensor.
                prompt_logits = prompt_logits[:, -1, :].clone()  # (1, vocab_size)
                pos = kv_single.get_pos()
                # Copy into decode cache for each sample
                for j in range(num_samples):
                    row_idx = i * num_samples + j
                    kv_cache_decode.k_cache[:, row_idx : row_idx + 1, :pos, :, :] = (
                        kv_single.k_cache[:, :, :pos, :, :]
                    )
                    kv_cache_decode.v_cache[:, row_idx : row_idx + 1, :pos, :, :] = (
                        kv_single.v_cache[:, :, :pos, :, :]
                    )
                    kv_cache_decode.cache_seqlens[row_idx] = pos
                    if kv_single.prev_embedding is not None:
                        if kv_cache_decode.prev_embedding is None:
                            kv_cache_decode.prev_embedding = torch.zeros(
                                total_rows,
                                1,
                                self.model.config.n_embd,
                                device=device,
                                dtype=dtype,
                            )
                        kv_cache_decode.prev_embedding[row_idx] = (
                            kv_single.prev_embedding[0]
                        )
                all_logits.append(prompt_logits.expand(num_samples, -1))
                del kv_single
            logits = torch.cat(all_logits, dim=0)  # (total_rows, vocab_size)

            # 2) Decode loop
            row_states = [
                RowState(prompt.copy())
                for prompt in prompts
                for _ in range(num_samples)
            ]
            num_generated = 0

            while True:
                if min_tokens is not None and num_generated < min_tokens:
                    logits[:, eos] = float("-inf")
                    logits[:, bos] = float("-inf")
                # Sample the next token for each row
                next_ids = sample_next_token(logits, rng, temperature, top_k)
                sampled_tokens = next_ids[:, 0].tolist()

                token_column = []
                token_masks = []
                for i, state in enumerate(row_states):
                    token_masks.append(1)
                    next_token = sampled_tokens[i]
                    token_column.append(next_token)
                    state.current_tokens.append(next_token)
                    if next_token == eos or next_token == bos:
                        state.completed = True

                if is_batched:
                    result = (
                        [
                            token_column[i * num_samples : (i + 1) * num_samples]
                            for i in range(num_prompts)
                        ],
                        [
                            token_masks[i * num_samples : (i + 1) * num_samples]
                            for i in range(num_prompts)
                        ],
                    )
                else:
                    result = (token_column, token_masks)

                if return_logits:
                    result = result + (logits,)
                yield result
                num_generated += 1

                if max_tokens is not None and num_generated >= max_tokens:
                    break
                if all(state.completed for state in row_states):
                    break

                # Prepare logits for next iteration
                ids = torch.tensor(
                    token_column, dtype=torch.long, device=device
                ).unsqueeze(1)
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)
                logits = logits[:, -1, :]
        except torch.cuda.OutOfMemoryError:
            # Dump the snapshot BEFORE the finally block frees the KV cache so
            # the snapshot captures the actual state at OOM (with live KV cache
            # and peak fragmentation), not the cleaned-up state afterwards.
            maybe_dump_memory_snapshot(
                f"OOM in Engine.generate (num_prompts={num_prompts}, max_prompt_len={max_prompt_len}, num_samples={num_samples})"
            )
            raise
        finally:
            # Explicitly free the KV cache. @torch.inference_mode() on a generator
            # can prevent proper frame teardown on GeneratorExit, leaving the huge
            # KV cache tensors alive. The finally block guarantees cleanup.
            if kv_cache_decode is not None:
                del kv_cache_decode.k_cache, kv_cache_decode.v_cache
                del kv_cache_decode

    def generate_batch(self, tokens, num_samples=1, return_logits=False, **kwargs):
        """
        Non-streaming batch generation that returns the final token sequences.
        Terminal tokens (eos, bos) are not included in the results.
        """
        eos = self.tokenizer.get_eos_token_id()
        bos = self.tokenizer.get_bos_token_id()

        is_batched = len(tokens) > 0 and isinstance(tokens[0], list)
        prompts = tokens if is_batched else [tokens]

        results = [p.copy() for p in prompts for _ in range(num_samples)]
        masks = [[0] * len(p) for p in prompts for _ in range(num_samples)]
        all_logits = (
            [[None] * len(p) for p in prompts for _ in range(num_samples)]
            if return_logits
            else None
        )
        completed = [False] * len(results)

        gen = self.generate(tokens, num_samples, return_logits=return_logits, **kwargs)
        for gen_output in gen:
            if return_logits:
                token_column, token_masks, logits_batch = gen_output
            else:
                token_column, token_masks = gen_output
                logits_batch = None

            if is_batched:
                token_column = [t for row in token_column for t in row]
                token_masks = [m for row in token_masks for m in row]

            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == eos or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
                        if return_logits:
                            all_logits[i].append(logits_batch[i])
            if all(completed):
                break
        # Explicitly close the generator to free the KV cache tensors immediately,
        # rather than waiting for GC (which may not run before the next allocation).
        gen.close()

        if is_batched:
            results = [
                results[i * num_samples : (i + 1) * num_samples]
                for i in range(len(prompts))
            ]
            masks = [
                masks[i * num_samples : (i + 1) * num_samples]
                for i in range(len(prompts))
            ]
            if return_logits:
                all_logits = [
                    all_logits[i * num_samples : (i + 1) * num_samples]
                    for i in range(len(prompts))
                ]

        if return_logits:
            return results, masks, all_logits
        return results, masks
