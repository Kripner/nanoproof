"""
Engine for efficient inference.

KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API:
- Tensors are (B, T, H, D) not (B, H, T, D)
- FA3 updates the cache in-place during flash_attn_with_kvcache
- Position tracked per batch element via cache_seqlens tensor

Engine supports batched generation with variable-length prompts:
- Single fused B=num_prompts prefill at T=max_prompt_len (right-padded with BOS)
- K/V copied into the decode cache and replicated num_samples times per prompt
- Per-row cache_seqlens=real_lens[i] keeps padded slots invisible to decode
- Decode then runs as a normal B=total_rows loop with FA3 / SDPA per-row positions
"""

import torch
import torch.nn.functional as F

from nanoproof.common import COMPUTE_DTYPE, maybe_dump_memory_snapshot
from nanoproof.model import norm


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

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens (per-row)."""
        self.cache_seqlens += num_tokens


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


@torch.inference_mode()
def sample_next_token_limited_replacement(logits, rng, num_samples=6, temperature=1.0, top_k=None, occurrences_cap=2):
    """
    Sample tokens with a per-token occurrence cap using Gumbel-Top-k.

    Args:
        logits: Tensor of shape (B, vocab_size).
        rng: torch.Generator.
        num_samples: Number of samples to draw per batch row.
        temperature: Sampling temperature. Must be > 0.
        top_k: Unsupported for now; must be None.
        max_per_token: Maximum number of times each token may appear.

    Returns:
        Tensor of shape (B, num_samples), where each token appears at most max_per_token times per batch row.
    """
    assert logits.ndim == 2, "logits must have shape (B, vocab_size)"
    assert temperature > 0.0, "temperature must be positive"
    assert top_k is None, "top_k is not supported for capped Gumbel sampling yet"
    assert num_samples > 0, "num_samples must be positive"
    assert occurrences_cap > 0, "occurrences_cap must be positive"
    assert occurrences_cap <= num_samples, "occurrences_cap should not exceed num_samples"

    B, V = logits.shape
    scores = logits / temperature  # (B, V)

    # Draw one Gumbel key per latent token copy.
    u = torch.rand(
        B, V, occurrences_cap, device=logits.device, dtype=logits.dtype, generator=rng,
    )  # (B, V, occurrences_cap)

    eps = torch.finfo(logits.dtype).tiny
    gumbel = -torch.log(-torch.log(u.clamp_min(eps)))

    # Broadcast token scores across latent copies.
    keys = scores.unsqueeze(-1) + gumbel  # (B, V, occurrences_cap)

    # Top-k over all latent copies.
    selected_flat = torch.topk(
        keys.flatten(start_dim=1),
        k=num_samples,
        dim=-1,
    ).indices  # (B, num_samples)

    # flatten order is:
    # token0_copy0, token0_copy1, ..., token0_copyC, token1_copy0, ...
    selected_tokens = selected_flat // occurrences_cap

    return selected_tokens

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
        return_logits=False,
        return_token_logprobs=False,
        first_token_occurrences_cap=None,
        seed=0,
    ):
        """
        Generate tokens from prompt(s). Accepts either list[int] (single prompt) or
        list[list[int]] (batched prompts).

        For variable-length batched prompts, each prompt is prefilled individually
        (no padding/masking needed), then decode proceeds in a single batch.

        Yields tuples in the order: (token_column, token_masks[, logits[, token_logprobs]]).
        Optional trailing elements appear in this order when their flags are set:
            - logits (return_logits=True): the (per-row) logits used for sampling
            - token_logprobs (return_token_logprobs=True): log_softmax(logits)[sampled_token]
              over the same logits used for sampling (post min_tokens masking).
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

        # 1) Batched padded prefill: one fused B=num_prompts forward pass.
        prompt_lengths = [len(p) for p in prompts]
        max_prompt_len = max(prompt_lengths)
        real_lens = torch.tensor(prompt_lengths, dtype=torch.int32, device=device)

        # Decode cache size: max prompt + space for generated tokens.
        kv_length_hint = (
            (max_prompt_len + max_tokens)
            if max_tokens is not None
            else self.model.config.sequence_len
        )

        kv_cache_decode = None
        kv_cache_prefill = None
        try:
            # Right-pad all prompts to max_prompt_len with BOS for the fused
            # forward. Pad slots are written as gibberish K/V in the prefill
            # cache but are never read during decode (cache_seqlens[i] keeps
            # row i's effective length at real_lens[i]).
            ids = torch.full(
                (num_prompts, max_prompt_len), bos, dtype=torch.long, device=device
            )
            for i, prompt in enumerate(prompts):
                ids[i, : len(prompt)] = torch.tensor(
                    prompt, dtype=torch.long, device=device
                )

            kv_cache_prefill = KVCache(
                batch_size=num_prompts,
                seq_len=max_prompt_len,
                device=device,
                dtype=dtype,
                **kv_kwargs,
            )
            prefill_logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
            # (num_prompts, max_prompt_len, vocab); read each row's last real position
            gathered = prefill_logits[
                torch.arange(num_prompts, device=device), real_lens.long() - 1, :
            ].clone()  # (num_prompts, vocab) - clone releases the full prefill_logits
            del prefill_logits
            logits = gathered.repeat_interleave(num_samples, dim=0)  # (total_rows, vocab)

            # Allocate decode cache and copy prefill K/V over with num_samples
            # replication. We copy the entire 0..max_prompt_len slab; per-row
            # cache_seqlens below ensures pad slots beyond real_lens[i] are
            # ignored at decode time.
            kv_cache_decode = KVCache(
                batch_size=total_rows,
                seq_len=kv_length_hint,
                device=device,
                dtype=dtype,
                **kv_kwargs,
            )
            kv_cache_decode.k_cache[:, :, :max_prompt_len, :, :] = (
                kv_cache_prefill.k_cache.repeat_interleave(num_samples, dim=1)
            )
            kv_cache_decode.v_cache[:, :, :max_prompt_len, :, :] = (
                kv_cache_prefill.v_cache.repeat_interleave(num_samples, dim=1)
            )
            kv_cache_decode.cache_seqlens = real_lens.repeat_interleave(num_samples)

            # Patch prev_embedding for smear: the prefill stored x[:, -1, :]
            # which is the PAD position's embedding for short rows. Replace
            # per-row with each prompt's real-last-token post-norm embedding.
            if kv_cache_prefill.prev_embedding is not None:
                last_token_ids = torch.tensor(
                    [p[-1] for p in prompts], dtype=torch.long, device=device
                )
                last_emb = norm(
                    self.model.transformer.wte(last_token_ids).to(dtype)
                )  # (num_prompts, n_embd)
                kv_cache_decode.prev_embedding = (
                    last_emb.repeat_interleave(num_samples, dim=0).unsqueeze(1)
                )

            del kv_cache_prefill
            kv_cache_prefill = None

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
                if (
                    num_generated == 0
                    and first_token_occurrences_cap is not None
                    and num_samples > 1
                ):
                    assert top_k is None, (
                        "top_k is not supported with first_token_occurrences_cap"
                    )
                    assert temperature > 0.0, (
                        "temperature must be > 0 with first_token_occurrences_cap"
                    )
                    # Rows for the same prompt share identical prefill logits;
                    # sample per-prompt with the cap, then expand back to rows.
                    prompt_logits = logits[::num_samples]  # (num_prompts, vocab)
                    sampled = sample_next_token_limited_replacement(
                        prompt_logits,
                        rng,
                        num_samples=num_samples,
                        temperature=temperature,
                        occurrences_cap=first_token_occurrences_cap,
                    )  # (num_prompts, num_samples)
                    next_ids = sampled.reshape(-1, 1)  # (total_rows, 1)
                else:
                    next_ids = sample_next_token(logits, rng, temperature, top_k)
                sampled_tokens = next_ids[:, 0].tolist()

                if return_token_logprobs:
                    token_logprobs = (
                        torch.log_softmax(logits, dim=-1)
                        .gather(1, next_ids)[:, 0]
                        .tolist()
                    )
                else:
                    token_logprobs = None

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
                if return_token_logprobs:
                    if is_batched:
                        result = result + (
                            [
                                token_logprobs[i * num_samples : (i + 1) * num_samples]
                                for i in range(num_prompts)
                            ],
                        )
                    else:
                        result = result + (token_logprobs,)
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
            # Explicitly free the KV caches. @torch.inference_mode() on a
            # generator can prevent proper frame teardown on GeneratorExit,
            # leaving the huge KV cache tensors alive. This finally block
            # guarantees cleanup of both prefill and decode caches.
            if kv_cache_prefill is not None:
                del kv_cache_prefill.k_cache, kv_cache_prefill.v_cache
                del kv_cache_prefill
            if kv_cache_decode is not None:
                del kv_cache_decode.k_cache, kv_cache_decode.v_cache
                del kv_cache_decode

    def generate_batch(
        self,
        tokens,
        num_samples=1,
        return_logits=False,
        return_logprobs=False,
        **kwargs,
    ):
        """
        Non-streaming batch generation that returns the final token sequences.
        Terminal tokens (eos, bos) are not included in the results.

        Return tuple is (results, masks) plus, in order, ``all_logits`` (when
        ``return_logits``) and ``logprob_sums`` (when ``return_logprobs``).
        ``logprob_sums`` is the per-row sum of model log-probs (under the same
        logits used for sampling) over the kept tokens, mirroring the ``masks``
        / ``results`` shape.
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
        logprob_sums = (
            [0.0] * (len(prompts) * num_samples) if return_logprobs else None
        )
        completed = [False] * len(results)

        gen = self.generate(
            tokens,
            num_samples,
            return_logits=return_logits,
            return_token_logprobs=return_logprobs,
            **kwargs,
        )
        for gen_output in gen:
            token_column, token_masks = gen_output[0], gen_output[1]
            idx = 2
            if return_logits:
                logits_batch = gen_output[idx]
                idx += 1
            else:
                logits_batch = None
            if return_logprobs:
                token_logprobs_batch = gen_output[idx]
                idx += 1
            else:
                token_logprobs_batch = None

            if is_batched:
                token_column = [t for row in token_column for t in row]
                token_masks = [m for row in token_masks for m in row]
                if return_logprobs:
                    token_logprobs_batch = [
                        lp for row in token_logprobs_batch for lp in row
                    ]

            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == eos or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
                        if return_logits:
                            all_logits[i].append(logits_batch[i])
                        if return_logprobs:
                            logprob_sums[i] += token_logprobs_batch[i]
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
            if return_logprobs:
                logprob_sums = [
                    logprob_sums[i * num_samples : (i + 1) * num_samples]
                    for i in range(len(prompts))
                ]

        out = (results, masks)
        if return_logits:
            out = out + (all_logits,)
        if return_logprobs:
            out = out + (logprob_sums,)
        return out
