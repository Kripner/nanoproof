from collections import deque

import torch
import torch.nn.functional as F


class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.
    Note that the .pos advances automatically after the last layer of the Transformer inserts.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # Each of K/V is of shape (B, H, T, D) and we have one per layer of the Transformer.
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0 # current position in time in the cache

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        """
        Prefill given another KV cache. Optionally expand along batch dim.
        This is used when we do batch 1 prefill and then want to generate
        multiple samples in parallel from there.
        """
        # 1) validate the shapes
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"
        
        # Extract dimensions explicitly
        self_layers, self_kv, self_batch, self_heads, self_seq, self_head_dim = self.kv_shape
        other_layers, other_kv, other_batch, other_heads, other_seq, other_head_dim = other.kv_shape
        
        # Validate dimensions
        assert self_layers == other_layers, f"Layer count mismatch: {self_layers} != {other_layers}"
        assert self_kv == other_kv, f"K/V dimension mismatch: {self_kv} != {other_kv}"
        assert self_heads == other_heads, f"Head count mismatch: {self_heads} != {other_heads}"
        assert self_head_dim == other_head_dim, f"Head dim mismatch: {self_head_dim} != {other_head_dim}"
        
        # Batch size can be expanded (other can be 1, self can be larger)
        assert self_batch == other_batch or other_batch == 1, f"Batch size mismatch: {self_batch} vs {other_batch} (other must be 1 or equal)"
        
        # Sequence length: self must be longer than other
        assert self_seq >= other_seq, f"Sequence length mismatch: {self_seq} < {other_seq}"
        
        # 2) initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        # 4) update the pos
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # Insert new keys/values to the cache and return the full cache so far
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # Dynamically grow the cache if needed
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024 # as much as we need plus buffer of 1024
            t_needed = (t_needed + 1023) & ~1023 # then round up to the nearest multiple of 1024
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape
        # Insert k, v into the cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1, :] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1, :] = v
        # Return the full cached keys/values up to current position (as a view)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1, :]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1, :]
        # Increment pos after the last layer of the Transformer processes
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


# -----------------------------------------------------------------------------
@torch.no_grad()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
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
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.completed = False # Whether this row has completed generation

class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(self, tokens, num_samples=1, max_tokens=None, min_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        Generate tokens from prompt(s). Accepts either list[int] (single prompt) or
        list[list[int]] (batched prompts).

        Yields:
            (token_column, token_masks) tuples where both are nested list[list[int]] of
            shape (num_prompts, num_samples) for batched input, or list[int] of shape
            (num_samples,) for single prompt. Masks: 1=sampled, 0=forced.
        """
        assert isinstance(tokens, list), "tokens must be a list"

        # Normalize input: convert single prompt to list of prompts
        is_batched = len(tokens) > 0 and isinstance(tokens[0], list)
        if is_batched:
            prompts = tokens
        else:
            assert isinstance(tokens[0], int), "expecting list of ints or list of lists of ints"
            prompts = [tokens]

        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        num_prompts = len(prompts)
        total_rows = num_prompts * num_samples

        eos = self.tokenizer.get_eos_token_id()
        bos = self.tokenizer.get_bos_token_id()

        # 1) Left-pad all prompts to max length and create attention mask
        prompt_lengths = [len(p) for p in prompts]
        max_prompt_len = max(prompt_lengths)
        padded_prompts = [[0] * (max_prompt_len - len(p)) + p for p in prompts]

        # Create attention masks if padding is needed
        decode_mask = None
        prefill_attn_mask = None
        if any(length != max_prompt_len for length in prompt_lengths):
            # prompt_mask[b, t] = True if position t is a real token (not padding) for prompt b
            prompt_mask = torch.zeros((num_prompts, max_prompt_len), dtype=torch.bool, device=device)
            for i, length in enumerate(prompt_lengths):
                prompt_mask[i, max_prompt_len - length:] = True
            # causal_mask[q, k] = True if query at position q can attend to key at position k
            causal_mask = torch.tril(torch.ones((max_prompt_len, max_prompt_len), dtype=torch.bool, device=device))
            # prefill_attn_mask combines prompt_mask and causal_mask: attend only to non-padding keys before the query position
            # shape: (num_prompts, 1, max_prompt_len, max_prompt_len) - the 1 broadcasts across heads
            prefill_attn_mask = (causal_mask.unsqueeze(0) & prompt_mask.unsqueeze(1)).unsqueeze(1)
            # decode_mask tracks which positions are valid for each row during generation (will be updated after each step)
            decode_mask = prompt_mask.repeat_interleave(num_samples, dim=0)

        # 2) Run batched prefill
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=num_prompts,
            seq_len=max_prompt_len,
            **kv_model_kwargs,
        )

        ids = torch.tensor(padded_prompts, dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill, attention_mask=prefill_attn_mask)
        logits = logits[:, -1, :]  # (num_prompts, vocab_size)

        # 3) Expand KV cache for num_samples per prompt
        kv_length_hint = (max_prompt_len + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=total_rows,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        # Initialize the decode cache from prefill cache, replicating for each sample
        dtype, dev = kv_cache_prefill.kv_cache.dtype, kv_cache_prefill.kv_cache.device
        kv_cache_decode.kv_cache = torch.empty(kv_cache_decode.kv_shape, dtype=dtype, device=dev)
        for i in range(num_prompts):
            src = kv_cache_prefill.kv_cache[:, :, i:i + 1, :, :max_prompt_len, :]
            for j in range(num_samples):
                kv_cache_decode.kv_cache[:, :, i * num_samples + j:i * num_samples + j + 1, :, :max_prompt_len, :] = src
        kv_cache_decode.pos = max_prompt_len
        del kv_cache_prefill  # no need to keep this memory around

        # Expand logits for num_samples per prompt
        logits = logits.repeat_interleave(num_samples, dim=0)  # (total_rows, vocab_size)

        # 4) Initialize row states and run generation loop
        row_states = [RowState(prompt.copy()) for prompt in prompts for _ in range(num_samples)]
        num_generated = 0

        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            if min_tokens is not None and num_generated < min_tokens:
                logits[:, eos] = float('-inf')
                logits[:, bos] = float('-inf')
            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                token_masks.append(1) # mask is 0 if forced, 1 if sampled
                next_token = sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On eos or bos, mark the row as completed
                if next_token == eos or next_token == bos:
                    state.completed = True

            if is_batched:
                # Yield shape (num_prompts, num_samples)
                yield ([token_column[i * num_samples:(i + 1) * num_samples] for i in range(num_prompts)],
                       [token_masks[i * num_samples:(i + 1) * num_samples] for i in range(num_prompts)])
            else:
                # Yield shape (num_samples,)
                yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)


            if decode_mask is not None:
                # Extend decode_mask with True for the new tokens
                decode_mask = torch.cat(
                    [decode_mask, torch.ones((total_rows, 1), dtype=torch.bool, device=device)], dim=1
                )
                logits = self.model.forward(
                    ids,
                    kv_cache=kv_cache_decode,
                    attention_mask=decode_mask.unsqueeze(1).unsqueeze(1),  # (B, 1, 1, T)
                )
            else:
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)
            logits = logits[:, -1, :]  # (B, vocab_size)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that returns the final token sequences.
        Terminal tokens (assistant_end, bos) are not included in the results.

        Returns:
            (results, masks): For batched input, both are list[list[list[int]]] of shape
            (num_prompts, num_samples, seq_len). For single prompt, both are
            list[list[int]] of shape (num_samples, seq_len). Masks: 1=sampled, 0=forced.
        """
        eos = self.tokenizer.get_eos_token_id()
        bos = self.tokenizer.get_bos_token_id()

        # Normalize input to list of prompts
        is_batched = len(tokens) > 0 and isinstance(tokens[0], list)
        prompts = tokens if is_batched else [tokens]

        # Work with flat structure internally (prompt0_sample0, prompt0_sample1, ..., prompt1_sample0, ...)
        results = [p.copy() for p in prompts for _ in range(num_samples)]
        masks = [[0] * len(p) for p in prompts for _ in range(num_samples)]
        completed = [False] * len(results)


        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            # Flatten nested output from generate() if batched
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
            # Stop if all rows are completed
            if all(completed):
                break

        # Reshape to nested structure for batched output
        if is_batched:
            results = [results[i * num_samples:(i + 1) * num_samples] for i in range(len(prompts))]
            masks = [masks[i * num_samples:(i + 1) * num_samples] for i in range(len(prompts))]
        return results, masks