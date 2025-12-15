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
@torch.inference_mode()
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
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, min_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        eos = self.tokenizer.get_eos_token_id()
        bos = self.tokenizer.get_bos_token_id()

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        if min_tokens is not None and 0 < min_tokens:
            logits[:, eos] = float('-inf')
            logits[:, bos] = float('-inf')
        if num_samples > 1:
            # Expand logits so that each initial token is sampled independently
            logits = logits.expand(num_samples, -1)
        next_ids = sample_next_token(logits, rng, temperature, top_k)  # (num_samples, 1)
        sampled_tokens = next_ids[:, 0].tolist()

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        first_iteration = True
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Get sampled tokens - either from prefill or from forward pass
            if first_iteration:
                # Use the tokens we already sampled from prefill
                first_iteration = False
            else:
                # Forward the model and get the next token for each row
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size) at last time step
                if min_tokens is not None and num_generated < min_tokens:
                    logits[:, eos] = float('-inf')
                    logits[:, bos] = float('-inf')
                next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
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

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1
            # Prepare ids for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (eos, bos) are not included in the results.
        """
        eos = self.tokenizer.get_eos_token_id()
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
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
        return results, masks