"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

import os
import torch
import pytest
from dataclasses import dataclass

from nanoproof.engine import KVCache, Engine
from nanoproof.common import COMPUTE_DTYPE

# -----------------------------------------------------------------------------
# Ensure deterministic behavior for reproducible tests

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Mock classes for testing Engine without loading a real model


@dataclass
class MockConfig:
    """Minimal config for Engine tests."""

    n_kv_head: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_layer: int = 2
    sequence_len: int = 128


class MockModel:
    """
    Mock model that returns uniform logits over the vocab.
    This ensures that with temperature > 0, different samples should
    (with very high probability) produce different tokens.
    """

    def __init__(self, vocab_size=262):
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self._device = "cpu"

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None, loss_reduction="mean"):
        """Return uniform logits so sampling is spread across vocab."""
        B, T = ids.shape
        # Simulate what a real transformer does: advance the KV cache
        if kv_cache is not None:
            # The model advances the cache after the last layer processes
            # In the real model, this happens inside CausalSelfAttention
            kv_cache.advance(T)
        # Uniform logits -> equal probability for all tokens
        logits = torch.zeros(B, T, self.vocab_size)
        return logits


class ByteTokenizer:
    """
    Simple byte-level tokenizer for testing.
    Tokens 0-255 are raw bytes, 256+ are special tokens.
    """

    def __init__(self):
        self._special_tokens = {
            "<|python_start|>": 256,
            "<|python_end|>": 257,
            "<|output_start|>": 258,
            "<|output_end|>": 259,
            "<|assistant_end|>": 260,
            "<|bos|>": 261,
        }
        self._bos = 261
        self._eos = 260

    def encode_special(self, s):
        return self._special_tokens[s]

    def get_bos_token_id(self):
        return self._bos

    def get_eos_token_id(self):
        return self._eos

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")

    def __call__(self, s, prepend=None):
        return self.encode(s, prepend=prepend)


# -----------------------------------------------------------------------------
# KVCache tests


def test_kv_cache_basic():
    """Test basic KVCache operations."""
    B, H, T, D, L = 2, 4, 64, 16, 2
    cache = KVCache(batch_size=B, num_heads=H, seq_len=T, head_dim=D, num_layers=L)
    assert cache.get_pos() == 0
    cache.advance(10)
    assert cache.get_pos() == 10
    cache.reset()
    assert cache.get_pos() == 0


def test_kv_cache_prefill():
    """Test KVCache prefill from batch=1 to batch=N."""
    B_src, B_dst = 1, 4
    H, T, D, L = 4, 32, 16, 2
    src = KVCache(batch_size=B_src, num_heads=H, seq_len=T, head_dim=D, num_layers=L)
    # Simulate a prefill
    src.k_cache[:, :, :8, :, :] = torch.randn(L, B_src, 8, H, D)
    src.v_cache[:, :, :8, :, :] = torch.randn(L, B_src, 8, H, D)
    src.advance(8)
    src.prev_embedding = torch.randn(1, 1, 64)

    dst = KVCache(batch_size=B_dst, num_heads=H, seq_len=T, head_dim=D, num_layers=L)
    dst.prefill(src)
    assert dst.get_pos() == 8
    assert dst.prev_embedding.shape == (B_dst, 1, 64)
    # Check data was copied
    for b in range(B_dst):
        assert torch.equal(dst.k_cache[:, b, :8, :, :], src.k_cache[:, 0, :8, :, :])


def test_kv_cache_get_layer_cache():
    """Test get_layer_cache returns correct views."""
    B, H, T, D, L = 1, 2, 16, 8, 3
    cache = KVCache(batch_size=B, num_heads=H, seq_len=T, head_dim=D, num_layers=L)
    for layer_idx in range(L):
        k, v = cache.get_layer_cache(layer_idx)
        assert k.shape == (B, T, H, D)
        assert v.shape == (B, T, H, D)
        assert k.data_ptr() == cache.k_cache[layer_idx].data_ptr()


# -----------------------------------------------------------------------------
# Engine tests


def test_multi_sample_first_token_diversity():
    """
    Test that when generating multiple samples, each sample gets an independently
    sampled first token (not a broadcast of the same token to all rows).
    """
    model = MockModel(vocab_size=262)
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    prompt_tokens = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"
    num_samples = 16

    first_tokens = []
    gen = engine.generate(
        prompt_tokens,
        num_samples=num_samples,
        max_tokens=1,
        temperature=1.0,
        seed=42,
    )
    for token_column, token_masks in gen:
        first_tokens = token_column

    unique_tokens = set(first_tokens)
    assert len(unique_tokens) > 1, (
        f"All {num_samples} samples got the same first token ({first_tokens[0]}). "
        f"With uniform logits, this is statistically impossible unless tokens are broadcast."
    )


def test_batched_generation_single_prompt():
    """
    Test that batched generation with a single prompt in the batch
    produces the same result as non-batched single prompt generation.
    """
    model = MockModel(vocab_size=262)
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"
    num_samples = 3
    generation_kwargs = dict(max_tokens=8, temperature=0.0, seed=0)

    # Generate non-batched
    single_results, single_masks = engine.generate_batch(
        prompt, num_samples=num_samples, **generation_kwargs
    )

    # Generate batched with single prompt
    batched_results, batched_masks = engine.generate_batch(
        [prompt], num_samples=num_samples, **generation_kwargs
    )

    assert single_results == batched_results[0]
    assert single_masks == batched_masks[0]


def test_batched_generation_stochastic():
    """
    Test that batched generation with temperature > 0 produces diverse outputs.
    """
    model = MockModel(vocab_size=262)
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    prompts = [
        [261, 72, 105],  # <bos> + "hi"
        [261, 72, 101, 108, 108, 111],  # <bos> + "Hello"
    ]

    num_samples = 4
    generation_kwargs = dict(max_tokens=16, temperature=1.0, seed=0)

    results, _ = engine.generate_batch(
        prompts, num_samples=num_samples, **generation_kwargs
    )

    assert len(results) == len(prompts)
    for prompt_idx, samples in enumerate(results):
        assert len(samples) == num_samples
        unique_samples = set(tuple(s) for s in samples)
        assert len(unique_samples) > 1, (
            f"All {num_samples} samples for prompt {prompt_idx} are identical."
        )


def test_batched_generation_variable_length():
    """
    Test that batched generation with variable-length prompts works correctly.
    Each prompt is prefilled individually, then decoded together.
    """
    model = MockModel(vocab_size=262)
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    # Prompts of different lengths
    prompts = [
        [261, 72],  # short
        [261, 72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100],  # long
    ]

    num_samples = 2
    generation_kwargs = dict(max_tokens=5, temperature=0.0, seed=0)

    # This should not crash even with different-length prompts
    results, masks = engine.generate_batch(
        prompts, num_samples=num_samples, **generation_kwargs
    )

    assert len(results) == len(prompts)
    for prompt_idx, samples in enumerate(results):
        assert len(samples) == num_samples
        for sample in samples:
            # Each sample should contain the original prompt + generated tokens
            assert len(sample) > len(prompts[prompt_idx])
