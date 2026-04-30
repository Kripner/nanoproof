"""
End-to-end correctness check for Engine.generate_batch.

Compares Engine output against the naive Transformer.generate reference
(model.py:481, batch=1, no KV cache, recomputes full forward each step) at
temperature=0. Builds a tiny real Transformer so RoPE/smear/cache effects
are exercised, unlike the mock-based tests in test_engine.py.

Run:
  python -m pytest tests/test_engine_correctness.py -v

Or directly (when pytest unavailable, e.g. troja venv):
  PYTHONPATH=. python -c "
  import tests.test_engine_correctness as m
  m.test_single_prompt(); print('test_single_prompt: PASS')
  m.test_same_length_batch(); print('test_same_length_batch: PASS')
  m.test_variable_length_batch(); print('test_variable_length_batch: PASS')
  "
"""

import os
import torch

from nanoproof.engine import Engine
from nanoproof.model import NetworkConfig, Transformer

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False


class _TinyTokenizer:
    """Minimal tokenizer interface (Engine only needs eos/bos ids)."""

    def __init__(self, eos=0, bos=1):
        self._eos = eos
        self._bos = bos

    def get_eos_token_id(self):
        return self._eos

    def get_bos_token_id(self):
        return self._bos


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_tiny_model(seed=0):
    """Tiny Transformer with scrambled c_proj weights.

    `init_weights` zero-initializes both attention and MLP `c_proj` weights,
    which means at-init both blocks contribute 0 to the residual stream and
    the model is effectively identity. That makes the model insensitive to
    RoPE / KV-cache effects for testing. We scramble c_proj after init so
    attention and MLP actually shape the output.
    """
    torch.manual_seed(seed)
    config = NetworkConfig(
        sequence_len=64,
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        window_pattern="LL",
    )
    model = Transformer(config)
    model.init_weights()
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    n_embd = config.n_embd
    s = 3**0.5 * n_embd**-0.5
    for block in model.transformer.h:
        block.attn.c_proj.weight.data.uniform_(-s, s, generator=g)
        block.mlp.c_proj.weight.data.uniform_(-s * 0.4, s * 0.4, generator=g)
    model.to(_device())
    model.eval()
    return model


def _ref_generate(model, prompt, max_tokens):
    """Reference: argmax sampling via Transformer.generate (no KV cache, B=1)."""
    return list(model.generate(prompt, max_tokens=max_tokens, temperature=0.0))


def _engine_generate_batched(engine, prompts, max_tokens):
    """Engine result for a batch of prompts at temp=0, forced to max_tokens
    decode steps via min_tokens=max_tokens (so eos/bos never terminate early)."""
    results, _ = engine.generate_batch(
        prompts,
        num_samples=1,
        min_tokens=max_tokens,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    # results[i][0] = prompts[i] + generated tokens
    return [results[i][0][len(prompts[i]):] for i in range(len(prompts))]


def test_single_prompt():
    model = _build_tiny_model()
    engine = Engine(model, _TinyTokenizer())
    prompt = [2, 3, 4, 5, 6]
    max_tokens = 4
    ref = _ref_generate(model, prompt, max_tokens)
    eng = _engine_generate_batched(engine, [prompt], max_tokens)[0]
    assert eng == ref, f"single prompt diverged.\n  ref: {ref}\n  eng: {eng}"


def test_same_length_batch():
    model = _build_tiny_model()
    engine = Engine(model, _TinyTokenizer())
    prompts = [
        [2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8],
        [4, 5, 6, 7, 8, 9],
        [5, 6, 7, 8, 9, 10],
    ]
    max_tokens = 4
    ref_per_prompt = [_ref_generate(model, p, max_tokens) for p in prompts]
    eng_per_prompt = _engine_generate_batched(engine, prompts, max_tokens)
    for i, (r, e) in enumerate(zip(ref_per_prompt, eng_per_prompt)):
        assert e == r, (
            f"prompt {i} (length {len(prompts[i])}): engine vs reference diverged."
            f"\n  ref: {r}\n  eng: {e}"
        )


def test_variable_length_batch():
    model = _build_tiny_model()
    engine = Engine(model, _TinyTokenizer())
    prompts = [
        [2, 3, 4, 5],
        [2, 3, 4, 5, 6, 7],
        [2, 3, 4, 5, 6, 7, 8, 9],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ]
    max_tokens = 4
    ref_per_prompt = [_ref_generate(model, p, max_tokens) for p in prompts]
    eng_per_prompt = _engine_generate_batched(engine, prompts, max_tokens)
    for i, (r, e) in enumerate(zip(ref_per_prompt, eng_per_prompt)):
        assert e == r, (
            f"prompt {i} (length {len(prompts[i])}): engine vs reference diverged."
            f"\n  ref: {r}\n  eng: {e}"
        )


if __name__ == "__main__":
    test_single_prompt()
    print("test_single_prompt: PASS")
    test_same_length_batch()
    print("test_same_length_batch: PASS")
    test_variable_length_batch()
    print("test_variable_length_batch: PASS")
