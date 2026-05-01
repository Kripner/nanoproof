"""
Standalone debug script for AMD MI250X NaN/inf investigation.

Loads the SFT model on a single GPU, runs the inference path on one fake state,
and walks each layer to report where NaN/inf first appears. Also probes:
  - Whether ROCm SDPA accepts enable_gqa
  - Whether bf16 vs fp32 changes things
  - Whether the "sharper attention" 1.2x multiplier blows up bf16 in attention

Run inside the container:
    python scripts/debug_amd_inference.py --model-path sft/.../model_NNN.pt
"""

import argparse
import os
import sys

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F


def has_bad(t):
    if not torch.is_tensor(t):
        return False
    if not torch.is_floating_point(t):
        return False
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


def stat(name, t):
    if not torch.is_tensor(t):
        return f"{name}: not a tensor"
    if not torch.is_floating_point(t):
        return f"{name}: dtype={t.dtype} shape={tuple(t.shape)}"
    nan = int(torch.isnan(t).sum())
    inf = int(torch.isinf(t).sum())
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return f"{name}: ALL non-finite (nan={nan} inf={inf})"
    return (
        f"{name}: dtype={t.dtype} shape={tuple(t.shape)} "
        f"nan={nan} inf={inf} "
        f"min={finite.min().item():.4g} max={finite.max().item():.4g} "
        f"absmax={finite.abs().max().item():.4g} "
        f"mean={finite.float().mean().item():.4g}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--state",
        default=(
            "x : Nat\n"
            "n : Nat\n"
            "h : x = n\n"
            "⊢ x + 0 = n"
        ),
        help="Fake state string for the forward pass (default: trivial Nat goal)",
    )
    parser.add_argument(
        "--probe-sdpa-backends",
        action="store_true",
        help="Re-run the forward pass under each SDPA backend (math/efficient/flash)",
    )
    parser.add_argument(
        "--cast-rotary-fp32",
        action="store_true",
        help="Force cos/sin to fp32 to test if rotary in bf16 is the culprit",
    )
    parser.add_argument(
        "--no-qk-scale",
        action="store_true",
        help="Disable the q*1.2 / k*1.2 sharper attention to see if that's the trigger",
    )
    args = parser.parse_args()

    print(f"torch={torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    print(f"  hip: {torch.version.hip}")
    print(f"  device 0: {torch.cuda.get_device_name(0)}")
    print(f"  capability: {torch.cuda.get_device_capability(0)}")

    from nanoproof.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON
    from nanoproof.flash_attention import (
        HAS_FA3, USE_FA3, _SUPPORTS_ENABLE_GQA, _FORCE_MATH_FOR_MASK,
    )
    from nanoproof.checkpoints import load_model
    from nanoproof.engine import Engine
    from nanoproof.inference import TacticModel

    print(f"COMPUTE_DTYPE={COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
    print(f"HAS_FA3={HAS_FA3} USE_FA3={USE_FA3} SDPA enable_gqa supported={_SUPPORTS_ENABLE_GQA}")
    print(f"_FORCE_MATH_FOR_MASK={_FORCE_MATH_FOR_MASK}")

    device = torch.device("cuda:0")
    model, tokenizer, _ = load_model(args.model_path, device, phase="eval")
    model.eval()
    print(f"Loaded model: cos.dtype={model.cos.dtype} sin.dtype={model.sin.dtype}")

    if args.cast_rotary_fp32:
        # patch rotary to fp32 for the duration
        model.cos = model.cos.float()
        model.sin = model.sin.float()
        # the model.forward asserts cos.dtype == COMPUTE_DTYPE; bypass by patching the assert
        import nanoproof.model as nmod
        nmod.COMPUTE_DTYPE = torch.float32  # noqa - quick hack
        from nanoproof import common as ncom
        ncom.COMPUTE_DTYPE = torch.float32
        print("(patched rotary to fp32 + flipped global COMPUTE_DTYPE)")

    if args.no_qk_scale:
        # Monkey-patch attention to skip the *1.2 multiplier
        from nanoproof import model as nmod
        orig_attn_fwd = nmod.CausalSelfAttention.forward

        def patched_fwd(self, x, cos_sin, window_size, kv_cache):
            # copy from the original but without q*1.2; k*1.2
            B, T, C = x.size()
            q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
            cos, sin = cos_sin
            q = nmod.apply_rotary_emb(q, cos, sin)
            k = nmod.apply_rotary_emb(k, cos, sin)
            q, k = nmod.norm(q), nmod.norm(k)
            # skip q*=1.2, k*=1.2 here
            if kv_cache is None:
                y = nmod.flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
            else:
                k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
                y = nmod.flash_attn.flash_attn_with_kvcache(
                    q, k_cache, v_cache, k=k, v=v,
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True, window_size=window_size,
                )
                if self.layer_idx == kv_cache.n_layers - 1:
                    kv_cache.advance(T)
            y = y.contiguous().view(B, T, -1)
            return self.c_proj(y)

        nmod.CausalSelfAttention.forward = patched_fwd
        print("(patched CausalSelfAttention to skip *1.2 sharper attention)")

    # Build inputs the same way TacticModel does
    bos = tokenizer.get_bos_token_id()
    suffix = "\n<|tactic|>"
    tokens = tokenizer(args.state + suffix, prepend=bos)
    print(f"Prompt tokens: {len(tokens)}")
    ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # ----- Test 1: bare model.forward without KV cache -----
    print("\n=== Test 1: model.forward(ids) (no KV cache, training-style fwd) ===")
    with torch.inference_mode():
        logits = model.forward(ids)
    print(stat("logits", logits))
    if has_bad(logits):
        print(">>> NaN/inf in training-style forward!")

    # ----- Test 2: per-block forward, find first layer that goes bad -----
    print("\n=== Test 2: per-layer instrumentation through Transformer.forward ===")
    walk_through_blocks(model, ids)

    # ----- Test 3: full inference path through Engine -----
    print("\n=== Test 3: Engine.generate_batch full path (single state) ===")
    engine = Engine(model, tokenizer)
    tactic_model = TacticModel(
        network=model,
        tokenizer=tokenizer,
        engine=engine,
        num_samples=6,
        seed=0,
        first_token_occurrences_cap=2,
        max_prompt_len=512,
        max_gen_tokens=24,
    )
    try:
        results = tactic_model.sample_tactic_from_str_batch([args.state])
    except RuntimeError as e:
        print(f">>> sample_tactic_from_str_batch raised: {e}")
        results = None
    if results is not None:
        print("  -> success")
        for r in results:
            print(" ", r)

    # ----- Test 4: batched heterogeneous prompts (the real RL load) -----
    print("\n=== Test 4: Engine.generate_batch with multiple heterogeneous prompts ===")
    states = [
        args.state,
        "x : Real\nh : x > 0\n⊢ x ≠ 0",
        "n : Nat\nh : n + 1 = 5\n⊢ n = 4",
        "n : Nat\nm : Nat\nh : n + 1 ≤ m\n⊢ n ≤ m",
        "x y : Real\nh1 : x ≤ y\nh2 : y ≤ x\n⊢ x = y",
        "p q : Prop\nh : p ∧ q\n⊢ q ∧ p",
        "n : Nat\nh : n ≠ 0\n⊢ 0 < n",
        "f : Nat → Nat\nh : ∀ n, f n = n\n⊢ f 5 = 5",
    ]
    print(f"  {len(states)} prompts, num_samples={tactic_model.num_samples}")
    try:
        results = tactic_model.sample_tactic_from_str_batch(states)
    except RuntimeError as e:
        print(f">>> raised: {e}")
        import traceback
        traceback.print_exc()
        results = None
    if results is not None:
        print("  -> success")
        for i, r in enumerate(results):
            print(f"  [{i}] {r}")

    # ----- Test 5: instrumented decode loop to find which iter first produces NaN -----
    print("\n=== Test 5: 36 prompts with per-decode-iter logits NaN check ===")
    long_state = "\n".join(
        [f"h{i} : a{i} = b{i}" for i in range(40)]
    ) + "\n⊢ a0 = b0"
    states36 = [args.state] * 8 + [long_state] * 8 + states[:8] + states[2:6] * 3
    instrument_and_run(tactic_model, states36)

    # ----- Test 6: SINGLE long prompt - exercises the explicit-mask sliding-window path
    print("\n=== Test 6: SINGLE long prompt (prompt > sliding-window=192) ===")
    print("  This exercises the SDPA explicit-mask sliding-window path on bf16.")
    instrument_and_run(tactic_model, [long_state])

    # ----- Test 7: training-style forward over a long sequence -----
    print("\n=== Test 7: training-style forward (no KV) on long seq (T=400) ===")
    bos = tokenizer.get_bos_token_id()
    long_tokens = tokenizer(long_state + "\n<|tactic|>", prepend=bos)
    print(f"  long_state tokenized: {len(long_tokens)} tokens")
    long_ids = torch.tensor([long_tokens], dtype=torch.long, device=device)
    walk_through_blocks(model, long_ids)

    # ----- Test 8: probe SDPA explicit-mask path directly with bf16 vs fp32 -----
    print("\n=== Test 8: direct SDPA explicit-mask sliding-window probe ===")
    probe_sdpa_mask_bug(device)


def instrument_and_run(tactic_model, states):
    """Patch Engine.generate to track NaN/inf at every decode step."""
    import nanoproof.engine as engine_mod

    bad_seen = {"first_iter": None, "rows": None, "logits_dtype": None}

    orig_sample_next = engine_mod.sample_next_token

    iter_counter = {"i": 0}

    def patched_sample_next(logits, rng, temperature=1.0, top_k=None):
        nan_mask = torch.isnan(logits).any(dim=-1) | torch.isinf(logits).any(dim=-1)
        max_abs = logits.abs().amax(dim=-1)
        if nan_mask.any() and bad_seen["first_iter"] is None:
            bad_seen["first_iter"] = iter_counter["i"]
            bad_seen["rows"] = nan_mask.nonzero(as_tuple=True)[0].tolist()
            bad_seen["logits_dtype"] = logits.dtype
            print(
                f"  [iter={iter_counter['i']}] FIRST NaN/inf in logits: rows={bad_seen['rows'][:8]}"
                f"... ({nan_mask.sum().item()}/{logits.size(0)} rows bad)"
            )
            print(f"     logits absmax per-row min/max: {max_abs.min().item():.3g} / {max_abs.max().item():.3g}")
        elif iter_counter["i"] % 4 == 0:
            print(
                f"  [iter={iter_counter['i']}] logits absmax per-row min/max: "
                f"{max_abs.min().item():.3g} / {max_abs.max().item():.3g} "
                f"(no NaN/inf)"
            )
        iter_counter["i"] += 1
        return orig_sample_next(logits, rng, temperature, top_k)

    engine_mod.sample_next_token = patched_sample_next
    try:
        try:
            tactic_model.sample_tactic_from_str_batch(states)
            print("  -> success")
        except RuntimeError as e:
            print(f"  >>> raised: {e}")
    finally:
        engine_mod.sample_next_token = orig_sample_next


def probe_sdpa_mask_bug(device):
    """Synthesize the exact heterogeneous-prefill SDPA call and compare bf16 vs fp32."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    B, H, Hkv, D = 4, 13, 13, 128
    T = 400  # > sliding window (192)
    window = 192

    torch.manual_seed(0)
    q_fp32 = torch.randn(B, H, T, D, device=device)
    k_fp32 = torch.randn(B, Hkv, T, D, device=device)
    v_fp32 = torch.randn(B, Hkv, T, D, device=device)

    row_idx = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
    col_idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    mask = (col_idx <= row_idx) & ((row_idx - col_idx) <= window)  # (T, T)

    # Reference: fp32 + MATH backend
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        ref = F.scaled_dot_product_attention(q_fp32, k_fp32, v_fp32, attn_mask=mask)
    print(f"  reference (fp32 MATH): {stat('out', ref)}")

    for dtype, name in [(torch.bfloat16, "bf16"), (torch.float16, "fp16"), (torch.float32, "fp32")]:
        q = q_fp32.to(dtype)
        k = k_fp32.to(dtype)
        v = v_fp32.to(dtype)
        for backend, bname in [
            (SDPBackend.MATH, "MATH"),
            (SDPBackend.EFFICIENT_ATTENTION, "EFFICIENT"),
            (SDPBackend.FLASH_ATTENTION, "FLASH"),
        ]:
            try:
                with sdpa_kernel(backends=[backend]):
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                tag = f"{name}/{bname}"
                if has_bad(out):
                    print(f"  *** {tag}: BAD -> {stat('out', out)}")
                else:
                    diff = (out.float() - ref).abs().max().item()
                    print(f"  {tag}: OK, max abs diff vs ref = {diff:.4g}")
            except Exception as e:
                print(f"  {name}/{bname}: raised {type(e).__name__}: {str(e)[:120]}")

    # Now try without explicit mask: is_causal + sliding window is not directly
    # supported, but try the smaller-window subset: full causal (no sliding)
    print("\n  --- bf16 + is_causal=True (no explicit mask) ---")
    q_bf16 = q_fp32.to(torch.bfloat16)
    k_bf16 = k_fp32.to(torch.bfloat16)
    v_bf16 = v_fp32.to(torch.bfloat16)
    for backend, bname in [
        (SDPBackend.MATH, "MATH"),
        (SDPBackend.EFFICIENT_ATTENTION, "EFFICIENT"),
        (SDPBackend.FLASH_ATTENTION, "FLASH"),
    ]:
        try:
            with sdpa_kernel(backends=[backend]):
                out = F.scaled_dot_product_attention(q_bf16, k_bf16, v_bf16, is_causal=True)
            print(f"  bf16/{bname} causal: {stat('out', out)}")
        except Exception as e:
            print(f"  bf16/{bname} causal: raised {type(e).__name__}: {str(e)[:120]}")


def walk_through_blocks(model, ids):
    """Re-implement Transformer.forward step by step, with NaN/inf checks."""
    from nanoproof.common import COMPUTE_DTYPE
    from nanoproof.model import norm

    with torch.inference_mode():
        B, T = ids.size()
        positions = torch.arange(T, device=ids.device).unsqueeze(0)
        cos = model.cos.squeeze(0).squeeze(1)[positions].unsqueeze(2)
        sin = model.sin.squeeze(0).squeeze(1)[positions].unsqueeze(2)
        cos_sin = (cos, sin)
        print(f"  rotary: {stat('cos', cos)}")
        print(f"  rotary: {stat('sin', sin)}")

        x = model.transformer.wte(ids)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)
        print(f"  after embed+norm: {stat('x', x)}")

        # smear (no kv_cache)
        if T > 1:
            gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(
                model.smear_gate(x[:, 1:, :24])
            )
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        print(f"  after smear: {stat('x', x)}")

        x0 = x
        backout = None
        backout_layer = model.config.n_layer // 2
        from nanoproof import common as ncom
        for i, block in enumerate(model.transformer.h):
            x_in = x.clone()
            # mimic Transformer.forward residual ladder
            window = model.window_sizes[i]
            x_pre_attn = norm(x_in)
            attn_out = block.attn(x_pre_attn, cos_sin, window, None)
            if has_bad(attn_out):
                print(f"  layer {i}: attn_out HAS BAD: {stat('attn_out', attn_out)}")
                # also check intermediates inside attn
                _walk_through_attn(block.attn, x_pre_attn, cos_sin, window, prefix=f"  layer {i} attn")
                return
            x_after_attn = x_in + attn_out
            mlp_out = block.mlp(norm(x_after_attn))
            if has_bad(mlp_out):
                print(f"  layer {i}: mlp_out HAS BAD: {stat('mlp_out', mlp_out)}")
                return
            x = x_after_attn + mlp_out

            if i == backout_layer:
                backout = x
            if has_bad(x):
                print(f"  layer {i}: residual HAS BAD: {stat('x', x)}")
                _walk_through_attn(block.attn, x_pre_attn, cos_sin, window, prefix=f"  layer {i} attn")
                return
            print(f"  layer {i:2d}: {stat('x', x)}")

        x = norm(x)
        logits = model.lm_head(x).float()
        print(f"  final logits: {stat('logits', logits)}")


def _walk_through_attn(attn, x, cos_sin, window_size, prefix):
    from nanoproof.model import apply_rotary_emb, norm, flash_attn

    print(f"{prefix}: input {stat('x', x)}")
    B, T, C = x.size()
    q = attn.c_q(x).view(B, T, attn.n_head, attn.head_dim)
    k = attn.c_k(x).view(B, T, attn.n_kv_head, attn.head_dim)
    v = attn.c_v(x).view(B, T, attn.n_kv_head, attn.head_dim)
    print(f"{prefix}: after Q/K/V proj  q={stat('q', q)}")
    print(f"{prefix}: after Q/K/V proj  k={stat('k', k)}")
    print(f"{prefix}: after Q/K/V proj  v={stat('v', v)}")
    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    print(f"{prefix}: after rotary       q={stat('q', q)}")
    print(f"{prefix}: after rotary       k={stat('k', k)}")
    q = norm(q)
    k = norm(k)
    print(f"{prefix}: after QK-norm      q={stat('q', q)}")
    print(f"{prefix}: after QK-norm      k={stat('k', k)}")
    q1 = q * 1.2
    k1 = k * 1.2
    print(f"{prefix}: after *1.2         q={stat('q', q1)}")
    print(f"{prefix}: after *1.2         k={stat('k', k1)}")
    y = flash_attn.flash_attn_func(q1, k1, v, causal=True, window_size=window_size)
    print(f"{prefix}: after flash_attn   y={stat('y', y)}")
    y = y.contiguous().view(B, T, -1)
    out = attn.c_proj(y)
    print(f"{prefix}: after c_proj       out={stat('out', out)}")


if __name__ == "__main__":
    main()
