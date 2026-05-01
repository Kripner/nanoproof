"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on non-Hopper GPUs (including Blackwell), MPS, and CPU.

Usage (drop-in replacement for FA3):
    from nanoproof.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""

import os

import torch
import torch.nn.functional as F

from nanoproof.common import COMPUTE_DTYPE


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    # ROCm/HIP also exposes torch.cuda, and gcnArch parses to (9, *) on
    # MI200/MI300, which would falsely satisfy the major==9 check below and
    # trigger an HF Hub fetch of NVIDIA-only kernels. Skip on AMD outright.
    if torch.version.hip is not None:
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        if major != 9:
            return None
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel

        return get_kernel("varunneal/flash-attention-3").flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _resolve_use_fa3():
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    if _override_impl == "fa3":
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == "sdpa":
        return False
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False


USE_FA3 = _resolve_use_fa3()


# =============================================================================
# SDPA helpers
# =============================================================================
def _detect_enable_gqa_support():
    """Probe whether F.scaled_dot_product_attention accepts enable_gqa.
    ROCm builds and some older CUDA builds reject the kwarg."""
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
    try:
        q = torch.zeros(1, 2, 1, 8, device=device)
        k = torch.zeros(1, 1, 1, 8, device=device)
        v = torch.zeros(1, 1, 1, 8, device=device)
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return True
    except (TypeError, RuntimeError):
        return False


_SUPPORTS_ENABLE_GQA = _detect_enable_gqa_support()


# ROCm 6.x's EFFICIENT_ATTENTION SDPA backend silently returns 100% NaN when
# given an explicit attn_mask (reproduced with bf16/fp16/fp32 Q/K/V on MI250X,
# torch 2.4 ROCm 6.3). The default SDPA dispatcher picks EFFICIENT for the
# shapes our model uses, so any attn_mask call (sliding window for T > window,
# heterogeneous decode) silently corrupts the activations. Force MATH for
# masked SDPA on ROCm. CUDA paths are untouched.
_FORCE_MATH_FOR_MASK = torch.cuda.is_available() and torch.version.hip is not None


def _sdpa(q, k, v, enable_gqa=False, **kwargs):
    """SDPA wrapper that emulates enable_gqa via head expansion when unsupported."""
    if enable_gqa and not _SUPPORTS_ENABLE_GQA:
        Hq, Hk = q.size(1), k.size(1)
        assert Hq % Hk == 0, f"GQA requires Hq ({Hq}) divisible by Hk ({Hk})"
        repeats = Hq // Hk
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    elif _SUPPORTS_ENABLE_GQA:
        kwargs["enable_gqa"] = enable_gqa
    if _FORCE_MATH_FOR_MASK and kwargs.get("attn_mask") is not None:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, **kwargs)
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return _sdpa(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return _sdpa(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return _sdpa(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    cache_seqlens=None,
    causal=False,
    window_size=(-1, -1),
):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=causal,
            window_size=window_size,
        )

    # SDPA fallback: manually manage KV cache. When cache_seqlens is uniform
    # we use the standard fast path (slice write + _sdpa_attention with
    # native is_causal / window-trim shortcuts). When it's heterogeneous
    # (variable-length prefill), we fall back to per-row writes and an
    # explicit per-row attention mask.
    B, T_new, H, D = q.shape
    device = q.device
    is_uniform = bool((cache_seqlens == cache_seqlens[0]).all().item())

    if is_uniform:
        pos = cache_seqlens[0].item()
        if k is not None and v is not None:
            k_cache[:, pos : pos + T_new, :, :] = k
            v_cache[:, pos : pos + T_new, :, :] = v
        end_pos = pos + T_new
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k_cache[:, :end_pos, :, :].transpose(1, 2)
        v_sdpa = v_cache[:, :end_pos, :, :].transpose(1, 2)
        enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
        y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)
        return y_sdpa.transpose(1, 2)

    # Heterogeneous: per-row write (vectorized for T_new=1 to avoid B
    # GPU<->CPU syncs from .item() inside a Python loop).
    if k is not None and v is not None:
        if T_new == 1:
            b_idx = torch.arange(B, device=device)
            k_cache[b_idx, cache_seqlens.long(), :, :] = k[:, 0, :, :]
            v_cache[b_idx, cache_seqlens.long(), :, :] = v[:, 0, :, :]
        else:
            for b in range(B):
                p = cache_seqlens[b].item()
                k_cache[b, p : p + T_new, :, :] = k[b]
                v_cache[b, p : p + T_new, :, :] = v[b]

    end_pos = int((cache_seqlens + T_new).max().item())
    q_pos = cache_seqlens.long().unsqueeze(1) + torch.arange(T_new, device=device).unsqueeze(0)
    k_pos = torch.arange(end_pos, device=device)
    mask = k_pos[None, None, :] <= q_pos[:, :, None]  # (B, T_new, end_pos)
    window = window_size[0]
    if 0 <= window < end_pos:
        mask = mask & ((q_pos[:, :, None] - k_pos[None, None, :]) <= window)

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_cache[:, :end_pos, :, :].transpose(1, 2)
    v_sdpa = v_cache[:, :end_pos, :, :].transpose(1, 2)
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa(
        q_sdpa, k_sdpa, v_sdpa, attn_mask=mask.unsqueeze(1), enable_gqa=enable_gqa
    )
    return y_sdpa.transpose(1, 2)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace

flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
