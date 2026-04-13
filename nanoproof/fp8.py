"""Minimal FP8 training for nanoproof - tensorwise dynamic scaling only.

Drop-in replacement for torchao's Float8Linear (~2000 lines) with ~150 lines.
We only need the "tensorwise" recipe (one scalar scale per tensor), not the full
generality of torchao (rowwise scaling, FSDP float8 all-gather, DTensor, tensor
subclass dispatch tables, etc.)

How FP8 training works
======================
A standard Linear layer does one matmul in forward and two in backward:
  forward:      output     = input      @ weight.T
  backward:     grad_input = grad_output @ weight
                grad_weight= grad_output.T @ input

FP8 training wraps each of these three matmuls with:
  1. Compute scale = FP8_MAX / max(|tensor|)  for each operand
  2. Quantize: fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
  3. Matmul via torch._scaled_mm (cuBLAS FP8 kernel, ~2x faster than bf16)
  4. Dequantize: _scaled_mm handles this internally using the inverse scales

The key insight: torch._scaled_mm and the float8 dtypes are PyTorch built-ins.
torchao is just orchestration around these primitives. We can call them directly.

FP8 dtype choice
================
There are two FP8 formats. We use both, following the standard convention:
  - float8_e4m3fn: 4-bit exponent, 3-bit mantissa, range [-448, 448]
    Higher precision (more mantissa bits), used for input and weight.
  - float8_e5m2:   5-bit exponent, 2-bit mantissa, range [-57344, 57344]
    Wider range (more exponent bits), used for gradients which can be large.

torch._scaled_mm layout requirements
=====================================
The cuBLAS FP8 kernel requires specific memory layouts:
  - First argument (A):  must be row-major (contiguous)
  - Second argument (B): must be column-major (B.t().contiguous().t())
If B is obtained by transposing a contiguous tensor (e.g. weight.t()), it is
already column-major, no copy needed. Otherwise we use _to_col_major().
"""

import torch
import torch.nn as nn

from nanoproof.common import COMPUTE_DTYPE

# Avoid division by zero when computing scale from an all-zeros tensor
EPS = 1e-12


@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    """Dynamically quantize a tensor to FP8 using tensorwise scaling."""
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.float().abs().max()
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale


def _to_col_major(x):
    """Rearrange a 2D tensor's memory to column-major layout."""
    return x.t().contiguous().t()


@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    """Custom autograd for the three FP8 GEMMs of a Linear layer."""

    @staticmethod
    def forward(ctx, input_2d, weight):
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)
        output = torch._scaled_mm(
            input_fp8, weight_fp8.t(),
            scale_a=input_inv, scale_b=weight_inv,
            out_dtype=input_2d.dtype, use_fast_accum=True,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8, w_col,
            scale_a=go_inv, scale_b=w_inv,
            out_dtype=grad_output.dtype, use_fast_accum=False,
        )
        go_T = go_fp8.t().contiguous()
        in_col = _to_col_major(in_fp8)
        grad_weight = torch._scaled_mm(
            go_T, in_col,
            scale_a=go_inv, scale_b=in_inv,
            out_dtype=grad_output.dtype, use_fast_accum=False,
        )
        return grad_input, grad_weight


class Float8Linear(nn.Linear):
    """Drop-in nn.Linear replacement that does FP8 compute."""

    def forward(self, input):
        input = input.to(COMPUTE_DTYPE)
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float8Matmul.apply(input_2d, self.weight)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod):
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


class Float8LinearConfig:
    """Minimal config matching torchao's API. Only tensorwise recipe is supported."""
    @staticmethod
    def from_recipe_name(recipe_name):
        if recipe_name != "tensorwise":
            raise ValueError(f"Only 'tensorwise' recipe is supported, got '{recipe_name}'.")
        return Float8LinearConfig()


def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
    """Replace nn.Linear layers with Float8Linear throughout a module."""
    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child))
    _convert(module)
    return module
