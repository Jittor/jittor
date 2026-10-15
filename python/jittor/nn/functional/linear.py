"""Functional linear transformation."""

import jittor as jt
from jittor._runtime.dispatch import select_kernel
from jittor.backends.cuda.kernels.cublas.lt_linear_cuda import (
    lt_linear_train_cuda as _lt_linear_train_cuda,
)

from ._amp import bias_for_compute_dtype
from .matrix import _check_matmul_shapes


def linear(x, weight, bias=None):
    """Return ``x * weight.T`` with an optional bias."""
    accumulator_kernel = select_kernel("linear_bias_accumulator", x, weight, bias) \
        if bias is not None else None
    if accumulator_kernel is not None:
        # The bias belongs to the GEMM's FP32 accumulator. Rounding the product
        # to BF16 before adding it introduces a second rounding, which can
        # change small residual gradients and even the direction of an update.
        # Keep BF16 GEMM operands and the native differentiable backward. The
        # CPU path retains its existing contraction and quantization boundaries.
        _check_matmul_shapes(x, weight, trans_b=True, op="linear", names=("input", "weight"))
        if len(weight.shape) != 2 or len(bias.shape) != 1 \
                or int(bias.shape[0]) != int(weight.shape[0]):
            raise jt.JittorShapeError("linear: bias must match the rank-2 weight's output dimension")
        return accumulator_kernel(x, weight, bias)
    # A large training product on CUDA: its GEMMs picked by timing, and the
    # bias gradient computed by the weight gradient's where that pays.
    fast = _lt_linear_train_cuda(x, weight, bias)
    if fast is not None:
        return fast
    x = jt.nn.matmul_transpose(x, weight, _op="linear", _names=("input", "weight"))
    if bias is None:
        return x
    # Under the amp register the product is already the compute dtype. A bias
    # that is still wider -- a float32 Parameter, which is what torch keeps --
    # must not lift the result back: torch's autocast casts the bias along with
    # the rest of the linear operator, and a fused linear would keep one dtype
    # here. See `_amp.bias_for_compute_dtype` for why the shim needs it and
    # native jittor does not.
    return x + bias_for_compute_dtype(x, bias)


__all__ = ["linear"]
