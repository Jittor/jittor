"""Functional linear transformation."""

import jittor as jt


def linear(x, weight, bias=None):
    """Return ``x * weight.T`` with an optional bias."""
    x = jt.nn.matmul_transpose(x, weight)
    if bias is None:
        return x
    # Under the amp register the product is already the compute dtype. A bias
    # that is still wider -- a float32 Parameter, which is what torch keeps --
    # must not lift the result back: torch's autocast casts the bias along with
    # the rest of the linear operator, and a fused linear would keep one dtype
    # here. Native jittor already keeps the compute dtype through the add (its
    # binary inference reads the register); the torch shim's torch-parity
    # promotion does not, so cast explicitly.
    if (jt.flags.amp_reg & (jt.amp_flags.prefer16 | jt.amp_flags.prefer32)
            and hasattr(bias, "cast") and x.dtype != bias.dtype):
        bias = bias.cast(x.dtype)
    return x + bias


__all__ = ["linear"]
