from ._code import code_with_attributes
from ._attributes import attribute_program, code_program, runner_for_alias
import os
from jittor_utils import env_or_try_find
import jittor_utils
import ctypes
import glob
import jittor.compiler as compiler
import jittor as jt
import math
import numpy as np

from typing import Union
from collections.abc import Sequence, Iterable


from ._code import acl_emit, acl_program
from .._soc import kernel_available

#: `jittor.nn.functional.pooling.average` imports jittor.nn, so this cannot be
#: imported at module scope; resolving it once keeps the statement out of every
#: pooling call, where it walked five package objects to return a module that
#: was already in sys.modules.
_pool_output_size = None


def _output_size_fn():
    global _pool_output_size
    if _pool_output_size is None:
        from jittor.nn.functional.pooling.average import _pool_output_size as fn
        _pool_output_size = fn
    return _pool_output_size


#: Assembled programs per (runner, geometry). `acl_code` re-derives its program
#: key from a freshly built attribute mapping on every call; a pooling geometry
#: is fixed by the module that owns it, so a keyed lookup answers instead.
_POOL_PROGRAMS = {}


def _pool_program(name, input_count, output_count, kernel, stride, padding,
                  dilation, ceil_mode, count_include_pad):
    key = (name, input_count, output_count, kernel, stride, padding, dilation,
           ceil_mode, count_include_pad)
    program = _POOL_PROGRAMS.get(key)
    if program is None:
        program = acl_program(
            name,
            input_count,
            output_count,
            attributes={
                "kernel_size": [kernel[0], kernel[1]],
                "poolStrides": [stride[0], stride[1]],
                "poolPads": [padding[0], padding[1]],
                "poolDilations": [dilation[0], dilation[1]],
                "poolCeil": ceil_mode,
                "countIncludePad": count_include_pad,
            },
        )
        _POOL_PROGRAMS[key] = program
    return program


class PoolACL(jt.Function):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=None,
        return_indices=None,
        ceil_mode=False,
        count_include_pad=True,
        op="maximum",
    ):
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        stride = stride if stride else kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        dilation = dilation if dilation else 1
        if dilation != 1:
            raise ValueError("ACL pooling only supports dilation=1")
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        for item in self.kernel_size:
            if item <= 0:
                raise RuntimeError(f"kernel_size must be greater than zero, but got {item}")
        for item in self.stride:
            if item <= 0:
                raise RuntimeError(f"stride must be greater than zero, but got {item}")
        for item in self.padding:
            if item < 0:
                raise RuntimeError(f"padding must be non-negative, but got {item}")
        self.op = op
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def _geometry(self):
        return (self.kernel_size, self.stride, self.padding, self.dilation,
                bool(self.ceil_mode), bool(self.count_include_pad))

    def execute(self, input):
        output_size = _output_size_fn()

        self.input = input
        output_height, output_width = (
            output_size(size, kernel, stride, padding, self.ceil_mode)
            for size, kernel, stride, padding in zip(
                input.shape[-2:], self.kernel_size, self.stride, self.padding
            )
        )

        output_shape = (input.shape[0], input.shape[1], output_height, output_width)

        inputs = [input]

        runner = "Maxpool" if self.op == "maximum" else "Avgpool"
        if self.op not in ("maximum", "mean"):
            raise ValueError("no this type pool")
        if not kernel_available(runner):
            # The installed CANN opp has no fixed-window pooling kernels for
            # this SoC (Ascend310P): compose pooling from strided views.
            self._composite = True
            return self._composite_forward(input, output_shape)
        self._composite = False

        if self.op == "maximum":
            result = acl_emit(
                _pool_program("Maxpool", 1, 2, *self._geometry()),
                inputs,
                [input.dtype, "int32"],
                [output_shape, output_shape],
            )
        else:
            result = acl_emit(
                _pool_program("Avgpool", 1, 1, *self._geometry()),
                inputs,
                [input.dtype],
                [output_shape],
            )

        if self.op == "maximum":
            self.index = result[1]

        if self.return_indices:
            return result[0], result[1]
        else:
            return result[0]

    def grad(self, grad_output):
        input = self.input
        if getattr(self, "_composite", False):
            return self._composite_grad(input, grad_output)
        output_shapes = [input.shape]
        output_dtypes = [input.dtype]
        if self.op == "maximum":
            result = acl_emit(
                _pool_program("MaxpoolBackward", 3, 1, *self._geometry()),
                [grad_output, input, self.index],
                output_dtypes,
                output_shapes,
            )[0]
        elif self.op == "mean":
            result = acl_emit(
                _pool_program("AvgpoolBackward", 2, 1, *self._geometry()),
                [grad_output, input],
                output_dtypes,
                output_shapes,
            )[0]
        else:
            raise ValueError("no this type pool")
        return result

    # ---- composite pooling (SoCs without fixed-window pool kernels) ----
    # The installed Ascend310P CANN opp ships no Maxpool/Avgpool aclnn kernels;
    # pooling is composed from strided SliceV2 views, concat and reduce, with
    # backward scattered through scatter_add. Everything stays on device.

    _NEG_INF = -3.4028234663852886e38

    def _padded(self, x, fill):
        """Pad to cover every output window, ceil_mode included."""
        ph, pw = self.padding
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = self._out_hw
        need_h = (oh - 1) * sh + kh
        need_w = (ow - 1) * sw + kw
        have_h = int(x.shape[-2]) + 2 * ph
        have_w = int(x.shape[-1]) + 2 * pw
        return jt.nn.pad(
            x,
            (pw, pw + max(0, need_w - have_w),
             ph, ph + max(0, need_h - have_h)),
            value=fill)

    def _cell_counts(self, input):
        """Per-cell divisor: window cells inside the *symmetrically* padded
        domain. For count_include_pad this is the divisor semantic; extended
        ceil_mode padding stays outside it."""
        ph, pw = self.padding
        ones = jt.full(tuple(input.shape), 1.0, dtype=input.dtype)
        domain = jt.nn.pad(ones, (pw, pw, ph, ph), value=1.0)
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = self._out_hw
        # A window reaching past the domain reads nothing, which is what the
        # include-pad divisor wants: it counts padded cells, not the extension.
        short_h = (oh - 1) * sh + kh - int(domain.shape[-2])
        short_w = (ow - 1) * sw + kw - int(domain.shape[-1])
        if short_h > 0 or short_w > 0:
            domain = jt.nn.pad(
                domain,
                (0, max(0, short_w), 0, max(0, short_h)), value=0.0)
        views = [
            domain[:, :, i:i + sh * oh:sh, j:j + sw * ow:sw].unsqueeze(-1)
            for i in range(kh) for j in range(kw)
        ]
        return jt.concat(views, dim=-1).sum(-1)

    def _windows(self, xp):
        """(N,C,oH,oW,kh*kw) stack of the strided pooling windows."""
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = self._out_hw
        views = [
            xp[:, :, i:i + sh * oh:sh, j:j + sw * ow:sw].unsqueeze(-1)
            for i in range(kh) for j in range(kw)
        ]
        return jt.concat(views, dim=-1)

    def _composite_forward(self, input, output_shape):
        self._out_hw = (int(output_shape[2]), int(output_shape[3]))
        if self.op == "maximum":
            xp = self._padded(input, self._NEG_INF)
            win = self._windows(xp)
            arg, out = jt.arg_reduce(win, "max", -1, False)
            self._arg = arg
            if self.return_indices:
                kh, kw = self.kernel_size
                sh, sw = self.stride
                oh, ow = self._out_hw
                width = int(xp.shape[-1])
                ohs = jt.arange(oh).reshape(1, 1, oh, 1)
                ows = jt.arange(ow).reshape(1, 1, 1, ow)
                # No Mod aclnn runner on this package; mod by hand.
                flat = ((ohs * sh + arg // kw) * width
                        + ows * sw + arg - (arg // kw) * kw)
                return out, flat.int32()
            return out
        xp = self._padded(input, 0.0)
        sums = self._windows(xp).sum(-1)
        if self.count_include_pad:
            return sums / self._cell_counts(input)
        ones = jt.full(tuple(input.shape), 1.0, dtype=input.dtype)
        counts = self._windows(self._padded(ones, 0.0)).sum(-1)
        return sums / counts

    def _composite_grad(self, input, grad_output):
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = self._out_hw
        n, c = int(input.shape[0]), int(input.shape[1])
        ohs = jt.arange(oh).reshape(1, 1, oh, 1)
        ows = jt.arange(ow).reshape(1, 1, 1, ow)
        g2 = grad_output.reshape(n, c, oh * ow)

        if self.op == "maximum":
            xp = self._padded(input, self._NEG_INF)
            win = self._windows(xp)
            arg = jt.arg_reduce(win, "max", -1, False)[0]
            width = int(xp.shape[-1])
            flat = ((ohs * sh + arg // kw) * width
                    + ows * sw + arg - (arg // kw) * kw)
            flat = flat.reshape(n, c, oh * ow)
            dx = jt.zeros(
                (n, c, int(xp.shape[-2]) * width), dtype=input.dtype
            ).scatter_add(-1, flat.int64(), g2)
            dx = dx.reshape(n, c, int(xp.shape[-2]), width)
        else:
            xp = self._padded(input, 0.0)
            width = int(xp.shape[-1])
            if self.count_include_pad:
                gcell = (grad_output / self._cell_counts(input)).reshape(
                    n, c, oh * ow)
            else:
                ones = jt.full(tuple(input.shape), 1.0, dtype=input.dtype)
                counts = self._windows(self._padded(ones, 0.0)).sum(-1)
                gcell = (grad_output / counts).reshape(n, c, oh * ow)
            dx = jt.zeros(
                (n, c, int(xp.shape[-2]) * width), dtype=input.dtype)
            for i in range(kh):
                for j in range(kw):
                    flat = ((ohs * sh + i) * width + ows * sw + j)
                    flat = flat.reshape(1, 1, oh * ow).broadcast(
                        (n, c, oh * ow))
                    dx = dx.scatter_add(-1, flat.int64(), gcell)
            dx = dx.reshape(n, c, int(xp.shape[-2]), width)

        ph, pw = self.padding
        height, width_in = int(input.shape[-2]), int(input.shape[-1])
        return dx[:, :, ph:ph + height, pw:pw + width_in]


class AdaptiveAvgPool2dACL(jt.Function):
    """CANN adaptive windows, including overlapping bins and upsampling."""

    def execute(self, x, output_size):
        from ._code import check_acl_float_dtype
        check_acl_float_dtype(x, "adaptive_avg_pool2d")
        self.input = x
        return acl_emit(
            acl_program("AdaptiveAvgPool2d", 1, 1,
                        attributes={"outputSize": list(output_size)}),
            [x], output_dtypes=[x.dtype],
            output_shapes=[tuple(x.shape[:-2]) + tuple(output_size)],
        )[0]

    def grad(self, dout):
        x = self.input
        return acl_emit(
            acl_program("AdaptiveAvgPool2dBackward", 2, 1, attributes={}),
            [dout, x], output_dtypes=[x.dtype], output_shapes=[x.shape],
        )[0], None
