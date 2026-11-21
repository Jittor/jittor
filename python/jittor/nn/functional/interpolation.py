"""Image interpolation operations exposed through :mod:`jittor.nn`."""

import math
from numbers import Integral, Real

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._core.var import device_scope_like
from jittor._runtime.dispatch import dispatch_context, try_dispatch
from ._layout import channels_last_source, channels_last_view, records_no_grad


def _bicubic(x, a, func):
    # normal ver
    if func == 1:
        return (a + 2) * (jt.abs(x) ** 3) - (a + 3) * (x**2) + 1
    if func == 2:
        return a * (jt.abs(x) ** 3) - 5 * a * (x**2) + 8 * a * jt.abs(x) - 4 * a
    return 0


def _antialias_size(input_size, size, scale_factor, recompute_scale_factor):
    if (size is None) == (scale_factor is None):
        raise ValueError("Specify either size or scale_factor for interpolation")
    if size is not None:
        if recompute_scale_factor:
            raise ValueError("recompute_scale_factor requires scale_factor")
        size = (size, size) if isinstance(size, Integral) else tuple(size)
        if len(size) != 2 or not all(isinstance(extent, Integral) for extent in size):
            raise ValueError("size must have two integer spatial dimensions")
        return tuple(map(int, size)), (None, None)
    factors = (scale_factor, scale_factor) if isinstance(scale_factor, Real) else tuple(scale_factor)
    if len(factors) != 2 or not all(
        isinstance(factor, Real) and math.isfinite(factor) and factor > 0 for factor in factors
    ):
        raise ValueError("scale_factor must have two positive finite spatial factors")
    size = tuple(int(extent * factor) for extent, factor in zip(input_size, factors))
    return size, (None, None) if recompute_scale_factor else factors


def _antialias_weights(in_size, out_size, ratio, mode, dtype):
    width = max(ratio, 1.0)
    support = width * (2 if mode == "bicubic" else 1)
    taps = min(in_size, 2 * math.ceil(support) + 1)
    output_index, tap = jt.index((out_size, taps))
    center = (output_index.cast(dtype) + 0.5) * jt.array(ratio, dtype=dtype)
    radius = jt.array(support, dtype=dtype)
    start = (center - radius + 0.5).floor_int().maximum(0)
    end = (center + radius + 0.5).floor_int().minimum(in_size)
    index = start + tap
    distance = (index.cast(dtype) + 0.5 - center).abs() / jt.array(width, dtype=dtype)
    if mode == "bicubic":
        weights = jt.ternary(distance < 1, _bicubic(distance, -0.5, 1),
                             jt.ternary(distance < 2, _bicubic(distance, -0.5, 2), 0))
    else:
        weights = (1 - distance).maximum(0)
    valid = index < end
    weights = jt.ternary(valid, weights, 0)
    total = weights.sum(1, keepdims=True)
    return jt.ternary(valid, index, in_size), weights / jt.ternary(total != 0, total, 1)


def _antialias_axis(image, extent, axis, ratio, mode, dtype, byte_image):
    # Index and filter coefficients depend on geometry alone.
    with jt.no_grad():
        index, weights = _antialias_weights(image.shape[axis], extent, ratio, mode, dtype)
        if byte_image:
            # The uint8 path uses rounded fixed-point coefficients and
            # saturates the intermediate image before the second pass.
            precision = ((32767.5 / weights.max()).log2().ceil() - 1).clamp(0, 22)
            scale = jt.array(2.0, dtype="float64") ** precision
            scaled = weights * scale
            rounded = (scaled.abs() + 0.5).floor()
            weights = jt.ternary(scaled < 0, -rounded, rounded)
    taps = index.shape[1]
    shape = list(image.shape)
    shape[axis] = extent
    indices = ["i0", "i1", "i2", "i3"]
    indices[axis] = f"@e0(i{axis}, i4)"
    samples = image.reindex([*shape, taps], indices, extras=[index])
    weights_shape = [1, 1, 1, 1, taps]
    weights_shape[axis] = extent
    result = (samples * weights.reshape(weights_shape)).sum(-1)
    if byte_image:
        result = (result / scale + 0.5).floor().clamp(0, 255)
    return result


class _AntialiasOutput(jt.Function):
    def execute(self, output, gradient_source):
        return output.detach().start_grad()

    def grad(self, gradient):
        return None, gradient


def _interpolate_antialiased(image, size, scale_factor, mode, align_corners,
                            recompute_scale_factor):
    if image.ndim != 4 or mode not in ("bilinear", "bicubic"):
        raise ValueError("antialias=True requires a 4-D input and bilinear or bicubic mode")
    backend = dispatch_context(image).backend
    if backend not in ("cpu", "cuda"):
        raise NotImplementedError(f"antialias=True is not implemented for {backend}")
    input_size = tuple(image.shape[-2:])
    size, factors = _antialias_size(input_size, size, scale_factor, recompute_scale_factor)
    if min(*input_size, *size) <= 0 or image.shape[1] <= 0:
        raise RuntimeError("antialiased interpolation requires positive channel and spatial sizes")
    dtype = dtype_name(image.dtype)
    supported = ("float32", "float64", "uint8") if backend == "cpu" else (
        "float16", "bfloat16", "float32", "float64"
    )
    if dtype not in supported:
        raise NotImplementedError(f"antialias=True is not implemented for {dtype} on {backend}")
    tracks_grad = not jt.flags.no_grad and image.requires_grad
    same_size = input_size == size
    if same_size and (align_corners or all(f in (None, 1) for f in factors)):
        result = image.clone()
        return result if tracks_grad else result.stop_grad()
    byte_image = dtype == "uint8"
    compute_dtype = "float64" if byte_image or dtype == "float64" else "float32"
    with device_scope_like(image):
        result = image.cast(compute_dtype)
        for axis in (3, 2):
            in_size = input_size[axis - 2]
            out_size = size[axis - 2]
            factor = factors[axis - 2]
            if align_corners:
                ratio = (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
            else:
                ratio = in_size / out_size if factor is None else 1.0 / factor
            if in_size == out_size and ratio == 1.0:
                continue
            if backend == "cpu" and in_size == out_size:
                # PyTorch's CPU forward copies an unchanged axis. Its backward
                # still uses the explicit scale for that axis.
                if not jt.flags.no_grad and result.requires_grad:
                    filtered = _antialias_axis(result, out_size, axis, ratio, mode, compute_dtype, byte_image)
                    result = _AntialiasOutput.apply(result, filtered)
                continue
            result = _antialias_axis(result, out_size, axis, ratio, mode, compute_dtype, byte_image)
        result = result.cast(dtype)
        if same_size and backend == "cpu":
            result = result.clone()
        elif same_size and tracks_grad:
            # CUDA's backward copies the gradient when both spatial sizes
            # match, even if the forward sampled with explicit factors.
            return _AntialiasOutput.apply(result, image)
        # Resizing stopped input must not create a new differentiable tensor.
        return result if tracks_grad else result.stop_grad()


class _ChannelsLast:
    """[N, H, W, C] storage, sampled through NCHW index lists.

    `_interpolate` indexes an image as (n, c, x, y); this answers the same
    samples from the NHWC tensor a channels-last activation is a view of, so
    the result comes out NHWC too, laid out for the next convolution.
    """

    def __init__(self, source):
        self.source = source
        n, h, w, c = source.shape
        self.shape = (n, c, h, w)

    @staticmethod
    def _nhwc(index):
        n, c, x, y = index
        return [n, x, y, c]

    def reindex(self, index):
        return self.source.reindex(self._nhwc(index))

    def reindex_var(self, index):
        return self.source.reindex_var(self._nhwc(index))


def _interpolate(img, x, y, ids, mode):
    if mode == "nearest":
        return img.reindex([*ids, x.floor_int(), y.floor_int()])
    if mode == "bilinear":
        fx, fy = x.floor_int(), y.floor_int()
        cx, cy = fx + 1, fy + 1
        dx, dy = x - fx, y - fy
        a = img.reindex_var([*ids, fx, fy])
        b = img.reindex_var([*ids, cx, fy])
        c = img.reindex_var([*ids, fx, cy])
        d = img.reindex_var([*ids, cx, cy])
        dnx, dny = 1 - dx, 1 - dy
        ab = dx * b + dnx * a
        cd = dx * d + dnx * c
        return ab * dny + cd * dy
    if mode == "bicubic":  # ugly ver.
        n, c, h, w = img.shape
        fx, fy = x.floor_int(), y.floor_int()
        dix, diy = x - fx, y - fy
        ax, ay = _bicubic(dix + 1, -0.75, 2), _bicubic(diy + 1, -0.75, 2)
        bx, by = _bicubic(dix, -0.75, 1), _bicubic(diy, -0.75, 1)
        cx, cy = _bicubic(1 - dix, -0.75, 1), _bicubic(1 - diy, -0.75, 1)
        dx, dy = _bicubic(2 - dix, -0.75, 2), _bicubic(2 - diy, -0.75, 2)
        afx = jt.maximum(jt.minimum(fx - 1, h - 1), 0)
        afy = jt.maximum(jt.minimum(fy - 1, w - 1), 0)
        bfx = jt.maximum(jt.minimum(fx, h - 1), 0)
        bfy = jt.maximum(jt.minimum(fy, w - 1), 0)
        cfx = jt.maximum(jt.minimum(fx + 1, h - 1), 0)
        cfy = jt.maximum(jt.minimum(fy + 1, w - 1), 0)
        dfx = jt.maximum(jt.minimum(fx + 2, h - 1), 0)
        dfy = jt.maximum(jt.minimum(fy + 2, w - 1), 0)
        a = ax * (
            img.reindex_var([*ids, afx, afy]) * ay
            + img.reindex_var([*ids, afx, bfy]) * by
            + img.reindex_var([*ids, afx, cfy]) * cy
            + img.reindex_var([*ids, afx, dfy]) * dy
        )
        b = bx * (
            img.reindex_var([*ids, bfx, afy]) * ay
            + img.reindex_var([*ids, bfx, bfy]) * by
            + img.reindex_var([*ids, bfx, cfy]) * cy
            + img.reindex_var([*ids, bfx, dfy]) * dy
        )
        c = cx * (
            img.reindex_var([*ids, cfx, afy]) * ay
            + img.reindex_var([*ids, cfx, bfy]) * by
            + img.reindex_var([*ids, cfx, cfy]) * cy
            + img.reindex_var([*ids, cfx, dfy]) * dy
        )
        d = dx * (
            img.reindex_var([*ids, dfx, afy]) * ay
            + img.reindex_var([*ids, dfx, bfy]) * by
            + img.reindex_var([*ids, dfx, cfy]) * cy
            + img.reindex_var([*ids, dfx, dfy]) * dy
        )
        return a + b + c + d
    raise ValueError("unsupported interpolation mode: {}".format(mode))


# TODO: tf_mode to another function
def resize(img, size, mode="nearest", align_corners=False, tf_mode=False):
    if img.dim() != 4:
        raise ValueError("Input shape must be `(N, C, H, W)`!")
    n, c, h, w = img.shape
    H, W = size
    if h <= 0 or w <= 0 or H <= 0 or W <= 0:
        raise RuntimeError(
            f"Input and output sizes should be greater than 0, but got input "
            f"(H: {h}, W: {w}) output (H: {H}, W: {W})"
        )
    if mode not in ("nearest", "nearest-exact", "bilinear", "bicubic", "area"):
        raise ValueError("unsupported interpolation mode: {}".format(mode))
    if mode == "nearest-exact" and (align_corners or tf_mode):
        raise ValueError("nearest-exact requires align_corners=False and tf_mode=False")
    fast = try_dispatch("nn.resize", img, (H, W), mode, align_corners, tf_mode)
    if fast is not None:
        return fast
    # A channels-last input is sampled from its NHWC storage and answered in
    # it; the indices are then laid out [N, H, W, C] for the same reason.
    source = None
    if mode in ("nearest", "nearest-exact", "bilinear", "bicubic") and records_no_grad(img):
        source = channels_last_source(img)
    if mode in ("nearest", "nearest-exact") and not align_corners:
        # Index expressions, not index tensors: `floor(i * h / H)` is the
        # integer quotient. Tensors held here are what an auto-flush inside
        # this call computes in full -- four int32/float maps of the output's
        # size, 1 GB at an SD1.5 VAE's last upsample -- instead of fusing them.
        if mode == "nearest-exact":
            # floor((i+0.5)*input/output), expressed as an integer quotient
            # to preserve the half-pixel tie rule without float rounding.
            if source is None:
                return img.reindex([n, c, H, W], [
                    "i0", "i1", f"(2LL*i2+1)*{h}/(2LL*{H})", f"(2LL*i3+1)*{w}/(2LL*{W})"])
            return channels_last_view(source.reindex([n, H, W, c], [
                "i0", f"(2LL*i1+1)*{h}/(2LL*{H})", f"(2LL*i2+1)*{w}/(2LL*{W})", "i3"]))
        if source is None:
            return img.reindex([n, c, H, W], ["i0", "i1", f"i2*{h}/{H}", f"i3*{w}/{W}"])
        return channels_last_view(
            source.reindex([n, H, W, c], ["i0", f"i1*{h}/{H}", f"i2*{w}/{W}", "i3"])
        )
    if source is None:
        nid, cid, hid, wid = jt.index((n, c, H, W))
    else:
        nid, hid, wid, cid = jt.index((n, H, W, c))
    # The sampling coordinates come from an int32 `jt.index` scaled by a Python
    # float, and int32 * float promotes to float32 whatever the image is. So a
    # float64 image was resampled with float32 weights: the reference test
    # `interpolate_bilinear [cpu/float64]` came back accurate to 1.7e-7 --
    # single-precision epsilon -- where it is checked at 1e-7. Doing the
    # coordinate arithmetic in the image's own floating type fixes that, and
    # leaves every narrower dtype where it was: float16 and bfloat16 want
    # float32 coordinates, which is what torch computes them in too.
    if img.dtype == "float64":
        hid, wid = hid.float64(), wid.float64()
    # Scale by the integer ratio rather than by a precomputed Python float.
    # `hid * (h / H)` rounds `h / H` first and then multiplies, so the error is
    # whatever that one division lost, scaled up by the index -- and the Python
    # float is materialised as a float32 constant, so for a float64 image the
    # rounding is a float32 rounding: 2/3 reached the kernel as 0.66666669.
    # `hid * h / H` multiplies exactly (h and H are small integers) and rounds
    # once, at the end. Same expression, and it is the order the reference
    # implementations use.
    if align_corners:
        x = hid * (h - 1) / max(1, H - 1)
        y = wid * (w - 1) / max(1, W - 1)
    elif mode == "bicubic":
        x = (hid + 0.5) * h / H - 0.5
        y = (wid + 0.5) * w / W - 0.5
    elif mode == "nearest":
        x = hid * h / H
        y = wid * w / W
    elif mode == "area":
        """
        Area interpolation uses AdaptivePool2D to resize origin images.
        """
        stride = (h // H, w // W)
        assert stride[0] > 0 and stride[1] > 0
        x, y = jt.meshgrid(jt.arange(0, H, 1), jt.arange(0, W, 1))
        startH = jt.floor(x * h / H).int32()
        endH = jt.ceil((x + 1) * h / H).int32()
        maxH = int(jt.max(endH - startH).data)
        startW = jt.floor(y * w / W).int32()
        endW = jt.ceil((y + 1) * w / W).int32()
        maxW = int(jt.max(endW - startW).data)
        pixel_count = (endH - startH) * (endW - startW)
        adaptive_output = img.reindex(
            [img.shape[0], img.shape[1], H, W, maxH, maxW],
            ["i0", "i1", "@e0(i2, i3) + i4", "@e2(i2, i3) + i5"],
            extras=[startH, endH, startW, endW],
            overflow_conditions=[
                "i4 >= @e1(i2, i3) - @e0(i2, i3)",
                "i5 >= @e3(i2, i3) - @e2(i2, i3)",
            ],
            overflow_value=0,
        )
        return adaptive_output.reduce("sum", [4, 5]) / pixel_count[None, None, ...]
    else:
        if tf_mode:
            x = hid * h / H
            if H > h:
                x = x.clamp(0, h - 1)
            y = wid * w / W
            if W > w:
                y = y.clamp(0, w - 1)
        else:
            # `(hid + 0.5) * h / H - 0.5`, not `hid * (h/H) + (h/H*0.5 - 0.5)`:
            # algebraically the same, one rounding instead of three, and the
            # same order as the reference.
            x = (hid + 0.5) * h / H - 0.5
            if H > h:
                x = x.clamp(0, h - 1)
            y = (wid + 0.5) * w / W - 0.5
            if W > w:
                y = y.clamp(0, w - 1)
    if source is not None:
        return channels_last_view(_interpolate(_ChannelsLast(source), x, y, (nid, cid), mode))
    return _interpolate(img, x, y, (nid, cid), mode)


def _integer_scaled_size(x, scale_factor, mode):
    factors = tuple(scale_factor) if isinstance(scale_factor, (tuple, list)) else (scale_factor, scale_factor)
    if len(factors) != 2:
        raise ValueError(f"{mode} scale_factor must have two spatial factors")
    factors = tuple(float(value) for value in factors)
    if any(value <= 0 or not value.is_integer() for value in factors):
        # Replacing an explicit fractional scale by floor(input*scale) changes
        # its coordinate mapping. Admit only exact integer scales here until
        # the scale itself can be carried through the resize kernel contract.
        raise NotImplementedError(f"{mode} supports positive integer scale_factor only; use size= for other output sizes")
    return tuple(int(size) * int(factor) for size, factor in zip(x.shape[-2:], factors))


def interpolate(
    X,
    size=None,
    scale_factor=None,
    mode="bilinear",
    align_corners=False,
    tf_mode=False,
):
    if mode == "nearest-exact" and scale_factor is not None:
        size = _integer_scaled_size(X, scale_factor, mode)
        return jt.nn.resize(X, size, mode, align_corners, tf_mode)
    upsampling = False
    if isinstance(scale_factor, (tuple, list)):
        size = _integer_scaled_size(X, scale_factor, mode)
        upsampling = any(target > source for target, source in zip(size, X.shape[-2:]))
    elif scale_factor is not None:
        size = [int(X.shape[-2] * scale_factor), int(X.shape[-1] * scale_factor)]
        upsampling = scale_factor > 1
    if isinstance(size, int):
        size = (size, size)
    if upsampling:
        return jt.nn.upsample(X, size, mode, align_corners, tf_mode)
    return jt.nn.resize(X, size, mode, align_corners, tf_mode)


__all__ = ["_bicubic", "_interpolate", "interpolate", "resize"]
