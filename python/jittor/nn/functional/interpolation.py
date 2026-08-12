"""Image interpolation operations exposed through :mod:`jittor.nn`."""

import jittor as jt


def _bicubic(x, a, func):
    # normal ver
    if func == 1:
        return (a + 2) * (jt.abs(x) ** 3) - (a + 3) * (x**2) + 1
    if func == 2:
        return a * (jt.abs(x) ** 3) - 5 * a * (x**2) + 8 * a * jt.abs(x) - 4 * a
    return 0


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
        ax, ay = jt.nn._bicubic(dix + 1, -0.75, 2), jt.nn._bicubic(diy + 1, -0.75, 2)
        bx, by = jt.nn._bicubic(dix, -0.75, 1), jt.nn._bicubic(diy, -0.75, 1)
        cx, cy = jt.nn._bicubic(1 - dix, -0.75, 1), jt.nn._bicubic(1 - diy, -0.75, 1)
        dx, dy = jt.nn._bicubic(2 - dix, -0.75, 2), jt.nn._bicubic(2 - diy, -0.75, 2)
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
    nid, cid, hid, wid = jt.index((n, c, H, W))
    if align_corners:
        x = hid * ((h - 1) / max(1, H - 1))
        y = wid * ((w - 1) / max(1, W - 1))
    elif mode == "bicubic":
        x = (hid + 0.5) * (h / H) - 0.5
        y = (wid + 0.5) * (w / W) - 0.5
    elif mode == "nearest":
        x = hid * (h / H)
        y = wid * (w / W)
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
            x = hid * (h / H)
            if H > h:
                x = x.clamp(0, h - 1)
            y = wid * (w / W)
            if W > w:
                y = y.clamp(0, w - 1)
        else:
            x = hid * (h / H) + (h / H * 0.5 - 0.5)
            if H > h:
                x = x.clamp(0, h - 1)
            y = wid * (w / W) + (w / W * 0.5 - 0.5)
            if W > w:
                y = y.clamp(0, w - 1)
    return jt.nn._interpolate(img, x, y, (nid, cid), mode)


def interpolate(
    X,
    size=None,
    scale_factor=None,
    mode="bilinear",
    align_corners=False,
    tf_mode=False,
):
    if scale_factor is not None:
        size = [int(X.shape[-2] * scale_factor), int(X.shape[-1] * scale_factor)]
    if isinstance(size, int):
        size = (size, size)
    if scale_factor is not None and scale_factor > 1:
        return jt.nn.upsample(X, size, mode, align_corners, tf_mode)
    return jt.nn.resize(X, size, mode, align_corners, tf_mode)


__all__ = ["_bicubic", "_interpolate", "interpolate", "resize"]
