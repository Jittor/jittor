"""Affine-grid construction and grid-sampling operations."""

import jittor as jt
import numpy as np


def grid_sample_v0(input, grid, mode="bilinear", padding_mode="zeros"):
    r"""
    Given an input and a flow-field grid, computes the output using input values and pixel locations from grid.

    grid specifies the sampling pixel locations normalized by the input spatial dimensions. Therefore, it should have most values in the range of [-1, 1]. For example, values x = -1, y = -1 is the left-top pixel of input, and values x = 1, y = 1 is the right-bottom pixel of input.

    Args:

        [in] input (var): the source input var, whose shape is (N, C, Hi, Wi)

        [in] grid (var): the pixel locations, whose shape is (N, Ho, Wo, 2)

        [in] mode (string): the interpolate way, default: bilinear.

        [in] padding_mode (string): the padding way, default: zeros.

        [out] output (var): the output var, whose shape is (N, C, Ho, Wo)

    Example:

        >>> x = jt.array([[[[1,2],[3,4]]]])
        >>> print(x)
        [[[[1 2]
        [3 4]]]]

        >>> grid = jt.array([[[[0.5, 0.5]]]])
        >>> print(x.shape, grid.shape)
        [1,1,2,2,], [1,1,2,2,]

        >>> nn.grid_sample(x, grid)
        [[[[3.25]]]]
    """
    assert padding_mode == "zeros"
    Ni, Ci, Hi, Wi = input.shape
    No, Ho, Wo, D = grid.shape
    assert D == 2
    assert Ni == No
    assert len(input.shape) == 4 and len(grid.shape)

    nid, cid, hid, wid = jt.index((Ni, Ci, Ho, Wo))
    x = ((grid[:, :, :, 1].unsqueeze(1).repeat([1, Ci, 1, 1]) + 1) / 2) * (Hi - 1)
    y = ((grid[:, :, :, 0].unsqueeze(1).repeat([1, Ci, 1, 1]) + 1) / 2) * (Wi - 1)
    return jt.nn._interpolate(input, x, y, (nid, cid), mode)


def linspace_from_neg_one(grid, num_steps, align_corners):
    if num_steps <= 1:
        return jt.array([], dtype=grid.dtype)
    # TODO: use jt.index
    ra = np.linspace(-1, 1, num_steps)
    if not align_corners:
        ra = ra * (num_steps - 1) / num_steps
    return jt.array(ra, dtype=grid.dtype)


def make_base_grid_4D(theta, N, C, H, W, align_corners):
    if jt.flags.use_acl:
        x = jt.nn.linspace_from_neg_one(theta, W, align_corners)
        x = x.reshape(1, 1, W, 1).broadcast((N, H, W, 1))
        y = jt.nn.linspace_from_neg_one(theta, H, align_corners)
        y = y.reshape(1, H, 1, 1).broadcast((N, H, W, 1))
        one = jt.ones((N, H, W, 1), dtype=theta.dtype)
        return jt.concat((x, y, one), dim=-1)
    base_grid = jt.zeros((N, H, W, 3), dtype=theta.dtype)
    base_grid[..., 0] = jt.nn.linspace_from_neg_one(theta, W, align_corners)
    base_grid[..., 1] = jt.unsqueeze(jt.nn.linspace_from_neg_one(theta, H, align_corners), -1)
    base_grid[..., -1] = 1
    return base_grid


def make_base_grid_5D(theta, N, C, D, H, W, align_corners):
    if jt.flags.use_acl:
        x = jt.nn.linspace_from_neg_one(theta, W, align_corners)
        x = x.reshape(1, 1, 1, W, 1).broadcast((N, D, H, W, 1))
        y = jt.nn.linspace_from_neg_one(theta, H, align_corners)
        y = y.reshape(1, 1, H, 1, 1).broadcast((N, D, H, W, 1))
        z = jt.nn.linspace_from_neg_one(theta, D, align_corners)
        z = z.reshape(1, D, 1, 1, 1).broadcast((N, D, H, W, 1))
        one = jt.ones((N, D, H, W, 1), dtype=theta.dtype)
        return jt.concat((x, y, z, one), dim=-1)
    base_grid = jt.zeros((N, D, H, W, 4), dtype=theta.dtype)
    base_grid[..., 0] = jt.nn.linspace_from_neg_one(theta, W, align_corners)
    base_grid[..., 1] = jt.unsqueeze(jt.nn.linspace_from_neg_one(theta, H, align_corners), -1)
    base_grid[..., 2] = jt.unsqueeze(
        jt.unsqueeze(jt.nn.linspace_from_neg_one(theta, D, align_corners), -1), -1
    )
    base_grid[..., -1] = 1
    return base_grid


def affine_grid_generator_4D(theta, N, C, H, W, align_corners):
    base_grid = jt.nn.make_base_grid_4D(theta, N, C, H, W, align_corners)
    grid = jt.nn.bmm(base_grid.reshape(N, H * W, 3), theta.transpose(0, 2, 1))
    return grid.reshape(N, H, W, 2)


def affine_grid_generator_5D(theta, N, C, D, H, W, align_corners):
    base_grid = jt.nn.make_base_grid_5D(theta, N, C, D, H, W, align_corners)
    grid = jt.nn.bmm(base_grid.reshape(N, D * H * W, 4), theta.transpose(0, 2, 1))
    return grid.reshape(N, D, H, W, 3)


def affine_grid(theta, size, align_corners=False):
    assert str(theta.dtype) in ["float", "float32", "float64"]
    assert min(size) > 0
    assert len(size) in [4, 5]
    if len(size) == 4:
        assert theta.ndim == 3 and theta.shape[-2] == 2 and theta.shape[-1] == 3
        return jt.nn.affine_grid_generator_4D(
            theta, size[0], size[1], size[2], size[3], align_corners
        )
    if len(size) == 5:
        assert theta.ndim == 3 and theta.shape[-2] == 3 and theta.shape[-1] == 4
        return jt.nn.affine_grid_generator_5D(
            theta, size[0], size[1], size[2], size[3], size[4], align_corners
        )


def grid_sampler_unnormalize(coord, size, align_corners):
    if align_corners:
        # unnormalize coord from [-1, 1] to [0, size - 1]
        return ((coord + 1) / 2) * (size - 1)
    # unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1) * size - 1) / 2


def clip_coordinates(x, clip_limit):
    return jt.clamp(x, min_v=0, max_v=clip_limit - 1)


def reflect_coordinates(x, twice_low, twice_high):
    if twice_low == twice_high:
        return jt.zeros_like(x)
    m = twice_low / 2
    span = (twice_high - twice_low) / 2
    x = (x - m).abs()
    # `fmod` returns same sign as `in`, which is positive after the `fabs` above.
    extra = x.mod(span)
    flips = (x / span).floor_int()
    result1 = extra + m
    result2 = span - extra + m
    con = flips % 2 == 0
    not_con = flips % 2 != 0
    result1[not_con] = 0.0
    result2[con] = 0.0
    return result1 + result2


def grid_sampler_compute_source_index(coord, size, padding_mode, align_corners):
    coord = jt.nn.grid_sampler_unnormalize(coord, size, align_corners)
    if padding_mode == "border":
        # clip coordinates to image borders
        coord = jt.nn.clip_coordinates(coord, size)
    elif padding_mode == "reflection":
        # reflect coordinates by image borders
        if align_corners:
            coord = jt.nn.reflect_coordinates(coord, 0, 2 * (size - 1))
        else:
            coord = jt.nn.reflect_coordinates(coord, -1, 2 * size - 1)
        # clip coordinates to image borders
        coord = jt.nn.clip_coordinates(coord, size)
    return coord


def grid_sampler_3d(X, grid, mode, padding_mode, align_corners):
    N = X.shape[0]
    C = X.shape[1]
    inp_D = X.shape[2]
    inp_H = X.shape[3]
    inp_W = X.shape[4]

    D = grid.shape[1]
    H = grid.shape[2]
    W = grid.shape[3]
    x = grid[:, :, :, :, 0]
    y = grid[:, :, :, :, 1]
    z = grid[:, :, :, :, 2]
    shape = [N, C, D, H, W]
    cid = jt.index(shape, dim=1)
    nid = jt.index(shape, dim=0)

    x = jt.nn.grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners)
    y = jt.nn.grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners)
    z = jt.nn.grid_sampler_compute_source_index(z, inp_D, padding_mode, align_corners)
    xid = x.reindex(shape, ["i0", "i2", "i3", "i4"])
    yid = y.reindex(shape, ["i0", "i2", "i3", "i4"])
    zid = z.reindex(shape, ["i0", "i2", "i3", "i4"])

    if mode == "nearest":
        return X.reindex([nid, cid, zid.round_int(), yid.round_int(), xid.round_int()])
    if mode == "bilinear":
        fx, fy, fz = xid.floor_int(), yid.floor_int(), zid.floor_int()
        cx, cy, cz = fx + 1, fy + 1, fz + 1
        dx, dy, dz = xid - fx, yid - fy, zid - fz
        dnx, dny, dnz = cx - xid, cy - yid, cz - zid
        a = X.reindex([nid, cid, fz, fy, fx])
        b = X.reindex([nid, cid, cz, fy, fx])
        c = X.reindex([nid, cid, fz, cy, fx])
        d = X.reindex([nid, cid, fz, fy, cx])
        e = X.reindex([nid, cid, fz, cy, cx])
        f = X.reindex([nid, cid, cz, fy, cx])
        g = X.reindex([nid, cid, cz, cy, fx])
        h = X.reindex([nid, cid, cz, cy, cx])
        return (
            a * dnx * dny * dnz
            + b * dnx * dny * dz
            + c * dnx * dy * dnz
            + d * dx * dny * dnz
            + e * dx * dy * dnz
            + f * dx * dny * dz
            + g * dnx * dy * dz
            + h * dx * dy * dz
        )


def grid_sampler_2d(X, grid, mode, padding_mode, align_corners):
    N = X.shape[0]
    C = X.shape[1]
    inp_H = X.shape[2]
    inp_W = X.shape[3]

    H = grid.shape[1]
    W = grid.shape[2]
    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]
    shape = [N, C, H, W]
    cid = jt.index(shape, dim=1)
    nid = jt.index(shape, dim=0)

    x = jt.nn.grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners)
    y = jt.nn.grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners)
    xid = x.reindex(shape, ["i0", "i2", "i3"])
    yid = y.reindex(shape, ["i0", "i2", "i3"])

    if mode == "nearest":
        return X.reindex([nid, cid, yid.round_int(), xid.round_int()])
    if mode == "bilinear":
        # xid,yid = (xid+0.00001),(yid+0.00001)
        fx, fy = xid.floor_int(), yid.floor_int()
        cx, cy = fx + 1, fy + 1
        dx, dy = xid - fx, yid - fy
        dnx, dny = cx - xid, cy - yid

        a = X.reindex([nid, cid, fy, fx], overflow_value=0.0)
        b = X.reindex([nid, cid, cy, fx], overflow_value=0.0)
        c = X.reindex([nid, cid, fy, cx], overflow_value=0.0)
        d = X.reindex([nid, cid, cy, cx], overflow_value=0.0)
        return a * dnx * dny + b * dnx * dy + c * dx * dny + d * dx * dy


def grid_sampler(X, grid, mode, padding_mode, align_corners):
    assert X.dtype == grid.dtype
    assert (X.ndim == 4 or X.ndim == 5) and X.ndim == grid.ndim
    assert X.shape[0] == grid.shape[0] and grid.shape[-1] == X.ndim - 2
    assert X.numel() > 0
    if X.ndim == 4:
        return jt.nn.grid_sampler_2d(X, grid, mode, padding_mode, align_corners)
    return jt.nn.grid_sampler_3d(X, grid, mode, padding_mode, align_corners)


def grid_sample(
    input,
    grid,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=False,
):
    assert mode in ["bilinear", "nearest"]
    assert padding_mode in ["zeros", "border", "reflection"]
    return jt.nn.grid_sampler(input, grid, mode, padding_mode, align_corners)


__all__ = [
    "affine_grid",
    "affine_grid_generator_4D",
    "affine_grid_generator_5D",
    "clip_coordinates",
    "grid_sample",
    "grid_sample_v0",
    "grid_sampler",
    "grid_sampler_2d",
    "grid_sampler_3d",
    "grid_sampler_compute_source_index",
    "grid_sampler_unnormalize",
    "linspace_from_neg_one",
    "make_base_grid_4D",
    "make_base_grid_5D",
    "reflect_coordinates",
]
