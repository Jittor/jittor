"""Tensor operations shared with the TorchVision compatibility facade."""

from operator import index

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._runtime.dispatch import try_dispatch

from .frontend import tensor_frontend


def _pack_rois(features, boxes):
    if isinstance(boxes, (list, tuple)):
        rows = []
        for image, coordinates in enumerate(boxes):
            if coordinates.ndim != 2 or coordinates.shape[1] != 4:
                raise ValueError("roi_align expects a list of [K, 4] boxes")
            if coordinates.shape[0]:
                batch = jt.full((coordinates.shape[0], 1), image, dtype=coordinates.dtype)
                rows.append(jt.concat((batch, coordinates), dim=1))
        return jt.concat(rows, dim=0) if rows else features.new_empty((0, 5))
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError("roi_align expects [K, 5] boxes")
    return boxes


def _roi_samples(start, extent, bins, ratio, size):
    bin_size = extent / bins
    if ratio > 0:
        grid, grid_size = ratio, ratio
    else:
        grid = bin_size.ceil().int32().maximum(0)
        grid_size = max(1, int(grid.max().item()))
        grid = grid.reshape((-1, 1, 1))
    dtype = start.dtype
    bin_index = jt.index((1, bins, 1), 1, dtype=dtype)
    samples = jt.index((1, 1, grid_size), 2, dtype=dtype)
    bin_size = bin_size.reshape((-1, 1, 1))
    coord = start.reshape((-1, 1, 1)) + bin_index * bin_size
    denom = grid if ratio > 0 else grid.maximum(1).cast(dtype)
    coord = coord + (samples + 0.5) * bin_size / denom
    valid = (coord >= -1) & (coord <= size) & (samples < grid)
    coord = coord.clamp(0, size - 1)
    low = coord.int32()
    return low, (low + 1).minimum(size - 1), coord - low.cast(dtype), valid, grid


def _roi_align_composite(features, boxes, output_size, scale, ratio, aligned):
    count, channels = boxes.shape[0], features.shape[1]
    height, width = features.shape[-2:]
    pooled_h, pooled_w = output_size
    # Accumulate low-precision feature maps in float32 on the portable path.
    dtype = "float64" if dtype_name(features.dtype) == "float64" else "float32"
    batch = boxes[:, 0].int32()
    coords = boxes[:, 1:].cast(dtype) * scale - (0.5 if aligned else 0)
    x0, y0, x1, y1 = (coords[:, i] for i in range(4))
    extent_y, extent_x = y1 - y0, x1 - x0
    if not aligned:
        extent_y, extent_x = extent_y.maximum(1), extent_x.maximum(1)
    iy0, iy1, dy, valid_y, ny = _roi_samples(y0, extent_y, pooled_h, ratio, height)
    ix0, ix1, dx, valid_x, nx = _roi_samples(x0, extent_x, pooled_w, ratio, width)
    samples_shape = (count, channels, pooled_h, pooled_w, dy.shape[-1], dx.shape[-1])

    def sample(iy, ix):
        return features.reindex(samples_shape, ("@e0(i0)", "i1", "@e1(i0,i2,i4)", "@e2(i0,i3,i5)"),
                                extras=[batch, iy, ix]).cast(dtype)

    y_shape = (count, 1, pooled_h, 1, dy.shape[-1], 1)
    x_shape = (count, 1, 1, pooled_w, 1, dx.shape[-1])
    dy, dx = dy.reshape(y_shape), dx.reshape(x_shape)
    hy, hx = 1 - dy, 1 - dx
    samples = ((hy * hx) * sample(iy0, ix0) + (hy * dx) * sample(iy0, ix1)
               + (dy * hx) * sample(iy1, ix0) + (dy * dx) * sample(iy1, ix1))
    samples = jt.ternary(valid_y.reshape(y_shape) & valid_x.reshape(x_shape), samples, 0)
    divisor = ratio * ratio if ratio > 0 else (ny * nx).reshape((count, 1, 1, 1)).maximum(1)
    return (samples.sum((-2, -1)) / divisor).cast(features.dtype)


def roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
    """Average bilinear samples over each output bin of each RoI."""
    if input.ndim != 4:
        raise ValueError("roi_align expects an [N, C, H, W] input")
    if isinstance(output_size, (tuple, list)):
        pooled_h, pooled_w = map(index, output_size)
    else:
        pooled_h = pooled_w = index(output_size)
    if min(pooled_h, pooled_w) < 0:
        raise ValueError("roi_align output dimensions must be nonnegative")
    with tensor_frontend(type(input), like=input):
        # TorchVision differentiates with respect to the input feature map.
        rois = _pack_rois(input, boxes)
        if dtype_name(input.dtype) != dtype_name(rois.dtype):
            raise TypeError("roi_align input and boxes must have the same dtype")
        rois = rois.detach()
        shape = (rois.shape[0], input.shape[1], pooled_h, pooled_w)
        if not all(shape) or not all(input.shape):
            return (input.sum() * 0).broadcast(shape)
        from jittor.backends.cuda.kernels.nn import roi_align_cuda  # noqa: F401

        args = ((pooled_h, pooled_w), float(spatial_scale), index(sampling_ratio), bool(aligned))
        result = try_dispatch("vision.roi_align", input, rois, *args)
        if result is not None:
            return result
        return _roi_align_composite(input, rois, *args)


def nms(boxes, scores, iou_threshold):
    """Keep non-overlapping boxes in decreasing score order."""
    return batched_nms(boxes, scores, None, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    """Apply greedy NMS independently to each category."""
    if boxes.ndim != 2 or boxes.shape[1] != 4 or tuple(scores.shape) != (boxes.shape[0],):
        raise ValueError("NMS expects boxes with shape [N, 4] and scores with shape [N]")
    if dtype_name(boxes.dtype) != dtype_name(scores.dtype):
        raise TypeError("NMS boxes and scores must have the same dtype")
    if idxs is not None and (tuple(idxs.shape) != tuple(scores.shape)
                             or dtype_name(idxs.dtype) not in ("int32", "int64")):
        raise ValueError("batched_nms expects integer categories with shape [N]")
    with tensor_frontend(type(boxes), like=boxes), jt.no_grad():
        if not boxes.shape[0]:
            return boxes.new_empty((0,), dtype="int64")
        if idxs is None:
            idxs = jt.zeros_like(scores, dtype="int32")
        threshold = float(iou_threshold)
        from jittor.backends.cuda.kernels.misc import batched_nms_cuda  # noqa: F401

        result = try_dispatch("vision.batched_nms", boxes, scores, idxs, threshold)
        if result is not None:
            return result
        order, _ = jt.argsort(scores, descending=True, stable=True)
        # Use a separate integer group key so large category IDs stay exact.
        sorted_classes = idxs[order]
        sorted_boxes = boxes[order]
        width = "max((Tx)0, min(@x(i,2), @x(j,2)) - max(@x(i,0), @x(j,0)))"
        height = "max((Tx)0, min(@x(i,3), @x(j,3)) - max(@x(i,1), @x(j,1)))"
        area = "(@x({i},2)-@x({i},0))*(@x({i},3)-@x({i},1))"
        inter = f"({width}*{height})"
        overlap = f"{inter}/({area.format(i='i')}+{area.format(i='j')}-{inter})"
        # Each category compacts in score order and the final sort restores the
        # global score order, including stable ties.
        keep_ranks = []
        for category in jt.unique(sorted_classes):
            rows = jt.where(sorted_classes == category)[0]
            selected = jt.candidate(sorted_boxes[rows], f"{overlap}>{threshold!r}")
            keep_ranks.append(rows[selected])
        _, ranks = jt.argsort(jt.concat(keep_ranks), stable=True)
        return order[ranks].int64()
