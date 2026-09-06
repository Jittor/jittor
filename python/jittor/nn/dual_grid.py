"""Dual-grid mesh finalization entry point."""

import jittor as jt
from jittor.backends.cuda.kernels.nn import dual_grid as _cuda_kernels


def finalize_dual_grid_mesh_cuda(
    coords, dual_vertices, quad_indices, valid_rows, split_weight, voxel_size, aabb_min
):
    """Transform dual vertices and split selected quads into triangles on CUDA.

    This low-level inference kernel returns ``None`` when tensors are not a
    compatible CUDA signature. ``quad_indices`` and ``valid_rows`` must use
    int32 or int64; connectivity discovery remains the caller's responsibility.
    """
    tensors = (
        coords,
        dual_vertices,
        quad_indices,
        valid_rows,
        split_weight,
        voxel_size,
        aabb_min,
    )
    if not all(isinstance(value, jt.Var) for value in tensors):
        return None
    if str(quad_indices.dtype) not in ("int32", "int64"):
        raise TypeError("quad_indices must use int32 or int64")
    if str(valid_rows.dtype) not in ("int32", "int64"):
        raise TypeError("valid_rows must use int32 or int64")
    return _cuda_kernels._finalize_dual_grid_mesh_cuda(*tensors)


__all__ = ["finalize_dual_grid_mesh_cuda"]
