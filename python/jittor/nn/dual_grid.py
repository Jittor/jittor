"""Dual-grid mesh finalization kernels."""

import jittor as jt


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
    if not (jt.flags.use_cuda and getattr(jt.flags, "no_grad", 0)):
        return None
    if getattr(jt.compiler, "has_acl", 0):
        return None
    try:
        devices = tuple(int(value.get_device()) for value in tensors)
        vertex_count = int(coords.shape[0])
        valid_count = int(valid_rows.shape[0])
        shapes = tuple(tuple(int(size) for size in value.shape) for value in tensors)
    except Exception:
        return None
    if any(device < 0 for device in devices) or len(set(devices)) != 1:
        return None
    if not (
        shapes[0] == (vertex_count, 3)
        and shapes[1] == (vertex_count, 3)
        and len(shapes[2]) == 2
        and shapes[2][1] == 4
        and shapes[3] == (valid_count,)
        and shapes[4] in ((vertex_count,), (vertex_count, 1))
        and shapes[5] == (3,)
        and shapes[6] == (3,)
    ):
        return None

    cuda_header = r"""
    template <typename C, typename V, typename S, typename A, typename O>
    __global__ void dual_grid_vertices(
            const C* coords, const V* vertices,
            const S* voxel_size, const A* aabb_min, O* out,
            int64_t total) {
        int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        for (; index < total; index += stride) {
            int column = (int)(index % 3);
            out[index] = (O)(((float)coords[index] + (float)vertices[index])
                * (float)voxel_size[column] + (float)aabb_min[column]);
        }
    }

    template <typename Q, typename R, typename W, typename O>
    __global__ void dual_grid_faces(
            const Q* quads, const R* rows, const W* weights, O* out,
            int64_t total) {
        int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        for (; index < total; index += stride) {
            int64_t row = index / 6;
            int lane = (int)(index - row * 6);
            int64_t base = (int64_t)rows[row] * 4;
            int q0 = (int)quads[base];
            int q1 = (int)quads[base + 1];
            int q2 = (int)quads[base + 2];
            int q3 = (int)quads[base + 3];
            float weight02 = (float)weights[q0] * (float)weights[q2];
            float weight13 = (float)weights[q1] * (float)weights[q3];
            int split1[6] = {0, 1, 2, 0, 2, 3};
            int split2[6] = {0, 1, 3, 3, 1, 2};
            int corner = weight02 > weight13 ? split1[lane] : split2[lane];
            out[index] = (O)quads[base + corner];
        }
    }
    """
    cuda_src = r"""
    @alias(coords, in0)
    @alias(vertices, in1)
    @alias(quads, in2)
    @alias(rows, in3)
    @alias(weights, in4)
    @alias(voxel_size, in5)
    @alias(aabb_min, in6)
    @alias(out_vertices, out0)
    @alias(out_faces, out1)
    int threads = 256;
    int64_t vertex_total = out_vertices->num;
    if (vertex_total) {
        int blocks = (int)((vertex_total + threads - 1) / threads);
        if (blocks > 4096) blocks = 4096;
        dual_grid_vertices<
            coords_type, vertices_type, voxel_size_type,
            aabb_min_type, out_vertices_type><<<blocks, threads>>>(
            coords_p, vertices_p, voxel_size_p, aabb_min_p,
            out_vertices_p, vertex_total);
        CHECK(0 == cudaGetLastError());
    }
    int64_t face_total = out_faces->num;
    if (face_total) {
        int blocks = (int)((face_total + threads - 1) / threads);
        if (blocks > 4096) blocks = 4096;
        dual_grid_faces<
            quads_type, rows_type, weights_type,
            out_faces_type><<<blocks, threads>>>(
            quads_p, rows_p, weights_p, out_faces_p, face_total);
        CHECK(0 == cudaGetLastError());
    }
    """
    return jt.code(
        [[vertex_count, 3], [valid_count * 2, 3]],
        [dual_vertices.dtype, quad_indices.dtype],
        list(tensors),
        cuda_header=cuda_header,
        cuda_src=cuda_src,
    )


__all__ = ["finalize_dual_grid_mesh_cuda"]
