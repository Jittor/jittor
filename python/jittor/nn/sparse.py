"""Sparse neural-network primitives."""

import jittor as jt


def _triple(value, name):
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError("%s must have three elements" % name)
        result = tuple(int(item) for item in value)
    else:
        result = (int(value),) * 3
    if any(item <= 0 for item in result):
        raise ValueError("%s entries must be positive" % name)
    return result


def build_submanifold_conv3d_neighbors(coords, kernel_size, dilation=1):
    """Build ``[point, kernel_tap]`` source indices without host sync.

    Coordinates are ``[batch, z, y, x]`` integer rows. Missing neighbors are
    encoded as ``-1``. CPU uses an unordered map and CUDA uses an open-addressed
    table that compares full coordinates, so signed and large coordinates do
    not rely on collision-prone packed integer keys.
    """
    if not isinstance(coords, jt.Var):
        raise TypeError("coords must be a jittor.Var")
    shape = tuple(int(size) for size in coords.shape)
    if len(shape) != 2 or shape[1] != 4:
        raise ValueError("coords must have shape [points, 4]")
    if str(coords.dtype) not in ("int32", "int64"):
        raise TypeError("coords must use int32 or int64")
    kernel = _triple(kernel_size, "kernel_size")
    dilation = _triple(dilation, "dilation")
    points = shape[0]
    taps = kernel[0] * kernel[1] * kernel[2]
    capacity = 1
    while capacity < max(1, points * 2):
        capacity *= 2

    cpu_header = r"""
    #include <cstdint>
    #include <unordered_map>
    struct SparseCoord3 {
        int64_t b, z, y, x;
        bool operator==(const SparseCoord3& other) const {
            return b == other.b && z == other.z && y == other.y && x == other.x;
        }
    };
    struct SparseCoord3Hash {
        size_t operator()(const SparseCoord3& value) const {
            uint64_t h = 1469598103934665603ULL;
            const int64_t values[4] = {value.b, value.z, value.y, value.x};
            for (int i = 0; i < 4; ++i) {
                uint64_t v = (uint64_t)values[i];
                h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                h *= 1099511628211ULL;
            }
            return (size_t)h;
        }
    };
    """
    cpu_src = r"""
    @alias(coords, in0)
    @alias(neighbors, out0)
    @alias(slots, out1)
    std::unordered_map<SparseCoord3, int, SparseCoord3Hash> table;
    table.reserve(%(capacity)d);
    for (int i = 0; i < %(capacity)d; ++i) slots_p[i] = -1;
    for (int i = 0; i < %(points)d; ++i) {
        SparseCoord3 key = {
            (int64_t)coords_p[i * 4], (int64_t)coords_p[i * 4 + 1],
            (int64_t)coords_p[i * 4 + 2], (int64_t)coords_p[i * 4 + 3]};
        if (table.find(key) == table.end()) table.emplace(key, i);
    }
    for (int i = 0; i < %(points)d; ++i) {
        int tap = 0;
        for (int kz = 0; kz < %(kd)d; ++kz)
        for (int ky = 0; ky < %(kh)d; ++ky)
        for (int kx = 0; kx < %(kw)d; ++kx, ++tap) {
            SparseCoord3 key = {
                (int64_t)coords_p[i * 4],
                (int64_t)coords_p[i * 4 + 1] + (kz - %(cz)d) * %(dd)d,
                (int64_t)coords_p[i * 4 + 2] + (ky - %(cy)d) * %(dh)d,
                (int64_t)coords_p[i * 4 + 3] + (kx - %(cx)d) * %(dw)d};
            auto found = table.find(key);
            neighbors_p[i * %(taps)d + tap] =
                found == table.end() ? -1 : found->second;
        }
    }
    """ % {
        "capacity": capacity,
        "points": points,
        "taps": taps,
        "kd": kernel[0],
        "kh": kernel[1],
        "kw": kernel[2],
        "cz": kernel[0] // 2,
        "cy": kernel[1] // 2,
        "cx": kernel[2] // 2,
        "dd": dilation[0],
        "dh": dilation[1],
        "dw": dilation[2],
    }
    cuda_header = r"""
    __device__ __forceinline__ unsigned long long sparse_coord_hash(
            long long b, long long z, long long y, long long x) {
        unsigned long long h = 1469598103934665603ULL;
        long long values[4] = {b, z, y, x};
        for (int i = 0; i < 4; ++i) {
            unsigned long long v = (unsigned long long)values[i];
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h *= 1099511628211ULL;
        }
        return h;
    }
    template <typename C>
    __device__ __forceinline__ bool sparse_coord_equal(
            const C* coords, int index,
            long long b, long long z, long long y, long long x) {
        return (long long)coords[index * 4] == b
            && (long long)coords[index * 4 + 1] == z
            && (long long)coords[index * 4 + 2] == y
            && (long long)coords[index * 4 + 3] == x;
    }
    template <typename C>
    __global__ void sparse_hash_insert(const C* coords, int* slots, int points) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= points) return;
        long long b = (long long)coords[index * 4];
        long long z = (long long)coords[index * 4 + 1];
        long long y = (long long)coords[index * 4 + 2];
        long long x = (long long)coords[index * 4 + 3];
        unsigned int slot = (unsigned int)(sparse_coord_hash(b, z, y, x)
            & (%(capacity)d - 1));
        for (int probe = 0; probe < %(capacity)d; ++probe) {
            int old = atomicCAS(slots + slot, -1, index);
            if (old == -1 || sparse_coord_equal(coords, old, b, z, y, x)) return;
            slot = (slot + 1) & (%(capacity)d - 1);
        }
    }
    template <typename C>
    __global__ void sparse_hash_lookup(
            const C* coords, const int* slots, int* neighbors, int total) {
        int flat = blockIdx.x * blockDim.x + threadIdx.x;
        if (flat >= total) return;
        int point = flat / %(taps)d;
        int tap = flat - point * %(taps)d;
        int kx = tap %% %(kw)d;
        int ky = (tap / %(kw)d) %% %(kh)d;
        int kz = tap / (%(kw)d * %(kh)d);
        long long b = (long long)coords[point * 4];
        long long z = (long long)coords[point * 4 + 1]
            + (kz - %(cz)d) * %(dd)d;
        long long y = (long long)coords[point * 4 + 2]
            + (ky - %(cy)d) * %(dh)d;
        long long x = (long long)coords[point * 4 + 3]
            + (kx - %(cx)d) * %(dw)d;
        unsigned int slot = (unsigned int)(sparse_coord_hash(b, z, y, x)
            & (%(capacity)d - 1));
        int found = -1;
        for (int probe = 0; probe < %(capacity)d; ++probe) {
            int index = slots[slot];
            if (index == -1) break;
            if (sparse_coord_equal(coords, index, b, z, y, x)) {
                found = index;
                break;
            }
            slot = (slot + 1) & (%(capacity)d - 1);
        }
        neighbors[flat] = found;
    }
    """ % {
        "capacity": capacity,
        "taps": taps,
        "kh": kernel[1],
        "kw": kernel[2],
        "cz": kernel[0] // 2,
        "cy": kernel[1] // 2,
        "cx": kernel[2] // 2,
        "dd": dilation[0],
        "dh": dilation[1],
        "dw": dilation[2],
    }
    cuda_src = r"""
    @alias(coords, in0)
    @alias(neighbors, out0)
    @alias(slots, out1)
    cudaMemset(slots_p, 0xff, sizeof(int) * %(capacity)d);
    int threads = 256;
    int point_blocks = (%(points)d + threads - 1) / threads;
    if (%(points)d) sparse_hash_insert<<<point_blocks, threads>>>(
        coords_p, slots_p, %(points)d);
    CHECK(0 == cudaGetLastError());
    int total = %(points)d * %(taps)d;
    int lookup_blocks = (total + threads - 1) / threads;
    if (total) sparse_hash_lookup<<<lookup_blocks, threads>>>(
        coords_p, slots_p, neighbors_p, total);
    CHECK(0 == cudaGetLastError());
    """ % {"capacity": capacity, "points": points, "taps": taps}
    neighbors, _ = jt.code(
        [[points, taps], [capacity]],
        ["int32", "int32"],
        [coords],
        cpu_header=cpu_header,
        cpu_src=cpu_src,
        cuda_header=cuda_header,
        cuda_src=cuda_src,
    )
    return neighbors


def submanifold_conv3d(feats, coords, weight, bias=None, dilation=1, neighbors=None):
    """Apply a stride-one submanifold 3-D convolution.

    ``weight`` uses ``[out, kd, kh, kw, in]`` layout. Neighbor discovery runs
    once and every tap is evaluated by one batched matrix multiplication, so
    the implementation has no per-tap Python dispatch or host synchronization.
    Pass a cached ``neighbors`` tensor to reuse topology.
    """
    if not all(isinstance(value, jt.Var) for value in (feats, coords, weight)):
        raise TypeError("feats, coords and weight must be jittor.Var values")
    feat_shape = tuple(int(size) for size in feats.shape)
    coord_shape = tuple(int(size) for size in coords.shape)
    weight_shape = tuple(int(size) for size in weight.shape)
    if len(feat_shape) != 2 or coord_shape != (feat_shape[0], 4):
        raise ValueError("feats and coords must have shapes [points, in] and [points, 4]")
    if len(weight_shape) != 5 or weight_shape[-1] != feat_shape[1]:
        raise ValueError("weight must have shape [out, kd, kh, kw, in]")
    kernel = weight_shape[1:4]
    taps = kernel[0] * kernel[1] * kernel[2]
    if neighbors is None:
        neighbors = build_submanifold_conv3d_neighbors(coords, kernel, dilation=dilation)
    if tuple(int(size) for size in neighbors.shape) != (feat_shape[0], taps):
        raise ValueError("neighbors must have shape [points, kernel_volume]")
    if str(neighbors.dtype) not in ("int32", "int64"):
        raise TypeError("neighbors must use int32 or int64")

    valid = neighbors >= 0
    safe_neighbors = jt.maximum(neighbors, 0)
    gathered = feats[safe_neighbors] * valid.unsqueeze(-1).cast(feats.dtype)
    kernels = weight.reshape((weight_shape[0], taps, feat_shape[1])).permute(1, 2, 0)
    out = jt.matmul(gathered.permute(1, 0, 2), kernels).sum(0)
    if bias is not None:
        if not isinstance(bias, jt.Var) or tuple(int(size) for size in bias.shape) != (
            weight_shape[0],
        ):
            raise ValueError("bias must have shape [out]")
        out = out + bias.reshape((1, weight_shape[0]))
    return out


__all__ = ["build_submanifold_conv3d_neighbors", "submanifold_conv3d"]
