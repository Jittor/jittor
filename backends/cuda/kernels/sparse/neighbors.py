"""CUDA hash-table construction and lookup for sparse convolution neighbors."""


def submanifold_neighbors_cuda_options(capacity, points, taps, kernel, dilation):
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
            if (old == -1) return;
            if (sparse_coord_equal(coords, old, b, z, y, x)) {
                // Duplicate coordinate. Which thread wins the CAS above is a
                // race, so keep the lowest point index: that is the first
                // occurrence, which is what the CPU table stores, and it makes
                // the two backends agree instead of silently disagreeing.
                atomicMin(slots + slot, index);
                return;
            }
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
    return {"cuda_header": cuda_header, "cuda_src": cuda_src}
