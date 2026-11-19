"""Small sorted top-k for wide CUDA inference rows."""

# Promotion to FP32 is exact for both 16-bit formats. The low mantissa
# bits after promotion cannot distinguish values of the same sign.
_FIRST_RADIX_BIT = {"float32": 0, "float16": 13, "bfloat16": 16}


def supports_wide_topk(x, k, dim=-1, largest=True, sorted=True):
    from jittor._core.dtypes import dtype_name
    from jittor._core.flags import _output_requires_grad

    return (len(x.shape) == 2 and dim in (-1, 1) and largest and sorted
            and type(k) is int and 1 <= k <= 32 and 0 < x.shape[0] <= 65535
            and 20000 <= x.shape[1] <= 2147483647
            and dtype_name(x.dtype) in _FIRST_RADIX_BIT
            and not _output_requires_grad(x))


_HEADER = """
#include <cub/block/block_radix_sort.cuh>
__device__ static inline unsigned order_key(float x) {
    unsigned bits = __float_as_uint(x);
    return isnan(x) ? 0xffffffffu : (bits & 0x80000000u ? ~bits : bits ^ 0x80000000u);
}
"""

_TILE = """
__global__ static void select_tile(@ARGS_DEF) {
    @PRECALC
    using Sort = cub::BlockRadixSort<unsigned, 128, 8, int>;
    __shared__ typename Sort::TempStorage storage;
    int row = blockIdx.y;
    int tile = blockIdx.x;
    unsigned keys[8];
    int ids[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int64_t col = int64_t(tile) * 1024 + threadIdx.x * 8 + j;
        int id = col < COLUMNS ? FETCH_ID : -1;
        ids[j] = id;
        keys[j] = id < 0 ? 0 : order_key(@in0(row, id));
    }
    Sort(storage).SortDescending(keys, ids, FIRST_RADIX_BIT, 32);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int rank = threadIdx.x * 8 + j;
        if (rank < SMALL_K) @out0(row, tile*SMALL_K + rank) = ids[j];
    }
}
select_tile<<<dim3(out0_shape1 / SMALL_K, out0_shape0), 128>>>(@ARGS);
"""

_ORDER = """
__global__ static void order_winners(@ARGS_DEF) {
    @PRECALC
    __shared__ int ids[32];
    __shared__ float vals[32];
    int lane = threadIdx.x, row = blockIdx.x;
    ids[lane] = -1;
    vals[lane] = 0.f;
    __syncthreads();
    if (lane < K) {
        int id = @in1(row,lane);
        unsigned cutoff = order_key(@in0(row,@in1(row,K-1)));
        bool tail = order_key(@in0(row,id)) == cutoff;
        int slot = 0;
        for (int j = 0; j < K; ++j) {
            int other = @in1(row,j);
            bool other_tail = order_key(@in0(row,other)) == cutoff;
            slot += (other_tail < tail || (other_tail == tail && other < id));
        }
        ids[slot] = id;
        vals[slot] = @in0(row,id);
    }
    // CUDA topk gathers strict winners in input order, then threshold ties.
    // Its small unstable bitonic sort also determines the published tie ranks.
    for (int size = 2; size <= 32; size *= 2) {
        for (int step = size / 2; step; step /= 2) {
            __syncthreads();
            if (lane < 16) {
                int a = 2*lane - (lane & (step-1)), b = a+step;
                float x = vals[a], y = vals[b];
                bool greater = x > y || (isnan(x) && !isnan(y));
                bool ordered = ids[b] == -1 || (ids[a] >= 0 && greater);
                if (ordered == bool(lane & (size/2))) {
                    int t = ids[a]; ids[a] = ids[b]; ids[b] = t;
                    vals[a] = y; vals[b] = x;
                }
            }
        }
    }
    __syncthreads();
    if (lane < K) {
        int id = ids[lane];
        @out0(row,lane) = @in0(row,id);
        @out1(row,lane) = id;
    }
}
order_winners<<<out0_shape0,32>>>(@ARGS);
"""


_GATHER_STABLE = """
__global__ static void gather_stable(@ARGS_DEF) {
    @PRECALC
    int row = blockIdx.x;
    int lane = threadIdx.x;
    if (lane < out0_shape1) {
        int id = @in1(row,lane);
        @out0(row,lane) = @in0(row,id);
        @out1(row,lane) = id;
    }
}
gather_stable<<<out0_shape0,32>>>(@ARGS);
"""


def wide_topk_cuda(x, k, dim=-1, largest=True, sorted=True):
    """Merge tile winners; stable radix selection preserves input order at the cutoff."""
    import jittor as jt
    from jittor._core.dtypes import dtype_name

    first_radix_bit = _FIRST_RADIX_BIT[dtype_name(x.dtype)]
    rows, cols = map(int, x.shape)
    ids = None
    while True:
        tiles = (cols + 1023) // 1024
        inputs = [x] if ids is None else [x, ids]
        fetch = 'int(col)' if ids is None else '@in1(row,col)'
        source = (_TILE.replace('COLUMNS', str(cols)).replace('FETCH_ID', fetch)
                       .replace('SMALL_K', str(k))
                       .replace('FIRST_RADIX_BIT', str(first_radix_bit)))
        ids = jt.code((rows, tiles * k), jt.int32, inputs, cuda_header=_HEADER, cuda_src=source)
        if tiles == 1:
            break
        cols = tiles * k
    if first_radix_bit:
        # Tiled radix selection retains input order within ties, as in the
        # full sort that handled low-precision rows before this fast path.
        return tuple(jt.code([(rows, k)] * 2, [x.dtype, jt.int64], [x, ids],
                             cuda_src=_GATHER_STABLE))
    return tuple(jt.code([(rows, k)] * 2, [x.dtype, jt.int64], [x, ids],
                         cuda_header=_HEADER, cuda_src=_ORDER.replace('K', str(k))))
