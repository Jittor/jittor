"""CUDA tensor kernels; CPU fallback sources are supplied by native callers."""

import jittor as jt
import numpy as np

from .topk_cuda import supports_wide_topk, wide_topk_cuda

from jittor._runtime.dispatch import optional_kernel, register_kernel
from jittor._runtime.backend_libraries import get_library_ops
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._core.flags import _output_requires_grad, _stop_grad_outputs


register_kernel("tensor.topk", "cuda", wide_topk_cuda, supports=supports_wide_topk)

_CUDA_CODE_BACKENDS = ("cuda", "rocm_legacy", "corex_legacy")


_COLLECTIVE_PACK_HEADER = r'''
struct PackSegment {
    const float* source;
    float* target;
    int64_t length;
    int64_t first_block;
};

struct PackLaunch {
    PackSegment segments[pack_segment_count];
};

__global__ static void pack_flat_kernel(PackLaunch launch, int64_t block_count) {
    constexpr int threads = 256;
    constexpr int per_block = threads * 4;
    int segment = 0;
    for (int64_t block = blockIdx.x; block < block_count; block += gridDim.x) {
        while (segment + 1 < pack_segment_count && launch.segments[segment + 1].first_block <= block)
            ++segment;
        const auto& item = launch.segments[segment];
        const auto start = (block - item.first_block) * per_block;
        const auto index = start + threadIdx.x * 4;
        if ((reinterpret_cast<uintptr_t>(item.source) | reinterpret_cast<uintptr_t>(item.target)) % 16 == 0) {
            if (index + 4 <= item.length)
                reinterpret_cast<float4*>(item.target)[index / 4] =
                    reinterpret_cast<const float4*>(item.source)[index / 4];
            else
                for (int lane = 0; lane < 4 && index + lane < item.length; ++lane)
                    item.target[index + lane] = item.source[index + lane];
        } else {
            for (int lane = 0; lane < 4; ++lane) {
                const auto element = start + threadIdx.x + lane * threads;
                if (element < item.length) item.target[element] = item.source[element];
            }
        }
    }
}
'''

_COLLECTIVE_PACK_SOURCE = r'''
PackLaunch launch;
int64_t offset = 0, blocks = 0;
@for(part, 0, IN_SIZE,
    const int64_t count@part = in@part@@_shape0;
    launch.segments[@part].source = in@part@@_p;
    launch.segments[@part].target = out0_p + offset;
    launch.segments[@part].length = count@part;
    launch.segments[@part].first_block = blocks;
    offset += count@part;
    blocks += (count@part + 1023) / 1024;
)
if (blocks) pack_flat_kernel<<<std::min<int64_t>(blocks, 65535), 256>>>(launch, blocks);
'''


@optional_kernel("collective.pack_flat", "cuda", dtypes=("float32",),
                 supports=lambda parts: 2 <= len(parts) <= 64 and
                 not _output_requires_grad(parts) and
                 all(part.ndim == 1 and part._storage_is_contiguous() for part in parts))
def _pack_flat_collective_cuda(parts):
    header = f"constexpr int pack_segment_count = {len(parts)};\n" + _COLLECTIVE_PACK_HEADER
    return jt.code([sum(part.numel() for part in parts)], jt.float32, parts,
                   cuda_header=header, cuda_src=_COLLECTIVE_PACK_SOURCE)


@optional_kernel("misc.repeat_interleave_dim0", _CUDA_CODE_BACKENDS,
                 supports=lambda x, repeats, dim, output_size, cpu_source=None: (
                     isinstance(repeats, jt.Var) and dim == 0 and output_size is not None))
def _repeat_interleave_dim0_cuda(x, repeats, dim, output_size, cpu_source=None):
    # int64 throughout. This used to cast the counts to int32, prefix-sum
    # them in int32 and index the output with an `int`, and cover that with
    # `assert output_size <= 2147483647` -- so the one case the fast path
    # could not do was refused rather than computed. Counting in int64
    # costs a wider prefix sum over one small vector and removes the limit.
    repeats = repeats.reshape(-1).int64()
    n = x.shape[0]
    out0 = int(output_size)
    assert repeats.shape[0] == n, \
        f"repeat_interleave: repeats length {repeats.shape[0]} != dim size {n}"
    if out0 == 0:
        new_shape = list(x.shape); new_shape[0] = 0
        return jt.zeros(new_shape, x.dtype)
    offsets = repeats.cumsum(0)
    inner = int(np.prod(x.shape[1:])) if x.ndim > 1 else 1
    out_shape = list(x.shape)
    out_shape[0] = out0
    return jt.code(
        out_shape,
        x.dtype,
        [x, offsets],
        cuda_header='''
        #include <stdint.h>
        template <typename X, typename R, typename O>
        __global__ void repeat_interleave_dim0_kernel(
            const X* __restrict__ x,
            const R* __restrict__ offsets,
            O* __restrict__ out,
            int64_t total,
            int n,
            int64_t inner) {
            int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
            int64_t stride = (int64_t)blockDim.x * gridDim.x;
            for (; linear < total; linear += stride) {
                // out_row and the offsets it is compared against are the
                // two quantities that count *outputs*, so they are the two
                // that leave int32 first.
                int64_t out_row = linear / inner;
                int lo = 0, hi = n - 1;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if ((int64_t)offsets[mid] > out_row) hi = mid;
                    else lo = mid + 1;
                }
                out[linear] = (O)x[(int64_t)lo * inner + (linear % inner)];
            }
        }
        ''',
        cuda_src=f'''
        @alias(x, in0)
        @alias(offsets, in1)
        @alias(out, out0)
        const int64_t total = out->num;
        const int n = x_shape0;
        const int64_t inner = {inner};
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        if (blocks > 4096) blocks = 4096;
        repeat_interleave_dim0_kernel<x_type, offsets_type, out_type>
            <<<blocks, threads>>>(x_p, offsets_p, out_p, total, n, inner);
        ''',
        cpu_src=cpu_source() if cpu_source is not None else ""
    )


@optional_kernel("misc.stack_no_grad", _CUDA_CODE_BACKENDS)
def _stack_no_grad_cuda_fast(xs, dim, cpu_source=None):
    if _output_requires_grad(xs):
        return None
    n = len(xs)
    if n not in (2, 3):
        return None
    if not xs:
        return None
    for x in xs:
        if not isinstance(x, jt.Var) or getattr(x, "_jittor_torch_force_cpu", False):
            return None
    base_shape = list(xs[0].shape)
    base_dtype = xs[0].dtype
    for x in xs[1:]:
        if list(x.shape) != base_shape or x.dtype != base_dtype:
            return None
    if dim < 0:
        dim += len(base_shape) + 1
    if dim < 0 or dim > len(base_shape):
        return None

    out_shape = base_shape[:dim] + [n] + base_shape[dim:]
    suffix = 1
    for size in base_shape[dim:]:
        suffix *= int(size)
    input_total = 1
    for size in base_shape:
        input_total *= int(size)
    if input_total == 0 or suffix == 0:
        return None
    # A 1-D input is already the flat view the kernel indexes; reshaping it
    # only built one more op per input.
    flat_inputs = list(xs) if len(base_shape) == 1 else [x.reshape([-1]) for x in xs]
    key = (n, suffix, cpu_source)
    sources = _STACK_SOURCES.get(key)
    if sources is None:
        if len(_STACK_SOURCES) >= 256:
            _STACK_SOURCES.clear()
        sources = _STACK_SOURCES[key] = _stack_sources(n, suffix, cpu_source)
    cuda_src, cpu_src = sources
    return _stop_grad_outputs(jt.code(
        [input_total * n], base_dtype, flat_inputs,
        cuda_src=cuda_src, cpu_src=cpu_src).reshape(out_shape))


# (n, suffix, cpu_source) -> (cuda_src, cpu_src) of the stack kernel. Both are
# fixed by the key, and building them was string work on every stack. `suffix`
# follows the input shape, so the table is emptied when it grows large.
_STACK_SOURCES = {}


def _stack_sources(n, suffix, cpu_source):
    write_lines = "\n".join(
        f"            @out(base_out + {i} * suffix + rem) = @in{i}(iid);"
        for i in range(n)
    )
    cuda_src = f"""
    __global__ void stack_kernel(@ARGS_DEF) {{
        @PRECALC
        index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        index_t step = blockDim.x * gridDim.x;
        const index_t suffix = {suffix};
        for (index_t iid = tid; iid < in0_shape0; iid += step) {{
            index_t prefix = iid / suffix;
            index_t rem = iid - prefix * suffix;
            index_t base_out = prefix * ({n} * suffix);
{write_lines}
        }}
    }}
    int block = 256;
    int grid = (in0_shape0 + block - 1) / block;
    if (grid > 65535) grid = 65535;
    stack_kernel<<<grid, block>>>(@ARGS);
    """
    cpu_src = cpu_source(suffix, n, write_lines) if cpu_source is not None else ""
    return cuda_src, cpu_src


def _unbind_cuda_source(n, suffix, write_lines):
    output = f"out{n - 1}_p"
    for i in range(n - 2, -1, -1):
        output = f"part == {i} ? out{i}_p : {output}"
    output_ptrs = " | ".join(f"(size_t)out{i}_p" for i in range(n))
    header = f"""
    namespace jittor {{
    template<typename Pack>
    __global__ static void unbind_packed_kernel(@ARGS_DEF) {{
        constexpr index_t width = sizeof(Pack) / sizeof(in0_type);
        constexpr index_t chunk = {suffix} / (width ? width : 1);
        if constexpr (width > 1 && chunk > 0) {{
            const index_t tile = blockIdx.x / {n};
            const index_t part = blockIdx.x - tile * {n};
            const Pack* packed = reinterpret_cast<const Pack*>(in0_p) + part * chunk;
            Pack* out = reinterpret_cast<Pack*>({output});
            const index_t total = out0_shape0 / width;
            const index_t step = blockDim.x * (gridDim.x / {n});
            for (index_t oid = tile * blockDim.x + threadIdx.x;
                 oid < total; oid += step) {{
                const index_t prefix = oid / chunk;
                const index_t base_in = oid + prefix * (({n} - 1) * chunk);
                out[oid] = packed[base_in];
            }}
        }}
    }}
    __global__ static void unbind_kernel(@ARGS_DEF) {{
        @PRECALC
        const index_t step = blockDim.x * gridDim.x;
        constexpr index_t suffix = {suffix};
        for (index_t oid = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
             oid < out0_shape0; oid += step) {{
            const index_t prefix = oid / suffix;
            const index_t base_in = oid + prefix * (({n} - 1) * suffix);
{write_lines}
        }}
    }}
    }} // namespace jittor
    """
    source = f"""
    constexpr int block = 256;
    constexpr int width16 = 16 / sizeof(in0_type);
    constexpr int width8 = 8 / sizeof(in0_type);
    const size_t alignment = (size_t)in0_p | {output_ptrs};
    if (width16 > 1 && {suffix} % width16 == 0 && (alignment & 15) == 0) {{
        const int grid = std::min<index_t>((out0_shape0 / width16 - 1) / block + 1, 65535);
        unbind_packed_kernel<uint4><<<grid * {n}, block>>>(@ARGS);
    }} else if (width8 > 1 && {suffix} % (width8 ? width8 : 1) == 0 && (alignment & 7) == 0) {{
        const int grid = std::min<index_t>((out0_shape0 / (width8 ? width8 : 1) - 1) / block + 1, 65535);
        unbind_packed_kernel<uint2><<<grid * {n}, block>>>(@ARGS);
    }} else {{
        const int grid = std::min<index_t>((out0_shape0 - 1) / block + 1, 65535);
        unbind_kernel<<<grid, block>>>(@ARGS);
    }}
    """
    return header, source


@optional_kernel("misc.unbind_no_grad", _CUDA_CODE_BACKENDS)
def _unbind_no_grad_cuda_fast(x, dim, cpu_source=None):
    if _output_requires_grad(x):
        return None
    if not isinstance(x, jt.Var) or getattr(x, "_jittor_torch_force_cpu", False):
        return None
    # Flattening a strided source materializes it before the copy kernel
    # below copies it again. Native indexing can retain each slice's storage
    # strides directly, including a permuted packed Q/K/V tensor.
    if not x._storage_is_contiguous():
        return None
    shape = list(x.shape)
    if not shape:
        return None
    if dim < 0:
        dim += len(shape)
    if dim < 0 or dim >= len(shape):
        return None
    n = int(shape[dim])
    if n not in (2, 3):
        return None
    out_shape = shape[:dim] + shape[dim + 1:]
    suffix = 1
    for size in shape[dim + 1:]:
        suffix *= int(size)
    out_total = 1
    for size in out_shape:
        out_total *= int(size)
    if out_total < 4096 or suffix == 0:
        return None
    if n == 2 and out_total < 1024 * 1024:
        return None

    flat = x.reshape([-1])
    write_lines = "\n".join(
        f"            @out{i}(oid) = @in0(base_in + {i} * suffix);"
        for i in range(n)
    )
    cuda_header, cuda_src = _unbind_cuda_source(n, suffix, write_lines)
    cpu_src = cpu_source(suffix, n, write_lines) if cpu_source is not None else ""
    outs = jt.code(
        [[out_total] for _ in range(n)],
        [x.dtype for _ in range(n)],
        [flat],
        cuda_header=cuda_header,
        cuda_src=cuda_src,
        cpu_src=cpu_src,
    )
    return _stop_grad_outputs([out.reshape(out_shape) for out in outs])


def _unique_code_cuda(*args, **kwargs):
    with jt.flag_scope(compile_options={"FLAGS:  --extended-lambda ": 1}):
        return jt.code(*args, **kwargs)


def _scan_2d_cuda(x, reverse):
    operations = get_library_ops("cub", load=True)
    if operations is None:
        raise RuntimeError("CUB is unavailable for CUDA cumsum")
    # float16/bfloat16 scan through a float32 buffer and come back narrow.
    # `cub_cumsum_op.cc` instantiates `cub::BlockScan<Tx, BLOCK_THREADS>` and
    # `DeviceScan::InclusiveSum` at the *element* type, so a half input is
    # accumulated in half: every partial is rounded, and a scan keeps every
    # partial, so the error grows along the axis. `jt.cumsum` over 4096 ones
    # answered 4092 in float16 and 3968 in bfloat16 -- the running sum stops
    # moving once it passes 2048 (resp. 256), which is where adding one no
    # longer changes the value. torch runs the scan in `acc_type<T>`, which is
    # float for both half types, and answers 4096; on a random input it
    # measured 1.78e-3 from the float64 reference where this was 3.34e-2.
    #
    # Done here rather than in the kernel because the CUB block scan's
    # accumulator type is the same template parameter as its storage type;
    # separating them is a rewrite of the kernel, and the extra float32 buffer
    # buys the same answer. The CPU kernel (`jittor/ops/scan.py::_scan_2d_cpu`)
    # does carry a separate accumulator and needs no buffer.
    from jittor._core.dtypes import dtype_name as _dtype_name
    dtype = x.dtype
    if _dtype_name(dtype) in ("float16", "bfloat16"):
        return operations.cub_cumsum(x.float32(), reverse).cast(dtype)
    return operations.cub_cumsum(x, reverse)


def unique_sort_header():
    return ('''
        #undef out
        #include <thrust/extrema.h>
        #include <thrust/device_ptr.h>
        #include <thrust/execution_policy.h>
        #include <thrust/device_vector.h>
        #include <thrust/sequence.h>

        #include <thrust/sequence.h>
        #include <thrust/sort.h>
        #include <thrust/unique.h>

        #include <cub/cub.cuh>
        #include <core/executor.h>
        '''
    )



def unique_sort_source():
    return ('''
            @alias(input_flatten, in0)
            @alias(indice, out)
            int dimlen = indice_shape0, dimsize = input_flatten_shape1;

            if (dimsize == 1) {
                size_t raw_allocation, d_allocation, temp_storage_bytes = 0;
                void *d_temp_storage = NULL;
                // Two allocations, not one block carved by hand. The old
                // code put the sorted keys at `raw_ptr + dimlen` -- 4*dimlen
                // bytes in, which is 8-byte aligned only when dimlen is
                // even, so an int64 or float64 input of odd length handed
                // cub a misaligned buffer. Carving the other way round
                // misaligns the int32 iota for 1- and 2-byte keys. Let the
                // allocator align each.
                size_t keys_bytes = dimlen * sizeof(input_flatten_type);
                size_t iota_bytes = dimlen * sizeof(int32_t);
                input_flatten_type* keys_out = (input_flatten_type*)runtime_executor().allocator->alloc(keys_bytes, raw_allocation);
                size_t iota_allocation;
                int32_t* raw_ptr = (int32_t*)runtime_executor().allocator->alloc(iota_bytes, iota_allocation);

                thrust::device_ptr<int32_t> arange_ptr = thrust::device_pointer_cast(raw_ptr);
                thrust::sequence(arange_ptr, arange_ptr + dimlen);

                cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input_flatten_p,
                                                keys_out, thrust::raw_pointer_cast(arange_ptr), indice_p, dimlen);
                d_temp_storage = runtime_executor().allocator->alloc(temp_storage_bytes, d_allocation);
                cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input_flatten_p,
                                                keys_out, thrust::raw_pointer_cast(arange_ptr), indice_p, dimlen);

                runtime_executor().allocator->free(raw_ptr, iota_bytes, iota_allocation);
                runtime_executor().allocator->free(keys_out, keys_bytes, raw_allocation);
                runtime_executor().allocator->free(d_temp_storage, temp_storage_bytes, d_allocation);
            } else {
                thrust::device_ptr<input_flatten_type> input_ptr = thrust::device_pointer_cast(input_flatten_p);
                thrust::device_ptr<int32_t> indice_ptr = thrust::device_pointer_cast(indice_p);

                thrust::sequence(indice_ptr, indice_ptr + dimlen);
                thrust::sort(thrust::device, indice_ptr, indice_ptr + dimlen,
                    [=] __device__ (int32_t a, int32_t b)->bool {
                        for(int i = 0; i < dimsize; ++i) {
                            input_flatten_type lhs = input_ptr[i + a * dimsize],
                                            rhs = input_ptr[i + b * dimsize];
                            if (lhs != rhs) return lhs < rhs;
                        }
                        return false;
                    });
            }
        '''
    )



def unique_compact_header():
    return ('''
            #undef out

            #include <thrust/extrema.h>
            #include <thrust/device_ptr.h>
            #include <thrust/execution_policy.h>

            #include <thrust/sequence.h>
            #include <thrust/unique.h>
            #include <thrust/sort.h>

            #include <thrust/scan.h>
            #include <thrust/fill.h>
            #include <core/executor.h>

            @alias(input_sorted, in0)
            @alias(diff, in1)
            @alias(indice, in2)
            @alias(output, out0)
            @alias(inverse, out1)
        '''
    )



def unique_compact_source(need_inverse):
    return (f"bool return_inverse = {int(need_inverse)};" +
        '''
            int dimlen = input_sorted_shape0, dimsize = input_sorted_shape1;
            size_t raw_allocation;
            int32_t* raw_ptr = (int32_t*)runtime_executor().allocator->alloc(2 * dimlen * sizeof(int), raw_allocation);

            thrust::device_ptr<int32_t> diff_ptr = thrust::device_pointer_cast(diff_p),
                                        inverse_ptr = thrust::device_pointer_cast(inverse_p),
                                        array_ptr = thrust::device_pointer_cast(raw_ptr),
                                        sum_ptr = thrust::device_pointer_cast(raw_ptr + dimlen),
                                        indice_ptr = thrust::device_pointer_cast(indice_p);
            thrust::device_ptr<input_sorted_type> input_ptr = thrust::device_pointer_cast(input_sorted_p);

            // diff has dimlen - 1 entries: diff[i] is whether row i + 1 starts
            // a new value. Row 0 is value 0.
            if (return_inverse && dimlen) {
                thrust::fill_n(sum_ptr, 1, 0);
                thrust::inclusive_scan(diff_ptr, diff_ptr + dimlen - 1, sum_ptr + 1);
                thrust::scatter(sum_ptr, sum_ptr + dimlen, indice_ptr, inverse_ptr);
            }

            thrust::sequence(array_ptr, array_ptr + dimlen);
            int32_t num = thrust::unique(array_ptr, array_ptr + dimlen,
                [=] __device__ (int32_t a, int32_t b)->bool {
                    for(int i = 0; i < dimsize; ++i) {
                        input_sorted_type lhs = input_ptr[i + a * dimsize],
                                        rhs = input_ptr[i + b * dimsize];
                        if (lhs != rhs) return false;
                    }
                    return true;
                }) - array_ptr;

            cudaMemcpy(output_p, raw_ptr, sizeof(int32_t) * num, cudaMemcpyDeviceToDevice);
            runtime_executor().allocator->free(raw_ptr, 2 * dimlen * sizeof(int32_t), raw_allocation);
            output->set_shape({ num });
        '''
    )



_SCATTER_ADD_DIM0_HEADER = r'''
__global__ static void scatter_init_copy_f32(float* __restrict__ dst, const float* __restrict__ src, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; i < n; i += stride) dst[i] = src[i];
}

template <typename IdxT>
__global__ static void scatter_add_rows_f32(
        float* __restrict__ out,
        const IdxT* __restrict__ idx,
        const float* __restrict__ src,
        int64_t E, int64_t N, int64_t C) {
    int64_t total = E * C;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; i < total; i += stride) {
        int64_t row = (C == 1) ? i : (i / C);
        int64_t col = (C == 1) ? 0 : (i - row * C);
        int64_t dst = static_cast<int64_t>(idx[row]);
        if (dst >= 0 && dst < N) {
            atomicAdd(out + dst * C + col, src[i]);
        }
    }
}
'''

_SCATTER_ADD_DIM0_SOURCE = r'''
int64_t n_out = out0->num;
if (n_out > 0) {
    int blocks = (int)std::min<int64_t>((n_out + 255) / 256, 65535);
    scatter_init_copy_f32<<<blocks, 256>>>(out0_p, in0_p, n_out);
}
int64_t E = in1->shape[0];
int64_t N = out0->shape[0];
int64_t C = out0->shape.size() > 1 ? out0->shape[1] : 1;
int64_t total = E * C;
if (total > 0 && N > 0) {
    int blocks = (int)std::min<int64_t>((total + 255) / 256, 65535);
    scatter_add_rows_f32<<<blocks, 256>>>(out0_p, in1_p, in2_p, E, N, C);
}
'''


def _extract_row_index_1d(index):
    if index.ndim == 1:
        return index
    if index.ndim != 2:
        return None
    if index._producer_name() == "broadcast_to":
        base = index._input(0)
        if base.ndim == 2 and int(base.shape[0]) == int(index.shape[0]) and int(base.shape[1]) == 1:
            return base.reshape((-1,))
    if index._is_view():
        strides = tuple(index._storage_strides())
        if len(strides) == 2 and strides[1] == 0:
            return index[:, 0]
    return None


def _supports_scatter_add_dim0(x, dim, index, src, reduce="void"):
    if reduce != "add" or dim != 0:
        return False
    if not (isinstance(x, jt.Var) and isinstance(index, jt.Var) and isinstance(src, jt.Var)):
        return False
    if _output_requires_grad(x, src):
        return False
    if _jittor_dtype_name(x.dtype) != "float32" or _jittor_dtype_name(src.dtype) != "float32":
        return False
    if _jittor_dtype_name(index.dtype) not in ("int32", "int64"):
        return False
    if x.ndim not in (1, 2) or index.ndim != x.ndim or src.ndim != x.ndim:
        return False
    if tuple(int(s) for s in src.shape) != tuple(int(s) for s in index.shape):
        return False
    if x.ndim == 2 and int(index.shape[1]) != int(x.shape[1]):
        return False
    return _extract_row_index_1d(index) is not None


@optional_kernel("tensor.scatter", _CUDA_CODE_BACKENDS, supports=_supports_scatter_add_dim0)
def _scatter_add_dim0_cuda(x, dim, index, src, reduce="void"):
    idx_1d = _extract_row_index_1d(index)
    out = jt.code(
        x.shape, x.dtype, [x, idx_1d, src],
        cuda_header=_SCATTER_ADD_DIM0_HEADER,
        cuda_src=_SCATTER_ADD_DIM0_SOURCE,
    )
    return _stop_grad_outputs(out)


for _backend in _CUDA_CODE_BACKENDS:
    register_kernel("misc.unique_code", _backend, _unique_code_cuda)
    register_kernel("misc.scan_2d", _backend, _scan_2d_cuda)
del _backend
