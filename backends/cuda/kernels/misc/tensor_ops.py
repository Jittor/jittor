"""CUDA tensor kernels; CPU fallback sources are supplied by native callers."""

import jittor as jt
import numpy as np

from jittor._runtime.dispatch import optional_kernel, register_kernel
from jittor._runtime.backend_libraries import get_library_ops
from jittor._core.flags import _output_requires_grad, _stop_grad_outputs


_CUDA_CODE_BACKENDS = ("cuda", "rocm_legacy", "corex_legacy")


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
    flat_inputs = [x.reshape([-1]) for x in xs]
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
    return _stop_grad_outputs(jt.code(
        [input_total * n], base_dtype, flat_inputs,
        cuda_src=cuda_src, cpu_src=cpu_src).reshape(out_shape))


@optional_kernel("misc.unbind_no_grad", _CUDA_CODE_BACKENDS)
def _unbind_no_grad_cuda_fast(x, dim, cpu_source=None):
    if _output_requires_grad(x):
        return None
    if not isinstance(x, jt.Var) or getattr(x, "_jittor_torch_force_cpu", False):
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
    cuda_src = f"""
    __global__ void unbind_kernel(@ARGS_DEF) {{
        @PRECALC
        index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        index_t step = blockDim.x * gridDim.x;
        const index_t suffix = {suffix};
        const index_t full_stride = suffix * {n};
        for (index_t oid = tid; oid < out0_shape0; oid += step) {{
            index_t prefix = oid / suffix;
            index_t rem = oid - prefix * suffix;
            index_t base_in = prefix * full_stride + rem;
{write_lines}
        }}
    }}
    int block = 256;
    int grid = (out0_shape0 + block - 1) / block;
    if (grid > 65535) grid = 65535;
    unbind_kernel<<<grid, block>>>(@ARGS);
    """
    cpu_src = cpu_source(suffix, n, write_lines) if cpu_source is not None else ""
    outs = jt.code(
        [[out_total] for _ in range(n)],
        [x.dtype for _ in range(n)],
        [flat],
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
        #include <executor.h>
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
            #include <executor.h>

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

            if (return_inverse) {
                thrust::inclusive_scan(diff_ptr, diff_ptr + dimlen, sum_ptr);
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



for _backend in _CUDA_CODE_BACKENDS:
    register_kernel("misc.unique_code", _backend, _unique_code_cuda)
    register_kernel("misc.scan_2d", _backend, _scan_2d_cuda)
del _backend
