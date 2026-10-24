"""CUDA fast paths for bounded dense FP32 last-axis means and sums.

One output row owns one thread-x group; there is no cross-block atomic fold.
Unsupported layouts, types and shapes remain on the existing native reduction.
"""
from functools import lru_cache
import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._runtime.dispatch import optional_kernel


def _product(shape):
    result = 1
    for size in shape:
        result *= size
    return result

def _last_axis_geometry(rows, columns):
    if type(rows) is not int or type(columns) is not int or not (rows > 0 and 128 <= columns <= 4096):
        raise ValueError('bounded positive last-axis geometry required')
    if rows * columns > (1 << 31) - 1:
        raise ValueError('32-bit logical indexing required')
    if columns == 128 and rows * columns * 4 > (1 << 31) - 1:
        raise ValueError('H128 FP32 byte indexing must not require iterator splitting')
    # H128 is NOT input-vectorized in Torch2.8 ReduceConfig (dim0 >128).
    groups = columns if columns == 128 else columns // 4
    power = lambda value: 1 << (value.bit_length() - 1)
    column_budget = min(power(groups), 512)
    row_budget = min(power(rows), 512)
    width = min(column_budget, 32)
    height = min(row_budget, 512 // width)
    width = min(column_budget, 512 // height)
    return width, height

@lru_cache(maxsize=32)
def _last_axis_mean_cuda_cls(width, height, keepdims, mean=True):
    if type(mean) is not bool:
        raise TypeError("mean mode must be an exact bool")
    class LastAxisMeanCUDA(jt.Function):
        def execute(self, x):
            self.input_shape = tuple(int(size) for size in x.shape)
            self.columns = self.input_shape[-1]
            rows = _product(self.input_shape[:-1])
            if not mean and self.columns != 128:
                raise ValueError("last-axis sum is restricted to H128")
            if _last_axis_geometry(rows, self.columns) != (width, height):
                raise ValueError('actual graph shape/launch mismatch')
            output_shape = self.input_shape[:-1] + ((1,) if keepdims else ())
            # Shared with native ReduceOp, including implicit broadcast VJP.
            source = r"""
                const int columns=in0->shape[in0->shape.size()-1];
                const int rows=in0->num/columns;
                jittor_ordered_sum::launch_last_fp32(in0_p,out0_p,rows,columns,MEAN_MODE);
                CHECK(0 == cudaGetLastError());
            """
            result = jt.code((rows,), 'float32', [x],
                cuda_header='#include "ordered_nonlast_sum.h"',
                cuda_src=source.replace('MEAN_MODE', 'true' if mean else 'false'))
            return result.reshape(output_shape)

        def grad(self, gradient):
            if gradient is None:
                return None
            if mean:
                return gradient.reshape(self.input_shape[:-1] + (1,)).broadcast(self.input_shape) / float(self.columns)
            return gradient.reshape(self.input_shape[:-1] + (1,)).broadcast(self.input_shape)

    return LastAxisMeanCUDA


def _supports_last_axis_mean(x, keepdims=False):
    if not isinstance(x, jt.Var) or type(keepdims) is not bool:
        return False
    shape = tuple(int(size) for size in x.shape)
    if not shape or any(size <= 0 for size in shape):
        return False
    if not 128 <= shape[-1] <= 4096 or _product(shape) > (1 << 31) - 1:
        return False
    if shape[-1] == 128 and _product(shape) * 4 > (1 << 31) - 1:
        return False
    return x._storage_is_contiguous()


@optional_kernel("tensor.mean.last_axis", "cuda", dtypes=("float32",),
                 supports=_supports_last_axis_mean)
def _last_axis_mean_cuda(x, keepdims=False):
    shape = tuple(int(size) for size in x.shape)
    width, height = _last_axis_geometry(_product(shape[:-1]), shape[-1])
    return _last_axis_mean_cuda_cls(width, height, keepdims).apply(x)


def _supports_large_last_axis_sum(x, keepdims=False):
    if not isinstance(x, jt.Var) or type(keepdims) is not bool:
        return False
    if _jittor_dtype_name(x.dtype) != 'float32':
        return False
    shape = tuple(int(size) for size in x.shape)
    if not shape or any(size <= 0 for size in shape):
        return False
    rows, columns = _product(shape[:-1]), shape[-1]
    return (1 <= rows <= 64 and (1 << 17) <= columns <= (1 << 18)
            and _product(shape) * 4 <= (1 << 31) - 1
            and x._storage_is_contiguous())


@lru_cache(maxsize=4)
def _large_last_axis_sum_cuda_cls(keepdims):
    """Four FP32 lanes, Torch2.8-compatible CTA fold, no atomic accumulation.

    The supported large rows use all 512 threads for each output. The logical
    input split is independent of its 2-D threadblock spelling: block-y folds
    then block-x folds form the same 512-lane tree. CUDA properties determine
    the CTA split; no machine/device ordinal is encoded in the calculation.
    """
    if type(keepdims) is not bool:
        raise TypeError('keepdims must be an exact bool')

    class LargeLastAxisSumCUDA(jt.Function):
        def execute(self, x):
            if not _supports_large_last_axis_sum(x, keepdims):
                raise ValueError('bounded dense FP32 large-row sum required')
            self.input_shape = tuple(int(size) for size in x.shape)
            columns = self.input_shape[-1]
            rows = _product(self.input_shape[:-1])
            partial_cap = (((columns + 511) // 512) + 15) // 16
            source = r'''
                __device__ __forceinline__ float large_sum_fold512(
                    float value, float* exchange) {
                    const int lane = threadIdx.x;
                    exchange[lane] = value;
                    for (int offset = 256; offset >= 32; offset >>= 1) {
                        __syncthreads();
                        if (lane < offset) {
                            value = value + exchange[lane + offset];
                            exchange[lane] = value;
                        }
                    }
                    __syncthreads();
                    for (int offset = 1; offset < 32; offset <<= 1)
                        value = value + __shfl_down_sync(0xffffffffu, value, offset);
                    return value;
                }

                __global__ static void large_sum_partials(
                    const float* input, float* partial, int columns, int cap) {
                    const int lane = threadIdx.x;
                    const int row = blockIdx.x;
                    const float* first = input + ((int64_t)row) * columns;
                    const int offset = (reinterpret_cast<uintptr_t>(first) & 15u) / sizeof(float);
                    const int head = offset ? 4 - offset : 0;
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                    if (blockIdx.y == 0 && offset && lane >= offset && lane < 4)
                        a0 = a0 + first[lane - offset];
                    const float* aligned = first + head;
                    const int remaining = columns - head;
                    const int stride = 512 * gridDim.y;
                    for (int index = lane + blockIdx.y * 512;
                         index * 4 + 3 < remaining; index += stride) {
                        const float4 value = reinterpret_cast<const float4*>(aligned)[index];
                        a0 = a0 + value.x;
                        a1 = a1 + value.y;
                        a2 = a2 + value.z;
                        a3 = a3 + value.w;
                    }
                    const int tail = remaining - remaining % 4;
                    if (blockIdx.y == 0 && tail + lane < remaining)
                        a0 = a0 + aligned[tail + lane];
                    float value = ((a0 + a1) + a2) + a3;
                    __shared__ float exchange[512];
                    value = large_sum_fold512(value, exchange);
                    if (lane == 0) partial[row * cap + blockIdx.y] = value;
                }

                __global__ static void large_sum_finish(
                    const float* partial, float* output, int cap, int ctas) {
                    const int lane = threadIdx.x;
                    const int row = blockIdx.x;
                    float value = 0.0f;
                    if (lane < ctas) value = value + partial[row * cap + lane];
                    __shared__ float exchange[512];
                    value = large_sum_fold512(value, exchange);
                    if (lane == 0) output[row] = value;
                }

                const int columns = in0->shape[in0->shape.size()-1];
                const int rows = in0->num / columns;
                const int cap = PARTIAL_CAP;
                int ordinal = -1;
                CHECK(cudaSuccess == cudaGetDevice(&ordinal));
                static thread_local int cached_ordinal = -1;
                static thread_local cudaDeviceProp properties;
                if (cached_ordinal != ordinal) {
                    CHECK(cudaSuccess == cudaGetDeviceProperties(&properties, ordinal));
                    cached_ordinal = ordinal;
                }
                const int target_grid = properties.multiProcessorCount
                    * (properties.maxThreadsPerMultiProcessor / 512);
                const int values_per_thread = (columns + 511) / 512;
                int ctas = 1;
                if (values_per_thread >= 256 && rows <= target_grid) {
                    const int capacity = (target_grid + rows - 1) / rows;
                    const int minimum_work = (values_per_thread + 15) / 16;
                    const int maximum_work = (values_per_thread + 255) / 256;
                    ctas = std::max(std::min(capacity, minimum_work), maximum_work);
                }
                CHECK(ctas >= 1 && ctas <= cap);
                large_sum_partials<<<dim3(rows, ctas), 512>>>(in0_p, out1_p, columns, cap);
                CHECK(0 == cudaGetLastError());
                large_sum_finish<<<rows, 512>>>(out1_p, out0_p, cap, ctas);
                CHECK(0 == cudaGetLastError());
                '''
            output, _partial = jt.code(
                [(rows,), (rows, partial_cap)], ['float32', 'float32'], [x],
                cuda_header='#include <stdint.h>\n#include <algorithm>',
                cuda_src=source.replace('PARTIAL_CAP', str(partial_cap)))
            output_shape = self.input_shape[:-1] + ((1,) if keepdims else ())
            return output.reshape(output_shape)

        def grad(self, gradient):
            if gradient is None:
                return None
            return gradient.reshape(self.input_shape[:-1] + (1,)).broadcast(self.input_shape)

    return LargeLastAxisSumCUDA


def _supports_bounded_last_axis_sum(x, keepdims=False):
    # Preserve the optional Python H128 contract. Public sums of width32..127
    # still reach the broader native ReduceOp/OrderedSum capability, as do
    # inverse-broadcast gradients; declining here is not backend fallback.
    if not isinstance(x, jt.Var) or type(keepdims) is not bool:
        return False
    shape = tuple(int(size) for size in x.shape)
    return (bool(shape) and all(size > 0 for size in shape)
            and _jittor_dtype_name(x.dtype) == 'float32'
            and shape[-1] == 128 and _product(shape)*4 <= (1 << 31)-1
            and x._storage_is_contiguous()
            and 'ordered_sum' in jt.core.backend_supported_capabilities('cuda'))


def _supports_last_axis_sum(x, keepdims=False):
    return (_supports_bounded_last_axis_sum(x, keepdims)
            or _supports_large_last_axis_sum(x, keepdims))

@optional_kernel("tensor.sum.last_axis", "cuda", dtypes=("float32",),
                 supports=_supports_last_axis_sum)
def _last_axis_sum_cuda(x, keepdims=False):
    if _supports_large_last_axis_sum(x, keepdims):
        return _large_last_axis_sum_cuda_cls(keepdims).apply(x)
    shape = tuple(int(size) for size in x.shape)
    axis = 1 << (len(shape)-1)
    return jt.ops.ordered_sum(x, axis, axis if keepdims else 0)


__all__ = ["_last_axis_mean_cuda", "_last_axis_sum_cuda"]


def _supports_large_last_axis_logsumexp(x, dim, keepdim=False):
    return (type(dim) is int and bool(x.shape)
            and dim in (-1, len(x.shape) - 1) and type(keepdim) is bool
            and _supports_large_last_axis_sum(x, keepdims=keepdim))


@lru_cache(maxsize=1)
def _large_last_axis_logsumexp_cuda_cls():
    class LogsumexpLastF32(jt.Function):
        def execute(self,x):
            if not _supports_large_last_axis_sum(x, keepdims=True):
                raise ValueError('bounded dense FP32 CUDA last-axis input required')
            maximum=x.max(-1,keepdims=True)
            # Match the reference's infinite-maximum stabilization: finite
            # inputs keep the existing forward bytes; all +/-inf rows return
            # +/-inf instead of the composite's inf-inf NaN.
            maximum=jt.where(maximum.abs()==float('inf'),jt.zeros_like(maximum),maximum)
            self.x=x
            self.result=maximum+(x-maximum).exp().sum(-1,keepdims=True).log()
            return self.result

        def grad(self,dy):
            if dy is None:
                return None
            return dy*(self.x-self.result).exp()
    return LogsumexpLastF32



@optional_kernel("nn.logsumexp", "cuda", dtypes=("float32",),
                 supports=_supports_large_last_axis_logsumexp)
def _large_last_axis_logsumexp_cuda(x, dim, keepdim=False):
    result = _large_last_axis_logsumexp_cuda_cls().apply(x)
    return result if keepdim else result.reshape(tuple(x.shape[:-1]))
