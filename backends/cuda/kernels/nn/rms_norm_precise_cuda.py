"""CUDA weighted RMS normalization with explicit FP32 rounding boundaries.

The row reductions use four independent accumulators and a fixed shuffle tree.
The same primitive owns the saved variance and the VJP's dot product. Frozen
affine weights retain input-dtype rounding before their ordinary promotion.
"""

from functools import lru_cache
import math

from jittor_core import Var
from jittor._core.function import Function
from jittor._core.flags import _output_requires_grad
from jittor._runtime.dispatch import optional_kernel


def _block_width(length, rows):
    # Match a last-axis FP32 reduction's logical row grouping. The kernel
    # launches one row per block; row grouping only selects its addition tree.
    columns = length // 4 if length > 128 else length
    column_power = min(512, 1 << (columns.bit_length() - 1))
    row_power = min(512, 1 << (rows.bit_length() - 1))
    initial_width = min(column_power, 32)
    height = min(row_power, 512 // initial_width)
    return min(column_power, 512 // height)


@lru_cache(maxsize=512)
def _row_reduce_cls(length, width, mean):
    import jittor as jt

    vectorized = int(length > 128)
    factor = "1.0f / %d.0f" % length if mean else "1.0f"

    class RowReduce(jt.Function):
        def execute(self, x):
            self.shape = x.shape
            return jt.code(x.shape[:-1] + [1], "float32", [x], cuda_src=f'''
            __global__ static void rms_row_reduce(
                    const float* x, float* output, int rows) {{
                __shared__ float partial[{width}];
                int lane = threadIdx.x, row = blockIdx.x;
                if (lane >= {width}) return;
                float value[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
                int start = 0, end = {length};
                if ({vectorized}) {{
                    // The virtual FP32 product is aligned at its allocation
                    // origin; individual non-multiple-of-four rows need a head.
                    int misalignment = (row * {length}) & 3;
                    if (misalignment) {{
                        if (lane >= misalignment && lane < 4)
                            value[0] = x[row * {length} + lane - misalignment];
                        start = 4 - misalignment;
                        end -= start;
                    }}
                    for (int index = lane; index * 4 + 3 < end; index += {width}) {{
                        #pragma unroll
                        for (int item = 0; item < 4; item++)
                            value[item] = __fadd_rn(value[item],
                                x[row * {length} + start + index * 4 + item]);
                    }}
                    int tail = end - end % 4;
                    if (lane < end - tail)
                        value[0] = __fadd_rn(value[0],
                            x[row * {length} + start + tail + lane]);
                }} else {{
                    for (int index = lane; index < end; index += 4 * {width}) {{
                        #pragma unroll
                        for (int item = 0; item < 4; item++)
                            if (index + item * {width} < end)
                                value[item] = __fadd_rn(value[item],
                                    x[row * {length} + index + item * {width}]);
                    }}
                }}
                float total = __fadd_rn(__fadd_rn(__fadd_rn(
                    value[0], value[1]), value[2]), value[3]);
                partial[lane] = total;
                __syncthreads();
                for (int offset = {width} / 2; offset >= 32; offset >>= 1) {{
                    if (lane < offset)
                        total = __fadd_rn(total, partial[lane + offset]);
                    partial[lane] = total;
                    __syncthreads();
                }}
                if (lane < 32) {{
                    unsigned mask = __activemask();
                    for (int offset = 1; offset < {min(width, 32)}; offset <<= 1)
                        total = __fadd_rn(total,
                            __shfl_down_sync(mask, total, offset));
                    if (lane == 0) output[row] = total * ({factor});
                }}
            }}
            int rows = in0->num / {length};
            rms_row_reduce<<<rows, {max(width, 32)}>>>(in0_p, out0_p, rows);
            CHECK(0 == cudaGetLastError());
            ''')

        def grad(self, gradient):
            with jt.flag_scope(amp_reg=jt.amp_flags.disabled):
                if mean:
                    gradient = gradient / length
                return gradient.broadcast(self.shape)

    return RowReduce


def _row_reduce(x, mean=False):
    length = int(x.shape[-1])
    rows = int(x.numel()) // length
    return _row_reduce_cls(length, _block_width(length, rows), mean)()(x)


class _ReciprocalSqrt(Function):
    def _run_call(self, x):
        import jittor as jt

        self.x = jt.ops.reshape(x, x.shape)
        return super()._run_call(x)

    def execute(self, x):
        import jittor as jt

        return jt.code(x.shape, "float32", [x], cuda_src=r'''
        __global__ static void rms_rsqrt(const float* x, float* y, int size) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < size) y[index] = rsqrtf(x[index]);
        }
        int size = in0->num;
        rms_rsqrt<<<(size + 255) / 256, 256>>>(in0_p, out0_p, size);
        CHECK(0 == cudaGetLastError());
        ''')

    def grad(self, gradient):
        import jittor as jt

        with jt.flag_scope(amp_reg=jt.amp_flags.disabled):
            # Re-evaluation retains the original graph edge for higher AD.
            value = _ReciprocalSqrt()(self.x)
            return (gradient * -.5) * ((value * value) * value)


class _SingleSquare(Function):
    """One AD edge for x**2, with the exact multiply used by its primal."""

    def _run_call(self, x):
        import jittor as jt

        self.x = jt.ops.reshape(x, x.shape)
        return super()._run_call(x)

    def execute(self, x):
        return x * x

    def grad(self, gradient):
        return gradient * (2 * self.x)


@lru_cache(maxsize=128)
def _ordered_broadcast_cls(shape):
    class OrderedBroadcast(Function):
        def execute(self, value):
            return value.broadcast(shape)

        def grad(self, gradient):
            return _row_reduce(gradient)

    return OrderedBroadcast


@lru_cache(maxsize=128)
def _precise_rms_cls(length, epsilon):
    class PreciseRMS:
        def __call__(self, x, weight):
            import jittor as jt

            with jt.flag_scope(amp_reg=jt.amp_flags.disabled):
                wide = x.float32()
                # Keep the direct and variance edges separate: a caller's
                # residual cotangent may arrive between them. Collapsing the
                # whole VJP changes the rounding of that three-term sum.
                variance = _row_reduce(_SingleSquare()(wide), mean=True)
                inverse = _ReciprocalSqrt()(variance + epsilon)
                expanded = _ordered_broadcast_cls(tuple(wide.shape))()(inverse)
                return weight * (wide * expanded).cast(x.dtype)

    return PreciseRMS


def _supports_precise_rms(x, weight, epsilon=1e-6):
    if not isinstance(x, Var) or not isinstance(weight, Var):
        return False
    if (_output_requires_grad(weight) or not x.shape
            or any(int(size) <= 0 for size in x.shape)
            or int(x.shape[-1]) > 4096 or x.numel() > 2147483647
            or tuple(weight.shape) != (int(x.shape[-1]),)
            or not x._storage_is_contiguous() or not weight._storage_is_contiguous()):
        return False
    try:
        value = float(epsilon)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(value) and value >= 0


@optional_kernel("nn.rms_norm.precise", "cuda",
                 dtypes=("float16", "bfloat16", "float32"),
                 supports=_supports_precise_rms)
def _rms_norm_precise_cuda(x, weight, epsilon=1e-6):
    return _precise_rms_cls(int(x.shape[-1]), float(epsilon))()(x, weight)
