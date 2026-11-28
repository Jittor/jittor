"""CUDA bilinear sampling for float32 inference with zero padding."""

from functools import lru_cache

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.dispatch import dispatch_context, register_kernel
from jittor._core.dtypes import dtype_name


def _supports(input, grid, mode, padding_mode, align_corners):
    return (mode == "bilinear" and padding_mode == "zeros"
            and type(align_corners) is bool
            and input.ndim == grid.ndim == 4
            and input.shape[0] == grid.shape[0] and grid.shape[-1] == 2
            and all(size > 0 for size in input.shape)
            and all(size > 0 for size in grid.shape)
            and not _output_requires_grad(input, grid))


_HEADER = r"""
__device__ static inline float zero_padded_pixel(
        const float* image, int64_t y, int64_t x, int64_t height, int64_t width) {
    return x >= 0 && x < width && y >= 0 && y < height
        ? image[y * width + x] : 0.0f;
}

template<bool align_corners>
__device__ static inline float grid_coordinate(float coord, int64_t size) {
    if (align_corners) return ((coord + 1.0f) / 2.0f) * float(size - 1);
    return ((coord + 1.0f) * float(size) - 1.0f) / 2.0f;
}

template<bool align_corners>
__device__ static inline float sample_bilinear(
        const float* image, const float* location, int64_t height, int64_t width) {
    const float x = grid_coordinate<align_corners>(location[0], width);
    const float y = grid_coordinate<align_corners>(location[1], height);
    float value = 0.0f;
    if (!isfinite(x) || !isfinite(y)) {
        value = NAN;
    } else if (x >= -1.0f && x < float(width) && y >= -1.0f && y < float(height)) {
        const int64_t left = int64_t(floorf(x)), top = int64_t(floorf(y));
        const int64_t right = left + 1, bottom = top + 1;
        const float dx = x - float(left), dy = y - float(top);
        const float dnx = float(right) - x, dny = float(bottom) - y;
        value = zero_padded_pixel(image, top, left, height, width) * dnx * dny
              + zero_padded_pixel(image, bottom, left, height, width) * dnx * dy
              + zero_padded_pixel(image, top, right, height, width) * dx * dny
              + zero_padded_pixel(image, bottom, right, height, width) * dx * dy;
    }
    return value;
}

template<bool align_corners>
__global__ static void bilinear_grid_sample(
        const float* input, const float* grid, float* output,
        int64_t channels, int64_t height, int64_t width, int64_t points,
        int64_t rows) {
    const int64_t first_point = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t point_stride = int64_t(gridDim.x) * blockDim.x;
    for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const float* image = input + row * height * width;
        const float* locations = grid + (row / channels) * points * 2;
        for (int64_t point = first_point; point < points; point += point_stride) {
            const float value = sample_bilinear<align_corners>(
                image, locations + point * 2, height, width);
            output[row * points + point] = value;
        }
    }
}
"""

_LAUNCH = r"""
    const int threads = 128;
    const int64_t points = int64_t(out0_shape2) * out0_shape3;
    const int64_t rows = int64_t(out0_shape0) * out0_shape1;
    const dim3 blocks(std::min<int64_t>((points + threads - 1) / threads, 65535),
                      std::min<int64_t>(rows, 65535));
    bilinear_grid_sample<ALIGN_CORNERS><<<blocks, threads>>>(
        in0_p, in1_p, out0_p, in0_shape1, in0_shape2, in0_shape3, points, rows);
    CHECK(0 == cudaGetLastError());
"""

_SOURCES = tuple(_LAUNCH.replace("ALIGN_CORNERS", align) for align in ("false", "true"))


def _grid_sample_cuda(input, grid, mode, padding_mode, align_corners):
    output_shape = (input.shape[0], input.shape[1], grid.shape[1], grid.shape[2])
    return jt.code(output_shape, input.dtype, [input, grid],
                   cuda_header=_HEADER, cuda_src=_SOURCES[align_corners], backend="cuda")


register_kernel("nn.grid_sample", "cuda", _grid_sample_cuda, dtypes={"float32"},
                supports=_supports)

def _dense_input(value, op):
    """Skip dense, size-preserving copies and reshapes while the graph is pending."""
    for _ in range(8):
        if op(value):
            return value
        if value._producer_name() not in ("reshape", "clone") or not value._storage_is_contiguous():
            return None
        inputs = value._inputs()
        if (len(inputs) != 1 or not inputs[0]._storage_is_contiguous()
                or inputs[0].numel() != value.numel()):
            return None
        value = inputs[0]
    return None


def _stacked_samples(value, shape):
    from jittor.backends.cuda.kernels.misc.tensor_ops import _stack_sources

    batches, channels, queries, samples = shape
    # Shape metadata alone cannot identify CodeOp. Compare its launch
    # program before reading the inputs as a stack or as a bilinear sampler.
    stacked = _dense_input(value, lambda t: (
        t.ndim == 5 and tuple(t.shape[:3]) == (batches, channels, queries)
        and t.shape[-2] * t.shape[-1] == samples))
    if stacked is None:
        return None
    levels, points = stacked.shape[-2:]
    if not 2 <= levels <= 4 or not 1 <= points <= 8:
        return None
    stack_src, _ = _stack_sources(levels, points, None)
    same_code = jt.core._pending_cuda_code_is
    stack = _dense_input(stacked, lambda t: same_code(t, stack_src, ""))
    if stack is None:
        return None
    inputs = []
    alignments = []
    for packed in stack._inputs():
        sampled = _dense_input(packed, lambda t: tuple(t.shape) == (batches, channels, queries, points))
        if sampled is None:
            return None
        sample = _dense_input(sampled, lambda t: any(same_code(t, src, _HEADER) for src in _SOURCES))
        if sample is None:
            return None
        operands = sample._inputs()
        if len(operands) != 2:
            return None
        image, grid = operands
        inputs.extend((image, grid))
        alignments.append(same_code(sample, _SOURCES[1], _HEADER))
    if len(alignments) != levels:
        return None
    return inputs, (points, tuple(alignments))


@lru_cache(maxsize=32)
def _reduction_source(points, alignments):
    arguments = []
    parameters = []
    terms = []
    for level, align in enumerate(alignments):
        src = 1 + 2 * level
        arguments.extend((f"in{src}_p", f"in{src + 1}_p",
                          f"in{src}_shape2", f"in{src}_shape3"))
        parameters.append(f"const float* image{level}, const float* grid{level}, "
                          f"int64_t height{level}, int64_t width{level}")
        terms.append(f"""
        const float* plane{level} = image{level} + row * height{level} * width{level};
        const float* sample{level} = grid{level} + (batch * queries + query) * {points * 2};
        #pragma unroll
        for (int p = 0; p < {points}; ++p) {{
            const float sampled = sample_bilinear<{str(align).lower()}>(
                plane{level}, sample{level} + p * 2, height{level}, width{level});
            sum += __fmul_rn(sampled, query_weights[{level * points} + p]);
        }}
        """)
    header = _HEADER + f"""
    __global__ static void reduce_bilinear_grids(
        const float* weights, float* out, int channels, int queries,
        int64_t rows, {', '.join(parameters)}) {{
        const int64_t first = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        const int64_t stride = int64_t(gridDim.x) * blockDim.x;
        for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {{
            const int64_t batch = row / channels;
            for (int64_t query = first; query < queries; query += stride) {{
                const float* query_weights = weights + (batch * queries + query) * {points * len(alignments)};
                float sum = 0.0f;
                {''.join(terms)}
                out[row * queries + query] = sum;
            }}
        }}
    }}
    """
    source = f"""
    const int threads = 128;
    const int64_t rows = int64_t(out0_shape0) * out0_shape1;
    const dim3 blocks(std::min<int64_t>((int64_t(out0_shape2) + threads - 1) / threads, 65535),
                      std::min<int64_t>(rows, 65535));
    reduce_bilinear_grids<<<blocks, threads>>>(in0_p, out0_p, out0_shape1, out0_shape2,
                                              rows, {', '.join(arguments)});
    CHECK(0 == cudaGetLastError());
    """
    return header, source


@lru_cache(maxsize=32)
def _tiled_reduction_source(points, alignments):
    arguments, parameters, setup, terms = [], [], [], []
    for level, align in enumerate(alignments):
        src = 1 + 2 * level
        arguments.extend((f"in{src}_p", f"in{src + 1}_p",
                          f"in{src}_shape2", f"in{src}_shape3"))
        parameters.append(f"const float* image{level}, const float* grid{level}, "
                          f"int height{level}, int width{level}")
        setup.append(f"""
        for (int p = threadIdx.y; p < {points}; p += blockDim.y) {{
            const float* location = grid{level} + (item * {points} + p) * 2;
            taps[{level * points} + p][lane] = make_tap<{str(align).lower()}>(
                location, height{level}, width{level});
        }}
        """)
        terms.append(f"""
        const float* plane{level} = image{level} + row * height{level} * width{level};
        #pragma unroll
        for (int p = 0; p < {points}; ++p) {{
            float value = apply_tap(plane{level}, taps[{level * points} + p][lane]);
            sum += __fmul_rn(value, query_weights[{level * points} + p]);
        }}
        """)
    header = _HEADER + r"""
    struct BilinearTap {
        int nw, sw, ne, se;
        float dx, dy, dnx, dny;
    };

    __device__ static inline int pixel_offset(int y, int x, int height, int width) {
        return x >= 0 && x < width && y >= 0 && y < height ? y * width + x : -1;
    }

    template<bool align_corners>
    __device__ static inline BilinearTap make_tap(const float* grid, int height, int width) {
        float x = grid_coordinate<align_corners>(grid[0], width);
        float y = grid_coordinate<align_corners>(grid[1], height);
        if (!isfinite(x) || !isfinite(y)) return {-2, -1, -1, -1, 0, 0, 0, 0};
        if (x < -1.0f || x >= float(width) || y < -1.0f || y >= float(height))
            return {-1, -1, -1, -1, 0, 0, 0, 0};
        int left = int(floorf(x)), top = int(floorf(y));
        int right = left + 1, bottom = top + 1;
        return {pixel_offset(top, left, height, width), pixel_offset(bottom, left, height, width),
                pixel_offset(top, right, height, width), pixel_offset(bottom, right, height, width),
                x - float(left), y - float(top), float(right) - x, float(bottom) - y};
    }

    __device__ static inline float apply_tap(const float* plane, const BilinearTap& tap) {
        if (tap.nw == -2) return NAN;
        float nw = tap.nw < 0 ? 0.0f : plane[tap.nw];
        float sw = tap.sw < 0 ? 0.0f : plane[tap.sw];
        float ne = tap.ne < 0 ? 0.0f : plane[tap.ne];
        float se = tap.se < 0 ? 0.0f : plane[tap.se];
        return nw * tap.dnx * tap.dny + sw * tap.dnx * tap.dy
             + ne * tap.dx * tap.dny + se * tap.dx * tap.dy;
    }
    """ + f"""
    __global__ static void reduce_bilinear_grids_tiled(
        const float* weights, float* out, int channels, int queries, {', '.join(parameters)}) {{
        // A warp writes one output channel over 32 queries. Four such warps
        // share their grid coordinates and neighbor indices.
        __shared__ BilinearTap taps[{points * len(alignments)}][32];
        int lane = threadIdx.x;
        int query = blockIdx.x * 32 + lane;
        int channel = blockIdx.y * blockDim.y + threadIdx.y;
        int64_t item = int64_t(blockIdx.z) * queries + query;
        if (query < queries) {{
            {''.join(setup)}
        }}
        __syncthreads();
        if (query >= queries || channel >= channels) return;
        int64_t row = int64_t(blockIdx.z) * channels + channel;
        const float* query_weights = weights + item * {points * len(alignments)};
        float sum = 0.0f;
        {''.join(terms)}
        out[row * queries + query] = sum;
    }}
    """
    source = f"""
    const dim3 threads(32, 4);
    const dim3 blocks((int64_t(out0_shape2) + threads.x - 1) / threads.x,
                      (out0_shape1 + threads.y - 1) / threads.y, out0_shape0);
    reduce_bilinear_grids_tiled<<<blocks, threads>>>(in0_p, out0_p, out0_shape1, out0_shape2,
                                                    {', '.join(arguments)});
    CHECK(0 == cudaGetLastError());
    """
    return header, source


def try_reduce_sampled_grids(product):
    """Weighted last-axis sum over stacked bilinear grids with shared channel weights."""
    if (product.ndim != 4 or not jt.flags.no_grad or jt.flags.amp_reg
            or not jt.flags.use_cuda or not 2 <= product.shape[-1] <= 32
            or product.numel() < 4096 or product._producer_op() != "binary.multiply"
            or dtype_name(product.dtype) != "float32"):
        return None
    if dispatch_context(product).backend != "cuda":
        return None
    pair = product._inputs()
    if len(pair) != 2:
        return None
    shape = tuple(product.shape)
    for sampled, weights in (pair, pair[::-1]):
        if (tuple(weights.shape) != shape or weights._storage_strides()[1] != 0
                or weights.dtype != product.dtype):
            continue
        result = _stacked_samples(sampled, shape)
        if result is not None:
            inputs, geometry = result
            batches, channels, queries, _samples = shape
            tiled = (batches <= 65535 and 16 <= channels <= 4 * 65535
                     and 128 <= queries < 1 << 31
                     and all(int(t.shape[-2]) * int(t.shape[-1]) < 1 << 31 for t in inputs[::2]))
            generate = _tiled_reduction_source if tiled else _reduction_source
            header, source = generate(*geometry)
            return jt.code(shape[:-1], product.dtype, [weights[:, 0], *inputs],
                           cuda_header=header, cuda_src=source, backend="cuda")
    return None
