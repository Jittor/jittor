"""CUDA bilinear sampling for float32 inference with zero padding."""

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.dispatch import register_kernel


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
            const float x = grid_coordinate<align_corners>(locations[point * 2], width);
            const float y = grid_coordinate<align_corners>(locations[point * 2 + 1], height);
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
