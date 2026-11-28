"""Bilinear half-pixel resize and placement into fixed-size canvases."""

import jittor as jt
from jittor._core.flags import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.dispatch import register_kernel


_SOURCE = r"""
__global__ static void resize_regions(
        const in0_type* input, const in1_type* regions, out0_type* output,
        int channels, int src_h, int src_w, int height, int width,
        int64_t size) {
    using Math = typename std::conditional<std::is_same<in0_type, double>::value, double, float>::type;
    const int64_t stride = int64_t(gridDim.x) * blockDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += stride) {
        const int x = i % width, y = (i / width) % height;
        const int64_t plane = i / width / height;
        const auto* box = regions + (plane / channels) * 4;
        if (x < box[0] || y < box[1] || x >= box[2] || y >= box[3]) {
            output[i] = out0_type(0);
            continue;
        }
        const Math scale_x = Math(src_w) / (Math(box[2]) - Math(box[0]));
        const Math scale_y = Math(src_h) / (Math(box[3]) - Math(box[1]));
        const Math xf = max(Math(0), scale_x * (Math(x) - Math(box[0]) + Math(0.5)) - Math(0.5));
        const Math yf = max(Math(0), scale_y * (Math(y) - Math(box[1]) + Math(0.5)) - Math(0.5));
        const int x0 = min(int(xf), src_w - 1), x1 = min(x0 + 1, src_w - 1);
        const int y0 = min(int(yf), src_h - 1), y1 = min(y0 + 1, src_h - 1);
        const Math dx = min(xf - Math(x0), Math(1)), dy = min(yf - Math(y0), Math(1));
        const auto* image = input + plane * src_h * src_w;
        const auto* top = image + int64_t(y0) * src_w;
        const auto* bottom = image + int64_t(y1) * src_w;
        const Math a = (Math(1) - dx) * Math(top[x0]) + dx * Math(top[x1]);
        const Math b = (Math(1) - dx) * Math(bottom[x0]) + dx * Math(bottom[x1]);
        output[i] = out0_type((Math(1) - dy) * a + dy * b);
    }
}

    constexpr int threads = 256;
    const int64_t size = int64_t(out0_shape0) * out0_shape1 * out0_shape2 * out0_shape3;
    const int blocks = std::min<int64_t>((size + threads - 1) / threads, 65535);
    resize_regions<<<blocks, threads>>>(in0_p, in1_p, out0_p, in0_shape1,
        in0_shape2, in0_shape3, out0_shape2, out0_shape3, size);
    CHECK(0 == cudaGetLastError());
"""


def _resize_regions_cuda(input, regions, output_size):
    shape = (*input.shape[:2], *output_size)
    return _stop_grad_outputs(jt.code(
        shape, input.dtype, [input.contiguous(), regions.contiguous()],
        cuda_src=_SOURCE, backend="cuda"))


register_kernel("vision.resize_regions", "cuda", _resize_regions_cuda,
                dtypes={"float32", "float64", "float16", "bfloat16", "int32", "int64"},
                supports=lambda input, regions, size: not _output_requires_grad(input))
