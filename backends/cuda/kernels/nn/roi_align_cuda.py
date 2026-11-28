"""Fused CUDA forward for RoI Align."""

import jittor as jt
from jittor._core.flags import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.dispatch import register_kernel


_HEADER = r"""
namespace jittor {
template<class T>
__device__ static inline T roi_coordinate(T value) { return value; }

__device__ static inline float roi_coordinate(float16 value) { return float(value); }

// Match the CUDA reference's contraction points independently of build flags.
__device__ static inline float roi_multiply_add(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}
__device__ static inline double roi_multiply_add(double a, double b, double c) {
    return __fma_rn(a, b, c);
}
__device__ static inline float16 roi_multiply_add(float16 a, float16 b, float16 c) {
    return __hadd_rn(__hmul_rn(a, b), c);
}

template<class T, class Storage>
__device__ static inline T roi_bilinear(
        const Storage* image, int height, int width, T y, T x) {
    const auto fy = roi_coordinate(y), fx = roi_coordinate(x);
    if (fy < -1 || fy > height || fx < -1 || fx > width)
        return T(0);
    if (isnan(fy) || isnan(fx))
        return T(NAN);
    y = fy > 0 ? y : T(0);
    x = fx > 0 ? x : T(0);
    const int y0 = min(int(roi_coordinate(y)), height - 1);
    const int x0 = min(int(roi_coordinate(x)), width - 1);
    const int y1 = min(y0 + 1, height - 1), x1 = min(x0 + 1, width - 1);
    const T dy = y0 == y1 ? T(0) : y - T(y0);
    const T dx = x0 == x1 ? T(0) : x - T(x0);
    const T hy = T(1) - dy, hx = T(1) - dx;
    const T w00 = hy * hx, w01 = hy * dx, w10 = dy * hx, w11 = dy * dx;
    const auto row0 = image + int64_t(y0) * width;
    const auto row1 = image + int64_t(y1) * width;
    const T top = roi_multiply_add(w01, T(row0[x1]), w00 * T(row0[x0]));
    const T partial = roi_multiply_add(w10, T(row1[x0]), top);
    return roi_multiply_add(w11, T(row1[x1]), partial);
}

template<class T, class Storage>
__global__ static void roi_align_forward(
        const Storage* features, const Storage* boxes, Storage* output,
        int batch, int channels, int height, int width, int64_t rois,
        int pooled_h, int pooled_w, int64_t bins_per_roi, T scale,
        int sampling_ratio, bool aligned) {
    const int64_t start = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(gridDim.x) * blockDim.x;
    for (int64_t roi = blockIdx.y; roi < rois; roi += gridDim.y) {
        const auto* box = boxes + roi * 5;
        const int image_index = int(roi_coordinate(T(box[0])));
        const T shift = aligned ? T(0.5f) : T(0);
        const T x0 = roi_multiply_add(T(box[1]), scale, -shift);
        const T y0 = roi_multiply_add(T(box[2]), scale, -shift);
        const T x1 = roi_multiply_add(T(box[3]), scale, -shift);
        const T y1 = roi_multiply_add(T(box[4]), scale, -shift);
        T extent_x = x1 - x0, extent_y = y1 - y0;
        if (!aligned) {
            if (extent_x < T(1)) extent_x = T(1);
            if (extent_y < T(1)) extent_y = T(1);
        }
        const T bin_h = extent_y / T(pooled_h), bin_w = extent_x / T(pooled_w);
        const int ny = sampling_ratio > 0 ? sampling_ratio : int(ceil(roi_coordinate(bin_h)));
        const int nx = sampling_ratio > 0 ? sampling_ratio : int(ceil(roi_coordinate(bin_w)));
        for (int64_t bin = start; bin < bins_per_roi; bin += stride) {
            auto* value = output + roi * bins_per_roi + bin;
            if (image_index < 0 || image_index >= batch || ny <= 0 || nx <= 0) {
                *value = Storage(0);
                continue;
            }
            const int ph = (bin / pooled_w) % pooled_h, pw = bin % pooled_w;
            const int channel = bin / pooled_w / pooled_h;
            const auto* image = features + (int64_t(image_index) * channels + channel) * height * width;
            const T y_base = roi_multiply_add(T(ph), bin_h, y0);
            const T x_base = roi_multiply_add(T(pw), bin_w, x0);
            T sum = T(0);
            for (int iy = 0; iy < ny; ++iy) {
                const T y = y_base + T(iy + 0.5f) * bin_h / T(ny);
                for (int ix = 0; ix < nx; ++ix) {
                    const T x = x_base + T(ix + 0.5f) * bin_w / T(nx);
                    sum += roi_bilinear(image, height, width, y, x);
                }
            }
            *value = Storage(sum / T(int64_t(ny) * nx));
        }
    }
}
}  // namespace jittor
"""

_LAUNCH = r"""
    // TorchVision's half kernel rounds the intermediate arithmetic in half.
    // Bfloat16 inputs use float coordinates, weights and accumulation.
    using Math = typename std::conditional<
        std::is_same<in0_type, bfloat16>::value, float, in0_type>::type;
    constexpr int threads = 128;
    const int64_t bins_per_roi = int64_t(out0_shape1) * out0_shape2 * out0_shape3;
    const dim3 blocks(std::min<int64_t>((bins_per_roi + threads - 1) / threads, 65535),
                      std::min<int64_t>(out0_shape0, 65535));
    roi_align_forward<<<blocks, threads>>>(
        in0_p, in1_p, out0_p, in0_shape0, in0_shape1, in0_shape2, in0_shape3,
        out0_shape0, out0_shape2, out0_shape3, bins_per_roi, Math(data["spatial_scale"]),
        int(data["sampling_ratio"]), bool(data["aligned"]));
    CHECK(0 == cudaGetLastError());
"""


def _supports(features, boxes, output_size, spatial_scale, sampling_ratio, aligned):
    return not _output_requires_grad(features)


def _roi_align_cuda(features, boxes, output_size, spatial_scale, sampling_ratio, aligned):
    shape = (boxes.shape[0], features.shape[1], *output_size)
    out = jt.code(shape, features.dtype, [features.contiguous(), boxes.contiguous()],
                  cuda_header=_HEADER, cuda_src=_LAUNCH, backend="cuda",
                  data=dict(spatial_scale=spatial_scale, sampling_ratio=float(sampling_ratio),
                            aligned=float(aligned)))
    return _stop_grad_outputs(out)


register_kernel("vision.roi_align", "cuda", _roi_align_cuda,
                dtypes={"float16", "bfloat16", "float32", "float64"}, supports=_supports)
