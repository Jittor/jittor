"""CUDA inference fusion for pointwise convolution over a single pixel."""

from functools import lru_cache

from jittor._core.dtypes import dtype_name
from jittor._runtime.dispatch import optional_kernel


_HEADER = r"""
template <typename T>
__device__ __forceinline__ float jt_pointwise_dot(
        const T* activation, const T* filter, int cin, int lane) {
    float a = 0.f, b = 0.f;
    int i = lane;
    for (; i + 32 < cin; i += 64) {
        a = fmaf(static_cast<float>(activation[i]), static_cast<float>(filter[i]), a);
        b = fmaf(static_cast<float>(activation[i + 32]), static_cast<float>(filter[i + 32]), b);
    }
    if (i < cin)
        a = fmaf(static_cast<float>(activation[i]), static_cast<float>(filter[i]), a);
    float total = a + b;
    for (int step = 16; step; step >>= 1)
        total += __shfl_down_sync(0xffffffffu, total, step);
    return total;
}

template <int WARPS, int ACT, typename T>
__global__ static void jt_pointwise_pixel(
        const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ bias,
        T* __restrict__ y, int batch, int cin, int cout) {
    int lane = threadIdx.x & 31;
    int item = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (item >= batch * cout) return;
    int channel = item % cout;
    const T* activation = x + static_cast<long long>(item / cout) * cin;
    const T* filter = w + static_cast<long long>(channel) * cin;
    float total = jt_pointwise_dot(activation, filter, cin, lane);
    if (lane == 0) {
        // A public convolution rounds before the bias add and the activation.
        T value = static_cast<T>(total);
        if (bias)
            value = static_cast<T>(static_cast<float>(value) + static_cast<float>(bias[channel]));
        if (ACT) {
            float v = static_cast<float>(value);
            float gate = 1.f / (1.f + expf(-v));
            value = static_cast<T>(ACT == 1 ? v * gate : gate);
        }
        y[item] = value;
    }
}

template <typename T>
__global__ static void jt_pointwise_sigmoid_gate(
        const T* __restrict__ feature, const T* __restrict__ x,
        const T* __restrict__ w, const T* __restrict__ bias, T* __restrict__ y,
        int cin, int cout, int spatial) {
    constexpr int channels = 16, pixels = 64;
    __shared__ T gates[channels];
    int channel0 = blockIdx.x * channels;
    int pixel0 = blockIdx.y * pixels;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    const T* activation = x + static_cast<long long>(blockIdx.z) * cin;
    for (int c = warp; c < channels; c += 4) {
        int channel = channel0 + c;
        if (channel < cout) {
            float total = jt_pointwise_dot(activation, w + static_cast<long long>(channel) * cin, cin, lane);
            if (lane == 0) {
                T value = static_cast<T>(total);
                if (bias)
                    value = static_cast<T>(static_cast<float>(value) + static_cast<float>(bias[channel]));
                gates[c] = static_cast<T>(1.f / (1.f + expf(-static_cast<float>(value))));
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < channels * pixels; i += blockDim.x) {
        int pixel = pixel0 + i / channels;
        int channel = channel0 + i % channels;
        if (pixel < spatial && channel < cout) {
            long long location = (static_cast<long long>(blockIdx.z) * spatial + pixel) * cout + channel;
            y[location] = static_cast<T>(static_cast<float>(feature[location]) *
                                         static_cast<float>(gates[i % channels]));
        }
    }
}
"""


@lru_cache(maxsize=16)
def _launch(warps, activation, has_bias):
    return f"""
    int batch = in0_shape0, cin = in0_shape1, cout = in1_shape0;
    jt_pointwise_pixel<{warps}, {activation}><<<(batch * cout + {warps} - 1) / {warps}, {warps * 32}>>>(
        in0_p, in1_p, (const in0_type*){"in2_p" if has_bias else "nullptr"},
        out0_p, batch, cin, cout);
    CHECK(cudaGetLastError() == cudaSuccess);
    """


@lru_cache(maxsize=2)
def _gate_launch(has_bias):
    return f"""
    int batch = in0_shape0, spatial = in0_shape1 * in0_shape2;
    int cin = in1_shape1, cout = in0_shape3;
    dim3 grid((cout + 15) / 16, (spatial + 63) / 64, batch);
    jt_pointwise_sigmoid_gate<<<grid, 128>>>(
        in0_p, in1_p, in2_p, (const in0_type*){"in3_p" if has_bias else "nullptr"},
        out0_p, cin, cout, spatial);
    CHECK(cudaGetLastError() == cudaSuccess);
    """


@optional_kernel("nn.conv2d.single_pixel.inference", "cuda")
def pointwise_pixel_cuda(x, weight, bias, padding, groups):
    import jittor as jt

    if (groups != 1 or tuple(padding) != (0, 0) or not jt.flags.no_grad
            or jt.flags.amp_reg or tuple(x.shape[2:]) != (1, 1)
            or tuple(weight.shape[2:]) != (1, 1)):
        return None
    n, cin = x.shape[:2]
    cout = weight.shape[0]
    if n <= 0 or cin <= 0 or cout <= 0 or not (256 <= n * cin * cout <= (1 << 20)):
        return None
    dtype = dtype_name(x.dtype)
    inputs = (x, weight) if bias is None else (x, weight, bias)
    if (dtype not in ("float16", "bfloat16") or
            any(not isinstance(value, jt.Var) or dtype_name(value.dtype) != dtype
                or not value._storage_is_contiguous() for value in inputs)):
        return None
    if bias is not None and tuple(bias.shape) != (cout,):
        return None

    warps = 1 if n * cout <= 128 else 4
    shape = n, cout, 1, 1

    def build(activation):
        return jt.code(shape, x.dtype, inputs, cuda_header=_HEADER,
                       cuda_src=_launch(warps, activation, bias is not None)).stop_grad()

    result = build(0)
    from jittor.nn.functional.activation import offer_activation

    stamps = tuple(value._saved_storage_version() for value in inputs)

    def activate(kind):
        if (kind in ("silu", "sigmoid") and jt.flags.no_grad and not jt.flags.amp_reg
                and tuple(value._saved_storage_version() for value in inputs) == stamps):
            result = build(1 if kind == "silu" else 2)
            if kind == "sigmoid":
                result._jt_pointwise_sigmoid_mul = (result._saved_storage_version(), multiply)
            return result
        return None

    def multiply(feature):
        from jittor._runtime.dispatch import dispatch_context
        from jittor.nn.functional._layout import channels_last_source, channels_last_view

        if (not jt.flags.no_grad or jt.flags.amp_reg or feature.dtype != x.dtype
                or feature.ndim != 4 or tuple(feature.shape[:2]) != (n, cout)
                or n > 65535 or tuple(value._saved_storage_version() for value in inputs) != stamps):
            return None
        height, width = feature.shape[2:]
        tiles = (height * width + 63) // 64
        if not tiles or n * cin * cout * tiles > (1 << 22):
            return None
        if dispatch_context(feature, *inputs).backend != "cuda":
            return None
        physical = channels_last_source(feature)
        if physical is None:
            return None
        out = jt.code(physical.shape, x.dtype, [physical, *inputs], cuda_header=_HEADER,
                      cuda_src=_gate_launch(bias is not None)).stop_grad()
        return channels_last_view(out)

    offer_activation(result, activate)
    return result
