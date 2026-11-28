"""CUDA inference fusion for pointwise convolution over a single pixel."""

from functools import lru_cache

from jittor._core.dtypes import dtype_name
from jittor._runtime.dispatch import optional_kernel


_HEADER = r"""
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
            return build(1 if kind == "silu" else 2)
        return None

    offer_activation(result, activate)
    return result
