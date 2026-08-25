"""CUDA inference fast path for modulated LayerNorm."""

import math

import jittor as jt

from .._cuda_inference import device_index


def _modulated_layer_norm_no_grad_cuda(x, scale, shift, eps):
    """Fuse inference BF16 LayerNorm with per-channel scale and shift."""
    if not all(isinstance(value, jt.Var) for value in (x, scale, shift)):
        return None
    if not (
        jt.flags.use_cuda
        and not getattr(jt.compiler, "has_acl", 0)
        and getattr(jt.flags, "no_grad", 0)
    ):
        return None
    autocast_probe = getattr(jt, "is_autocast_enabled", None)
    if callable(autocast_probe):
        try:
            if bool(autocast_probe()):
                return None
        except Exception:
            return None
    try:
        shape = tuple(int(size) for size in x.shape)
        scale_shape = tuple(int(size) for size in scale.shape)
        shift_shape = tuple(int(size) for size in shift.shape)
        devices = tuple(device_index(value) for value in (x, scale, shift))
        eps_value = float(eps)
    except Exception:
        return None
    if len(shape) not in (2, 3) or any(size <= 0 for size in shape):
        return None
    hidden = shape[-1]
    if (
        hidden > 4096
        or scale_shape not in ((hidden,), (1, hidden))
        or shift_shape != scale_shape
        or int(scale.numel()) != hidden
        or int(shift.numel()) != hidden
        or any(device < 0 for device in devices)
        or len(set(devices)) != 1
        or not math.isfinite(eps_value)
        or eps_value <= 0
    ):
        return None
    if any(str(value.dtype) != "bfloat16" for value in (x, scale, shift)):
        return None

    return jt.code(
        x.shape,
        x.dtype,
        [x, scale, shift],
        cuda_src=r"""
        __device__ __forceinline__ float warp_sum(float value) {
            for (int offset = 16; offset > 0; offset >>= 1)
                value += __shfl_down_sync(0xffffffff, value, offset);
            return value;
        }
        __device__ __forceinline__ double warp_sum_double(double value) {
            for (int offset = 16; offset > 0; offset >>= 1)
                value += __shfl_down_sync(0xffffffff, value, offset);
            return value;
        }
        __global__ static void modulated_layer_norm(
                const in0_type* x, const in1_type* scale,
                const in2_type* shift, out0_type* y) {
            int row = blockIdx.x;
            int tid = threadIdx.x;
            int lane = tid & 31;
            int warp = tid >> 5;
            __shared__ float warp_buf[4];
            __shared__ float mean_shared;
            __shared__ float inv_std_shared;
            __shared__ double warp_double_buf[4];
            __shared__ double mean_double_shared;
            __shared__ double inv_std_double_shared;
            __shared__ int use_double;

            float sum = 0.0f;
            for (int dim = tid; dim < %(hidden)d; dim += blockDim.x)
                sum += static_cast<float>(x[row * %(hidden)d + dim]);
            sum = warp_sum(sum);
            if (lane == 0) warp_buf[warp] = sum;
            __syncthreads();
            if (warp == 0) {
                float total = lane < 4 ? warp_buf[lane] : 0.0f;
                total = warp_sum(total);
                if (lane == 0) {
                    use_double = !isfinite(total);
                    if (!use_double) mean_shared = total / %(hidden)d.0f;
                }
            }
            __syncthreads();
            if (!use_double) {
                float mean = mean_shared;
                float variance = 0.0f;
                for (int dim = tid; dim < %(hidden)d; dim += blockDim.x) {
                    float delta = static_cast<float>(
                        x[row * %(hidden)d + dim]) - mean;
                    variance += delta * delta;
                }
                variance = warp_sum(variance);
                if (lane == 0) warp_buf[warp] = variance;
                __syncthreads();
                if (warp == 0) {
                    float total = lane < 4 ? warp_buf[lane] : 0.0f;
                    total = warp_sum(total);
                    if (lane == 0) {
                        use_double = !isfinite(total);
                        if (!use_double)
                            inv_std_shared = rsqrtf(
                                total / %(hidden)d.0f + %(eps).9gf);
                    }
                }
                __syncthreads();
            }
            if (use_double) {
                double double_sum = 0.0;
                for (int dim = tid; dim < %(hidden)d; dim += blockDim.x)
                    double_sum += static_cast<double>(
                        x[row * %(hidden)d + dim]);
                double_sum = warp_sum_double(double_sum);
                if (lane == 0) warp_double_buf[warp] = double_sum;
                __syncthreads();
                if (warp == 0) {
                    double total = lane < 4 ? warp_double_buf[lane] : 0.0;
                    total = warp_sum_double(total);
                    if (lane == 0)
                        mean_double_shared = total / %(hidden)d.0;
                }
                __syncthreads();
                double double_mean = mean_double_shared;
                double double_variance = 0.0;
                for (int dim = tid; dim < %(hidden)d; dim += blockDim.x) {
                    double delta = static_cast<double>(
                        x[row * %(hidden)d + dim]) - double_mean;
                    double_variance += delta * delta;
                }
                double_variance = warp_sum_double(double_variance);
                if (lane == 0) warp_double_buf[warp] = double_variance;
                __syncthreads();
                if (warp == 0) {
                    double total = lane < 4 ? warp_double_buf[lane] : 0.0;
                    total = warp_sum_double(total);
                    if (lane == 0)
                        inv_std_double_shared = 1.0 / sqrt(
                            total / %(hidden)d.0 + %(eps_double).17g);
                }
                __syncthreads();
            }

            for (int dim = tid; dim < %(hidden)d; dim += blockDim.x) {
                int index = row * %(hidden)d + dim;
                float normalized_value = use_double
                    ? static_cast<float>(
                        (static_cast<double>(x[index]) - mean_double_shared)
                        * inv_std_double_shared)
                    : (static_cast<float>(x[index]) - mean_shared)
                        * inv_std_shared;
                out0_type normalized = out0_type(normalized_value);
                out0_type mod_scale = out0_type(
                    1.0f + static_cast<float>(scale[dim]));
                out0_type product = out0_type(
                    static_cast<float>(normalized)
                    * static_cast<float>(mod_scale));
                y[index] = out0_type(
                    static_cast<float>(product)
                    + static_cast<float>(shift[dim]));
            }
        }
        int rows = in0->num / %(hidden)d;
        modulated_layer_norm<<<rows, 128>>>(
            in0_p, in1_p, in2_p, out0_p);
        CHECK(0 == cudaGetLastError());
        """ % {
            "eps": eps_value,
            "eps_double": eps_value,
            "hidden": hidden,
        },
    )


__all__ = ["_modulated_layer_norm_no_grad_cuda"]
