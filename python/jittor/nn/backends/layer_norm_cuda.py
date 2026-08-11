"""CUDA inference fast path for :func:`jittor.nn.layer_norm`."""

import os

import jittor as jt


def _layer_norm_no_grad_cuda(
        x, normalized_shape, weight, bias, eps, *, allow_bfloat16=False):
    if not (jt.flags.use_cuda and not getattr(jt.compiler, "has_acl", 0)
            and getattr(jt.flags, "no_grad", 0)):
        return None
    input_dtype = str(x.dtype)
    supported_dtypes = ("float16", "float32")
    if allow_bfloat16:
        supported_dtypes += ("bfloat16",)
    if len(normalized_shape) != 1 or input_dtype not in supported_dtypes:
        return None
    hidden = int(normalized_shape[0])
    var_affine = isinstance(weight, jt.Var) and isinstance(bias, jt.Var)
    scalar_affine = not isinstance(weight, jt.Var) and not isinstance(bias, jt.Var)
    if not var_affine:
        if not scalar_affine or os.environ.get("JITTOR_LAYERNORM_SCALAR_FAST", "1") == "0":
            return None
    if int(x.shape[-1]) != hidden:
        return None
    eps_value = float(eps)
    if scalar_affine:
        scale_value = float(weight)
        offset_value = float(bias)
        scale_literal = f"{scale_value:.9e}f"
        offset_literal = f"{offset_value:.9e}f"
        y = jt.code(
            x.shape,
            x.dtype,
            [x],
            cuda_src=f"""
            __device__ __forceinline__ float warp_sum(float value) {{
                for (int offset = 16; offset > 0; offset >>= 1)
                    value += __shfl_down_sync(0xffffffff, value, offset);
                return value;
            }}
            __device__ __forceinline__ double warp_sum_double(double value) {{
                for (int offset = 16; offset > 0; offset >>= 1)
                    value += __shfl_down_sync(0xffffffff, value, offset);
                return value;
            }}
            __global__ static void kernel(
                    in0_type* x, out0_type* y, int hidden) {{
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
                for (int j = tid; j < hidden; j += blockDim.x)
                    sum += static_cast<float>(x[row * hidden + j]);
                sum = warp_sum(sum);
                if (lane == 0) warp_buf[warp] = sum;
                __syncthreads();
                if (warp == 0) {{
                    float total = lane < 4 ? warp_buf[lane] : 0.0f;
                    total = warp_sum(total);
                    if (lane == 0) {{
                        use_double = !isfinite(total);
                        if (!use_double) mean_shared = total / hidden;
                    }}
                }}
                __syncthreads();
                if (!use_double) {{
                    float mean = mean_shared;
                    float var = 0.0f;
                    for (int j = tid; j < hidden; j += blockDim.x) {{
                        float d = static_cast<float>(
                            x[row * hidden + j]) - mean;
                        var += d * d;
                    }}
                    var = warp_sum(var);
                    if (lane == 0) warp_buf[warp] = var;
                    __syncthreads();
                    if (warp == 0) {{
                        float total = lane < 4 ? warp_buf[lane] : 0.0f;
                        total = warp_sum(total);
                        if (lane == 0) {{
                            use_double = !isfinite(total);
                            if (!use_double)
                                inv_std_shared = rsqrtf(
                                    total / hidden + {eps_value:.9g}f);
                        }}
                    }}
                    __syncthreads();
                }}
                if (use_double) {{
                    double double_sum = 0.0;
                    for (int j = tid; j < hidden; j += blockDim.x)
                        double_sum += static_cast<double>(
                            x[row * hidden + j]);
                    double_sum = warp_sum_double(double_sum);
                    if (lane == 0) warp_double_buf[warp] = double_sum;
                    __syncthreads();
                    if (warp == 0) {{
                        double total = lane < 4 ? warp_double_buf[lane] : 0.0;
                        total = warp_sum_double(total);
                        if (lane == 0)
                            mean_double_shared = total / hidden;
                    }}
                    __syncthreads();
                    double double_mean = mean_double_shared;
                    double double_var = 0.0;
                    for (int j = tid; j < hidden; j += blockDim.x) {{
                        double d = static_cast<double>(
                            x[row * hidden + j]) - double_mean;
                        double_var += d * d;
                    }}
                    double_var = warp_sum_double(double_var);
                    if (lane == 0) warp_double_buf[warp] = double_var;
                    __syncthreads();
                    if (warp == 0) {{
                        double total = lane < 4 ? warp_double_buf[lane] : 0.0;
                        total = warp_sum_double(total);
                        if (lane == 0)
                            inv_std_double_shared = 1.0 / sqrt(
                                total / hidden + {eps_value:.17g});
                    }}
                    __syncthreads();
                    double double_inv_std = inv_std_double_shared;
                    for (int j = tid; j < hidden; j += blockDim.x)
                        y[row * hidden + j] = out0_type(
                            (static_cast<double>(x[row * hidden + j])
                             - double_mean)
                            * double_inv_std * {scale_literal}
                            + {offset_literal});
                }} else {{
                    float mean = mean_shared;
                    float inv_std = inv_std_shared;
                    for (int j = tid; j < hidden; j += blockDim.x)
                        y[row * hidden + j] = out0_type(
                            (static_cast<float>(x[row * hidden + j]) - mean)
                            * inv_std * {scale_literal} + {offset_literal});
                }}
            }}
            int rows = in0->num / {hidden};
            kernel<<<rows, 128>>>(in0_p, out0_p, {hidden});
            """,
        )
        return y
    if int(weight.numel()) != hidden or int(bias.numel()) != hidden:
        return None
    y = jt.code(
        x.shape,
        x.dtype,
        [x, weight, bias],
        cuda_src=f"""
        __device__ __forceinline__ float warp_sum(float value) {{
            for (int offset = 16; offset > 0; offset >>= 1)
                value += __shfl_down_sync(0xffffffff, value, offset);
            return value;
        }}
        __device__ __forceinline__ double warp_sum_double(double value) {{
            for (int offset = 16; offset > 0; offset >>= 1)
                value += __shfl_down_sync(0xffffffff, value, offset);
            return value;
        }}
        __global__ static void kernel(
                in0_type* x, in1_type* weight, in2_type* bias,
                out0_type* y, int hidden) {{
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
            for (int j = tid; j < hidden; j += blockDim.x)
                sum += static_cast<float>(x[row * hidden + j]);
            sum = warp_sum(sum);
            if (lane == 0) warp_buf[warp] = sum;
            __syncthreads();
            if (warp == 0) {{
                float total = lane < 4 ? warp_buf[lane] : 0.0f;
                total = warp_sum(total);
                if (lane == 0) {{
                    use_double = !isfinite(total);
                    if (!use_double) mean_shared = total / hidden;
                }}
            }}
            __syncthreads();
            if (!use_double) {{
                float mean = mean_shared;
                float var = 0.0f;
                for (int j = tid; j < hidden; j += blockDim.x) {{
                    float d = static_cast<float>(
                        x[row * hidden + j]) - mean;
                    var += d * d;
                }}
                var = warp_sum(var);
                if (lane == 0) warp_buf[warp] = var;
                __syncthreads();
                if (warp == 0) {{
                    float total = lane < 4 ? warp_buf[lane] : 0.0f;
                    total = warp_sum(total);
                    if (lane == 0) {{
                        use_double = !isfinite(total);
                        if (!use_double)
                            inv_std_shared = rsqrtf(
                                total / hidden + {eps_value:.9g}f);
                    }}
                }}
                __syncthreads();
            }}
            if (use_double) {{
                double double_sum = 0.0;
                for (int j = tid; j < hidden; j += blockDim.x)
                    double_sum += static_cast<double>(
                        x[row * hidden + j]);
                double_sum = warp_sum_double(double_sum);
                if (lane == 0) warp_double_buf[warp] = double_sum;
                __syncthreads();
                if (warp == 0) {{
                    double total = lane < 4 ? warp_double_buf[lane] : 0.0;
                    total = warp_sum_double(total);
                    if (lane == 0)
                        mean_double_shared = total / hidden;
                }}
                __syncthreads();
                double double_mean = mean_double_shared;
                double double_var = 0.0;
                for (int j = tid; j < hidden; j += blockDim.x) {{
                    double d = static_cast<double>(
                        x[row * hidden + j]) - double_mean;
                    double_var += d * d;
                }}
                double_var = warp_sum_double(double_var);
                if (lane == 0) warp_double_buf[warp] = double_var;
                __syncthreads();
                if (warp == 0) {{
                    double total = lane < 4 ? warp_double_buf[lane] : 0.0;
                    total = warp_sum_double(total);
                    if (lane == 0)
                        inv_std_double_shared = 1.0 / sqrt(
                            total / hidden + {eps_value:.17g});
                }}
                __syncthreads();
                double double_inv_std = inv_std_double_shared;
                for (int j = tid; j < hidden; j += blockDim.x) {{
                    double scale = static_cast<double>(weight[j]);
                    double offset = static_cast<double>(bias[j]);
                    y[row * hidden + j] = out0_type(
                        (static_cast<double>(x[row * hidden + j])
                         - double_mean)
                        * double_inv_std * scale + offset);
                }}
            }} else {{
                float mean = mean_shared;
                float inv_std = inv_std_shared;
                for (int j = tid; j < hidden; j += blockDim.x) {{
                    float scale = static_cast<float>(weight[j]);
                    float offset = static_cast<float>(bias[j]);
                    y[row * hidden + j] = out0_type(
                        (static_cast<float>(x[row * hidden + j]) - mean)
                        * inv_std * scale + offset);
                }}
            }}
        }}
        int rows = in0->num / {hidden};
        kernel<<<rows, 128>>>(
            in0_p, in1_p, in2_p, out0_p, {hidden});
        """,
    )
    return y
