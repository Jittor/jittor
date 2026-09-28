"""CUDA fast path for 4-D affine batch normalization, training and eval.

Every kernel here is one of two shapes. A *reduction* over one channel's
``N * H * W`` elements runs on a grid of (channel, segment) blocks, each
segment folding its part into a partial result that a one-thread-per-channel
*finish* kernel combines. The *elementwise* part -- applying the per-channel
affine map -- runs on a grid over the whole tensor.

It used to be one block per channel doing all three: a block's threads walked
every element of the channel, three times forward and twice backward. Early
ResNet-50 layers have 64 channels, which is 64 blocks on a 128-SM card, each
streaming 800 K elements through 1024 threads; the two kernels were 27 ms of a
96 ms training step on a 4090, against PyTorch's 74 ms for the whole step.
"""

from functools import lru_cache
import math

import jittor as jt
from jittor._core.dtypes import dtype_name as _dtype_name
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.backend_libraries import library_resource
from jittor._runtime.dispatch import optional_kernel

#: Blocks a reduction aims for: a few per SM on current parts. More segments
#: add partial results to combine; fewer leave SMs idle on narrow layers.
_TARGET_BLOCKS = 1024
_THREADS = 256
#: A segment is never shorter than this many elements per thread, so a wide
#: layer is not cut into segments that do nothing but write a partial.
_MIN_PER_THREAD = 8

_PAIR = """
struct JtBnPair { float a, b; };
struct JtBnPairSum {
    __device__ JtBnPair operator()(const JtBnPair& x, const JtBnPair& y) const {
        return {x.a + y.a, x.b + y.b};
    }
};
"""

_WELFORD = """
struct JtBnWelford { float n, mean, m2; };
struct JtBnWelfordSum {
    __device__ JtBnWelford operator()(const JtBnWelford& a,
                                      const JtBnWelford& b) const {
        float n = a.n + b.n;
        if (n == 0.0f) return a;
        float delta = b.mean - a.mean;
        float wb = b.n / n;
        return {n, a.mean + delta * wb, a.m2 + b.m2 + delta * delta * a.n * wb};
    }
};
"""


def _segments(channels, count):
    wanted = -(-_TARGET_BLOCKS // channels)
    most = max(1, count // (_THREADS * _MIN_PER_THREAD))
    segments = max(1, min(wanted, most))
    return segments, -(-count // segments)


def _header(*structs):
    return f"#include <{library_resource('cub', 'home')}cub/cub.cuh>\n" + "".join(structs)


def _channel_loop(channels, spatial, count, per_segment, body):
    """A block's walk over its segment of one channel; `body` sees `index`."""
    return f"""
    int channel = blockIdx.x;
    long long begin = (long long)blockIdx.y * {per_segment};
    long long end = begin + {per_segment};
    if (end > {count}) end = {count};
    for (long long item = begin + threadIdx.x; item < end; item += {_THREADS}) {{
        long long sample = item / {spatial};
        long long offset = item - sample * {spatial};
        long long index = (sample * {channels} + channel) * {spatial} + offset;
        {body}
    }}
    """


def _elementwise(name, args, scalar_body, vector_body, total, spatial, channels,
                 vector):
    """Elementwise kernels over the tensor; a body sees the item `i` and its
    channel `c`.

    `name` runs one element per item. With `vector` = 4 there is also
    `name`_v4, whose item is four consecutive elements of one channel's plane
    loaded and stored as one float4 -- usable when the plane is a multiple of
    four and every pointer is 16-byte aligned, which `_launch` checks.
    """
    def kernel(kernel_name, body, width):
        return f"""
    __global__ static void {kernel_name}({args}) {{
        long long stride = (long long)gridDim.x * blockDim.x;
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
                i < {total // width}LL; i += stride) {{
            int c = (int)((i * {width} / {spatial}) % {channels});
            {body}
        }}
    }}
    """
    source = kernel(name, scalar_body, 1)
    if vector == 4:
        source += kernel(name + "_v4", vector_body, 4)
    return source


def _launch(name, args, pointers, total, vector):
    def blocks(items):
        return max(1, min(-(-items // _THREADS), 65535 * 8))
    scalar = f"{name}<<<{blocks(total)}, {_THREADS}>>>({args});"
    if vector != 4:
        return scalar + "\n"
    aligned = " | ".join(f"(size_t){p}" for p in pointers)
    return (f"if ((({aligned}) & 15) == 0) "
            f"{name}_v4<<<{blocks(total // 4)}, {_THREADS}>>>({args});\n"
            f"else {scalar}\n")


def _vector(x, spatial):
    # By the runtime's own dtype name: the torch frontend spells it
    # "torch.float32", which silently kept every call on the scalar kernels.
    return 4 if spatial % 4 == 0 and _dtype_name(x.dtype) == "float32" else 1


def _per_channel(channels):
    """Launch bounds of a one-thread-per-channel kernel."""
    return f"{-(-channels // _THREADS)}, {_THREADS}"


@lru_cache(maxsize=128)
def _batch_norm_cuda_cls(batch, channels, spatial, eps, vector):
    count = batch * spatial
    total = count * channels
    segments, per_segment = _segments(channels, count)
    header = _header(_WELFORD, _PAIR)
    parts = segments * channels
    apply_v4 = """
        float4 v = reinterpret_cast<const float4*>(x)[i];
        float k = coef[c], b = coef[%d + c];
        reinterpret_cast<float4*>(y)[i] =
            make_float4(v.x * k + b, v.y * k + b, v.z * k + b, v.w * k + b);
    """ % channels
    apply_body = """
        y[i] = out0_type(static_cast<float>(x[i]) * coef[c] + coef[%d + c]);
    """ % channels
    grad_v4 = """
        float4 g = reinterpret_cast<const float4*>(grad_y)[i];
        float4 v = reinterpret_cast<const float4*>(x)[i];
        float k1 = coef[c], k2 = coef[%d + c], k3 = coef[%d + c];
        reinterpret_cast<float4*>(grad_x)[i] = make_float4(
            k1 * g.x + k2 * v.x + k3, k1 * g.y + k2 * v.y + k3,
            k1 * g.z + k2 * v.z + k3, k1 * g.w + k2 * v.w + k3);
    """ % (channels, 2 * channels)
    grad_body = """
        grad_x[i] = out0_type(coef[c] * static_cast<float>(grad_y[i])
            + coef[%d + c] * static_cast<float>(x[i]) + coef[%d + c]);
    """ % (channels, 2 * channels)

    class BatchNormCUDA(jt.Function):
        def execute(self, x, weight, bias):
            y, mean, var, rstd, partial, coef = jt.code(
                [x.shape, (channels,), (channels,), (channels,),
                 (3 * parts,), (2 * channels,)],
                [x.dtype, "float32", "float32", "float32", "float32", "float32"],
                [x, weight, bias],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void batch_norm_statistics(
                        const in0_type* x, float* partial) {{
                    typedef cub::BlockReduce<JtBnWelford, {_THREADS}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    // Welford per element. Shifted sums and squares lost a
                    // factor of twenty against the two-pass variance over the
                    // 800 K elements of an early ResNet-50 channel.
                    JtBnWelford local{{0.0f, 0.0f, 0.0f}};
                    {_channel_loop(channels, spatial, count, per_segment, '''
                        float value = static_cast<float>(x[index]);
                        local.n += 1.0f;
                        float delta = value - local.mean;
                        local.mean += delta * __frcp_rn(local.n);
                        local.m2 += delta * (value - local.mean);
                    ''')}
                    JtBnWelford total = BlockReduce(storage).Reduce(local, JtBnWelfordSum());
                    if (threadIdx.x == 0) {{
                        int slot = blockIdx.y * {channels} + channel;
                        partial[slot] = total.n;
                        partial[{parts} + slot] = total.mean;
                        partial[{2 * parts} + slot] = total.m2;
                    }}
                }}
                __global__ static void batch_norm_finish(
                        const float* partial, const in1_type* weight,
                        const in2_type* bias, float* mean, float* var,
                        float* rstd, float* coef) {{
                    int c = blockIdx.x * blockDim.x + threadIdx.x;
                    if (c >= {channels}) return;
                    JtBnWelford total{{0.0f, 0.0f, 0.0f}};
                    for (int s = 0; s < {segments}; s++) {{
                        int slot = s * {channels} + c;
                        JtBnWelford part{{partial[slot], partial[{parts} + slot],
                                          partial[{2 * parts} + slot]}};
                        total = JtBnWelfordSum()(total, part);
                    }}
                    float variance = total.m2 / total.n;
                    float r = rsqrtf(variance + {eps:.9g}f);
                    float k = r * static_cast<float>(weight[c]);
                    mean[c] = total.mean;
                    var[c] = variance;
                    rstd[c] = r;
                    coef[c] = k;
                    coef[{channels} + c] = static_cast<float>(bias[c]) - total.mean * k;
                }}
                {_elementwise("batch_norm_apply",
                              "const in0_type* x, const float* coef, out0_type* y",
                              apply_body, apply_v4, total, spatial, channels, vector)}
                batch_norm_statistics<<<dim3({channels}, {segments}), {_THREADS}>>>(
                    in0_p, out4_p);
                batch_norm_finish<<<{_per_channel(channels)}>>>(
                    out4_p, in1_p, in2_p, out1_p, out2_p, out3_p, out5_p);
                {_launch("batch_norm_apply", "in0_p, out5_p, out0_p",
                         ("in0_p", "out0_p"), total, vector)}
                CHECK(0 == cudaGetLastError());
                """,
            )
            self.saved = x, mean, rstd, weight
            # The statistics the step computed anyway, for the running
            # buffers; outside the tape, like the buffers themselves.
            self.statistics = mean.stop_grad(), var.stop_grad()
            return y

        def grad(self, grad_y):
            x, mean, rstd, weight = self.saved
            grad_x, grad_weight, grad_bias, partial, coef = jt.code(
                [grad_y.shape, weight.shape, weight.shape,
                 (2 * parts,), (3 * channels,)],
                [grad_y.dtype, weight.dtype, weight.dtype, "float32", "float32"],
                [grad_y, x, mean, rstd, weight],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void batch_norm_backward_sums(
                        const in0_type* grad_y, const in1_type* x,
                        const in2_type* mean, float* partial) {{
                    typedef cub::BlockReduce<JtBnPair, {_THREADS}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    float center = static_cast<float>(mean[blockIdx.x]);
                    JtBnPair local{{0.0f, 0.0f}};
                    {_channel_loop(channels, spatial, count, per_segment, '''
                        float dy = static_cast<float>(grad_y[index]);
                        local.a += dy;
                        local.b += dy * (static_cast<float>(x[index]) - center);
                    ''')}
                    JtBnPair total = BlockReduce(storage).Reduce(local, JtBnPairSum());
                    if (threadIdx.x == 0) {{
                        int slot = blockIdx.y * {channels} + channel;
                        partial[slot] = total.a;
                        partial[{parts} + slot] = total.b;
                    }}
                }}
                __global__ static void batch_norm_backward_finish(
                        const float* partial, const in2_type* mean,
                        const in3_type* rstd, const in4_type* weight,
                        out1_type* grad_weight, out2_type* grad_bias,
                        float* coef) {{
                    int c = blockIdx.x * blockDim.x + threadIdx.x;
                    if (c >= {channels}) return;
                    float sum_dy = 0.0f, sum_dy_centered = 0.0f;
                    for (int s = 0; s < {segments}; s++) {{
                        sum_dy += partial[s * {channels} + c];
                        sum_dy_centered += partial[{parts} + s * {channels} + c];
                    }}
                    float r = static_cast<float>(rstd[c]);
                    float sum_dy_xhat = sum_dy_centered * r;
                    grad_bias[c] = out2_type(sum_dy);
                    grad_weight[c] = out1_type(sum_dy_xhat);
                    // grad_x = r * w * (dy - mean(dy) - xhat * mean(dy * xhat)),
                    // written as k1 * dy + k2 * x + k3.
                    float k1 = r * static_cast<float>(weight[c]);
                    float k2 = -k1 * r * sum_dy_xhat / {count}.0f;
                    float k3 = -k1 * sum_dy / {count}.0f
                        - k2 * static_cast<float>(mean[c]);
                    coef[c] = k1;
                    coef[{channels} + c] = k2;
                    coef[{2 * channels} + c] = k3;
                }}
                {_elementwise("batch_norm_backward_apply",
                              "const in0_type* grad_y, const in1_type* x, "
                              "const float* coef, out0_type* grad_x",
                              grad_body, grad_v4, total, spatial, channels, vector)}
                batch_norm_backward_sums<<<dim3({channels}, {segments}), {_THREADS}>>>(
                    in0_p, in1_p, in2_p, out3_p);
                batch_norm_backward_finish<<<{_per_channel(channels)}>>>(
                    out3_p, in2_p, in3_p, in4_p, out1_p, out2_p, out4_p);
                {_launch("batch_norm_backward_apply", "in0_p, in1_p, out4_p, out0_p",
                         ("in0_p", "in1_p", "out0_p"), total, vector)}
                CHECK(0 == cudaGetLastError());
                """,
            )
            return grad_x, grad_weight, grad_bias

    return BatchNormCUDA


def _supports_batch_norm_training(x, weight, bias, eps):
    if not (
        _output_requires_grad(x, weight, bias)
        and isinstance(weight, jt.Var)
        and isinstance(bias, jt.Var)
    ):
        return False
    shape = tuple(int(size) for size in x.shape)
    if (
        len(shape) != 4
        or any(size <= 0 for size in shape)
        or int(weight.numel()) != shape[1]
        or int(bias.numel()) != shape[1]
        or not math.isfinite(float(eps))
        or float(eps) <= 0.0
    ):
        return False
    return True


@optional_kernel("nn.batch_norm.training_statistics", ("cuda", "rocm_legacy", "corex_legacy"),
                 dtypes=("float32",),
                 supports=_supports_batch_norm_training)
def _batch_norm_cuda_statistics(x, weight, bias, eps):
    """``(y, mean, var)``: the output and the biased batch statistics it used."""
    shape = tuple(int(size) for size in x.shape)
    spatial = shape[2] * shape[3]
    cls = _batch_norm_cuda_cls(shape[0], shape[1], spatial, float(eps),
                               _vector(x, spatial))
    # The call's own context, which `execute` writes the statistics onto.
    call = cls()._new_call_context()
    y = call._run_call(x, weight, bias)
    mean, var = call.statistics
    return y, mean, var


@optional_kernel("nn.batch_norm.training", ("cuda", "rocm_legacy", "corex_legacy"),
                 dtypes=("float32",),
                 supports=_supports_batch_norm_training)
def _batch_norm_cuda(x, weight, bias, eps):
    return _batch_norm_cuda_statistics(x, weight, bias, eps)[0]


@lru_cache(maxsize=128)
def _batch_norm_eval_cuda_cls(batch, channels, spatial, eps, vector):
    count = batch * spatial
    total = count * channels
    segments, per_segment = _segments(channels, count)
    header = _header(_PAIR)
    parts = segments * channels
    apply_v4 = """
        float4 v = reinterpret_cast<const float4*>(x)[i];
        float k = coef[c], b = coef[%d + c];
        reinterpret_cast<float4*>(y)[i] =
            make_float4(v.x * k + b, v.y * k + b, v.z * k + b, v.w * k + b);
    """ % channels
    apply_body = """
        y[i] = out0_type(static_cast<float>(x[i]) * coef[c] + coef[%d + c]);
    """ % channels
    grad_v4 = """
        float4 g = reinterpret_cast<const float4*>(grad_y)[i];
        float k = coef[c];
        reinterpret_cast<float4*>(grad_x)[i] =
            make_float4(g.x * k, g.y * k, g.z * k, g.w * k);
    """
    grad_body = """
        grad_x[i] = out0_type(static_cast<float>(grad_y[i]) * coef[c]);
    """
    # coef = [weight * rstd, bias - mean * weight * rstd, rstd]
    coefficients = f"""
    __global__ static void batch_norm_eval_coefficients(
            const in1_type* weight, const in2_type* bias, const in3_type* mean,
            const in4_type* variance, float* coef) {{
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= {channels}) return;
        float r = rsqrtf(static_cast<float>(variance[c]) + {eps:.9g}f);
        float k = r * static_cast<float>(weight[c]);
        coef[c] = k;
        coef[{channels} + c] = static_cast<float>(bias[c])
            - static_cast<float>(mean[c]) * k;
        coef[{2 * channels} + c] = r;
    }}
    """

    class BatchNormEvalCUDA(jt.Function):
        def execute(self, x, weight, bias, running_mean, running_var):
            y, coef = jt.code(
                [x.shape, (3 * channels,)],
                [x.dtype, "float32"],
                [x, weight, bias, running_mean, running_var],
                cuda_src=f"""
                {coefficients}
                {_elementwise("batch_norm_eval_apply",
                              "const in0_type* x, const float* coef, out0_type* y",
                              apply_body, apply_v4, total, spatial, channels, vector)}
                batch_norm_eval_coefficients<<<{_per_channel(channels)}>>>(
                    in1_p, in2_p, in3_p, in4_p, out1_p);
                {_launch("batch_norm_eval_apply", "in0_p, out1_p, out0_p",
                         ("in0_p", "out0_p"), total, vector)}
                CHECK(0 == cudaGetLastError());
                """,
            )
            self.saved = x, weight, bias, running_mean, running_var
            return y

        def grad(self, grad_y):
            x, weight, bias, running_mean, running_var = self.saved
            grad_x, grad_weight, grad_bias, partial, coef = jt.code(
                [grad_y.shape, weight.shape, weight.shape,
                 (2 * parts,), (3 * channels,)],
                [grad_y.dtype, weight.dtype, weight.dtype, "float32", "float32"],
                [grad_y, weight, bias, running_mean, running_var, x],
                cuda_header=header,
                cuda_src=f"""
                {coefficients}
                __global__ static void batch_norm_eval_backward_sums(
                        const in0_type* grad_y, const in5_type* x,
                        const float* coef, const in3_type* mean, float* partial) {{
                    typedef cub::BlockReduce<JtBnPair, {_THREADS}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    float center = static_cast<float>(mean[blockIdx.x]);
                    float r = coef[{2 * channels} + blockIdx.x];
                    JtBnPair local{{0.0f, 0.0f}};
                    {_channel_loop(channels, spatial, count, per_segment, '''
                        float dy = static_cast<float>(grad_y[index]);
                        local.a += dy * (static_cast<float>(x[index]) - center) * r;
                        local.b += dy;
                    ''')}
                    JtBnPair total = BlockReduce(storage).Reduce(local, JtBnPairSum());
                    if (threadIdx.x == 0) {{
                        int slot = blockIdx.y * {channels} + channel;
                        partial[slot] = total.a;
                        partial[{parts} + slot] = total.b;
                    }}
                }}
                __global__ static void batch_norm_eval_backward_finish(
                        const float* partial, out1_type* grad_weight,
                        out2_type* grad_bias) {{
                    int c = blockIdx.x * blockDim.x + threadIdx.x;
                    if (c >= {channels}) return;
                    float gw = 0.0f, gb = 0.0f;
                    for (int s = 0; s < {segments}; s++) {{
                        gw += partial[s * {channels} + c];
                        gb += partial[{parts} + s * {channels} + c];
                    }}
                    grad_weight[c] = out1_type(gw);
                    grad_bias[c] = out2_type(gb);
                }}
                {_elementwise("batch_norm_eval_backward_apply",
                              "const in0_type* grad_y, const float* coef, out0_type* grad_x",
                              grad_body, grad_v4, total, spatial, channels, vector)}
                batch_norm_eval_coefficients<<<{_per_channel(channels)}>>>(
                    in1_p, in2_p, in3_p, in4_p, out4_p);
                batch_norm_eval_backward_sums<<<dim3({channels}, {segments}), {_THREADS}>>>(
                    in0_p, in5_p, out4_p, in3_p, out3_p);
                batch_norm_eval_backward_finish<<<{_per_channel(channels)}>>>(
                    out3_p, out1_p, out2_p);
                {_launch("batch_norm_eval_backward_apply", "in0_p, out4_p, out0_p",
                         ("in0_p", "out0_p"), total, vector)}
                CHECK(0 == cudaGetLastError());
                """,
            )
            return grad_x, grad_weight, grad_bias, None, None

    return BatchNormEvalCUDA


def _supports_batch_norm_eval(x, weight, bias, running_mean, running_var, eps):
    values = (x, weight, bias, running_mean, running_var)
    if not (
        _output_requires_grad(values)
        and all(isinstance(value, jt.Var) for value in values)
    ):
        return False
    shape = tuple(int(size) for size in x.shape)
    if (
        len(shape) != 4
        or any(size <= 0 for size in shape)
        or any(int(value.numel()) != shape[1] for value in values[1:])
        or not math.isfinite(float(eps))
        or float(eps) <= 0.0
    ):
        return False
    return True


@optional_kernel("nn.batch_norm.eval", ("cuda", "rocm_legacy", "corex_legacy"),
                 dtypes=("float32",),
                 supports=_supports_batch_norm_eval)
def _batch_norm_eval_cuda(x, weight, bias, running_mean, running_var, eps):
    # The kernel indexes channels as NCHW. A channels-last view is left to the
    # elementwise form, which keeps its layout; copying it dense here would
    # undo it for the next convolution too.
    if not x._storage_is_contiguous():
        return None
    shape = tuple(int(size) for size in x.shape)
    spatial = shape[2] * shape[3]
    cls = _batch_norm_eval_cuda_cls(shape[0], shape[1], spatial, float(eps),
                                    _vector(x, spatial))
    return cls.apply(x, weight, bias, running_mean, running_var)
