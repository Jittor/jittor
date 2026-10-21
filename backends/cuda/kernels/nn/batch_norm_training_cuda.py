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
from jittor.nn.functional.activation import offer_activation
from jittor.nn.functional._layout import channels_last_source, channels_last_view, take_channels_last

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
    # A whole number of float4s, so that the vector reductions cut the same
    # segments; `count` is a multiple of four whenever they run.
    per_segment = -(-count // segments)
    per_segment += -per_segment % 4
    return segments, per_segment


def _header(*structs):
    return f"#include <{library_resource('cub', 'home')}cub/cub.cuh>\n" + "".join(structs)


def _channel_loop(channels, spatial, count, per_segment, body, width=1):
    """A block's walk over its segment of one channel; `body` sees `index`.

    With `width` 4 the walk is in float4s -- `index` counts them -- which needs
    a plane that is a multiple of four; `count` and `per_segment` stay in
    elements. The position within the plane is carried from one step to the
    next rather than divided out of the item number: a 64-bit division per
    element held the reductions to 700 GB/s where the elementwise passes over
    the same tensor run at 930.
    """
    plane = spatial // width
    step_samples, step_offset = divmod(_THREADS, plane)
    return f"""
    int channel = blockIdx.x;
    long long begin = (long long)blockIdx.y * {per_segment // width};
    long long end = begin + {per_segment // width};
    if (end > {count // width}) end = {count // width};
    long long item = begin + threadIdx.x;
    long long sample = item / {plane};
    long long offset = item - sample * {plane};
    #pragma unroll 4
    for (; item < end; item += {_THREADS}) {{
        long long index = (sample * {channels} + channel) * {plane} + offset;
        {body}
        sample += {step_samples};
        offset += {step_offset};
        if (offset >= {plane}) {{ offset -= {plane}; sample++; }}
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


#: What the training batch norm can apply to its output in the same pass, as
#: (forward of z, gradient given the output's gradient gs and z).
_ACTIVATIONS = {
    "": ("return z;", "return gs;"),
    "relu": ("return z > 0.0f ? z : 0.0f;", "return z > 0.0f ? gs : 0.0f;"),
}


@lru_cache(maxsize=128)
def _batch_norm_cuda_cls(batch, channels, spatial, eps, vector, act="", residual=False):
    """The training batch norm, applying ``act`` to its output -- or, with
    ``residual``, to its output plus a residual input of the same shape.

    With a residual the activation's gradient is taken from the output, which
    is kept (it is the block's output, which the next layer keeps anyway), and
    the backward kernels see that gradient already applied; relu is the one
    activation whose gradient the output determines.
    """
    count = batch * spatial
    total = count * channels
    segments, per_segment = _segments(channels, count)
    forward_act, grad_act = _ACTIVATIONS[act]
    if residual:
        assert act == "relu", act
        grad_act = _ACTIVATIONS[""][1]
    header = _header(_WELFORD, _PAIR) + f"""
    __device__ __forceinline__ float jt_bn_act(float z) {{ {forward_act} }}
    __device__ __forceinline__ float jt_bn_act_grad(float gs, float z) {{ {grad_act} }}
    """
    parts = segments * channels
    apply_v4 = """
        float4 v = reinterpret_cast<const float4*>(x)[i];
        float k = coef[c], b = coef[%d + c];
        reinterpret_cast<float4*>(y)[i] = make_float4(
            jt_bn_act(v.x * k + b), jt_bn_act(v.y * k + b),
            jt_bn_act(v.z * k + b), jt_bn_act(v.w * k + b));
    """ % channels
    apply_body = """
        y[i] = out0_type(jt_bn_act(static_cast<float>(x[i]) * coef[c] + coef[%d + c]));
    """ % channels
    residual_v4 = """
        float4 v = reinterpret_cast<const float4*>(x)[i];
        float4 s = reinterpret_cast<const float4*>(r)[i];
        float k = coef[c], b = coef[%d + c];
        reinterpret_cast<float4*>(y)[i] = make_float4(
            jt_bn_act(v.x * k + b + s.x), jt_bn_act(v.y * k + b + s.y),
            jt_bn_act(v.z * k + b + s.z), jt_bn_act(v.w * k + b + s.w));
    """ % channels
    residual_body = """
        y[i] = out0_type(jt_bn_act(static_cast<float>(x[i]) * coef[c] + coef[%d + c]
                                   + static_cast<float>(r[i])));
    """ % channels
    # The backward takes the gradient through the activation from the value it
    # had, recomputed from x and the forward's coefficients (`fcoef`).
    grad_v4 = """
        float4 gs = reinterpret_cast<const float4*>(grad_y)[i];
        float4 v = reinterpret_cast<const float4*>(x)[i];
        float fk = fcoef[c], fb = fcoef[%d + c];
        float4 g = make_float4(
            jt_bn_act_grad(gs.x, v.x * fk + fb), jt_bn_act_grad(gs.y, v.y * fk + fb),
            jt_bn_act_grad(gs.z, v.z * fk + fb), jt_bn_act_grad(gs.w, v.w * fk + fb));
        float k1 = coef[c], k2 = coef[%d + c], k3 = coef[%d + c];
        reinterpret_cast<float4*>(grad_x)[i] = make_float4(
            k1 * g.x + k2 * v.x + k3, k1 * g.y + k2 * v.y + k3,
            k1 * g.z + k2 * v.z + k3, k1 * g.w + k2 * v.w + k3);
    """ % (channels, channels, 2 * channels)
    grad_body = """
        float v = static_cast<float>(x[i]);
        float g = jt_bn_act_grad(static_cast<float>(grad_y[i]), v * fcoef[c] + fcoef[%d + c]);
        grad_x[i] = out0_type(coef[c] * g + coef[%d + c] * v + coef[%d + c]);
    """ % (channels, channels, 2 * channels)

    statistics_launch = f"batch_norm_statistics<<<dim3({channels}, {segments}), {_THREADS}>>>(in0_p, out3_p);"
    statistics_v4 = ""
    sums_launch = (f"batch_norm_backward_sums<<<dim3({channels}, {segments}), {_THREADS}>>>("
                   "in0_p, in1_p, in2_p, in5_p, out3_p);")
    sums_v4 = ""
    if vector == 4:
        # Four elements a step: each group of four is folded into the
        # running Welford state at once, one reciprocal per four elements.
        statistics_body = """
            float4 v = x[index];
            float m4 = (v.x + v.y + v.z + v.w) * 0.25f;
            float d0 = v.x - m4, d1 = v.y - m4, d2 = v.z - m4, d3 = v.w - m4;
            float n = local.n + 4.0f;
            float delta = m4 - local.mean;
            float w = 4.0f * __frcp_rn(n);
            local.mean += delta * w;
            local.m2 += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3
                + delta * delta * local.n * w;
            local.n = n;
        """
        statistics_v4 = f"""
            __global__ static void batch_norm_statistics_v4(
                    const float4* x, float* partial) {{
                typedef cub::BlockReduce<JtBnWelford, {_THREADS}> BlockReduce;
                __shared__ typename BlockReduce::TempStorage storage;
                JtBnWelford local{{0.0f, 0.0f, 0.0f}};
                {_channel_loop(channels, spatial, count, per_segment, statistics_body, 4)}
                JtBnWelford total = BlockReduce(storage).Reduce(local, JtBnWelfordSum());
                if (threadIdx.x == 0) {{
                    int slot = blockIdx.y * {channels} + channel;
                    partial[slot] = total.n;
                    partial[{parts} + slot] = total.mean;
                    partial[{2 * parts} + slot] = total.m2;
                }}
            }}
        """
        statistics_launch = (
            f"if (((size_t)in0_p & 15) == 0) batch_norm_statistics_v4<<<dim3({channels}, "
            f"{segments}), {_THREADS}>>>((const float4*)in0_p, out3_p);\n"
            f"else {statistics_launch}")
        sums_body = """
            float4 v = x[index], g = grad_y[index];
            float d0 = jt_bn_act_grad(g.x, v.x * fk + fb);
            float d1 = jt_bn_act_grad(g.y, v.y * fk + fb);
            float d2 = jt_bn_act_grad(g.z, v.z * fk + fb);
            float d3 = jt_bn_act_grad(g.w, v.w * fk + fb);
            local.a += (d0 + d1) + (d2 + d3);
            local.b += (d0 * (v.x - center) + d1 * (v.y - center))
                + (d2 * (v.z - center) + d3 * (v.w - center));
        """
        sums_v4 = f"""
                __global__ static void batch_norm_backward_sums_v4(
                        const float4* grad_y, const float4* x,
                        const float* mean, const float* fcoef, float* partial) {{
                    typedef cub::BlockReduce<JtBnPair, {_THREADS}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    float center = mean[blockIdx.x];
                    float fk = fcoef[blockIdx.x], fb = fcoef[{channels} + blockIdx.x];
                    JtBnPair local{{0.0f, 0.0f}};
                    {_channel_loop(channels, spatial, count, per_segment, sums_body, 4)}
                    JtBnPair total = BlockReduce(storage).Reduce(local, JtBnPairSum());
                    if (threadIdx.x == 0) {{
                        int slot = blockIdx.y * {channels} + channel;
                        partial[slot] = total.a;
                        partial[{parts} + slot] = total.b;
                    }}
                }}
        """
        sums_launch = (
            f"if ((((size_t)in0_p | (size_t)in1_p) & 15) == 0) batch_norm_backward_sums_v4"
            f"<<<dim3({channels}, {segments}), {_THREADS}>>>((const float4*)in0_p, "
            f"(const float4*)in1_p, in2_p, in5_p, out3_p);\n"
            f"else {sums_launch}")

    def statistics(x, weight, bias):
        """mean, var, rstd and the (scale, shift) the apply uses."""
        mean, var, rstd, partial, coef = jt.code(
            [(channels,), (channels,), (channels,), (3 * parts,), (2 * channels,)],
            ["float32", "float32", "float32", "float32", "float32"],
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
            {statistics_v4}
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
            {statistics_launch}
            batch_norm_finish<<<{_per_channel(channels)}>>>(
                out3_p, in1_p, in2_p, out0_p, out1_p, out2_p, out4_p);
            CHECK(0 == cudaGetLastError());
            """,
        )
        return mean, var, rstd, coef

    def apply(x, coef, r=None):
        if r is not None:
            return jt.code(
                x.shape, x.dtype, [x, coef, r],
                cuda_header=header,
                cuda_src=f"""
                {_elementwise("batch_norm_apply_residual",
                              "const in0_type* x, const in1_type* coef, "
                              "const in2_type* r, out0_type* y",
                              residual_body, residual_v4, total, spatial, channels, vector)}
                {_launch("batch_norm_apply_residual", "in0_p, in1_p, in2_p, out0_p",
                         ("in0_p", "in2_p", "out0_p"), total, vector)}
                CHECK(0 == cudaGetLastError());
                """,
            )
        return jt.code(
            x.shape, x.dtype, [x, coef],
            cuda_header=header,
            cuda_src=f"""
            {_elementwise("batch_norm_apply",
                          "const in0_type* x, const in1_type* coef, out0_type* y",
                          apply_body, apply_v4, total, spatial, channels, vector)}
            {_launch("batch_norm_apply", "in0_p, in1_p, out0_p",
                     ("in0_p", "out0_p"), total, vector)}
            CHECK(0 == cudaGetLastError());
            """,
        )

    class BatchNormCUDA(jt.Function):
        # The statistics and the output are separate operators so that an
        # activation taken into the pass (`_batch_norm_cuda_statistics`) can
        # reuse a call's statistics -- which the running buffers read -- and
        # replace only its output.
        def execute(self, x, weight, bias, *rest):
            r, stats = (rest[0], rest[1:]) if residual else (None, rest)
            if stats:
                mean, var, rstd, coef = stats
            else:
                mean, var, rstd, coef = statistics(x, weight, bias)
            y = apply(x, coef, r)
            self.output = y if residual else None
            self.stats_given = len(stats)
            self.saved = x, mean, rstd, weight, coef
            # The statistics the step computed anyway, for the running
            # buffers; outside the tape, like the buffers themselves.
            self.statistics = mean.stop_grad(), var.stop_grad()
            self.all_statistics = tuple(v.stop_grad() for v in (mean, var, rstd, coef))
            return y

        def grad(self, grad_y):
            x, mean, rstd, weight, fcoef = self.saved
            if residual:
                # relu's gradient, from the output; the residual's is the same.
                grad_y = grad_y * (self.output > 0.0).cast(grad_y.dtype)
            grad_x, grad_weight, grad_bias, partial, coef = jt.code(
                [grad_y.shape, weight.shape, weight.shape,
                 (2 * parts,), (3 * channels,)],
                [grad_y.dtype, weight.dtype, weight.dtype, "float32", "float32"],
                [grad_y, x, mean, rstd, weight, fcoef],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void batch_norm_backward_sums(
                        const in0_type* grad_y, const in1_type* x,
                        const in2_type* mean, const in5_type* fcoef, float* partial) {{
                    typedef cub::BlockReduce<JtBnPair, {_THREADS}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    float center = static_cast<float>(mean[blockIdx.x]);
                    float fk = fcoef[blockIdx.x], fb = fcoef[{channels} + blockIdx.x];
                    JtBnPair local{{0.0f, 0.0f}};
                    {_channel_loop(channels, spatial, count, per_segment, '''
                        float v = static_cast<float>(x[index]);
                        float dy = jt_bn_act_grad(static_cast<float>(grad_y[index]),
                                                  v * fk + fb);
                        local.a += dy;
                        local.b += dy * (v - center);
                    ''')}
                    JtBnPair total = BlockReduce(storage).Reduce(local, JtBnPairSum());
                    if (threadIdx.x == 0) {{
                        int slot = blockIdx.y * {channels} + channel;
                        partial[slot] = total.a;
                        partial[{parts} + slot] = total.b;
                    }}
                }}
                {sums_v4}
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
                              "const float* fcoef, const float* coef, out0_type* grad_x",
                              grad_body, grad_v4, total, spatial, channels, vector)}
                {sums_launch}
                batch_norm_backward_finish<<<{_per_channel(channels)}>>>(
                    out3_p, in2_p, in3_p, in4_p, out1_p, out2_p, out4_p);
                {_launch("batch_norm_backward_apply", "in0_p, in1_p, in5_p, out4_p, out0_p",
                         ("in0_p", "in1_p", "out0_p"), total, vector)}
                CHECK(0 == cudaGetLastError());
                """,
            )
            grad_residual = (grad_y,) if residual else ()
            return (grad_x, grad_weight, grad_bias) + grad_residual + (None,) * self.stats_given

    return BatchNormCUDA


def _nhwc_blocks(channels, count):
    """Launch shape of an NHWC reduction: lanes over float4s of channels,
    rows over the rest of the block and over (row) segments."""
    lanes = channels // 4
    tx = min(lanes, 32)
    ty = _THREADS // tx
    tiles = -(-lanes // tx)
    segments = max(1, min(-(-_TARGET_BLOCKS // tiles), count // (ty * _MIN_PER_THREAD)))
    per_segment = -(-count // segments)
    return tx, ty, tiles, segments, per_segment


def _nhwc_reduction(name, args, fields, setup, body, store, tx, ty, tiles, per_segment, count):
    """A kernel folding rows of NHWC float4s into `fields` per lane, then over
    the block's rows (shared memory tree), then storing through `store`."""
    width = 1
    while width < ty:
        width *= 2
    declare = "\n".join(f"float {f}[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};" for f in fields)
    save = "\n".join(f"for (int k = 0; k < 4; k++) shared[{i * 4} + k][ty][tx] = {f}[k];"
                      for i, f in enumerate(fields))
    load = "\n".join(f"for (int k = 0; k < 4; k++) {f}[k] = shared[{i * 4} + k][ty][tx];"
                      for i, f in enumerate(fields))
    return f"""
    __global__ static void {name}({args}) {{
        __shared__ float shared[{4 * len(fields)}][{ty}][{tx}];
        int tx = threadIdx.x % {tx}, ty = threadIdx.x / {tx};
        int lane = blockIdx.x * {tx} + tx;
        long long begin = (long long)blockIdx.y * {per_segment};
        long long end = begin + {per_segment};
        if (end > {count}) end = {count};
        {declare}
        if (lane < {tiles * tx} && lane * 4 < CHANNELS) {{
            {setup}
            for (long long r = begin + ty; r < end; r += {ty}) {{
                long long item = r * (CHANNELS / 4) + lane;
                {body}
            }}
        }}
        {save}
        __syncthreads();
        for (int span = {width // 2}; span > 0; span >>= 1) {{
            if (ty < span && ty + span < {ty}) {{
                {load.replace("[ty][tx]", "[ty + span][tx]").replace("] = shared", "] = shared").replace("float ", "")}
            }}
            __syncthreads();
        }}
        if (ty == 0 && lane * 4 < CHANNELS) {{
            {load}
            {store}
        }}
    }}
    """


@lru_cache(maxsize=128)
def _batch_norm_nhwc_cls(count, channels, eps, act="", residual=False):
    """The training batch norm over dense NHWC storage: `x` is [N, H, W, C]
    with `count` = N * H * W rows, the channels innermost.

    What a channels-last convolution hands out in training
    (`jittor.nn.backends.cudnn.channels_last_training`). The arithmetic,
    the statistics' partial layout and the finish kernels are those of
    `_batch_norm_cuda_cls`; only the walks differ: a reduction's lanes cover
    four channels each and its threads step over rows, so a warp reads 512
    contiguous bytes of one row, and an elementwise pass finds a float4's
    channels as `item % (C / 4)`.
    """
    assert channels % 4 == 0, channels
    total = count * channels
    tx, ty, tiles, segments, per_segment = _nhwc_blocks(channels, count)
    parts = segments * channels
    forward_act, grad_act = _ACTIVATIONS[act]
    if residual:
        assert act == "relu", act
        grad_act = _ACTIVATIONS[""][1]
    header = _header(_WELFORD, _PAIR) + f"""
    #define CHANNELS {channels}
    __device__ __forceinline__ float jt_bn_act(float z) {{ {forward_act} }}
    __device__ __forceinline__ float jt_bn_act_grad(float gs, float z) {{ {grad_act} }}
    __device__ __forceinline__ float4 jt_bn_ld4(const float* p, int lane) {{
        return reinterpret_cast<const float4*>(p)[lane];
    }}
    """
    grid = f"dim3({tiles}, {segments}), {_THREADS if tx * ty == _THREADS else tx * ty}"
    elementwise_blocks = max(1, min(-(-(total // 4) // _THREADS), 65535 * 8))

    def elementwise(name, args, body):
        return f"""
        __global__ static void {name}({args}) {{
            long long stride = (long long)gridDim.x * blockDim.x;
            for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
                    i < {total // 4}LL; i += stride) {{
                int lane = (int)(i % (CHANNELS / 4));
                {body}
            }}
        }}
        """

    statistics_kernel = _nhwc_reduction(
        "batch_norm_statistics_nhwc", "const float* x_, float* partial",
        ("n", "mean", "m2"), "const float4* x = (const float4*)x_;", """
            float4 v4 = x[item];
            float v[4] = {v4.x, v4.y, v4.z, v4.w};
            float count = n[0] + 1.0f, inv = __frcp_rn(count);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float delta = v[k] - mean[k];
                mean[k] += delta * inv;
                m2[k] += delta * (v[k] - mean[k]);
                n[k] = count;
            }
        """, f"""
            for (int k = 0; k < 4; k++) {{
                int slot = blockIdx.y * CHANNELS + lane * 4 + k;
                partial[slot] = n[k];
                partial[{parts} + slot] = mean[k];
                partial[{2 * parts} + slot] = m2[k];
            }}
        """, tx, ty, tiles, per_segment, count)
    # The tree step folds Welford states, not sums: patch its combine.
    assert "n[k] = shared[0 + k][ty + span][tx]" in statistics_kernel
    statistics_kernel = statistics_kernel.replace(
        """for (int k = 0; k < 4; k++) n[k] = shared[0 + k][ty + span][tx];
for (int k = 0; k < 4; k++) mean[k] = shared[4 + k][ty + span][tx];
for (int k = 0; k < 4; k++) m2[k] = shared[8 + k][ty + span][tx];""",
        """for (int k = 0; k < 4; k++) {
                    JtBnWelford a{shared[0 + k][ty][tx], shared[4 + k][ty][tx], shared[8 + k][ty][tx]};
                    JtBnWelford b{shared[0 + k][ty + span][tx], shared[4 + k][ty + span][tx], shared[8 + k][ty + span][tx]};
                    JtBnWelford c = JtBnWelfordSum()(a, b);
                    shared[0 + k][ty][tx] = c.n; shared[4 + k][ty][tx] = c.mean; shared[8 + k][ty][tx] = c.m2;
                }""")
    sums_kernel = _nhwc_reduction(
        "batch_norm_backward_sums_nhwc",
        "const float* grad_y_, const float* x_, const float* mean_, const float* fcoef, float* partial",
        ("a", "b"), """
            const float4* grad_y = (const float4*)grad_y_;
            const float4* x = (const float4*)x_;
            float4 m4 = jt_bn_ld4(mean_, lane), k4 = jt_bn_ld4(fcoef, lane);
            float4 b4 = jt_bn_ld4(fcoef + CHANNELS, lane);
            float center[4] = {m4.x, m4.y, m4.z, m4.w};
            float fk[4] = {k4.x, k4.y, k4.z, k4.w}, fb[4] = {b4.x, b4.y, b4.z, b4.w};
        """, """
            float4 g4 = grad_y[item], v4 = x[item];
            float g[4] = {g4.x, g4.y, g4.z, g4.w}, v[4] = {v4.x, v4.y, v4.z, v4.w};
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float d = jt_bn_act_grad(g[k], v[k] * fk[k] + fb[k]);
                a[k] += d;
                b[k] += d * (v[k] - center[k]);
            }
        """, f"""
            for (int k = 0; k < 4; k++) {{
                int slot = blockIdx.y * CHANNELS + lane * 4 + k;
                partial[slot] = a[k];
                partial[{parts} + slot] = b[k];
            }}
        """, tx, ty, tiles, per_segment, count)
    assert "a[k] = shared[0 + k][ty + span][tx]" in sums_kernel
    sums_kernel = sums_kernel.replace(
        """for (int k = 0; k < 4; k++) a[k] = shared[0 + k][ty + span][tx];
for (int k = 0; k < 4; k++) b[k] = shared[4 + k][ty + span][tx];""",
        """for (int k = 0; k < 4; k++) {
                    shared[0 + k][ty][tx] += shared[0 + k][ty + span][tx];
                    shared[4 + k][ty][tx] += shared[4 + k][ty + span][tx];
                }""")

    assert "n[k] = shared[0 + k][ty + span][tx]" not in statistics_kernel
    assert "a[k] = shared[0 + k][ty + span][tx]" not in sums_kernel
    apply_kernel = elementwise(
        "batch_norm_apply_nhwc", "const float* x, const float* coef, float* y", """
            float4 v = reinterpret_cast<const float4*>(x)[i];
            float4 k = jt_bn_ld4(coef, lane), b = jt_bn_ld4(coef + CHANNELS, lane);
            reinterpret_cast<float4*>(y)[i] = make_float4(
                jt_bn_act(v.x * k.x + b.x), jt_bn_act(v.y * k.y + b.y),
                jt_bn_act(v.z * k.z + b.z), jt_bn_act(v.w * k.w + b.w));
        """)
    residual_kernel = elementwise(
        "batch_norm_apply_residual_nhwc",
        "const float* x, const float* coef, const float* r, float* y", """
            float4 v = reinterpret_cast<const float4*>(x)[i];
            float4 s = reinterpret_cast<const float4*>(r)[i];
            float4 k = jt_bn_ld4(coef, lane), b = jt_bn_ld4(coef + CHANNELS, lane);
            reinterpret_cast<float4*>(y)[i] = make_float4(
                jt_bn_act(v.x * k.x + b.x + s.x), jt_bn_act(v.y * k.y + b.y + s.y),
                jt_bn_act(v.z * k.z + b.z + s.z), jt_bn_act(v.w * k.w + b.w + s.w));
        """)
    backward_apply_kernel = elementwise(
        "batch_norm_backward_apply_nhwc",
        "const float* grad_y, const float* x, const float* fcoef, const float* coef, float* grad_x", """
            float4 gs = reinterpret_cast<const float4*>(grad_y)[i];
            float4 v = reinterpret_cast<const float4*>(x)[i];
            float4 fk = jt_bn_ld4(fcoef, lane), fb = jt_bn_ld4(fcoef + CHANNELS, lane);
            float4 k1 = jt_bn_ld4(coef, lane), k2 = jt_bn_ld4(coef + CHANNELS, lane);
            float4 k3 = jt_bn_ld4(coef + 2 * CHANNELS, lane);
            float g0 = jt_bn_act_grad(gs.x, v.x * fk.x + fb.x);
            float g1 = jt_bn_act_grad(gs.y, v.y * fk.y + fb.y);
            float g2 = jt_bn_act_grad(gs.z, v.z * fk.z + fb.z);
            float g3 = jt_bn_act_grad(gs.w, v.w * fk.w + fb.w);
            reinterpret_cast<float4*>(grad_x)[i] = make_float4(
                k1.x * g0 + k2.x * v.x + k3.x, k1.y * g1 + k2.y * v.y + k3.y,
                k1.z * g2 + k2.z * v.z + k3.z, k1.w * g3 + k2.w * v.w + k3.w);
        """)
    launch = f"<<<{elementwise_blocks}, {_THREADS}>>>"

    def statistics(x, weight, bias):
        mean, var, rstd, partial, coef = jt.code(
            [(channels,), (channels,), (channels,), (3 * parts,), (2 * channels,)],
            ["float32", "float32", "float32", "float32", "float32"],
            [x, weight, bias],
            cuda_header=header,
            cuda_src=f"""
            {statistics_kernel}
            __global__ static void batch_norm_finish(
                    const float* partial, const in1_type* weight,
                    const in2_type* bias, float* mean, float* var,
                    float* rstd, float* coef) {{
                // A warp per channel: a narrow layer has a thousand segments.
                int c = blockIdx.x;
                JtBnWelford total{{0.0f, 0.0f, 0.0f}};
                for (int s = threadIdx.x; s < {segments}; s += 32) {{
                    int slot = s * {channels} + c;
                    JtBnWelford part{{partial[slot], partial[{parts} + slot],
                                      partial[{2 * parts} + slot]}};
                    total = JtBnWelfordSum()(total, part);
                }}
                for (int offset = 16; offset > 0; offset >>= 1) {{
                    JtBnWelford other{{__shfl_down_sync(0xffffffffu, total.n, offset),
                                       __shfl_down_sync(0xffffffffu, total.mean, offset),
                                       __shfl_down_sync(0xffffffffu, total.m2, offset)}};
                    total = JtBnWelfordSum()(total, other);
                }}
                if (threadIdx.x) return;
                float variance = total.m2 / total.n;
                float r = rsqrtf(variance + {eps:.9g}f);
                float k = r * static_cast<float>(weight[c]);
                mean[c] = total.mean;
                var[c] = variance;
                rstd[c] = r;
                coef[c] = k;
                coef[{channels} + c] = static_cast<float>(bias[c]) - total.mean * k;
            }}
            batch_norm_statistics_nhwc<<<{grid}>>>(in0_p, out3_p);
            batch_norm_finish<<<{channels}, 32>>>(
                out3_p, in1_p, in2_p, out0_p, out1_p, out2_p, out4_p);
            CHECK(0 == cudaGetLastError());
            """,
        )
        return mean, var, rstd, coef

    def apply(x, coef, r=None):
        if r is not None:
            return jt.code(
                x.shape, x.dtype, [x, coef, r],
                cuda_header=header,
                cuda_src=f"""
                {residual_kernel}
                batch_norm_apply_residual_nhwc{launch}(in0_p, in1_p, in2_p, out0_p);
                CHECK(0 == cudaGetLastError());
                """,
            )
        return jt.code(
            x.shape, x.dtype, [x, coef],
            cuda_header=header,
            cuda_src=f"""
            {apply_kernel}
            batch_norm_apply_nhwc{launch}(in0_p, in1_p, out0_p);
            CHECK(0 == cudaGetLastError());
            """,
        )

    class BatchNormNHWC(jt.Function):
        # As `_batch_norm_cuda_cls`'s class, over [N, H, W, C] storage.
        def execute(self, x, weight, bias, *rest):
            r, stats = (rest[0], rest[1:]) if residual else (None, rest)
            if stats:
                mean, var, rstd, coef = stats
            else:
                mean, var, rstd, coef = statistics(x, weight, bias)
            y = apply(x, coef, r)
            self.output = y if residual else None
            self.stats_given = len(stats)
            self.saved = x, mean, rstd, weight, coef
            self.statistics = mean.stop_grad(), var.stop_grad()
            self.all_statistics = tuple(v.stop_grad() for v in (mean, var, rstd, coef))
            return y

        def grad(self, grad_y):
            x, mean, rstd, weight, fcoef = self.saved
            if residual:
                grad_y = grad_y * (self.output > 0.0).cast(grad_y.dtype)
            if not grad_y._storage_is_contiguous() or grad_y._storage_offset():
                grad_y = grad_y.clone()
            grad_x, grad_weight, grad_bias, partial, coef = jt.code(
                [grad_y.shape, weight.shape, weight.shape, (2 * parts,), (3 * channels,)],
                [grad_y.dtype, weight.dtype, weight.dtype, "float32", "float32"],
                [grad_y, x, mean, rstd, weight, fcoef],
                cuda_header=header,
                cuda_src=f"""
                {sums_kernel}
                __global__ static void batch_norm_backward_finish(
                        const float* partial, const in2_type* mean,
                        const in3_type* rstd, const in4_type* weight,
                        out1_type* grad_weight, out2_type* grad_bias,
                        float* coef) {{
                    int c = blockIdx.x;
                    float sum_dy = 0.0f, sum_dy_centered = 0.0f;
                    for (int s = threadIdx.x; s < {segments}; s += 32) {{
                        sum_dy += partial[s * {channels} + c];
                        sum_dy_centered += partial[{parts} + s * {channels} + c];
                    }}
                    for (int offset = 16; offset > 0; offset >>= 1) {{
                        sum_dy += __shfl_down_sync(0xffffffffu, sum_dy, offset);
                        sum_dy_centered += __shfl_down_sync(0xffffffffu, sum_dy_centered, offset);
                    }}
                    if (threadIdx.x) return;
                    float r = static_cast<float>(rstd[c]);
                    float sum_dy_xhat = sum_dy_centered * r;
                    grad_bias[c] = out2_type(sum_dy);
                    grad_weight[c] = out1_type(sum_dy_xhat);
                    float k1 = r * static_cast<float>(weight[c]);
                    float k2 = -k1 * r * sum_dy_xhat / {count}.0f;
                    float k3 = -k1 * sum_dy / {count}.0f
                        - k2 * static_cast<float>(mean[c]);
                    coef[c] = k1;
                    coef[{channels} + c] = k2;
                    coef[{2 * channels} + c] = k3;
                }}
                {backward_apply_kernel}
                batch_norm_backward_sums_nhwc<<<{grid}>>>(in0_p, in1_p, in2_p, in5_p, out3_p);
                batch_norm_backward_finish<<<{channels}, 32>>>(
                    out3_p, in2_p, in3_p, in4_p, out1_p, out2_p, out4_p);
                batch_norm_backward_apply_nhwc{launch}(in0_p, in1_p, in5_p, out4_p, out0_p);
                CHECK(0 == cudaGetLastError());
                """,
            )
            grad_residual = (grad_y,) if residual else ()
            return (grad_x, grad_weight, grad_bias) + grad_residual + (None,) * self.stats_given

    return BatchNormNHWC


def _nhwc_source(x):
    """The dense NHWC storage `x` reads, when the NHWC kernels can serve it."""
    shape = tuple(int(size) for size in x.shape)
    if len(shape) != 4 or shape[1] % 4 or _dtype_name(x.dtype) != "float32":
        return None
    return channels_last_source(x)


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
    source = _nhwc_source(x)
    if source is None and shape[1] % 4 == 0 and _dtype_name(x.dtype) == "float32":
        # A convolution's output it offered channels-last: take it, and the
        # layout carries through this and on into the next convolution.
        offered = take_channels_last(x)
        if offered is not None:
            source = _nhwc_source(offered)
    if source is not None:
        return _batch_norm_nhwc_statistics(source, shape, weight, bias, eps)
    spatial = shape[2] * shape[3]
    key = (shape[0], shape[1], spatial, float(eps), _vector(x, spatial))
    cls = _batch_norm_cuda_cls(*key)
    # The call's own context, which `execute` writes the statistics onto.
    call = cls()._new_call_context()
    y = call._run_call(x, weight, bias)
    mean, var = call.statistics

    def fuse_activation(act):
        # `relu(y)` asks for this while y is still unexecuted: the same
        # statistics, which the running buffers read and which therefore run
        # anyway, and an output with the activation applied in the same pass
        # -- and its gradient taken inside the backward's. y stays a graph node
        # nobody runs unless something else reads it.
        if act not in _ACTIVATIONS:
            return None
        fused = _batch_norm_cuda_cls(*key, act)()._new_call_context()
        return fused._run_call(x, weight, bias, *call.all_statistics)

    def fuse_residual(act, r):
        # `relu(y + r)`, a bottleneck's output: the add in the same pass too.
        # A ResNet-50 training step spent 5.0 ms on the add and relu alone.
        if (act != "relu" or tuple(int(size) for size in r.shape) != shape
                or _dtype_name(r.dtype) != _dtype_name(x.dtype)):
            return None
        fused = _batch_norm_cuda_cls(*key, act, True)()._new_call_context()
        return fused._run_call(x, weight, bias, r, *call.all_statistics)
    offer_activation(y, fuse_activation, residual=fuse_residual)
    return y, mean, var


def _batch_norm_nhwc_statistics(source, shape, weight, bias, eps):
    """`_batch_norm_cuda_statistics` for an NCHW view of NHWC storage: the
    kernels read the storage as it lies, and the output goes out the same
    way, so a channels-last convolution chain stays channels-last."""
    key = (shape[0] * shape[2] * shape[3], shape[1], float(eps))
    call = _batch_norm_nhwc_cls(*key)()._new_call_context()
    out = call._run_call(source, weight, bias)
    y = channels_last_view(out)
    mean, var = call.statistics
    storage_shape = (shape[0], shape[2], shape[3], shape[1])

    def fuse_activation(act):
        if act not in _ACTIVATIONS:
            return None
        fused = _batch_norm_nhwc_cls(*key, act)()._new_call_context()
        return channels_last_view(fused._run_call(source, weight, bias, *call.all_statistics))

    def fuse_residual(act, r, storage=False):
        # `r` is an NCHW view of NHWC storage or, found through an add kept
        # channels-last, that storage itself.
        if act != "relu" or _dtype_name(r.dtype) != "float32":
            return None
        if storage:
            ok = tuple(int(size) for size in r.shape) == storage_shape \
                and r._storage_is_contiguous() and not r._storage_offset()
            residual = r if ok else None
        else:
            residual = _nhwc_source(r) if tuple(int(size) for size in r.shape) == shape else None
        if residual is None:
            return None
        fused = _batch_norm_nhwc_cls(*key, act, True)()._new_call_context()
        return channels_last_view(
            fused._run_call(source, weight, bias, residual, *call.all_statistics))
    offer_activation(y, fuse_activation, residual=fuse_residual, storage=out)
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
