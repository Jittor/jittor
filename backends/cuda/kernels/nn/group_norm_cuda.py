"""CUDA fast path for 4-D group normalization, float32 and half precision.

Laid out like the batch normalization next to it (batch_norm_training_cuda.py):
a group's statistics are a reduction over its ``C / G * H * W`` contiguous
elements, run on a grid of (group, segment) blocks whose partial results a
one-thread-per-group kernel combines, and applying them is elementwise over
the whole tensor. The affine gradients are the same per-channel reductions as
batch norm's.

It used to be one block per (sample, group) doing everything. An SD1.5 UNet
at batch 2 has 32 groups, so every GroupNorm ran 64 blocks on a 128-SM card,
and half precision -- which is how the UNet runs -- was not accepted at all:
it went down the generic path, four fused kernels and 31 ms of a 20-step
sample, against PyTorch's 5 ms.
"""

from functools import lru_cache
import math

import jittor as jt
from jittor._core.dtypes import dtype_name as _dtype_name
from jittor._runtime.backend_libraries import library_resource
from jittor._runtime.dispatch import optional_kernel
from jittor.nn.functional._layout import channels_last_source, channels_last_view, records_no_grad
from jittor.nn.functional.activation import offer_activation

from .batch_norm_training_cuda import (
    _PAIR, _THREADS, _WELFORD, _elementwise, _launch, _per_channel, _segments,
)


def _header():
    return f"#include <{library_resource('cub', 'home')}cub/cub.cuh>\n" + _WELFORD + _PAIR


#: What `_group_norm_cuda_cls` can apply to its output in the same pass, as
#: (forward of z, gradient given the output's gradient gs and z).
_ACTIVATIONS = {
    "": ("return z;", "return gs;"),
    "silu": ("return z / (1.0f + __expf(-z));",
             "float s = 1.0f / (1.0f + __expf(-z)); return gs * s * (1.0f + z * (1.0f - s));"),
}


@lru_cache(maxsize=128)
def _group_norm_cuda_cls(shape, num_groups, eps, vector=1, act=""):
    batch, channels, height, width = shape
    spatial = height * width
    channels_per_group = channels // num_groups
    group_size = channels_per_group * spatial
    rows = batch * num_groups
    total = rows * group_size
    row_segments, per_row_segment = _segments(rows, group_size)
    row_parts = row_segments * rows
    per_sample = batch * spatial
    channel_segments, per_channel_segment = _segments(channels, per_sample)
    channel_parts = channel_segments * channels
    forward_act, grad_act = _ACTIVATIONS[act]
    # The activation, and its gradient from the normalized value, which the
    # backward recomputes from x rather than store: xhat * w + b.
    header = _header() + f"""
    __device__ __forceinline__ float jt_gn_act(float z) {{ {forward_act} }}
    __device__ __forceinline__ float jt_gn_act_grad(float gs, float z) {{ {grad_act} }}
    """
    # For an item i of the tensor: its (sample, group) row and its channel.
    locate = f"""
        long long row = i * WIDTH / {group_size};
        int channel = (int)((i * WIDTH / {spatial}) % {channels});
    """

    def body(width, text):
        return locate.replace("WIDTH", str(width)) + text

    apply_scalar = body(1, """
        float k = static_cast<float>(rstd[row]) * static_cast<float>(weight[channel]);
        float b = static_cast<float>(bias[channel]) - static_cast<float>(mean[row]) * k;
        y[i] = out0_type(jt_gn_act(static_cast<float>(x[i]) * k + b));
    """)
    apply_v4 = body(4, """
        float k = rstd[row] * static_cast<float>(weight[channel]);
        float b = static_cast<float>(bias[channel]) - mean[row] * k;
        float4 v = reinterpret_cast<const float4*>(x)[i];
        reinterpret_cast<float4*>(y)[i] = make_float4(
            jt_gn_act(v.x * k + b), jt_gn_act(v.y * k + b),
            jt_gn_act(v.z * k + b), jt_gn_act(v.w * k + b));
    """)
    grad_scalar = body(1, """
        float r = rstd[row], w = static_cast<float>(weight[channel]);
        float xhat = (static_cast<float>(x[i]) - mean[row]) * r;
        float g = jt_gn_act_grad(static_cast<float>(grad_y[i]),
                                 xhat * w + static_cast<float>(bias[channel])) * w;
        grad_x[i] = out0_type(r * (g - coef[row] - xhat * coef[%d + row]));
    """ % rows)
    grad_v4 = body(4, """
        float r = rstd[row], w = static_cast<float>(weight[channel]);
        float shift = static_cast<float>(bias[channel]);
        float center = mean[row], mg = coef[row], mgx = coef[%d + row];
        float4 dy = reinterpret_cast<const float4*>(grad_y)[i];
        float4 v = reinterpret_cast<const float4*>(x)[i];
        float hx = (v.x - center) * r, hy = (v.y - center) * r;
        float hz = (v.z - center) * r, hw = (v.w - center) * r;
        float4 out;
        out.x = r * (jt_gn_act_grad(dy.x, hx * w + shift) * w - mg - hx * mgx);
        out.y = r * (jt_gn_act_grad(dy.y, hy * w + shift) * w - mg - hy * mgx);
        out.z = r * (jt_gn_act_grad(dy.z, hz * w + shift) * w - mg - hz * mgx);
        out.w = r * (jt_gn_act_grad(dy.w, hw * w + shift) * w - mg - hw * mgx);
        reinterpret_cast<float4*>(grad_x)[i] = out;
    """ % rows)

    def row_loop(text):
        return f"""
        int row = blockIdx.x;
        long long begin = (long long)blockIdx.y * {per_row_segment};
        long long end = begin + {per_row_segment};
        if (end > {group_size}) end = {group_size};
        const long long base = (long long)row * {group_size};
        for (long long j = begin + threadIdx.x; j < end; j += {_THREADS}) {{
            int channel = (row % {num_groups}) * {channels_per_group} + (int)(j / {spatial});
            {text}
        }}
        """

    class GroupNormCUDA(jt.Function):
        def execute(self, x, weight, bias):
            # Only the two per-group statistics are carried to the backward,
            # as torch's native_group_norm does; the backward recomputes xhat
            # from x reading the same bytes a stored copy would.
            y, mean, rstd, partial = jt.code(
                [x.shape, (rows,), (rows,), (3 * row_parts,)],
                [x.dtype, "float32", "float32", "float32"],
                [x, weight, bias],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void group_norm_statistics(
                        const in0_type* x, float* partial) {{
                    typedef cub::BlockReduce<JtBnWelford, {_THREADS}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    // Welford per element. Shifted sums and squares lost a
                    // factor of twenty against the two-pass variance over the
                    // 800 K elements of an early ResNet-50 channel.
                    JtBnWelford local{{0.0f, 0.0f, 0.0f}};
                    {row_loop('''
                        float value = static_cast<float>(x[base + j]);
                        local.n += 1.0f;
                        float delta = value - local.mean;
                        local.mean += delta * __frcp_rn(local.n);
                        local.m2 += delta * (value - local.mean);
                    ''')}
                    JtBnWelford total = BlockReduce(storage).Reduce(local, JtBnWelfordSum());
                    if (threadIdx.x == 0) {{
                        int slot = blockIdx.y * {rows} + row;
                        partial[slot] = total.n;
                        partial[{row_parts} + slot] = total.mean;
                        partial[{2 * row_parts} + slot] = total.m2;
                    }}
                }}
                __global__ static void group_norm_finish(
                        const float* partial, float* mean, float* rstd) {{
                    int row = blockIdx.x * blockDim.x + threadIdx.x;
                    if (row >= {rows}) return;
                    JtBnWelford total{{0.0f, 0.0f, 0.0f}};
                    for (int s = 0; s < {row_segments}; s++) {{
                        int slot = s * {rows} + row;
                        JtBnWelford part{{partial[slot], partial[{row_parts} + slot],
                                          partial[{2 * row_parts} + slot]}};
                        total = JtBnWelfordSum()(total, part);
                    }}
                    mean[row] = total.mean;
                    rstd[row] = rsqrtf(total.m2 / total.n + {eps:.9g}f);
                }}
                {_elementwise("group_norm_apply",
                              "const in0_type* x, const in1_type* weight, "
                              "const in2_type* bias, const float* mean, "
                              "const float* rstd, out0_type* y",
                              apply_scalar, apply_v4, total, spatial, channels, vector)}
                group_norm_statistics<<<dim3({rows}, {row_segments}), {_THREADS}>>>(
                    in0_p, out3_p);
                group_norm_finish<<<{_per_channel(rows)}>>>(out3_p, out1_p, out2_p);
                {_launch("group_norm_apply", "in0_p, in1_p, in2_p, out1_p, out2_p, out0_p",
                         ("in0_p", "out0_p"), total, vector)}
                CHECK(0 == cudaGetLastError());
                """,
            )
            self.saved = x, mean, rstd, weight, bias
            return y

        def grad(self, grad_y):
            x, mean, rstd, weight, bias = self.saved
            grad_x, grad_weight, grad_bias, row_partial, channel_partial, coef = jt.code(
                [grad_y.shape, weight.shape, weight.shape,
                 (2 * row_parts,), (2 * channel_parts,), (2 * rows,)],
                [grad_y.dtype, weight.dtype, weight.dtype, "float32", "float32", "float32"],
                [grad_y, x, mean, rstd, weight, bias],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void group_norm_backward_row_sums(
                        const in0_type* grad_y, const in1_type* x, const float* mean,
                        const float* rstd, const in4_type* weight, const in5_type* bias,
                        float* partial) {{
                    typedef cub::BlockReduce<JtBnPair, {_THREADS}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    float center = mean[blockIdx.x], r = rstd[blockIdx.x];
                    JtBnPair local{{0.0f, 0.0f}};
                    {row_loop('''
                        float w = static_cast<float>(weight[channel]);
                        float xhat = (static_cast<float>(x[base + j]) - center) * r;
                        float g = jt_gn_act_grad(static_cast<float>(grad_y[base + j]),
                            xhat * w + static_cast<float>(bias[channel])) * w;
                        local.a += g;
                        local.b += g * xhat;
                    ''')}
                    JtBnPair total = BlockReduce(storage).Reduce(local, JtBnPairSum());
                    if (threadIdx.x == 0) {{
                        int slot = blockIdx.y * {rows} + row;
                        partial[slot] = total.a;
                        partial[{row_parts} + slot] = total.b;
                    }}
                }}
                __global__ static void group_norm_backward_channel_sums(
                        const in0_type* grad_y, const in1_type* x, const float* mean,
                        const float* rstd, const in4_type* weight, const in5_type* bias,
                        float* partial) {{
                    typedef cub::BlockReduce<JtBnPair, {_THREADS}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    int channel = blockIdx.x;
                    int group = channel / {channels_per_group};
                    long long begin = (long long)blockIdx.y * {per_channel_segment};
                    long long end = begin + {per_channel_segment};
                    if (end > {per_sample}) end = {per_sample};
                    JtBnPair local{{0.0f, 0.0f}};
                    for (long long item = begin + threadIdx.x; item < end; item += {_THREADS}) {{
                        long long sample = item / {spatial};
                        long long offset = item - sample * {spatial};
                        long long index = (sample * {channels} + channel) * {spatial} + offset;
                        long long row = sample * {num_groups} + group;
                        float xhat = (static_cast<float>(x[index]) - mean[row]) * rstd[row];
                        float dy = jt_gn_act_grad(static_cast<float>(grad_y[index]),
                            xhat * static_cast<float>(weight[channel])
                            + static_cast<float>(bias[channel]));
                        local.a += dy * xhat;
                        local.b += dy;
                    }}
                    JtBnPair total = BlockReduce(storage).Reduce(local, JtBnPairSum());
                    if (threadIdx.x == 0) {{
                        int slot = blockIdx.y * {channels} + channel;
                        partial[slot] = total.a;
                        partial[{channel_parts} + slot] = total.b;
                    }}
                }}
                __global__ static void group_norm_backward_finish(
                        const float* row_partial, const float* channel_partial,
                        out1_type* grad_weight, out2_type* grad_bias, float* coef) {{
                    int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i < {rows}) {{
                        float g = 0.0f, gx = 0.0f;
                        for (int s = 0; s < {row_segments}; s++) {{
                            g += row_partial[s * {rows} + i];
                            gx += row_partial[{row_parts} + s * {rows} + i];
                        }}
                        coef[i] = g / {group_size}.0f;
                        coef[{rows} + i] = gx / {group_size}.0f;
                    }}
                    if (i < {channels}) {{
                        float gw = 0.0f, gb = 0.0f;
                        for (int s = 0; s < {channel_segments}; s++) {{
                            gw += channel_partial[s * {channels} + i];
                            gb += channel_partial[{channel_parts} + s * {channels} + i];
                        }}
                        grad_weight[i] = out1_type(gw);
                        grad_bias[i] = out2_type(gb);
                    }}
                }}
                {_elementwise("group_norm_backward_apply",
                              "const in0_type* grad_y, const in1_type* x, "
                              "const float* mean, const float* rstd, "
                              "const in4_type* weight, const in5_type* bias, "
                              "const float* coef, out0_type* grad_x",
                              grad_scalar, grad_v4, total, spatial, channels, vector)}
                group_norm_backward_row_sums<<<dim3({rows}, {row_segments}), {_THREADS}>>>(
                    in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, out3_p);
                group_norm_backward_channel_sums<<<dim3({channels}, {channel_segments}),
                                                   {_THREADS}>>>(
                    in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, out4_p);
                group_norm_backward_finish<<<{_per_channel(max(rows, channels))}>>>(
                    out3_p, out4_p, out1_p, out2_p, out5_p);
                {_launch("group_norm_backward_apply",
                         "in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, out5_p, out0_p",
                         ("in0_p", "in1_p", "out0_p"), total, vector)}
                CHECK(0 == cudaGetLastError());
                """,
            )
            return grad_x, grad_weight, grad_bias

    return GroupNormCUDA


@lru_cache(maxsize=128)
def _group_norm_nhwc_source(shape, num_groups, eps, act=""):
    """Forward-only group norm over dense NHWC memory.

    What a channels-last activation -- an NCHW view of NHWC storage, which a
    half-precision convolution hands out when nothing records a gradient --
    is normalized with, in place: a group is ``C / G`` consecutive channels at
    every spatial position rather than one contiguous run, and the output is
    written in the same order so the layout carries on to the next
    convolution.
    """
    batch, height, width, channels = shape
    spatial = height * width
    channels_per_group = channels // num_groups
    group_size = channels_per_group * spatial
    rows = batch * num_groups
    total = rows * group_size
    row_segments, per_row_segment = _segments(rows, group_size)
    row_parts = row_segments * rows
    header = _header() + f"""
    __device__ __forceinline__ float jt_gn_act(float z) {{ {_ACTIVATIONS[act][0]} }}
    """
    source = f"""
    __global__ static void group_norm_nhwc_statistics(const in0_type* x, float* partial) {{
        typedef cub::BlockReduce<JtBnWelford, {_THREADS}> BlockReduce;
        __shared__ typename BlockReduce::TempStorage storage;
        int row = blockIdx.x;
        int sample = row / {num_groups}, group = row % {num_groups};
        long long begin = (long long)blockIdx.y * {per_row_segment};
        long long end = begin + {per_row_segment};
        if (end > {group_size}) end = {group_size};
        const long long base = (long long)sample * {spatial} * {channels}
            + group * {channels_per_group};
        JtBnWelford local{{0.0f, 0.0f, 0.0f}};
        for (long long j = begin + threadIdx.x; j < end; j += {_THREADS}) {{
            long long position = j / {channels_per_group};
            int inner = (int)(j - position * {channels_per_group});
            float value = static_cast<float>(x[base + position * {channels} + inner]);
            local.n += 1.0f;
            float delta = value - local.mean;
            local.mean += delta * __frcp_rn(local.n);
            local.m2 += delta * (value - local.mean);
        }}
        JtBnWelford sum = BlockReduce(storage).Reduce(local, JtBnWelfordSum());
        if (threadIdx.x == 0) {{
            int slot = blockIdx.y * {rows} + row;
            partial[slot] = sum.n;
            partial[{row_parts} + slot] = sum.mean;
            partial[{2 * row_parts} + slot] = sum.m2;
        }}
    }}
    __global__ static void group_norm_nhwc_finish(
            const float* partial, float* mean, float* rstd) {{
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= {rows}) return;
        JtBnWelford sum{{0.0f, 0.0f, 0.0f}};
        for (int s = 0; s < {row_segments}; s++) {{
            int slot = s * {rows} + row;
            JtBnWelford part{{partial[slot], partial[{row_parts} + slot],
                              partial[{2 * row_parts} + slot]}};
            sum = JtBnWelfordSum()(sum, part);
        }}
        mean[row] = sum.mean;
        rstd[row] = rsqrtf(sum.m2 / sum.n + {eps:.9g}f);
    }}
    __global__ static void group_norm_nhwc_apply(
            const in0_type* x, const in1_type* weight, const in2_type* bias,
            const float* mean, const float* rstd, out0_type* y) {{
        long long stride = (long long)gridDim.x * blockDim.x;
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
                i < {total}LL; i += stride) {{
            int channel = (int)(i % {channels});
            long long row = (i / ((long long){spatial} * {channels})) * {num_groups}
                + channel / {channels_per_group};
            float k = rstd[row] * static_cast<float>(weight[channel]);
            float b = static_cast<float>(bias[channel]) - mean[row] * k;
            y[i] = out0_type(jt_gn_act(static_cast<float>(x[i]) * k + b));
        }}
    }}
    group_norm_nhwc_statistics<<<dim3({rows}, {row_segments}), {_THREADS}>>>(in0_p, out3_p);
    group_norm_nhwc_finish<<<{_per_channel(rows)}>>>(out3_p, out1_p, out2_p);
    group_norm_nhwc_apply<<<{max(1, min(-(-total // _THREADS), 65535 * 8))}, {_THREADS}>>>(
        in0_p, in1_p, in2_p, out1_p, out2_p, out0_p);
    CHECK(0 == cudaGetLastError());
    """
    return header, source, rows, 3 * row_parts


def _group_norm_nhwc(x, num_groups, weight, bias, eps):
    """Group norm of a channels-last view that records no gradient, or None."""
    if not records_no_grad(x, weight, bias):
        return None
    source = channels_last_source(x)
    if source is None:
        return None
    shape = tuple(int(size) for size in source.shape)

    def build(act):
        header, cuda_src, rows, parts = _group_norm_nhwc_source(
            shape, int(num_groups), float(eps), act)
        y, _, _, _ = jt.code(
            [source.shape, (rows,), (rows,), (parts,)],
            [source.dtype, "float32", "float32", "float32"],
            [source, weight, bias], cuda_header=header, cuda_src=cuda_src)
        return channels_last_view(y)
    y = build("")
    # As in `_group_norm_cuda`: `silu(y)` takes the activation into the pass.
    offer_activation(y, lambda act: build(act) if act in _ACTIVATIONS else None)
    return y


def _supports_group_norm(x, num_groups, weight, bias, eps):
    if not (
        isinstance(weight, jt.Var)
        and isinstance(bias, jt.Var)
    ):
        return False
    shape = tuple(int(size) for size in x.shape)
    if len(shape) != 4 or any(size <= 0 for size in shape):
        return False
    channels = shape[1]
    num_groups = int(num_groups)
    if (
        num_groups <= 0
        or channels % num_groups
        or int(weight.numel()) != channels
        or int(bias.numel()) != channels
        or not math.isfinite(float(eps))
        or float(eps) <= 0.0
    ):
        return False
    return True


@optional_kernel("nn.group_norm", ("cuda", "rocm_legacy", "corex_legacy"),
                 dtypes=("float32", "float16", "bfloat16"),
                 supports=_supports_group_norm)
def _group_norm_cuda(x, num_groups, weight, bias, eps):
    if not x._storage_is_contiguous():
        # A channels-last activation stays channels-last; the kernels below
        # would have it copied dense first, undoing the layout for the next
        # convolution as well.
        nhwc = _group_norm_nhwc(x, num_groups, weight, bias, eps)
        if nhwc is not None:
            return nhwc
    shape = tuple(int(size) for size in x.shape)
    num_groups = int(num_groups)
    spatial = shape[2] * shape[3]
    vector = 4 if spatial % 4 == 0 and _dtype_name(x.dtype) == "float32" else 1
    cls = _group_norm_cuda_cls(shape, num_groups, float(eps), vector)
    y = cls.apply(x, weight, bias)

    def fuse_activation(act):
        # `silu(y)` asks for this while y is still unexecuted: the same group
        # norm with the activation applied in its last pass, and its gradient
        # taken inside the backward's. y itself stays a graph node nobody
        # runs unless something else reads it.
        if act not in _ACTIVATIONS:
            return None
        fused = _group_norm_cuda_cls(shape, num_groups, float(eps), vector, act)
        return fused.apply(x, weight, bias)
    offer_activation(y, fuse_activation)
    return y


__all__ = ["_group_norm_cuda"]
