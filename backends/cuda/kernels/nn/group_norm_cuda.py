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
    _PAIR, _THREADS, _WELFORD, _elementwise, _launch, _nhwc_reduction, _per_channel, _segments,
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
    # The backward reads x and the gradient once for its sums, a plane -- one
    # channel of one sample -- or a chunk of one a block: every sum it needs
    # is a per-plane sum of dy' and dy' * xhat (dy' the gradient through the
    # activation), weighted by the channel's weight for the per-group ones.
    planes = batch * channels
    per_chunk = 4096
    chunks = max(1, -(-spatial // per_chunk))
    plane_parts = chunks * planes
    plane_vector = 4 if spatial % 4 == 0 and vector == 4 else 1
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
            # One pass for the sums, `group_norm_backward_plane_sums`; then
            # per channel its weight and bias gradients, over samples and
            # chunks, and per (sample, group) row the two means the input
            # gradient needs -- its channels' plane sums weighted by the
            # channel's weight (`group_norm_backward_finish`).
            x, mean, rstd, weight, bias = self.saved
            grad_x, grad_weight, grad_bias, partial, coef = jt.code(
                [grad_y.shape, weight.shape, weight.shape, (2 * plane_parts,), (2 * rows,)],
                [grad_y.dtype, weight.dtype, weight.dtype, "float32", "float32"],
                [grad_y, x, mean, rstd, weight, bias],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void group_norm_backward_plane_sums(
                        const in0_type* grad_y, const in1_type* x, const float* mean,
                        const float* rstd, const in4_type* weight, const in5_type* bias,
                        float* partial) {{
                    // A warp per (plane, chunk): a deep layer's plane is 64
                    // elements, which left a block of 256 threads idle but 16.
                    int item = blockIdx.x * {_THREADS // 32} + threadIdx.x / 32;
                    if (item >= {plane_parts}) return;
                    int lane = threadIdx.x % 32;
                    int plane = item % {planes}, chunk = item / {planes};
                    int channel = plane % {channels};
                    int row = (plane / {channels}) * {num_groups} + channel / {channels_per_group};
                    float center = mean[row], r = rstd[row];
                    float w = static_cast<float>(weight[channel]);
                    float shift = static_cast<float>(bias[channel]);
                    long long base = (long long)plane * {spatial};
                    int begin = chunk * {per_chunk};
                    int end = begin + {per_chunk};
                    if (end > {spatial}) end = {spatial};
                    float a = 0.0f, b = 0.0f;
                    if ({plane_vector} == 4 && (((size_t)(grad_y + base) | (size_t)(x + base)) & 15) == 0) {{
                        const float4* g4 = reinterpret_cast<const float4*>(grad_y + base);
                        const float4* x4 = reinterpret_cast<const float4*>(x + base);
                        for (int j = begin / 4 + lane; j < end / 4; j += 32) {{
                            float4 gv = g4[j], xv = x4[j];
                            float h0 = (xv.x - center) * r, h1 = (xv.y - center) * r;
                            float h2 = (xv.z - center) * r, h3 = (xv.w - center) * r;
                            float d0 = jt_gn_act_grad(gv.x, h0 * w + shift);
                            float d1 = jt_gn_act_grad(gv.y, h1 * w + shift);
                            float d2 = jt_gn_act_grad(gv.z, h2 * w + shift);
                            float d3 = jt_gn_act_grad(gv.w, h3 * w + shift);
                            a += (d0 + d1) + (d2 + d3);
                            b += (d0 * h0 + d1 * h1) + (d2 * h2 + d3 * h3);
                        }}
                    }} else {{
                        for (int j = begin + lane; j < end; j += 32) {{
                            float xhat = (static_cast<float>(x[base + j]) - center) * r;
                            float dy = jt_gn_act_grad(static_cast<float>(grad_y[base + j]),
                                                      xhat * w + shift);
                            a += dy;
                            b += dy * xhat;
                        }}
                    }}
                    for (int offset = 16; offset > 0; offset >>= 1) {{
                        a += __shfl_down_sync(0xffffffffu, a, offset);
                        b += __shfl_down_sync(0xffffffffu, b, offset);
                    }}
                    if (lane == 0) {{
                        partial[item] = a;
                        partial[{plane_parts} + item] = b;
                    }}
                }}
                __global__ static void group_norm_backward_finish(
                        const float* partial, const in4_type* weight,
                        out1_type* grad_weight, out2_type* grad_bias, float* coef) {{
                    int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i < {channels}) {{
                        float gw = 0.0f, gb = 0.0f;
                        for (int n = 0; n < {batch}; n++)
                            for (int k = 0; k < {chunks}; k++) {{
                                int slot = k * {planes} + n * {channels} + i;
                                gb += partial[slot];
                                gw += partial[{plane_parts} + slot];
                            }}
                        grad_weight[i] = out1_type(gw);
                        grad_bias[i] = out2_type(gb);
                    }}
                    if (i < {rows}) {{
                        int n = i / {num_groups}, first = (i % {num_groups}) * {channels_per_group};
                        float g = 0.0f, gx = 0.0f;
                        for (int c = first; c < first + {channels_per_group}; c++) {{
                            float w = static_cast<float>(weight[c]);
                            for (int k = 0; k < {chunks}; k++) {{
                                int slot = k * {planes} + n * {channels} + c;
                                g += w * partial[slot];
                                gx += w * partial[{plane_parts} + slot];
                            }}
                        }}
                        coef[i] = g / {group_size}.0f;
                        coef[{rows} + i] = gx / {group_size}.0f;
                    }}
                }}
                {_elementwise("group_norm_backward_apply",
                              "const in0_type* grad_y, const in1_type* x, "
                              "const float* mean, const float* rstd, "
                              "const in4_type* weight, const in5_type* bias, "
                              "const float* coef, out0_type* grad_x",
                              grad_scalar, grad_v4, total, spatial, channels, vector)}
                group_norm_backward_plane_sums<<<{-(-plane_parts // (_THREADS // 32))}, {_THREADS}>>>(
                    in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, out3_p);
                group_norm_backward_finish<<<{_per_channel(max(rows, channels))}>>>(
                    out3_p, in4_p, out1_p, out2_p, out4_p);
                {_launch("group_norm_backward_apply",
                         "in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, out4_p, out0_p",
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
    // Per row, how many of its segments have stored their partial. The last
    // one finishes the row and puts its count back to zero, so the next call
    // -- a replay of a captured step too -- finds it zero. One module per
    // shape, and one stream: no two calls of this kernel overlap.
    __device__ unsigned int jt_gn_nhwc_done[{rows}];
    """
    source = f"""
    __global__ static void group_norm_nhwc_statistics(
            const in0_type* x, float* partial, float* mean, float* rstd) {{
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
        // The row's last segment to store its partial finishes the row: a
        // kernel of its own cost 3.5 us a call of an SD1.5 sampling step's
        // 49, launch and all, for 64 rows of arithmetic.
        __shared__ bool last;
        if (threadIdx.x == 0) {{
            int slot = blockIdx.y * {rows} + row;
            partial[slot] = sum.n;
            partial[{row_parts} + slot] = sum.mean;
            partial[{2 * row_parts} + slot] = sum.m2;
            __threadfence();
            last = atomicAdd(&jt_gn_nhwc_done[row], 1u) == {row_segments - 1}u;
        }}
        __syncthreads();
        if (!last || threadIdx.x >= 32) return;
        __threadfence();
        JtBnWelford total{{0.0f, 0.0f, 0.0f}};
        for (int s = threadIdx.x; s < {row_segments}; s += 32) {{
            int slot = s * {rows} + row;
            JtBnWelford part{{__ldcg(partial + slot), __ldcg(partial + {row_parts} + slot),
                              __ldcg(partial + {2 * row_parts} + slot)}};
            total = JtBnWelfordSum()(total, part);
        }}
        for (int offset = 16; offset > 0; offset >>= 1) {{
            JtBnWelford other{{__shfl_down_sync(0xffffffffu, total.n, offset),
                               __shfl_down_sync(0xffffffffu, total.mean, offset),
                               __shfl_down_sync(0xffffffffu, total.m2, offset)}};
            total = JtBnWelfordSum()(total, other);
        }}
        if (threadIdx.x == 0) {{
            mean[row] = total.mean;
            rstd[row] = rsqrtf(total.m2 / total.n + {eps:.9g}f);
            jt_gn_nhwc_done[row] = 0u;
        }}
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
    group_norm_nhwc_statistics<<<dim3({rows}, {row_segments}), {_THREADS}>>>(
        in0_p, out3_p, out1_p, out2_p);
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


def _nhwc_sample_blocks(batch, spatial, channels):
    """Launch shape of a per-sample NHWC reduction: lanes over float4s of
    channels, threads over positions, (segments of positions, samples) on the
    grid's y and z."""
    lanes = channels // 4
    tx = min(lanes, 32)
    ty = _THREADS // tx
    tiles = -(-lanes // tx)
    segments = max(1, min(-(-1024 // (tiles * batch)), spatial // (ty * 8)))
    per_segment = -(-spatial // segments)
    return tx, ty, tiles, segments, per_segment


@lru_cache(maxsize=128)
def _group_norm_nhwc_backward_source(shape, num_groups, act=""):
    """The backward of `_group_norm_nhwc_source`'s normalization, over the
    same dense NHWC storage: per (sample, channel) the sums of dz and
    dz * xhat over the positions, dz the gradient through the activation;
    from them per channel the weight and bias gradients and per (sample,
    group) the two means the input gradient needs; then that gradient,
    elementwise. The NCHW backward's arithmetic, walked the NHWC way: a
    warp reads 512 contiguous bytes of one position's channels."""
    batch, height, width, channels = shape
    spatial = height * width
    cpg = channels // num_groups
    rows = batch * num_groups
    group_size = cpg * spatial
    total = batch * spatial * channels
    tx, ty, tiles, segments, per_segment = _nhwc_sample_blocks(batch, spatial, channels)
    parts = segments * batch * channels
    grad_act = _ACTIVATIONS[act][1]
    header = _header() + f"""
    #define CHANNELS {channels}
    __device__ __forceinline__ float jt_gn_act_grad(float gs, float z) {{ {grad_act} }}
    """
    sums = _nhwc_reduction(
        "group_norm_nhwc_backward_sums",
        "const in0_type* grad_y_, const in1_type* x_, const float* mean, const float* rstd, "
        "const in4_type* weight, const in5_type* bias, float* partial",
        ("a", "b"), f"""
            const long long sample_base = (long long)blockIdx.z * {spatial} * (CHANNELS / 4);
            const float4* grad_y = (const float4*)grad_y_ + sample_base;
            const float4* x = (const float4*)x_ + sample_base;
            float center[4], rs[4], w[4], shift[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) {{
                int c = lane * 4 + k;
                int row = blockIdx.z * {num_groups} + c / {cpg};
                center[k] = mean[row];
                rs[k] = rstd[row];
                w[k] = static_cast<float>(weight[c]);
                shift[k] = static_cast<float>(bias[c]);
            }}
        """, """
            float4 g4 = grad_y[item], v4 = x[item];
            float g[4] = {g4.x, g4.y, g4.z, g4.w}, v[4] = {v4.x, v4.y, v4.z, v4.w};
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float xhat = (v[k] - center[k]) * rs[k];
                float d = jt_gn_act_grad(g[k], xhat * w[k] + shift[k]);
                a[k] += d;
                b[k] += d * xhat;
            }
        """, f"""
            for (int k = 0; k < 4; k++) {{
                int slot = (blockIdx.y * {batch} + blockIdx.z) * CHANNELS + lane * 4 + k;
                partial[slot] = a[k];
                partial[{parts} + slot] = b[k];
            }}
        """, tx, ty, tiles, per_segment, spatial)
    assert "a[k] = shared[0 + k][ty + span][tx]" in sums
    sums = sums.replace(
        """for (int k = 0; k < 4; k++) a[k] = shared[0 + k][ty + span][tx];
for (int k = 0; k < 4; k++) b[k] = shared[4 + k][ty + span][tx];""",
        """for (int k = 0; k < 4; k++) {
                    shared[0 + k][ty][tx] += shared[0 + k][ty + span][tx];
                    shared[4 + k][ty][tx] += shared[4 + k][ty + span][tx];
                }""")
    return header, sums, (tx, ty, tiles, segments, parts, rows, group_size, total, spatial, cpg)


@lru_cache(maxsize=128)
def _group_norm_nhwc_training_cls(shape, num_groups, eps, act=""):
    """Training group norm over dense NHWC storage `x` [N, H, W, C]: what a
    channels-last convolution hands out in training, normalized and handed
    on in the same layout. See `_group_norm_nhwc_backward_source`."""
    batch, height, width, channels = shape
    assert channels % 4 == 0, channels
    fwd_header, fwd_src, rows, fwd_parts = _group_norm_nhwc_source(shape, num_groups, eps, act)
    header, sums, dims = _group_norm_nhwc_backward_source(shape, num_groups, act)
    tx, ty, tiles, segments, parts, rows, group_size, total, spatial, cpg = dims
    elementwise_blocks = max(1, min(-(-(total // 4) // _THREADS), 65535 * 8))

    class GroupNormNHWC(jt.Function):
        def execute(self, x, weight, bias):
            y, mean, rstd, _ = jt.code(
                [x.shape, (rows,), (rows,), (fwd_parts,)],
                [x.dtype, "float32", "float32", "float32"],
                [x, weight, bias], cuda_header=fwd_header, cuda_src=fwd_src)
            self.saved = x, mean, rstd, weight, bias
            return y

        def grad(self, grad_y):
            x, mean, rstd, weight, bias = self.saved
            if not grad_y._storage_is_contiguous() or grad_y._storage_offset():
                grad_y = grad_y.clone()
            grad_x, grad_weight, grad_bias, _, _ = jt.code(
                [grad_y.shape, weight.shape, weight.shape, (2 * parts,), (2 * rows,)],
                [grad_y.dtype, weight.dtype, weight.dtype, "float32", "float32"],
                [grad_y, x, mean, rstd, weight, bias],
                cuda_header=header,
                cuda_src=f"""
                {sums}
                __global__ static void group_norm_nhwc_backward_finish(
                        const float* partial, const in4_type* weight,
                        out1_type* grad_weight, out2_type* grad_bias, float* coef) {{
                    int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i < CHANNELS) {{
                        float gw = 0.0f, gb = 0.0f;
                        for (int s = 0; s < {segments * batch}; s++) {{
                            gb += partial[s * CHANNELS + i];
                            gw += partial[{parts} + s * CHANNELS + i];
                        }}
                        grad_weight[i] = out1_type(gw);
                        grad_bias[i] = out2_type(gb);
                    }}
                    if (i < {rows}) {{
                        int n = i / {num_groups}, first = (i % {num_groups}) * {cpg};
                        float g = 0.0f, gx = 0.0f;
                        for (int c = first; c < first + {cpg}; c++) {{
                            float w = static_cast<float>(weight[c]);
                            for (int s = 0; s < {segments}; s++) {{
                                int slot = (s * {batch} + n) * CHANNELS + c;
                                g += w * partial[slot];
                                gx += w * partial[{parts} + slot];
                            }}
                        }}
                        coef[i] = g / {group_size}.0f;
                        coef[{rows} + i] = gx / {group_size}.0f;
                    }}
                }}
                __global__ static void group_norm_nhwc_backward_apply(
                        const in0_type* grad_y, const in1_type* x, const float* mean,
                        const float* rstd, const in4_type* weight, const in5_type* bias,
                        const float* coef, out0_type* grad_x) {{
                    long long stride = (long long)gridDim.x * blockDim.x;
                    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
                            i < {total // 4}LL; i += stride) {{
                        int lane = (int)(i % (CHANNELS / 4));
                        int n = (int)(i / ({spatial}LL * (CHANNELS / 4)));
                        float4 dy = reinterpret_cast<const float4*>(grad_y)[i];
                        float4 v = reinterpret_cast<const float4*>(x)[i];
                        float g[4] = {{dy.x, dy.y, dy.z, dy.w}}, xv[4] = {{v.x, v.y, v.z, v.w}};
                        float out[4];
                        #pragma unroll
                        for (int k = 0; k < 4; k++) {{
                            int c = lane * 4 + k;
                            int row = n * {num_groups} + c / {cpg};
                            float r = rstd[row], w = static_cast<float>(weight[c]);
                            float xhat = (xv[k] - mean[row]) * r;
                            float d = jt_gn_act_grad(g[k], xhat * w + static_cast<float>(bias[c])) * w;
                            out[k] = r * (d - coef[row] - xhat * coef[{rows} + row]);
                        }}
                        reinterpret_cast<float4*>(grad_x)[i] = make_float4(out[0], out[1], out[2], out[3]);
                    }}
                }}
                group_norm_nhwc_backward_sums<<<dim3({tiles}, {segments}, {batch}), {tx * ty}>>>(
                    in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, out3_p);
                group_norm_nhwc_backward_finish<<<{_per_channel(max(rows, channels))}>>>(
                    out3_p, in4_p, out1_p, out2_p, out4_p);
                group_norm_nhwc_backward_apply<<<{elementwise_blocks}, {_THREADS}>>>(
                    in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, out4_p, out0_p);
                CHECK(0 == cudaGetLastError());
                """,
            )
            return grad_x, grad_weight, grad_bias

    return GroupNormNHWC


def _group_norm_nhwc_training(source, num_groups, weight, bias, eps):
    """`_group_norm_cuda` for an NCHW view of dense NHWC storage that records
    a gradient: normalized as it lies and handed on the same way, so a
    channels-last convolution chain stays channels-last."""
    shape = tuple(int(size) for size in source.shape)
    if shape[3] % 4 or _dtype_name(source.dtype) != "float32":
        return None
    key = (shape, int(num_groups), float(eps))
    y = channels_last_view(_group_norm_nhwc_training_cls(*key).apply(source, weight, bias))

    def fuse_activation(act):
        if act not in _ACTIVATIONS:
            return None
        return channels_last_view(
            _group_norm_nhwc_training_cls(*key, act).apply(source, weight, bias))
    offer_activation(y, fuse_activation)
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
        source = channels_last_source(x)
        if source is not None:
            nhwc = _group_norm_nhwc_training(source, num_groups, weight, bias, eps)
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
