"""Fixed-order CUDA spatial means for gradient-free image features."""

from functools import lru_cache

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.dispatch import optional_kernel
from jittor.nn.functional._layout import channels_last_source


def _supports_spatial_mean(x, keepdim=False):
    if len(x.shape) != 4 or _output_requires_grad(x):
        return False
    if dtype_name(x.dtype) != "float32" and jt.flags.amp_reg & jt.amp_flags.reduce16_no_fp32_acc:
        return False
    n, c, h, w = map(int, x.shape)
    return n > 0 and c > 0 and 32 <= h * w <= 16384 and int(x.numel()) < 2**31


@lru_cache(maxsize=256)
def _spatial_source(rows, spatial, low_precision):
    # Four independent accumulators retain a reproducible FP32 reduction tree.
    # A warp handles each short row; long rows use all sixteen warps, folding
    # each warp before the shared-memory fold. No atomic summation is needed.
    vector = spatial >= 128
    split = spatial >= 8192
    stride = 512 if split else 32
    rows_per_block = 1 if split else (4 if low_precision and rows < 512 else 16)
    warps = 16 if split else rows_per_block
    scale = f"1.f / {spatial}" if low_precision else "1.f"
    finish_scale = "1.f" if low_precision else f"float({rows}) / float({rows * spatial})"
    return f'''
    __global__ static void jt_spatial_mean(const in0_type* x, out0_type* y) {{
        __shared__ float shared[512];
        int lane = threadIdx.x, warp = threadIdx.y;
        int row = blockIdx.x * {rows_per_block} + ({0 if split else 'warp'});
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int idx = lane + {32 if split else 0} * warp;
        if (row < {rows}) {{
            const in0_type* input = x + (long long)row * {spatial};
            constexpr float scale = {scale};
            if ({int(vector)}) {{
                for (; idx * 4 + 3 < {spatial}; idx += {stride}) {{
                    a0 += float(input[idx * 4]) * scale;
                    a1 += float(input[idx * 4 + 1]) * scale;
                    a2 += float(input[idx * 4 + 2]) * scale;
                    a3 += float(input[idx * 4 + 3]) * scale;
                }}
            }} else {{
                for (; idx + 3 * {stride} < {spatial}; idx += 4 * {stride}) {{
                    a0 += float(input[idx]) * scale;
                    a1 += float(input[idx + {stride}]) * scale;
                    a2 += float(input[idx + 2 * {stride}]) * scale;
                    a3 += float(input[idx + 3 * {stride}]) * scale;
                }}
                if (idx < {spatial}) a0 += float(input[idx]) * scale;
                if (idx + {stride} < {spatial}) a1 += float(input[idx + {stride}]) * scale;
                if (idx + 2 * {stride} < {spatial}) a2 += float(input[idx + 2 * {stride}]) * scale;
                if (idx + 3 * {stride} < {spatial}) a3 += float(input[idx + 3 * {stride}]) * scale;
            }}
        }}
        float total = ((a0 + a1) + a2) + a3;
        __syncthreads();
        for (int offset = 16; offset; offset /= 2)
            total += __shfl_down_sync(0xffffffffu, total, offset);
        if ({int(split)}) {{
            shared[warp * 32 + lane] = total;
            for (int offset = 8; offset; offset /= 2) {{
                __syncthreads();
                if (warp < offset) {{
                    total += shared[(warp + offset) * 32 + lane];
                    shared[warp * 32 + lane] = total;
                }}
            }}
        }}
        if (lane == 0 && ({int(not split)} || warp == 0) && row < {rows})
            y[row] = static_cast<out0_type>(total * ({finish_scale}));
    }}
    jt_spatial_mean<<<({rows} + {rows_per_block} - 1) / {rows_per_block}, dim3(32, {warps})>>>(
        in0_p, out0_p);
    '''


_NHWC_SPATIAL_TILE = 256
_NHWC_HEADER = rf'''
constexpr int spatial_tile = {_NHWC_SPATIAL_TILE};
constexpr int channel_tile = 16;
constexpr int spatial_threads = 8;

template <typename In, typename Out>
__global__ static void jt_spatial_mean_nhwc(const In* input, Out* output,
                                            int channels, int spatial, int tiles,
                                            int channel_tiles) {{
    __shared__ float sums[spatial_threads][channel_tile];
    int row = blockIdx.x / channel_tiles;
    int channel = blockIdx.x % channel_tiles * channel_tile + threadIdx.x;
    int first = (row % tiles) * spatial_tile;
    int end = min(first + spatial_tile, spatial);
    float sum = 0.f;
    if (channel < channels) {{
        const In* pixels = input + (row / tiles) * spatial * channels + channel;
        // Scale before the sum so a finite BF16 mean can stay within FP32 range.
        float scale = 1.f / spatial;
        for (int p = first + threadIdx.y; p < end; p += spatial_threads)
            sum += float(pixels[p * channels]) * scale;
    }}
    sums[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y == 0 && channel < channels) {{
        #pragma unroll
        for (int y = 1; y < spatial_threads; ++y) sum += sums[y][threadIdx.x];
        output[row * channels + channel] = static_cast<Out>(sum);
    }}
}}

template <typename Out>
__global__ static void jt_spatial_mean_finish(const float* partial, Out* output,
                                              int channels, int tiles, int rows) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    const float* values = partial + i / channels * tiles * channels + i % channels;
    float sum = 0.f;
    for (int tile = 0; tile < tiles; ++tile) sum += values[tile * channels];
    output[i] = static_cast<Out>(sum);
}}
'''

_NHWC_LAUNCH = r'''
    int spatial = in0_shape1 * in0_shape2, channels = in0_shape3;
    int tiles = (spatial + spatial_tile - 1) / spatial_tile;
    int channel_tiles = (channels + channel_tile - 1) / channel_tile;
    int blocks = in0_shape0 * tiles * channel_tiles;
    dim3 threads(channel_tile, spatial_threads);
'''


def _spatial_mean_nhwc(source, shape, spatial):
    n, _, _, c = map(int, source.shape)
    tiles = (spatial + _NHWC_SPATIAL_TILE - 1) // _NHWC_SPATIAL_TILE
    if tiles == 1:
        return jt.code(shape, source.dtype, [source], cuda_header=_NHWC_HEADER,
                       cuda_src=_NHWC_LAUNCH + r'''
            jt_spatial_mean_nhwc<<<blocks, threads>>>(in0_p, out0_p, channels, spatial,
                                                     tiles, channel_tiles);
        ''')
    # Large maps split the spatial work across blocks even at batch size one.
    result, _ = jt.code([shape, (n, tiles, c)], [source.dtype, "float32"], [source],
                        cuda_header=_NHWC_HEADER, cuda_src=_NHWC_LAUNCH + r'''
        jt_spatial_mean_nhwc<<<blocks, threads>>>(in0_p, out1_p, channels, spatial,
                                                 tiles, channel_tiles);
        int rows = in0_shape0 * channels;
        jt_spatial_mean_finish<<<(rows + 127) / 128, 128>>>(out1_p, out0_p, channels, tiles, rows);
    ''')
    return result


@optional_kernel("nn.spatial_mean", "cuda", dtypes={"float32", "float16", "bfloat16"},
                 supports=_supports_spatial_mean)
def _spatial_mean_cuda(x, keepdim=False):
    n, c, h, w = map(int, x.shape)
    spatial = h * w
    shape = (n, c, 1, 1) if keepdim else (n, c)
    if not x._storage_is_contiguous():
        source = channels_last_source(x)
        if source is None or n * c * spatial < 1024:
            return None
        return _spatial_mean_nhwc(source, shape, spatial).stop_grad()
    if n * c < 16 or (spatial >= 128 and spatial % 4):
        return None
    return jt.code(shape, x.dtype, [x],
                   cuda_src=_spatial_source(n * c, spatial, dtype_name(x.dtype) != "float32")).stop_grad()
