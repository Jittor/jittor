# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms in LICENSE.txt.
"""FP32 depthwise stencils over dense channels-last storage, without gradients."""
from functools import lru_cache

from jittor._core.dtypes import dtype_name

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad


@lru_cache(maxsize=128)
def _source(n, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw):
    tile = 4
    oh = (h + 2*ph - dh*(kh-1) - 1)//sh + 1
    ow = (w + 2*pw - dw*(kw-1) - 1)//sw + 1
    width_tiles = (ow+tile-1)//tile
    cv = c//4
    total = n*oh*width_tiles*cv
    span = (tile-1)*sw+(kw-1)*dw+1
    header = f'''
    template<bool aligned>
    __device__ __forceinline__ float4 dw_load4(const float* p, int i) {{
        if (aligned) return ((const float4*)p)[i];
        p += 4*i;
        return make_float4(p[0],p[1],p[2],p[3]);
    }}
    template<bool aligned>
    __device__ __forceinline__ void dw_store4(float* p, int i, float4 v) {{
        if (aligned) {{ ((float4*)p)[i]=v; return; }}
        p += 4*i;
        p[0]=v.x; p[1]=v.y; p[2]=v.z; p[3]=v.w;
    }}
    template<bool aligned>
    __global__ static void nhwc_depthwise_width_tile(const float* x, const float* weight, float* y) {{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= {total}) return;
        int ch = idx % {cv};
        int pixel = idx / {cv};
        int col = (pixel % {width_tiles}) * {tile};
        int row = pixel / {width_tiles} % {oh};
        int batch = pixel / {width_tiles*oh};
        float4 result[{tile}];
        #pragma unroll
        for (int t=0; t<{tile}; ++t) result[t] = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int ky=0; ky<{kh}; ++ky) {{
            int iy = row*{sh}-{ph}+ky*{dh};
            if ((unsigned)iy >= {h}) continue;
            float4 values[{span}];
            #pragma unroll
            for (int t=0; t<{span}; ++t) {{
                int ix = col*{sw}-{pw}+t;
                values[t] = (unsigned)ix < {w}
                    ? dw_load4<aligned>(x,((batch*{h}+iy)*{w}+ix)*{cv}+ch)
                    : make_float4(0.f, 0.f, 0.f, 0.f);
            }}
            #pragma unroll
            for (int kx=0; kx<{kw}; ++kx) {{
                float4 k = dw_load4<aligned>(weight,(ky*{kw}+kx)*{cv}+ch);
                #pragma unroll
                for (int t=0; t<{tile}; ++t) {{
                    int ix = (col+t)*{sw}-{pw}+kx*{dw};
                    if ((unsigned)ix < {w}) {{
                        float4 v = values[t*{sw}+kx*{dw}];
                        result[t].x=fmaf(v.x,k.x,result[t].x);
                        result[t].y=fmaf(v.y,k.y,result[t].y);
                        result[t].z=fmaf(v.z,k.z,result[t].z);
                        result[t].w=fmaf(v.w,k.w,result[t].w);
                    }}
                }}
            }}
        }}
        #pragma unroll
        for (int t=0; t<{tile}; ++t)
            if (col+t<{ow}) dw_store4<aligned>(y,((batch*{oh}+row)*{ow}+col+t)*{cv}+ch,result[t]);
    }}
    '''
    launch = f'''
    if ((((uintptr_t)in0_p | (uintptr_t)in1_p | (uintptr_t)out0_p) & 15) == 0)
        nhwc_depthwise_width_tile<true><<<({total}+255)/256,256>>>(in0_p,in1_p,out0_p);
    else
        nhwc_depthwise_width_tile<false><<<({total}+255)/256,256>>>(in0_p,in1_p,out0_p);
    CHECK(cudaGetLastError()==cudaSuccess);
    '''
    return header, launch

def supports_depthwise_nhwc(x, weight, bias, stride, padding, dilation, groups, **kwargs):
    from jittor.nn.functional._layout import channels_last_source

    eligible = (jt.flags.use_cuda and not jt.flags.amp_reg
            and kwargs.get('_depthwise_fast_path', True)
            and dtype_name(x.dtype) == 'float32'
            and dtype_name(weight.dtype) == 'float32'
            and (bias is None or (isinstance(bias, jt.Var) and dtype_name(bias.dtype) == 'float32'))
            and not _output_requires_grad(x, weight, bias)
            and groups == int(x.shape[1]) == int(weight.shape[0])
            and int(x.shape[1]) >= 32 and int(x.shape[1]) % 4 == 0
            and int(weight.shape[1]) == 1
            and 65536 <= x.numel() <= 2147483647
            and all(1 <= int(k) <= 7 for k in weight.shape[-2:])
            and all(int(d) == 1 for d in dilation)
            and all(1 <= int(v) <= 2 for v in stride)
            and channels_last_source(x) is not None)
    if not eligible:
        return False
    n, c, h, w = map(int, x.shape)
    kh, kw = map(int, weight.shape[-2:])
    oh = (h + 2*padding[0] - kh)//stride[0] + 1
    ow = (w + 2*padding[1] - kw)//stride[1] + 1
    return oh > 0 and ow > 0 and n*c*oh*ow <= 2147483647

def depthwise_nhwc(x, weight, bias, stride, padding, dilation, groups, **kwargs):
    from jittor.nn.functional._layout import channels_last_source, channels_last_view
    from jittor.nn.functional._amp import bias_for_compute_dtype

    dense = channels_last_source(x)
    n, c, h, w = map(int, x.shape)
    kh, kw = map(int, weight.shape[-2:])
    oh = (h + 2*padding[0] - dilation[0]*(kh-1)-1)//stride[0]+1
    ow = (w + 2*padding[1] - dilation[1]*(kw-1)-1)//stride[1]+1
    # The filter conversion stays in the graph, so an in-place weight update
    # cannot leave a stale packed filter behind.
    filt = weight._storage_permute((2, 3, 1, 0)).contiguous()
    header, launch = _source(n, c, h, w, kh, kw, *stride, *padding, *dilation)
    y = jt.code((n, oh, ow, c), 'float32', [dense, filt],
                cuda_header=header, cuda_src=launch).stop_grad()
    if bias is not None:
        y = y + bias_for_compute_dtype(y, bias)
    return channels_last_view(y).stop_grad()
