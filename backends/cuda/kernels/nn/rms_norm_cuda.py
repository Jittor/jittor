"""CUDA inference kernels for multi-head RMS normalization."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import math
from functools import lru_cache

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.dispatch import native_rule, optional_kernel

from ._inference import cached_source


_autocast_probe = None


def _autocast_enabled():
    # Resolved once: the attribute lookup showed up in a decode profile at
    # thousands of calls per step, in front of a one-microsecond kernel.
    global _autocast_probe
    if _autocast_probe is None:
        _autocast_probe = getattr(jt, "is_autocast_enabled", None) or False
    probe = _autocast_probe
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:
        return True


def _rms_norm_contract(x, gamma, epsilon, residual=None):
    if not isinstance(x, jt.Var) or not isinstance(gamma, jt.Var):
        return None
    if residual is not None and not isinstance(residual, jt.Var):
        return None
    if _output_requires_grad(x, gamma, residual):
        return None
    try:
        if _autocast_enabled():
            return None
        x_shape = tuple(int(size) for size in x.shape)
        gamma_shape = tuple(int(size) for size in gamma.shape)
        epsilon_value = float(epsilon)
    except Exception:
        return None
    if not x_shape or any(size <= 0 for size in x_shape):
        return None
    hidden_size = x_shape[-1]
    if hidden_size > 16384 or gamma_shape != (hidden_size,):
        return None
    if residual is not None and tuple(int(size) for size in residual.shape) != x_shape:
        return None
    if _jittor_dtype_name(x.dtype) not in ("float16", "bfloat16", "float32"):
        return None
    if residual is not None and _jittor_dtype_name(residual.dtype) != _jittor_dtype_name(x.dtype):
        return None
    if _jittor_dtype_name(gamma.dtype) not in ("float16", "bfloat16", "float32"):
        return None
    if not math.isfinite(epsilon_value) or epsilon_value <= 0:
        return None

    threads = 32
    while threads < min(hidden_size, 1024):
        threads *= 2
    return hidden_size, threads, threads // 32, epsilon_value


_RMS_NORM_SOURCE = r"""
    __device__ __forceinline__ float warp_sum(float value) {
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        return value;
    }
    __global__ static void rms_norm(
            const in0_type* __restrict__ x,
            const in1_type* __restrict__ gamma,
            out0_type* __restrict__ y,
            float eps) {
        long long row = blockIdx.x;
        long long row_offset = row * %(hidden_size)dLL;
        int tid = threadIdx.x;
        int lane = tid & 31;
        int warp = tid >> 5;
        __shared__ float warp_buf[%(launch_warps)d];
        __shared__ float inverse_rms;
        if (sizeof(in0_type) == 4 && sizeof(out0_type) == 4 && (%(hidden_size)d %% 4 == 0)
                && ((((size_t)x | (size_t)y) & 15u) == 0)) {
            constexpr int kVecN = %(hidden_size)d / 4;
            constexpr int kVecPer = (kVecN + %(launch_threads)d - 1) / %(launch_threads)d;
            constexpr bool kVecCache = kVecPer <= 4;
            const float4* x4 = reinterpret_cast<const float4*>(x + row_offset);
            float4* y4 = reinterpret_cast<float4*>(y + row_offset);
            float4 cache[kVecCache ? kVecPer : 1];
            float sum = 0.0f;
            if (kVecCache) {
                #pragma unroll
                for (int i = 0; i < kVecPer; ++i) {
                    int v = tid + i * %(launch_threads)d;
                    if (v < kVecN) {
                        float4 val = x4[v];
                        cache[i] = val;
                        sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
                    }
                }
            } else {
                for (int v = tid; v < kVecN; v += %(launch_threads)d) {
                    float4 val = x4[v];
                    sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
                }
            }
            sum = warp_sum(sum);
            if (lane == 0) warp_buf[warp] = sum;
            __syncthreads();
            if (warp == 0) {
                float total = lane < %(launch_warps)d ? warp_buf[lane] : 0.0f;
                total = warp_sum(total);
                if (lane == 0)
                    inverse_rms = rsqrtf(total / %(hidden_size)d.0f + eps);
            }
            __syncthreads();
            float inv = inverse_rms;
            bool g_vec = (sizeof(in1_type) == 4) && ((((size_t)gamma) & 15u) == 0);
            const float4* g4 = g_vec ? reinterpret_cast<const float4*>(gamma) : nullptr;
            if (kVecCache) {
                #pragma unroll
                for (int i = 0; i < kVecPer; ++i) {
                    int v = tid + i * %(launch_threads)d;
                    if (v < kVecN) {
                        float4 val = cache[i];
                        float4 gv;
                        if (g_vec) {
                            gv = g4[v];
                        } else {
                            int d = v * 4;
                            gv = make_float4(
                                static_cast<float>(gamma[d]),
                                static_cast<float>(gamma[d + 1]),
                                static_cast<float>(gamma[d + 2]),
                                static_cast<float>(gamma[d + 3]));
                        }
                        float4 outv;
                        outv.x = (val.x * inv) * gv.x;
                        outv.y = (val.y * inv) * gv.y;
                        outv.z = (val.z * inv) * gv.z;
                        outv.w = (val.w * inv) * gv.w;
                        y4[v] = outv;
                    }
                }
            } else {
                for (int v = tid; v < kVecN; v += %(launch_threads)d) {
                    float4 val = x4[v];
                    float4 gv;
                    if (g_vec) {
                        gv = g4[v];
                    } else {
                        int d = v * 4;
                        gv = make_float4(
                            static_cast<float>(gamma[d]),
                            static_cast<float>(gamma[d + 1]),
                            static_cast<float>(gamma[d + 2]),
                            static_cast<float>(gamma[d + 3]));
                    }
                    float4 outv;
                    outv.x = (val.x * inv) * gv.x;
                    outv.y = (val.y * inv) * gv.y;
                    outv.z = (val.z * inv) * gv.z;
                    outv.w = (val.w * inv) * gv.w;
                    y4[v] = outv;
                }
            }
            return;
        }
        if (sizeof(in0_type) == 2 && sizeof(out0_type) == 2 && (%(hidden_size)d %% 8 == 0)
                && ((((size_t)x | (size_t)y) & 15u) == 0)) {
            constexpr int kVecN = %(hidden_size)d / 8;
            constexpr int kVecPer = (kVecN + %(launch_threads)d - 1) / %(launch_threads)d;
            constexpr bool kVecCache = kVecPer <= 2;
            const uint4* x8 = reinterpret_cast<const uint4*>(x + row_offset);
            uint4* y8 = reinterpret_cast<uint4*>(y + row_offset);
            uint4 cache[kVecCache ? kVecPer : 1];
            float sum = 0.0f;
            if (kVecCache) {
                #pragma unroll
                for (int i = 0; i < kVecPer; ++i) {
                    int v = tid + i * %(launch_threads)d;
                    if (v < kVecN) {
                        uint4 raw = x8[v];
                        cache[i] = raw;
                        const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            float val = static_cast<float>(vp[k]);
                            sum += val * val;
                        }
                    }
                }
            } else {
                for (int v = tid; v < kVecN; v += %(launch_threads)d) {
                    uint4 raw = x8[v];
                    const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        float val = static_cast<float>(vp[k]);
                        sum += val * val;
                    }
                }
            }
            sum = warp_sum(sum);
            if (lane == 0) warp_buf[warp] = sum;
            __syncthreads();
            if (warp == 0) {
                float total = lane < %(launch_warps)d ? warp_buf[lane] : 0.0f;
                total = warp_sum(total);
                if (lane == 0)
                    inverse_rms = rsqrtf(total / %(hidden_size)d.0f + eps);
            }
            __syncthreads();
            float inv = inverse_rms;
            bool g_vec = (sizeof(in1_type) == 2) && ((((size_t)gamma) & 15u) == 0);
            const uint4* g8 = g_vec ? reinterpret_cast<const uint4*>(gamma) : nullptr;
            if (kVecCache) {
                #pragma unroll
                for (int i = 0; i < kVecPer; ++i) {
                    int v = tid + i * %(launch_threads)d;
                    if (v < kVecN) {
                        uint4 raw = cache[i];
                        const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                        uint4 outraw;
                        out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                        if (g_vec) {
                            uint4 graw = g8[v];
                            const in1_type* gp = reinterpret_cast<const in1_type*>(&graw);
                            #pragma unroll
                            for (int k = 0; k < 8; ++k) {
                                float normalized = static_cast<float>(vp[k]) * inv;
                                if (%(round_before_weight)d)
                                    normalized = static_cast<float>(out0_type(normalized));
                                op[k] = out0_type(normalized * static_cast<float>(gp[k]));
                            }
                        } else {
                            int d = v * 8;
                            #pragma unroll
                            for (int k = 0; k < 8; ++k) {
                                float normalized = static_cast<float>(vp[k]) * inv;
                                if (%(round_before_weight)d)
                                    normalized = static_cast<float>(out0_type(normalized));
                                op[k] = out0_type(normalized * static_cast<float>(gamma[d + k]));
                            }
                        }
                        y8[v] = outraw;
                    }
                }
            } else {
                for (int v = tid; v < kVecN; v += %(launch_threads)d) {
                    uint4 raw = x8[v];
                    const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                    uint4 outraw;
                    out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                    if (g_vec) {
                        uint4 graw = g8[v];
                        const in1_type* gp = reinterpret_cast<const in1_type*>(&graw);
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            float normalized = static_cast<float>(vp[k]) * inv;
                            if (%(round_before_weight)d)
                                normalized = static_cast<float>(out0_type(normalized));
                            op[k] = out0_type(normalized * static_cast<float>(gp[k]));
                        }
                    } else {
                        int d = v * 8;
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            float normalized = static_cast<float>(vp[k]) * inv;
                            if (%(round_before_weight)d)
                                normalized = static_cast<float>(out0_type(normalized));
                            op[k] = out0_type(normalized * static_cast<float>(gamma[d + k]));
                        }
                    }
                    y8[v] = outraw;
                }
            }
            return;
        }
        constexpr int kPer = (%(hidden_size)d + %(launch_threads)d - 1) / %(launch_threads)d;
        constexpr bool kCache = kPer <= 16;
        float cache[kCache ? kPer : 1];
        float sum = 0.0f;
        if (kCache) {
            #pragma unroll
            for (int i = 0; i < kPer; ++i) {
                int dim = tid + i * %(launch_threads)d;
                if (dim < %(hidden_size)d) {
                    float value = static_cast<float>(x[row_offset + dim]);
                    cache[i] = value;
                    sum += value * value;
                }
            }
        } else {
            for (int dim = tid; dim < %(hidden_size)d; dim += %(launch_threads)d) {
                float value = static_cast<float>(x[row_offset + dim]);
                sum += value * value;
            }
        }
        sum = warp_sum(sum);
        if (lane == 0) warp_buf[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float total = lane < %(launch_warps)d ? warp_buf[lane] : 0.0f;
            total = warp_sum(total);
            if (lane == 0)
                inverse_rms = rsqrtf(total / %(hidden_size)d.0f + eps);
        }
        __syncthreads();
        float inv = inverse_rms;
        if (kCache) {
            #pragma unroll
            for (int i = 0; i < kPer; ++i) {
                int dim = tid + i * %(launch_threads)d;
                if (dim < %(hidden_size)d) {
                    float normalized = cache[i] * inv;
                    if (%(round_before_weight)d)
                        normalized = static_cast<float>(out0_type(normalized));
                    y[row_offset + dim] = out0_type(normalized * static_cast<float>(gamma[dim]));
                }
            }
        } else {
            for (int dim = tid; dim < %(hidden_size)d; dim += %(launch_threads)d) {
                long long index = row_offset + dim;
                float normalized = static_cast<float>(x[index]) * inv;
                if (%(round_before_weight)d)
                    normalized = static_cast<float>(out0_type(normalized));
                y[index] = out0_type(normalized * static_cast<float>(gamma[dim]));
            }
        }
    }
    float eps = static_cast<float>(data["eps"]);
    long long rows = in0->num / %(hidden_size)d;
    rms_norm<<<(unsigned)rows, %(launch_threads)d>>>(in0_p, in1_p, out0_p, eps);
    CHECK(0 == cudaGetLastError());
    """


@lru_cache(maxsize=256)
def _format_rms_norm_source(template, hidden_size, threads, warps,
                            round_before_weight):
    # Cap block size at 512 threads so SM89 (1536 threads/SM) runs 3 blocks/SM
    # at 100% occupancy instead of 1 block/SM (66.7%) with 1024 threads.
    launch_threads = min(threads, 512)
    return template % {
        "hidden_size": hidden_size,
        "threads": threads,
        "warps": warps,
        "launch_threads": launch_threads,
        "launch_warps": launch_threads // 32,
        "round_before_weight": int(round_before_weight),
    }


def _rms_norm_source(hidden_size, threads, warps, epsilon_value=None, round_before_weight=True):
    """The CUDA source of the kernel below for one contract.

    Also what the native module call (`src/bindings/pyjt/py_module_call.cc`)
    builds the same operator from, so the two cannot drift apart.
    ``epsilon_value`` is accepted for caller compatibility; epsilon is passed
    at runtime via ``CodeOp::data["eps"]`` so different eps values share one
    compiled kernel.
    """
    return _format_rms_norm_source(
        _RMS_NORM_SOURCE, hidden_size, threads, warps, round_before_weight)


@native_rule("rms_norm_inference")
def _rms_norm_supported(x, gamma, epsilon=1e-6):
    return _rms_norm_contract(x, gamma, epsilon) is not None


@optional_kernel("nn.rms_norm.inference", ("cuda", "rocm_legacy", "corex_legacy"),
                 supports=_rms_norm_supported)
def _rms_norm_cuda(x, gamma, epsilon=1e-6):
    """Inference-only fused CUDA RMSNorm, or ``None`` when unsupported."""
    if not _autocast_enabled():
        result = jt.core._try_rms_norm_cuda(x, gamma, epsilon, _rms_norm_source)
        if result is not None:
            return result
    return _rms_norm_cuda_code(x, gamma, epsilon, round_before_weight=True)


_WARP_RMS_NORM_4096 = r"""
    __global__ static void warp_rms_norm_4096(@ARGS_DEF, float eps) {
        @PRECALC
        constexpr int width = 4096;
        const int row = blockIdx.x * blockDim.y + threadIdx.y;
        const int lane = threadIdx.x;
        if (row >= in0_shape0) return;
        // Accumulate the four interleaved streams independently, as the
        // float32 last-axis reduction does before rounding the BF16 output.
        float accumulators[4] = {};
        for (int column = lane * 4; column < width; column += 32 * 4) {
            #pragma unroll
            for (int item = 0; item < 4; ++item) {
                const float x = %(load_reduction)s;
                accumulators[item] = __fadd_rn(accumulators[item], __fmul_rn(x, x));
            }
        }
        float sum = accumulators[0];
        #pragma unroll
        for (int item = 1; item < 4; ++item)
            sum = __fadd_rn(sum, accumulators[item]);
        for (int offset = 16; offset > 0; offset /= 2)
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
        const float mean = __shfl_sync(0xffffffff, sum, 0) * (1.0f / width);
        const float scale = rsqrtf(__fadd_rn(mean, eps));
        for (int column = lane; column < width; column += 32) {
            const float x = %(load_output)s;
            const float weight = %(weight)s;
            @out0(row, column) = out0_type(__fmul_rn(__fmul_rn(x, scale), weight));
            %(store_residual)s
        }
    }
    float eps = static_cast<float>(data["eps"]);
    warp_rms_norm_4096<<<(in0_shape0 + 7) / 8, dim3(32, 8)>>>(@ARGS, eps);
    CHECK(0 == cudaGetLastError());
    """


@lru_cache(maxsize=32)
def _warp_rms_norm_source(residual=False, weight_offset=0.0):
    def load(column):
        x = f"float(@in0(row, {column}))"
        return f"__fadd_rn({x}, float(@in1(row, {column})))" if residual else x

    weight = "float(in2_p[column])" if residual else "in1_p[column]"
    if weight_offset:
        weight = f"__fadd_rn({weight}, static_cast<float>({weight_offset!r}))"
    return _WARP_RMS_NORM_4096 % {
        "load_reduction": load("column + item"),
        "load_output": load("column"),
        "weight": weight,
        "store_residual": "@out1(row, column) = out1_type(x);" if residual else "",
    }


def _is_wide_bfloat16_row(x):
    return x.ndim == 2 and x.shape[1] == 4096 and _jittor_dtype_name(x.dtype) == "bfloat16"


@optional_kernel("nn.rms_norm.single_round.inference", "cuda", supports=_rms_norm_supported)
def _rms_norm_single_round_cuda(x, gamma, epsilon=1e-6):
    if _is_wide_bfloat16_row(x) and _jittor_dtype_name(gamma.dtype) == "float32":
        source = _warp_rms_norm_source()
        return _stop_grad_outputs(
            jt.code(x.shape, x.dtype, [x, gamma], cuda_src=source,
                    data={"eps": float(epsilon)}, backend="cuda"))
    return _rms_norm_cuda_code(x, gamma, epsilon, round_before_weight=False)


def _rms_norm_cuda_code(x, gamma, epsilon, *, round_before_weight):
    contract = _rms_norm_contract(x, gamma, epsilon)
    if contract is None:
        return None
    hidden_size, threads, warps, epsilon_value = contract
    cuda_src = _rms_norm_source(hidden_size, threads, warps, epsilon_value, round_before_weight)
    return _stop_grad_outputs(
        jt.code(x.shape, x.dtype, [x, gamma], cuda_src=cuda_src,
                data={"eps": epsilon_value}))


def _add_rms_norm_supported(x, residual, gamma, epsilon=1e-6):
    return _rms_norm_contract(x, gamma, epsilon, residual=residual) is not None


@optional_kernel("nn.add_rms_norm.inference", ("cuda", "rocm_legacy", "corex_legacy"),
                 supports=_add_rms_norm_supported)
def _fused_add_rms_norm_cuda(x, residual, gamma, epsilon=1e-6):
    """Inference-only fused residual add and CUDA RMSNorm."""
    return _fused_add_rms_norm_cuda_code(x, residual, gamma, epsilon, round_before_weight=True)


@optional_kernel("nn.add_rms_norm.single_round.inference", "cuda",
                 supports=_add_rms_norm_supported)
def _fused_add_rms_norm_single_round_cuda(x, residual, gamma, epsilon=1e-6):
    return _fused_add_rms_norm_cuda_code(x, residual, gamma, epsilon, round_before_weight=False)


def _fp32_residual_supported(x, residual, gamma, epsilon=1e-6, weight_offset=0.0):
    return math.isfinite(weight_offset) and _add_rms_norm_supported(x, residual, gamma, epsilon)


@optional_kernel("nn.add_rms_norm.fp32_residual.inference", "cuda",
                 supports=_fp32_residual_supported)
def _fused_add_rms_norm_fp32_residual_cuda(x, residual, gamma, epsilon=1e-6, weight_offset=0.0):
    if _is_wide_bfloat16_row(x):
        source = _warp_rms_norm_source(residual=True, weight_offset=float(weight_offset))
        return _stop_grad_outputs(jt.code(
            [x.shape, x.shape], [x.dtype, x.dtype], [x, residual, gamma],
            cuda_src=source, data={"eps": float(epsilon)}, backend="cuda"))
    return _fused_add_rms_norm_cuda_code(
        x, residual, gamma, epsilon, round_before_weight=False,
        fp32_residual=True, weight_offset=weight_offset)


def _fused_add_rms_norm_cuda_code(x, residual, gamma, epsilon, *,
                                  round_before_weight, fp32_residual=False, weight_offset=0.0):
    contract = _rms_norm_contract(x, gamma, epsilon, residual=residual)
    if contract is None:
        return None
    hidden_size, threads, warps, epsilon_value = contract
    launch_threads = min(threads, 256 if (fp32_residual or hidden_size <= 1024) else 512)
    launch_warps = launch_threads // 32
    cuda_src = cached_source(r"""
    __device__ __forceinline__ float warp_sum(float value) {
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        return value;
    }
    __device__ __forceinline__ float4 jt_add_rms_inf_ldcs_f4(const float4* p) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        return __ldcs(p);
    #else
        return *p;
    #endif
    }
    __device__ __forceinline__ uint4 jt_add_rms_inf_ldcs_u4(const uint4* p) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        return __ldcs(p);
    #else
        return *p;
    #endif
    }
    __global__ __launch_bounds__(%(launch_threads)d) static void fused_add_rms_norm(
            const in0_type* __restrict__ x, const in1_type* __restrict__ residual,
            const in2_type* __restrict__ gamma, out0_type* __restrict__ y,
            out1_type* __restrict__ residual_out, float eps) {
        long long row = blockIdx.x;
        long long row_offset = row * %(hidden_size)dLL;
        int tid = threadIdx.x;
        int lane = tid & 31;
        int warp = tid >> 5;
        __shared__ float warp_buf[%(launch_warps)d];
        __shared__ float inverse_rms;
        if (sizeof(in0_type) == 4 && sizeof(in1_type) == 4
                && sizeof(out0_type) == 4 && sizeof(out1_type) == 4
                && (%(hidden_size)d %% 4 == 0)
                && ((((size_t)x | (size_t)residual | (size_t)y | (size_t)residual_out) & 15u) == 0)) {
            constexpr int kVecN = %(hidden_size)d / 4;
            constexpr int kVecPer = (kVecN + %(launch_threads)d - 1) / %(launch_threads)d;
            constexpr bool kVecCache = kVecPer <= 4;
            const float4* x4 = reinterpret_cast<const float4*>(x + row_offset);
            const float4* r4 = reinterpret_cast<const float4*>(residual + row_offset);
            float4* y4 = reinterpret_cast<float4*>(y + row_offset);
            float4* t4 = reinterpret_cast<float4*>(residual_out + row_offset);
            float4 cache[kVecCache ? kVecPer : 1];
            float sum = 0.0f;
            if (kVecCache) {
                #pragma unroll
                for (int i = 0; i < kVecPer; ++i) {
                    int v = tid + i * %(launch_threads)d;
                    if (v < kVecN) {
                        float4 xv = jt_add_rms_inf_ldcs_f4(x4 + v);
                        float4 rv = jt_add_rms_inf_ldcs_f4(r4 + v);
                        float4 tv = make_float4(xv.x + rv.x, xv.y + rv.y, xv.z + rv.z, xv.w + rv.w);
                        t4[v] = tv;
                        cache[i] = tv;
                        sum += tv.x * tv.x + tv.y * tv.y + tv.z * tv.z + tv.w * tv.w;
                    }
                }
            } else {
                for (int v = tid; v < kVecN; v += %(launch_threads)d) {
                    float4 xv = jt_add_rms_inf_ldcs_f4(x4 + v);
                    float4 rv = jt_add_rms_inf_ldcs_f4(r4 + v);
                    float4 tv = make_float4(xv.x + rv.x, xv.y + rv.y, xv.z + rv.z, xv.w + rv.w);
                    t4[v] = tv;
                    sum += tv.x * tv.x + tv.y * tv.y + tv.z * tv.z + tv.w * tv.w;
                }
            }
            sum = warp_sum(sum);
            if (lane == 0) warp_buf[warp] = sum;
            __syncthreads();
            if (warp == 0) {
                float total = lane < %(launch_warps)d ? warp_buf[lane] : 0.0f;
                total = warp_sum(total);
                if (lane == 0)
                    inverse_rms = rsqrtf(total / %(hidden_size)d.0f + eps);
            }
            __syncthreads();
            float inv = inverse_rms;
            bool g_vec = (sizeof(in2_type) == 4) && ((((size_t)gamma) & 15u) == 0);
            const float4* g4 = g_vec ? reinterpret_cast<const float4*>(gamma) : nullptr;
            if (kVecCache) {
                #pragma unroll
                for (int i = 0; i < kVecPer; ++i) {
                    int v = tid + i * %(launch_threads)d;
                    if (v < kVecN) {
                        float4 tv = cache[i];
                        float4 gv;
                        if (g_vec) {
                            gv = g4[v];
                        } else {
                            int d = v * 4;
                            gv = make_float4(
                                static_cast<float>(gamma[d]),
                                static_cast<float>(gamma[d + 1]),
                                static_cast<float>(gamma[d + 2]),
                                static_cast<float>(gamma[d + 3]));
                        }
                        if (%(has_weight_offset)d) {
                            gv.x += %(weight_offset)sf;
                            gv.y += %(weight_offset)sf;
                            gv.z += %(weight_offset)sf;
                            gv.w += %(weight_offset)sf;
                        }
                        float4 outv;
                        outv.x = (tv.x * inv) * gv.x;
                        outv.y = (tv.y * inv) * gv.y;
                        outv.z = (tv.z * inv) * gv.z;
                        outv.w = (tv.w * inv) * gv.w;
                        y4[v] = outv;
                    }
                }
            } else {
                for (int v = tid; v < kVecN; v += %(launch_threads)d) {
                    float4 tv = t4[v];
                    float4 gv;
                    if (g_vec) {
                        gv = g4[v];
                    } else {
                        int d = v * 4;
                        gv = make_float4(
                            static_cast<float>(gamma[d]),
                            static_cast<float>(gamma[d + 1]),
                            static_cast<float>(gamma[d + 2]),
                            static_cast<float>(gamma[d + 3]));
                    }
                    if (%(has_weight_offset)d) {
                        gv.x += %(weight_offset)sf;
                        gv.y += %(weight_offset)sf;
                        gv.z += %(weight_offset)sf;
                        gv.w += %(weight_offset)sf;
                    }
                    float4 outv;
                    outv.x = (tv.x * inv) * gv.x;
                    outv.y = (tv.y * inv) * gv.y;
                    outv.z = (tv.z * inv) * gv.z;
                    outv.w = (tv.w * inv) * gv.w;
                    y4[v] = outv;
                }
            }
            return;
        }
        if (!%(fp32_residual)d && sizeof(in0_type) == 2 && sizeof(in1_type) == 2
                && sizeof(out0_type) == 2 && sizeof(out1_type) == 2
                && (%(hidden_size)d %% 8 == 0)
                && ((((size_t)x | (size_t)residual | (size_t)y | (size_t)residual_out) & 15u) == 0)) {
            constexpr int kVecN = %(hidden_size)d / 8;
            constexpr int kVecPer = (kVecN + %(launch_threads)d - 1) / %(launch_threads)d;
            constexpr bool kVecCache = kVecPer <= 2;
            const uint4* x8 = reinterpret_cast<const uint4*>(x + row_offset);
            const uint4* r8 = reinterpret_cast<const uint4*>(residual + row_offset);
            uint4* y8 = reinterpret_cast<uint4*>(y + row_offset);
            uint4* t8 = reinterpret_cast<uint4*>(residual_out + row_offset);
            uint4 cache[kVecCache ? kVecPer : 1];
            float sum = 0.0f;
            if (kVecCache) {
                #pragma unroll
                for (int i = 0; i < kVecPer; ++i) {
                    int v = tid + i * %(launch_threads)d;
                    if (v < kVecN) {
                        uint4 xraw = jt_add_rms_inf_ldcs_u4(x8 + v);
                        uint4 rraw = jt_add_rms_inf_ldcs_u4(r8 + v);
                        uint4 traw;
                        const in0_type* xp = reinterpret_cast<const in0_type*>(&xraw);
                        const in1_type* rp = reinterpret_cast<const in1_type*>(&rraw);
                        out1_type* tp = reinterpret_cast<out1_type*>(&traw);
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            out1_type tv = out1_type(static_cast<float>(xp[k]) + static_cast<float>(rp[k]));
                            tp[k] = tv;
                            float val = static_cast<float>(tv);
                            sum += val * val;
                        }
                        t8[v] = traw;
                        cache[i] = traw;
                    }
                }
            } else {
                for (int v = tid; v < kVecN; v += %(launch_threads)d) {
                    uint4 xraw = jt_add_rms_inf_ldcs_u4(x8 + v);
                    uint4 rraw = jt_add_rms_inf_ldcs_u4(r8 + v);
                    uint4 traw;
                    const in0_type* xp = reinterpret_cast<const in0_type*>(&xraw);
                    const in1_type* rp = reinterpret_cast<const in1_type*>(&rraw);
                    out1_type* tp = reinterpret_cast<out1_type*>(&traw);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        out1_type tv = out1_type(static_cast<float>(xp[k]) + static_cast<float>(rp[k]));
                        tp[k] = tv;
                        float val = static_cast<float>(tv);
                        sum += val * val;
                    }
                    t8[v] = traw;
                }
            }
            sum = warp_sum(sum);
            if (lane == 0) warp_buf[warp] = sum;
            __syncthreads();
            if (warp == 0) {
                float total = lane < %(launch_warps)d ? warp_buf[lane] : 0.0f;
                total = warp_sum(total);
                if (lane == 0)
                    inverse_rms = rsqrtf(total / %(hidden_size)d.0f + eps);
            }
            __syncthreads();
            float inv = inverse_rms;
            bool g_vec = (sizeof(in2_type) == 2) && ((((size_t)gamma) & 15u) == 0);
            const uint4* g8 = g_vec ? reinterpret_cast<const uint4*>(gamma) : nullptr;
            if (kVecCache) {
                #pragma unroll
                for (int i = 0; i < kVecPer; ++i) {
                    int v = tid + i * %(launch_threads)d;
                    if (v < kVecN) {
                        uint4 traw = cache[i];
                        const out1_type* tp = reinterpret_cast<const out1_type*>(&traw);
                        uint4 outraw;
                        out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                        if (g_vec) {
                            uint4 graw = g8[v];
                            const in2_type* gp = reinterpret_cast<const in2_type*>(&graw);
                            #pragma unroll
                            for (int k = 0; k < 8; ++k) {
                                float normalized = static_cast<float>(tp[k]) * inv;
                                if (%(round_before_weight)d)
                                    normalized = static_cast<float>(out0_type(normalized));
                                float scale = static_cast<float>(gp[k]);
                                if (%(has_weight_offset)d) scale += %(weight_offset)sf;
                                op[k] = out0_type(normalized * scale);
                            }
                        } else {
                            int d = v * 8;
                            #pragma unroll
                            for (int k = 0; k < 8; ++k) {
                                float normalized = static_cast<float>(tp[k]) * inv;
                                if (%(round_before_weight)d)
                                    normalized = static_cast<float>(out0_type(normalized));
                                float scale = static_cast<float>(gamma[d + k]);
                                if (%(has_weight_offset)d) scale += %(weight_offset)sf;
                                op[k] = out0_type(normalized * scale);
                            }
                        }
                        y8[v] = outraw;
                    }
                }
            } else {
                for (int v = tid; v < kVecN; v += %(launch_threads)d) {
                    uint4 traw = t8[v];
                    const out1_type* tp = reinterpret_cast<const out1_type*>(&traw);
                    uint4 outraw;
                    out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                    int d = v * 8;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        float normalized = static_cast<float>(tp[k]) * inv;
                        if (%(round_before_weight)d)
                            normalized = static_cast<float>(out0_type(normalized));
                        float scale = static_cast<float>(gamma[d + k]);
                        if (%(has_weight_offset)d) scale += %(weight_offset)sf;
                        op[k] = out0_type(normalized * scale);
                    }
                    y8[v] = outraw;
                }
            }
            return;
        }
        constexpr int kPer = (%(hidden_size)d + %(launch_threads)d - 1) / %(launch_threads)d;
        constexpr bool kCache = kPer <= 16;
        float cache[kCache ? kPer : 1];
        float sum = 0.0f;
        if (kCache) {
            #pragma unroll
            for (int i = 0; i < kPer; ++i) {
                int dim = tid + i * %(launch_threads)d;
                if (dim < %(hidden_size)d) {
                    long long index = row_offset + dim;
                    float value = static_cast<float>(x[index]) + static_cast<float>(residual[index]);
                    if (!%(fp32_residual)d) value = static_cast<float>(out1_type(value));
                    residual_out[index] = out1_type(value);
                    cache[i] = value;
                    sum += value * value;
                }
            }
        } else {
            for (int dim = tid; dim < %(hidden_size)d; dim += %(launch_threads)d) {
                long long index = row_offset + dim;
                float value = static_cast<float>(x[index]) + static_cast<float>(residual[index]);
                if (!%(fp32_residual)d) value = static_cast<float>(out1_type(value));
                residual_out[index] = out1_type(value);
                sum += value * value;
            }
        }
        sum = warp_sum(sum);
        if (lane == 0) warp_buf[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float total = lane < %(launch_warps)d ? warp_buf[lane] : 0.0f;
            total = warp_sum(total);
            if (lane == 0)
                inverse_rms = rsqrtf(total / %(hidden_size)d.0f + eps);
        }
        __syncthreads();
        float inv = inverse_rms;
        if (kCache) {
            #pragma unroll
            for (int i = 0; i < kPer; ++i) {
                int dim = tid + i * %(launch_threads)d;
                if (dim < %(hidden_size)d) {
                    float normalized = cache[i] * inv;
                    if (%(round_before_weight)d)
                        normalized = static_cast<float>(out0_type(normalized));
                    float scale = static_cast<float>(gamma[dim]);
                    if (%(has_weight_offset)d) scale += %(weight_offset)sf;
                    y[row_offset + dim] = out0_type(normalized * scale);
                }
            }
        } else {
            for (int dim = tid; dim < %(hidden_size)d; dim += %(launch_threads)d) {
                long long index = row_offset + dim;
                float value = %(fp32_residual)d
                    ? (static_cast<float>(x[index]) + static_cast<float>(residual[index]))
                    : static_cast<float>(residual_out[index]);
                float normalized = value * inv;
                if (%(round_before_weight)d)
                    normalized = static_cast<float>(out0_type(normalized));
                float scale = static_cast<float>(gamma[dim]);
                if (%(has_weight_offset)d) scale += %(weight_offset)sf;
                y[index] = out0_type(normalized * scale);
            }
        }
    }
    float eps = static_cast<float>(data["eps"]);
    long long rows = in0->num / %(hidden_size)d;
    fused_add_rms_norm<<<(unsigned)rows, %(launch_threads)d>>>(
        in0_p, in1_p, in2_p, out0_p, out1_p, eps);
    CHECK(0 == cudaGetLastError());
    """, {
        "hidden_size": hidden_size,
        "launch_threads": launch_threads,
        "launch_warps": launch_warps,
        "round_before_weight": int(round_before_weight),
        "fp32_residual": int(fp32_residual),
        "has_weight_offset": int(bool(weight_offset)),
        "weight_offset": repr(float(weight_offset)),
    })
    return _stop_grad_outputs(jt.code(
        [x.shape, x.shape], [x.dtype, x.dtype], [x, residual, gamma],
        cuda_src=cuda_src,
        data={"eps": epsilon_value},
    ))


def _multihead_rms_norm_contract(x, gamma, scale=None, min_norm=1e-12):
    if not isinstance(x, jt.Var) or not isinstance(gamma, jt.Var):
        return None
    if _output_requires_grad(x, gamma):
        return None
    if _autocast_enabled():
        return None
    if _jittor_dtype_name(x.dtype) != "bfloat16" or _jittor_dtype_name(gamma.dtype) != "float32":
        return None

    try:
        x_shape = tuple(int(size) for size in x.shape)
        gamma_shape = tuple(int(size) for size in gamma.shape)
        scale_value = math.sqrt(float(x_shape[-1])) if scale is None else float(scale)
        min_norm_value = float(min_norm)
    except Exception:
        return None
    if len(x_shape) < 2:
        return None
    num_heads, head_dim = x_shape[-2:]
    if (
        num_heads <= 0
        or head_dim <= 0
        or head_dim > 4096
        or gamma_shape != (num_heads, head_dim)
        or any(size <= 0 for size in x_shape[:-2])
    ):
        return None
    if not math.isfinite(scale_value) or not math.isfinite(min_norm_value) or min_norm_value <= 0:
        return None
    return num_heads, head_dim, scale_value, min_norm_value


def _multihead_rms_norm_supported(x, gamma, scale=None, min_norm=1e-12):
    return _multihead_rms_norm_contract(x, gamma, scale, min_norm) is not None


@optional_kernel("nn.multihead_rms_norm.inference", ("cuda", "rocm_legacy", "corex_legacy"),
                 supports=_multihead_rms_norm_supported)
def multihead_rms_norm_cuda(x, gamma, scale=None, min_norm=1e-12):
    """Fused inference normalization for matching (..., heads, dim) / (heads, dim).

    The default scale is sqrt(head_dim); unsupported contracts return None.
    """
    num_heads, head_dim, scale_value, min_norm_value = _multihead_rms_norm_contract(
        x, gamma, scale, min_norm)

    if head_dim <= 256:
        rows_per_block = 8
        cuda_src = cached_source(r"""
        __device__ __forceinline__ float warp_sum(float value) {
            for (int offset = 16; offset > 0; offset >>= 1)
                value += __shfl_down_sync(0xffffffff, value, offset);
            return value;
        }
        __global__ static void multihead_rms_norm(
                const in0_type* x, const in1_type* gamma, out0_type* y,
                int rows) {
            int warp = threadIdx.x >> 5;
            int lane = threadIdx.x & 31;
            int row = blockIdx.x * %(rows_per_block)d + warp;
            if (row >= rows) return;

            float sum = 0.0f;
            for (int dim = lane; dim < %(head_dim)d; dim += 32) {
                float value = static_cast<float>(
                    x[row * %(head_dim)d + dim]);
                sum += value * value;
            }
            sum = warp_sum(sum);
            float denominator = sqrtf(
                __shfl_sync(0xffffffff, sum, 0));
            if (denominator < %(min_norm).9g)
                denominator = %(min_norm).9g;
            float factor = %(scale).9g / denominator;

            int head = row %% %(num_heads)d;
            for (int dim = lane; dim < %(head_dim)d; dim += 32) {
                int index = row * %(head_dim)d + dim;
                float value = static_cast<float>(x[index]);
                float weight = static_cast<float>(
                    gamma[head * %(head_dim)d + dim]);
                y[index] = out0_type(value * weight * factor);
            }
        }
        int rows = in0->num / %(head_dim)d;
        multihead_rms_norm<<<
            (rows + %(rows_per_block)d - 1) / %(rows_per_block)d, 256>>>(
                in0_p, in1_p, out0_p, rows);
        CHECK(0 == cudaGetLastError());
        """, {
            "head_dim": head_dim,
            "min_norm": min_norm_value,
            "num_heads": num_heads,
            "rows_per_block": rows_per_block,
            "scale": scale_value,
        })
        return _stop_grad_outputs(
            jt.code(x.shape, x.dtype, [x, gamma], cuda_src=cuda_src))

    threads = 32
    while threads < min(head_dim, 1024):
        threads *= 2
    warps = threads // 32
    cuda_src = r"""
    __device__ __forceinline__ float warp_sum(float value) {
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        return value;
    })
    __global__ static void multihead_rms_norm(
            const in0_type* x, const in1_type* gamma, out0_type* y) {
        int row = blockIdx.x;
        int tid = threadIdx.x;
        int lane = tid & 31;
        int warp = tid >> 5;
        __shared__ float warp_buf[%(warps)d];
        __shared__ float denominator;
        float sum = 0.0f;
        for (int dim = tid; dim < %(head_dim)d; dim += blockDim.x) {
            float value = static_cast<float>(x[row * %(head_dim)d + dim]);
            sum += value * value;
        }
        sum = warp_sum(sum);
        if (lane == 0) warp_buf[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float total = lane < %(warps)d ? warp_buf[lane] : 0.0f;
            total = warp_sum(total);
            if (lane == 0) {
                denominator = sqrtf(total);
                if (denominator < %(min_norm).9g) denominator = %(min_norm).9g;
            }
        }
        __syncthreads();
        int head = row %% %(num_heads)d;
        for (int dim = tid; dim < %(head_dim)d; dim += blockDim.x) {
            float value = static_cast<float>(x[row * %(head_dim)d + dim]);
            float weight = static_cast<float>(
                gamma[head * %(head_dim)d + dim]);
            y[row * %(head_dim)d + dim] = out0_type(
                value / denominator * weight * %(scale).9g);
        }
    }
    int rows = in0->num / %(head_dim)d;
    multihead_rms_norm<<<rows, %(threads)d>>>(in0_p, in1_p, out0_p);
    CHECK(0 == cudaGetLastError());
    """ % {
        "warps": warps,
        "head_dim": head_dim,
        "min_norm": min_norm_value,
        "num_heads": num_heads,
        "scale": scale_value,
        "threads": threads,
    }
    return _stop_grad_outputs(
        jt.code(x.shape, x.dtype, [x, gamma], cuda_src=cuda_src))


__all__ = ["multihead_rms_norm_cuda"]
