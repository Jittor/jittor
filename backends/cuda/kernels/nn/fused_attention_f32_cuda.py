"""Memory-efficient float32 attention as a CUDA ``nn.fused_attention`` kernel.

cuDNN's fused attention (``cudnn_attention_cuda.py``) is float16/bfloat16 only,
and float32 attention built from matmuls writes the whole ``[..., Lq, Lk]``
score matrix and its softmax to device memory: a 4096-token SD1.5 attention
call with its backward held 2 GiB and took 15 ms. PyTorch answers float32 with
its memory-efficient kernel, which computes the softmax a tile of keys at a
time and never stores the scores. This is that algorithm (FlashAttention-2):

forward
    A block owns 64 queries. It walks the keys a tile at a time, keeping for
    each query a running maximum, a running sum and the weighted values, all
    rescaled whenever the maximum moves. It writes the output and one
    log-sum-exp per query, which is all the backward needs.
backward
    A block owns a tile of keys. It walks the queries a tile at a time,
    rebuilds the probabilities from the log-sum-exp, accumulates the key and
    value gradients in registers, and adds its share of each query gradient
    into global memory.

Supports head dimensions up to 128, causal (top-left aligned, as the
composite path builds it) or not, with an optional boolean or additive float
``attn_mask`` broadcast to ``[batch, heads, queries, keys]``, as PyTorch
accepts it. A mask that needs a gradient, or dropout, declines.
"""

import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._core.flags import _output_requires_grad
from jittor._runtime.dispatch import register_kernel

_MAX_HEAD_DIM = 128
#: Below this many blocks a forward grid leaves SMs idle; see `_forward`.
_SM_BLOCKS = 128

_KERNELS = r"""
#include <cfloat>

namespace mea {

// The mask of one (batch, head), broadcast by zero strides: MASK is 0 for
// none, 1 for a boolean one (false drops the key) and 2 for an additive float
// bias, PyTorch's two forms of `attn_mask`.
struct Mask {
    const void* data;
    int heads;
    long long batch_stride, head_stride, row_stride, col_stride;
    __device__ __forceinline__ long long base(int bh) const {
        return (long long)(bh / heads) * batch_stride + (long long)(bh % heads) * head_stride;
    }
};

template <int MASK>
__device__ __forceinline__ float masked(float s, const Mask& mask, long long at) {
    if (MASK == 1) return static_cast<const bool*>(mask.data)[at] ? s : -INFINITY;
    if (MASK == 2) return s + static_cast<const float*>(mask.data)[at];
    return s;
}

} // namespace mea

// The forward, register-tiled. 256 threads as 16 row groups x 16 lanes; a
// thread holds RPT query rows x 4 keys of a score tile and RPT rows x D/16
// output columns. Q and K sit in shared memory transposed, [D][rows], so a
// thread reads its RPT queries and its 4 keys at one depth as one float4 each:
// two loads per 16 multiply-adds, where the kernel this replaced made eight
// scalar ones -- 16.2 ms of a Qwen3-0.6B training step's forward, now 9.6.
// The probabilities go through shared memory transposed as well, so the P.V
// loop reads a key's RPT probabilities as one float4.
namespace mea_fwd {

constexpr int THREADS = 256;

__device__ __forceinline__ float lanes16_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    return fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
}

__device__ __forceinline__ float lanes16_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    return v + __shfl_xor_sync(0xffffffff, v, 1);
}

// What the mask leaves of a ROWS x COLS tile, the same answer in every thread
// of the block: 0 nothing, 1 some of it, 2 all of it. A tile hidden whole is
// skipped, and a boolean mask is not read inside a tile it shows whole -- so
// an explicit causal mask, as Transformers builds one under compilation, costs
// what `is_causal` does: before, the kernel declined it, and every [L, L]
// score matrix was written out instead.
template <int MASK, bool CAUSAL, int ROWS, int COLS>
__device__ __forceinline__ int tile_visibility(const mea::Mask& mask, long long base, int r0,
                                               int c0, int lq, int lk) {
    bool any = false, all = true;
    for (int i = threadIdx.x; i < ROWS * COLS; i += THREADS) {
        const int row = r0 + i / COLS, col = c0 + i % COLS;
        if (row >= lq || col >= lk || (CAUSAL && col > row)) continue;
        const long long at = base + row * mask.row_stride + col * mask.col_stride;
        const bool shown = MASK == 1 ? static_cast<const bool*>(mask.data)[at]
                                     : static_cast<const float*>(mask.data)[at] != -INFINITY;
        any |= shown;
        all &= shown;
    }
    if (!__syncthreads_or(any)) return 0;
    return MASK == 1 && __syncthreads_and(all) ? 2 : 1;
}

template <int RPT>
__device__ __forceinline__ void load_rows(const float* p, float (&out)[RPT]) {
    if (RPT == 4) {
        const float4 v = *reinterpret_cast<const float4*>(p);
        out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
    } else {
        #pragma unroll
        for (int r = 0; r < RPT; r++) out[r] = p[r];
    }
}

template <int D, int RPT, bool CAUSAL, int MASK>
__global__ void __launch_bounds__(THREADS) forward(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, float* __restrict__ o, float* __restrict__ lse,
        int lq, int lk, float scale, mea::Mask mask) {
    constexpr int BQ = 16 * RPT, BK = 64;
    constexpr int QP = BQ + 4, KP = BK + 4, DPT = (D + 15) / 16;
    constexpr int KV = D * KP > BK * D ? D * KP : BK * D;
    extern __shared__ float smem[];
    float* sq = smem;               // Q^T, [D][QP]
    float* skv = sq + D * QP;       // K^T, [D][KP], then V, [BK][D]
    float* sp = skv + KV;           // P^T, [BK][QP]
    const int bh = blockIdx.y, q0 = blockIdx.x * BQ;
    const int ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
    const float* qb = q + (size_t)bh * lq * D;
    const float* kb = k + (size_t)bh * lk * D;
    const float* vb = v + (size_t)bh * lk * D;
    const long long mask_base = MASK ? mask.base(bh) : 0;
    for (int i = threadIdx.x; i < BQ * D; i += THREADS) {
        const int r = i / D, c = i - r * D;
        sq[c * QP + r] = q0 + r < lq ? qb[(size_t)(q0 + r) * D + c] * scale : 0.f;
    }

    float m[RPT], l[RPT], acc[RPT][DPT];
    #pragma unroll
    for (int r = 0; r < RPT; r++) {
        m[r] = -INFINITY; l[r] = 0.f;
        #pragma unroll
        for (int c = 0; c < DPT; c++) acc[r][c] = 0.f;
    }
    const int kend = CAUSAL ? min(lk, q0 + BQ) : lk;
    for (int k0 = 0; k0 < kend; k0 += BK) {
        const int shown = MASK ? tile_visibility<MASK, CAUSAL, BQ, BK>(
            mask, mask_base, q0, k0, lq, lk) : 2;
        if (!shown) continue;
        __syncthreads();
        for (int i = threadIdx.x; i < BK * D; i += THREADS) {
            const int r = i / D, c = i - r * D;
            skv[c * KP + r] = k0 + r < lk ? kb[(size_t)(k0 + r) * D + c] : 0.f;
        }
        __syncthreads();
        float s[RPT][4];
        #pragma unroll
        for (int r = 0; r < RPT; r++)
            #pragma unroll
            for (int c = 0; c < 4; c++) s[r][c] = 0.f;
        #pragma unroll 4
        for (int d = 0; d < D; d++) {
            float qv[RPT];
            load_rows<RPT>(sq + d * QP + ty * RPT, qv);
            const float4 kv = *reinterpret_cast<const float4*>(skv + d * KP + tx * 4);
            #pragma unroll
            for (int r = 0; r < RPT; r++) {
                s[r][0] += qv[r] * kv.x;
                s[r][1] += qv[r] * kv.y;
                s[r][2] += qv[r] * kv.z;
                s[r][3] += qv[r] * kv.w;
            }
        }
        #pragma unroll
        for (int r = 0; r < RPT; r++) {
            const int row = q0 + ty * RPT + r;
            float top = -INFINITY;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                const int col = k0 + tx * 4 + c;
                if (col >= lk || (CAUSAL && col > row)) s[r][c] = -INFINITY;
                else if (MASK && shown == 1 && row < lq)
                    s[r][c] = mea::masked<MASK>(s[r][c], mask, mask_base
                        + row * mask.row_stride + col * mask.col_stride);
                top = fmaxf(top, s[r][c]);
            }
            const float next = fmaxf(m[r], lanes16_max(top));
            const float keep = next == -INFINITY ? 1.f : __expf(m[r] - next);
            float sum = 0.f;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                s[r][c] = next == -INFINITY ? 0.f : __expf(s[r][c] - next);
                sum += s[r][c];
            }
            l[r] = l[r] * keep + lanes16_sum(sum);
            m[r] = next;
            #pragma unroll
            for (int c = 0; c < DPT; c++) acc[r][c] *= keep;
        }
        #pragma unroll
        for (int c = 0; c < 4; c++)
            #pragma unroll
            for (int r = 0; r < RPT; r++) sp[(tx * 4 + c) * QP + ty * RPT + r] = s[r][c];
        __syncthreads();
        for (int i = threadIdx.x; i < BK * D; i += THREADS) {
            const int r = i / D, c = i - r * D;
            skv[r * D + c] = k0 + r < lk ? vb[(size_t)(k0 + r) * D + c] : 0.f;
        }
        __syncthreads();
        #pragma unroll 2
        for (int j = 0; j < BK; j++) {
            float p[RPT];
            load_rows<RPT>(sp + j * QP + ty * RPT, p);
            #pragma unroll
            for (int c = 0; c < DPT; c++) {
                const int col = tx + c * 16;
                const float vv = col < D ? skv[j * D + col] : 0.f;
                #pragma unroll
                for (int r = 0; r < RPT; r++) acc[r][c] += p[r] * vv;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < RPT; r++) {
        const int row = q0 + ty * RPT + r;
        if (row >= lq) continue;
        const float inv = l[r] > 0.f ? 1.f / l[r] : 0.f;
        float* ob = o + ((size_t)bh * lq + row) * D;
        #pragma unroll
        for (int c = 0; c < DPT; c++) {
            const int col = tx + c * 16;
            if (col < D) ob[col] = acc[r][c] * inv;
        }
        if (tx == 0) lse[(size_t)bh * lq + row] = l[r] > 0.f ? m[r] + __logf(l[r]) : -INFINITY;
    }
}

} // namespace mea_fwd

// The backward, register-tiled. A block owns 32 keys and walks the queries 32
// at a time; 128 threads take a different share in each phase so that every
// shared-memory read feeds several multiply-adds:
//   S and dP   2 queries x 4 keys a thread; Q, dO as [D][34], K, V as [D][36]
//              -- a float2 and a float4 per operand and depth
//   dV and dK  4 keys x D/16 columns a thread; P, dS read as float4, dO and Q
//              down a column, whose stride of 34 floats keeps the lanes on
//              distinct banks
//   dQ         4 queries x D/16 columns, added into global memory, as the
//              other key blocks add theirs
// 38.8 ms of the same step's backward before, 25.6 after.
namespace mea_bwd {

constexpr int THREADS = 128;
constexpr int BQ = 32, BK = 32, QP = BQ + 2, KP = BK + 4, PP = BK + 4;

template <int MASK, bool CAUSAL>
__device__ __forceinline__ int tile_visibility(const mea::Mask& mask, long long base, int r0,
                                               int c0, int lq, int lk) {
    bool any = false, all = true;
    for (int i = threadIdx.x; i < BQ * BK; i += THREADS) {
        const int row = r0 + i / BK, col = c0 + i % BK;
        if (row >= lq || col >= lk || (CAUSAL && col > row)) continue;
        const long long at = base + row * mask.row_stride + col * mask.col_stride;
        const bool shown = MASK == 1 ? static_cast<const bool*>(mask.data)[at]
                                     : static_cast<const float*>(mask.data)[at] != -INFINITY;
        any |= shown;
        all &= shown;
    }
    if (!__syncthreads_or(any)) return 0;
    return MASK == 1 && __syncthreads_and(all) ? 2 : 1;
}

template <int D, bool CAUSAL, int MASK>
__global__ void __launch_bounds__(THREADS) backward(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ dout,
        const float* __restrict__ lse, const float* __restrict__ delta,
        float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dv,
        int lq, int lk, float scale, mea::Mask mask) {
    constexpr int DPT = (D + 15) / 16;
    extern __shared__ float smem[];
    float* sk = smem;               // K^T [D][KP]
    float* sv = sk + D * KP;        // V^T [D][KP]
    float* sq = sv + D * KP;        // Q^T [D][QP]
    float* sdo = sq + D * QP;       // dO^T [D][QP]
    float* sp = sdo + D * QP;       // P  [BQ][PP]
    float* sds = sp + BQ * PP;      // dS [BQ][PP]
    const int bh = blockIdx.y, k0 = blockIdx.x * BK;
    const int tid = threadIdx.x;
    const size_t qoff = (size_t)bh * lq * D, koff = (size_t)bh * lk * D;
    const long long mask_base = MASK ? mask.base(bh) : 0;
    for (int i = tid; i < BK * D; i += THREADS) {
        const int r = i / D, c = i - r * D;
        const bool in = k0 + r < lk;
        sk[c * KP + r] = in ? k[koff + (size_t)(k0 + r) * D + c] : 0.f;
        sv[c * KP + r] = in ? v[koff + (size_t)(k0 + r) * D + c] : 0.f;
    }
    // Phase A: rows ay*2 +{0,1}, keys ax*4 +{0..3}.
    const int ay = tid >> 3, ax = tid & 7;
    // Phases B and C: a group of 4 keys (B) or 4 queries (C), and a lane
    // whose columns are lane + 16c.
    const int group = tid >> 4, lane = tid & 15;
    float gk[4][DPT], gv[4][DPT];
    #pragma unroll
    for (int r = 0; r < 4; r++)
        #pragma unroll
        for (int c = 0; c < DPT; c++) gk[r][c] = gv[r][c] = 0.f;

    // Query tiles wholly above the diagonal see none of these keys.
    const int qstart = CAUSAL ? (k0 / BQ) * BQ : 0;
    for (int q0 = qstart; q0 < lq; q0 += BQ) {
        const int shown = MASK ? tile_visibility<MASK, CAUSAL>(
            mask, mask_base, q0, k0, lq, lk) : 2;
        if (!shown) continue;
        __syncthreads();
        for (int i = tid; i < BQ * D; i += THREADS) {
            const int r = i / D, c = i - r * D;
            const bool in = q0 + r < lq;
            sq[c * QP + r] = in ? q[qoff + (size_t)(q0 + r) * D + c] : 0.f;
            sdo[c * QP + r] = in ? dout[qoff + (size_t)(q0 + r) * D + c] : 0.f;
        }
        __syncthreads();
        float s[2][4], dp[2][4];
        #pragma unroll
        for (int r = 0; r < 2; r++)
            #pragma unroll
            for (int c = 0; c < 4; c++) s[r][c] = dp[r][c] = 0.f;
        #pragma unroll 4
        for (int d = 0; d < D; d++) {
            const float2 qv = *reinterpret_cast<const float2*>(sq + d * QP + ay * 2);
            const float2 ov = *reinterpret_cast<const float2*>(sdo + d * QP + ay * 2);
            const float4 kv = *reinterpret_cast<const float4*>(sk + d * KP + ax * 4);
            const float4 vv = *reinterpret_cast<const float4*>(sv + d * KP + ax * 4);
            const float qr[2] = {qv.x, qv.y}, orr[2] = {ov.x, ov.y};
            const float kc[4] = {kv.x, kv.y, kv.z, kv.w}, vc[4] = {vv.x, vv.y, vv.z, vv.w};
            #pragma unroll
            for (int r = 0; r < 2; r++)
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    s[r][c] += qr[r] * kc[c];
                    dp[r][c] += orr[r] * vc[c];
                }
        }
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            const int row = q0 + ay * 2 + r;
            const float row_lse = row < lq ? lse[(size_t)bh * lq + row] : -INFINITY;
            const float row_delta = row < lq ? delta[(size_t)bh * lq + row] : 0.f;
            float pr[4], dsr[4];
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                const int col = k0 + ax * 4 + c;
                const bool dead = row >= lq || col >= lk || (CAUSAL && col > row)
                                  || row_lse == -INFINITY;
                float score = s[r][c] * scale;
                if (MASK && shown == 1 && !dead)
                    score = mea::masked<MASK>(score, mask, mask_base
                        + row * mask.row_stride + col * mask.col_stride);
                pr[c] = dead || score == -INFINITY ? 0.f : __expf(score - row_lse);
                dsr[c] = pr[c] * (dp[r][c] - row_delta);
            }
            const int at = (ay * 2 + r) * PP + ax * 4;
            *reinterpret_cast<float4*>(sp + at) = make_float4(pr[0], pr[1], pr[2], pr[3]);
            *reinterpret_cast<float4*>(sds + at) = make_float4(dsr[0], dsr[1], dsr[2], dsr[3]);
        }
        __syncthreads();
        // Phase B: dV += P^T dO and dK += dS^T Q, keys group*4 +{0..3}.
        #pragma unroll 2
        for (int i = 0; i < BQ; i++) {
            const float4 p4 = *reinterpret_cast<const float4*>(sp + i * PP + group * 4);
            const float4 ds4 = *reinterpret_cast<const float4*>(sds + i * PP + group * 4);
            const float pk[4] = {p4.x, p4.y, p4.z, p4.w}, dk4[4] = {ds4.x, ds4.y, ds4.z, ds4.w};
            #pragma unroll
            for (int c = 0; c < DPT; c++) {
                const int col = lane + c * 16;
                const float ov = col < D ? sdo[col * QP + i] : 0.f;
                const float qv = col < D ? sq[col * QP + i] : 0.f;
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    gv[r][c] += pk[r] * ov;
                    gk[r][c] += dk4[r] * qv;
                }
            }
        }
        // Phase C: dQ += dS K, queries group*4 +{0..3}.
        float gq[4][DPT];
        #pragma unroll
        for (int r = 0; r < 4; r++)
            #pragma unroll
            for (int c = 0; c < DPT; c++) gq[r][c] = 0.f;
        #pragma unroll 2
        for (int j = 0; j < BK; j++) {
            float dsr[4];
            #pragma unroll
            for (int r = 0; r < 4; r++) dsr[r] = sds[(group * 4 + r) * PP + j];
            #pragma unroll
            for (int c = 0; c < DPT; c++) {
                const int col = lane + c * 16;
                const float kv = col < D ? sk[col * KP + j] : 0.f;
                #pragma unroll
                for (int r = 0; r < 4; r++) gq[r][c] += dsr[r] * kv;
            }
        }
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            const int row = q0 + group * 4 + r;
            if (row >= lq) continue;
            float* qrow = dq + qoff + (size_t)row * D;
            #pragma unroll
            for (int c = 0; c < DPT; c++) {
                const int col = lane + c * 16;
                if (col < D) atomicAdd(qrow + col, gq[r][c] * scale);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int key = k0 + group * 4 + r;
        if (key >= lk) continue;
        #pragma unroll
        for (int c = 0; c < DPT; c++) {
            const int col = lane + c * 16;
            if (col < D) {
                dk[koff + (size_t)key * D + col] = gk[r][c] * scale;
                dv[koff + (size_t)key * D + col] = gv[r][c];
            }
        }
    }
}

} // namespace mea_bwd
"""


def _launch(kernel, grid, smem, args, threads):
    return f"""
    auto fn = {kernel};
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});
    fn<<<dim3({grid}), {threads}, {smem}>>>({args});
    """


def _mask_layout(mask, shape):
    """(kind, contiguous mask, strides) for an `attn_mask`, or None to decline.

    kind is the kernel's MASK. The mask broadcasts to `shape` = (batch, heads,
    queries, keys) from the right, as PyTorch broadcasts it; a broadcast
    dimension gets stride 0.
    """
    if mask is None:
        return 0, None, (0, 0, 0, 0)
    dtype = _jittor_dtype_name(mask.dtype)
    if dtype == "bool":
        kind = 1
    elif dtype == "float32":
        # A mask that is learned would need its own gradient.
        if _output_requires_grad(mask):
            return None
        kind = 2
    else:
        return None
    dims = tuple(int(size) for size in mask.shape)
    if len(dims) > 4:
        return None
    dims = (1,) * (4 - len(dims)) + dims
    if any(size not in (1, full) for size, full in zip(dims, shape)):
        return None
    strides, step = [], 1
    for size in reversed(dims):
        strides.append(0 if size == 1 else step)
        step *= size
    return kind, mask.reshape(dims).stop_grad(), tuple(reversed(strides))


def _mask_args(layout, heads, index):
    kind, _, (sb, sh, sq, sk) = layout
    data = f"in{index}_p" if kind else "nullptr"
    return f"mea::Mask{{{data}, {heads}, {sb}LL, {sh}LL, {sq}LL, {sk}LL}}"


def _smem_register_forward(d, rpt):
    bq, bk = 16 * rpt, 64
    return (d * (bq + 4) + max(d * (bk + 4), bk * d) + bk * (bq + 4)) * 4


def _forward(query, key, value, scale, causal, layout=(0, None, (0, 0, 0, 0))):
    b, h, lq, d = (int(size) for size in query.shape)
    # Four query rows a thread; one where that leaves the grid smaller than
    # the device, which is what BERT at batch 1 (24 tiles of 64) did.
    rpt = 4 if -(-lq // 64) * b * h >= _SM_BLOCKS else 1
    bq = 16 * rpt
    smem = _smem_register_forward(d, rpt)
    kernel = (f"mea_fwd::forward<{d}, {rpt}, {str(bool(causal)).lower()}, {layout[0]}>")
    inputs = [query, key, value] + ([layout[1]] if layout[0] else [])
    return jt.code(
        [query.shape, (b, h, lq)], ["float32", "float32"], inputs,
        cuda_header=_KERNELS,
        cuda_src=_launch(kernel, f"(in0->shape[2] + {bq} - 1) / {bq}, in0->shape[0] * in0->shape[1]",
                         smem, f"in0_p, in1_p, in2_p, out0_p, out1_p, in0->shape[2], "
                               f"in1->shape[2], {float(scale)!r}f, {_mask_args(layout, h, 3)}",
                         threads="mea_fwd::THREADS"))


def _smem_register_backward(d):
    return (2 * d * (32 + 4) + 2 * d * (32 + 2) + 2 * 32 * (32 + 4)) * 4


def _backward(query, key, value, grad_out, lse, delta, scale, causal,
              layout=(0, None, (0, 0, 0, 0))):
    d = int(query.shape[3])
    h = int(query.shape[1])
    smem = _smem_register_backward(d)
    kernel = f"mea_bwd::backward<{d}, {str(bool(causal)).lower()}, {layout[0]}>"
    inputs = [query, key, value, grad_out, lse, delta] + ([layout[1]] if layout[0] else [])
    return jt.code(
        [query.shape, key.shape, value.shape], ["float32"] * 3,
        inputs,
        cuda_header=_KERNELS,
        cuda_src="cudaMemsetAsync(out0_p, 0, out0->size, 0);\n" + _launch(
            kernel, "(in1->shape[2] + 31) / 32, in0->shape[0] * in0->shape[1]",
            smem, f"in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, out0_p, out1_p, out2_p, "
                  f"in0->shape[2], in1->shape[2], {float(scale)!r}f, {_mask_args(layout, h, 6)}",
            threads="mea_bwd::THREADS"))


class _FusedAttentionF32(jt.Function):
    def execute(self, query, key, value, scale, causal, layout):
        self.scale, self.causal, self.layout = scale, causal, layout
        out, lse = _forward(query, key, value, scale, causal, layout)
        self.saved = (query, key, value, out, lse)
        return out

    def grad(self, grad_out):
        query, key, value, out, lse = self.saved
        grad_out = grad_out.float32()
        delta = (grad_out * out).sum(-1)
        grad_query, grad_key, grad_value = _backward(
            query, key, value, grad_out, lse, delta, self.scale, self.causal, self.layout)
        return grad_query, grad_key, grad_value, None, None, None


def _supports(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    return all(_jittor_dtype_name(t.dtype) == "float32" for t in (query, key, value))


def _fused_attention_f32(query, key, value, attn_mask=None, dropout_p=0.0,
                         is_causal=False, scale=None):
    """Run float32 attention tile by tile, or return None to decline."""
    if float(dropout_p or 0.0) != 0.0:
        return None
    if len(query.shape) != 4 or tuple(key.shape) != tuple(value.shape) \
            or len(key.shape) != 4 or tuple(query.shape[:2]) != tuple(key.shape[:2]) \
            or query.shape[3] != key.shape[3]:
        return None
    if not 0 < int(query.shape[3]) <= _MAX_HEAD_DIM:
        return None
    # A mask used to decline outright, and Transformers builds an explicit
    # causal one whenever `torch.compiler.is_compiling()` -- so a captured
    # Qwen3 step materialized every [L, L] score matrix instead.
    b, h, lq, _ = (int(size) for size in query.shape)
    lk = int(key.shape[2])
    training = _output_requires_grad(query, key, value)
    layout = _mask_layout(attn_mask, (b, h, lq, lk))
    if layout is None:
        return None
    scale = float(scale) if scale is not None else float(query.shape[3]) ** -0.5
    if training:
        return _FusedAttentionF32.apply(query, key, value, scale, bool(is_causal), layout)
    return _forward(query, key, value, scale, bool(is_causal), layout)[0]


register_kernel("nn.fused_attention", "cuda", _fused_attention_f32, supports=_supports)
