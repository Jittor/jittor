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
#: Score matrices smaller than this go to the matmul composite when there is
#: no backward; see `_fused_attention_f32`.
_SMALL_SCORES = 8 << 20

_KERNELS = r"""
#include <cfloat>

// Threads are laid out as 16 row groups x 8 lanes. The eight lanes that share
// a row group sit next to each other in a warp, so a row's reduction is three
// shuffles. Shared rows are padded by one float, so the eight lanes reading
// eight different rows at the same column hit eight different banks.
namespace mea {

constexpr int THREADS = 128;

__device__ __forceinline__ float group_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    return fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
}

__device__ __forceinline__ float group_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    return v + __shfl_xor_sync(0xffffffff, v, 1);
}

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

// What the mask leaves of a ROWS x COLS tile, the same answer in every thread
// of the block: 0 nothing, 1 some of it, 2 all of it. A tile hidden whole is
// skipped, and a boolean mask is not read inside a tile it shows whole -- so
// an explicit causal mask, as Transformers builds one under compilation, costs
// what `is_causal` does: before, the kernel declined it, and every [L, L]
// score matrix was written out instead.
template <int MASK, bool CAUSAL, int ROWS, int COLS>
__device__ __forceinline__ int tile_visibility(const Mask& mask, long long base, int r0,
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

template <int D, int ROWS>
__device__ __forceinline__ void load_tile(float* dst, const float* src, int first,
                                          int limit, float scale) {
    constexpr int DP = D + 1;
    for (int i = threadIdx.x; i < ROWS * D; i += THREADS) {
        int r = i / D, c = i - r * D;
        dst[r * DP + c] = first + r < limit ? src[(size_t)(first + r) * D + c] * scale : 0.f;
    }
}

template <int D, int BQ, int BK, bool CAUSAL, int MASK>
__global__ void __launch_bounds__(THREADS) forward(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, float* __restrict__ o, float* __restrict__ lse,
        int lq, int lk, float scale, Mask mask) {
    constexpr int DP = D + 1, PP = BK + 1;
    constexpr int RPT = BQ / 16, CPT = BK / 8, DPT = (D + 7) / 8;
    extern __shared__ float smem[];
    float* sq = smem;
    float* skv = sq + BQ * DP;
    float* sp = skv + BK * DP;
    const int bh = blockIdx.y, q0 = blockIdx.x * BQ;
    const int ty = threadIdx.x >> 3, tx = threadIdx.x & 7;
    const float* qb = q + (size_t)bh * lq * D;
    const float* kb = k + (size_t)bh * lk * D;
    const float* vb = v + (size_t)bh * lk * D;
    const long long mask_base = MASK ? mask.base(bh) : 0;
    load_tile<D, BQ>(sq, qb, q0, lq, scale);

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
        load_tile<D, BK>(skv, kb, k0, lk, 1.f);
        __syncthreads();
        float s[RPT][CPT];
        #pragma unroll
        for (int r = 0; r < RPT; r++)
            #pragma unroll
            for (int c = 0; c < CPT; c++) s[r][c] = 0.f;
        for (int d = 0; d < D; d++) {
            float qv[RPT], kv[CPT];
            #pragma unroll
            for (int r = 0; r < RPT; r++) qv[r] = sq[(ty * RPT + r) * DP + d];
            #pragma unroll
            for (int c = 0; c < CPT; c++) kv[c] = skv[(tx + c * 8) * DP + d];
            #pragma unroll
            for (int r = 0; r < RPT; r++)
                #pragma unroll
                for (int c = 0; c < CPT; c++) s[r][c] += qv[r] * kv[c];
        }
        #pragma unroll
        for (int r = 0; r < RPT; r++) {
            const int row = q0 + ty * RPT + r;
            float top = -INFINITY;
            #pragma unroll
            for (int c = 0; c < CPT; c++) {
                const int col = k0 + tx + c * 8;
                if (col >= lk || (CAUSAL && col > row)) s[r][c] = -INFINITY;
                else if (MASK && shown == 1 && row < lq)
                    s[r][c] = masked<MASK>(s[r][c], mask, mask_base
                        + row * mask.row_stride + col * mask.col_stride);
                top = fmaxf(top, s[r][c]);
            }
            const float next = fmaxf(m[r], group_max(top));
            const float keep = next == -INFINITY ? 1.f : __expf(m[r] - next);
            float sum = 0.f;
            #pragma unroll
            for (int c = 0; c < CPT; c++) {
                s[r][c] = next == -INFINITY ? 0.f : __expf(s[r][c] - next);
                sum += s[r][c];
            }
            l[r] = l[r] * keep + group_sum(sum);
            m[r] = next;
            #pragma unroll
            for (int c = 0; c < DPT; c++) acc[r][c] *= keep;
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < RPT; r++)
            #pragma unroll
            for (int c = 0; c < CPT; c++) sp[(ty * RPT + r) * PP + tx + c * 8] = s[r][c];
        load_tile<D, BK>(skv, vb, k0, lk, 1.f);
        __syncthreads();
        for (int j = 0; j < BK; j++) {
            float vv[DPT];
            #pragma unroll
            for (int c = 0; c < DPT; c++) {
                const int col = tx + c * 8;
                vv[c] = col < D ? skv[j * DP + col] : 0.f;
            }
            #pragma unroll
            for (int r = 0; r < RPT; r++) {
                const float p = sp[(ty * RPT + r) * PP + j];
                #pragma unroll
                for (int c = 0; c < DPT; c++) acc[r][c] += p * vv[c];
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
            const int col = tx + c * 8;
            if (col < D) ob[col] = acc[r][c] * inv;
        }
        if (tx == 0) lse[(size_t)bh * lq + row] = l[r] > 0.f ? m[r] + __logf(l[r]) : -INFINITY;
    }
}

template <int D, int BQ, int BK, bool CAUSAL, int MASK>
__global__ void __launch_bounds__(THREADS) backward(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ dout,
        const float* __restrict__ lse, const float* __restrict__ delta,
        float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dv,
        int lq, int lk, float scale, Mask mask) {
    constexpr int DP = D + 1, PP = BK + 1;
    constexpr int RPT = BQ / 16, CPT = BK / 8, KPT = BK / 16, DPT = (D + 7) / 8;
    extern __shared__ float smem[];
    float* sk = smem;
    float* sv = sk + BK * DP;
    float* sq = sv + BK * DP;
    float* sdo = sq + BQ * DP;
    float* sp = sdo + BQ * DP;
    float* sds = sp + BQ * PP;
    const int bh = blockIdx.y, k0 = blockIdx.x * BK;
    const int ty = threadIdx.x >> 3, tx = threadIdx.x & 7;
    const size_t qoff = (size_t)bh * lq * D, koff = (size_t)bh * lk * D;
    const long long mask_base = MASK ? mask.base(bh) : 0;
    load_tile<D, BK>(sk, k + koff, k0, lk, 1.f);
    load_tile<D, BK>(sv, v + koff, k0, lk, 1.f);

    float gk[KPT][DPT], gv[KPT][DPT];
    #pragma unroll
    for (int r = 0; r < KPT; r++)
        #pragma unroll
        for (int c = 0; c < DPT; c++) gk[r][c] = gv[r][c] = 0.f;

    // Query tiles wholly above the diagonal see none of these keys.
    const int qstart = CAUSAL ? (k0 / BQ) * BQ : 0;
    for (int q0 = qstart; q0 < lq; q0 += BQ) {
        const int shown = MASK ? tile_visibility<MASK, CAUSAL, BQ, BK>(
            mask, mask_base, q0, k0, lq, lk) : 2;
        if (!shown) continue;
        __syncthreads();
        load_tile<D, BQ>(sq, q + qoff, q0, lq, 1.f);
        load_tile<D, BQ>(sdo, dout + qoff, q0, lq, 1.f);
        __syncthreads();
        float s[RPT][CPT], dp[RPT][CPT];
        #pragma unroll
        for (int r = 0; r < RPT; r++)
            #pragma unroll
            for (int c = 0; c < CPT; c++) s[r][c] = dp[r][c] = 0.f;
        for (int d = 0; d < D; d++) {
            float qv[RPT], ov[RPT], kv[CPT], vv[CPT];
            #pragma unroll
            for (int r = 0; r < RPT; r++) {
                qv[r] = sq[(ty * RPT + r) * DP + d];
                ov[r] = sdo[(ty * RPT + r) * DP + d];
            }
            #pragma unroll
            for (int c = 0; c < CPT; c++) {
                kv[c] = sk[(tx + c * 8) * DP + d];
                vv[c] = sv[(tx + c * 8) * DP + d];
            }
            #pragma unroll
            for (int r = 0; r < RPT; r++)
                #pragma unroll
                for (int c = 0; c < CPT; c++) {
                    s[r][c] += qv[r] * kv[c];
                    dp[r][c] += ov[r] * vv[c];
                }
        }
        #pragma unroll
        for (int r = 0; r < RPT; r++) {
            const int row = q0 + ty * RPT + r;
            const float row_lse = row < lq ? lse[(size_t)bh * lq + row] : -INFINITY;
            const float row_delta = row < lq ? delta[(size_t)bh * lq + row] : 0.f;
            #pragma unroll
            for (int c = 0; c < CPT; c++) {
                const int col = k0 + tx + c * 8;
                const bool dead = row >= lq || col >= lk || (CAUSAL && col > row)
                                  || row_lse == -INFINITY;
                float score = s[r][c] * scale;
                if (MASK && shown == 1 && !dead)
                    score = masked<MASK>(score, mask, mask_base
                        + row * mask.row_stride + col * mask.col_stride);
                const float p = dead || score == -INFINITY ? 0.f : __expf(score - row_lse);
                sp[(ty * RPT + r) * PP + tx + c * 8] = p;
                sds[(ty * RPT + r) * PP + tx + c * 8] = p * (dp[r][c] - row_delta);
            }
        }
        __syncthreads();
        // dV += P^T dO and dK += dS^T Q, for this thread's keys.
        for (int i = 0; i < BQ; i++) {
            float ov[DPT], qv[DPT];
            #pragma unroll
            for (int c = 0; c < DPT; c++) {
                const int col = tx + c * 8;
                ov[c] = col < D ? sdo[i * DP + col] : 0.f;
                qv[c] = col < D ? sq[i * DP + col] : 0.f;
            }
            #pragma unroll
            for (int r = 0; r < KPT; r++) {
                const float p = sp[i * PP + ty * KPT + r];
                const float ds = sds[i * PP + ty * KPT + r];
                #pragma unroll
                for (int c = 0; c < DPT; c++) {
                    gv[r][c] += p * ov[c];
                    gk[r][c] += ds * qv[c];
                }
            }
        }
        // dQ += dS K, added into global memory: other key tiles add theirs.
        #pragma unroll
        for (int r = 0; r < RPT; r++) {
            const int row = q0 + ty * RPT + r;
            if (row >= lq) continue;
            float gq[DPT];
            #pragma unroll
            for (int c = 0; c < DPT; c++) gq[c] = 0.f;
            for (int j = 0; j < BK; j++) {
                const float ds = sds[(ty * RPT + r) * PP + j];
                #pragma unroll
                for (int c = 0; c < DPT; c++) {
                    const int col = tx + c * 8;
                    if (col < D) gq[c] += ds * sk[j * DP + col];
                }
            }
            float* qrow = dq + qoff + (size_t)row * D;
            #pragma unroll
            for (int c = 0; c < DPT; c++) {
                const int col = tx + c * 8;
                if (col < D) atomicAdd(qrow + col, gq[c] * scale);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < KPT; r++) {
        const int key = k0 + ty * KPT + r;
        if (key >= lk) continue;
        #pragma unroll
        for (int c = 0; c < DPT; c++) {
            const int col = tx + c * 8;
            if (col < D) {
                dk[koff + (size_t)key * D + col] = gk[r][c] * scale;
                dv[koff + (size_t)key * D + col] = gv[r][c];
            }
        }
    }
}

} // namespace mea
"""


def _tiles(head_dim):
    """(forward BQ, forward BK, backward BQ, backward BK) for a head dimension.

    Chosen so every tile set fits the 99 KB of shared memory a block may have.
    """
    return (64, 64, 32, 64) if head_dim <= 64 else (64, 32, 32, 32)


def _smem_forward(d, bq, bk):
    return ((bq + bk) * (d + 1) + bq * (bk + 1)) * 4


def _smem_backward(d, bq, bk):
    return ((2 * bk + 2 * bq) * (d + 1) + 2 * bq * (bk + 1)) * 4


def _launch(kernel, grid, smem, args):
    return f"""
    auto fn = {kernel};
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});
    fn<<<dim3({grid}), mea::THREADS, {smem}>>>({args});
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


def _forward(query, key, value, scale, causal, layout=(0, None, (0, 0, 0, 0))):
    b, h, lq, d = (int(size) for size in query.shape)
    bq, bk, _, _ = _tiles(d)
    # A grid smaller than the device takes narrower query tiles: BERT at
    # batch 1 (12 heads, 128 tokens) is 24 blocks of 64 queries, which ran at
    # half the speed of the matmul composite on a 128-SM card.
    if -(-lq // bq) * b * h < _SM_BLOCKS:
        bq = 16
    smem = _smem_forward(d, bq, bk)
    kernel = f"mea::forward<{d}, {bq}, {bk}, {str(bool(causal)).lower()}, {layout[0]}>"
    inputs = [query, key, value] + ([layout[1]] if layout[0] else [])
    return jt.code(
        [query.shape, (b, h, lq)], ["float32", "float32"], inputs,
        cuda_header=_KERNELS,
        cuda_src=_launch(kernel, f"(in0->shape[2] + {bq} - 1) / {bq}, in0->shape[0] * in0->shape[1]",
                         smem, f"in0_p, in1_p, in2_p, out0_p, out1_p, in0->shape[2], "
                               f"in1->shape[2], {float(scale)!r}f, {_mask_args(layout, h, 3)}"))


def _backward(query, key, value, grad_out, lse, delta, scale, causal,
              layout=(0, None, (0, 0, 0, 0))):
    d = int(query.shape[3])
    h = int(query.shape[1])
    _, _, bq, bk = _tiles(d)
    smem = _smem_backward(d, bq, bk)
    kernel = f"mea::backward<{d}, {bq}, {bk}, {str(bool(causal)).lower()}, {layout[0]}>"
    inputs = [query, key, value, grad_out, lse, delta] + ([layout[1]] if layout[0] else [])
    return jt.code(
        [query.shape, key.shape, value.shape], ["float32"] * 3,
        inputs,
        cuda_header=_KERNELS,
        cuda_src="cudaMemsetAsync(out0_p, 0, out0->size, 0);\n" + _launch(
            kernel, f"(in1->shape[2] + {bk} - 1) / {bk}, in0->shape[0] * in0->shape[1]",
            smem, f"in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, out0_p, out1_p, out2_p, "
                  f"in0->shape[2], in1->shape[2], {float(scale)!r}f, {_mask_args(layout, h, 6)}"))


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
    # Without a backward the kernel saves only the score matrix's traffic, and
    # a small one is cheaper through the matmuls: BERT's 12 x 128 x 128 per
    # layer took 24 us here against 11 us composed.
    if not training and b * h * lq * lk * 4 < _SMALL_SCORES:
        return None
    layout = _mask_layout(attn_mask, (b, h, lq, lk))
    if layout is None:
        return None
    scale = float(scale) if scale is not None else float(query.shape[3]) ** -0.5
    if training:
        return _FusedAttentionF32.apply(query, key, value, scale, bool(is_causal), layout)
    return _forward(query, key, value, scale, bool(is_causal), layout)[0]


register_kernel("nn.fused_attention", "cuda", _fused_attention_f32, supports=_supports)
