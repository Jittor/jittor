"""cuBLASLt for `nn.Linear`: the bias folded into the GEMM, and the algorithm
chosen by measurement rather than by the first heuristic.

Two separate things, both measured on this machine:

**The bias.** The portable path is a matmul and then a broadcast add, so every
linear layer is two kernels and two operators. Folding the add into the GEMM's
epilogue removes one of each. For a decode step of an 8-layer transformer that
is 32 of the step's operators, and dropping the add entirely (wrong answers,
just to see the ceiling) measured 7.3% of the step.

**The algorithm.** cuBLAS picks by heuristic and its first answer is not always
good. `fc2` of the same model -- [2048,2048] x [2048,512]^T -- measured 509 us
through `cublasGemmEx`, which is exactly what PyTorch's `x @ W.t()` measures
too; but PyTorch's `Linear` goes through cuBLASLt and measures 424.6. Timing
cuBLASLt's own heuristic candidates one by one: the first is 509.4 us and the
sixth is 422.3. So the difference is which kernel gets picked, not arithmetic.
This picks by timing them, once per problem shape, and remembers the winner.

Determinism is kept: the algorithm is chosen once and then fixed, and this
cuBLASLt exposes only deterministic reduction schemes (NONE, INPLACE,
COMPUTE_TYPE, OUTPUT_TYPE -- there is no atomic one to pick by accident), so
the answer does not move from run to run. A split-k candidate can still sum in
a different order than the non-split kernel, which is also true of whatever
cuBLAS's own heuristic picks.

**float16 is the same op, not a second one.** The compute dtype is not a
parameter of this module: it is `dtype_infer(x, weight)`'s answer, the same
inference `CublasMatmulOp` uses, so an autocast scope falls out of it rather
than needing a branch here. That matters beyond style -- deciding it any other
way is how `nn.Linear` came to return float32 out of a float16 graph, one
linear layer at a time. The operands are cast to that dtype exactly as the
portable path casts them, and the accumulate stays float32, which is what
`cublas_gemm_mode` picks for float16.

The one attribute that must *not* follow the operands is the scale type: it
belongs to the compute type, and cuBLASLt answers every fp16 query with
`CUBLAS_STATUS_INVALID_VALUE` if it is handed `CUDA_R_16F` there. That failure
is silent -- the fallback below is correct, so the only symptom is speed -- and
it cost a 2.4x regression before it was traced.

Kept out on purpose:

  - **Anything that needs a gradient.** This op has no backward source, so a
    training step goes the portable way. Silently losing a gradient would be
    the worst possible outcome of a speedup.
  - Anything whose compute dtype is not float16 or float32 (bfloat16, float64),
    or not dense, or that leaves a non-contiguous operand; those fall back to
    the portable path.
"""

import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._runtime.core_api import _output_requires_grad


#: Heuristic candidates to time. Beyond this the returns are gone and the
#: one-off cost is not.
_CANDIDATES = 8

#: Per-problem workspace. cuBLASLt reports what each candidate wants; the
#: fast ones for these shapes asked for 8 MB.
_WORKSPACE = 32 << 20

#: `dsize_` as `src/type/nano_string.h` defines it: 2**code is the byte width.
_DSIZE = {"float16": 1, "bfloat16": 1, "float32": 2, "float64": 3}

_HEADER = r"""
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

static cublasLtHandle_t jt_lt_handle() {
    static cublasLtHandle_t handle = nullptr;
    if (!handle) cublasLtCreate(&handle);
    return handle;
}

static cublasHandle_t jt_blas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);
    return handle;
}

static void* jt_lt_workspace(size_t bytes) {
    static void* ws = nullptr;
    static size_t have = 0;
    if (have < bytes) {
        if (ws) cudaFree(ws);
        if (cudaMalloc(&ws, bytes) != cudaSuccess) { ws = nullptr; have = 0; }
        else have = bytes;
    }
    return ws;
}

// Chosen once per problem shape. Each shape compiles its own kernel, so a
// function-local static here IS per shape.
//
// Templated on the element type: the epilogue bias must be the same type as
// the output matrix, so the float16 route has a half version of this.
template <typename T>
__global__ static void jt_lt_add_bias(T* out, const T* bias,
                                      int total, int cout) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = out[i] + bias[i % cout];
}

struct JtLtChoice {
    bool ready = false;
    bool usable = false;
    cublasLtMatmulAlgo_t algo;
};
"""


def compute_dtype_name(x, weight):
    """`dtype_infer(x, weight)` -- jittor's own answer, from Python.

    Mirrors `float_dtype` in `src/type/nano_string.h`, which is the rule
    `dtype_infer` applies to two non-scalar floating operands: an explicit
    `prefer32` wins, then `prefer16`, and only with neither set does the
    operands' own width decide. Reimplemented rather than reached for because
    the binding exposes `binary_dtype_infer` but not `dtype_infer`.

    Both operands here are always tensors, never scalars, so the `has_scalar`
    branch of `float_dtype` is not reachable and is not reproduced.
    """
    amp = jt.flags.amp_reg
    if amp & jt.amp_flags.prefer32:
        return "float32"
    if amp & jt.amp_flags.prefer16:
        return _bfloat16_or(x, weight, "float16")
    dsize = max(_DSIZE[_jittor_dtype_name(x.dtype)],
                _DSIZE[_jittor_dtype_name(weight.dtype)])
    if dsize == 3:
        return "float64"
    if dsize == 2:
        return "float32"
    return _bfloat16_or(x, weight, "float16")


def _bfloat16_or(x, weight, fallback):
    if ("bfloat16" in (_jittor_dtype_name(x.dtype), _jittor_dtype_name(weight.dtype))):
        return "bfloat16"
    return fallback


def _supports(x, weight, bias, *args, **kwargs):
    # Only cuda_src is provided, so this op is CUDA-only; under use_cuda=0 the
    # portable path is the only correct one.
    if not jt.flags.use_cuda:
        return False
    # Forward only: this op has no backward source, so anything that needs a
    # gradient has to go the portable way. Silently losing the gradient would
    # be the worst possible outcome of a speedup.
    if _output_requires_grad(x, weight, bias):
        return False
    if not isinstance(x, jt.Var) or not isinstance(weight, jt.Var):
        return False
    if not isinstance(bias, jt.Var):
        return False
    for v in (x, weight, bias):
        if _jittor_dtype_name(v.dtype) not in ("float16", "float32"):
            return False
        if not v._storage_is_contiguous():
            return False
    # bfloat16 and float64 would need their own descriptors and their own
    # accumulate rule; the portable path already has both.
    if compute_dtype_name(x, weight) not in ("float16", "float32"):
        return False
    if len(weight.shape) != 2 or len(bias.shape) != 1:
        return False
    if len(x.shape) < 2 or int(x.shape[-1]) != int(weight.shape[1]):
        return False
    if int(bias.shape[0]) != int(weight.shape[0]):
        return False
    rows = 1
    for d in x.shape[:-1]:
        rows *= int(d)
    # A tiny problem is all overhead: the autotune costs more than it can ever
    # return, and cuBLAS's heuristic is fine at that size.
    return rows * int(weight.shape[0]) * int(weight.shape[1]) >= (1 << 18)


def _source(rows, cin, cout, dtype):
    # The one place the element type enters the kernel. The accumulate is
    # float32 for both -- `cublas_gemm_mode`'s choice for float16 -- so alpha
    # and beta stay floats either way.
    ct = "CUDA_R_16F" if dtype == "float16" else "CUDA_R_32F"
    kt = "half" if dtype == "float16" else "float"
    return f"""
    const int rows = {rows}, cin = {cin}, cout = {cout};
    const float alpha = 1.0f, beta = 0.0f;
    cublasLtHandle_t lt = jt_lt_handle();

    cublasLtMatmulDesc_t op = nullptr;
    // The scale type follows the *compute* type and the float alpha/beta, not
    // the operand type. Handing it the operand type is rejected outright --
    // cuBLASLt answers `CUBLAS_STATUS_INVALID_VALUE` to
    // `cublasLtMatmulAlgoGetHeuristic` for every algorithm, and the only trace
    // of it is the fallback running underneath (which is how the whole fused
    // route came to be 2.4x slower than not using it at all while still
    // producing correct numbers).
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
    void* biasp = (void*)in2_p;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasp, sizeof(biasp));

    // Column-major, which is what cuBLAS speaks: the row-major product
    // A[rows,cin] * B[cout,cin]^T is computed as B^T * A with the operands
    // swapped, so `la` describes the weight and `lb` the activations.
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtMatrixLayoutCreate(&la, {ct}, cin, cout, cin);
    cublasLtMatrixLayoutCreate(&lb, {ct}, cin, rows, cin);
    cublasLtMatrixLayoutCreate(&lc, {ct}, cout, rows, cout);

    void* ws = jt_lt_workspace({_WORKSPACE});
    size_t wsize = ws ? (size_t){_WORKSPACE} : 0;

    static JtLtChoice choice;
    if (!choice.ready) {{
        choice.ready = true;
        cublasLtMatmulPreference_t pref = nullptr;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsize, sizeof(wsize));
        cublasLtMatmulHeuristicResult_t cand[{_CANDIDATES}];
        int found = 0;
        cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref,
                                       {_CANDIDATES}, cand, &found);
        cudaEvent_t beg, end;
        cudaEventCreate(&beg); cudaEventCreate(&end);
        float best = 1e30f;
        for (int c = 0; c < found; c++) {{
            auto once = [&]() {{
                // Explicit stream: jittor runs on `cudaStreamPerThread`, which
                // does not synchronise with the legacy default stream.
                return cublasLtMatmul(lt, op, &alpha, in1_p, la, in0_p, lb, &beta,
                                      out0_p, lc, out0_p, lc, &cand[c].algo,
                                      ws, wsize, cudaStreamPerThread);
            }};
            if (once() != CUBLAS_STATUS_SUCCESS) continue;
            cudaDeviceSynchronize();
            cudaEventRecord(beg, cudaStreamPerThread);
            for (int r = 0; r < 3; r++) once();
            cudaEventRecord(end, cudaStreamPerThread);
            if (cudaEventSynchronize(end) != cudaSuccess) continue;
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, beg, end);
            if (ms > 0.0f && ms < best) {{
                best = ms;
                choice.algo = cand[c].algo;
                choice.usable = true;
            }}
        }}
        cudaEventDestroy(beg); cudaEventDestroy(end);
        cublasLtMatmulPreferenceDestroy(pref);
    }}

    if (choice.usable) {{
        cublasLtMatmul(lt, op, &alpha, in1_p, la, in0_p, lb, &beta,
                       out0_p, lc, out0_p, lc, &choice.algo, ws, wsize,
                       cudaStreamPerThread);
    }} else {{
        // Nothing cuBLASLt offered was usable. The portable GEMM plus a bias
        // add is still correct, so do that rather than answer with garbage.
        // The handle is hoisted: creating and destroying one per call here
        // turned a 17 s decode into a 48 s one, so a route that has fallen
        // back must at least not pay for that on every layer.
        cublasHandle_t h = jt_blas_handle();
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, cout, rows, cin, &alpha,
                     in1_p, {ct}, cin, in0_p, {ct}, cin, &beta,
                     out0_p, {ct}, cout,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        int total = rows * cout;
        jt_lt_add_bias<{kt}><<<(total + 255) / 256, 256>>>(out0_p, in2_p, total, cout);
    }}

    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
    """


def lt_linear_cuda(x, weight, bias):
    """`x @ weight.T + bias` in one kernel, or None if this cannot serve it.

    No reshape on either side. A dense rank-3 activation is already the
    [rows, cin] matrix the GEMM wants, and the output shape is declared
    directly -- flattening and unflattening instead added two operators per
    layer, which on a 128-token prefill cost more than the kernel saved
    (2.90 -> 3.03 ms).
    """
    if not _supports(x, weight, bias):
        return None
    shape = [int(d) for d in x.shape]
    cout = int(weight.shape[0])
    cin = int(weight.shape[1])
    rows = 1
    for d in shape[:-1]:
        rows *= int(d)
    # The truth of this op is `linear()`'s answer, so the operands are cast to
    # exactly what the portable path would cast them to and nothing here
    # second-guesses the dtype. Under an autocast scope that is float16 for all
    # three; with no scope it is float32 and the casts are no-ops.
    dtype = compute_dtype_name(x, weight)
    if _jittor_dtype_name(x.dtype) != dtype:
        x = x.cast(dtype)
    if _jittor_dtype_name(weight.dtype) != dtype:
        weight = weight.cast(dtype)
    if _jittor_dtype_name(bias.dtype) != dtype:
        bias = bias.cast(dtype)
    return jt.code(shape[:-1] + [cout], dtype, [x, weight, bias],
                   cuda_header=_HEADER, cuda_src=_source(rows, cin, cout, dtype))


__all__ = ["lt_linear_cuda", "compute_dtype_name"]
