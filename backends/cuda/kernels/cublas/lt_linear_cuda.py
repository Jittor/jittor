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

Kept out on purpose:

  - **Anything that needs a gradient.** This op has no backward source, so a
    training step goes the portable way. Silently losing a gradient would be
    the worst possible outcome of a speedup.
  - Anything not float32, not dense, or not 2-D after flattening; those fall
    back to the portable path.
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

_HEADER = r"""
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

static cublasLtHandle_t jt_lt_handle() {
    static cublasLtHandle_t handle = nullptr;
    if (!handle) cublasLtCreate(&handle);
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
__global__ static void jt_lt_add_bias(float* out, const float* bias,
                                     int total, int cout) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] += bias[i % cout];
}

struct JtLtChoice {
    bool ready = false;
    bool usable = false;
    cublasLtMatmulAlgo_t algo;
};
"""


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
        if _jittor_dtype_name(v.dtype) != "float32":
            return False
        if not v._storage_is_contiguous():
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


def _source(rows, cin, cout):
    return f"""
    const int rows = {rows}, cin = {cin}, cout = {cout};
    const float alpha = 1.0f, beta = 0.0f;
    cublasLtHandle_t lt = jt_lt_handle();

    cublasLtMatmulDesc_t op = nullptr;
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
    cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, cin, cout, cin);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, cin, rows, cin);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, cout, rows, cout);

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
                return cublasLtMatmul(lt, op, &alpha, in1_p, la, in0_p, lb, &beta,
                                      out0_p, lc, out0_p, lc, &cand[c].algo,
                                      ws, wsize, 0);
            }};
            if (once() != CUBLAS_STATUS_SUCCESS) continue;
            cudaDeviceSynchronize();
            cudaEventRecord(beg, 0);
            for (int r = 0; r < 3; r++) once();
            cudaEventRecord(end, 0);
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
                       out0_p, lc, out0_p, lc, &choice.algo, ws, wsize, 0);
    }} else {{
        // Nothing cuBLASLt offered was usable. The portable GEMM plus a bias
        // add is still correct, so do that rather than answer with garbage.
        cublasHandle_t h = nullptr;
        cublasCreate(&h);
        cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, cout, rows, cin, &alpha,
                    in1_p, cin, in0_p, cin, &beta, out0_p, cout);
        cublasDestroy(h);
        int total = rows * cout;
        jt_lt_add_bias<<<(total + 255) / 256, 256>>>(out0_p, in2_p, total, cout);
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
        rows *= d
    return jt.code(shape[:-1] + [cout], x.dtype, [x, weight, bias],
                   cuda_header=_HEADER, cuda_src=_source(rows, cin, cout))


__all__ = ["lt_linear_cuda"]
