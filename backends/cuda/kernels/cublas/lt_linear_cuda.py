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

**Training: the weight gradient timed, the bias gradient folded where it is
cheap.** `lt_linear_train_cuda` keeps the portable forward and takes the
backward's dW through cuBLASLt with the algorithm picked by timing, as the
forward route does: ViT-B/16's 72 weight-gradient GEMMs went 28.6 -> 27.5 ms.
The portable backward sums the output gradient over every row for `db`, and
jittor fuses that sum into whatever produced the gradient -- BERT's GELU
backward became a column reduction, 123 us a call where the elementwise kernel
alone is 96. cuBLASLt's BGRADB epilogue answers the same sums, but on this
cuBLASLt as a kernel of its own that reads the gradient again: 5-16 us while
the gradient is still in L2, 270 us for ViT's 148 MB one, against ~60 us for
the sum fused into its producer. So the fold is taken only for a gradient
that fits the L2 cache (`_FOLD_L2_FRACTION`); a larger one keeps the fused
sum.

Kept out on purpose:

  - **A forward that needs a gradient.** `lt_linear_cuda` has no backward
    source, so a training step takes `lt_linear_train_cuda` or the portable
    way. Silently losing a gradient would be the worst possible outcome of a
    speedup.
  - Anything whose compute dtype is not float16 or float32 (bfloat16, float64),
    or not dense, or that leaves a non-contiguous operand; those fall back to
    the portable path.
"""

import functools
import os

import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._runtime.core_api import _output_requires_grad


#: Heuristic candidates to time. Beyond this the returns are gone and the
#: one-off cost is not.
_CANDIDATES = 8

#: Per-problem workspace. cuBLASLt reports what each candidate wants; the
#: fast ones for these shapes asked for 8 MB. It is borrowed from the tensor
#: pool on every call, so asking for more than the winners use only leaves a
#: larger hole in that pool between calls.
_WORKSPACE = 8 << 20

#: `dsize_` as `src/type/nano_string.h` defines it: 2**code is the byte width.
_DSIZE = {"float16": 1, "bfloat16": 1, "float32": 2, "float64": 3}

_HEADER = r"""
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/executor.h"
#include "mem/allocator.h"
#include "runtime/float32_precision.h"

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

// The workspace of one call, from the executor's temporary pool, as cuDNN's
// are (see `CudnnWorkspace`). It was a function-local static -- and every
// problem shape compiles its own kernel, so that was one 32 MB cudaMalloc per
// shape, held for the life of the process and outside every pool: 512 MB of
// an SD1.5 UNet's device memory, for sixteen shapes.
struct JtLtWorkspace {
    void* ptr = nullptr;
    size_t size = 0, allocation = 0;
    jittor::Allocator* allocator = nullptr;
    explicit JtLtWorkspace(size_t bytes) : size(bytes) {
        allocator = jittor::runtime_executor().temp_allocator;
        ptr = allocator->alloc(size, allocation);
    }
    ~JtLtWorkspace() { if (ptr) allocator->free(ptr, size, allocation); }
    JtLtWorkspace(const JtLtWorkspace&) = delete;
    JtLtWorkspace& operator=(const JtLtWorkspace&) = delete;
};

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

_WGRAD_HEADER = _HEADER + r"""
#include <cuda_bf16.h>
// The bias gradient when cuBLASLt offers no algorithm: g's column sums.
template <typename T>
__global__ static void jt_lt_column_sum(const T* g, T* out, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float s = 0.f;
    for (int r = 0; r < rows; r++) s += (float)g[(size_t)r * cols + c];
    out[c] = (T)s;
}
"""

#: cuBLASLt's name and the kernel's for each element type the training route
#: serves. bfloat16 is here though the forward route leaves it out: BGRADB
#: takes it with the same float32 accumulate.
_LT_TYPES = {"float32": ("CUDA_R_32F", "float"), "float16": ("CUDA_R_16F", "half"),
             "bfloat16": ("CUDA_R_16BF", "__nv_bfloat16")}


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

    // float32 follows the float32 matmul policy (`allow_tf32`,
    // `set_float32_matmul_precision`), as `cublas_gemm_mode` does for the
    // portable path: this route used to compute every float32 linear layer in
    // full float32 on the SIMT kernels while the policy asked for TF32 -- BERT
    // inference ran its GEMMs 24% slower than PyTorch's for no difference the
    // caller asked for. The tier is read when the op runs, so each tier keeps
    // its own measured algorithm.
    int tier = {"jittor::float32_matmul_tier()" if dtype == "float32" else "0"};
    cublasComputeType_t compute = tier == jittor::F32_HIGH ? CUBLAS_COMPUTE_32F_FAST_TF32
        : tier == jittor::F32_MEDIUM ? CUBLAS_COMPUTE_32F_FAST_16BF : CUBLAS_COMPUTE_32F;
    cublasLtMatmulDesc_t op = nullptr;
    // The scale type follows the *compute* type and the float alpha/beta, not
    // the operand type. Handing it the operand type is rejected outright --
    // cuBLASLt answers `CUBLAS_STATUS_INVALID_VALUE` to
    // `cublasLtMatmulAlgoGetHeuristic` for every algorithm, and the only trace
    // of it is the fallback running underneath (which is how the whole fused
    // route came to be 2.4x slower than not using it at all while still
    // producing correct numbers).
    cublasLtMatmulDescCreate(&op, compute, CUDA_R_32F);
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

    JtLtWorkspace workspace({_WORKSPACE});
    void* ws = workspace.ptr;
    size_t wsize = ws ? (size_t){_WORKSPACE} : 0;

    static JtLtChoice choices[3];
    JtLtChoice& choice = choices[tier < 0 || tier > 2 ? 0 : tier];
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
                     compute, CUBLAS_GEMM_DEFAULT);
        int total = rows * cout;
        jt_lt_add_bias<{kt}><<<(total + 255) / 256, 256>>>(out0_p, in2_p, total, cout);
    }}

    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
    """


#: The share of the L2 cache an output gradient may fill and still have its
#: bias gradient folded into the weight gradient's GEMM; see the module notes.
_FOLD_L2_FRACTION = 0.75

#: The L2 cache of each CUDA device, by ordinal, as the driver reports it.
_L2_BYTES = {}


def _l2_bytes():
    """The current CUDA device's L2 cache in bytes, or 0 if the driver cannot say."""
    index = int(jt.core.current_device())
    if index in _L2_BYTES:
        return _L2_BYTES[index]
    import ctypes
    size = 0
    try:
        driver = ctypes.CDLL("nvcuda" if os.name == "nt" else "libcuda.so.1")
    except OSError:
        driver = None
    if driver is not None:
        dev, value = ctypes.c_int(0), ctypes.c_int(0)
        # CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
        if (driver.cuInit(0) == 0 and driver.cuDeviceGet(ctypes.byref(dev), index) == 0
                and driver.cuDeviceGetAttribute(ctypes.byref(value), 38, dev) == 0):
            size = int(value.value)
    _L2_BYTES[index] = size
    return size


#: The three products of a linear layer in cuBLAS's column-major terms, the
#: row-major operands being `x` [rows, cin], `W` [cout, cin], `g` [rows, cout]:
#:   fwd  y  = x W^T   inputs (x, W)
#:   dx   dx = g W     inputs (g, W)
#:   dw   dW = g^T x   inputs (x, g) -- computed as dW^T = x^T g, so the bias
#:                     gradient, B summed over k, is BGRADB
#: Each is (m, n, k, A, op(A), A's stored rows and columns, B, op(B), B's
#: stored rows and columns); a stored matrix's leading dimension is its rows,
#: and C's is m.
_GEMMS = {
    "fwd": ("cout", "rows", "cin", "in1_p", "CUBLAS_OP_T", "cin", "cout",
            "in0_p", "CUBLAS_OP_N", "cin", "rows"),
    "dx": ("cin", "rows", "cout", "in1_p", "CUBLAS_OP_N", "cin", "cout",
           "in0_p", "CUBLAS_OP_N", "cout", "rows"),
    "dw": ("cin", "cout", "rows", "in0_p", "CUBLAS_OP_N", "cin", "rows",
           "in1_p", "CUBLAS_OP_T", "cout", "rows"),
}


@functools.lru_cache(maxsize=256)
def _gemm_source(kind, rows, cin, cout, dtype, fold_bias=False):
    ct, kt = _LT_TYPES[dtype]
    m, n, k, a, ta, a_rows, a_cols, b, tb, b_rows, b_cols = _GEMMS[kind]
    lda, ldb, ldc = a_rows, b_rows, m
    fold_bias = fold_bias and kind == "dw"
    epilogue = "CUBLASLT_EPILOGUE_BGRADB" if fold_bias else "CUBLASLT_EPILOGUE_DEFAULT"
    column_sum = (f"jt_lt_column_sum<{kt}><<<(cout + 255) / 256, 256>>>(\n"
                  f"            ({kt}*)in1_p, ({kt}*)out1_p, rows, cout);") if fold_bias else ""
    bias_pointer = ("void* biasp = (void*)out1_p;\n    cublasLtMatmulDescSetAttribute(op, "
                    "CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasp, sizeof(biasp));") if fold_bias else ""

    return f"""
    const int rows = {rows}, cin = {cin}, cout = {cout};
    const float alpha = 1.0f, beta = 0.0f;
    cublasLtHandle_t lt = jt_lt_handle();
    int tier = {"jittor::float32_matmul_tier()" if dtype == "float32" else "0"};
    cublasComputeType_t compute = tier == jittor::F32_HIGH ? CUBLAS_COMPUTE_32F_FAST_TF32
        : tier == jittor::F32_MEDIUM ? CUBLAS_COMPUTE_32F_FAST_16BF : CUBLAS_COMPUTE_32F;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatmulDescCreate(&op, compute, CUDA_R_32F);
    cublasOperation_t ta = {ta}, tb = {tb};
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    cublasLtEpilogue_t ep = {epilogue};
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
    {bias_pointer}
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtMatrixLayoutCreate(&la, {ct}, {a_rows}, {a_cols}, {lda});
    cublasLtMatrixLayoutCreate(&lb, {ct}, {b_rows}, {b_cols}, {ldb});
    cublasLtMatrixLayoutCreate(&lc, {ct}, {m}, {n}, {ldc});

    JtLtWorkspace workspace({_WORKSPACE});
    void* ws = workspace.ptr;
    size_t wsize = ws ? (size_t){_WORKSPACE} : 0;
    auto once = [&](const cublasLtMatmulAlgo_t* algo) {{
        return cublasLtMatmul(lt, op, &alpha, {a}, la, {b}, lb, &beta,
                              out0_p, lc, out0_p, lc, algo, ws, wsize, cudaStreamPerThread);
    }};

    static JtLtChoice choices[3];
    JtLtChoice& choice = choices[tier < 0 || tier > 2 ? 0 : tier];
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
            if (once(&cand[c].algo) != CUBLAS_STATUS_SUCCESS) continue;
            cudaDeviceSynchronize();
            cudaEventRecord(beg, cudaStreamPerThread);
            for (int r = 0; r < 3; r++) once(&cand[c].algo);
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
        once(&choice.algo);
    }} else {{
        cublasHandle_t h = jt_blas_handle();
        cublasGemmEx(h, {ta}, {tb}, {m}, {n}, {k}, &alpha,
                     {a}, {ct}, {lda}, {b}, {ct}, {ldb}, &beta, out0_p, {ct}, {ldc},
                     compute, CUBLAS_GEMM_DEFAULT);
        {column_sum}
    }}

    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
    """


def _rows_of(t):
    rows = 1
    for d in t.shape[:-1]:
        rows *= int(d)
    return rows


def lt_linear_gemm_cuda(kind, a, b, rows, cin, cout):
    """One product of a linear layer through cuBLASLt, the algorithm picked by
    timing once per shape; see `_GEMMS` for `kind` and its operands. Returns
    [rows, cout] for fwd, [rows, cin] for dx, [cout, cin] for dw."""
    dtype = _jittor_dtype_name(a.dtype)
    shape = {"fwd": list(a.shape[:-1]) + [cout], "dx": list(a.shape[:-1]) + [cin],
             "dw": [cout, cin]}[kind]
    return jt.code(shape, dtype, [a, b], cuda_header=_WGRAD_HEADER,
                   cuda_src=_gemm_source(kind, rows, cin, cout, dtype))


def lt_linear_wgrad_cuda(x, grad, fold_bias=True):
    """dW of `x @ W.T + b` for the output gradient `grad` -- and db with it,
    from the same cuBLASLt call, when `fold_bias`.

    `x` and `grad` are dense, of one dtype, with the rows in their leading
    dimensions; dW is [cout, cin] and db [cout], in that dtype. Returns
    (dW, db), or dW alone without `fold_bias`.
    """
    cin, cout = int(x.shape[-1]), int(grad.shape[-1])
    rows = _rows_of(x)
    if not fold_bias:
        return lt_linear_gemm_cuda("dw", x, grad, rows, cin, cout)
    dtype = _jittor_dtype_name(x.dtype)
    return jt.code([(cout, cin), (cout,)], [dtype, dtype], [x, grad], cuda_header=_WGRAD_HEADER,
                   cuda_src=_gemm_source("dw", rows, cin, cout, dtype, True))


class _LtLinearProduct(jt.Function):
    """`x @ W.T` -- each of its three products a cuBLASLt GEMM picked by
    timing -- whose backward also answers for the bias added after it.

    The bias, when there is one, is an input so that its gradient comes from
    here -- folded into the weight gradient's GEMM, or summed where that is
    cheaper; the caller adds a detached copy of it. Adding it in here instead
    would make the sum this Function's output -- and a Function's output is a
    tape, which shares its input's storage, so the sum would have to be
    written out rather than fused into whatever reads it.
    """

    def execute(self, x, weight, *bias):
        self.x, self.weight, self.has_bias = x, weight, bool(bias)
        self.wants_dx = not x.is_stop_grad()
        self.rows, self.cout, self.cin = _rows_of(x), int(weight.shape[0]), int(weight.shape[1])
        return lt_linear_gemm_cuda("fwd", x, weight, self.rows, self.cin, self.cout)

    def grad(self, grad):
        if grad is None:
            return (None, None) + (None,) * self.has_bias
        x, rows, cin, cout = self.x, self.rows, self.cin, self.cout
        dense = (_jittor_dtype_name(grad.dtype) == _jittor_dtype_name(x.dtype)
                 and grad._storage_is_contiguous() and x._storage_is_contiguous())
        db = None
        if not dense:
            rows_x = x.reshape((-1, cin))
            rows_g = grad.reshape((-1, cout))
            dw = jt.nn.matmul(rows_g.transpose(), rows_x)
            if self.has_bias:
                db = rows_g.sum(0)
            dx = jt.nn.matmul(grad, self.weight) if self.wants_dx else None
        else:
            if self.has_bias and grad.nbytes <= _l2_bytes() * _FOLD_L2_FRACTION:
                dw, db = lt_linear_wgrad_cuda(x, grad)
            else:
                dw = lt_linear_gemm_cuda("dw", x, grad, rows, cin, cout)
                if self.has_bias:
                    # Over the leading dimensions as they are: a reshape in
                    # between keeps the sum out of the kernel producing `grad`.
                    db = grad.sum(tuple(range(grad.ndim - 1)))
            dx = lt_linear_gemm_cuda("dx", grad, self.weight, rows, cin, cout) \
                if self.wants_dx else None
        return (dx, dw) + ((db,) if self.has_bias else ())


def _supports_training(x, weight, bias):
    if not jt.flags.use_cuda or jt.flags.no_grad:
        return False
    if not (isinstance(x, jt.Var) and isinstance(weight, jt.Var)):
        return False
    if bias is not None and not isinstance(bias, jt.Var):
        return False
    # A frozen weight leaves only the input gradient, which the portable path
    # serves as well; an autocast register casts inside the portable path,
    # which this does not reproduce.
    if weight.is_stop_grad() or jt.flags.amp_reg:
        return False
    dtype = _jittor_dtype_name(x.dtype)
    if dtype not in _LT_TYPES or _jittor_dtype_name(weight.dtype) != dtype:
        return False
    if bias is not None and (_jittor_dtype_name(bias.dtype) != dtype or len(bias.shape) != 1
                             or int(bias.shape[0]) != int(weight.shape[0])):
        return False
    if len(weight.shape) != 2 or len(x.shape) < 2:
        return False
    cout, cin = int(weight.shape[0]), int(weight.shape[1])
    if int(x.shape[-1]) != cin or not x._storage_is_contiguous():
        return False
    rows = _rows_of(x)
    # The route costs a Function call -- tapes, a context, a second operator
    # in the backward -- about 70 us of host time a layer, forward and
    # backward together. A product too small for a better GEMM or for its
    # bias sum to matter pays that for nothing: a DDPM UNet's forty-odd
    # 16-row time-embedding projections made its eager step 64.0 -> 68.3 ms
    # at an unchanged device time.
    return rows >= _MIN_ROWS and rows * cin * cout >= _MIN_PRODUCT


#: The smallest problem the training route takes: rows, and rows x cin x cout.
_MIN_ROWS = 1024
_MIN_PRODUCT = 1 << 28


def lt_linear_train_cuda(x, weight, bias=None):
    """`x @ weight.T (+ bias)` for a call that records gradients, its three
    GEMMs picked by timing and its bias gradient folded into the weight
    gradient's where that pays, or None if this cannot serve it.

    cuBLAS's heuristic pick is not the fastest for every shape: Qwen3's MLP
    weight gradients, [1024, 3072] over 2048 rows, ran 219 us against the
    172 of the best of cuBLASLt's candidates, and its 151936-wide output
    projection 8.23 ms against 7.50."""
    if not _supports_training(x, weight, bias):
        return None
    if bias is None:
        return _LtLinearProduct.apply(x, weight)
    return _LtLinearProduct.apply(x, weight, bias) + bias.detach()


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


__all__ = ["lt_linear_cuda", "lt_linear_train_cuda", "lt_linear_wgrad_cuda",
           "compute_dtype_name"]
