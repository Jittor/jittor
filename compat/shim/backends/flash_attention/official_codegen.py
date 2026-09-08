"""FlashAttention official codegen implementation."""
from __future__ import annotations
import pathlib

def _official_stub_source(build_dir: str, root: pathlib.Path) -> str:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    path = _facade.pathlib.Path(build_dir) / "flashattn_jittor_bwd_stubs.cu"
    compiled = _facade._official_compiled_specs(root)
    fwd_lines = []
    split_lines = []
    bwd_lines = []
    for dtype in _facade._OFFICIAL_FLASH_ATTN_DTYPES:
        ctype = "cutlass::half_t" if dtype == "fp16" else "cutlass::bfloat16_t"
        for dim in _facade._OFFICIAL_FLASH_ATTN_HEAD_DIMS:
            for causal in (False, True):
                cbool = "true" if causal else "false"
                if ("fwd", dtype, dim, causal) not in compiled:
                    fwd_lines.append("JT_FLASHATTN_FWD_STUB(%s, %s, %s)" % (ctype, dim, cbool))
                if ("split", dtype, dim, causal) not in compiled:
                    split_lines.append("JT_FLASHATTN_SPLIT_FWD_STUB(%s, %s, %s)" % (ctype, dim, cbool))
                if ("bwd", dtype, dim, causal) not in compiled:
                    bwd_lines.append("JT_FLASHATTN_BWD_STUB(%s, %s, %s)" % (
                        ctype, dim, cbool))

    body = r'''
#include <stdexcept>
#include <cuda_runtime.h>
#include "namespace_config.h"
#include <cutlass/numeric_types.h>
#include "flash.h"

namespace FLASH_NAMESPACE {
template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_(Flash_fwd_params&, cudaStream_t) {
    throw std::runtime_error("flashattn_jittor official backend was built without this forward kernel; set JITTOR_FLASH_ATTN_HEAD_DIMS=all or include the requested head dimension");
}

template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params&, cudaStream_t) {
    throw std::runtime_error("flashattn_jittor official backend was built without this split forward kernel; set JITTOR_FLASH_ATTN_HEAD_DIMS=all or include the requested head dimension");
}

template<typename T, int Headdim, bool Is_causal>
void run_mha_bwd_(Flash_bwd_params&, cudaStream_t) {
    throw std::runtime_error("flashattn_jittor official backend was built without this backward kernel; set JITTOR_FLASH_ATTN_HEAD_DIMS=all or include the requested head dimension");
}

#define JT_FLASHATTN_FWD_STUB(DTYPE, HDIM, CAUSAL) \
template void run_mha_fwd_<DTYPE, HDIM, CAUSAL>(Flash_fwd_params&, cudaStream_t);

#define JT_FLASHATTN_SPLIT_FWD_STUB(DTYPE, HDIM, CAUSAL) \
template void run_mha_fwd_splitkv_dispatch<DTYPE, HDIM, CAUSAL>(Flash_fwd_params&, cudaStream_t);

#define JT_FLASHATTN_BWD_STUB(DTYPE, HDIM, CAUSAL) \
template void run_mha_bwd_<DTYPE, HDIM, CAUSAL>(Flash_bwd_params&, cudaStream_t);

%s
%s
%s
#undef JT_FLASHATTN_FWD_STUB
#undef JT_FLASHATTN_SPLIT_FWD_STUB
#undef JT_FLASHATTN_BWD_STUB
} // namespace FLASH_NAMESPACE
''' % ("\n".join(fwd_lines), "\n".join(split_lines), "\n".join(bwd_lines))
    try:
        old = path.read_text(encoding="utf-8")
    except OSError:
        old = None
    if old != body:
        path.write_text(body, encoding="utf-8")
    return _facade.os.fspath(path)


def _official_packed_source(build_dir: str) -> str:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    path = _facade.pathlib.Path(build_dir) / "flashattn_jittor_packed_fwd.cu"
    body = r'''
#include <cmath>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "hardware_info.h"
#include "flash.h"
#include "static_switch.h"

namespace jtorch { namespace detail {
void data_ptrs(std::initializer_list<jtorch::Tensor> tensors, void** out);
}} // namespace jtorch::detail

namespace FLASH_NAMESPACE {

static at::Tensor jt_readonly_tensor(py::handle obj, const char *name) {
    TORCH_CHECK(!obj.is_none(), name, " must be a Jittor Var");
    TORCH_CHECK(::jtorch::detail::is_jittor_var(obj.ptr()), name, " must be a Jittor Var");
    return ::jtorch::detail::tensor_from_pyvar_readonly(obj.ptr());
}

static inline int jt_round_multiple(int x, int m) {
    return (x + m - 1) / m * m;
}

static void jt_run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        });
    });
}

static void jt_check_cuda(const at::Tensor &x, const char *name) {
    TORCH_CHECK(x.is_cuda(), name, " must be on CUDA");
}

static at::TensorOptions jt_cuda_options(at::ScalarType dtype) {
    return torch::dtype(dtype).device(torch::kCUDA);
}

static void jt_fill_params(
        Flash_fwd_params &params,
        at::ScalarType dtype,
        int batch_size,
        int seqlen_q,
        int seqlen_k,
        int num_heads,
        int num_heads_k,
        int head_size,
        void *q_ptr,
        void *k_ptr,
        void *v_ptr,
        int64_t q_batch_stride,
        int64_t k_batch_stride,
        int64_t v_batch_stride,
        int64_t q_row_stride,
        int64_t k_row_stride,
        int64_t v_row_stride,
        int64_t q_head_stride,
        int64_t k_head_stride,
        int64_t v_head_stride,
        at::Tensor &out,
        void *out_ptr,
        void *cu_seqlens_q,
        void *cu_seqlens_k,
        at::Tensor &softmax_lse,
        void *softmax_lse_ptr,
        float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        bool unpadded_lse) {
    params = {};
    params.is_bf16 = dtype == torch::kBFloat16;
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_ptr = out_ptr;
    params.o_batch_stride = out.stride(0);
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);
    params.p_ptr = nullptr;
    params.softmax_lse_ptr = softmax_lse_ptr;
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = jt_round_multiple(seqlen_q, 128);
    params.seqlen_k_rounded = jt_round_multiple(seqlen_k, 128);
    params.d = head_size;
    params.d_rounded = jt_round_multiple(head_size, head_size <= 128 ? 32 : 64);
    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k);
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.p_dropout = 1.f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.f;
    params.scale_softmax_rp_dropout = softmax_scale;
    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }
    if (is_causal) { window_size_right = 0; }
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.is_causal = window_size_left < 0 && window_size_right == 0;
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    params.is_seqlens_k_cumulative = true;
    params.unpadded_lse = unpadded_lse;
    params.num_splits = 1;
    params.total_q = unpadded_lse ? out.size(0) : 0;
}

static void jt_finish_and_run(Flash_fwd_params &params, void *rng_state_ptr) {
    params.rng_state = reinterpret_cast<uint64_t *>(rng_state_ptr);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    jt_run_mha_fwd(params, stream);
}

at::Tensor
jt_fwd(py::handle q_obj,
       py::handle k_obj,
       py::handle v_obj,
       float softmax_scale,
       bool is_causal,
       int window_size_left,
       int window_size_right) {
    auto q = jt_readonly_tensor(q_obj, "q");
    auto k = jt_readonly_tensor(k_obj, "k");
    auto v = jt_readonly_tensor(v_obj, "v");
    at::cuda::CUDAGuard device_guard{0};
    TORCH_CHECK(q.dim() == 4, "q must be [batch, seqlen_q, heads, dim]");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k/v must be [batch, seqlen_k, heads, dim]");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "q must be fp16 or bf16");
    TORCH_CHECK(k.dtype() == q.dtype(), "k dtype mismatch");
    TORCH_CHECK(v.dtype() == q.dtype(), "v dtype mismatch");
    const int batch_size = q.size(0);
    const int seqlen_q = q.size(1);
    const int seqlen_k = k.size(1);
    const int num_heads = q.size(2);
    const int num_heads_k = k.size(2);
    const int head_size = q.size(3);
    TORCH_CHECK(batch_size == k.size(0) && batch_size == v.size(0), "q/k/v batch mismatch");
    TORCH_CHECK(k.size(1) == v.size(1), "k/v seqlen mismatch");
    TORCH_CHECK(k.size(2) == v.size(2), "k/v heads mismatch");
    TORCH_CHECK(head_size == k.size(3) && head_size == v.size(3), "q/k/v head dim mismatch");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide query heads");
    auto opts = jt_cuda_options(q.dtype());
    auto out = torch::empty({batch_size, seqlen_q, num_heads, head_size}, opts);
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(torch::kFloat));
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    void *ptrs[6];
    ::jtorch::detail::data_ptrs({q, k, v, out, softmax_lse, rng_state}, ptrs);
    jt_check_cuda(q, "q");
    jt_check_cuda(k, "k");
    jt_check_cuda(v, "v");
    Flash_fwd_params params;
    jt_fill_params(params, q.dtype(), batch_size, seqlen_q, seqlen_k,
                   num_heads, num_heads_k, head_size,
                   ptrs[0], ptrs[1], ptrs[2],
                   q.stride(0), k.stride(0), v.stride(0),
                   q.stride(1), k.stride(1), v.stride(1),
                   q.stride(2), k.stride(2), v.stride(2),
                   out, ptrs[3], nullptr, nullptr, softmax_lse, ptrs[4], softmax_scale,
                   is_causal, window_size_left, window_size_right, false);
    jt_finish_and_run(params, ptrs[5]);
    return out;
}

at::Tensor
jt_varlen_fwd(py::handle q_obj,
              py::handle k_obj,
              py::handle v_obj,
              py::handle cu_seqlens_q_obj,
              py::handle cu_seqlens_k_obj,
              int max_seqlen_q,
              int max_seqlen_k,
              float softmax_scale,
              bool is_causal,
              int window_size_left,
              int window_size_right) {
    auto q = jt_readonly_tensor(q_obj, "q");
    auto k = jt_readonly_tensor(k_obj, "k");
    auto v = jt_readonly_tensor(v_obj, "v");
    auto cu_seqlens_q = jt_readonly_tensor(cu_seqlens_q_obj, "cu_seqlens_q");
    auto cu_seqlens_k = jt_readonly_tensor(cu_seqlens_k_obj, "cu_seqlens_k");
    at::cuda::CUDAGuard device_guard{0};
    TORCH_CHECK(q.dim() == 3, "q must be [total_q, heads, dim]");
    TORCH_CHECK(k.dim() == 3 && v.dim() == 3, "k/v must be [total_k, heads, dim]");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "q must be fp16 or bf16");
    TORCH_CHECK(k.dtype() == q.dtype(), "k dtype mismatch");
    TORCH_CHECK(v.dtype() == q.dtype(), "v dtype mismatch");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 && cu_seqlens_k.dtype() == torch::kInt32,
                "cu_seqlens tensors must be int32");
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int total_q = q.size(0);
    const int num_heads = q.size(1);
    const int num_heads_k = k.size(1);
    const int head_size = q.size(2);
    TORCH_CHECK(cu_seqlens_k.numel() == cu_seqlens_q.numel(), "cu_seqlens batch mismatch");
    TORCH_CHECK(k.size(0) == v.size(0), "k/v total length mismatch");
    TORCH_CHECK(k.size(1) == v.size(1), "k/v heads mismatch");
    TORCH_CHECK(head_size == k.size(2) && head_size == v.size(2), "q/k/v head dim mismatch");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide query heads");
    auto opts = jt_cuda_options(q.dtype());
    auto out = torch::empty({total_q, num_heads, head_size}, opts);
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(torch::kFloat));
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    void *ptrs[8];
    ::jtorch::detail::data_ptrs({q, k, v, cu_seqlens_q, cu_seqlens_k,
                                 out, softmax_lse, rng_state}, ptrs);
    jt_check_cuda(q, "q");
    jt_check_cuda(k, "k");
    jt_check_cuda(v, "v");
    jt_check_cuda(cu_seqlens_q, "cu_seqlens_q");
    jt_check_cuda(cu_seqlens_k, "cu_seqlens_k");
    Flash_fwd_params params;
    jt_fill_params(params, q.dtype(), batch_size, max_seqlen_q, max_seqlen_k,
                   num_heads, num_heads_k, head_size,
                   ptrs[0], ptrs[1], ptrs[2],
                   0, 0, 0,
                   q.stride(0), k.stride(0), v.stride(0),
                   q.stride(1), k.stride(1), v.stride(1),
                   out, ptrs[5],
                   ptrs[3],
                   ptrs[4],
                   softmax_lse, ptrs[6], softmax_scale, is_causal,
                   window_size_left, window_size_right, true);
    jt_finish_and_run(params, ptrs[7]);
    return out;
}

at::Tensor
jt_varlen_qkvpacked_fwd(py::handle qkv_obj,
                        py::handle cu_seqlens_obj,
                        int max_seqlen,
                        float softmax_scale,
                        bool is_causal,
                        int window_size_left,
                        int window_size_right) {
    auto qkv = jt_readonly_tensor(qkv_obj, "qkv");
    auto cu_seqlens = jt_readonly_tensor(cu_seqlens_obj, "cu_seqlens");
    at::cuda::CUDAGuard device_guard{0};
    TORCH_CHECK(qkv.dim() == 4, "qkv must be [total, 3, heads, dim]");
    TORCH_CHECK(qkv.size(1) == 3, "qkv packed dimension must be 3");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(qkv.dtype() == torch::kFloat16 || qkv.dtype() == torch::kBFloat16,
                "qkv must be fp16 or bf16");
    const int batch_size = cu_seqlens.numel() - 1;
    const int total_q = qkv.size(0);
    const int num_heads = qkv.size(2);
    const int head_size = qkv.size(3);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256 && head_size % 8 == 0, "unsupported head size");
    auto opts = jt_cuda_options(qkv.dtype());
    auto out = torch::empty({total_q, num_heads, head_size}, opts);
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(torch::kFloat));
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    void *ptrs[5];
    ::jtorch::detail::data_ptrs({qkv, cu_seqlens, out, softmax_lse, rng_state}, ptrs);
    jt_check_cuda(qkv, "qkv");
    jt_check_cuda(cu_seqlens, "cu_seqlens");
    char *base = reinterpret_cast<char *>(ptrs[0]);
    const int64_t elem = qkv.element_size();
    const int64_t fused_stride = qkv.stride(1);
    Flash_fwd_params params;
    jt_fill_params(params, qkv.dtype(), batch_size, max_seqlen, max_seqlen,
                   num_heads, num_heads, head_size,
                   base,
                   base + fused_stride * elem,
                   base + 2 * fused_stride * elem,
                   0, 0, 0,
                   qkv.stride(0), qkv.stride(0), qkv.stride(0),
                   qkv.stride(2), qkv.stride(2), qkv.stride(2),
                   out, ptrs[2],
                   ptrs[1],
                   ptrs[1],
                   softmax_lse, ptrs[3], softmax_scale, is_causal,
                   window_size_left, window_size_right, true);
    jt_finish_and_run(params, ptrs[4]);
    return out;
}

at::Tensor
jt_varlen_kvpacked_fwd(py::handle q_obj,
                       py::handle kv_obj,
                       py::handle cu_seqlens_q_obj,
                       py::handle cu_seqlens_k_obj,
                       int max_seqlen_q,
                       int max_seqlen_k,
                       float softmax_scale,
                       bool is_causal,
                       int window_size_left,
                       int window_size_right) {
    auto q = jt_readonly_tensor(q_obj, "q");
    auto kv = jt_readonly_tensor(kv_obj, "kv");
    auto cu_seqlens_q = jt_readonly_tensor(cu_seqlens_q_obj, "cu_seqlens_q");
    auto cu_seqlens_k = jt_readonly_tensor(cu_seqlens_k_obj, "cu_seqlens_k");
    at::cuda::CUDAGuard device_guard{0};
    TORCH_CHECK(q.dim() == 3, "q must be [total_q, heads, dim]");
    TORCH_CHECK(kv.dim() == 4 && kv.size(1) == 2, "kv must be [total_k, 2, heads, dim]");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "q must be fp16 or bf16");
    TORCH_CHECK(kv.dtype() == q.dtype(), "kv dtype mismatch");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 && cu_seqlens_k.dtype() == torch::kInt32,
                "cu_seqlens tensors must be int32");
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int total_q = q.size(0);
    const int num_heads = q.size(1);
    const int num_heads_k = kv.size(2);
    const int head_size = q.size(2);
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide query heads");
    TORCH_CHECK(head_size == kv.size(3), "q/kv head dim mismatch");
    auto opts = jt_cuda_options(q.dtype());
    auto out = torch::empty({total_q, num_heads, head_size}, opts);
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(torch::kFloat));
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    void *ptrs[7];
    ::jtorch::detail::data_ptrs({q, kv, cu_seqlens_q, cu_seqlens_k,
                                 out, softmax_lse, rng_state}, ptrs);
    jt_check_cuda(q, "q");
    jt_check_cuda(kv, "kv");
    jt_check_cuda(cu_seqlens_q, "cu_seqlens_q");
    jt_check_cuda(cu_seqlens_k, "cu_seqlens_k");
    char *kv_base = reinterpret_cast<char *>(ptrs[1]);
    const int64_t elem = kv.element_size();
    const int64_t fused_stride = kv.stride(1);
    Flash_fwd_params params;
    jt_fill_params(params, q.dtype(), batch_size, max_seqlen_q, max_seqlen_k,
                   num_heads, num_heads_k, head_size,
                   ptrs[0],
                   kv_base,
                   kv_base + fused_stride * elem,
                   0, 0, 0,
                   q.stride(0), kv.stride(0), kv.stride(0),
                   q.stride(1), kv.stride(2), kv.stride(2),
                   out, ptrs[4],
                   ptrs[2],
                   ptrs[3],
                   softmax_lse, ptrs[5], softmax_scale, is_causal,
                   window_size_left, window_size_right, true);
    jt_finish_and_run(params, ptrs[6]);
    return out;
}

at::Tensor
jt_qkvpacked_fwd(py::handle qkv_obj,
                 float softmax_scale,
                 bool is_causal,
                 int window_size_left,
    int window_size_right) {
    auto qkv = jt_readonly_tensor(qkv_obj, "qkv");
    at::cuda::CUDAGuard device_guard{0};
    TORCH_CHECK(qkv.dim() == 5 && qkv.size(2) == 3, "qkv must be [batch, seqlen, 3, heads, dim]");
    TORCH_CHECK(qkv.dtype() == torch::kFloat16 || qkv.dtype() == torch::kBFloat16,
                "qkv must be fp16 or bf16");
    const int batch_size = qkv.size(0);
    const int seqlen = qkv.size(1);
    const int num_heads = qkv.size(3);
    const int head_size = qkv.size(4);
    auto opts = jt_cuda_options(qkv.dtype());
    auto out = torch::empty({batch_size, seqlen, num_heads, head_size}, opts);
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen}, opts.dtype(torch::kFloat));
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    void *ptrs[4];
    ::jtorch::detail::data_ptrs({qkv, out, softmax_lse, rng_state}, ptrs);
    jt_check_cuda(qkv, "qkv");
    char *base = reinterpret_cast<char *>(ptrs[0]);
    const int64_t elem = qkv.element_size();
    const int64_t fused_stride = qkv.stride(2);
    Flash_fwd_params params;
    jt_fill_params(params, qkv.dtype(), batch_size, seqlen, seqlen,
                   num_heads, num_heads, head_size,
                   base,
                   base + fused_stride * elem,
                   base + 2 * fused_stride * elem,
                   qkv.stride(0), qkv.stride(0), qkv.stride(0),
                   qkv.stride(1), qkv.stride(1), qkv.stride(1),
                   qkv.stride(3), qkv.stride(3), qkv.stride(3),
                   out, ptrs[1], nullptr, nullptr, softmax_lse, ptrs[2], softmax_scale,
                   is_causal, window_size_left, window_size_right, false);
    jt_finish_and_run(params, ptrs[3]);
    return out;
}

at::Tensor
jt_kvpacked_fwd(py::handle q_obj,
                py::handle kv_obj,
                float softmax_scale,
                bool is_causal,
                int window_size_left,
    int window_size_right) {
    auto q = jt_readonly_tensor(q_obj, "q");
    auto kv = jt_readonly_tensor(kv_obj, "kv");
    at::cuda::CUDAGuard device_guard{0};
    TORCH_CHECK(q.dim() == 4, "q must be [batch, seqlen_q, heads, dim]");
    TORCH_CHECK(kv.dim() == 5 && kv.size(2) == 2, "kv must be [batch, seqlen_k, 2, heads, dim]");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "q must be fp16 or bf16");
    TORCH_CHECK(kv.dtype() == q.dtype(), "kv dtype mismatch");
    const int batch_size = q.size(0);
    const int seqlen_q = q.size(1);
    const int seqlen_k = kv.size(1);
    const int num_heads = q.size(2);
    const int num_heads_k = kv.size(3);
    const int head_size = q.size(3);
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide query heads");
    TORCH_CHECK(head_size == kv.size(4), "q/kv head dim mismatch");
    auto opts = jt_cuda_options(q.dtype());
    auto out = torch::empty({batch_size, seqlen_q, num_heads, head_size}, opts);
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(torch::kFloat));
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    void *ptrs[5];
    ::jtorch::detail::data_ptrs({q, kv, out, softmax_lse, rng_state}, ptrs);
    jt_check_cuda(q, "q");
    jt_check_cuda(kv, "kv");
    char *kv_base = reinterpret_cast<char *>(ptrs[1]);
    const int64_t elem = kv.element_size();
    const int64_t fused_stride = kv.stride(2);
    Flash_fwd_params params;
    jt_fill_params(params, q.dtype(), batch_size, seqlen_q, seqlen_k,
                   num_heads, num_heads_k, head_size,
                   ptrs[0],
                   kv_base,
                   kv_base + fused_stride * elem,
                   q.stride(0), kv.stride(0), kv.stride(0),
                   q.stride(1), kv.stride(1), kv.stride(1),
                   q.stride(2), kv.stride(3), kv.stride(3),
                   out, ptrs[2], nullptr, nullptr, softmax_lse, ptrs[3], softmax_scale,
                   is_causal, window_size_left, window_size_right, false);
    jt_finish_and_run(params, ptrs[4]);
    return out;
}

} // namespace FLASH_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &FLASH_NAMESPACE::jt_fwd, "Jittor direct forward");
    m.def("varlen_fwd", &FLASH_NAMESPACE::jt_varlen_fwd, "Jittor direct varlen forward");
    m.def("varlen_qkvpacked_fwd", &FLASH_NAMESPACE::jt_varlen_qkvpacked_fwd, "Jittor direct varlen qkvpacked forward");
    m.def("varlen_kvpacked_fwd", &FLASH_NAMESPACE::jt_varlen_kvpacked_fwd, "Jittor direct varlen kvpacked forward");
    m.def("qkvpacked_fwd", &FLASH_NAMESPACE::jt_qkvpacked_fwd, "Jittor direct qkvpacked forward");
    m.def("kvpacked_fwd", &FLASH_NAMESPACE::jt_kvpacked_fwd, "Jittor direct kvpacked forward");
}
'''
    try:
        old = path.read_text(encoding="utf-8")
    except OSError:
        old = None
    if old != body:
        path.write_text(body, encoding="utf-8")
    return _facade.os.fspath(path)
