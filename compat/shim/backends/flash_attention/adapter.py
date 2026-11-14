"""FlashAttention adapter implementation."""
from __future__ import annotations
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
import pathlib
from types import ModuleType
from typing import Optional, Tuple

def _window_size_pair(window_size, window_size_left=-1, window_size_right=-1) -> Tuple[int, int]:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    if window_size is not None:
        try:
            return int(window_size[0]), int(window_size[1])
        except _facade.EXPECTED as exc:
            _facade.swallowed("shim/backends/flash_attention.py _window_size_pair: return int(window_size[0]), int(window_size[1])", exc)
    return int(window_size_left), int(window_size_right)


def _flashattn_result(result, return_attn_probs: bool = False):
    if return_attn_probs:
        return result[0], result[1], result[2]
    return result[0]


def _dtype_name(x) -> str:
    return _jittor_dtype_name(getattr(x, "dtype", ""))


def _native_supported_dtype(x) -> bool:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    return _facade._dtype_name(x) in ("float16", "bfloat16")


def _float32_cast_target():
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    raw = (_facade.os.environ.get("JITTOR_FLASH_ATTN_CAST_FLOAT32") or "").strip().lower()
    if raw in ("1", "true", "yes", "on", "bf16", "bfloat16"):
        return "bfloat16"
    if raw in ("fp16", "float16", "half"):
        return "float16"
    return None


def _maybe_cast_float32_tensor(x, target: Optional[str]):
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    if target and _facade._dtype_name(x) == "float32":
        return x.to(target)
    return x


def _mark_readonly_borrow(*tensors):
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    if _facade._torch_ext_borrow_inputs_enabled():
        return []
    saved = []
    for tensor in tensors:
        if tensor is None:
            continue
        try:
            old_value = getattr(tensor, _facade._READONLY_BORROW_ATTR)
        except AttributeError:
            old_value = _facade._MISSING_ATTR
        except (AttributeError, TypeError) as exc:
            _facade.swallowed("shim/backends/flash_attention.py _mark_readonly_borrow: old_value = getattr(tensor, _READONLY_BORROW_ATTR)", exc)
            continue
        try:
            setattr(tensor, _facade._READONLY_BORROW_ATTR, True)
        except (AttributeError, TypeError) as exc:
            _facade.swallowed("shim/backends/flash_attention.py _mark_readonly_borrow: setattr(tensor, _READONLY_BORROW_ATTR, True)", exc)
            continue
        saved.append((tensor, old_value))
    return saved


def _restore_readonly_borrow(saved) -> None:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    for tensor, old_value in reversed(saved):
        try:
            if old_value is _facade._MISSING_ATTR:
                delattr(tensor, _facade._READONLY_BORROW_ATTR)
            else:
                setattr(tensor, _facade._READONLY_BORROW_ATTR, old_value)
        except (AttributeError, TypeError) as exc:
            _facade.swallowed("shim/backends/flash_attention.py _restore_readonly_borrow: if old_value is _MISSING_ATTR:", exc)


def _make_official_backend(low_level: ModuleType, root: pathlib.Path,
                           packed_low_level: Optional[ModuleType] = None) -> ModuleType:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    import jittor as jt

    mod = _facade.ModuleType("flashattn_jittor_official")
    mod.__file__ = _facade.os.fspath(root)
    mod._flashattn_jittor_official = True
    mod._flashattn_jittor_low_level = low_level
    mod._flashattn_jittor_packed_low_level = packed_low_level
    mod._flashattn_jittor_head_dims = tuple(_facade._official_head_dims(root))
    mod._flashattn_jittor_dtypes = tuple(_facade._official_dtypes())
    mod._flashattn_jittor_training = True
    mod._flashattn_jittor_packed_split_stats = _facade._PACKED_SPLIT_STATS
    low_fwd = low_level.fwd
    low_varlen_fwd = low_level.varlen_fwd
    low_bwd = low_level.bwd
    low_varlen_bwd = low_level.varlen_bwd
    packed_split_enabled = _facade._packed_split_enabled()

    def _check_args(dropout_p, softcap, alibi_slopes, return_attn_probs):
        dropout = float(dropout_p or 0.0)
        if dropout < 0.0 or dropout >= 1.0:
            raise RuntimeError(
                "flashattn_jittor official backend requires 0 <= dropout_p < 1")
        if softcap not in (0, 0.0, None) and float(softcap) > 0.0:
            raise RuntimeError("flashattn_jittor official backend does not support softcap")
        if alibi_slopes is not None:
            raise RuntimeError("flashattn_jittor official backend does not support alibi_slopes")
        if return_attn_probs and dropout == 0.0:
            raise RuntimeError(
                "flashattn_jittor official backend requires dropout_p > 0 "
                "when return_attn_probs is enabled")

    def _grad_enabled(*tensors):
        return not getattr(jt.flags, "no_grad", 0) and any(
            bool(getattr(tensor, "requires_grad", False)) for tensor in tensors)

    def _check_dropout_backward(q, dropout, needs_grad):
        head_dim = int(q.shape[-1])
        cuda_archs = tuple(getattr(jt.flags, "cuda_archs", ()))
        if (needs_grad and dropout > 0.0
                and not _facade._official_dropout_backward_supported(head_dim, cuda_archs)):
            arch_label = ",".join("sm%s" % arch for arch in cuda_archs) or "unknown"
            raise RuntimeError(
                "flashattn_jittor official backend does not support dropout "
                "backward for head dimension %s on CUDA architecture %s; "
                "upstream supports head dimensions above 192 with dropout "
                "only on sm80 or sm90" % (head_dim, arch_label))

    def _cast_result(result, dtype, return_attn_probs):
        if not return_attn_probs:
            return result.to(dtype)
        return result[0].to(dtype), result[1], result[2]

    def _opaque_capture_boundary():
        from jittor._runtime import step_capture
        if step_capture.active():
            step_capture.refuse('this FlashAttention call requires its Python extension boundary')

    def _graph_dense_forward(q, k, v, scale, causal, wl, wr):
        address = int(packed_low_level._raw_dense_fwd_address())
        b, m, h, d = map(int, q.shape)
        n, hk = int(k.shape[1]), int(k.shape[2])
        if (len(k.shape) != 4 or tuple(k.shape) != tuple(v.shape)
                or int(k.shape[0]) != b or int(k.shape[3]) != d
                or hk <= 0 or h % hk or not 0 < d <= 256 or d % 8):
            raise RuntimeError('unsupported dense FlashAttention graph shape')
        inputs = [value.contiguous() for value in (q, k, v)]
        header = '''
#include "stream_compat.h"
using FlashDenseLaunch = void (*)(void*, void*, void*, void*, float*,
    int, int, int, int, int, int, float, bool, int, int, bool, cudaStream_t);
'''
        source = (
            'int device; CHECK(cudaGetDevice(&device) == cudaSuccess);\n'
            'auto launch = reinterpret_cast<FlashDenseLaunch>(uintptr_t(%dULL));\n' % address
            + 'launch(in0_p, in1_p, in2_p, out0_p, out1_p, '
            + ', '.join(map(str, (b, m, n, h, hk, d)))
            + ', %sf, %s, %d, %d, %s, jittor::cuda_compute_stream(device));\n' % (
                repr(float(scale)), str(bool(causal)).lower(), wl, wr,
                str(_facade._dtype_name(q) == 'bfloat16').lower())
            + 'CHECK(cudaGetLastError() == cudaSuccess);')
        out, _lse = jt.code([(b, m, h, d), (b, h, m)], [q.dtype, 'float32'], inputs,
                            cuda_header=header, cuda_src=source)
        return out

    def _can_graph_dense(q, k, v):
        from jittor._runtime.dispatch import dispatch_context
        return (all(len(value.shape) == 4 and all(int(size) > 0 for size in value.shape)
                    and isinstance(value, jt.Var) for value in (q, k, v))
                and dispatch_context(q, k, v).backend == 'cuda'
                and _facade._dtype_name(q) == _facade._dtype_name(k) == _facade._dtype_name(v)
                and tuple(k.shape) == tuple(v.shape) and q.shape[0] == k.shape[0]
                and q.shape[3] == k.shape[3] and 0 < int(q.shape[3]) <= 256
                and int(q.shape[3]) % 8 == 0 and int(q.shape[2]) % int(k.shape[2]) == 0)

    class _DenseNativeAttention(jt.Function):
        def __init__(self, dropout, scale, causal, wl, wr, deterministic,
                     return_attn_probs):
            self.config = (
                float(dropout), float(scale), bool(causal), int(wl), int(wr),
                bool(deterministic), bool(return_attn_probs),
            )

        def execute(self, q, k, v):
            dropout, scale, causal, wl, wr, _, return_attn_probs = self.config
            saved = _facade._mark_readonly_borrow(q, k, v)
            try:
                result = low_fwd(
                    q, k, v, None, None, dropout, scale, causal, wl, wr,
                    0.0, return_attn_probs, None)
            finally:
                _facade._restore_readonly_borrow(saved)
            out, softmax_lse, probability, rng_state = result
            # The C++ extension returns a frozen factory allocation. This
            # Function supplies its backward, so the output must be eligible
            # before Function tapes it under the explicit-grad policy.
            out.start_grad()
            self.saved = (q, k, v, out, softmax_lse, rng_state)
            if return_attn_probs:
                return out, softmax_lse, probability
            return out

        def grad(self, grad_out, *unused):
            q, k, v, out, softmax_lse, rng_state = self.saved
            dropout, scale, causal, wl, wr, deterministic, _ = self.config
            grad_out = grad_out.contiguous()
            saved = _facade._mark_readonly_borrow(
                grad_out, q, k, v, out, softmax_lse, rng_state)
            try:
                result = low_bwd(
                    grad_out, q, k, v, out, softmax_lse,
                    None, None, None, None, dropout, scale, causal, wl, wr,
                    0.0, deterministic, None, rng_state)
            finally:
                _facade._restore_readonly_borrow(saved)
            for gradient in result[:3]:
                gradient._set_first_order_only()
            return result[0], result[1], result[2]

    class _VarlenNativeAttention(jt.Function):
        def __init__(self, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                     max_seqlen_k, dropout, scale, causal, wl, wr,
                     deterministic, return_attn_probs):
            self.cu_seqlens = (cu_seqlens_q, cu_seqlens_k)
            self.config = (
                int(max_seqlen_q), int(max_seqlen_k), float(dropout),
                float(scale), bool(causal), int(wl), int(wr),
                bool(deterministic), bool(return_attn_probs),
            )

        def execute(self, q, k, v):
            cu_q, cu_k = self.cu_seqlens
            max_q, max_k, dropout, scale, causal, wl, wr, _, return_probs = self.config
            saved = _facade._mark_readonly_borrow(q, k, v, cu_q, cu_k)
            try:
                result = low_varlen_fwd(
                    q, k, v, None, cu_q, cu_k, None, None, None, None,
                    max_q, max_k, dropout, scale, False, causal, wl, wr,
                    0.0, return_probs, None)
            finally:
                _facade._restore_readonly_borrow(saved)
            out, softmax_lse, probability, rng_state = result
            out.start_grad()
            self.saved = (q, k, v, out, softmax_lse, rng_state)
            if return_probs:
                return out, softmax_lse, probability
            return out

        def grad(self, grad_out, *unused):
            q, k, v, out, softmax_lse, rng_state = self.saved
            cu_q, cu_k = self.cu_seqlens
            max_q, max_k, dropout, scale, causal, wl, wr, deterministic, _ = self.config
            grad_out = grad_out.contiguous()
            saved = _facade._mark_readonly_borrow(
                grad_out, q, k, v, out, softmax_lse, rng_state, cu_q, cu_k)
            try:
                result = low_varlen_bwd(
                    grad_out, q, k, v, out, softmax_lse,
                    None, None, None, cu_q, cu_k, None, max_q, max_k,
                    dropout, scale, False, causal, wl, wr, 0.0,
                    deterministic, None, rng_state)
            finally:
                _facade._restore_readonly_borrow(saved)
            for gradient in result[:3]:
                gradient._set_first_order_only()
            return result[0], result[1], result[2]

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                        window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                        deterministic=False, return_attn_probs=False, *args, **kwargs):
        _check_args(dropout_p, softcap, alibi_slopes, return_attn_probs)
        if not (_facade._native_supported_dtype(q) and _facade._native_supported_dtype(k) and _facade._native_supported_dtype(v)):
            target = _facade._float32_cast_target()
            if target and _facade._dtype_name(q) == _facade._dtype_name(k) == _facade._dtype_name(v) == "float32":
                q0 = q
                q = _facade._maybe_cast_float32_tensor(q, target)
                k = _facade._maybe_cast_float32_tensor(k, target)
                v = _facade._maybe_cast_float32_tensor(v, target)
                result = flash_attn_func(
                    q, k, v, dropout_p, softmax_scale, causal, window_size,
                    softcap, alibi_slopes, deterministic, return_attn_probs,
                    *args, **kwargs)
                return _cast_result(result, q0.dtype, return_attn_probs)
            return None
        wl, wr = _facade._window_size_pair(kwargs.get("window_size", window_size))
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        # A Function rebinds saved tapes to its original inputs after forward.
        # Normalize before taping: the extension bridge exposes dense strides,
        # while clone() may preserve a Torch input's noncontiguous layout.
        q, k, v = (x.contiguous() if not x.is_contiguous() else x for x in (q, k, v))
        needs_grad = _grad_enabled(q, k, v)
        dropout = float(dropout_p or 0.0)
        _check_dropout_backward(q, dropout, needs_grad)
        from jittor._runtime import step_capture
        if step_capture.active():
            if (not needs_grad and dropout == 0.0 and not return_attn_probs
                    and packed_low_level is not None
                    and hasattr(packed_low_level, '_raw_dense_fwd_address')
                    and _can_graph_dense(q, k, v)):
                return _graph_dense_forward(q, k, v, float(softmax_scale), causal, wl, wr)
            # Opaque extension launches are not lazy graph nodes. A replay
            # would otherwise reuse settled outputs unrelated to new inputs.
            step_capture.refuse('this FlashAttention call requires its Python extension boundary')
        if (packed_low_level is not None and not needs_grad and dropout == 0.0
                and not return_attn_probs):
            # Materialize logical views before exposing pointers to the native
            # bridge, so their physical storage matches the reported strides.
            return packed_low_level.fwd(q, k, v, float(softmax_scale), bool(causal), wl, wr)
        if needs_grad:
            return _DenseNativeAttention(
                dropout, softmax_scale, causal, wl, wr, deterministic,
                return_attn_probs)(q, k, v)
        saved = _facade._mark_readonly_borrow(q, k, v, alibi_slopes)
        try:
            result = low_fwd(q, k, v, None, alibi_slopes, dropout,
                             float(softmax_scale), bool(causal), wl, wr,
                             0.0, bool(return_attn_probs), None)
        finally:
            _facade._restore_readonly_borrow(saved)
        return _facade._flashattn_result(result, return_attn_probs)

    def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None,
                                  causal=False, window_size=(-1, -1), softcap=0.0,
                                  alibi_slopes=None, deterministic=False,
                                  return_attn_probs=False, *args, **kwargs):
        _opaque_capture_boundary()
        if (packed_low_level is not None and _facade._native_supported_dtype(qkv)
                and not _grad_enabled(qkv) and float(dropout_p or 0.0) == 0.0
                and not return_attn_probs):
            _check_args(dropout_p, softcap, alibi_slopes, return_attn_probs)
            wl, wr = _facade._window_size_pair(kwargs.get("window_size", window_size))
            scale = qkv.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
            return packed_low_level.qkvpacked_fwd(qkv, scale, bool(causal), wl, wr)
        if packed_split_enabled:
            split = _facade._split_qkvpacked_cuda(qkv)
            if split is not None:
                return flash_attn_func(split[0], split[1], split[2], dropout_p,
                                       softmax_scale, causal, window_size,
                                       softcap, alibi_slopes, deterministic,
                                       return_attn_probs, *args, **kwargs)
        return flash_attn_func(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2],
                               dropout_p, softmax_scale, causal, window_size,
                               softcap, alibi_slopes, deterministic,
                               return_attn_probs, *args, **kwargs)

    def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None,
                                 causal=False, window_size=(-1, -1), softcap=0.0,
                                 alibi_slopes=None, deterministic=False,
                                 return_attn_probs=False, *args, **kwargs):
        _opaque_capture_boundary()
        if (packed_low_level is not None and _facade._native_supported_dtype(q)
                and _facade._native_supported_dtype(kv) and not _grad_enabled(q, kv)
                and float(dropout_p or 0.0) == 0.0 and not return_attn_probs):
            _check_args(dropout_p, softcap, alibi_slopes, return_attn_probs)
            wl, wr = _facade._window_size_pair(kwargs.get("window_size", window_size))
            scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
            return packed_low_level.kvpacked_fwd(q, kv, scale, bool(causal), wl, wr)
        if packed_split_enabled:
            split = _facade._split_kvpacked_cuda(kv)
            if split is not None:
                return flash_attn_func(q, split[0], split[1],
                                       dropout_p, softmax_scale, causal,
                                       window_size, softcap, alibi_slopes,
                                       deterministic, return_attn_probs,
                                       *args, **kwargs)
        return flash_attn_func(q, kv[:, :, 0], kv[:, :, 1],
                               dropout_p, softmax_scale, causal, window_size,
                               softcap, alibi_slopes, deterministic,
                               return_attn_probs, *args, **kwargs)

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                               max_seqlen_q, max_seqlen_k,
                               dropout_p=0.0, softmax_scale=None, causal=False,
                               window_size=(-1, -1), softcap=0.0,
                               alibi_slopes=None, deterministic=False,
                               return_attn_probs=False, block_table=None,
                               *args, **kwargs):
        _opaque_capture_boundary()
        _check_args(dropout_p, softcap, alibi_slopes, return_attn_probs)
        if not (_facade._native_supported_dtype(q) and _facade._native_supported_dtype(k) and _facade._native_supported_dtype(v)):
            target = _facade._float32_cast_target()
            if target and _facade._dtype_name(q) == _facade._dtype_name(k) == _facade._dtype_name(v) == "float32":
                q0 = q
                q = _facade._maybe_cast_float32_tensor(q, target)
                k = _facade._maybe_cast_float32_tensor(k, target)
                v = _facade._maybe_cast_float32_tensor(v, target)
                result = flash_attn_varlen_func(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    dropout_p, softmax_scale, causal, window_size, softcap,
                    alibi_slopes, deterministic, return_attn_probs, block_table,
                    *args, **kwargs)
                return _cast_result(result, q0.dtype, return_attn_probs)
            return None
        wl, wr = _facade._window_size_pair(kwargs.get("window_size", window_size))
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        seqused_k = kwargs.get("seqused_k", None)
        leftpad_k = kwargs.get("leftpad_k", None)
        q, k, v = (x.contiguous() if not x.is_contiguous() else x for x in (q, k, v))
        cu_seqlens_q = cu_seqlens_q.contiguous()
        cu_seqlens_k = cu_seqlens_k.contiguous()
        needs_grad = _grad_enabled(q, k, v)
        dropout = float(dropout_p or 0.0)
        _check_dropout_backward(q, dropout, needs_grad)
        simple_varlen = (seqused_k is None and leftpad_k is None
                         and block_table is None and alibi_slopes is None)
        if (packed_low_level is not None and simple_varlen and not needs_grad
                and dropout == 0.0 and not return_attn_probs):
            return packed_low_level.varlen_fwd(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                int(max_seqlen_q), int(max_seqlen_k),
                float(softmax_scale), bool(causal), wl, wr)
        if needs_grad:
            if not simple_varlen:
                raise RuntimeError(
                    "flashattn_jittor native varlen backward does not support "
                    "seqused_k, leftpad_k, block_table, or alibi_slopes")
            return _VarlenNativeAttention(
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout, softmax_scale, causal, wl, wr, deterministic,
                return_attn_probs)(q, k, v)
        saved = _facade._mark_readonly_borrow(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            seqused_k, leftpad_k, block_table, alibi_slopes,
        )
        try:
            result = low_varlen_fwd(
                q, k, v, None, cu_seqlens_q, cu_seqlens_k,
                seqused_k, leftpad_k,
                block_table, alibi_slopes, int(max_seqlen_q), int(max_seqlen_k),
                dropout, float(softmax_scale), False, bool(causal),
                wl, wr, 0.0, bool(return_attn_probs), None)
        finally:
            _facade._restore_readonly_borrow(saved)
        return _facade._flashattn_result(result, return_attn_probs)

    def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen,
                                         dropout_p=0.0, softmax_scale=None,
                                         causal=False, window_size=(-1, -1),
                                         softcap=0.0, alibi_slopes=None,
                                         deterministic=False,
                                         return_attn_probs=False, *args, **kwargs):
        _opaque_capture_boundary()
        if (packed_low_level is not None and _facade._native_supported_dtype(qkv)
                and not _grad_enabled(qkv) and float(dropout_p or 0.0) == 0.0
                and not return_attn_probs):
            _check_args(dropout_p, softcap, alibi_slopes, return_attn_probs)
            wl, wr = _facade._window_size_pair(kwargs.get("window_size", window_size))
            head_dim = int(qkv.shape[-1])
            scale = head_dim ** -0.5 if softmax_scale is None else float(softmax_scale)
            padding = (-head_dim) % 8
            if padding:
                # Match upstream FlashAttention: pad Q/K/V but retain the
                # original scale, then discard padded output channels.
                import torch.nn.functional as F
                qkv = F.pad(qkv, (0, padding))
            # The pointer bridge requires physical packed storage, even when
            # an already aligned head did not need padding.
            if not qkv.is_contiguous():
                qkv = qkv.contiguous()
            out = packed_low_level.varlen_qkvpacked_fwd(
                qkv, cu_seqlens, int(max_seqlen), scale, bool(causal), wl, wr)
            return out[..., :head_dim] if padding else out
        head_dim = int(qkv.shape[-1])
        padding = (-head_dim) % 8
        if padding and _grad_enabled(qkv):
            # The native training kernel also requires aligned heads. Pad as
            # ordinary differentiable Torch operations, retaining the scale
            # of the original head and slicing the output back afterward.
            import torch.nn.functional as F
            scale = head_dim ** -0.5 if softmax_scale is None else float(softmax_scale)
            padded = F.pad(qkv, (0, padding))
            q, k, v = (padded[:, i].contiguous() for i in range(3))
            result = flash_attn_varlen_func(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                dropout_p, scale, causal, window_size, softcap, alibi_slopes,
                deterministic, return_attn_probs, *args, **kwargs)
            if return_attn_probs:
                return (result[0][..., :head_dim],) + tuple(result[1:])
            return result[..., :head_dim]
        if packed_split_enabled:
            split = _facade._split_qkvpacked_cuda(qkv)
            if split is not None:
                return flash_attn_varlen_func(
                    split[0], split[1], split[2], cu_seqlens, cu_seqlens,
                    max_seqlen, max_seqlen, dropout_p, softmax_scale, causal,
                    window_size, softcap, alibi_slopes, deterministic,
                    return_attn_probs, *args, **kwargs)
        return flash_attn_varlen_func(qkv[:, 0], qkv[:, 1], qkv[:, 2],
                                      cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                                      dropout_p, softmax_scale, causal, window_size,
                                      softcap, alibi_slopes, deterministic,
                                      return_attn_probs, *args, **kwargs)

    def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k,
                                        max_seqlen_q, max_seqlen_k,
                                        dropout_p=0.0, softmax_scale=None,
                                        causal=False, window_size=(-1, -1),
                                        softcap=0.0, alibi_slopes=None,
                                        deterministic=False,
                                        return_attn_probs=False, *args, **kwargs):
        _opaque_capture_boundary()
        if (packed_low_level is not None and _facade._native_supported_dtype(q)
                and _facade._native_supported_dtype(kv) and not _grad_enabled(q, kv)
                and float(dropout_p or 0.0) == 0.0 and not return_attn_probs):
            _check_args(dropout_p, softcap, alibi_slopes, return_attn_probs)
            wl, wr = _facade._window_size_pair(kwargs.get("window_size", window_size))
            scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
            return packed_low_level.varlen_kvpacked_fwd(
                q, kv, cu_seqlens_q, cu_seqlens_k,
                int(max_seqlen_q), int(max_seqlen_k), scale,
                bool(causal), wl, wr)
        if packed_split_enabled:
            split = _facade._split_kvpacked_cuda(kv)
            if split is not None:
                return flash_attn_varlen_func(
                    q, split[0], split[1], cu_seqlens_q, cu_seqlens_k,
                    max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale,
                    causal, window_size, softcap, alibi_slopes, deterministic,
                    return_attn_probs, *args, **kwargs)
        return flash_attn_varlen_func(q, kv[:, 0], kv[:, 1], cu_seqlens_q,
                                      cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                      dropout_p, softmax_scale, causal, window_size,
                                      softcap, alibi_slopes, deterministic,
                                      return_attn_probs, *args, **kwargs)

    mod.flash_attn_func = flash_attn_func
    mod.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
    mod.flash_attn_kvpacked_func = flash_attn_kvpacked_func
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    mod.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    mod.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
    return mod
