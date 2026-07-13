"""Runtime patches for running TRELLIS.2 through the Jittor torch shim.

These patches live in Jittor rather than in TRELLIS.2 so the upstream project
and its native extension dependencies can stay unmodified.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from typing import Dict, Tuple


_FALSEY = {"0", "false", "no", "off"}
_CU_SEQLENS_CACHE: Dict[Tuple[str, Tuple[int, ...], str], object] = {}
_DENSE_ATTENTION_MODULE = "trellis2.modules.attention.full_attn"
_SPARSE_ATTENTION_MODULE = "trellis2.modules.sparse.attention.full_attn"
_LAYOUT_INFO_KEY = "_jittor_torch_layout_info"
_CU_SEQLENS_KEY = "_jittor_torch_cu_seqlens_int32"


def _is_falsey(value) -> bool:
    return str(value or "").strip().lower() in _FALSEY


def _cached_cu_seqlens(torch, lengths, device):
    key = (str(device), tuple(int(x) for x in lengths), "int32")
    out = _CU_SEQLENS_CACHE.get(key)
    if out is None:
        values = [0]
        total = 0
        for length in key[1]:
            total += int(length)
            values.append(total)
        out = torch.tensor(values, dtype=torch.int32, device=device)
        _CU_SEQLENS_CACHE[key] = out
    return out


def _cache_get(obj, key):
    cache = getattr(obj, "_cache", None)
    if isinstance(cache, dict):
        return cache.get(key)
    getter = getattr(obj, "get_spatial_cache", None)
    if getter is not None:
        try:
            return getter(key)
        except Exception:
            return None
    return None


def _cache_set(obj, key, value) -> None:
    cache = getattr(obj, "_cache", None)
    if isinstance(cache, dict):
        cache[key] = value
        return
    setter = getattr(obj, "register_spatial_cache", None)
    if setter is not None:
        try:
            setter(key, value)
        except Exception:
            pass


def _layout_info(obj):
    info = _cache_get(obj, _LAYOUT_INFO_KEY)
    if info is not None:
        return info
    lengths = tuple(item.stop - item.start for item in obj.layout)
    max_len = max(lengths) if lengths else 0
    info = (lengths, max_len)
    _cache_set(obj, _LAYOUT_INFO_KEY, info)
    return info


def _cached_varlen_cu_seqlens(torch, obj):
    lengths, max_len = _layout_info(obj)
    device = obj.device
    device_key = str(device)
    cached = _cache_get(obj, _CU_SEQLENS_KEY)
    if cached is not None and cached[0] == device_key:
        return lengths, max_len, cached[1]
    cu = _cached_cu_seqlens(torch, lengths, device)
    _cache_set(obj, _CU_SEQLENS_KEY, (device_key, cu))
    return lengths, max_len, cu


def _dense_lengths(n, length):
    length = int(length)
    return (length,) * int(n), length


def _flash_funcs(mod, names):
    funcs = mod.__dict__.get("_jittor_torch_flash_attn_funcs")
    if funcs is None:
        import flash_attn

        funcs = tuple(getattr(flash_attn, name) for name in names)
        mod.__dict__["_jittor_torch_flash_attn_funcs"] = funcs
    return funcs


def _patch_dense_attention_module(mod) -> bool:
    if getattr(mod, "_jittor_torch_fast_dense_attn", False):
        return True

    try:
        import importlib

        torch = importlib.import_module("torch")
        from trellis2.modules.attention import config
    except Exception:
        return False

    original = getattr(mod, "scaled_dot_product_attention", None)
    if original is None:
        return False

    def scaled_dot_product_attention(*args, **kwargs):
        if config.BACKEND != "flash_attn":
            return original(*args, **kwargs)
        qkvpacked_func, kvpacked_func, attn_func = _flash_funcs(
            mod,
            ("flash_attn_qkvpacked_func", "flash_attn_kvpacked_func", "flash_attn_func"),
        )
        if not kwargs:
            if len(args) == 1:
                return qkvpacked_func(args[0])
            if len(args) == 2:
                return kvpacked_func(args[0], args[1])
            if len(args) == 3:
                return attn_func(args[0], args[1], args[2])
            return original(*args, **kwargs)

        num_all_args = len(args) + len(kwargs)
        if num_all_args == 1 and (args or "qkv" in kwargs):
            return qkvpacked_func(args[0] if args else kwargs["qkv"])
        if num_all_args == 2 and all(key in kwargs for key in ("q", "kv")[len(args):]):
            q = args[0] if len(args) > 0 else kwargs["q"]
            kv = args[1] if len(args) > 1 else kwargs["kv"]
            return kvpacked_func(q, kv)
        if num_all_args == 3 and all(key in kwargs for key in ("q", "k", "v")[len(args):]):
            q = args[0] if len(args) > 0 else kwargs["q"]
            k = args[1] if len(args) > 1 else kwargs["k"]
            v = args[2] if len(args) > 2 else kwargs["v"]
            return attn_func(q, k, v)
        return original(*args, **kwargs)

    scaled_dot_product_attention._jittor_torch_fast_dense_attn = True
    scaled_dot_product_attention._jittor_torch_original = original
    mod.scaled_dot_product_attention = scaled_dot_product_attention

    refs = sys.modules.get("trellis2.modules.attention.modules")
    if refs is not None:
        try:
            refs.scaled_dot_product_attention = scaled_dot_product_attention
        except Exception:
            pass
    return True


def _patch_sparse_attention_module(mod) -> bool:
    if getattr(mod, "_jittor_torch_fast_sparse_attn", False):
        return True

    try:
        import importlib

        torch = importlib.import_module("torch")
        from trellis2.modules.sparse import config
        from trellis2.modules.sparse.basic import VarLenTensor
    except Exception:
        return False

    original = getattr(mod, "sparse_scaled_dot_product_attention", None)
    if original is None:
        return False

    def sparse_scaled_dot_product_attention(*args, **kwargs):
        if config.ATTN != "flash_attn":
            return original(*args, **kwargs)
        if kwargs:
            return original(*args, **kwargs)
        varlen_func, qkvpacked_func, kvpacked_func = _flash_funcs(
            mod,
            (
                "flash_attn_varlen_func",
                "flash_attn_varlen_qkvpacked_func",
                "flash_attn_varlen_kvpacked_func",
            ),
        )
        num_all_args = len(args)
        if num_all_args not in (1, 2, 3):
            return original(*args, **kwargs)

        if num_all_args == 1:
            qkv = args[0]
            if not isinstance(qkv, VarLenTensor):
                return original(*args, **kwargs)
            _, max_q, cu_q = _cached_varlen_cu_seqlens(torch, qkv)
            out = qkvpacked_func(qkv.feats, cu_q, max_q)
            return qkv.replace(out)

        if num_all_args == 2:
            q, kv = args
            if not (
                isinstance(q, VarLenTensor)
                and isinstance(kv, (VarLenTensor, torch.Tensor))
                or isinstance(q, torch.Tensor)
                and isinstance(kv, VarLenTensor)
            ):
                return original(*args, **kwargs)
            if isinstance(q, VarLenTensor):
                s = q
                q_seqlen, max_q, cu_q = _cached_varlen_cu_seqlens(torch, q)
                q_feats = q.feats
            else:
                s = None
                n, l, h, c = q.shape
                q_seqlen, max_q = _dense_lengths(n, l)
                q_feats = q.reshape(n * l, h, c)
                cu_q = _cached_cu_seqlens(torch, q_seqlen, q.device)
            if isinstance(kv, VarLenTensor):
                kv_seqlen, max_kv, cu_kv = _cached_varlen_cu_seqlens(torch, kv)
                kv_feats = kv.feats
            else:
                n, l, _, h, c = kv.shape
                kv_seqlen, max_kv = _dense_lengths(n, l)
                kv_feats = kv.reshape(n * l, 2, h, c)
                cu_kv = _cached_cu_seqlens(torch, kv_seqlen, q.device)
            if q_seqlen == kv_seqlen:
                cu_kv = cu_q
            out = kvpacked_func(q_feats, kv_feats, cu_q, cu_kv, max_q, max_kv)
            return s.replace(out) if s is not None else out.reshape(n, l, h, -1)

        q, k, v = args
        if isinstance(q, VarLenTensor):
            s = q
            q_seqlen, max_q, cu_q = _cached_varlen_cu_seqlens(torch, q)
            q_feats = q.feats
        else:
            s = None
            n, l, h, ci = q.shape
            q_seqlen, max_q = _dense_lengths(n, l)
            q_feats = q.reshape(n * l, h, ci)
            cu_q = _cached_cu_seqlens(torch, q_seqlen, q.device)
        if isinstance(k, VarLenTensor):
            if not isinstance(v, VarLenTensor):
                return original(*args, **kwargs)
            kv_seqlen, max_kv, cu_kv = _cached_varlen_cu_seqlens(torch, k)
            k_feats = k.feats
            v_feats = v.feats
        else:
            n, l, h, ci = k.shape
            kv_seqlen, max_kv = _dense_lengths(n, l)
            k_feats = k.reshape(n * l, h, ci)
            v_feats = v.reshape(n * l, h, v.shape[-1])
            cu_kv = _cached_cu_seqlens(torch, kv_seqlen, q.device)
        if q_seqlen == kv_seqlen:
            cu_kv = cu_q
        out = varlen_func(q_feats, k_feats, v_feats, cu_q, cu_kv, max_q, max_kv)
        return s.replace(out) if s is not None else out.reshape(n, l, h, -1)

    sparse_scaled_dot_product_attention._jittor_torch_fast_sparse_attn = True
    sparse_scaled_dot_product_attention._jittor_torch_original = original
    mod.sparse_scaled_dot_product_attention = sparse_scaled_dot_product_attention

    refs = sys.modules.get("trellis2.modules.sparse.attention.modules")
    if refs is not None:
        try:
            refs.sparse_scaled_dot_product_attention = sparse_scaled_dot_product_attention
        except Exception:
            pass
    return True


def _patch_loaded_dense_attention() -> bool:
    mod = sys.modules.get(_DENSE_ATTENTION_MODULE)
    return mod is not None and _patch_dense_attention_module(mod)


def _patch_loaded_sparse_attention() -> bool:
    mod = sys.modules.get(_SPARSE_ATTENTION_MODULE)
    return mod is not None and _patch_sparse_attention_module(mod)


def install() -> None:
    if _is_falsey(os.environ.get("JITTOR_TRELLIS_RUNTIME_PATCHES")):
        return
    dense_done = _patch_loaded_dense_attention()
    sparse_done = _patch_loaded_sparse_attention()
    if dense_done and sparse_done:
        return
    for finder in sys.meta_path:
        if isinstance(finder, _TrellisRuntimeFinder):
            return
    sys.meta_path.insert(0, _TrellisRuntimeFinder())


class _TrellisRuntimeLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module) -> None:
        self.loader.exec_module(module)
        if module.__name__ == _DENSE_ATTENTION_MODULE:
            _patch_dense_attention_module(module)
        if module.__name__ == _SPARSE_ATTENTION_MODULE:
            _patch_sparse_attention_module(module)


class _TrellisRuntimeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (_DENSE_ATTENTION_MODULE, _SPARSE_ATTENTION_MODULE):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        if not isinstance(spec.loader, _TrellisRuntimeLoader):
            spec.loader = _TrellisRuntimeLoader(spec.loader)
        return spec
