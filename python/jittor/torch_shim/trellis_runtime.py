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
_SPARSE_ATTENTION_MODULE = "trellis2.modules.sparse.attention.full_attn"


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
        arg_names_dict = {
            1: ["qkv"],
            2: ["q", "kv"],
            3: ["q", "k", "v"],
        }
        num_all_args = len(args) + len(kwargs)
        if config.ATTN != "flash_attn" or num_all_args not in arg_names_dict:
            return original(*args, **kwargs)
        for key in arg_names_dict[num_all_args][len(args):]:
            if key not in kwargs:
                return original(*args, **kwargs)

        if "flash_attn" not in mod.__dict__:
            import flash_attn

            mod.__dict__["flash_attn"] = flash_attn
        flash_attn = mod.__dict__["flash_attn"]

        if num_all_args == 1:
            qkv = args[0] if args else kwargs["qkv"]
            if not isinstance(qkv, VarLenTensor):
                return original(*args, **kwargs)
            q_seqlen = [s.stop - s.start for s in qkv.layout]
            cu_q = _cached_cu_seqlens(torch, q_seqlen, qkv.device)
            out = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.feats, cu_q, max(q_seqlen)
            )
            return qkv.replace(out)

        if num_all_args == 2:
            q = args[0] if len(args) > 0 else kwargs["q"]
            kv = args[1] if len(args) > 1 else kwargs["kv"]
            if not (
                isinstance(q, VarLenTensor)
                and isinstance(kv, (VarLenTensor, torch.Tensor))
                or isinstance(q, torch.Tensor)
                and isinstance(kv, VarLenTensor)
            ):
                return original(*args, **kwargs)
            if isinstance(q, VarLenTensor):
                s = q
                q_seqlen = [item.stop - item.start for item in q.layout]
                q_feats = q.feats
            else:
                s = None
                n, l, h, c = q.shape
                q_seqlen = [l] * n
                q_feats = q.reshape(n * l, h, c)
            if isinstance(kv, VarLenTensor):
                kv_seqlen = [item.stop - item.start for item in kv.layout]
                kv_feats = kv.feats
            else:
                n, l, _, h, c = kv.shape
                kv_seqlen = [l] * n
                kv_feats = kv.reshape(n * l, 2, h, c)
            cu_q = _cached_cu_seqlens(torch, q_seqlen, q.device)
            cu_kv = cu_q if q_seqlen == kv_seqlen else _cached_cu_seqlens(torch, kv_seqlen, q.device)
            out = flash_attn.flash_attn_varlen_kvpacked_func(
                q_feats, kv_feats, cu_q, cu_kv, max(q_seqlen), max(kv_seqlen)
            )
            return s.replace(out) if s is not None else out.reshape(n, l, h, -1)

        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
        if isinstance(q, VarLenTensor):
            s = q
            q_seqlen = [item.stop - item.start for item in q.layout]
            q_feats = q.feats
        else:
            s = None
            n, l, h, ci = q.shape
            q_seqlen = [l] * n
            q_feats = q.reshape(n * l, h, ci)
        if isinstance(k, VarLenTensor):
            if not isinstance(v, VarLenTensor):
                return original(*args, **kwargs)
            kv_seqlen = [item.stop - item.start for item in k.layout]
            k_feats = k.feats
            v_feats = v.feats
        else:
            n, l, h, ci = k.shape
            kv_seqlen = [l] * n
            k_feats = k.reshape(n * l, h, ci)
            v_feats = v.reshape(n * l, h, v.shape[-1])
        cu_q = _cached_cu_seqlens(torch, q_seqlen, q.device)
        cu_kv = cu_q if q_seqlen == kv_seqlen else _cached_cu_seqlens(torch, kv_seqlen, q.device)
        out = flash_attn.flash_attn_varlen_func(
            q_feats, k_feats, v_feats, cu_q, cu_kv, max(q_seqlen), max(kv_seqlen)
        )
        return s.replace(out) if s is not None else out.reshape(n, l, h, -1)

    sparse_scaled_dot_product_attention._jittor_torch_fast_sparse_attn = True
    sparse_scaled_dot_product_attention._jittor_torch_original = original
    mod.sparse_scaled_dot_product_attention = sparse_scaled_dot_product_attention
    return True


def _patch_loaded_sparse_attention() -> bool:
    mod = sys.modules.get(_SPARSE_ATTENTION_MODULE)
    return mod is not None and _patch_sparse_attention_module(mod)


def install() -> None:
    if _is_falsey(os.environ.get("JITTOR_TRELLIS_RUNTIME_PATCHES")):
        return
    if _patch_loaded_sparse_attention():
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
        if module.__name__ == _SPARSE_ATTENTION_MODULE:
            _patch_sparse_attention_module(module)


class _TrellisRuntimeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _SPARSE_ATTENTION_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        if not isinstance(spec.loader, _TrellisRuntimeLoader):
            spec.loader = _TrellisRuntimeLoader(spec.loader)
        return spec
