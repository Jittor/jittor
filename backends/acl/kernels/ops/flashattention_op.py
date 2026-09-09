from ._code import code_with_attributes
from ._attributes import attribute_program, code_program, runner_for_alias
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
import os
from jittor_utils import env_or_try_find
import jittor_utils
import ctypes
import glob
import jittor.compiler as compiler
import jittor as jt
import math
import numpy as np

from typing import Union
from collections import OrderedDict
from collections.abc import Sequence, Iterable


_causal_mask_cache = OrderedDict()
_causal_mask_cache_limit = 16


from ._code import acl_code as flashattention_cmd


class FlashAttentionACL:
    def __init__(
        self,
        headnum,
        layout="BNSD",
        prefix=None,
        qstart=None,
        kvstart=None,
        scale=1.0,
        prob=1.0,
        pretokens=2147483647,
        nexttokens=2147483647,
        innerprecise=0,
        sparsemode=0,
        psetype=1,
    ):
        self.headnum = headnum
        self.layout = layout
        self.scale = scale
        self.prob = prob
        self.pretokens = pretokens
        self.nexttokens = nexttokens
        self.innerprecise = innerprecise
        self.sparsemode = sparsemode
        self.psetype = psetype
        self.prefix = prefix
        self.qstart = qstart
        self.kvstart = kvstart

    def __call__(
        self,
        q,
        k,
        v,
        realshift=None,
        dropMask=None,
        paddingMask=None,
        attenMask=None,
    ):
        if self.layout == "BSH":
            B, SQ, H = q.shape
            N = self.headnum
            D = H // N
        elif self.layout == "SBH":
            SQ, B, H = q.shape
            SKV = k.shape[0]
            N = self.headnum
            D = H // N
        elif self.layout == "BSND":
            B, SQ, N, D = q.shape
            SKV = k.shape[1]
        elif self.layout == "BNSD":
            B, N, SQ, D = q.shape
        else:
            raise ValueError(f"got invalid input layout {self.layout}")

        output_shape = (B, N, SQ, 8)

        prefix = self.prefix if self.prefix is not None else [0 for _ in range(B)]
        qstart = self.qstart if self.qstart is not None else [0 for _ in range(B)]
        kvstart = self.kvstart if self.kvstart is not None else [0 for _ in range(B)]
        has_realshift = realshift is not None
        has_dropmask = dropMask is not None
        has_paddingmask = paddingMask is not None
        has_attenmask = attenMask is not None

        dummy = jt.empty((1,), q.dtype)
        realshift = realshift if realshift is not None else dummy
        dropMask = dropMask if dropMask is not None else dummy
        paddingMask = paddingMask if paddingMask is not None else dummy
        attenMask = attenMask if attenMask is not None else dummy

        attributes = {
            "scale": self.scale,
            "keepProb": self.prob,
            "preToken": self.pretokens,
            "nextToken": self.nexttokens,
            "headNum": self.headnum,
            "inputLayout": self.layout,
            "innerPrecise": self.innerprecise,
            "sparseMode": self.sparsemode,
            "psetype": self.psetype,
            "prefix": list(prefix),
            "qStartIdx": list(qstart),
            "kvStartIdx": list(kvstart),
            "hasRealshift": bool(has_realshift),
            "hasDropmask": bool(has_dropmask),
            "hasPaddingmask": bool(has_paddingmask),
            "hasAttentmask": bool(has_attenmask),
        }

        inputs = [q, k, v, realshift, dropMask, paddingMask, attenMask]

        result = flashattention_cmd(
            "FlashAttention",
            inputs,
            output_dtypes=["float", "float", q.dtype],
            output_shapes=[output_shape, output_shape, q.shape],
            attributes=attributes,
            multi_grad_output=2,
            multi_grad_input_count=3,
            multi_grad_src=code_program(
                [
                    "\n            // aclop\n            FlashAttentionBackwardOpRunner op;\n            op.add(in0, true);\n            op.add(in1, true);\n            op.add(in2, true);\n            op.add(dout, true);\n            op.add(in3, true);\n            op.add(in4, true);\n            op.add(in5, true);\n            op.add(in6, true);\n            op.add(pout0, true);\n            op.add(pout1, true);\n            op.add(pout2, true);\n            op.add(out0, false);\n            op.add(out1, false);\n            op.add(out2, false);\n            ",
                    "\n            op.run();\n            ",
                ]
            ),
            multi_grad_attributes=attributes,
        )
        return result[2]


class IncreFlashAttentionACL(jt.Function):
    def __init__(self, headnum, key_value_headnum, scale, layout="BNSD", innerprecise=0):
        self.headnum = headnum
        self.key_value_headnum = key_value_headnum
        self.scale = scale
        self.layout = layout
        self.innerprecise = innerprecise

    def execute(self, q, k, v):
        attr_code = code_program(
            [
                '\n        op.jt_name = "increflashattention";\n        ',
                attribute_program(
                    "IncreFlashAttention",
                    {
                        "scale": self.scale,
                        "headNum": self.headnum,
                        "keyValueHeadNum": self.key_value_headnum,
                        "inputLayout": self.layout,
                        "innerPrecise": self.innerprecise,
                    },
                    variable="op",
                ),
                "\n        ",
            ]
        )
        result = flashattention_cmd(
            "IncreFlashAttention",
            [q, k, v],
            output_dtypes=[q.dtype],
            output_shapes=[q.shape],
            attr_code=attr_code,
        )
        return result[0]


class PagedIncreFlashAttentionACL(jt.Function):
    def __init__(
        self,
        headnum,
        key_value_headnum,
        block_size,
        scale,
        actual_seq_lengths,
        layout="BNSD",
        innerprecise=0,
    ):
        self.headnum = int(headnum)
        self.key_value_headnum = int(key_value_headnum)
        self.block_size = int(block_size)
        self.scale = float(scale)
        self.actual_seq_lengths = [int(length) for length in actual_seq_lengths]
        self.layout = str(layout)
        self.innerprecise = int(innerprecise)

    def execute(self, q, kv_cache, block_table):
        attr_code = attribute_program(
            "IncreFlashAttention",
            {
                "scale": self.scale,
                "headNum": self.headnum,
                "keyValueHeadNum": self.key_value_headnum,
                "inputLayout": self.layout,
                "innerPrecise": self.innerprecise,
                "blockSize": self.block_size,
                "hasBlockTable": True,
                "actualSeqLengths": self.actual_seq_lengths,
            },
        )
        result = flashattention_cmd(
            "IncreFlashAttention",
            [q, kv_cache, block_table],
            output_dtypes=[q.dtype],
            output_shapes=[q.shape],
            attr_code=attr_code,
        )
        return result[0]


class KVCacheMemcpyACL(jt.Function):
    def __init__(self, block_size, slots):
        self.block_size = int(block_size)
        self.slots = [int(slot) for slot in slots]

    def execute(self, key, value, kv_cache):
        attr_code = code_program(
            [
                '\n        op.jt_name = "kv_cache_memcpy";\n        ',
                attribute_program(
                    "KVCacheMemcpy",
                    {"blockSize": self.block_size, "slots": list(self.slots)},
                    variable="op",
                ),
                "\n        ",
            ]
        )
        result = flashattention_cmd(
            "KVCacheMemcpy", [key, value], outputs=[kv_cache], attr_code=attr_code
        )
        return result[0]


def _causal_mask(query_length, source_length):
    key = (int(query_length), int(source_length))
    cached = _causal_mask_cache.get(key)
    if cached is not None:
        _causal_mask_cache.move_to_end(key)
        return cached
    mask = jt.array(np.triu(np.ones(key, dtype=np.bool_), 1))
    _causal_mask_cache[key] = mask
    if len(_causal_mask_cache) > _causal_mask_cache_limit:
        _causal_mask_cache.popitem(last=False)
    return mask


def _compressed_causal_mask():
    # CANN sparse mode 2 consumes this fixed-size optimized causal mask.
    return _causal_mask(2048, 2048)


def scaled_dot_product_attention_acl(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
):
    """Return fused ACL SDPA for a verified inference/training subset."""
    if not (compiler.has_acl and jt.flags.use_cuda and jt.flags.use_acl):
        return None
    training = not getattr(jt.flags, "no_grad", 0)
    if float(dropout_p or 0.0) != 0.0:
        return None
    if not all(isinstance(tensor, jt.Var) for tensor in (query, key, value)):
        return None

    q_shape = tuple(query.shape)
    k_shape = tuple(key.shape)
    v_shape = tuple(value.shape)
    if not (len(q_shape) == len(k_shape) == len(v_shape) == 4):
        return None
    if q_shape[0] != k_shape[0] or q_shape[0] != v_shape[0]:
        return None
    if k_shape[-2] != v_shape[-2]:
        return None
    if q_shape[-1] != k_shape[-1] or q_shape[-1] != v_shape[-1]:
        return None
    if _jittor_dtype_name(query.dtype) != _jittor_dtype_name(key.dtype) or _jittor_dtype_name(
        query.dtype
    ) != _jittor_dtype_name(value.dtype):
        return None
    if _jittor_dtype_name(query.dtype) not in ("float32", "bfloat16"):
        return None

    query_heads = int(q_shape[-3])
    key_heads = int(k_shape[-3])
    value_heads = int(v_shape[-3])
    if key_heads <= 0 or key_heads != value_heads:
        return None
    if query_heads != key_heads:
        if not enable_gqa or query_heads % key_heads != 0:
            return None
    if training and (_jittor_dtype_name(query.dtype) != "float32" or query_heads != key_heads):
        return None
    head_dim = int(q_shape[-1])
    if head_dim <= 0 or head_dim > 256 or head_dim % 8 != 0:
        return None

    query_length = int(q_shape[-2])
    source_length = int(k_shape[-2])
    if query_length <= 0 or source_length <= 0:
        return None

    real_shift = None
    if attn_mask is not None:
        if is_causal or not isinstance(attn_mask, jt.Var):
            return None
        if training and not attn_mask.is_stop_grad():
            return None
        mask_dtype = _jittor_dtype_name(attn_mask.dtype)
        if _jittor_dtype_name(mask_dtype) != "float32":
            return None
        mask_shape = tuple(attn_mask.shape)
        if len(mask_shape) == 2:
            mask_shape = (1, 1) + mask_shape
            real_shift = attn_mask.reshape(mask_shape)
        elif len(mask_shape) == 4:
            real_shift = attn_mask
        else:
            return None
        target_shape = (int(q_shape[0]), query_heads, query_length, source_length)
        if any(actual not in (1, expected) for actual, expected in zip(mask_shape, target_shape)):
            return None
        if mask_shape != target_shape:
            real_shift = real_shift.broadcast(target_shape)

    causal_mask = None
    sparse_mode = 0
    if is_causal:
        if query_length != source_length:
            return None
        if query_length > 1:
            causal_mask = _compressed_causal_mask()
            sparse_mode = 2
    scale_factor = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)
    if (
        _jittor_dtype_name(query.dtype) == "bfloat16"
        and query_length == 1
        and attn_mask is None
        and not is_causal
    ):
        scaled_dot_product_attention_acl.backend_name = "acl_incre_flash_attention_v4"
        return IncreFlashAttentionACL(query_heads, key_heads, scale_factor)(query, key, value)

    scaled_dot_product_attention_acl.backend_name = "acl_flash_attention_score_v2"
    return FlashAttentionACL(
        query_heads,
        "BNSD",
        scale=scale_factor,
        sparsemode=sparse_mode,
        psetype=0 if real_shift is not None else 1,
    )(query, key, value, real_shift, None, None, causal_mask)


scaled_dot_product_attention_acl.backend_name = "acl_flash_attention_score_v2"
