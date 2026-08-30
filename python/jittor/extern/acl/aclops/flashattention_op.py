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

    def __init__(self,
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
                 psetype=1):
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
        if self.layout == 'BSH':
            B, SQ, H = q.shape
            SKV = k.shape[1]
            N = self.headnum
            D = H // N
        elif self.layout == 'SBH':
            SQ, B, H = q.shape
            SKV = k.shape[0]
            N = self.headnum
            D = H // N
        elif self.layout == 'BSND':
            B, SQ, N, D = q.shape
            SKV = k.shape[1]
        elif self.layout == 'BNSD':
            B, N, SQ, D = q.shape
            SKV = k.shape[2]
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

        attr_code = f"""
        op.jt_name = "flashattention";
        FlashAttentionAttr *attr = new FlashAttentionAttr();
        attr->scale = {self.scale};
        attr->keepProb = {self.prob};
        attr->preToken = {self.pretokens};
        attr->nextToken = {self.nexttokens};
        attr->headNum = {self.headnum};
        attr->inputLayout = "{self.layout}";
        attr->innerPrecise = {self.innerprecise};
        attr->sparseMode = {self.sparsemode};
        attr->psetype = {self.psetype};
        attr->prefix = {{ {", ".join(map(str, prefix))} }};
        attr->qStartIdx = {{ {", ".join(map(str, qstart))} }};
        attr->kvStartIdx = {{ {", ".join(map(str, kvstart))} }};
        attr->hasRealshift = {"true" if has_realshift else "false"};
        attr->hasDropmask = {"true" if has_dropmask else "false"};
        attr->hasPaddingmask = {"true" if has_paddingmask else "false"};
        attr->hasAttentmask = {"true" if has_attenmask else "false"};
        op.op_attr.reset(attr);
        """
        grad_attr_code = attr_code.replace(
            'op.jt_name = "flashattention";',
            'op.jt_name = "flashattentionbackward";',
        )

        inputs = [
            q, k, v, realshift, dropMask, paddingMask, attenMask
        ]

        result = flashattention_cmd(
            "FlashAttention",
            inputs,
            output_dtypes=["float", "float", q.dtype],
            output_shapes=[output_shape, output_shape, q.shape],
            attr_code=attr_code,
            multi_grad_output=2,
            multi_grad_input_count=3,
            multi_grad_src=f"""
            // aclop
            FlashAttentionBackwardOpRunner op;
            op.add(in0, true);
            op.add(in1, true);
            op.add(in2, true);
            op.add(dout, true);
            op.add(in3, true);
            op.add(in4, true);
            op.add(in5, true);
            op.add(in6, true);
            op.add(pout0, true);
            op.add(pout1, true);
            op.add(pout2, true);
            op.add(out0, false);
            op.add(out1, false);
            op.add(out2, false);
            {grad_attr_code}
            op.run();
            """)
        return result[2]


class IncreFlashAttentionACL(jt.Function):

    def __init__(self, headnum, key_value_headnum, scale,
                 layout="BNSD", innerprecise=0):
        self.headnum = headnum
        self.key_value_headnum = key_value_headnum
        self.scale = scale
        self.layout = layout
        self.innerprecise = innerprecise

    def execute(self, q, k, v):
        attr_code = f"""
        op.jt_name = "increflashattention";
        IncreFlashAttentionAttr *attr = new IncreFlashAttentionAttr();
        attr->scale = {self.scale};
        attr->headNum = {self.headnum};
        attr->keyValueHeadNum = {self.key_value_headnum};
        attr->inputLayout = "{self.layout}";
        attr->innerPrecise = {self.innerprecise};
        op.op_attr.reset(attr);
        """
        result = flashattention_cmd(
            "IncreFlashAttention", [q, k, v],
            output_dtypes=[q.dtype], output_shapes=[q.shape],
            attr_code=attr_code)
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


def scaled_dot_product_attention_acl(
        query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False):
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
    if str(query.dtype) != str(key.dtype) or str(query.dtype) != str(value.dtype):
        return None
    if str(query.dtype) not in ("float32", "bfloat16"):
        return None

    query_heads = int(q_shape[-3])
    key_heads = int(k_shape[-3])
    value_heads = int(v_shape[-3])
    if key_heads <= 0 or key_heads != value_heads:
        return None
    if query_heads != key_heads:
        if not enable_gqa or query_heads % key_heads != 0:
            return None
    if training and (
            str(query.dtype) != "float32" or attn_mask is not None
            or is_causal or query_heads != key_heads):
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
        mask_dtype = str(attn_mask.dtype)
        if mask_dtype != "float32":
            return None
        mask_shape = tuple(attn_mask.shape)
        if len(mask_shape) == 2:
            mask_shape = (1, 1) + mask_shape
            real_shift = attn_mask.reshape(mask_shape)
        elif len(mask_shape) == 4:
            real_shift = attn_mask
        else:
            return None
        target_shape = (
            int(q_shape[0]), query_heads, query_length, source_length)
        if any(actual not in (1, expected)
               for actual, expected in zip(mask_shape, target_shape)):
            return None
        if mask_shape != target_shape:
            real_shift = real_shift.broadcast(target_shape)

    causal_mask = None
    if is_causal:
        if query_length != source_length:
            return None
        if query_length > 1:
            causal_mask = _causal_mask(query_length, source_length)
    scale_factor = (1.0 / math.sqrt(head_dim) if scale is None
                    else float(scale))
    if (str(query.dtype) == "bfloat16" and query_length == 1
            and attn_mask is None and not is_causal):
        scaled_dot_product_attention_acl.backend_name = \
            "acl_incre_flash_attention_v4"
        return IncreFlashAttentionACL(
            query_heads, key_heads, scale_factor)(query, key, value)

    scaled_dot_product_attention_acl.backend_name = \
        "acl_flash_attention_score_v2"
    return FlashAttentionACL(
        query_heads, "BNSD", scale=scale_factor,
        psetype=0 if real_shift is not None else 1,
    )(query, key, value, real_shift, None, None, causal_mask)


scaled_dot_product_attention_acl.backend_name = "acl_flash_attention_score_v2"
