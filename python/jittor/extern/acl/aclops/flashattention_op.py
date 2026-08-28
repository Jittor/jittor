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


def flashattention_cmd(name: str,
                       inputs: list,
                       output_dtypes: list = None,
                       output_shapes: list = None,
                       attr_code: str = "",
                       attr_header: str = "",
                       outputs: list = None):
    attr_header = "\nnamespace jittor{" + attr_header + "}\n"

    cuda_header = '''
    #include "acl/aclops/aclops.h"
    '''
    outputs_ = []
    if outputs is not None:
        outputs_ = outputs
    else:
        assert output_dtypes is not None
        assert output_shapes is not None
        assert len(output_dtypes) == len(output_shapes)
        for i in range(len(output_shapes)):
            outputs_.append(jt.empty(output_shapes[i], output_dtypes[i]))
    input_code = ''
    for i in range(len(inputs)):
        input_code += f"op.add(in{i}, true);\n"

    output_code = ''
    for i in range(len(outputs_)):
        output_code += f"op.add(out{i}, false);\n"
    return jt.code(outputs=outputs_,
                   inputs=inputs,
                   cuda_header=attr_header + cuda_header,
                   cuda_src=f"""
   
    // aclop
    {name}OpRunner op;
    {input_code}
    {output_code}
    {attr_code}
    op.run();""")


class FlashAttentionACL(jt.Function):

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

    def execute(
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

        self.q = q
        self.k = k
        self.v = v

        self.prefix = (self.prefix if self.prefix is not None
                       else [0 for _ in range(B)])
        self.qstart = (self.qstart if self.qstart is not None
                       else [0 for _ in range(B)])
        self.kvstart = (self.kvstart if self.kvstart is not None
                        else [0 for _ in range(B)])

        self.hasRealshift = realshift is not None
        self.hasDropmask = dropMask is not None
        self.hasPaddingmask = paddingMask is not None
        self.hasAttenmask = attenMask is not None

        dummy = jt.empty((1,), q.dtype)
        self.realshift = realshift if realshift is not None else dummy
        self.dropMask = dropMask if dropMask is not None else dummy
        self.paddingMask = paddingMask if paddingMask is not None else dummy
        self.attenMask = attenMask if attenMask is not None else dummy

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
        attr->prefix = {{ {", ".join(map(str, self.prefix))} }};
        attr->qStartIdx = {{ {", ".join(map(str, self.qstart))} }};
        attr->kvStartIdx = {{ {", ".join(map(str, self.kvstart))} }};
        attr->hasRealshift = {"true" if self.hasRealshift else "false"};
        attr->hasDropmask = {"true" if self.hasDropmask else "false"};
        attr->hasPaddingmask = {"true" if self.hasPaddingmask else "false"};
        attr->hasAttentmask = {"true" if self.hasAttenmask else "false"};
        op.op_attr.reset(attr);
        """

        inputs = [
            q, k, v, self.realshift, self.dropMask, self.paddingMask,
            self.attenMask
        ]

        result = flashattention_cmd(
            "FlashAttention",
            inputs,
            output_dtypes=["float", "float", q.dtype],
            output_shapes=[output_shape, output_shape, q.shape],
            attr_code=attr_code)

        self.maxout = result[0]
        self.sumout = result[1]
        self.attenout = result[2]

        return self.attenout

    def grad(self, dy):
        attr_code = f"""
        op.jt_name = "flashattentionbackward";
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
        attr->prefix = {{ {", ".join(map(str, self.prefix))} }};
        attr->qStartIdx = {{ {", ".join(map(str, self.qstart))} }};
        attr->kvStartIdx = {{ {", ".join(map(str, self.kvstart))} }};
        attr->hasRealshift = {"true" if self.hasRealshift else "false"};
        attr->hasDropmask = {"true" if self.hasDropmask else "false"};
        attr->hasPaddingmask = {"true" if self.hasPaddingmask else "false"};
        attr->hasAttentmask = {"true" if self.hasAttenmask else "false"};
        op.op_attr.reset(attr);
        """
        inputs = [
            self.q, self.k, self.v, dy, self.realshift, self.dropMask,
            self.paddingMask, self.attenMask, self.maxout, self.sumout,
            self.attenout
        ]

        result = flashattention_cmd(
            "FlashAttentionBackward",
            inputs,
            output_dtypes=[self.q.dtype, self.k.dtype, self.v.dtype],
            output_shapes=[self.q.shape, self.k.shape, self.v.shape],
            attr_code=attr_code)
        return result


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
    """Return fused ACL SDPA for the verified inference subset, else ``None``."""
    if not (compiler.has_acl and jt.flags.use_cuda and jt.flags.use_acl
            and getattr(jt.flags, "no_grad", 0)):
        return None
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
    if str(query.dtype) != "float32":
        return None

    query_heads = int(q_shape[-3])
    key_heads = int(k_shape[-3])
    value_heads = int(v_shape[-3])
    if key_heads <= 0 or key_heads != value_heads:
        return None
    if query_heads != key_heads:
        if not enable_gqa or query_heads % key_heads != 0:
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
    return FlashAttentionACL(
        query_heads, "BNSD", scale=scale_factor,
        psetype=0 if real_shift is not None else 1,
    )(query, key, value, real_shift, None, None, causal_mask)
