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
from collections.abc import Sequence, Iterable

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
            D = H / N
        elif self.layout == 'SBH':
            SQ, B, H = q.shape
            SKV = k.shape[0]
            N = self.headnum
            D = H / N
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

        self.prefix = self.prefix if self.prefix else [0 for _ in range(B)]
        self.qstart = self.qstart if self.qstart else [0 for _ in range(B)]
        self.kvstart = self.kvstart if self.kvstart else [
            0 for _ in range(B)
        ]

        self.hasRealshift = (not realshift == None)
        self.hasDropmask = (not dropMask == None)
        self.hasPaddingmask = (not paddingMask == None)
        self.hasAttenmask = (not attenMask == None)

        # 待定，目前设为nullptr
        self.realshift = realshift if realshift else jt.zeros(
            B, N, SQ, SKV)
        self.dropMask = dropMask if dropMask else jt.ones(B, N, SQ, SKV)
        self.paddingMask = paddingMask if paddingMask else jt.zeros(
            B, N, SQ, SKV)
        self.attenMask = attenMask if attenMask else jt.zeros(SQ, SKV)

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