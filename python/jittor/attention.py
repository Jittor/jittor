# ***************************************************************
# Copyright (c) 2020 Jittor. Authors:
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
#
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import jittor as jt
from jittor import init, Module, nn
import numpy as np
import math

class MultiheadAttention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        assert dropout==0, "TODO: dropout>0"

        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, ("Self-attention requires query, key and " "value to be of the same size")

        #TODO: quant_noise
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert not add_bias_kv, "TODO: add_bias_kv=True"
        self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False
        
    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            init.xavier_uniform_(self.k_proj.weight)
            init.xavier_uniform_(self.v_proj.weight)
            init.xavier_uniform_(self.q_proj.weight)

        # init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            init.xavier_normal_(self.bias_v)

    def execute(
        self,
        query,
        key = None,
        value = None,
        key_padding_mask = None,
        incremental_state = None,
        need_weights = True,
        static_kv = False,
        attn_mask = None,
        before_softmax = False,
        need_head_weights = False,
    ):
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim
        assert list(query.shape) == [tgt_len, bsz, embed_dim]

        assert incremental_state is None, "TODO: incremental_state is not None"
        saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q = q*self.scaling

        assert self.bias_k is None, "TODO: self.bias_k is not None:"

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        if k is not None:
            k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        if v is not None:
            v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)

        assert saved_state is None, "TODO: saved_state is not None"
        assert k is not None
        src_len = k.shape[1]

        assert key_padding_mask is None, "TODO: key_padding_mask is not None"
        assert not self.add_zero_attn, "TODO: self.add_zero_attn=True"

        attn_weights = nn.bmm(q, k.transpose(0, 2, 1))

        assert list(attn_weights.shape) == [bsz * self.num_heads, tgt_len, src_len]

        assert attn_mask is None, "TODO: attn_mask is not None"
        assert key_padding_mask is None, "TODO: key_padding_mask is not None"
        
        if before_softmax:
            return attn_weights, v
        
        attn_weights_float = nn.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)

        assert v is not None
        attn = nn.bmm(attn_weights, v)
        assert list(attn.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.shape[1] == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(1, 0, 2).view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0, 2, 3)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dims=[0])

        return attn, attn_weights
