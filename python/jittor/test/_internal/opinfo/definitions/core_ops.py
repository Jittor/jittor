# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Core OpInfos: elementwise unary/binary, basic reductions, matmul, softmax, norms.

The seed of the registry. Other ``definitions/*.py`` modules add their domains; the
aggregator concatenates them all into ``op_db``.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, refs, np, jt, nn, F)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo


def sample_matmul(op_info, device, dtype, requires_grad):
    shapes = [((3, 4), (4, 5)), ((5, 5), (5, 5))]
    return [SampleInput(make_tensor(*a, dtype=dtype, requires_grad=requires_grad, seed=400 + i),
                        make_tensor(*b, dtype=dtype, requires_grad=requires_grad, seed=450 + i))
            for i, (a, b) in enumerate(shapes)]


def sample_bmm(op_info, device, dtype, requires_grad):
    return [SampleInput(make_tensor(2, 3, 4, dtype=dtype, requires_grad=requires_grad, seed=460),
                        make_tensor(2, 4, 5, dtype=dtype, requires_grad=requires_grad, seed=461))]


def sample_softmax(op_info, device, dtype, requires_grad):
    return [SampleInput(make_tensor(3, 5, dtype=dtype, requires_grad=requires_grad, seed=470), dim=-1),
            SampleInput(make_tensor(2, 3, 4, dtype=dtype, requires_grad=requires_grad, seed=471), dim=1)]


def sample_layer_norm(op_info, device, dtype, requires_grad):
    out = []
    for i, (shape, ns) in enumerate([((3, 6), (6,)), ((2, 3, 4), (4,))]):
        C = ns[0]
        out.append(SampleInput(
            make_tensor(*shape, dtype=dtype, requires_grad=requires_grad, seed=480 + i), ns,
            make_tensor(C, dtype=dtype, requires_grad=requires_grad, seed=482 + i),
            make_tensor(C, dtype=dtype, requires_grad=requires_grad, seed=484 + i), eps=1e-5))
    return out


def sample_group_norm(op_info, device, dtype, requires_grad):
    N, C, H, W, G = 2, 6, 4, 4, 3
    return [SampleInput(
        make_tensor(N, C, H, W, dtype=dtype, requires_grad=requires_grad, seed=490), G,
        make_tensor(C, dtype=dtype, requires_grad=requires_grad, seed=491),
        make_tensor(C, dtype=dtype, requires_grad=requires_grad, seed=492), eps=1e-5)]


op_db = [
    # ---- unary elementwise (forward pinned to numpy; backward gradchecked) ----
    UnaryUfuncInfo("exp", ref=np.exp, op=jt.exp),
    UnaryUfuncInfo("log", ref=np.log, domain=(0.2, 4.0), op=jt.log),
    UnaryUfuncInfo("sin", ref=np.sin, op=jt.sin),
    UnaryUfuncInfo("cos", ref=np.cos, op=jt.cos),
    UnaryUfuncInfo("tanh", ref=np.tanh, op=jt.tanh),
    UnaryUfuncInfo("sqrt", ref=np.sqrt, domain=(0.1, 4.0), op=jt.sqrt),
    UnaryUfuncInfo("abs", ref=np.abs, op=jt.abs),
    UnaryUfuncInfo("negative", ref=np.negative, op=lambda x: -x),
    UnaryUfuncInfo("sigmoid", ref=sigmoid_ref, op=jt.sigmoid),
    UnaryUfuncInfo("relu", ref=lambda x: np.maximum(x, 0), op=nn.relu),
    UnaryUfuncInfo("gelu", ref=gelu_ref, op=nn.gelu),
    UnaryUfuncInfo("silu", ref=silu_ref, op=nn.silu),

    # ---- binary elementwise ----
    BinaryUfuncInfo("add", ref=np.add, op=lambda a, b: a + b),
    BinaryUfuncInfo("sub", ref=np.subtract, op=lambda a, b: a - b),
    BinaryUfuncInfo("mul", ref=np.multiply, op=lambda a, b: a * b),
    BinaryUfuncInfo("div", ref=np.divide, op=lambda a, b: a / b),
    BinaryUfuncInfo("maximum", ref=np.maximum, op=jt.maximum),
    BinaryUfuncInfo("minimum", ref=np.minimum, op=jt.minimum),

    # ---- reductions (sweep dim/keepdims -- closes the reduce-backward hole) ----
    ReductionOpInfo("sum", ref=reduce_ref(np.sum), op=jt.sum),
    ReductionOpInfo("mean", ref=reduce_ref(np.mean), op=jt.mean),
    ReductionOpInfo("prod", ref=reduce_ref(np.prod), op=jt.prod, supports_gradgrad=False),

    # ---- matmul family ----
    OpInfo("matmul", op=jt.matmul, ref=np.matmul, sample_inputs_func=sample_matmul),
    OpInfo("bmm", op=jt.matmul, ref=np.matmul, sample_inputs_func=sample_bmm),

    # ---- softmax / log_softmax ----
    OpInfo("softmax", op=nn.softmax, ref=softmax_ref, sample_inputs_func=sample_softmax),
    OpInfo("log_softmax", op=nn.log_softmax, ref=log_softmax_ref, sample_inputs_func=sample_softmax),

    # ---- normalization ----
    # supports_gradgrad=False: jittor's stable norm backward (d4c7927a) is a
    # jt.Function whose backward is not itself differentiable -> no 2nd derivative.
    # Surfaced by gradgradcheck; declared (as torch does) so the gap is recorded.
    OpInfo("layer_norm", op=F.layer_norm, ref=layer_norm_ref,
           sample_inputs_func=sample_layer_norm, supports_gradgrad=False),
    OpInfo("group_norm", op=F.group_norm, ref=group_norm_ref,
           sample_inputs_func=sample_group_norm, supports_gradgrad=False),
]
