"""High-value public operators need OpInfo or an explicit stronger test route."""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS = REPO_ROOT / "tests" / "opinfo" / "definitions"
OPINFO_CONSTRUCTORS = {
    "OpInfo", "UnaryUfuncInfo", "BinaryUfuncInfo", "ReductionOpInfo",
}

REQUIRED_PUBLIC_OPERATORS = {
    "setitem",
    "index_put",
    "nonzero",
    "unique",
    "bincount",
    "einsum",
    "ctc_loss",
    "rms_norm",
    "rotary_embedding",
    "paged_attention",
    "fused_moe",
    "conv_transpose3d",
}

ALTERNATIVE_COVERAGE = {
    "setitem": {
        "nodeid": "tests/core/test_setitem_core.py::TestSetitemOverwrite::test_slice",
        "reason": "setitem mutates and returns None; its dedicated battery covers forward and gradients",
    },
    "index_put": {
        "nodeid": "tests/compat/torch/test_torch_compat.py::test_torch_compat",
        "reason": "index_put is a Torch-shim alias with mutation and duplicate-index accumulation contracts",
    },
    "bincount": {
        "nodeid": (
            "tests/compat/torch/test_torch_compat_reduce_shape.py::"
            "TestVarStd::test_bincount_argwhere_segment_reduce"
        ),
        "reason": "bincount is a Torch-shim API and its output-length semantics have a direct NumPy oracle",
    },
    "ctc_loss": {
        "nodeid": "tests/core/test_misc_op.py::TestPad::test_ctc_loss",
        "reason": "CTC uses ragged lengths and a dynamic-programming oracle outside the generic OpInfo shape",
    },
    "rotary_embedding": {
        "nodeid": (
            "tests/nn/test_serving_ops.py::TestRotaryEmbedding::"
            "test_neox_style_rotates_the_two_halves"
        ),
        "reason": "the plan's rope spelling is the public rotary_embedding API with a written NumPy oracle",
    },
    "paged_attention": {
        "nodeid": "tests/nn/test_paged_attention.py::TestPagedAttention::test_single_request_prefill",
        "reason": "paged KV-cache layout needs a multi-input request fixture and written attention oracle",
    },
    "fused_moe": {
        "nodeid": "tests/nn/test_fused_moe.py::TestFusedMoE::test_many_tokens_dispatch_per_expert",
        "reason": "expert routing needs structured weights and a token-by-token independent oracle",
    },
    "conv_transpose3d": {
        "nodeid": "tests/backends/cuda/test_cudnn_op.py::TestCudnnConvOp::test_conv_transpose3d",
        "reason": "the maintained test compares CUDA/cuDNN forward and gradients with the native path",
    },
}


def _opinfo_names():
    names = set()
    for path in DEFINITIONS.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if isinstance(node.func, ast.Name):
                constructor = node.func.id
            elif isinstance(node.func, ast.Attribute):
                constructor = node.func.attr
            else:
                continue
            if constructor not in OPINFO_CONSTRUCTORS:
                continue
            if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                names.add(node.args[0].value)
    return names


def _assert_nodeid_exists(nodeid):
    parts = nodeid.split("::")
    assert len(parts) in (2, 3), "unsupported nodeid shape: {}".format(nodeid)
    path = REPO_ROOT / parts[0]
    assert path.is_file(), "coverage node file is missing: {}".format(nodeid)
    body = ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body
    for index, name in enumerate(parts[1:]):
        kinds = (ast.ClassDef,) if index == 0 and len(parts) == 3 else (
            ast.FunctionDef, ast.AsyncFunctionDef)
        node = next((item for item in body
                     if isinstance(item, kinds) and item.name == name), None)
        assert node is not None, "coverage node is missing: {}".format(nodeid)
        body = node.body


def test_high_value_public_operators_have_opinfo_or_a_justified_test_route():
    opinfo = _opinfo_names()
    alternatives = set(ALTERNATIVE_COVERAGE)
    assert alternatives <= REQUIRED_PUBLIC_OPERATORS
    assert alternatives.isdisjoint(opinfo), "remove alternatives after adding their OpInfo"

    missing = REQUIRED_PUBLIC_OPERATORS - opinfo
    assert missing == alternatives, (
        "public operators without OpInfo or a justified test route: {}".format(
            sorted(missing - alternatives))
    )
    for name, coverage in ALTERNATIVE_COVERAGE.items():
        assert coverage["reason"].strip(), "{} needs a reason".format(name)
        _assert_nodeid_exists(coverage["nodeid"])
