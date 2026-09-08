"""Exercise the actual Python owners and CodeOp payload without claiming NPU execution."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[4]
OPS = ROOT / "backends/acl/kernels/ops"


@pytest.fixture
def pipeline(monkeypatch):
    class Tensor:
        def __init__(self, shape, dtype="float32"):
            self.shape, self.dtype = tuple(shape), dtype

        @property
        def ndim(self):
            return len(self.shape)

        def __len__(self):
            return len(self.shape)

        def numel(self):
            import math

            return math.prod(self.shape)

    calls = []
    jt = types.ModuleType("jittor")
    jt.Function = object
    jt.empty = jt.zeros = Tensor

    def code(*args, **kwargs):
        if args:
            kwargs["inputs"] = args[2]
        calls.append(kwargs)
        return kwargs.get("outputs") or [
            Tensor(shape, dtype) for shape, dtype in zip(args[0], args[1])
        ]

    jt.code = code
    monkeypatch.setitem(sys.modules, "jittor", jt)
    compiler = types.ModuleType("jittor.compiler")
    jt.compiler = compiler
    monkeypatch.setitem(sys.modules, compiler.__name__, compiler)
    utils = types.ModuleType("jittor_utils")
    utils.env_or_try_find = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "jittor_utils", utils)
    dtypes = types.ModuleType("jittor._core.dtypes")
    dtypes.dtype_name = str
    monkeypatch.setitem(sys.modules, dtypes.__name__, dtypes)
    package = types.ModuleType("acl_attribute_pipeline")
    package.__path__ = [str(OPS)]
    monkeypatch.setitem(sys.modules, package.__name__, package)

    def load(name):
        qualified = package.__name__ + "." + name
        if qualified in sys.modules:
            return sys.modules[qualified]
        spec = importlib.util.spec_from_file_location(qualified, OPS / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        spec.loader.exec_module(module)
        return module

    # Register imports through monkeypatch so each test restores the full module table.
    for name in ("acl_data", "_attributes", "_code"):
        load(name)
    return load, Tensor, calls


def test_dimensions_are_runtime_data_and_backward_keeps_its_own_values(pipeline):
    load, Tensor, calls = pipeline
    module = load("softmax_op")
    x = Tensor((2, 3), "float16")
    first, second = module.SoftmaxACL(), module.SoftmaxACL()
    first.execute(x, 0)
    second.execute(x, 1)
    assert calls[0]["cuda_src"] == calls[1]["cuda_src"]
    assert calls[0]["data"] != calls[1]["data"]
    assert calls[0]["outputs"][0].dtype == "float16"
    first.grad(Tensor((2, 3)))
    second.grad(Tensor((2, 3)))
    data = load("_attributes").attribute_data
    assert calls[2]["data"] == data("SoftmaxBackward", {"dim": 0})
    assert calls[3]["data"] == data("SoftmaxBackward", {"dim": 1})
    assert all(call["backend"] == "acl" for call in calls)
    assert all("attr->dim" not in call["cuda_src"] for call in calls)


def test_scalar_vector_and_composite_backward_owners_use_the_channel(pipeline):
    load, Tensor, calls = pipeline
    data = load("_attributes").attribute_data
    x = Tensor((2, 3))
    load("triu_op").TriuACL().execute(x, -(1 << 63))
    assert calls[-1]["data"] == data("Triu", {"diagonal": -(1 << 63)})
    flip = load("flip_op").FlipACL()
    flip.execute(x, (1, 0))
    flip.grad(x)
    assert calls[-1]["data"] == calls[-2]["data"] == data("Flip", {"axes": [1, 0]})
    cumulative = load("cumsum_op").CumsumACL()
    cumulative.execute(x, -1)
    cumulative.grad(x)
    assert [call["data"] for call in calls[-3:]] == [
        data("Flip", {"axes": [-1]}),
        data("Cumsum", {"dim": -1}),
        data("Flip", {"axes": [-1]}),
    ]
    assert calls[-2]["inputs"][0] is not x
    gather = load("gather_scatter_op").GatherACL()
    index = Tensor((2, 3), "int64")
    gather.execute(x, 1, index)
    gather.grad(x)
    assert calls[-1]["data"] == data("Scatter", {"axis": 1, "reduction": 1})
    scatter = load("gather_scatter_op").ScatterACL()
    scatter.execute(x, 0, index, x, "mul")
    assert calls[-1]["data"] == data("Scatter", {"axis": 0, "reduction": 2})
    scatter.grad(x)
    assert calls[-1]["data"] == data("Gather", {"dim": 0})


def test_generated_and_runtime_attribute_sources_cannot_be_mixed(pipeline):
    load, Tensor, calls = pipeline
    code = load("_code").acl_code
    kwargs = dict(output_shapes=[(2, 3)], output_dtypes=["float32"], attributes={"dim": 1})
    with pytest.raises(ValueError, match="mutually exclusive"):
        code("Softmax", [Tensor((2, 3))], attr_code="op.run();", **kwargs)
    with pytest.raises(ValueError, match="reserved"):
        code("Softmax", [Tensor((2, 3))], extra_data={"acl_attr.version": 1}, **kwargs)
    assert not calls


def test_encoded_values_reach_real_cpp_attribute_types(pipeline, tmp_path):
    load, _, _ = pipeline
    encode = load("_attributes").attribute_data
    spec = importlib.util.spec_from_file_location(
        "attribute_sdk_stub", ROOT / "agent/skills/acl-host-syntax-check/make_cann_stub.py"
    )
    stubber = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stubber)
    stub = tmp_path / "sdk"
    stubber.build(ROOT / "backends/acl", stub)
    cases = [
        ("Softmax", {"dim": -1}, "dynamic_cast<SoftmaxAttr*>(runner.op_attr.get())->dim == -1"),
        (
            "SoftmaxBackward",
            {"dim": 2},
            "dynamic_cast<SoftmaxAttr*>(runner.op_attr.get())->dim == 2",
        ),
        (
            "Triu",
            {"diagonal": -(1 << 63)},
            "dynamic_cast<TriuAttr*>(runner.op_attr.get())->diagonal == std::numeric_limits<int64_t>::min()",
        ),
        (
            "Flip",
            {"axes": [2, 0]},
            "dynamic_cast<ReduceAttr*>(runner.op_attr.get())->axes == std::vector<int64_t>({2, 0})",
        ),
        ("Cumsum", {"dim": -2}, "dynamic_cast<GatherAttr*>(runner.op_attr.get())->dim == -2"),
        ("Gather", {"dim": 1}, "dynamic_cast<GatherAttr*>(runner.op_attr.get())->dim == 1"),
        (
            "Scatter",
            {"axis": -1, "reduction": 2},
            "dynamic_cast<ScatterAttr*>(runner.op_attr.get())->axis == -1 && "
            "dynamic_cast<ScatterAttr*>(runner.op_attr.get())->reduction == 2",
        ),
    ]
    cases.extend(
        [
            (
                "Conv2dBackward",
                {
                    "convStrides": [2, 1],
                    "convPads": [1, 0],
                    "convDilations": [1, 2],
                    "group": 3,
                    "convOutPads": [0, 0],
                },
                "dynamic_cast<ConvAttr*>(runner.op_attr.get())->group == 3 && dynamic_cast<ConvAttr*>(runner.op_attr.get())->convStrides[0] == 2",
            ),
            (
                "LayerNormBackward",
                {"eps": 0.125, "normalizedShape": [3, 4]},
                "dynamic_cast<LayerNormAttr*>(runner.op_attr.get())->size == 2 && dynamic_cast<LayerNormAttr*>(runner.op_attr.get())->eps == 0.125",
            ),
            (
                "GroupNormBackward",
                {"batch": 2, "channels": 6, "spatialSize": 12, "groups": 3, "eps": 0.25},
                "dynamic_cast<GroupNormAttr*>(runner.op_attr.get())->spatialSize == 12",
            ),
            (
                "AvgpoolBackward",
                {
                    "kernel_size": [3, 2],
                    "poolStrides": [2, 1],
                    "poolPads": [1, 0],
                    "poolDilations": [1, 1],
                    "poolCeil": True,
                    "countIncludePad": False,
                },
                "dynamic_cast<PoolAttr*>(runner.op_attr.get())->poolCeil && !dynamic_cast<PoolAttr*>(runner.op_attr.get())->countIncludePad",
            ),
            (
                "UpsampleNearest2dBackward",
                {"outputSize": [4, 6], "inputSize": [1, 2, 2, 3]},
                "dynamic_cast<UpsampleNearest2dAttr*>(runner.op_attr.get())->inputSize[3] == 3",
            ),
            (
                "StridedSliceAssignV2",
                {"begins": [-(1 << 63)], "ends": [(1 << 63) - 1], "steps": [2], "axes": [0]},
                "dynamic_cast<StrideAttr*>(runner.op_attr.get())->ends[0] == std::numeric_limits<int64_t>::max()",
            ),
            (
                "Range",
                {"start": -(1 << 63), "end": (1 << 63) - 1, "step": 3},
                "dynamic_cast<RangeAttr*>(runner.op_attr.get())->start == std::numeric_limits<int64_t>::min()",
            ),
            (
                "Dropout",
                {"p": 0.25, "train": True, "seed": (1 << 63) - 1, "offset": 2},
                "dynamic_cast<DropoutAttr*>(runner.op_attr.get())->p == 0.25 && dynamic_cast<DropoutAttr*>(runner.op_attr.get())->seed == std::numeric_limits<int64_t>::max()",
            ),
            (
                "MatMul",
                {"mode": 2, "cube_math_type": 1},
                'runner.cube_math_type == 1 && runner.jt_name == "matmul_trans_0"',
            ),
            (
                "Roll",
                {"shifts": [-3, 2], "dims": [0, 1]},
                "runner.shifts[0] == -3 && runner.dims[1] == 1",
            ),
            (
                "IncreFlashAttention",
                {
                    "scale": 0.5,
                    "headNum": 8,
                    "keyValueHeadNum": 2,
                    "inputLayout": "BNSD",
                    "innerPrecise": 0,
                    "blockSize": 16,
                    "hasBlockTable": True,
                    "actualSeqLengths": [10, 12],
                },
                'dynamic_cast<IncreFlashAttentionAttr*>(runner.op_attr.get())->inputLayout == "BNSD" && dynamic_cast<IncreFlashAttentionAttr*>(runner.op_attr.get())->actualSeqLengths[1] == 12',
            ),
        ]
    )
    body = []
    for name, attributes, condition in cases:
        wire = encode(name, attributes)
        entries = ",".join(
            "{" + json.dumps(key) + "," + repr(value) + "}" for key, value in wire.items()
        )
        body.append(
            "{ Runner runner{" + json.dumps(name) + "}; Map data{" + entries + "}; "
            "apply_acl_code_attributes(runner, data); assert(" + condition + "); "
            "assert(!runner.jt_name.empty()); }"
        )
        if name == "StridedSliceAssignV2":
            body.append(
                '{ Runner runner{"StridedSliceAssignV2", "stridedsliceassignv2_grad"}; Map data{'
                + entries + '}; apply_acl_code_attributes(runner, data); '
                'assert(runner.jt_name == "stridedsliceassignv2_grad"); }'
            )
        if name == "Range":
            body.append(
                '{ Runner runner{"Softmax"}; Map data{' + entries + '}; bool rejected=false; '
                'try { apply_acl_code_attributes(runner,data,"acl_attr.","Range"); } '
                'catch (const InternalInvariantError&) { rejected=true; } assert(rejected); }'
            )
    unit = tmp_path / "attributes.cc"
    unit.write_text(
        """
#include "aclops/acl_code_attributes.h"
#include <cassert>
#include <limits>
#include <unordered_map>
using namespace jittor;
// Only the carrier is a stand-in. The decoder, schemas, attribute classes and
// field assignment code above are the exact production definitions.
struct Runner {
    string name, jt_name;
    std::unique_ptr<AclOpAttr> op_attr;
    int cube_math_type = 0;
    vector<int64_t> shifts, dims;
};
using Map = std::unordered_map<string, double>;
int main() {
"""
        + "\n".join(body)
        + "\n}\n"
    )
    executable = tmp_path / "attributes"
    result = subprocess.run(
        [
            os.environ.get("CXX", "g++"),
            "-std=c++14",
            "-pthread",
            "-I" + str(stub),
            "-I" + str(stub / "acl"),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "backends/acl/include"),
            str(unit),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_normalization_preserves_numpy_scalars_without_truncating_axes(pipeline):
    import numpy as np

    load, _, _ = pipeline
    encode = load("_attributes").attribute_data
    assert encode("Softmax", {"dim": np.int64(-1)}) == encode("Softmax", {"dim": -1})
    assert encode("Flip", {"axes": np.array([1, 0], dtype=np.int32)}) == encode(
        "Flip", {"axes": [1, 0]}
    )
    assert encode("RmsNorm", {"eps": np.float32(0.5)}) == encode("RmsNorm", {"eps": 0.5})
    with pytest.raises(load("acl_data").AclDataUserError, match="int64"):
        encode("Softmax", {"dim": np.float64(1.5)})
    with pytest.raises(load("acl_data").AclDataUserError, match="int64"):
        encode("Softmax", {"dim": True})


def test_complete_forward_backward_payloads_are_disjoint(pipeline):
    load, Tensor, calls = pipeline
    norms = load("norms_op")
    x, weight, bias = Tensor((2, 6, 4)), Tensor((6,)), Tensor((6,))
    norms.GroupNormACL(3, 0.125)(x, weight, bias)
    first = calls[-1]
    assert "GroupNormBackward_op" in " ".join(first["data"])
    assert "GroupNorm_op" in " ".join(first["data"])
    assert first["data"]["multi_grad"] == 1
    assert len(first["cuda_grad_src"]) == 1
    norms.LayerNormACL((6, 4), eps=0.125)(x, weight, bias)
    assert "LayerNormBackward_op" in " ".join(calls[-1]["data"])
    load("matmul_op").MatmulACL()(Tensor((2, 3)), Tensor((3, 4)))
    assert len(calls[-1]["cuda_grad_src"]) == 2
    assert "matmul_grad_x1" in " ".join(calls[-1]["data"])
    assert "matmul_grad_x2" in " ".join(calls[-1]["data"])
    load("transpose_op").TransPoseACL()(Tensor((2, 3, 4)), (1, 2, 0))
    assert "transpose_backward" in " ".join(calls[-1]["data"])
    load("upsample_op").UpsampleNearest2dACL()(Tensor((1, 2, 3, 4)), (6, 8))
    assert "UpsampleNearest2dBackward_op" in " ".join(calls[-1]["data"])
    dropout = load("dropout_op").DropoutACL()
    dropout.execute(x, 0.25, True)
    dropout.execute(x, 0.5, False)
    assert calls[-1]["cuda_src"] == calls[-2]["cuda_src"]
    assert calls[-1]["data"] != calls[-2]["data"]
