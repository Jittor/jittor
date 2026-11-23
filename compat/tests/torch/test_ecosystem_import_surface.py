"""The import-time API surface torchao, transformers and peft touch."""
import importlib
import typing
from types import SimpleNamespace

import pytest
import torch


def test_optimizer_required_is_a_module_level_sentinel():
    from torch.optim.optimizer import Optimizer, required

    assert required is not None
    assert required is not Optimizer


def test_triton_ops_meta_reports_no_tuned_entry():
    from torch.sparse import _triton_ops_meta, SparseSemiStructuredTensor
    from torch.sparse._triton_ops_meta import get_meta, minimize, update

    assert torch.sparse is importlib.import_module("torch.sparse")
    assert torch.sparse._triton_ops_meta is _triton_ops_meta
    assert _triton_ops_meta is importlib.import_module("torch.sparse._triton_ops_meta")
    assert torch.sparse.SparseSemiStructuredTensor is SparseSemiStructuredTensor
    assert get_meta("bsr_dense_addmm", (1024, 1024)) is None
    assert update("bsr_dense_addmm", (1024, 1024), (0, "float32", 0.5), {}) is None
    with pytest.raises(NotImplementedError):
        minimize("bsr_dense_addmm", (1024,), (0, "float32", 0.5), [{}],
                 None, None, None, None)


def test_library_impl_accepts_the_torch_2_6_call_form():
    library = torch.library.Library("impl_form_probe", "FRAGMENT")
    library.define("double(Tensor x) -> Tensor")

    @torch.library.impl(library, "double", "CPU")
    def double(x):
        return x * 2

    result = torch.ops.impl_form_probe.double(torch.tensor([1.0, 2.0], device="cpu"))
    assert result.tolist() == [2.0, 4.0]


def test_infer_schema_accepts_the_optional_scalar_union():
    from torch._library.infer_schema import infer_schema

    def op(x: torch.Tensor,
           qmin: typing.Union[int, float, bool, None] = None) -> torch.Tensor:
        return x

    schema = infer_schema(op, mutates_args={})
    assert isinstance(schema, str)
    assert schema.endswith("-> Tensor")
    assert "Scalar?" in schema


def test_signed_sub_byte_dtypes_exist():
    assert torch.int1 is not torch.int2
    assert torch.int7 is not torch.uint7


def test_pytree_register_constant_returns_the_class():
    class Constant:
        pass

    assert torch.utils._pytree.register_constant(Constant) is Constant


def test_prims_common_layout_helpers():
    from torch._prims_common import make_contiguous_strides_for, suggest_memory_format

    tensor = torch.zeros((2, 3, 4, 5))
    assert suggest_memory_format(tensor) == torch.contiguous_format
    assert make_contiguous_strides_for((2, 3)) == (3, 1)


def test_metadata_probes_propagate_runtime_errors():
    from torch._prims_common import suggest_memory_format

    class FailedMetadata:
        @property
        def shape(self):
            raise RuntimeError("tensor metadata read failed")

        @property
        def dtype(self):
            raise RuntimeError("tensor metadata read failed")

    invalid = SimpleNamespace()
    assert suggest_memory_format(invalid) == torch.contiguous_format
    assert not torch._has_compatible_shallow_copy_type(invalid, invalid)

    tensor = FailedMetadata()
    with pytest.raises(RuntimeError, match="tensor metadata read failed"):
        suggest_memory_format(tensor)
    with pytest.raises(RuntimeError, match="tensor metadata read failed"):
        torch._has_compatible_shallow_copy_type(tensor, tensor)


def test_functional_collectives_are_identity_in_one_process():
    from torch.distributed._functional_collectives import all_reduce, wait_tensor

    tensor = torch.tensor([1.0, 2.0])
    assert all_reduce(tensor) is tensor
    assert wait_tensor(tensor) is tensor


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_functional_collectives_copy_single_rank_buffers(device):
    from torch.distributed import _functional_collectives as collectives

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    source = torch.tensor([1.0, 2.0, 3.0], device=device)
    stacked = torch.zeros((1, 3), device=device)
    reduced = torch.zeros_like(source)
    assert collectives.all_gather_into_tensor(stacked, source) is stacked
    assert collectives.reduce_scatter_tensor(reduced, stacked) is reduced
    assert reduced.tolist() == [1.0, 2.0, 3.0]
    assert collectives.wait_tensor(collectives.AsyncCollectiveTensor(reduced)) is reduced
    if device == "cuda":
        assert stacked.location() == reduced.location() == "device"


def test_functional_collectives_refuse_multi_rank_requests_without_mutation(monkeypatch):
    from jittor.compat import functional_collectives as owner
    from torch.distributed import _functional_collectives as collectives

    source = torch.tensor([1.0, 2.0, 3.0], device="cpu")
    output = torch.zeros_like(source)
    ops = torch.ops._c10d_functional
    assert ops.all_gather_into_tensor.default(source, 1, "world") is source
    with pytest.raises(NotImplementedError, match="single-process.*group_size=2"):
        ops.all_gather_into_tensor.default(source, 2, "world")
    with pytest.raises(NotImplementedError, match="single-process.*group_size=2"):
        ops.reduce_scatter_tensor.default(source, "sum", 2, "world")
    legacy = torch.ops.c10d_functional
    assert legacy.all_gather_into_tensor.default(source, "", [0], 1) is source
    assert legacy.wait_tensor.default(source) is source
    assert ops.wait_tensor.default(source) is source
    with pytest.raises(NotImplementedError, match="single-process.*group_size=2"):
        legacy.all_gather_into_tensor.default(source, "", [0, 1], 2)
    with pytest.raises(NotImplementedError, match="c10d.scatter_ is not supported"):
        torch.ops.c10d.scatter_.default(output, source)
    two_rank_group = SimpleNamespace(size=lambda: 2)
    with pytest.raises(NotImplementedError, match="single-process.*group_size=2"):
        collectives.all_gather_into_tensor(output, source, two_rank_group)
    assert output.tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="output shape"):
        collectives.all_gather_into_tensor(torch.zeros((2, 3), device="cpu"), source)
    monkeypatch.setattr(owner, "_world_size", lambda: 2)
    with pytest.raises(NotImplementedError, match="single-process.*world_size=2"):
        collectives.all_reduce(source)
    with pytest.raises(NotImplementedError, match="single-process.*world_size=2"):
        ops.all_gather_into_tensor.default(source, 1, "world")


@pytest.mark.parametrize("namespace", ["_c10d_functional", "c10d", "c10d_functional"])
def test_unknown_collective_has_no_dispatch_entry(namespace):
    with pytest.raises(AttributeError, match="not_a_registered_collective"):
        getattr(torch.ops, namespace).not_a_registered_collective


def test_onnx_and_nn_import_helpers_use_their_existing_owners():
    from jittor.compat.torch.installers import compiler
    from jittor.ops.tuples import _pair
    from torch.nn.modules.batchnorm import BatchNorm2d
    from torch.nn.modules.instancenorm import InstanceNorm2d
    from torch.nn.modules.utils import _pair as published_pair
    from torch.onnx.symbolic_helper import parse_args

    assert parse_args is compiler._api_onnx_parse_args
    assert torch.onnx.register_custom_op_symbolic is compiler._api_onnx_register_custom_op_symbolic
    assert published_pair is _pair
    assert BatchNorm2d is torch.nn.BatchNorm2d
    assert InstanceNorm2d is torch.nn.InstanceNorm2d


def test_aten_overload_falls_back_to_the_eager_function():
    tensor = torch.tensor([1.0, 2.0])
    out = torch.ops.aten.empty_like.default(tensor)
    assert tuple(out.shape) == (2,)


def test_inductor_config_holds_the_tuning_default():
    from torch._inductor.config import max_autotune_gemm_search_space

    assert max_autotune_gemm_search_space == "DEFAULT"
