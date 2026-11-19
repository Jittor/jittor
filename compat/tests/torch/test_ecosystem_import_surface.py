"""The import-time API surface torchao, transformers and peft touch."""
import typing

import pytest
import torch


def test_optimizer_required_is_a_module_level_sentinel():
    from torch.optim.optimizer import Optimizer, required

    assert required is not None
    assert required is not Optimizer


def test_triton_ops_meta_reports_no_tuned_entry():
    from torch.sparse._triton_ops_meta import get_meta, minimize, update

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

    result = torch.ops.impl_form_probe.double(torch.tensor([1.0, 2.0]))
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


def test_functional_collectives_are_identity_in_one_process():
    from torch.distributed._functional_collectives import all_reduce, wait_tensor

    tensor = torch.tensor([1.0, 2.0])
    assert all_reduce(tensor) is tensor
    assert wait_tensor(tensor) is tensor


def test_aten_overload_falls_back_to_the_eager_function():
    tensor = torch.tensor([1.0, 2.0])
    out = torch.ops.aten.empty_like.default(tensor)
    assert tuple(out.shape) == (2,)


def test_inductor_config_holds_the_tuning_default():
    from torch._inductor.config import max_autotune_gemm_search_space

    assert max_autotune_gemm_search_space == "DEFAULT"
