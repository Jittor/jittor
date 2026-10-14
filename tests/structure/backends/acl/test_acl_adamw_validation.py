"""Validate real CPU metadata without executing the ACL AdamW kernel."""

import jittor as jt
import pytest

from jittor.backends.acl.kernels.ops.adamw_op import fused_adamw_acl
from jittor.compat.torch.tensor_state import compatibility_owner


@pytest.fixture(params=("native", "tensor"))
def make_tensor(request):
    torch = compatibility_owner(jt)
    assert torch is not jt and torch.Tensor is not jt.Var

    def make(shape=(2, 3), dtype="float32"):
        if request.param == "native":
            value = jt.empty(shape, dtype=dtype)
            assert type(value) is jt.Var
        else:
            value = torch.empty(shape, dtype=getattr(torch, dtype), device="cpu")
            assert type(value) is torch.Tensor
        assert value.device_id < 0
        return value

    with jt.flag_scope(use_cuda=0):
        yield make


def tensor_lists(make, dtypes=("float32", "float32")):
    return [[make(dtype=dtype) for dtype in dtypes] for _ in range(4)]


def invoke(groups, step):
    return fused_adamw_acl(*groups, step, 0.001, 0.9, 0.999, 0.01, 1e-8)


@pytest.mark.parametrize("step_dtype", ("float32", "int64"))
@pytest.mark.parametrize("step_shape", ((), (1,), (1, 1)))
def test_supported_rows_and_single_value_steps_reach_the_cpu_guard(
        make_tensor, step_dtype, step_shape):
    # Dtypes must agree within a row; different rows may have different dtypes.
    groups = tensor_lists(make_tensor, ("bfloat16", "float16", "float32"))
    step = make_tensor(step_shape, step_dtype)
    with pytest.raises(RuntimeError, match="^fused AdamW ACL op requires the ACL backend$"):
        invoke(groups, step)


@pytest.mark.parametrize("bad_list", (None, 1, 2, 3))
def test_empty_or_unequal_tensor_lists_are_rejected(make_tensor, bad_list):
    groups = tensor_lists(make_tensor)
    if bad_list is None:
        groups = [[], [], [], []]
    else:
        groups[bad_list].pop()
    with pytest.raises(ValueError, match="^fused AdamW TensorLists must have one or more entries$"):
        invoke(groups, make_tensor((), "float32"))


@pytest.mark.parametrize("bad_list", (0, 1, 2, 3))
def test_each_row_requires_identical_shapes_even_with_equal_numel(make_tensor, bad_list):
    groups = tensor_lists(make_tensor)
    # The second row catches validation that only checks the first row.
    groups[bad_list][1] = make_tensor((3, 2))
    with pytest.raises(ValueError, match="^fused AdamW tensors must have identical shapes$"):
        invoke(groups, make_tensor((), "float32"))


@pytest.mark.parametrize("bad_list", (0, 1, 2, 3))
def test_each_row_requires_identical_dtypes(make_tensor, bad_list):
    groups = tensor_lists(make_tensor)
    groups[bad_list][1] = make_tensor(dtype="float16")
    with pytest.raises(TypeError, match="^fused AdamW tensors must have identical dtypes$"):
        invoke(groups, make_tensor((), "float32"))


@pytest.mark.parametrize("dtype", ("float64", "int32", "int64", "bool"))
def test_matching_unsupported_row_dtypes_are_rejected(make_tensor, dtype):
    groups = tensor_lists(make_tensor, ("float32", dtype))
    with pytest.raises(TypeError, match="^fused AdamW requires bfloat16, float16, or float32$"):
        invoke(groups, make_tensor((), "float32"))


@pytest.mark.parametrize("shape", ((0,), (2,), (1, 2)))
def test_step_requires_exactly_one_element(make_tensor, shape):
    with pytest.raises(TypeError, match="^fused AdamW step must be one float32 or int64 value$"):
        invoke(tensor_lists(make_tensor), make_tensor(shape, "float32"))


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float64", "int32", "bool"))
def test_step_rejects_unsupported_scalar_dtypes(make_tensor, dtype):
    with pytest.raises(TypeError, match="^fused AdamW step must be one float32 or int64 value$"):
        invoke(tensor_lists(make_tensor), make_tensor((), dtype))


def test_shape_errors_precede_dtype_and_step_errors(make_tensor):
    groups = tensor_lists(make_tensor)
    groups[3][1] = make_tensor((3, 2), "float64")
    with pytest.raises(ValueError, match="^fused AdamW tensors must have identical shapes$"):
        invoke(groups, make_tensor((2,), "int32"))


def test_dtype_mismatch_precedes_supported_dtype_and_step_checks(make_tensor):
    groups = tensor_lists(make_tensor, ("float64", "float32"))
    groups[3][0] = make_tensor(dtype="float32")
    with pytest.raises(TypeError, match="^fused AdamW tensors must have identical dtypes$"):
        invoke(groups, make_tensor((2,), "int32"))
