"""Real COO metadata storage and explicit unsupported boundaries."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess

from _helpers.child_process import child_env, default_timeout

import numpy as np
import pytest

import torch


@pytest.mark.parametrize("array", [
    np.array([[False, True, False], [True, False, True]], dtype=bool),
    np.zeros((2, 3), dtype=bool),
    np.ones((2, 2), dtype=bool),
    np.array([False, True, True], dtype=bool),
    np.arange(24).reshape(2, 3, 4) % 3 == 0,
    np.zeros((0, 3), dtype=bool),
])
def test_bool_coo_values_indices_and_dense_roundtrip(array):
    dense = torch.from_numpy(array)
    sparse = dense.to_sparse()
    assert isinstance(sparse, torch.Tensor)
    assert not isinstance(sparse, torch.nn.Parameter)
    assert torch.is_tensor(sparse)
    assert sparse.is_sparse and not sparse.is_sparse_csr
    assert sparse.layout == torch.sparse_coo
    assert sparse.dtype == torch.bool
    assert sparse.device == dense.device
    assert tuple(sparse.shape) == array.shape
    assert sparse.sparse_dim() == array.ndim
    assert sparse.dense_dim() == 0
    assert sparse.is_coalesced()
    assert sparse.coalesce() is sparse
    assert sparse._nnz() == np.count_nonzero(array)
    assert sparse.numel() == array.size
    assert sparse.indices().dtype == torch.int64
    assert sparse.values().dtype == torch.bool
    np.testing.assert_array_equal(sparse.indices().cpu().numpy(), np.asarray(np.nonzero(array)))
    np.testing.assert_array_equal(sparse.values().cpu().numpy(), array[np.nonzero(array)])
    np.testing.assert_array_equal(sparse.to_dense().cpu().numpy(), array)


def test_sparse_nonpersistent_buffer_participates_in_module_protocol():
    model = torch.nn.Linear(2, 3)
    sparse = torch.tensor([[False, True], [True, False]], dtype=torch.bool).to_sparse()
    model.register_buffer("alignment", sparse, persistent=False)
    assert dict(model.named_buffers())["alignment"] is sparse
    assert model.get_buffer("alignment") is sparse
    assert model._buffers["alignment"] is sparse
    assert "alignment" not in model.state_dict()
    assert "alignment" not in dict(model.named_parameters())
    assert model.cpu() is model
    assert model.to(device="cpu", dtype=torch.float32) is model
    assert model.alignment.dtype == torch.bool
    assert model.alignment.device.type == "cpu"
    np.testing.assert_array_equal(model.alignment.to_dense().numpy(), [[False, True], [True, False]])
    original_parameters = dict(model.named_parameters())
    model.double()
    for name, parameter in model.named_parameters():
        assert parameter is original_parameters[name]
        assert parameter.dtype == torch.float64
    assert model.alignment.dtype == torch.bool
    assert model.alignment is sparse
    assert set(model.state_dict()) == {"weight", "bias"}


def test_sparse_copy_and_detach_preserve_storage_contract():
    sparse = torch.tensor([[True, False], [False, True]], dtype=torch.bool).to_sparse()
    assert sparse.to("cpu") is sparse
    assert sparse.cpu() is sparse
    assert sparse.to(dtype=torch.bool) is sparse
    assert sparse.to_sparse() is sparse
    detached = sparse.detach()
    copied = sparse.clone()
    moved_copy = sparse.to("cpu", copy=True)
    assert detached is not sparse
    assert copied is not sparse
    assert moved_copy is not sparse
    assert copied.values() is not sparse.values()
    assert copied.indices() is not sparse.indices()
    assert not detached.requires_grad
    assert detached.grad is None
    for result in (detached, copied, moved_copy):
        np.testing.assert_array_equal(result.to_dense().numpy(), sparse.to_dense().numpy())
        assert result.layout == torch.sparse_coo


def test_sparse_scope_is_explicit_and_does_not_densify_implicitly():
    sparse = torch.tensor([[True, False]], dtype=torch.bool).to_sparse()
    with pytest.raises(TypeError, match="to_dense"):
        sparse.numpy()
    with pytest.raises(NotImplementedError, match="bool"):
        torch.ones(2, 3).to_sparse()
    with pytest.raises(NotImplementedError, match="hybrid"):
        torch.ones(2, 3, dtype=torch.bool).to_sparse(1)
    with pytest.raises(NotImplementedError, match="bool"):
        sparse.to(dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="not supported"):
        sparse + sparse
    with pytest.raises(NotImplementedError, match="persistent"):
        torch.nn.Module().register_buffer("sparse", sparse)
    with pytest.raises(NotImplementedError, match="layout"):
        torch.zeros(2, 3, layout=torch.sparse_coo)
    with pytest.raises(RuntimeError, match="gradients"):
        sparse.requires_grad_(True)


def test_sparse_owner_is_not_a_dense_var_and_dense_identity_is_unchanged():
    import jittor as jt
    from jittor.compat.torch.sparse_frontend import SparseCOOTensor
    from jittor.sparse.coo import SparseVar

    dense = torch.tensor([[False, True]], dtype=torch.bool)
    sparse = dense.to_sparse()
    assert isinstance(sparse, SparseVar)
    assert isinstance(sparse, SparseCOOTensor)
    assert not isinstance(sparse, jt.Var)
    assert isinstance(sparse._coo, SparseVar)
    assert not hasattr(sparse, "_dense")
    assert isinstance(dense, torch.Tensor)
    assert isinstance(dense, torch.BoolTensor)
    assert isinstance(dense, jt.Var)
    assert not isinstance(dense, SparseCOOTensor)
    assert not isinstance(object(), torch.Tensor)
    assert not isinstance(sparse, torch.FloatTensor)
    assert not isinstance(sparse, torch.BoolTensor)


def test_sparse_copy_preserves_buffer_identity_and_owns_new_coordinates():
    target = torch.tensor([[True, False, False], [False, False, False]], dtype=torch.bool).to_sparse()
    source_array = np.array([[False, True, True], [True, False, False]], dtype=bool)
    source = torch.from_numpy(source_array).to_sparse()
    model = torch.nn.Module()
    model.register_buffer("alignment", target, persistent=False)
    assert target.copy_(source, non_blocking=True) is target
    assert model.get_buffer("alignment") is target
    assert target._nnz() == 3
    assert target.indices() is not source.indices()
    assert target.values() is not source.values()
    np.testing.assert_array_equal(target.to_dense().numpy(), source_array)
    source.values().fill_(False)
    np.testing.assert_array_equal(target.to_dense().numpy(), source_array)
    assert target.copy_(target) is target
    with pytest.raises(NotImplementedError, match="COO source"):
        target.copy_(torch.from_numpy(source_array))
    with pytest.raises(RuntimeError, match="shapes"):
        target.copy_(torch.ones(1, 3, dtype=torch.bool).to_sparse())
    np.testing.assert_array_equal(target.to_dense().numpy(), source_array)


def test_dense_and_sparse_layouts_match_public_tensor_kinds():
    floating = torch.ones(2, 3)
    boolean = torch.ones(2, 3, dtype=torch.bool)
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    for dense in (floating, boolean, parameter):
        assert dense.layout == torch.strided
        assert dense.is_sparse is False
        assert dense.is_sparse_csr is False
        with pytest.raises(AttributeError):
            dense.layout = torch.sparse_coo
    assert isinstance(floating, torch.FloatTensor)
    assert not isinstance(floating, torch.BoolTensor)
    assert isinstance(boolean, torch.BoolTensor)
    assert not isinstance(boolean, torch.FloatTensor)
    sparse = boolean.to_sparse()
    assert sparse.layout == torch.sparse_coo
    assert sparse.is_sparse is True
    assert sparse.is_sparse_csr is False
    assert not isinstance(sparse, torch.BoolTensor)
    with pytest.raises(AttributeError):
        sparse.layout = torch.strided


def test_dense_layout_metadata_does_not_modify_native_var():
    import jittor as jt

    assert torch.Tensor is not jt.Var
    for name in ("layout", "is_sparse", "is_sparse_csr"):
        assert name in vars(torch.Tensor)
        assert not hasattr(jt.Var, name)


@pytest.mark.parametrize("data,shape", [
    ([], (0,)),
    ([], (1, 0)),
    ([False], (1,)),
    ([False], (1, 1)),
    ([True], (1,)),
    ([True], (1, 1)),
    ([False, False], (2,)),
    ([True, False], (1, 2)),
])
def test_sparse_truth_and_length_follow_logical_shape(data, shape):
    sparse = torch.tensor(data, dtype=torch.bool).reshape(shape).to_sparse()
    assert len(sparse) == shape[0]
    if len(data) == 1:
        assert bool(sparse) is data[0]
    elif not data:
        with pytest.raises(RuntimeError, match="no values is ambiguous"):
            bool(sparse)
    else:
        with pytest.raises(RuntimeError, match="more than one value is ambiguous"):
            bool(sparse)


def test_sparse_single_element_truth_reads_stored_value():
    sparse = torch.tensor([True], dtype=torch.bool).to_sparse()
    assert bool(sparse) is True
    sparse.values().fill_(False)
    assert bool(sparse) is False


def test_sparse_equality_rejects_unsupported_comparison_instead_of_identity():
    sparse = torch.tensor([False], dtype=torch.bool).to_sparse()
    for other in (sparse, True, False):
        with pytest.raises(NotImplementedError):
            sparse == other
        with pytest.raises(NotImplementedError):
            sparse != other
    assert isinstance(hash(sparse), int)
    assert {sparse: 1}[sparse] == 1
    assert sparse in {sparse}


@pytest.mark.parametrize("replacement", ["attribute", "buffer_mapping", "placeholder"])
def test_persistent_sparse_replacement_cannot_bypass_unsupported_state_boundary(replacement):
    model = torch.nn.Module()
    original = None if replacement == "placeholder" else torch.zeros(1, dtype=torch.bool)
    model.register_buffer("tracked", original)
    sparse = torch.tensor([True], dtype=torch.bool).to_sparse()
    if replacement == "buffer_mapping":
        model._buffers["tracked"] = sparse
    else:
        model.tracked = sparse
    assert dict(model.named_buffers())["tracked"] is sparse
    with pytest.raises(NotImplementedError, match="persistent sparse"):
        model.state_dict()


def test_explicit_sparse_buffer_with_private_name_is_enumerated():
    model = torch.nn.Module()
    sparse = torch.tensor([[False, True]], dtype=torch.bool).to_sparse()
    model.register_buffer("_alignment", sparse, persistent=False)
    assert dict(model.named_buffers())["_alignment"] is sparse
    assert model.get_buffer("_alignment") is sparse
    assert model._buffers["_alignment"] is sparse
    assert "_alignment" not in model.state_dict()
    assert model.cpu() is model
    assert dict(model.named_buffers())["_alignment"].device.type == "cpu"

@pytest.mark.parametrize("factory", ["eye", "rand", "randn"])
def test_sparse_layout_factory_never_silently_returns_dense(factory):
    options = {"layout": torch.sparse_coo}
    if factory != "eye":
        options["generator"] = torch.Generator().manual_seed(17)
    with pytest.raises(NotImplementedError, match="sparse COO"):
        getattr(torch, factory)(2, **options)


# Same public operation sequence runs in the active shim and a clean, genuine
# PyTorch interpreter. Bool/int metadata is compared exactly (no tolerance).
_PROBE_PATH = Path(__file__).resolve().parents[3] / "agent/skills/jittor-torch-diff/sparse_metadata_probe.py"
# Read literal case metadata during collection, without executing the probe.
_INVENTORY_NODE = next(
    node.value for node in ast.parse(_PROBE_PATH.read_text(encoding="utf-8")).body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "CASES"
            for target in node.targets)
)
_COLLECTED_SPARSE_CASES = ast.literal_eval(_INVENTORY_NODE)
assert isinstance(_COLLECTED_SPARSE_CASES, (list, tuple)) and _COLLECTED_SPARSE_CASES
assert all(isinstance(name, str) and name for name in _COLLECTED_SPARSE_CASES)
assert len(_COLLECTED_SPARSE_CASES) == len(set(_COLLECTED_SPARSE_CASES))


@pytest.fixture(scope="module")
def sparse_probe():
    spec = importlib.util.spec_from_file_location("sparse_metadata_probe", _PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert tuple(module.CASES) == tuple(_COLLECTED_SPARSE_CASES)
    return module


@pytest.fixture(scope="module")
def sparse_torch_oracle(tmp_path_factory, sparse_probe):
    python = os.environ.get("REAL_TORCH_PYTHON", "").strip()
    if not python:
        if os.environ.get("JITTOR_REQUIRE_REAL_TORCH", "").lower() not in ("", "0", "false", "no", "off"):
            pytest.fail("REAL_TORCH_PYTHON is required for sparse COO differential tests")
        pytest.skip("REAL_TORCH_PYTHON is not configured")
    output = tmp_path_factory.mktemp("sparse-torch-oracle") / "oracle.json"
    env = child_env(without_torch_mode=True, repo_paths=False)
    result = subprocess.run([python, str(_PROBE_PATH), "--output", str(output)],
                            env=env, cwd=str(output.parent), capture_output=True,
                            text=True, timeout=default_timeout())
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["runtime"] == "pytorch" and report["device"] == "cpu"
    assert set(report["cases"]) == set(sparse_probe.CASES)
    assert hasattr(torch, "_torch_compat_install_context"), "candidate must be the Jittor shim"
    print("sparse COO oracle: {} from {}".format(report["version"], report["origin"]))
    return report


@pytest.mark.parametrize("sparse_case", _COLLECTED_SPARSE_CASES)
def test_bool_coo_matches_independent_pytorch(sparse_case, sparse_probe, sparse_torch_oracle):
    import jittor as jt

    with jt.runtime.scope(use_cuda=0):
        actual = sparse_probe.run_case(torch, sparse_case)
    assert actual == sparse_torch_oracle["cases"][sparse_case], "COO contract diverged: " + sparse_case
