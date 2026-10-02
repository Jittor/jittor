"""Dense Tensor.is_sparse protocol against a separate binary PyTorch process.

Run independently with JITTOR_TEST_DEVICES=cpu or npu. These checks do not
claim sparse tensor creation, layouts or arithmetic support.
"""
import inspect
import json
import os
import subprocess

import pytest
import torch

from _helpers.child_process import child_env, default_timeout


def _dense_records(t, device):
    leaf = t.tensor([[1., -2.], [3., 4.]], dtype=t.float32,
                    device=device, requires_grad=True)
    (leaf * leaf).sum().backward()
    assert leaf.grad is not None
    cases = {
        "scalar": t.tensor(3., dtype=t.float32, device=device),
        "empty": t.empty((0, 2), dtype=t.float32, device=device),
        "regular": leaf,
        "transpose": leaf.transpose(0, 1),
        "detach": leaf.detach(),
        "gradient": leaf.grad,
    }
    records = {}
    for name, tensor in cases.items():
        # Assert placement before copying values to CPU for serialization.
        assert tensor.device.type == device.split(":")[0]
        assert tensor.dtype == t.float32
        assert tensor.is_sparse is False
        try:
            tensor.is_sparse = True
        except AttributeError as error:
            readonly_error = type(error).__name__
        else:
            raise AssertionError("Tensor.is_sparse must be read-only")
        assert tensor.is_sparse is False
        records[name] = {
            "is_sparse": tensor.is_sparse,
            "readonly_error": readonly_error,
            "shape": list(tensor.shape),
            "device": tensor.device.type,
            "dtype": str(tensor.dtype),
            "values": tensor.detach().cpu().tolist(),
        }
    return records


def test_dense_sparse_flag_matches_binary_torch(tmp_path):
    oracle = os.environ.get("REAL_TORCH_PYTHON", "").strip()
    if not oracle:
        reason = "REAL_TORCH_PYTHON is not configured; dense sparse-flag oracle unverified"
        if os.environ.get("JITTOR_REQUIRE_REAL_TORCH", "").lower() in ("1", "true", "yes", "on"):
            pytest.fail(reason)
        pytest.skip(reason)
    selected = os.environ.get("JITTOR_TEST_DEVICES", "cpu").strip().lower()
    assert selected in ("cpu", "npu", "npu:0"), "Run CPU and NPU in separate processes"
    device = "npu:0" if selected.startswith("npu") else "cpu"
    source = (
        "import json\nimport torch\n"
        "assert not hasattr(torch, '_torch_compat_install_context')\n"
        "assert hasattr(torch._C, '_c10d_init')\n"
        + ("import torch_npu\nassert torch.npu.is_available()\n" if device.startswith("npu") else "")
        + inspect.getsource(_dense_records)
        + "\nprint(json.dumps({'version': torch.__version__, 'records': _dense_records(torch, %r)}))\n" % device
    )
    result = subprocess.run(
        [oracle, "-c", source],
        env=child_env(without_torch_mode=True, repo_paths=False),
        cwd=str(tmp_path), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=default_timeout())
    assert result.returncode == 0, result.stdout + result.stderr
    expected = json.loads(result.stdout.strip().splitlines()[-1])
    import jittor as jt
    if device.startswith("npu"):
        assert jt.compiler.has_acl, "NPU verification requires the real ACL backend"
        assert jt.flags.use_acl and jt.flags.use_cuda, "ACL dispatch must be active"
    before = jt.core.backend_fallback_count()
    actual = _dense_records(torch, device)
    jt.sync_all()
    assert jt.core.backend_fallback_count() == before, "Dense flag probe used backend fallback"
    assert actual == expected["records"], expected["version"]
    assert actual["gradient"]["values"] == [[2., -4.], [6., 8.]]


def test_dense_sparse_flag_frontend_owner_and_scoped_fidelity():
    import jittor as jt
    from jittor.compat.torch.fidelity import Fidelity, fidelity_of
    from jittor.compat.torch.installers.tensor.method_api import _api_is_sparse

    descriptor = torch.Tensor.is_sparse
    assert isinstance(descriptor, property)
    assert descriptor.fget is _api_is_sparse
    assert descriptor.fset is None
    # Publishing the frontend property must not mutate the native Var class.
    assert getattr(jt.Var, "is_sparse", None) is not descriptor
    record = fidelity_of("torch.Tensor.is_sparse")
    assert record.implementation is _api_is_sparse
    assert record.level is Fidelity.EXACT
    assert "dense native Var" in record.detail
    assert "no sparse COO" in record.detail
