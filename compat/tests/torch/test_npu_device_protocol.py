"""CPU refusal and actual NPU selection against independent torch_npu."""
import inspect
import json
import os
import subprocess

import pytest
import torch

from _helpers.child_process import child_env, default_timeout


def _trajectory(t):
    count = t.npu.device_count()
    assert t.npu.is_available() and count > 0
    original = t.npu.current_device()
    records = []
    arguments = [0, "npu:%d" % (count - 1), t.device("npu:0"), None, t.device("npu"), -1]
    try:
        for argument in arguments:
            t.npu.set_device(argument)
            selected = t.npu.current_device()
            x = t.tensor([selected + 1., selected + 2.], dtype=t.float32, device="npu")
            y = x.square() + 3
            t.npu.synchronize(t.device("npu", selected))
            assert t.npu.current_device() == selected
            assert x.device.type == y.device.type == "npu"
            assert x.device.index == y.device.index == selected
            records.append({"selected": selected, "values": y.cpu().tolist()})
        for method in (t.npu.set_device, t.npu.synchronize):
            try:
                method(count)
            except RuntimeError:
                continue
            raise AssertionError("Out-of-range device was silently accepted")
    finally:
        t.npu.set_device(original)
    return {"count": count, "records": records}


def test_device_protocol_on_selected_backend(tmp_path):
    import jittor as jt
    selected = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
    assert selected in ("cpu", "npu"), "Select one backend per process"
    if selected == "cpu":
        assert not jt.compiler.has_acl and not jt.flags.use_cuda
        assert torch.npu.device_count() == 0
        assert torch.npu.is_available() is False
        for operation in (torch.npu.current_device, lambda: torch.npu.set_device(0), torch.npu.synchronize):
            with pytest.raises(RuntimeError, match="ACL"):
                operation()
        return
    assert jt.compiler.has_acl and jt.flags.use_cuda
    oracle = os.environ.get("REAL_TORCH_PYTHON", "")
    if not oracle:
        pytest.fail("NPU verification requires an independent REAL_TORCH_PYTHON")
    source = (
        "import json,torch\nassert not hasattr(torch, '_torch_compat_install_context')\n"
        "import torch_npu\nassert torch.npu.is_available()\n"
        + inspect.getsource(_trajectory)
        + "\nprint(json.dumps(_trajectory(torch)))\n")
    result = subprocess.run(
        [oracle, "-c", source], cwd=str(tmp_path),
        env=child_env(without_torch_mode=True, repo_paths=False),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=default_timeout())
    assert result.returncode == 0, result.stdout + result.stderr
    reference = json.loads(result.stdout.strip().splitlines()[-1])
    before = jt.core.backend_fallback_count()
    actual = _trajectory(torch)
    jt.sync_all(True)
    assert jt.core.backend_fallback_count() == before
    assert actual == reference
    print("NPU_DEVICE_ORACLE=" + json.dumps(actual))


@pytest.mark.parametrize("argument", ["cpu", "cuda:0", torch.device("cpu")])
def test_npu_device_rejects_other_backends(argument):
    for operation in (torch.npu.set_device, torch.npu.synchronize):
        with pytest.raises(ValueError, match="npu"):
            operation(argument)


def test_negative_device_selection_does_not_change_runtime():
    import jittor as jt
    original = int(jt.current_device())
    assert torch.npu.set_device(-1) is None
    assert int(jt.current_device()) == original


def test_set_device_requires_an_argument():
    with pytest.raises(TypeError):
        torch.npu.set_device()


def test_npu_device_public_owners_and_fidelity():
    from jittor.compat.torch.fidelity import Fidelity, fidelity_of
    from jittor.compat.torch.installers.cuda import npu
    for name in ("is_available", "device_count", "current_device", "set_device", "synchronize"):
        assert getattr(torch.npu, name) is getattr(npu, name)
        record = fidelity_of("torch.npu." + name)
        assert record.implementation is getattr(torch.npu, name)
        assert record.level == Fidelity.APPROXIMATE
    assert torch.mps.is_available() is False
