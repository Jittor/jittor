"""Real singleton FSDP2 full-state placement, not multi-rank/PPO acceptance.

JITTOR_TEST_DEVICES selects cpu/cuda/npu; each runs in a fresh child/cache.
CUDA deliberately retains the unset device_id=-1 sentinel. NPU explicitly
initializes its real current device; its negative-flag routing is covered by
the separate host load-plan contract, not claimed as NPU hardware evidence.
"""
import json
import os

import pytest

from _helpers import capability as _test_capability
from _helpers.child_process import run_python_child


_CHILD = r'''
import json
import os
import numpy as np
import jittor as jt
import torch
from _helpers import capability as _test_capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

kind = os.environ["JITTOR_FSDP_STATE_TEST_DEVICE"]
assert kind in ("cpu", "cuda", "npu"), kind
assert hasattr(torch, "_torch_compat_install_context"), "this is a shim behavior test"
assert torch.distributed.get_world_size() == 1
assert torch.distributed.get_rank() == 0
ordinal = None
if kind != "cpu":
    backend = "acl" if kind == "npu" else "cuda"
    capability = _test_capability.check_accelerator(backend, backend=jt)
    assert capability.enabled, (kind, capability.state.value, capability.reason)
    if kind == "npu":
        # Unlike CUDA's implicit default, the NPU facade refuses an unset
        # native current device. Initialize it through its public owner.
        torch.npu.set_device(int(os.environ.get("JITTOR_TEST_DEVICE_INDEX", "0")))
    # Never set_device for CUDA: that would mask the original cuda:-1 bug.
    ordinal = getattr(torch, kind).current_device()
    assert ordinal >= 0

expected = {"weight": np.array([[2., 3.]], dtype=np.float32),
            "bias": np.array([4.], dtype=np.float32)}
with forbid_backend_fallbacks(), torch.device("cpu"):
    before = jt.core.backend_fallback_count()
    model = torch.nn.Linear(2, 1, device="cpu", dtype=torch.float32)
    checkpoint = {name: torch.tensor(array, device="cpu", dtype=torch.float32)
                  for name, array in expected.items()}
    for value in checkpoint.values():
        value.sync()
        assert value.device.type == "cpu" and value.location() == "cpu"
    mesh = init_device_mesh(kind, (1,), mesh_dim_names=("dp",))
    assert mesh.size() == 1 and mesh.device_type == kind
    fully_shard(model, mesh=mesh)
    raw_device_id = int(jt.flags.device_id)
    if kind == "cuda":
        assert raw_device_id == -1, "CUDA sentinel precondition was masked: %s" % raw_device_id
    result = set_model_state_dict(model, checkpoint, options=StateDictOptions(
        full_state_dict=True, broadcast_from_rank0=True, strict=True))
    assert result.missing_keys == [] and result.unexpected_keys == []
    parameters = dict(model.named_parameters())
    assert set(parameters) == set(expected), set(parameters)
    resident = {}
    actual = {}
    # Check ALL actual parameter storage before reading any result on CPU.
    # Singleton parameters and their local/full views retain logical shapes.
    for name, parameter in parameters.items():
        assert tuple(parameter.shape) == expected[name].shape, (name, "public shape changed")
        parameter.sync()
        assert parameter.device.type == kind, (name, parameter.device, kind)
        assert parameter.location() == ("cpu" if kind == "cpu" else "device")
        value = parameter.to_local()
        assert tuple(value.shape) == expected[name].shape, (name, "singleton local shape changed")
        full = parameter.full_tensor()
        assert tuple(full.shape) == expected[name].shape, (name, "full logical shape changed")
        value.sync()
        assert value.dtype == torch.float32, (name, value.dtype)
        native_backend, native_ordinal = jt.core.dispatch_context([value])
        assert value.device.type == kind, (name, value.device, kind)
        assert value.location() == ("cpu" if kind == "cpu" else "device"), (name, value.location())
        if kind == "cpu":
            assert native_backend == "cpu", (name, native_backend)
        else:
            allowed = ("acl", "acl_legacy") if kind == "npu" else ("cuda",)
            assert native_backend in allowed, (name, native_backend)
            assert int(value.device_id) == ordinal and int(native_ordinal) == ordinal, (name, native_ordinal)
        resident[name] = value
        actual[name] = {"backend": native_backend, "ordinal": int(native_ordinal),
                        "location": value.location(), "dtype": str(value.dtype),
                        "local_shape": list(value.shape)}
    for name, value in resident.items():
        np.testing.assert_array_equal(
            value.detach().cpu().numpy(), expected[name])
    # The original dictionary remains CPU resident and byte/value unchanged.
    for name, value in checkpoint.items():
        value.sync()
        assert value.device.type == "cpu" and value.location() == "cpu"
        np.testing.assert_array_equal(value.numpy(), expected[name])
    assert int(jt.flags.device_id) == raw_device_id, "state loading changed ambient device selection"
    fallback_delta = jt.core.backend_fallback_count() - before
    assert fallback_delta == 0, fallback_delta
print("FSDP_FULL_STATE_DEVICE_OK " + json.dumps({
    "device": kind, "world_size": 1, "rank": 0, "raw_device_id": raw_device_id,
    "current_device": ordinal, "actual_parameters": actual,
    "parameter_count": len(parameters), "fallback_delta": fallback_delta,
}, sort_keys=True), flush=True)
'''


_TRAINING_CHILD = r'''
import json
import os
import numpy as np
import jittor as jt
import torch
from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

kind = os.environ["JITTOR_FSDP_STATE_TEST_DEVICE"]
assert kind in ("cpu", "cuda", "npu")
assert hasattr(torch, "_torch_compat_install_context")
assert torch.distributed.get_world_size() == 1
ordinal = None
if kind != "cpu":
    backend = "acl" if kind == "npu" else "cuda"
    assert capability.check_accelerator(backend, backend=jt).enabled
    if kind == "npu":
        torch.npu.set_device(int(os.environ.get("JITTOR_TEST_DEVICE_INDEX", "0")))
    ordinal = getattr(torch, kind).current_device()
    assert ordinal >= 0
if kind == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((1, 2), device="cpu"))
        self.bias = torch.nn.Parameter(torch.zeros((1,), device="cpu"))
        self.gain = torch.nn.Parameter(torch.tensor(0., device="cpu"))
        self.frozen = torch.nn.Parameter(torch.zeros((2, 2), device="cpu"), requires_grad=False)
        self.weight_alias = self.weight
    def forward(self, x):
        return (x @ self.weight.t() + self.bias) * self.gain

def check(value, expected, exact=False):
    expected = np.asarray(expected)
    assert tuple(value.shape) == expected.shape, "no public reshape exemptions"
    local = value.to_local() if callable(getattr(value, "to_local", None)) else value
    assert tuple(local.shape) == expected.shape
    local.sync()
    backend, index = jt.core.dispatch_context([local])
    assert local.device.type == kind and local.dtype == torch.float32
    assert local.location() == ("cpu" if kind == "cpu" else "device")
    assert backend in (("acl", "acl_legacy") if kind == "npu" else (kind,))
    if kind != "cpu":
        assert int(index) == ordinal and int(local.device_id) == ordinal
    actual = np.array(local.detach().cpu().numpy(), copy=True)
    assert np.isfinite(actual).all()
    if exact:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    # Host inspection must not migrate the live parameter/gradient.
    assert local.location() == ("cpu" if kind == "cpu" else "device")

with forbid_backend_fallbacks(), torch.device("cpu"):
    before = jt.core.backend_fallback_count()
    model = Model()
    identities = {n: id(p) for n, p in model.named_parameters(remove_duplicate=False)}
    fully_shard(model, mesh=init_device_mesh(kind, (1,), mesh_dim_names=("dp",)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    inputs = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    for weight, bias, gain in (([2., 3.], 4., 1.), ([-1., 5.], -2., 2.)):
        optimizer.zero_grad(set_to_none=True)
        model.reshard()
        expected = {"weight": np.array([weight], dtype=np.float32),
                    "bias": np.array([bias], dtype=np.float32),
                    "gain": np.array(gain, dtype=np.float32),
                    "frozen": np.arange(4, dtype=np.float32).reshape(2, 2)}
        expected["weight_alias"] = expected["weight"].copy()
        checkpoint = {n: torch.tensor(a, device="cpu") for n, a in expected.items()}
        result = set_model_state_dict(model, checkpoint, options=StateDictOptions(
            full_state_dict=True, broadcast_from_rank0=True, strict=True))
        assert result.missing_keys == [] and result.unexpected_keys == []
        named = dict(model.named_parameters(remove_duplicate=False))
        assert {n: id(p) for n, p in named.items()} == identities
        assert model.weight is model.weight_alias and not model.frozen.requires_grad
        for name, p in named.items():
            check(p, expected[name], exact=True)
            check(p.full_tensor(), expected[name], exact=True)
            assert checkpoint[name].device.type == "cpu"
            np.testing.assert_array_equal(checkpoint[name].numpy(), expected[name])
        x = torch.tensor(inputs, device=kind, requires_grad=True)
        y = model(x)
        loss = y.square().sum()
        loss.backward()
        z = inputs.astype(np.float64) @ np.asarray(weight, dtype=np.float64)[:, None] + bias
        gradients = {"weight": 2 * gain * gain * z.T @ inputs.astype(np.float64),
                     "bias": np.array([2 * gain * gain * z.sum()]),
                     "gain": np.array(2 * gain * np.sum(z * z))}
        check(y, z * gain)
        check(loss, np.array(np.sum((z * gain) ** 2)))
        assert x.grad is not None
        check(x.grad, 2 * gain * gain * z * np.asarray(weight)[None, :])
        for name, gradient in gradients.items():
            p = getattr(model, name)
            assert p.grad is not None, (name, "missing full forward gradient")
            check(p.grad, gradient)
        assert model.frozen.grad is None
        optimizer.step()
        for name, gradient in gradients.items():
            check(getattr(model, name), expected[name].astype(np.float64) - 0.001 * gradient)
    fallback_delta = jt.core.backend_fallback_count() - before
    assert fallback_delta == 0
print("FSDP_FULL_STATE_TRAINING_OK " + json.dumps({
    "device": kind, "world_size": 1, "rank": 0, "loads": 2,
    "parameter_count": 5, "fallback_delta": fallback_delta,
    "atol": 1e-5, "rtol": 1e-5, "ppo_acceptance": False,
}), flush=True)
'''


@pytest.mark.parametrize("contract", ("placement", "training"))
@pytest.mark.parametrize(
    "device_type", os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","),
    ids=lambda device: "device_" + device.strip())
def test_public_singleton_full_state_actual_device(device_type, contract):
    kind = device_type.strip()
    assert kind in ("cpu", "cuda", "npu"), "select explicit cpu/cuda/npu devices"
    if kind != "cpu":
        import jittor as jt
        backend = "acl" if kind == "npu" else "cuda"
        capability = _test_capability.check_accelerator(backend, backend=jt)
        if not capability.enabled:
            pytest.skip("accelerator prerequisite: %s %s: %s" %
                        (kind, capability.state.value, capability.reason))
    environment = dict(os.environ)
    # This is a genuine singleton, not a child that inherits a Ray/MPI rank.
    for key in tuple(environment):
        if (key.startswith(("OMPI_", "PMI_", "PMIX_", "JITTOR_VERL_"))
                or key in ("REAL_TORCH_SITE", "MASTER_ADDR", "MASTER_PORT",
                           "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
                           "JT_NCCL_ROOTINFO_FILE", "JT_HCCL_ROOTINFO_FILE")):
            environment.pop(key)
    environment.update({
        "JITTOR_TORCH_SHIM": "1", "JITTOR_TEST_DEVICES": kind,
        "JITTOR_FSDP_STATE_TEST_DEVICE": kind,
        "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1",
        "JITTOR_TORCH_DISTRIBUTED_AUTO_INIT": "0", "device_id": "-1",
        "use_cuda": "0" if kind == "cpu" else "1", "backend_fallback": "error",
        "cache_name": environment.get("cache_name", "default") + "_fsdp2_full_state_device_" + kind,
    })
    child = _CHILD if contract == "placement" else _TRAINING_CHILD
    completed = run_python_child(["-B", "-c", child], env=environment, inherit=False,
                                 merge_stderr=True)
    assert completed.returncode == 0, completed.stdout[-6000:]
    prefix = "FSDP_FULL_STATE_DEVICE_OK " if contract == "placement" else "FSDP_FULL_STATE_TRAINING_OK "
    records = [json.loads(line[len(prefix):]) for line in completed.stdout.splitlines()
               if line.startswith(prefix)]
    assert len(records) == 1, completed.stdout[-6000:]
    record = records[0]
    assert record["device"] == kind
    assert record["world_size"] == 1 and record["rank"] == 0
    assert record["parameter_count"] == (2 if contract == "placement" else 5)
    assert record["fallback_delta"] == 0
    if contract == "training":
        assert record["loads"] == 2 and record["ppo_acceptance"] is False
    if kind == "cuda" and contract == "placement":
        assert record["raw_device_id"] == -1
