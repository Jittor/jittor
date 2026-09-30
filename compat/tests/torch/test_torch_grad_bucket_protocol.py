"""GradBucket's import protocol, without claiming DDP bucket-hook support.

The independent PyTorch child checks the public type and constructor contract.
The remaining tests pin the shim's explicit unsupported boundary; they do not
manufacture reducer-owned instances or exercise gradient communication.
"""

import inspect
import json
import os
import subprocess

import pytest
import torch

from _helpers.child_process import child_env, default_timeout
from jittor.compat.torch.fidelity import Fidelity, fidelity_of
from jittor.compat.torch.installers import distributed as owner


_METHODS = ("index", "buffer", "gradients", "parameters", "is_last", "set_buffer")


def _protocol_snapshot(torch_module):
    bucket = torch_module.distributed.GradBucket
    constructor_errors = []
    for args, kwargs in (((), {}), ((0,), {}), ((), {"buffer": None})):
        try:
            bucket(*args, **kwargs)
        except Exception as error:
            # Record the observed category: an unexpected exception still fails
            # the explicit TypeError assertion in the caller.
            constructor_errors.append(type(error).__name__)
        else:
            constructor_errors.append("constructed")
    return {
        "is_type": isinstance(bucket, type),
        "same_owner": bucket is torch_module._C._distributed_c10d.GradBucket,
        "methods": {name: callable(getattr(bucket, name, None)) for name in _METHODS},
        "constructor_errors": constructor_errors,
    }


def test_grad_bucket_protocol_matches_independent_pytorch(tmp_path):
    oracle = os.environ.get("REAL_TORCH_PYTHON", "").strip()
    if not oracle:
        reason = "REAL_TORCH_PYTHON is not configured; independent PyTorch unverified"
        required = os.environ.get("JITTOR_REQUIRE_REAL_TORCH", "").strip().lower()
        if required in ("1", "true", "yes", "on"):
            pytest.fail(reason)
        pytest.skip(reason)

    source = (
        "import json\nimport torch\n"
        "assert not hasattr(torch, '_torch_compat_install_context'), "
        "'oracle resolved to the Jittor torch shim'\n"
        "assert hasattr(torch._C, '_c10d_init'), 'oracle lacks binary c10d'\n"
        "_METHODS = %r\n" % (_METHODS,)
        + inspect.getsource(_protocol_snapshot)
        + "\nprint(json.dumps({'version': torch.__version__, "
        "'origin': torch.__file__, 'protocol': _protocol_snapshot(torch)}))\n"
    )
    completed = subprocess.run(
        [oracle, "-c", source],
        env=child_env(without_torch_mode=True, repo_paths=False),
        cwd=str(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=default_timeout(),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    reference = json.loads(completed.stdout.strip().splitlines()[-1])
    expected = {
        "is_type": True,
        "same_owner": True,
        "methods": {name: True for name in _METHODS},
        "constructor_errors": ["TypeError"] * 3,
    }
    assert reference["protocol"] == expected, reference
    assert _protocol_snapshot(torch) == reference["protocol"], reference


def test_grad_bucket_uses_one_owner_and_declares_unimplemented_hooks():
    bucket = torch.distributed.GradBucket
    assert bucket is owner.GradBucket
    assert bucket is torch._C._distributed_c10d.GradBucket
    record = fidelity_of("torch.distributed.GradBucket")
    assert record.implementation is bucket
    assert record.level is Fidelity.UNIMPLEMENTED
    assert record.detail.strip(), "The unsupported scope must be documented"


@pytest.mark.parametrize("args,kwargs", [((), {}), ((0,), {}), ((), {"buffer": None})])
def test_grad_bucket_rejects_public_construction(args, kwargs):
    with pytest.raises(TypeError):
        torch.distributed.GradBucket(*args, **kwargs)


@pytest.mark.parametrize("method", _METHODS)
def test_grad_bucket_methods_refuse_unimplemented_computation(method):
    # Exercise only the shim's rejection guard, without allocating a fake
    # GradBucket. Invalid-self behaviour is not a PyTorch parity claim.
    arguments = (None, None) if method == "set_buffer" else (None,)
    with pytest.raises(NotImplementedError):
        getattr(torch.distributed.GradBucket, method)(*arguments)
