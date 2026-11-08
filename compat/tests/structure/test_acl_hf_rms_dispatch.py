"""HF RMSNorm dispatch preserves original arithmetic on an active ACL backend.

Execute the actual dispatcher definitions with host sentinels: no runtime import,
CANN initialization or JIT is needed to prove which owner receives the call.
"""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


_OWNER = Path(__file__).resolve().parents[2] / "torch/installers/nn/module_methods.py"


def namespace(*, acl=True, cuda=True, compiled=True, backend=None):
    class Var:
        def __init__(self, dtype):
            self.dtype = dtype

    calls = []
    marker = object()

    def training(*args):
        calls.append("training_kernel")
        return marker

    def bfloat16(*args):
        calls.append("bf16_kernel_probe")
        return None

    owner = ast.parse(_OWNER.read_text())
    names = {"_standard_rms_norm", "_dispatch_module_call"}
    program = ast.Module(
        body=[
            node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name in names
        ]
    )
    if "type_ignores" in ast.Module._fields:
        program.type_ignores = []
    resolved = (
        backend
        if backend is not None
        else "acl"
        if acl and cuda and compiled
        else "cuda"
        if cuda
        else "cpu"
    )
    flags = SimpleNamespace(use_acl=acl, use_cuda=cuda, amp_reg=89)
    @contextmanager
    def flag_scope(**values):
        old = {key: getattr(flags, key) for key in values}
        try:
            for key, value in values.items(): setattr(flags, key, value)
            yield
        finally:
            for key, value in old.items(): setattr(flags, key, value)
    scope = {
        "jt": SimpleNamespace(
            Var=Var,
            compiler=SimpleNamespace(has_acl=compiled),
            core=SimpleNamespace(
                dispatch_context=lambda values: (resolved, 0 if resolved != "cpu" else -1)
            ),
            flags=flags, flag_scope=flag_scope,
        ),
        "_jittor_dtype_name": str,
        "_acl_bfloat16_rms_norm": bfloat16,
        "_backend_hooks": SimpleNamespace(rms_norm_training_cuda=training, rms_norm_cuda=training),
        "_rms_norm_training_cuda": training,
        "_rms_norm_cuda": training,
        "_prefer_forward": lambda cls: True,
        "_ORIG_MODULE_DISPATCH_CALL": lambda *args, **kwargs: "native_execute",
    }
    exec(compile(ast.fix_missing_locations(program), str(_OWNER), "exec"), scope)

    class ExampleRMSNorm:
        def __init__(self, dtype):
            self.weight = Var(dtype)
            self.variance_epsilon = 1e-6

        def forward(self, value):
            if resolved == "acl":
                assert flags.amp_reg == 0
            calls.append("original_hf_forward")
            return ("original", value.dtype)

    return scope, Var, ExampleRMSNorm, calls, marker


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_active_acl_runs_original_forward_with_amp_suspended(dtype):
    scope, Var, Rms, calls, _ = namespace()
    assert scope["_standard_rms_norm"](Rms(dtype), (Var(dtype),), {}) == ("original", dtype)
    assert calls == ["original_hf_forward"]
    assert scope["jt"].flags.amp_reg == 89


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_active_acl_module_call_reaches_the_original_hf_class_forward(dtype):
    scope, Var, Rms, calls, _ = namespace()
    assert scope["_dispatch_module_call"](Rms(dtype), Var(dtype)) == ("original", dtype)
    assert calls == ["original_hf_forward"]


@pytest.mark.parametrize(
    "mode",
    [
        dict(acl=False, cuda=False),
        dict(acl=False, cuda=True),
        dict(acl=True, cuda=False),
        dict(acl=True, cuda=True, compiled=False),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_non_acl_modes_preserve_existing_kernel_resolution(mode, dtype):
    scope, Var, Rms, calls, marker = namespace(**mode)
    assert scope["_standard_rms_norm"](Rms(dtype), (Var(dtype),), {}) is marker
    assert calls == ["bf16_kernel_probe", "training_kernel"]


def test_mixed_dtype_and_unrecognized_signatures_remain_on_original_forward():
    scope, Var, Rms, calls, _ = namespace(acl=False, cuda=False)
    assert scope["_standard_rms_norm"](Rms("float32"), (Var("bfloat16"),), {}) is None
    assert scope["_standard_rms_norm"](Rms("float32"), (), {}) is None
    assert scope["_standard_rms_norm"](Rms("float32"), (Var("float32"),), {"extra": True}) is None
    assert calls == []


def test_instance_forward_still_has_priority_over_all_class_policies():
    scope, Var, Rms, calls, _ = namespace()
    instance = Rms("float32")
    instance.forward = lambda value: "instance_override"
    assert scope["_dispatch_module_call"](instance, Var("float32")) == "instance_override"
    assert calls == []


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("backend,acl,cuda", [("acl", False, False), ("cpu", True, True)])
def test_actual_tensor_backend_has_priority_over_default_flags(dtype, backend, acl, cuda):
    scope, Var, Rms, calls, marker = namespace(backend=backend, acl=acl, cuda=cuda)
    result = scope["_standard_rms_norm"](Rms(dtype), (Var(dtype),), {})
    if backend == "acl":
        assert result == ("original", dtype) and calls == ["original_hf_forward"]
        assert scope["jt"].flags.amp_reg == 89
    else:
        assert result is marker and calls == ["bf16_kernel_probe", "training_kernel"]
