"""Exercise the bridge's real binding path without a compiler or GPU launch."""
import ast
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import types
from unittest import mock

import pytest


class Operand:
    device_id = 0


class CapturedCompile(Exception):
    pass


@pytest.fixture
def bind_launch():
    path = Path(__file__).resolve().parents[2] / "triton/backend.py"
    tree = ast.parse(path.read_text())
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and node.name in {"_run", "_scalar_sig", "_launch_alignment", "JittorTritonError"}]
    namespace = {
        "os": os,
        "_is_tensor": lambda value: isinstance(value, Operand),
        "_tensor_is_cuda": lambda value: True,
        "_ptr_sig": lambda value: "*fp32",
        "_tensor_ptr": lambda value: 0x1000,
        "_is_var": lambda value: False,
        "_stats_on": lambda: False,
        "_fast_sync_enabled": lambda: True,
        "_sync_before_launch_enabled": lambda fast_sync: False,
        "_Driver": types.SimpleNamespace(get=lambda device: types.SimpleNamespace(ensure_ctx=lambda: None)),
    }
    spec = importlib.util.spec_from_file_location(
        "triton_argument_binding", path.with_name("argument_binding.py"))
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    namespace["bind_arguments"] = binding.bind_arguments
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)

    def compile_kernel(kernel, signature, constants, options, device, aligned=()):
        raise CapturedCompile(signature, constants, options, device)

    namespace["_compile"] = compile_kernel

    def capture(kernel, args, kwargs):
        jittor = types.ModuleType("jittor")
        jittor.sync_all = mock.Mock()
        with mock.patch.dict(sys.modules, {"jittor": jittor}):
            with pytest.raises(CapturedCompile) as captured:
                namespace["_run"](kernel, args, kwargs, (1,))
        return captured.value.args

    capture.error = namespace["JittorTritonError"]
    return capture


def test_cached_signature_binds_defaults_keywords_and_constexpr_without_inspection(bind_launch):
    def function(pointer, /, count=3, *, optional=None, BLOCK: "tl.constexpr" = 64):
        pass

    kernel = types.SimpleNamespace(fn=function, signature=inspect.signature(function))
    with mock.patch.object(inspect, "signature", side_effect=AssertionError("reinspected kernel")):
        signature, constants, options, device = bind_launch(kernel, (Operand(),), {"num_warps": 4})
        assert signature == {"pointer": "*fp32", "count": "i32", "optional": "constexpr", "BLOCK": "constexpr"}
        assert constants == {"optional": None, "BLOCK": 64}
        assert options == {"num_warps": 4}
        assert device == 0
        _, constants, _, _ = bind_launch(kernel, (Operand(), 5), {"BLOCK": 128})
        assert constants == {"optional": None, "BLOCK": 128}
        for args, kwargs in (((Operand(), 5), {"count": 6}), ((), {"pointer": Operand()}),
                             ((Operand(),), {"unknown": 1}), ((), {})):
            with pytest.raises(bind_launch.error, match="could not bind args"):
                bind_launch(kernel, args, kwargs)


def test_upstream_param_constexpr_metadata_remains_authoritative(bind_launch):
    def function(pointer, count=3):
        pass

    kernel = types.SimpleNamespace(
        fn=function, signature=inspect.signature(function),
        params=[types.SimpleNamespace(name="pointer", is_constexpr=False),
                types.SimpleNamespace(name="count", is_constexpr=True)])
    _, constants, _, _ = bind_launch(kernel, (Operand(),), {})
    assert constants == {"count": 3}


def test_cached_signature_lifetime_matches_upstream_and_replacement_is_observed(bind_launch):
    def function(pointer, BLOCK: "tl.constexpr" = 64):
        pass

    kernel = types.SimpleNamespace(fn=function, signature=inspect.signature(function))
    function.__defaults__ = (128,)
    assert bind_launch(kernel, (Operand(),), {})[1] == {"BLOCK": 64}
    kernel.signature = inspect.signature(function)
    assert bind_launch(kernel, (Operand(),), {})[1] == {"BLOCK": 128}


@pytest.mark.parametrize("signature", [None, "unrelated callable attribute"])
def test_uncached_callable_inspection_stays_live_and_runs_once(bind_launch, signature):
    def function(pointer, BLOCK: "tl.constexpr" = 64):
        pass

    kernel = types.SimpleNamespace(fn=function, signature=signature)
    with mock.patch.object(inspect, "signature", wraps=inspect.signature) as inspect_call:
        assert bind_launch(kernel, (Operand(),), {})[1] == {"BLOCK": 64}
        function.__defaults__ = (128,)
        assert bind_launch(kernel, (Operand(),), {})[1] == {"BLOCK": 128}
        assert inspect_call.call_count == 2


def test_bare_callable_remains_supported(bind_launch):
    def function(pointer, BLOCK: "tl.constexpr" = 64):
        pass

    assert bind_launch(function, (Operand(),), {})[1] == {"BLOCK": 64}


def test_tensor_inventory_preserves_device_and_missing_operand_checks(bind_launch):
    def function(left, right=None):
        pass

    kernel = types.SimpleNamespace(fn=function, signature=inspect.signature(function))
    other = Operand()
    other.device_id = 1
    with pytest.raises(bind_launch.error, match="operands on devices 0 and 1"):
        bind_launch(kernel, (Operand(), other), {})
    with pytest.raises(bind_launch.error, match="no tensor arguments"):
        bind_launch(kernel, (3,), {})
    assert bind_launch(kernel, (other, None), {})[3] == 1
