"""Exercise native operator registration generation without importing Jittor."""

import ast
from pathlib import Path
import re

import pytest

from _helpers.op_registration_generator import load_op_registration_generator as _load_generator


ROOT = Path(__file__).resolve().parents[2]
JITTOR = ROOT / "python/jittor"


@pytest.mark.parametrize("backend,mask", [
    (None, None),
    ("cpu", "OpBackendCpu"),
    ("accelerator", "OpBackendAccelerator"),
    ("both", "OpBackendAny"),
])
def test_generator_registers_typed_definition_with_explicit_or_class_backend(backend, mask, tmp_path):
    header = JITTOR / "src/ops/array_op.h"
    source = _load_generator(tmp_path)([str(header)], backend=backend)
    assert '#include "ops/op_registration.h"' in source
    registration = next(line.strip() for line in source.splitlines()
                        if "register_op_definition<ArrayOp>" in line)
    assert f'R"({header.with_suffix(".cc")})"' in registration
    assert "op_constructor_entry(&make_array)" in registration
    assert "VAR_MEMBER_NAME_AND_OFFSET(output, ArrayOp)" in registration
    assert registration.endswith(f", {mask});" if mask else "});")
    assert "op_registe(" not in source


def test_generator_rejects_unknown_backend_before_reading_headers():
    with pytest.raises(ValueError, match="backend must be"):
        _load_generator()(["missing_op.h"], backend="typo")


@pytest.mark.parametrize("relative", ["python/jittor/src/ops", "backends/cuda/kernels/cublas",
                                      "python/jittor/extern/mkl/ops"])
def test_generator_preserves_every_operator_definition(relative, tmp_path):
    headers = sorted((ROOT / relative).glob("*_op.h"))
    assert headers
    source = _load_generator(tmp_path)([str(header) for header in headers], export="registration_test")
    registered = re.findall(r'register_op_definition<\w+>\(\{ "([^"]+)"', source)
    expected = [header.stem[:-3] for header in headers]
    assert registered == expected
    assert "PYJT_MODULE_INIT(registration_test)" in source


def test_custom_library_forwards_backend_into_registration_generator():
    tree = ast.parse((JITTOR / "compiler.py").read_text(encoding="utf8"))
    custom = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "compile_custom_ops")
    assert custom.args.args[-1].arg == "backend"
    assert isinstance(custom.args.defaults[-1], ast.Constant)
    assert custom.args.defaults[-1].value is None
    calls = [node for node in ast.walk(custom) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "gen_jit_op_maker"]
    assert len(calls) == 1
    argument = next(arg.value for arg in calls[0].keywords if arg.arg == "backend")
    assert isinstance(argument, ast.Name) and argument.id == "backend"


@pytest.mark.parametrize("relative,expected", [
    ("compile_extern.py", {"mkl": "cpu", "culib": "accelerator", "cutt": "accelerator",
                           "nccl": "accelerator", "hccl": "accelerator", "mpi": None}),
    ("extern/rocm/rocm_compiler.py", {"rocmlib": "accelerator"}),
])
def test_optional_libraries_declare_their_actual_backend_family(relative, expected):
    tree = ast.parse((JITTOR / relative).read_text(encoding="utf8"))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        function = node.value.func
        name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", None)
        if name != "compile_custom_ops":
            continue
        target = node.targets[0].id
        assert target in expected
        backend = next((ast.literal_eval(arg.value) for arg in node.value.keywords
                        if arg.arg == "backend"), None)
        assert backend == expected[target]
        found.add(target)
    assert found == set(expected)
