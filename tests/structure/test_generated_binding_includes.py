"""Quoted core includes in binding templates resolve after physical moves."""

from pathlib import Path
import re
import ast
import os


ROOT = Path(__file__).resolve().parents[2]


def test_binding_templates_reference_real_core_headers():
    for relative in ("python/jittor/build/compiler.py", "python/jittor/build/codegen.py",
                     "python/jittor/build/compilation.py", "python/jittor/build/pyjt_compiler.py",
                     "python/jittor/build/utils/__init__.py", "src/third_party/miniz.h"):
        source = (ROOT / relative).read_text()
        for include in re.findall(r'#include "([^"{}]+)"', source):
            assert (ROOT / "src" / include).is_file(), (relative, include)


def test_core_stamp_scans_relocated_core(tmp_path):
    path = ROOT / "python/jittor/build/compiler.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "core_source_signature")
    core = tmp_path / "src"
    (core / "core").mkdir(parents=True)
    header = core / "core/common.h"
    header.write_text("first")
    def missing_backend(*args):
        raise FileNotFoundError
    namespace = dict(os=os, jittor_path=str(tmp_path / "python/jittor"),
                     core_root=lambda tree: str(core), backend_root=missing_backend)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    signature = namespace["core_source_signature"]
    before = signature()
    assert "src/core/common.h" in before
    header.write_text("second version")
    assert signature() != before
