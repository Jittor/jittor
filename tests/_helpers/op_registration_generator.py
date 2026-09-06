"""Load registration code generation without importing the Jittor runtime."""

import ast
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
JITTOR = ROOT / "python/jittor"


def load_op_registration_generator(cache_path=None):
    compiler_path = JITTOR / "compiler.py"
    tree = ast.parse(compiler_path.read_text(encoding="utf8"))
    names = {"parse_var_members", "gen_jit_op_maker"}
    patterns = {"_VAR_MEMBER_DECL", "_VAR_MEMBER_LOOSE"}
    selected = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in patterns
            for target in node.targets))
    ]
    binding_tree = ast.parse((JITTOR / "pyjt_compiler.py").read_text(encoding="utf8"))
    selected.extend(node for node in binding_tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "parse_attrs")
    namespace = {
        "os": os,
        "re": re,
        "json": json,
        "cache_path": os.fspath(cache_path) if cache_path is not None else None,
        "jittor_path": str(JITTOR),
        "LOG": SimpleNamespace(vv=lambda *args: None, vvvv=lambda *args: None),
    }
    resource_path = ROOT / "python/jittor_utils/backend_resources.py"
    resource_tree = ast.parse(resource_path.read_text(encoding="utf8"))
    resource_functions = [node for node in resource_tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == "backend_root"]
    exec(compile(ast.Module(body=resource_functions, type_ignores=[]),
                 str(resource_path), "exec"), namespace)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(compiler_path), "exec"), namespace)
    namespace["pyjt_compiler"] = SimpleNamespace(parse_attrs=namespace["parse_attrs"])
    return namespace["gen_jit_op_maker"]
