"""Load registration code generation without importing the Jittor runtime."""

import ast
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
JITTOR = ROOT / "python/jittor"


def compiler_state_free_ast(tree):
    """Lift explicit compiler-state reads into the existing fake namespace."""
    class LiftState(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if (node.level == 1 and node.module is None and len(node.names) == 1
                    and node.names[0].name == "compiler"
                    and node.names[0].asname == "_compiler_state"):
                return None
            return node

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "_compiler_state":
                return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
            return self.generic_visit(node)

    return ast.fix_missing_locations(LiftState().visit(tree))


def load_op_registration_generator(cache_path=None):
    compiler_path = JITTOR / "build" / "codegen.py"
    tree = compiler_state_free_ast(ast.parse(compiler_path.read_text(encoding="utf8")))
    names = {"parse_var_members", "gen_jit_op_maker"}
    patterns = {"_VAR_MEMBER_DECL", "_VAR_MEMBER_LOOSE"}
    selected = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in patterns
            for target in node.targets))
    ]
    binding_tree = ast.parse((JITTOR / "build" / "pyjt_compiler.py").read_text(encoding="utf8"))
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
    resource_path = ROOT / "python/jittor/build/utils/backend_resources.py"
    resource_tree = ast.parse(resource_path.read_text(encoding="utf8"))
    resource_functions = [node for node in resource_tree.body
                          if isinstance(node, ast.FunctionDef) and node.name in {"backend_root", "core_root"}]
    exec(compile(ast.Module(body=resource_functions, type_ignores=[]),
                 str(resource_path), "exec"), namespace)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(compiler_path), "exec"), namespace)
    namespace["pyjt_compiler"] = SimpleNamespace(parse_attrs=namespace["parse_attrs"])
    return namespace["gen_jit_op_maker"]
