# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Zheng-Ning Liu <lzhengning@com>
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

""" This file implements generation of stub files for Jittor C extensions.

In detail, autocompletion of the following functions are supported.
- functions in __init__.py
- functions in jittor.core.ops
- attributes of jittor.flags
- methods of jittor.Var

Prerequisite:
- mypy for automatic stub generation, installation: pip install mypy

Usage: python3 tools/build/gen_pyi.py

This is a repository tool: it regenerates a file that is committed and
shipped (``python/jittor/__init__.pyi``), so it runs against a checkout
and does not belong inside the installed package.

"""

import ast
import os
import pprint
import re
import inspect
import subprocess
import sys
import tempfile
from pathlib import Path


def runtime_module(runtime=None):
    if runtime is None:
        import jittor as runtime
    return runtime

def add_indent(s: str, n=1):
    for _ in range(n):
        s = '\t' + s.replace('\n', '\n\t', s.count('\n')-1)
    return s

def ctype_to_python(type_str):
    if type_str == "bool":
        return "bool"
    if type_str in ["int", "int32", "uint", "uint8", "int64", "uint64", "size_t"]:
        return "int"
    if type_str in ["float32", "float64"]:
        return "float"
    if type_str in ["string", "string&&", "NanoString", "char*", "const char*"]:
        return "str"
    if type_str in ["vector<int>"]:
        return "List[int]"
    if type_str in ["vector<string>&&", "vector<NanoString>&&"]:
        return "List[str]"
    if type_str == "VarHolder*":
        return "Var"
    if type_str in ["vector<VarHolder*>", "vector<VarHolder*>&&"]:
        return "List[Var]"
    if type_str in ["vector_to_tuple<VarHolder*>"]:
        return "Tuple[Var, ...]"
    if type_str == "NanoVector":
        return "Tuple[int, ...]"
    if type_str == "vector<NanoVector>&&":
        return "List[Tuple[int, ...]]"
    if type_str in ["FetchFunc", "FetchFunc&&", "NumpyFunc&&"]:
        return "Callable"
    if type_str == "vector<NumpyFunc>&&":
        return "List[Callable]"
    if type_str == "PyObject*":
        # PyObject is an opaque reference here; the concrete converter owns
        # its accepted input protocol (array, dtype, callback, etc.).
        return "object"
    if type_str == "VarSlices&&":
        return "slice"
    if type_str in ["ArrayArgs", "ArrayArgs&&", "DataView"]:
        return "numpy.ndarray"
    if type_str == 'ItemData':
        return "float | int | bool"
    if type_str == "void":
        return ""
    if type_str in ("loop_options_t", "loop_options_t&&"):
        return "Dict[str, int]"
    if type_str in ("DataMap", "DataMap&&"):
        return "Dict[str, float]"
    print(f"[warning] Unknown ctype: {type_str}, do not write type hinting")
    return ""

def cval_to_python(val_str: str, python_type=""):
    if val_str == "{}":
        # C++ aggregate initialization says nothing about a Python dict.
        if python_type.startswith("List["):
            return "[]"
        if python_type.startswith("Tuple["):
            return "()"
        return "..."
    if val_str == "false":
        return "False"
    if val_str == "true":
        return "True"
    if val_str.startswith("ns_"):
        return f'"{val_str[3:]}"'
    if val_str == "NanoVector()":
        return "()"
    return val_str


def run_stubgen(jittor_path, cache_path):
    # The caller explicitly requested generation. Never remove repository
    # caches or share a fixed output directory with another invocation.
    with tempfile.TemporaryDirectory(prefix="jittor-stubs-", dir=cache_path) as stubpath:
        subprocess.run(
            [sys.executable, "-c", "from mypy.stubgen import main; main()",
             "-m", "jittor", "-o", stubpath, "-q"],
            check=True,
        )
        stubfile = Path(stubpath) / "jittor" / "__init__.pyi"
        mypy_content = stubfile.read_text(encoding="utf-8")
    # Remove the follow type redirection
    unused_content = ["ori_int = int\n",
                      "ori_float = float\n",
                      "ori_bool = bool\n",
                      "int = int32\n",
                      "float = float32\n",
                      "double = float64\n",
                      "\nflags: Any\n"]
    for unused in unused_content:
        mypy_content = mypy_content.replace(unused, "")
    Path(jittor_path, "__init__.pyi").write_text(mypy_content, encoding="utf-8")

def gen_ops_stub(jittor_path, runtime=None):
    jittor = runtime_module(runtime)
    f = open(os.path.join(jittor_path, "__init__.pyi"), "a")
    f.write("from typing import List, Tuple, Callable, overload\n")
    f.write("import numpy\n")

    var_hint = "class Var:\n\t'''Variable that stores multi-dimensional data.'''\n"
    var_methods = set()

    def decl_to_param_hints(decl):
        param_decl = re.findall(r".+ [a-zA-Z_0-9]+\((.*)\)", decl)[0]
        if not param_decl.strip():
            return []
        param_hints = []
        for param_str in param_decl.split(','):
            if "=" in param_str:
                template = r"\s*(.+)\s+([a-zA-Z_0-9]+)\s*=\s*(.+)"
                param_type, param_name, param_val = re.findall(template, param_str)[0]
                param_type = ctype_to_python(param_type)
                param_val = cval_to_python(param_val, param_type)
            else:
                param_type, param_name = param_str.strip().rsplit(' ', maxsplit=1)
                param_type = ctype_to_python(param_type)
                param_val = ""

            hint = param_name
            if param_type:
                hint += ": " + param_type
            if param_val:
                hint += "=" + param_val
            param_hints.append(hint)
        return param_hints

    def generate_var_hint(decorators, return_type, param_hints, docstring):
        hint = add_indent(decorators) if decorators else ""
        hint += f"\tdef {func_name}("
        hint += ", ".join(['self'] + param_hints) + ")"
        hint += f"-> {return_type}" if return_type else ""
        hint += ":"
        if docstring:
            hint += add_indent(f"\n'''{docstring}'''\n", 2) + "\t\t...\n"
        else:
            hint += " ...\n"
        return hint

    for func_name, func in jittor.ops.__dict__.items():
        if func_name.startswith("__"):
            continue
        # Exclude a function that overrides the builtin bool:
        #       def bool(x: Var) -> Var: ...
        # It will confuse the IDE. So we ignore this function in pyi.
        if func_name == "bool":
            continue

        docstrings = []
        declarations = []
        for i, doc in enumerate(re.split(r"Declaration:\n(.+)\n", func.__doc__)):
            if i % 2 == 0:
                if not doc.strip() and docstrings:  
                    # if the current docstring is empty, use the last docstring
                    docstrings.append(docstrings[-1])
                else:
                    docstrings.append(doc.replace("'''", '"""').strip())
            else:
                declarations.append(doc)

        for i in range(len(declarations)):
            decl = declarations[i]
            docstring = docstrings[i]

            decorators = "@overload\n" if len(declarations) > 1 else ""
            return_type = ctype_to_python(decl.split(' ', maxsplit=1)[0])
            param_hints = decl_to_param_hints(decl)

            func_text = decorators
            func_text += f"def {func_name}"
            func_text += "(" + ", ".join(param_hints) + ")"
            func_text += f"-> {return_type}" if return_type else ""
            func_text += ":\n"
            if docstring:
                func_text += add_indent(f"'''{docstring}'''\n") + "\t...\n"
            else:
                func_text += " ...\n"

            f.write(func_text)

            if not param_hints or "Var" not in param_hints[0]:
                continue
            var_methods.add(func_name)
            var_hint += generate_var_hint(decorators, return_type, param_hints[1:], docstring)

    for func_name, func in jittor.Var.__dict__.items():
        if func_name.startswith("__") or func_name in var_methods:
            continue
        if func_name in ["int", "float", "double", "bool", "long"]:
            continue
        if func.__doc__ is None:
            continue
        docstring = func.__doc__[:func.__doc__.find("Declaration:")]
        docstring = docstring.replace("'''", '"""').strip()
        declarations = re.findall(r"Declaration:\n(.+)\n", func.__doc__)

        for decl in declarations:
            decl = decl.replace("inline ", "")
            decorators = "@overload\n" if len(declarations) > 1 else ""
            if isinstance(func, property) or inspect.isgetsetdescriptor(func):
                decorators = "@property\n"
            return_type = re.findall(r"(.+) [a-zA-Z_0-9]+\(.*\)", decl)[0].split()[-1]
            return_type = ctype_to_python(return_type)
            param_hints = decl_to_param_hints(decl)

            var_hint += generate_var_hint(decorators, return_type, param_hints, docstring)

    f.write(var_hint)
    f.close()

def gen_flags_stub(jittor_path, runtime=None):
    jittor = runtime_module(runtime)
    f = open(os.path.join(jittor_path, "__init__.pyi"), "a")
    f.write("class Flags:\n")
    f.write("\t'''A set of flags to configure jittor running behaviors'''\n")

    for attr_name, attr in jittor.Flags.__dict__.items():
        if attr_name.startswith("__"):
            continue
        docstring = attr.__doc__
        docstring = attr.__doc__[:attr.__doc__.find("Declaration:")]
        docbody = re.findall(r"\(type.+default.+\):(.+)", docstring)[0].strip()
        docbody += "." if not docbody.endswith('.') else ""
        attr_type, attr_val = re.findall(r"\(type:(.+), default:(.+)\)", docstring)[0]
        attr_type = ctype_to_python(attr_type)
        attr_type = attr_type if attr_type else "Any"
        f.write(f"\t{attr_name}: {attr_type}\n")
        f.write(f"\t'''{docbody} Default: {attr_val}'''\n")

    f.write("flags: Flags\n")
    f.write("'''Jittor running time flags instance'''\n")
    f.close()


def synchronize_public_exports(jittor_path, runtime=None):
    """Make the generated stub describe exactly the declared root surface."""
    jittor = runtime_module(runtime)
    stub_path = os.path.join(jittor_path, "__init__.pyi")
    with open(stub_path, encoding="utf-8") as handle:
        content = handle.read()
    content = "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
    tree = ast.parse(content)
    declared = set()
    stub_exports = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_") or node.name == "__version__":
                declared.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in targets
            ):
                stub_exports.append(tuple(ast.literal_eval(node.value)))
            declared.update(
                target.id for target in targets
                if isinstance(target, ast.Name)
                and (not target.id.startswith("_") or target.id == "__version__")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module in ("typing", "collections", "collections.abc"):
                continue
            for alias in node.names:
                if alias.name != "*":
                    declared.add(alias.asname or alias.name)
                elif node.module == "jittor_core":
                    declared.update(
                        name for name in dir(jittor.jittor_core)
                        if not name.startswith("_"))
                elif node.module == "jittor_core.ops":
                    declared.update(
                        name for name in dir(jittor.ops)
                        if not name.startswith("_"))

    exports = tuple(jittor.__all__)
    if stub_exports and stub_exports != [exports]:
        raise RuntimeError("generated stub __all__ does not match jittor.__all__")
    missing = sorted(set(exports) - declared)
    manifest = (
        "__all__ = " + pprint.pformat(exports, width=88) + "\n\n"
        if not stub_exports else ""
    )
    fallback = ""
    if missing:
        fallback = "\n# Public names whose precise type is not inferred yet.\n"
        fallback += "\n".join("{}: Any".format(name) for name in missing) + "\n"
    with open(stub_path, "w", encoding="utf-8") as handle:
        handle.write(manifest)
        handle.write(content)
        handle.write(fallback)

def binding_properties(header):
    """Read the same explicit property markers consumed by pyjt."""
    result = {}
    pattern = r"//\s*@pyjt\(([^)]+)\)\s*(?:inline\s+)?(\w+)\s+\w+\("
    for names, cpp_type in re.findall(pattern, header):
        for name in names.split(","):
            name = name.strip()
            if name.startswith("__get__"):
                result.setdefault(name[7:], {})["type"] = (
                    "NanoString" if cpp_type == "NanoString" else ctype_to_python(cpp_type)
                )
            elif name.startswith("__set__"):
                result.setdefault(name[7:], {})["writable"] = True
    return result


def repair_stub_content(content, properties):
    """Normalize generated declarations without importing or executing Jittor."""
    content = content.replace(", numpy2cupy as numpy2cupy", "")
    content = content.replace("from .misc.concatenation import", "from .ops.concatenation import")
    content = content.replace(
        "from .nn import attention as attention, baddbmm as baddbmm, bmm as bmm, bmm_transpose as bmm_transpose, matmul as matmul",
        "from .nn import attention as attention\nfrom .nn.functional.matrix import baddbmm as baddbmm, bmm as bmm, bmm_transpose as bmm_transpose, matmul as matmul",
    )
    if "import builtins as _builtins" not in content:
        content = "import builtins as _builtins\nimport numpy as _numpy\nfrom typing import Dict, MutableMapping\n" + content
    if not re.search(r"^(?:def numpy2cupy\(|numpy2cupy:)", content, re.M):
        content += "\n# Optional CuPy adapter mutates numpy-code argument records in place.\nnumpy2cupy: Optional[Callable[[object, MutableMapping[str, object]], None]]\n"
    tree = ast.parse(content)
    lines = content.splitlines(True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    def location(node):
        return offsets[node.lineno - 1] + node.col_offset, offsets[node.end_lineno - 1] + node.end_col_offset
    def annotation_text(node):
        a, b = location(node)
        text = content[a:b]
        text = re.sub(r"\bTuple\[(int|Var)\]", r"Tuple[\1, ...]", text)
        text = re.sub(r"(?<![\w.])(int|float|bool)\b", r"_builtins.\1", text)
        return re.sub(r"(?<![\w.])numpy\.", "_numpy.", text)
    imported = {
        alias.asname or alias.name.split(".")[0]
        for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in imported:
            a = offsets[node.lineno - 1]
            b = offsets[node.end_lineno]
            edits.append((a, b, ""))
    var = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Var")
    def is_setter(node):
        return isinstance(node, ast.FunctionDef) and any(
            isinstance(d, ast.Attribute) and d.attr == "setter" for d in node.decorator_list
        )
    methods = {node.name: node for node in var.body if isinstance(node, ast.FunctionDef) and not is_setter(node)}
    setters = [node for node in var.body if is_setter(node) and node.name in properties]
    for setter in setters:
        a = offsets[min(d.lineno for d in setter.decorator_list) - 1]
        edits.append((a, offsets[setter.end_lineno], ""))
    for scope in [tree, var]:
        definitions = [n for n in scope.body if isinstance(n, ast.FunctionDef)]
        for node in definitions:
            if sum(n.name == node.name for n in definitions) == 1:
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "overload":
                        edits.append((offsets[decorator.lineno - 1], offsets[decorator.lineno], ""))
    for node in ast.walk(tree):
        if any(node is child for setter in setters for child in ast.walk(setter)):
            continue
        annotation = node.annotation if isinstance(node, (ast.arg, ast.AnnAssign)) else node.returns if isinstance(node, ast.FunctionDef) else None
        if annotation is not None:
            # Property types are supplied by their actual C++ getter declaration.
            if isinstance(node, ast.FunctionDef) and node in var.body and node.name in properties:
                continue
            a, b = location(annotation)
            edits.append((a, b, annotation_text(annotation)))
        if isinstance(node, ast.FunctionDef):
            params = getattr(node.args, "posonlyargs", []) + node.args.args
            for arg, default in zip(params[-len(node.args.defaults):], node.args.defaults):
                if isinstance(default, ast.Dict) and not default.keys:
                    typ = annotation_text(arg.annotation) if arg.annotation else ""
                    if typ.startswith("List["):
                        a, b = location(default)
                        edits.append((a, b, "[]"))
    additions = []
    for name, spec in properties.items():
        typ = spec.get("type")
        if not typ:
            continue
        typ = re.sub(r"(?<![\w.])(int|float|bool)\b", r"_builtins.\1", typ).replace("numpy.", "_numpy.")
        node = methods.get(name)
        if node is not None:
            indent = lines[node.lineno - 1][:node.col_offset]
            if not any(isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list):
                a = offsets[node.lineno - 1]
                edits.append((a, a, indent + "@property\n"))
            if node.returns:
                a, b = location(node.returns)
                edits.append((a, b, typ))
            else:
                a = content.index(")", offsets[node.lineno - 1]) + 1
                edits.append((a, a, " -> " + typ))
        else:
            additions.append("\t@property\n\tdef " + name + "(self) -> " + typ + ": ...\n")
        if spec.get("writable"):
            setter = "\t@" + name + ".setter\n\tdef " + name + "(self, value: " + typ + ") -> None: ...\n"
            if node is not None:
                a = offsets[node.end_lineno]
                edits.append((a, a, setter))
            else:
                additions.append(setter)
    if additions:
        a = offsets[var.end_lineno]
        edits.append((a, a, "".join(additions)))
    for a, b, replacement in sorted(edits, reverse=True):
        content = content[:a] + replacement + content[b:]
    ast.parse(content)
    return content


def repair_existing_stub(path):
    path = Path(path)
    header = path.parents[2] / "src/core/var_holder.h"
    content = repair_stub_content(path.read_text(encoding="utf-8"), binding_properties(header.read_text(encoding="utf-8")))
    path.write_text(content, encoding="utf-8")


def get_pyi(jittor_path=None, cache_path=None):
    jittor = runtime_module()
    if jittor_path is None:
        jittor_path = jittor.flags.jittor_path
    if cache_path is None:
        import jittor_utils
        cache_path = jittor_utils.cache_path

    run_stubgen(jittor_path, cache_path)
    gen_ops_stub(jittor_path)
    gen_flags_stub(jittor_path)
    synchronize_public_exports(jittor_path)
    repair_existing_stub(Path(jittor_path) / "__init__.pyi")

    print(f"Generated stubfile: {os.path.join(jittor_path, '__init__.pyi')}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--repair-existing":
        repair_existing_stub(sys.argv[2])
    else:
        get_pyi()
