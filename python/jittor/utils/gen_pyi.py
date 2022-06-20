# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved.
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
- mypy for automatic stub generation

Usage: python3 -m jittor.utils.gen_pyi

"""

import os
import re
import shutil
import jittor

def add_indent(s: str, n=1):
    for _ in range(n):
        s = '\t' + s.replace('\n', '\n\t', s.count('\n')-1)
    return s

def ctype_to_python(type_str):
    if type_str == "bool":
        return "bool"
    if type_str in ["int", "uint", "int64", "uint64", "size_t"]:
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
    if type_str == "NanoVector":
        return "Tuple[int]"
    if type_str == "vector<NanoVector>&&":
        return "List[Tuple[int]]"
    if type_str in ["FetchFunc", "FetchFunc&&", "NumpyFunc&&"]:
        return "Callable"
    if type_str == "vector<NumpyFunc>&&":
        return "List[Callable]"
    if type_str == "PyObject*":
        return "float | int | numpy.ndarray | Var"
    if type_str == "VarSlices&&":
        return "slice"
    if type_str in ["ArrayArgs", "ArrayArgs&&", "DataView"]:
        return "numpy.ndarray"
    if type_str == 'ItemData':
        return "float | int | bool"
    if type_str == "void":
        return ""
    print(f"[warning] Unknown ctype: {type_str}, do not write type hinting")
    return ""

def cval_to_python(val_str: str):
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

    # for __init__.py functions
    stubpath = os.path.join(cache_path, 'stubs')
    stubfile = os.path.join(stubpath, "jittor", "__init__.pyi")
    os.system(f"stubgen -m jittor -o {stubpath} -q")
    with open(stubfile) as f:
        mypy_content = f.read()

    f = open(stubfile, "w")
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
    f.write(mypy_content)

    shutil.move(stubfile, os.path.join(jittor_path, "__init__.pyi"))
    shutil.rmtree(stubpath)
    shutil.rmtree(os.path.expanduser(".mypy_cache"))

def gen_ops_stub(jittor_path):
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
                param_val = cval_to_python(param_val)
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
            hint += f" ...\n"
        return hint

    for func_name, func in jittor.ops.__dict__.items():
        if func_name.startswith("__"):
            continue
        # Exclude a function that overrides the builtin bool:
        #       def bool(x: Var) -> Var: ...
        # It will confuse the IDE. So we ignore this function in pyi.
        if func_name == "bool":
            continue

        docstring = func.__doc__[:func.__doc__.find("Declaration:")]
        docstring = docstring.replace("'''", '"""').strip()
        declarations = re.findall(r"Declaration:\n(.+)\n", func.__doc__)

        for decl in declarations:
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
                func_text += f" ...\n"

            f.write(func_text)

            if not "Var" in param_hints[0]:
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
            return_type = re.findall(r"(.+) [a-zA-Z_0-9]+\(.*\)", decl)[0].split()[-1]
            return_type = ctype_to_python(return_type)
            param_hints = decl_to_param_hints(decl)

            var_hint += generate_var_hint(decorators, return_type, param_hints, docstring)

    f.write(var_hint)
    f.close()

def gen_flags_stub(jittor_path):
    f = open(os.path.join(jittor_path, "__init__.pyi"), "a")
    f.write("class Flags:\n")
    f.write("\t'''A set of flags to configure jittor running behaviors'''\n")

    for attr_name, attr in jittor.Flags.__dict__.items():
        if attr_name.startswith("__"):
            continue
        docstring = attr.__doc__
        docstring = attr.__doc__[:attr.__doc__.find("Declaration:")]
        docbody = re.findall("\(type.+default.+\):(.+)", docstring)[0].strip()
        docbody += "." if not docbody.endswith('.') else ""
        attr_type, attr_val = re.findall(r"\(type:(.+), default:(.+)\)", docstring)[0]
        attr_type = ctype_to_python(attr_type)
        attr_type = attr_type if attr_type else "Any"
        f.write(f"\t{attr_name}: {attr_type}\n")
        f.write(f"\t'''{docbody} Default: {attr_val}'''\n")

    f.write("flags: Flags\n")
    f.write("'''Jittor running time flags instance'''\n")
    f.close()

def get_pyi(jittor_path=None, cache_path=None):
    if jittor_path is None:
        jittor_path = jittor.flags.jittor_path
    if cache_path is None:
        import jittor_utils
        cache_path = jittor_utils.cache_path

    run_stubgen(jittor_path, cache_path)
    gen_ops_stub(jittor_path)
    gen_flags_stub(jittor_path)

    print(f"Generated stubfile: {os.path.join(jittor_path, '__init__.pyi')}")


if __name__ == "__main__":
    get_pyi()