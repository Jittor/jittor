# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import subprocess as sp
import json
import os
import re
import sys
import glob
import inspect
import datetime
import threading
import platform
import ctypes
from ctypes import cdll
from ctypes.util import find_library

import jittor_utils as jit_utils
from jittor_utils import (
    LOG,
    run_cmd,
    find_exe,
    try_find_exe,
    # There used to be a byte-identical copy of this in this module, shadowing
    # the one here; fixing one meant missing the other.
    env_or_try_find,
    cc_path,
    cc_type,
    cache_path,
)
from jittor_utils import env_config
from jittor_utils.env_config import build_env, build_flag
from jittor_utils.compiler_flags import remove_flags, shsplit
from . import pyjt_compiler
from ._runtime.flag_policy import flag_category
from jittor_utils import lock
from jittor_utils import install_cuda
from jittor import __version__
import hashlib
from functools import partial
from jittor_utils import backend_discovery as _backend_discovery
from jittor_utils import build_config as _build_config_api
from jittor_utils import backend_resources as _backend_resources_api
from jittor_utils.backend_resources import backend_root, core_root
from jittor_utils.build_config import BuildConfig, BuildContext, BuildSource, ModuleBuildServices


def _module_build_services(config):
    return ModuleBuildServices(
        pyjt_compiler.compile_single, fix_cl_flags, config.cc_path,
        jit_utils.cache_path, config.jittor_path,
    )


def make_backend_context(config=None, *, publish_library=None, mpi_compile_flags=""):
    config = build_config if config is None else config
    return BuildContext(
        config=config,
        compile_module=partial(jit_utils.compile_module, services=_module_build_services(config)),
        compile=compile, compile_custom_ops=compile_custom_ops,
        publish_library=publish_library, make_cache_dir=make_cache_dir,
        load_library=ctypes.CDLL, mpi_compile_flags=mpi_compile_flags, so=so,
        native_core=globals().get("core"),
    )


def compile_backend_sources(config, common_flags):
    objects, commands = [], []
    directory = os.path.join(config.cache_path, "obj_files", "backends")
    os.makedirs(directory, exist_ok=True)
    for source in config.backend_sources:
        path = os.path.abspath(source.path)
        identity = json.dumps((path, source.language, source.flags, source.compiler),
                              separators=(",", ":"))
        tag = hashlib.sha256(identity.encode("utf8")).hexdigest()[:12]
        output = os.path.join(directory, os.path.basename(path) + "." + tag + ".o")
        flags = remove_flags(common_flags + " " + source.flags,
                             ['-l', '-L', '-Wl,', '.lib', '-shared'])
        driver = source.compiler or config.cc_path
        if source.language in ("cuda", "hip"):
            driver = source.compiler or config.kernel_compiler or config.nvcc_path
            if not driver:
                raise RuntimeError("backend source requires a compiler: " + path)
            if config.convert_nvcc_flags is not None:
                flags = config.convert_nvcc_flags(flags)
        if "nan_checker" in path:
            flags = remove_flags(flags, ["--use_fast_math", "-Ofast"]) + " -O2 "
        command = '"%s" "%s" %s -c -o "%s"' % (driver, path, flags, output)
        commands.append(fix_cl_flags(command))
        objects.append(output)
    if commands:
        jit_utils.run_cmds(commands, config.cache_path, config.jittor_path,
                           "Compiling " + config.backend + " backend")
    return objects

def find_jittor_path():
    return os.path.dirname(__file__)

def make_cache_dir(cache_path):
    if not os.path.isdir(cache_path):
        LOG.i(f"Create cache dir: {cache_path}")
        os.mkdir(cache_path)

def moveback_flags(flags, rm_flags):
    flags = shsplit(flags)
    output = []
    output2 = []
    for s in flags:
        ss = s.replace("\"", "")
        for rm in rm_flags:
            if ss.startswith(rm) or ss.endswith(rm):
                output2.append(s)
                break
        else:
            output.append(s)
    return " ".join(output+output2)

def map_flags(flags, func):
    flags = shsplit(flags)
    output = []
    for s in flags:
        output.append(func(s))
    return " ".join(output)

def _source_path(name):
    """Resolve a source named relative to the package, or pass an absolute one.

    `4.15` moved the C++ core out of the Python package to the top level, so a
    name beginning with ``src/`` no longer sits under ``jittor_path``. Routing
    it through ``core_root`` here is why the dozens of relative names in the
    core file lists below did not each have to learn where the core went --
    and why ``jit_utils_core_files`` can stay in the same spelling as the list
    it is later removed from (``files.remove`` matches strings, not paths).
    """
    if os.path.isabs(name):
        return name
    head = name.replace("\\", "/").split("/", 1)
    if head[0] == "src" and len(head) == 2:
        return os.path.join(core_root(jittor_path), head[1])
    return os.path.join(jittor_path, name)


def compile(compiler, flags, inputs, output, combind_build=False, cuda_flags="", obj_dirname="obj_files", return_cmd=False):
    def do_compile(cmd):
        if jit_utils.cc:
            return jit_utils.cc.cache_compile(cmd, cache_path, jittor_path)
        else:
            run_cmd(cmd)
            return True
    base_output = os.path.basename(output).split('.')[0]
    if os.name == 'nt':
        # windows do not combind build, need gen def
        combind_build = False
        # windows need xxxx.lib
        afile = output.rsplit('.', 1)[0] + ".lib"
        afile = os.path.join(cache_path, afile)
        # The MinGW branch that used to live here read a name (`link`) that is
        # defined neither in this function nor at module scope, so executing it
        # raised NameError unconditionally: it has never run. Rather than leave
        # code that cannot work, Windows builds go through cl.
    if not os.path.isabs(output):
        output = os.path.join(cache_path, output)
    # don't recompile object file in inputs
    obj_files = []
    ex_obj_files = []
    new_inputs = []
    obj_dir = os.path.join(cache_path, obj_dirname)
    os.makedirs(obj_dir, exist_ok=True)
    for name in inputs:
        if name[-1] in 'oab':
            ex_obj_files.append(name)
        else:
            new_inputs.append(_source_path(name))
            obj_files.append(os.path.join(
                obj_dir, os.path.basename(name)+".o"))
    inputs = new_inputs
    cm = lambda s: f"\"{s}\""
    cms = lambda arr: [f"\"{s}\"" for s in arr ]

    if len(inputs) == 1 or combind_build:
        cmd = f"\"{compiler}\" {' '.join(cms(inputs))} {flags} -o {cm(output)}"
        if return_cmd:
            return fix_cl_flags(cmd)
        return do_compile(fix_cl_flags(cmd))
    if return_cmd:
        # The split compile/link path runs several commands; there is no single
        # one to hand back, and silently compiling instead would be worse.
        raise ValueError("return_cmd requires a single-command build")
    # split compile object file and link
    # remove -l -L flags when compile object files
    oflags = remove_flags(flags, ['-l', '-L', '-Wl,', '.lib', '-shared'])
    cmds = []
    for input, obj_file in zip(inputs, obj_files):
        cc = compiler
        nflags = oflags
        cmd = f"{cm(input)} {nflags} {lto_flags} -c -o {cm(obj_file)}"
        if input.endswith(".cu"):
            if has_cuda or has_rocm:
                cmd = f"\"{nvcc_path}\" {cuda_flags} {cmd}"
                cmd = convert_nvcc_flags(fix_cl_flags(cmd))
            else:
                continue
        else:
            cmd = f"\"{cc}\" {cmd}"
            cmd = fix_cl_flags(cmd)
        if "nan_checker" in input:
            # nan checker needs to disable fast_math 
            if "--use_fast_math" in cmd:
                cmd = cmd.replace("--use_fast_math", "")
            if "-Ofast" in cmd:
                cmd = cmd.replace("-Ofast", "-O2")
        cmds.append(cmd)
    jit_utils.run_cmds(cmds, cache_path, jittor_path, "Compiling "+base_output)
    obj_files += ex_obj_files
    if os.name == 'nt':
        # dumpdef is a Windows build resource and ships with the build package.
        # Keep the lookup relative to the installed package so wheel builds do
        # not depend on repository-only tools/ files.
        dumpdef_path = os.path.join(jittor_path, "build", "dumpdef.py")
        cmd = f"\"{sys.executable}\" \"{dumpdef_path}\" {' '.join(cms(obj_files))} -Fo: \"{output}.def\""
        do_compile(fix_cl_flags(cmd))
    cmd = f"\"{compiler}\" {' '.join(cms(obj_files))} -o {cm(output)} {flags} {lto_flags}"
    return do_compile(fix_cl_flags(cmd))

def gen_jit_tests():
    all_src = glob.glob(core_root(jittor_path)+"/**/*.cc", recursive=True)
    jit_declares = []
    re_def = re.compile("JIT_TEST\\((.*?)\\)")
    names = set()
    test_defs = []
    
    for src_name in all_src:
        with open(src_name, 'rb') as f:
            src = f.read().decode('utf8')
        defs = re_def.findall(src)
        for name in defs:
            LOG.vv(f"Find test {name} from {src_name}")
            assert name not in names, f"Conflict test name {name}"
            names.add(name)
            jit_declares.append(f"JIT_TEST({name});")
            test_defs.append(f"""
                /* From {src_name} */
                // @pyjt({name})
                static inline void test_{name}() {{ jit_test_{name}(); }} 
            """)
    
    jit_declares = "\n    ".join(jit_declares)
    jit_src = f"""
    #pragma once
    #include "core/common.h"

    void expect_error(std::function<void()> func) {{
        try {{ func(); }}
        catch (...) {{ return; }}
        CHECK(0) << "Missing error";
    }}

    namespace jittor {{
    
    {jit_declares}

    // @pyjt(tests)
    // @attrs(submodule)
    namespace tests {{
        {"".join(test_defs)}
    }}

    }} // jittor
    """
    LOG.vvvv(jit_src)
    with open(os.path.join(cache_path, "gen", "jit_tests.h"), 'w', encoding='utf8') as f:
        f.write(jit_src)

def strip_cxx_comments(src):
    """Remove /* */ and // comments, so a scanner reads only live code.

    ``gen_jit_flags`` regexes DEFINE_FLAG out of the sources. Without this,
    a *commented-out* definition is indistinguishable from a real one: 27
    lines of commented-out flags in src/utils/flags.cc gave twelve flags a
    second definition, and which default and doc string reached ``jt.flags``
    depended on the order glob happened to return the files in.
    """
    out = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == '\\' else 1
            out.append(src[i:min(j+1, n)])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i+2)
            i = n if j < 0 else j + 2
            out.append(" ")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def gen_jit_flags():
    all_src = glob.glob(core_root(jittor_path)+"/**/*.cc", recursive=True)
    jit_declares = []
    re_def = re.compile("DEFINE_(RUNTIME_)?FLAG(_WITH_SETTER)?\\((.*?)\\);", re.DOTALL)

    flags_defs = []
    visit = {}
    
    for src_name in all_src:
        with open(src_name, 'rb') as f:
            src = f.read().decode("utf8")
        defs = re_def.findall(strip_cxx_comments(src))
        for runtime, _, args in defs:
            args = args.split(",")
            type = args[0].strip()
            name = args[1].strip()
            if not has_cuda and "cuda" in name and name not in ("use_cuda", "cuda_kernel_math"):
                if name != "use_cuda_host_allocator":
                    continue
            default = args[2].strip()
            doc = ",".join(args[3:])
            doc = eval(f"({doc})")
            LOG.vv(f"Find define {name} from {src_name}")
            if name in visit:
                continue
            visit[name] = 1
            category = flag_category(name)
            declaration = "DECLARE_RUNTIME_FLAG" if runtime else "DECLARE_FLAG"
            getter = f"runtime_flag_{name}()" if runtime else name
            jit_declares.append(f"{declaration}({type}, {name});")
            alias = []
            if name == "use_cuda":
                alias = ["use_device", "use_acl", "use_rocm", "use_corex"]
            elif name == "auto_mixed_precision_level":
                alias = ["amp_level"]
            deprecated_aliases = alias if name == "use_cuda" else []
            binding_names = [name] if deprecated_aliases else [name] + alias
            get_names = ",".join(["__get__"+a for a in binding_names])
            set_names = ",".join(["__set__"+a for a in binding_names])
            guard = f'check_startup_config_write("{name}"); ' if category == "startup" else ""
            setter = f"{guard}set_{name}(v);"
            if category == "counter":
                setter = f'throw std::runtime_error("{name} is a read-only runtime counter");'
            flags_defs.append(f"""
                /* {name}(type:{type}, default:{default}): {doc} */
                // @pyjt({get_names})
                {type} _get_{name}() {{ return {getter}; }}
                // @pyjt({set_names})
                void _set_{name}({type} v) {{ {setter} }}
                {f'''// @pyjt({set_names})
                void _set_{name}(bool v) {{ {setter} }}
                ''' if type=="int" else ""}
            """)
            for alias_name in deprecated_aliases:
                warning = (
                    f'if (PyErr_WarnEx(PyExc_DeprecationWarning, "jt.flags.{alias_name} '
                    'is a deprecated accelerator-mode alias; use jt.runtime.use_cuda '
                    'or explicit backend queries", 1) < 0) '
                    'throw std::runtime_error("deprecated backend alias");'
                )
                flags_defs.append(f'''
                    // @pyjt(__get__{alias_name})
                    {type} _get_{alias_name}() {{ {warning} return {getter}; }}
                    // @pyjt(__set__{alias_name})
                    void _set_{alias_name}({type} v) {{ {warning} set_{name}(v); }}
                    // @pyjt(__set__{alias_name})
                    void _set_{alias_name}(bool v) {{ {warning} set_{name}(v); }}
                ''')
    
    jit_declares = "\n    ".join(jit_declares)
    jit_src = f"""
    #include "utils/flags.h"
    #include <Python.h>
    #include "runtime/configuration.h"
    #include <stdexcept>

    namespace jittor {{
    
    {jit_declares}

    // @pyjt(Flags)
    struct _Flags {{
        // @pyjt(__init__)
        _Flags() {{}}
        {"".join(flags_defs)}
    }};

    }} // jittor
    """
    LOG.vvvv(jit_src)
    with open(os.path.join(cache_path, "gen", "jit_flags.h"), 'w', encoding='utf8') as f:
        f.write(jit_src)

# The Var* members of an op struct.  Their names and byte offsets go into the
# op registry (VAR_MEMBER_NAME_AND_OFFSET -> offsetof), and VarRelayManager
# relays a member by looking it up there by name.
#
# A member that is not picked up here is not an error anywhere downstream: it
# simply never gets relayed, so the relayed op runs with whatever that member
# happened to point at last.  So only the canonical spelling is accepted, and a
# declaration that looks like a member of pointer-to-Var type but is not spelled
# that way fails the build instead of being silently skipped.
_VAR_MEMBER_DECL = re.compile(r"^[ \t]*Var[ \t]*\*(?P<names>[^;=(){}]*);", re.M)
# the same thing however spelled: "jittor::Var*", "const Var*", extra spaces
_VAR_MEMBER_LOOSE = re.compile(r"^[ \t]*(?:const[ \t]+)?(?:\w+::)*Var[ \t]*\*", re.M)


def parse_var_members(src, header="<src>"):
    """Names of the Var* members declared in an op header, in declaration order."""
    def statement(start):
        end = src.find(";", start)
        return src[start:end + 1 if end != -1 else len(src)].strip()

    strict_starts = {m.start() for m in _VAR_MEMBER_DECL.finditer(src)}
    unparsed = [
        statement(m.start())
        for m in _VAR_MEMBER_LOOSE.finditer(src)
        if m.start() not in strict_starts and "(" not in statement(m.start())
    ]
    assert not unparsed, (
        f"{header}: Var member declaration(s) this build step cannot read: "
        f"{unparsed}. Write them as `Var* name;` or `Var* a, * b;` -- a member "
        f"it cannot read is left out of the op registry, and relaying then "
        f"leaves that member pointing at whatever it pointed at last, silently.")
    names = []
    for m in _VAR_MEMBER_DECL.finditer(src):
        text = m.group("names")
        for c in "*,":
            text = text.replace(c, " ")
        names.extend(text.split())
    return names


def gen_jit_op_maker(op_headers, export=False, extra_flags="", backend=None):
    def compose_backend_source(operator, segments):
        import tempfile

        parts = []
        for source_path in segments:
            with open(source_path, encoding="utf-8") as source_file:
                text = source_file.read()
            parts.append('#line 1 ' + json.dumps(source_path) + '\n' + text + '\n')
        content = "".join(parts)
        directory = os.path.join(cache_path, "backend_sources")
        os.makedirs(directory, exist_ok=True)
        destination = os.path.join(directory, operator + "_cuda.cc")
        try:
            with open(destination, encoding="utf-8") as current:
                if current.read() == content:
                    return destination
        except FileNotFoundError:
            pass
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                             prefix=operator + ".", suffix=".tmp", delete=False) as output:
                temporary = output.name
                output.write(content)
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)
        return destination

    backend_masks = {
        None: None,
        "cpu": "OpBackendCpu",
        "accelerator": "OpBackendAccelerator",
        "both": "OpBackendAny",
    }
    if backend not in backend_masks:
        raise ValueError("backend must be None, 'cpu', 'accelerator', or 'both'")
    backend_mask = backend_masks[backend]
    def add_src(
        cc_func_name,
        cc_args, 
        op_name, 
        op_args, 
        src, 
        pybind_name, 
        py_args, 
        jit_cc_src, 
        doc_string, 
        attrs
    ):
        has_ir = set(["add", "sub", "mul", "matmul", "truediv", "floordiv", "mod", "divmod", "pow", "lshift", "rshift", "and", "xor", "or"])
        pybind_names = [ s.strip() for s in pybind_name.split(",")]
        cc_make_args = [ arg.replace("VarHolder*", "Var*") for arg in cc_args ]
        op_make_args = [ arg.replace("->var", "") for arg in op_args ]
        py_args = [ arg.replace("Var*", "VarHolder*")  for arg in py_args ]
        op_args = []
        cc_args_with_default = []
        for i, arg in enumerate(cc_args):
            pre_arg = arg.split()[-1].split('=')[0]
            op_arg = None
            if arg.startswith("VarHolder*"):
                op_arg = pre_arg+"->var"
            elif arg.startswith("vector<VarHolder*>"):
                op_arg = f"convert({pre_arg})"
            if "&&" in arg:
                if op_arg == None:
                    op_arg = "move("+pre_arg+")"
                op_make_args[i] = "move("+pre_arg+")"
            if op_arg==None: op_arg = pre_arg
            op_args.append(op_arg)
            py_arg = py_args[i]
            if "_a=" not in py_arg:
                cc_args_with_default.append(arg)
                continue
            py_arg = py_arg.split("_a=")[1]
            cc_args_with_default.append(arg + "=" + py_arg)
        cc_args = cc_args_with_default
        # steps of Op creation:
        # 1. new op
        # 2. new output var (create_output in op constructor)
        # 3. take over op's output VarPtr from outputs_holder
        # 4. set op's output
        # 5. set op's input
        # 6. infer shape(op->init())
        if "multiple_outputs" not in attrs:
            jit_cc_src.append(f"""
            VarPtr make_{cc_func_name}({", ".join(cc_make_args)}) {{
                auto _op = new {op_name}({", ".join(op_make_args)});
                if (_op->outputs_holder.size() != 1) {{
                    delete _op;
                    LOGf << "Wrong output size of" << \"{op_name}\";
                }}
                if (_op->flag(OpFlags::_forwarded)) {{
                    VarPtr _out(move(_op->outputs_holder[0]));
                    delete _op;
                    return _out;
                }}
                {{
                    JT_GBP_SCOPE(gbp_edge_table);
                    _op->outputs_holder[0]->set_inputs({{_op}});
                }}
                VarPtr _out(move(_op->outputs_holder[0]));
                {{
                    JT_GBP_SCOPE(gbp_edge_table);
                    {src.replace("->var","")};
                }}
                _op->init();
                return _out;
            }}
            """)
        else:
            jit_cc_src.append(f"""
            vector<VarPtr> make_{cc_func_name}({", ".join(cc_make_args)}) {{
                auto _op = new {op_name}({", ".join(op_make_args)});
                if (_op->flag(OpFlags::_forwarded)) {{
                    vector<VarPtr> _outs = move(_op->outputs_holder);
                    delete _op;
                    return _outs;
                }}
                vector<VarPtr> _outs = move(_op->outputs_holder);
                {{
                    JT_GBP_SCOPE(gbp_edge_table);
                    for (uint i=0; i<_outs.size(); i++)
                        _outs[i]->set_inputs({{_op}});
                    {src.replace("->var","")};
                }}
                _op->init();
                return _outs;
            }}
            """)
        if pybind_name == 'None':
            return
        pyjt_names = []
        for pybind_name in pybind_names:
            if pybind_name.startswith("__"):
                pyjt_names.append("Var."+pybind_name)
            else:
                pyjt_names.append(pybind_name)
                if len(cc_args)>0 and cc_args[0].startswith("VarHolder* "):
                    pyjt_names.append("Var."+pybind_name)
        if "multiple_outputs" in attrs:
            jit_cc_src.append(f"""
            /*{doc_string}*/
            // @pyjt({",".join(pyjt_names)})
            vector_to_tuple<VarHolder*> {cc_func_name}({", ".join(cc_args)}) {{
                {   f'return make_vh_vector(make_{cc_func_name}({", ".join(op_args)}));'
                    if "replace_outputs" not in attrs else
                    f'''auto rt = make_vh_vector(make_{cc_func_name}({", ".join(op_args)}));
                    ASSERT(rt.size() == outputs.size());
                    for (int i=0; i<outputs.size(); i++)
                        outputs[i]->assign(rt[i]);
                    return rt;
                    '''}
            }}
            """)
        else:
            jit_cc_src.append(f"""
            /*{doc_string}*/
            // @pyjt({",".join(pyjt_names)})
            VarHolder* {cc_func_name}({", ".join(cc_args)}) {{
                return new VarHolder(make_{cc_func_name}({", ".join(op_args)}));
            }}
            """)
        need_ir_define = False
        ir_name = None
        for pybind_name in pybind_names:
            if pybind_name.startswith("__") and pybind_name[2:-2] in has_ir:
                need_ir_define = True
                assert ir_name is None
                ir_name = pybind_name[2:-2]
        if need_ir_define:
            assert len(cc_args)>0 and cc_args[0].startswith("VarHolder* ")
            this = cc_args[0].split()[-1]
            jit_cc_src.append(f"""
            // @pyjt(Var.__i{ir_name}__)
            // @attrs(return_self)
            VarHolder* i{cc_func_name}({", ".join(cc_args)}) {{
                *{this} = make_{cc_func_name}({", ".join(op_args)});
                return {this};
            }}
            """)
            assert len(cc_args)>1 and cc_args[1].startswith("VarHolder* "), cc_args
            r_cc_args = [cc_args[1], cc_args[0]] + cc_args[2:]
            r_py_args = [py_args[1], py_args[0]] + py_args[2:]
            jit_cc_src.append(f"""
            VarHolder* r{cc_func_name}({", ".join(r_cc_args)}) {{
                return new VarHolder(make_{cc_func_name}({", ".join(op_args)}));
            }}
            """)

    jit_cc_src = []
    jit_headers = ""
    initer = []
    pybind_reg = '(/\\*(.*?)\\*/\\s*)?(//\\s*@pybind\\(([^\\n]*)\\)\\s*)?'
    pybind_attrs_reg = pybind_reg + '(//\\s*@attrs\\(([^\\n]*)\\)\\s*)?'
    for header in op_headers:
        # xxx_xxx_op
        name = os.path.basename(header)
        name = os.path.splitext(name)[0]
        # xxx_xxx
        assert name.endswith("_op")
        func_name = name[:-3]
        # XxxXxxOp
        name2 = map(lambda s:s[:1].upper() + s[1:], name.split('_'))
        name2 = "".join(name2)
        with open(header, encoding='utf8') as f:
            src = f.read()
        # XxxXxxOp(args)
        res = re.findall(pybind_attrs_reg + '[^~]('+name2+"\\([^\\n]*\\))", src, re.S)
        assert len(res) >= 1, "Wrong op args in " + header
        # registe op
        cc_name = header[:-2] + ".cc"
        constructors = []
        for i in range(len(res)):
            name = 'make_'+func_name+'_'*i
            constructors.append(f"op_constructor_entry(&{name})")
        constructors = ",".join(constructors)
        var_member = parse_var_members(src, header)
        LOG.vv("var_member "+str(var_member))
        var_member_src = [ f"VAR_MEMBER_NAME_AND_OFFSET({name}, {name2})" for name in var_member ]
        var_member_src = ",".join(var_member_src)
        mask_arg = f", {backend_mask}" if backend_mask is not None else ""
        # Optional backend sources are composed into the first definition, not
        # registered as a replacement (which would invalidate persistent JIT keys).
        builtin_directories = (os.path.join(core_root(jittor_path), "ops"),
                               os.path.join(core_root(jittor_path), "ops", "composite"))
        if os.path.realpath(os.path.dirname(header)) in tuple(map(os.path.realpath, builtin_directories)):
            kernel_directory = os.path.join(backend_root(jittor_path, "cuda"), "kernels", "core")
            accelerator_source = os.path.join(kernel_directory, func_name + "_op.cc")
            prefix_source = os.path.join(kernel_directory, func_name + "_prefix.cc")
            if os.path.isfile(prefix_source):
                accelerator_source = compose_backend_source(func_name, [prefix_source, cc_name])
            if os.path.isfile(accelerator_source):
                mask = backend_mask if backend_mask is not None else name2 + "::backend_mask"
                accelerator_flags = ' -I"' + kernel_directory + '" '
                mask_arg = ", " + mask + ", " + json.dumps(accelerator_source) + ", " + json.dumps(accelerator_flags)
        initer.append(f'\n        register_op_definition<{name2}>({{ "{func_name}", R"({cc_name})", extra_flags, {{{constructors}}}, {{{var_member_src}}} }}{mask_arg});')
        for hid, h_def in enumerate(res):
            h_def = list(h_def)
            # // @attrs(...)
            attrs = {}
            if h_def[4] != "":
                attrs = pyjt_compiler.parse_attrs(h_def[5])
            del h_def[4:6]
            # /* doc_string */
            # // @pybind(bind_name)
            # XxxXxxOp(args_def)
            doc_string = h_def[1].strip()
            h_def = h_def[2:]
            args_def = h_def[2][len(name2)+1:-1]
            bind_name = h_def[1]
            if bind_name == "":
                bind_name = func_name
            if args_def=="":
                args = []
            else:
                args = list(map(lambda s: s.split()[-1].split('=')[0], args_def.split(',')))
            # py_args: "arg"_a=default
            py_args = []
            new_args_def = []
            new_args = []
            # source of convert VarHolder* to Var*
            vh2v_src = []
            more_src = []
            for arg, arg_def in zip(args, args_def.split(',')):
                py_arg = f'"{arg}"_a'
                if '=' in arg_def:
                    py_arg += "=" + arg_def.split('=')[-1]
                    arg_def = arg_def.split('=')[0]
                py_args.append(py_arg)
                arg_type = arg_def[:-(len(arg)+1)].strip()
                if arg_type == "Var*":
                    new_args_def.append("VarHolder* " + arg)
                    vh2v_src.append(arg + "->var")
                    new_args.append(arg + "->var")
                elif arg_type.startswith("vector<Var*>"):
                    new_args_def.append(
                        arg_type.replace("Var", "VarHolder")+' '+arg)
                    new_args.append(arg)
                    more_src.append(f"_op->add_inputs({arg});")
                elif arg_type.startswith("VarSlices"):
                    new_args_def.append(arg_def)
                    new_args.append(arg)
                    more_src.append(f"""
                        vector<Var*> svars;
                        for (int i=0; i<_op->vs.n; i++)
                            if (_op->vs.slices[i].is_var())
                                svars.push_back(_op->vs.slices[i].var);
                        _op->add_inputs(svars);""")
                else:
                    new_args_def.append(arg_def)
                    new_args.append(arg)
            vh2v_src = "_op->set_inputs({" + ", ".join(vh2v_src) + "});" + \
                "".join(more_src)
            LOG.vvvv(f"Find op: {name2} args: {new_args}")
            # if header.startswith("src/"):
            #     jit_headers += f"#include \"{header[4:]}\"\n"
            # else:
            jit_headers += f"#include \"{header}\"\n"
            add_src(
                func_name+'_'*hid,
                new_args_def,
                name2,
                new_args,
                vh2v_src,
                bind_name,
                py_args,
                jit_cc_src,
                doc_string,
                attrs
            )
            if func_name in ["binary", "unary", "reduce"]:
                # generate binary op alias
                with open(os.path.join(core_root(jittor_path), f"ops/{func_name}_op.cc"), encoding="utf-8") as f:
                    src = f.read()
                src = src.split(f"unordered_set<string> {func_name}_ops = ""{")[1].split("};")[0]
                match_result = re.findall(pybind_reg + "\"([a-z_A-Z0-9]*)\"", src, re.S)
                # remove /* doc_string */ pattern
                res2 = [ (_[3], _[4]) for _ in match_result ]
                LOG.vvvv(f"All supported {func_name} ops: {res2}")
                # remove op args
                if func_name == "reduce":
                    args_def = new_args_def[:1] + new_args_def[2:]
                    py_args_s = py_args[:1] + py_args[2:]
                else:
                    args_def = new_args_def[:-1]
                    py_args_s = py_args[:-1]
                # find the last type id(float64)
                # add "_" suffix for all function
                if func_name == "unary":
                    last_tid = res2.index(("","float64"))
                # for each functor
                for tid, (bind_name, func_name2) in enumerate(res2):
                    # get certain op doc_string
                    doc_string2 = match_result[tid][1].strip()
                    if len(doc_string2) == 0:
                        doc_string2 = doc_string
                    # add _ for types
                    if func_name == "unary" and tid <= last_tid:
                        func_name3 = func_name2 + "_"
                    elif func_name == "reduce":
                        func_name4 = func_name2
                        func_name2 = "reduce_" + func_name2
                        func_name3 = func_name2
                    else:
                        func_name3 = func_name2
                    if len(bind_name) == 0:
                        bind_name = func_name2
                    if func_name == "reduce":
                        args = new_args[:1] + [f'ns_{func_name4}'] + new_args[2:]
                    else:
                        args = new_args[:-1] + [f'ns_{func_name2}']
                    add_src(
                        func_name3+'_'*hid,
                        args_def,
                        name2,
                        args,
                        vh2v_src,
                        bind_name,
                        py_args_s,
                        jit_cc_src,
                        doc_string2,
                        attrs
                    )

    jit_src = f"""
    #pragma once
    #include "bindings/pyjt/py_obj_holder.h"
    #include "core/var.h"
    #include "core/var_holder.h"
    #include "ops/composite/op_registration.h"
    #include "utils/graph_build_profile.h"
    {jit_headers}
    
    namespace jittor {{
    // fix make_array(py::array) undefine reference
    #pragma GCC visibility push(default)
    #define JIT_NAMESPACE {export+"_maker" if export else "jit_op_maker"}
    // @pyjt(ops)
    // @attrs(submodule{",core_name="+export if export else ""})
    namespace JIT_NAMESPACE {{
    {"".join(jit_cc_src)}

    void initer() {{
        string extra_flags = R"({extra_flags})";
        {"".join(initer)}
    }}
    int caller = (initer(), 0);
    
    }} // JIT_NAMESPACE
    }} // jittor
    {f'''
    namespace jittor {{
    extern void pyjt_def_{export}(PyObject*);
    }}

    static void init_module(PyModuleDef* mdef, PyObject* m) {{
        mdef->m_doc = "User defined custom ops";
        jittor::pyjt_def_{export}(m);
    }}
    PYJT_MODULE_INIT({export});

    ''' if export else ""}
    """
    return jit_src

@lock.lock_scope()
def compile_custom_op(header, source, op_name, warp=True):
    """Compile a single custom op
    header: code of op header, not path
    source: code of op source, not path
    op_name: op_name of this op, it will used for 
        generation of header and source files, if the 
        type name of op is XxxXxxOp, op_name should be
        xxx_xxx
    warp: if true, warp a snippet for header and source
    """
    if warp:
        header = f"""
        #pragma once
        #include "core/op.h"
        #include "core/var.h"
        namespace jittor {{
        {header}
        }}
        """
        source = f"""
        #include "{op_name}_op.h"
        namespace jittor {{
        {source}
        }}
        """
    cops_dir = os.path.join(cache_path, "custom_ops")
    make_cache_dir(cops_dir)
    hname = os.path.join(cops_dir, op_name+"_op.h")
    ccname = os.path.join(cops_dir, op_name+"_op.cc")
    with open(hname, 'w', encoding='utf8') as f:
        f.write(header)
    with open(ccname, 'w', encoding='utf8') as f:
        f.write(source)
    m = compile_custom_ops([hname, ccname])
    return getattr(m, op_name)

@lock.lock_scope()
def compile_custom_ops(
    filenames, 
    extra_flags="", 
    return_module=False,
    dlopen_flags=None,
    gen_name_ = "",
    backend=None):
    """Compile custom ops
    filenames: path of op source files, filenames must be
        pairs of xxx_xxx_op.cc and xxx_xxx_op.h, and the 
        type name of op must be XxxXxxOp.
    extra_flags: extra compile flags
    return_module: return module rather than ops(default: False)
    backend: None uses the op class declaration; cpu, accelerator, or both
        explicitly selects the registered backend family for this library.
    return: compiled ops
    """
    if dlopen_flags is None:
        dlopen_flags = os.RTLD_GLOBAL | os.RTLD_NOW
        if platform.system() == 'Linux':
            dlopen_flags |= os.RTLD_DEEPBIND

    srcs = {}
    headers = {}
    builds = []
    includes = []
    pyjt_includes = []
    for name in filenames:
        name = os.path.realpath(name)
        if name.endswith(".cc") or name.endswith(".cpp") or name.endswith(".cu"):
            builds.append(name)
        if name.endswith(".h"):
            dirname = os.path.dirname(name)
            if dirname.endswith("inc"):
                includes.append(dirname)
            with open(name, "r", encoding='utf8') as f:
                if "@pyjt" in f.read():
                    pyjt_includes.append(name)
        bname = os.path.basename(name)
        bname = os.path.splitext(bname)[0]
        if bname.endswith("_op"):
            bname = bname[:-3]
            if name.endswith(".cc"):
                srcs[bname] = name
            elif name.endswith(".h"):
                includes.append(os.path.dirname(name))
                headers[bname] = name
    assert len(srcs) == len(headers), "Source and header names not match"
    for name in srcs:
        assert name in headers, f"Header of op {name} not found"
    gen_name = "gen_ops_" + "_".join(headers.keys())
    if gen_name_ != "":
        gen_name = gen_name_
    if len(gen_name) > 50:
        gen_name = gen_name[:50] + "___hash" + hashlib.md5(gen_name.encode()).hexdigest()[:6]

    include_dirs = sorted(list(set(includes)))
    includes = "".join(map(lambda x: f" -I\"{x}\" ", include_dirs))
    LOG.vvvv(f"Include flags:{includes}")

    op_extra_flags = includes + extra_flags

    lib_path = os.path.join(cache_path, "custom_ops")
    make_cache_dir(lib_path)
    gen_src_fname = os.path.join(lib_path, gen_name+".cc")
    gen_head_fname = os.path.join(lib_path, gen_name+".h")
    gen_lib = os.path.join(lib_path, gen_name+extension_suffix)
    libname = gen_name + lib_suffix
    op_extra_flags += f" -L\"{lib_path}\" -l\"{libname}\" "

    # Everything below -- generating the op maker, running pyjt over the
    # headers, and handing the translation units to the compile pool -- is
    # skipped when the stamp says this library was already built from exactly
    # these inputs. See :func:`product_build_is_current` for why the cheap
    # answer is worth having.
    stamp_sources = {
        "files": _stat_signature([os.path.realpath(name)
                                  for name in filenames]),
        "includes": _include_tree_signature(_custom_op_include_dirs(
            filenames, include_dirs, extra_flags)),
        "core_sources": core_source_signature(),
    }
    stamp_ingredients = custom_op_build_ingredients(
        gen_name, extra_flags, includes, backend)
    if product_build_is_current(gen_lib, stamp_sources, stamp_ingredients):
        LOG.v(f"custom op lib {gen_name} is current, skipping build")
        return _import_custom_op_lib(gen_name, dlopen_flags, return_module)
    if not build_is_allowed():
        _refuse_build("the custom op library " + gen_name)

    gen_src = gen_jit_op_maker(headers.values(), export=gen_name,
                             extra_flags=op_extra_flags, backend=backend)
    pyjt_compiler.compile_single(gen_head_fname, gen_src_fname, src=gen_src)
    # gen src initialize first
    builds.insert(0, gen_src_fname)

    def insert_anchor(gen_src, anchor_str, insert_str):
        # insert insert_str after anchor_str into gen_src
        return gen_src.replace(anchor_str, anchor_str+insert_str, 1)

    for name in pyjt_includes:
        LOG.v("handle pyjt_include ", name)
        bname = os.path.basename(name).split(".")[0]
        gen_src_fname = os.path.join(cache_path, "custom_ops", gen_name+"_"+bname+".cc")
        pyjt_compiler.compile_single(name, gen_src_fname)
        builds.insert(1, gen_src_fname)
        gen_src = insert_anchor(gen_src,
            "namespace jittor {",
            f"extern void pyjt_def_{bname}(PyObject* m);")
        gen_src = insert_anchor(gen_src,
            "init_module(PyModuleDef* mdef, PyObject* m) {",
            f"jittor::pyjt_def_{bname}(m);")

    with open(gen_head_fname, "w", encoding='utf8') as f:
        f.write(gen_src)

    LOG.vvv(f"Build custum ops lib:{gen_lib}")
    LOG.vvvv(f"Build sources:{builds}")
    compile(cc_path, extra_flags+cc_flags+opt_flags+includes, builds, gen_lib)

    # After the compile, so the stamp records the product that exists now. The
    # source signature is the pre-build one on purpose: a source edited during
    # the build must leave the stamp stale rather than claim to cover the edit.
    _write_product_build_stamp(gen_lib, stamp_sources, stamp_ingredients)

    return _import_custom_op_lib(gen_name, dlopen_flags, return_module)


def _import_custom_op_lib(gen_name, dlopen_flags, return_module):
    """dlopen a built custom-op library and hand back its ops."""
    LOG.vvv(f"Import custum ops lib:{gen_name}")
    lib_path = os.path.join(cache_path, "custom_ops")
    if lib_path not in os.sys.path:
        os.sys.path.append(lib_path)
    # unlock scope when initialize
    # NOTE: use __import__ (returns the module object) instead of
    # `exec("import X"); locals()["X"]`. Since Python 3.13 (PEP 667) exec() no
    # longer writes back into an optimized function's locals(), so the old
    # pattern raised KeyError and broke `import jittor` from source on 3.13.
    with lock.unlock_scope():
        with jit_utils.import_scope(dlopen_flags):
            mod = __import__(gen_name)
    if return_module:
        return mod
    return mod.ops


def get_full_path_of_executable(name):
    full_path = os.path.abspath(name)
    while os.path.islink(full_path):
        full_path = os.path.realpath(full_path)
    if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
        return full_path
    return get_full_path_of_executable(find_exe(name))

cuda_wheel_stack = None
cuda_include_dirs = []
cuda_lib_dirs = []
cuda_runtime_lib = ""
_loaded_cuda_libraries = {}


def _cuda_library_sort_key(path):
    name = os.path.basename(path)
    return (name.count("."), len(name), name)


def find_cuda_library(name, component=None):
    """Find a CUDA library, including wheel-only versioned SONAME files."""

    if cuda_wheel_stack:
        path = cuda_wheel_stack.find_library(name, component)
        if path:
            return path
    dirs = cuda_lib_dirs or [globals().get("cuda_lib", ""), globals().get("cuda_bin", "")]
    if os.name == "nt":
        patterns = (name + "64*.dll", name + "*.dll")
    elif platform.system() == "Darwin":
        patterns = ("lib" + name + ".dylib", "lib" + name + ".*.dylib")
    else:
        patterns = ("lib" + name + ".so", "lib" + name + ".so.*")
    matches = []
    for directory in dirs:
        if not directory:
            continue
        for pattern in patterns:
            matches.extend(glob.glob(os.path.join(directory, pattern)))
    matches = [os.path.abspath(path) for path in matches if os.path.isfile(path)]
    if not matches:
        return None
    matches.sort(key=_cuda_library_sort_key)
    return matches[0]


def cuda_library_link_flags(name, path=None):
    path = path or find_cuda_library(name)
    if not path:
        raise RuntimeError("CUDA library lib%s was not found" % name)
    directory = os.path.dirname(path)
    if os.name == "nt":
        return '-L"%s" -l%s' % (directory, name)
    return '-L"%s" -l:%s' % (directory, os.path.basename(path))


def preload_cuda_library(name, required=False):
    """Load a CUDA library and its wheel dependencies into the global scope."""

    if cuda_wheel_stack:
        paths = cuda_wheel_stack.preload_paths(name)
    else:
        path = find_cuda_library(name)
        paths = [path] if path else []
    if required and not paths:
        raise RuntimeError("CUDA library lib%s was not found" % name)
    for path in paths:
        if path not in _loaded_cuda_libraries:
            _loaded_cuda_libraries[path] = ctypes.CDLL(path, dlopen_flags)
    return _loaded_cuda_libraries.get(paths[-1]) if paths else None

def check_cuda():
    if not nvcc_path:
        return
    global cc_flags, has_cuda, is_cuda, core_link_flags, cuda_dir, cuda_lib, cuda_include, cuda_home, cuda_bin
    global cuda_include_dirs, cuda_lib_dirs, cuda_runtime_lib
    global cuda_sdk_flags, cuda_link_flags
    cuda_dir = os.path.dirname(get_full_path_of_executable(nvcc_path))
    cuda_bin = cuda_dir
    cuda_home = os.path.abspath(os.path.join(cuda_dir, ".."))
    # try default nvidia-cuda-toolkit in Ubuntu 20.04
    # assert cuda_dir.endswith("bin") and "cuda" in cuda_dir.lower(), f"Wrong cuda_dir: {cuda_dir}"
    cuda_include = os.path.abspath(os.path.join(cuda_dir, "..", "include"))
    cuda_lib = os.path.abspath(os.path.join(cuda_dir, "..", "lib64"))
    if nvcc_path == "/usr/bin/nvcc":
        # this nvcc is install by package manager
        cuda_lib = "/usr/lib/x86_64-linux-gnu"
    cuda_include_dirs = [cuda_include]
    cuda_lib_dirs = [cuda_lib, cuda_bin]
    if cuda_wheel_stack:
        cuda_include_dirs = cuda_wheel_stack.include_dirs() + cuda_include_dirs
        cuda_lib_dirs = cuda_wheel_stack.lib_dirs() + cuda_lib_dirs
    cuda_include_dirs = list(dict.fromkeys(
        path for path in cuda_include_dirs if os.path.isdir(path)
    ))
    cuda_lib_dirs = list(dict.fromkeys(
        path for path in cuda_lib_dirs if os.path.isdir(path)
    ))
    cuda_include2 = os.path.join(backend_root(jittor_path, "cuda"), "include")
    cc_flags += " -DHAS_ACCELERATOR -DHAS_CUDA -DIS_CUDA "
    cuda_sdk_flags = "".join(f' -I"{path}"' for path in cuda_include_dirs)
    cuda_sdk_flags += f" -I\"{cuda_include2}\" "
    if os.name == 'nt':
        cuda_lib = os.path.abspath(os.path.join(cuda_dir, "..", "lib", "x64"))
        # cc_flags += f" \"{cuda_lib}\\cudart.lib\" "
        cuda_link_flags = f" -lcudart -L\"{cuda_lib}\" -L\"{cuda_bin}\" "
    else:
        cuda_runtime_lib = find_cuda_library("cudart")
        if not cuda_runtime_lib:
            raise RuntimeError(
                "CUDA compiler was found, but libcudart was not found in %s"
                % cuda_lib_dirs
            )
        cuda_link_flags = " " + cuda_library_link_flags("cudart", cuda_runtime_lib) + " "
    is_cuda = has_cuda = 1

def _write_jit_utils_cache_key(files, output):
    """Write ``<output>.key`` from a child process.

    ``cache_compile`` records the key as a side effect of building, so the only
    way to get one is to run a build. Running it here would replace a library
    this process already has mapped; running it in a child leaves this process
    with a single, settled file to import.
    """
    cmd = compile(cc_path, cc_flags+f" {opt_flags} ", files, output, True, return_cmd=True)
    # Import from the directory that actually holds the freshly built library,
    # which is not necessarily this module's ``cache_path`` once a CUDA suffix
    # has been appended to it.
    script = (
        "import sys\n"
        "sys.path.insert(0, {module_dir!r})\n"
        "import jit_utils_core\n"
        "jit_utils_core.cache_compile({cmd!r}, {cache!r}, {jittor!r})\n"
    ).format(module_dir=os.path.dirname(os.path.abspath(output)),
             cache=cache_path, cmd=cmd, jittor=jittor_path)
    try:
        result = sp.run([sys.executable, "-c", script],
                        stdout=sp.PIPE, stderr=sp.STDOUT)
    except OSError as error:
        LOG.v("cache key child could not start: %s" % error)
        return
    if result.returncode != 0:
        LOG.v("cache key child failed: " + result.stdout.decode("utf8", "replace"))


#: Exit code used when jit_utils was rebuilt and the process must be restarted.
#: Distinct from 1 so a caller can tell "rerun me" apart from a real failure.
JIT_UTILS_UPDATED_EXIT_CODE = 3


def check_cache_compile():
    files = [
        "src/utils/cache_compile.cc",
        "src/utils/log.cc",
        "src/utils/tracer.cc",
        "src/utils/jit_utils.cc",
        "src/utils/str_utils.cc",
    ]
    if os.name == 'nt':
        files = [ x.replace('/', '\\') for x in files ]
    global jit_utils_core_files
    jit_utils_core_files = files
    output = jit_utils.cache_path+'/jit_utils_core'+extension_suffix
    recompile = compile(cc_path, cc_flags+f" {opt_flags} ", files, output, True)
    if recompile and jit_utils.cc:
        # A non-zero code, because nothing the caller asked for was done. With
        # exit 0 a CI job or a training script that touched src/utils/*.cc
        # "succeeded" without running a single line of the user's program, and
        # the only trace was one line of log among hundreds.
        LOG.e("jit_utils was rebuilt and cannot be reloaded in this process. "
              "Nothing else has run: rerun the same command.")
        sys.exit(JIT_UTILS_UPDATED_EXIT_CODE)
    if not jit_utils.cc:
        # The cache key is written by cache_compile, which needs this very
        # library to be importable first. Doing that after importing it would
        # relink the file while this process has it mapped: the linker unlinks
        # the output before writing, so the inode this process holds no longer
        # matches the one on disk, and the loader later resolves jittor_core's
        # dependency to a *second* copy of the same library. Two copies mean two
        # sets of the flags and log-capture state defined in these files, which
        # is why log_capture_scope silently returned nothing after a cold build.
        # Generate the key in a short-lived child, then import the settled file.
        _write_jit_utils_cache_key(files, output)
        with jit_utils.import_scope(import_flags):
            jit_utils.try_import_jit_utils_core()
        if not jit_utils.cc:
            raise RuntimeError(
                f"jit_utils_core was built at {output} but could not be "
                f"imported. The build directory is on sys.path; a stale or "
                f"partially written file there is the usual cause -- remove "
                f"it and run again.")
        if not os.path.isfile(output + ".key"):
            # The child could not run; fall back to the original behaviour so a
            # missing key never blocks startup. The duplicate mapping above is
            # the cost, and the next run starts from a cached key.
            compile(cc_path, cc_flags+f" {opt_flags} ", files, output, True)

def query_cuda_archs():
    """Compute capabilities of the GPUs on this machine, or None if unknown.

    Runs a short child because the CUDA driver cannot be loaded and unloaded
    inside this process. Only bare integers are accepted from its output: the
    child inherits stdout and stderr from a full Jittor import, and *any* line
    it happens to log used to be spliced into the cache directory name -- one
    stray "Create lock file: ..." produced a cache directory named after that
    message, and a second process that did not log it built into a different
    one.

    The answer is cached against the driver version, so the usual import does
    not start a second interpreter at all. That child is also the one that has
    deadlocked: it was started while this process held the build lock, and
    anything it did that needed the lock waited for a holder that was waiting
    for it.
    """
    driver = ""
    try:
        with open("/proc/driver/nvidia/version") as f:
            driver = f.readline().strip()
    except OSError:
        pass
    return jit_utils.probe.cached(
        "cuda_archs", [], lambda: _read_cuda_archs(), extra=driver)


def _read_cuda_archs():
    try:
        child = sp.run(
            [sys.executable, "-m", "jittor_utils.query_cuda_cc"],
            stdout=sp.PIPE, stderr=sp.PIPE,
            env=env_config.child_env(log_v=0, log_silent=1))
    except OSError as error:
        LOG.v(f"could not query cuda archs: {error}")
        return None
    if child.returncode != 0:
        LOG.v("could not query cuda archs: "
              + child.stderr.decode("utf8", "replace"))
        return None
    archs = set()
    for token in child.stdout.decode("utf8", "replace").split():
        if token.isdigit():
            archs.add(token)
        else:
            LOG.v(f"ignoring non-numeric output from query_cuda_cc: {token!r}")
    return sorted(archs)

def parse_nvcc_arch_list(text):
    """The integers in an ``nvcc --list-gpu-arch`` listing."""
    archs = set()
    for token in text.split():
        token = token.strip()
        if token.startswith("compute_") and token[len("compute_"):].isdigit():
            archs.add(int(token[len("compute_"):]))
    return sorted(archs)


def _read_nvcc_archs(nvcc):
    try:
        child = sp.run([nvcc, "--list-gpu-arch"], stdout=sp.PIPE, stderr=sp.PIPE)
    except OSError as error:
        LOG.v(f"could not list nvcc gpu archs: {error}")
        return []
    if child.returncode != 0:
        # --list-gpu-arch arrived in CUDA 11.1. Older toolkits fall back to
        # the version table in the caller.
        LOG.v("could not list nvcc gpu archs: "
              + child.stderr.decode("utf8", "replace"))
        return []
    return parse_nvcc_arch_list(child.stdout.decode("utf8", "replace"))


def query_nvcc_archs(nvcc):
    """Virtual architectures this nvcc can generate code for, newest last.

    This used to be a literal: ``max_arch = 90``, with a version ladder below
    it for older toolkits. A literal ceiling ages into a wrong answer -- every
    CUDA release after the one it was written for reads as 90 -- and the
    consequence is not a missing optimisation but a build that cannot run.
    Empty if this nvcc is too old to be asked (before CUDA 11.1).
    """
    return jit_utils.probe.cached("nvcc_archs:" + nvcc, [nvcc],
                                  lambda: _read_nvcc_archs(nvcc))


def select_cuda_archs(requested, max_arch, min_arch=30):
    """The architectures to actually compile for, sorted and deduplicated.

    Anything below ``min_arch`` is dropped: nvcc cannot target it at all.
    Anything above ``max_arch`` is replaced by ``max_arch``, which is only
    survivable because ``cuda_arch_flags`` keeps that architecture's PTX for
    the driver to JIT -- see the warning below.
    """
    archs = []
    for arch in requested:
        arch = int(arch)
        if arch < min_arch:
            LOG.w(f"CUDA arch({arch})<{min_arch} is not supported")
            continue
        if arch > max_arch:
            LOG.w(f"CUDA arch({arch}) is newer than anything this nvcc can "
                  f"generate code for (highest is {max_arch}). Building for "
                  f"compute_{max_arch} and embedding its PTX, which the driver "
                  f"JIT-compiles for sm_{arch} the first time each kernel runs. "
                  f"A CUDA toolkit that knows sm_{arch} removes that cost.")
            arch = max_arch
        archs.append(arch)
    return sorted(set(archs))


def cuda_arch_flags(archs):
    """``-gencode`` for each architecture, keeping PTX for the newest one.

    The old form was ``-arch=compute_<min>`` followed by one ``-code=sm_X`` per
    architecture. That has two faults. Every cubin is compiled from the
    *lowest* virtual architecture, so an sm_90 cubin is built from compute_70
    sources and none of the newer instructions are available to it. And the
    product contains cubins only: no PTX. A GPU newer than every ``sm_`` in
    the list therefore has neither a loadable cubin nor anything to JIT, and
    fails at launch with "no kernel image is available for execution on the
    device". PTX for the highest architecture is the only thing that makes a
    build forward compatible at all.
    """
    archs = sorted(set(int(arch) for arch in archs))
    if not archs:
        return ""
    # Two things about the spelling, both learned the hard way.
    #
    # `--generate-code=...` rather than `-gencode ...`: cache_compile parses
    # this command line to find the source files, and takes any token that is
    # not itself an option to be one. Split across two tokens, the value is
    # read as a file name and every CUDA operator fails to compile with
    # "Source read failed: arch=compute_89,...".
    #
    # Two clauses rather than the `code=[sm_X,compute_X]` list: the command
    # goes to a shell, where square brackets are a glob pattern. Unmatched
    # patterns survive today, but nothing here needs to depend on that.
    parts = [f" --generate-code=arch=compute_{arch},code=sm_{arch} "
             for arch in archs]
    top = archs[-1]
    parts.append(f" --generate-code=arch=compute_{top},code=compute_{top} ")
    return "".join(parts)


def check_pybt(gdb_path, python_path):
    if gdb_path=='' or python_path=='':
        return False
    return True
    # TODO: prev we use below code to check has py-bt or nor
    # but it is too slow, so we comment it,
    # find a better way to check py-bt exist

    # ret = sp.getoutput(f"{gdb_path} --batch {python_path} -ex 'help py-bt'")
    # if 'python frame' in ret:
    #     LOG.v("py-bt found in gdb.")
    #     return True
    # return False

def check_debug_flags():
    global is_debug
    is_debug = 0
    if build_flag("debug"):
        is_debug = 1
        global cc_flags
        cc_flags += " -g -DNODE_MEMCHECK "

def check_save_mem_flags():
    """Warn when the unfinished swapping build is explicitly enabled.

    ``jt_config_macro_flags`` adds the normalized define with the other
    source-controlled JT_* macros later in command-line construction.
    """
    flags = jit_utils.save_mem_build_flags()
    if not flags:
        return
    LOG.w("JT_SAVE_MEM=1: memory swapping is built in. It is unfinished -- "
          "share_with, migrate, the dual allocator and foreign allocators are "
          "all still on its TODO list (src/mem/swap.h) -- and it needs its own "
          "build, so this configuration compiles into a cache directory of "
          "its own.")

cc_flags = " "
# os.RTLD_NOW | os.RTLD_GLOBAL cause segfault when import torch first
import_flags = os.RTLD_NOW | os.RTLD_GLOBAL
if platform.system() == 'Linux':
    import_flags |= os.RTLD_DEEPBIND
# if cc_type=="icc":
#     # weird link problem, icc omp library may conflict and cause segfault
#     import_flags = os.RTLD_NOW | os.RTLD_GLOBAL
dlopen_flags = os.RTLD_NOW | os.RTLD_GLOBAL
if platform.system() == 'Linux':
    import_flags |= os.RTLD_DEEPBIND

with jit_utils.import_scope(import_flags):
    jit_utils.try_import_jit_utils_core()

jittor_path = find_jittor_path()
if os.name == 'nt':
    # prevent windows recompile
    jittor_path = jittor_path.lower()
check_debug_flags()
check_save_mem_flags()

sys.path.append(cache_path)
LOG.i(f"Jittor({__version__}) src: {jittor_path}")
LOG.i(f"{jit_utils.cc_type} at {jit_utils.cc_path}{jit_utils.get_version(jit_utils.cc_path)}")
LOG.i(f"cache_path: {cache_path}")

with jit_utils.import_scope(import_flags):
    jit_utils.try_import_jit_utils_core()

python_path = sys.executable
# sometime python do not return the correct sys executable
# this will happend when multiple python version installed
ex_python_path = python_path + '.' + str(sys.version_info.minor)
if os.path.isfile(ex_python_path):
    python_path = ex_python_path

def _discover_cuda_compiler(requested_backend):
    if requested_backend not in (None, "cuda"):
        return ""
    nvcc = None
    if install_cuda.has_installation() or os.name == 'nt':
        nvcc = install_cuda.install_cuda()
        if nvcc:
            nvcc = try_find_exe(nvcc)
    if not nvcc:
        nvcc = env_or_try_find('nvcc_path', 'nvcc') or \
            try_find_exe('/usr/local/cuda/bin/nvcc') or \
            try_find_exe('/usr/bin/nvcc') or \
            try_find_exe('/opt/cuda/bin/nvcc')
    if not nvcc:
        nvcc = install_cuda.install_cuda()
        if nvcc:
            nvcc = try_find_exe(nvcc)
    return build_env("nvcc_path", nvcc or "")


_requested_backend = _backend_discovery.requested_backend()
nvcc_path = _discover_cuda_compiler(_requested_backend)
gdb_path = env_or_try_find('gdb_path', 'gdb')
addr2line_path = try_find_exe('addr2line')
has_pybt = check_pybt(gdb_path, python_path)

if nvcc_path:
    # gen cuda key for cache_path
    cu = "cu"
    v = jit_utils.get_version(nvcc_path)[1:-1]
    nvcc_version = list(map(int,v.split('.')))
    cu += v
    cuda_wheel_stack = install_cuda.get_cuda_wheel_stack(v)
    if cuda_wheel_stack:
        cu += "_" + cuda_wheel_stack.fingerprint
    archs = query_cuda_archs()
    if archs is not None:
        if len(archs) == 0:
            LOG.e("No GPU Device Found!")
        cu += "_sm_" + "_".join(archs)
    LOG.i("cuda key:", cu)
    cache_path = os.path.join(cache_path, cu)
    # Ahead of the plain cache directory, which is already on the path. Any run
    # that imports Jittor without nvcc builds a CPU-only jittor_core into that
    # parent directory, and appending here would leave it earlier on the path --
    # so it wins the import and every CUDA op afterwards fails with
    # "Op ... doesn't have cuda version", for this run and every later one
    # sharing the cache.
    parent_cache = os.path.dirname(cache_path)
    if cache_path in sys.path:
        sys.path.remove(cache_path)
    sys.path.insert(
        sys.path.index(parent_cache) if parent_cache in sys.path else len(sys.path),
        cache_path,
    )


def check_clang_latest_supported_cpu():
    output = run_cmd('clang --print-supported-cpus')
    def find_latest_chip_version(pattern_prefix):
        apple_cpus = [l.strip() for l in output.split('\n') if pattern_prefix in l]
        apple_cpu_id = max([int(cpu[7:]) for cpu in apple_cpus])
        return pattern_prefix + str(apple_cpu_id)
    if 'apple-m' in output:
        return find_latest_chip_version('apple-m')
    else:
        return find_latest_chip_version('apple-a')

# cc_flags += " -Wall -Werror -Wno-unknown-pragmas -std=c++14 -fPIC "
cc_flags += " -Wall -Wno-unknown-pragmas -std=c++14 -fPIC "
# 1. Arch/CPU specific optimization
if platform.machine() in ["x86_64", "AMD64"]:
    cc_flags += " -march=native " 
elif platform.machine() == 'arm64' and platform.system() == "Darwin":
    cc_flags += f" -mcpu={check_clang_latest_supported_cpu()} "
cc_flags += " -fdiagnostics-color=always "

#: Build-configuration macros the C++ sources test with `#ifdef`, and which the
#: environment can turn on. There are only a few, they change rarely, and
#: `tests/compiler/test_jt_config_macros.py` greps the sources and fails if
#: this tuple stops matching -- so the list cannot drift without anyone
#: noticing.
#:
#: This used to be discovered by `cache_compile.cc`, which scanned every source
#: it was already reading for dependency tracking and rewrote the compiler
#: command line in place when it found one. That coupled two unrelated jobs:
#: deciding the command line, which must happen *before* a compile, and
#: collecting dependencies, which the compiler can only report *after* one.
#: While they shared a scanner, the compiler's own `-MD -MF` could not be used
#: for dependencies at all -- the first, cold compile would have gone out
#: without its `-D`. Deciding the flags here, from a declared list, is what
#: separates them (task 9.21).
JT_CONFIG_MACROS = (
    "JT_CHECK_NAN",
    "JT_GRAPH_BUILD_PROFILE",
    "JT_HAS_HALF_SIMD",
    "JT_HCCL_NO_MPI",
    "JT_NCCL_NO_MPI",
    "JT_SAVE_MEM",
    "JT_SYNC",
    "JT_bfs_executor",
)

def jt_config_macro_flags(environ=None):
    """`-D` flags for the JT_* macros this environment turns on.

    A variable that is unset, empty or "0" is off, which is the same rule the
    scanner used. Everything downstream derives from cc_flags, so setting them
    here reaches the nvcc command line too.
    """
    environ = os.environ if environ is None else environ
    flags = []
    for name in JT_CONFIG_MACROS:
        if name == "JT_SAVE_MEM":
            value = jit_utils.save_mem_build_flags(environ)
            if value:
                flags.append(value)
            continue
        value = environ.get(name)
        if value in (None, "", "0"):
            continue
        flags.append(f" -D{name}={value} ")
    return "".join(flags)

cc_flags += jt_config_macro_flags()
# 2. Non standard include path
if platform.system() == 'Darwin':
    # TODO: if not using apple clang, there is no need to add -lomp
    cc_flags += " -undefined dynamic_lookup -lomp "
    if os.environ.get("CONDA_PREFIX", None):
        cc_flags += f" -L{os.path.join(os.environ['CONDA_PREFIX'], 'lib')} "
    # if platform.machine() == "arm64":
    #     cc_flags += " -I/opt/homebrew/include -L/opt/homebrew/lib  "
    # Homebrew does not symlink the openmp library (libomp >= 15.0.6) into /opt/homebrew/lib
    homebrew_openmp_paths = [
        "/opt/homebrew/opt/libomp",
        "/usr/local/opt/libomp"
    ]
    for openmp_path in homebrew_openmp_paths:
        if os.path.exists(openmp_path):
            cc_flags += f" -I{openmp_path}/include -L{openmp_path}/lib"

# 3. User specified flags
#
# These are *appended*. That is now the only meaning the setting has: the native
# `cc_flags` flag used to be replaced wholesale by the same variable in
# `log.h`'s static initializer, and then overwritten again by the
# `flags.cc_flags = ...` assignment at the end of this file, so one name had two
# documented behaviours of which only this one ever survived. The core no longer
# reads any compiler-owned build flag from the environment (see
# `compiler_owned_flag_names` in `src/utils/log.cc`).
_user_cc_flags = build_env("cc_flags")
if _user_cc_flags is not None:
    cc_flags += _user_cc_flags + ' '

cc_flags += " -lstdc++ -ldl -shared "

opt_flags = ""

py_include = jit_utils.get_py3_include_path()
LOG.v(f"py_include: {py_include}")
extension_suffix = jit_utils.get_py3_extension_suffix()
lib_suffix = extension_suffix.rsplit(".", 1)[0]
LOG.v(f"extension_suffix: {extension_suffix}")
so = ".so" if os.name != 'nt' else ".dll"


kernel_opt_flags = build_env("kernel_flags", "") + opt_flags
if platform.system() == 'Darwin':
    # TODO: if not using apple clang, cannot add -Xpreprocessor
    kernel_opt_flags += " -Xpreprocessor -fopenmp "
elif cc_type != 'cl':
    kernel_opt_flags += " -fopenmp "
def fix_cl_flags(cmd):
    output = shsplit(cmd)
    output2 = []
    libpaths = []
    for s in output:
        if s.startswith("-l:"):
            output2.append(s)
        elif s.startswith("-l") and ("cpython" in s or "lib" in s):
            if platform.system() == 'Darwin':
                fname = s[2:] + ".so"
                for path in reversed(libpaths):
                    full = os.path.join(path, fname).replace("\"", "")
                    if os.path.isfile(full):
                        output2.append(full)
                        break
                else:
                    output2.append(s)
            else:
                output2.append(f"-l:{s[2:]}.so")
        elif s.startswith("-L"):
            libpaths.append(s[2:])
            output2.append(f"{s} -Wl,-rpath,{s[2:]}")
        else:
            output2.append(s)
    return " ".join(output2)

if os.name == 'nt':
    if cc_type == 'g++':
        pass
    elif cc_type == 'cl':
        py3_link_path = jit_utils.get_py3_link_path()
        cc_flags = remove_flags(cc_flags, ["-f", "-m"])
        cc_flags = cc_flags.replace("-std=c++14", "-std=c++17")
        cc_flags = cc_flags.replace("-lstdc++", "")
        cc_flags = cc_flags.replace("-ldl", "")
        cc_flags += f" -L\"{py3_link_path}\" -lpython3{sys.version_info.minor} "
        cc_flags += " -EHa -MD -utf-8 "
        import jittor_utils
        if jittor_utils.msvc_path:
            mp = jittor_utils.msvc_path
            cc_flags += f' -nologo -I"{mp}\\VC\\include" -I"{mp}\\win10_kits\\include\\ucrt" -I"{mp}\\win10_kits\\include\\shared" -I"{mp}\\win10_kits\\include\\um" -DNOMINMAX '
            cc_flags += f' -L"{mp}\\VC\\lib" -L"{mp}\\win10_kits\\lib\\um\\x64" -L"{mp}\\win10_kits\\lib\\ucrt\\x64" '
        win_libpaths = {}
        def fix_cl_flags(cmd):
            cmd = cmd.replace(".o ", ".obj ")
            cmd = cmd.replace(".o\"", ".obj\"")
            if cmd.endswith(".o"): cmd += "bj"
            if " -o " in cmd:
                if " -shared " in cmd:
                    cmd = cmd.replace(" -o ", " -Fe: ")
                    output = shsplit(cmd.split("-Fe:")[1].strip())[0]
                    base_output = os.path.basename(output).split('.')[0]
                    cmd += f" -DEF:{output}.def -IGNORE:4102 -IGNORE:4197 -IGNORE:4217 "

                elif " -c -o " in cmd:
                    cmd = cmd.replace(" -c -o ", " -c -Fo: ")
            flags = shsplit(cmd)
            output = []
            output2 = []
            for f in flags:
                if f.startswith("-link"):
                    pass
                elif f.startswith("-l"):
                    output2.append(f[2:]+".lib")
                elif f.startswith("-LIB"):
                    output2.append(f)
                elif f.startswith("-LD"):
                    output.append(f)
                elif f.startswith("-L"):
                    path = f[2:].replace("\"", "")
                    if path not in win_libpaths:
                        win_libpaths[path] = 1
                        os.add_dll_directory(path)
                        os.environ["PATH"] = f";{path};" + os.environ["PATH"]
                    output2.append("-LIBPATH:"+f[2:])
                elif ".lib" in f:
                    output2.append(f)
                elif f.startswith("-DEF:"):
                    output2.append(f)
                elif f.startswith("-W") or f.startswith("-f"):
                    pass
                elif f.startswith("-std="):
                    output.append(f.replace("=", ":"))
                else:
                    output.append(f)
            cmd = " ".join(output)
            if len(output2):
                cmd += " -link " + " ".join(output2)
            cmd = cmd.replace("-include", "-FI")
            cmd = cmd.replace("-shared", "-LD")
            return cmd

if ' -O' not in cc_flags:
    if build_flag("debug"):
        opt_flags += " -O0 "
    else:
        opt_flags += " -O2 "
    kernel_opt_flags += " -Ofast "
lto_flags = ""
if build_flag("enable_lto"):
    if cc_type == "icc":
        lto_flags = " -flto -ipo -ipo-c "
    elif cc_type == "g++":
        lto_flags = " -flto -fuse-linker-plugin "
    else:
        lto_flags = " -flto "

make_cache_dir(cache_path)
make_cache_dir(os.path.join(cache_path, "jit"))
make_cache_dir(os.path.join(cache_path, "obj_files"))
make_cache_dir(os.path.join(cache_path, "gen"))
make_cache_dir(os.path.join(cache_path, "tmp"))
ck_path = os.path.join(cache_path, "checkpoints")
make_cache_dir(ck_path)

# build cache_compile
cc_flags += f" -I\"{core_root(jittor_path)}\" "
cc_flags += f" -I\"{os.path.join(jittor_path, 'extern')}\" "
cc_flags += f" -I\"{backend_root(jittor_path, 'cuda')}\" "

cc_flags += py_include

check_cache_compile()
LOG.v(f"Get cache_compile: {jit_utils.cc}")

# check cuda
is_cuda = has_cuda = has_acl = has_rocm = has_corex = 0
cuda_sdk_flags = cuda_link_flags = ""
check_cuda()
if _requested_backend == "cuda" and not has_cuda:
    raise RuntimeError("JT_BACKEND=cuda requires a usable CUDA compiler; "
                       "set JT_BUILD_NVCC_PATH")
nvcc_flags = build_env("nvcc_flags", "")
def convert_nvcc_flags(value):
    return value
if has_cuda:
    nvcc_flags += cc_flags + cuda_sdk_flags
    def convert_nvcc_flags(nvcc_flags):
        # nvcc don't support -Wall option
        if os.name == 'nt':
            nvcc_flags = nvcc_flags.replace("-fp:", "-Xcompiler -fp:")
            nvcc_flags = nvcc_flags.replace("-EH", "-Xcompiler -EH")
            nvcc_flags = nvcc_flags.replace("-M", "-Xcompiler -M")
            nvcc_flags = nvcc_flags.replace("-utf", "-Xcompiler -utf")
            nvcc_flags = nvcc_flags.replace("-nologo", "")
            nvcc_flags = nvcc_flags.replace("-std:", "-std=")
            nvcc_flags = nvcc_flags.replace("-Fo:", "-o")
            nvcc_flags = nvcc_flags.replace("-LD", "-shared")
            nvcc_flags = nvcc_flags.replace("-LIBPATH:", "-L")
            nvcc_flags = nvcc_flags.replace("-link", "")
            def func(x):
                if ".lib" not in x: return x
                x = x.replace("\"", "")
                a = os.path.dirname(x)
                b = os.path.basename(x)
                if not b.endswith(".lib"):
                    return x
                return f"-L\"{a}\" -l{b[:-4]}"
            nvcc_flags = map_flags(nvcc_flags, func)
        if nvcc_version >= [11,4]:
            nvcc_flags = nvcc_flags.replace("-std=c++17", "-std=c++14 -Xcompiler -std:c++14")
        else:
            nvcc_flags = nvcc_flags.replace("-std=c++17", "")
        nvcc_flags = nvcc_flags.replace("-Wall", "")
        nvcc_flags = nvcc_flags.replace("-Wno-unknown-pragmas", "")
        nvcc_flags = nvcc_flags.replace("-fopenmp", "")
        nvcc_flags = nvcc_flags.replace("-march", "-Xcompiler -march")
        nvcc_flags = nvcc_flags.replace("-Werror", "")
        nvcc_flags = nvcc_flags.replace("-fPIC", "-Xcompiler -fPIC")
        nvcc_flags = nvcc_flags.replace("-fdiagnostics", "-Xcompiler -fdiagnostics")
        nvcc_flags += f" -x cu --cudart=shared -ccbin=\"{cc_path}\" --use_fast_math "
        # nvcc warning is noise
        nvcc_flags += " -w "
        nvcc_flags += f" -I\"{os.path.join(backend_root(jittor_path, 'cuda'), 'include')}\" "
        if build_flag("cuda_debug"):
            nvcc_flags += " -G "
        return nvcc_flags
    nvcc_flags = convert_nvcc_flags(nvcc_flags)

build_config = BuildConfig(
    backend="cuda" if has_cuda else "cpu", cc_path=cc_path, cc_type=cc_type,
    cc_flags=cc_flags, nvcc_path=nvcc_path, nvcc_flags=nvcc_flags,
    kernel_flags=kernel_opt_flags, cache_path=cache_path, jittor_path=jittor_path,
    has_cuda=bool(has_cuda), is_cuda=bool(is_cuda), has_accelerator=bool(has_cuda),
    convert_nvcc_flags=convert_nvcc_flags,
    backend_sources=(
        BuildSource(os.path.join(backend_root(jittor_path, "cuda"), "runtime/driver.cc"), flags=cuda_sdk_flags),
        BuildSource(os.path.join(backend_root(jittor_path, "cuda"), "runtime/nan_checker.cc"), flags=cuda_sdk_flags),
        BuildSource(os.path.join(backend_root(jittor_path, "cuda"), "kernels/debug/nan_checker.cu"),
                    language="cuda", flags=cuda_sdk_flags),
    ) if has_cuda else (),
    backend_link_flags=cuda_link_flags,
    extension_compile_flags=cuda_sdk_flags,
    kernel_compiler=nvcc_path if has_cuda else cc_path,
    kernel_language="cuda" if has_cuda else "cxx",
    kernel_compile_flags=cuda_sdk_flags,
    kernel_device_link=bool(has_cuda),
    kernel_source_roots=(os.path.join(backend_root(jittor_path, "cuda"), "kernels/core"),) if has_cuda else (),
)
_backend_provider = _backend_discovery.load_backend_provider(_requested_backend)
backend_modules = () if _backend_provider is None else (_backend_provider,)
if _backend_provider is not None:
    configured = _backend_provider.configure(make_backend_context(build_config))
    if not isinstance(configured, BuildConfig):
        raise TypeError("backend configure(context) must return BuildConfig")
    build_config = configured
for _env_name, _env_value in build_config.environment.items():
    os.environ[_env_name] = _env_value

# Only the bootstrap publishes compatibility names. Providers return values;
# they never mutate this module or append into its core source inventory.
cc_path, cc_type, cc_flags = build_config.cc_path, build_config.cc_type, build_config.cc_flags
nvcc_path, nvcc_flags = build_config.nvcc_path, build_config.nvcc_flags
kernel_opt_flags = build_config.kernel_flags
jittor_path, cache_path = build_config.jittor_path, build_config.cache_path
has_accelerator = build_config.has_accelerator
has_cuda, has_acl = build_config.has_cuda, build_config.has_acl
has_rocm, has_corex = build_config.has_rocm, build_config.has_corex
hipcc_path, tikcc_path = build_config.hipcc_path, build_config.tikcc_path
setup_fake_cuda_lib = build_config.setup_fake_cuda_lib
extra_core_files = build_config.extra_core_files
convert_nvcc_flags = build_config.convert_nvcc_flags or convert_nvcc_flags

is_cuda = build_config.is_cuda

# build core
core_output_name = 'jittor_core' + extension_suffix
core_output_path = os.path.join(cache_path, core_output_name)
cc_flags += f' -I\"{cache_path}\" -L\"{cache_path}\" -L\"{jit_utils.cache_path}\" '
cc_flags += f" -l\"jit_utils_core{lib_suffix}\" "

#: The flags the core itself is compiled with, frozen here.
#:
#: ``cc_flags`` keeps growing after this point -- ``-ljittor_core`` goes on as
#: soon as the core is linked, and it is what JIT ops are compiled with. So it
#: cannot be what the stamp records: read again after the import it no longer
#: matches what the build used, every check reports "stale", and the whole
#: point of the stamp is lost. That is not a hypothetical; it is what the
#: first version of this did.
core_cc_flags = cc_flags

#: Source files the core was built from, in compile order. Filled in by
#: :func:`build_core` -- from the build itself, or from the stamp it left
#: behind when there was nothing to build.
files = []

#: Stamp version. Bump it when the meaning of a recorded field changes, so an
#: older stamp is treated as "no stamp" instead of being misread as current.
CORE_BUILD_STAMP_VERSION = 2
CORE_GENERATOR_SIGNATURE_VERSION = 1


def core_build_stamp_path():
    return core_output_path + ".build_stamp.json"


def core_source_signature(root=None):
    """``{relative path: [mtime_ns, size]}`` for every core source file.

    ``root`` overrides the tree that is walked. It exists for the tests: the
    startup configuration on this module is frozen after bootstrap (see
    ``_runtime/state.py``), so ``jittor_path`` can no longer be patched, and a
    walk that can only ever look at the real checkout cannot be shown to
    notice a new or same-size-edited file.

    A walk of ``src/`` and ``extern/`` rather than the individual globs the
    generators use, so that a source or header that did not exist at the last
    build shows up too -- a dependency list recorded from that build can only
    name files that were already there.

    External headers (the toolkit's, Python's, the standard library's) are
    deliberately absent. ``cache_path`` is already partitioned by compiler
    version, Python version and cuda key, so a toolchain change lands in a
    different cache directory and gets a cold build rather than a stale one.
    Costs about 3 ms for ~600 files, which is what makes it affordable on
    every import.
    """
    tree = jittor_path if root is None else root
    signature = {}
    roots = [("src", core_root(tree) if root is None else os.path.join(tree, "src")),
             ("extern", os.path.join(tree, "extern"))]
    for backend in ("cpu", "cuda", "acl", "rocm", "corex"):
        try:
            backend_directory = backend_root(tree, backend)
        except FileNotFoundError:
            continue
        roots.append(("backends/" + backend, backend_directory))
    for top, root_dir in roots:
        for directory, _, names in os.walk(root_dir):
            for name in names:
                path = os.path.join(directory, name)
                try:
                    info = os.stat(path)
                except OSError:
                    continue
                signature[os.path.join(top, os.path.relpath(path, root_dir))] = \
                    [info.st_mtime_ns, info.st_size]
    return signature


def core_generator_signature():
    """Describe Python code that generates the core translation units.

    These files live outside ``src/`` and ``extern/``, so the source walk used
    by the core stamp cannot see edits to them.  Keep both a cheap stat record
    and a content digest: an editor or checkout normally changes mtime, while
    the digest also catches an in-place same-size/same-mtime replacement.
    Missing files are represented explicitly so a partial installation cannot
    accidentally reuse a stamp made by a complete one.
    """
    paths = [
        os.path.abspath(__file__),
        os.path.abspath(pyjt_compiler.__file__),
        os.path.abspath(_build_config_api.__file__),
        os.path.abspath(_backend_discovery.__file__),
        os.path.abspath(_backend_resources_api.__file__),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_runtime", "flag_policy.py"),
    ]
    records = {}
    for path in paths:
        relative = os.path.relpath(path, jittor_path)
        try:
            info = os.stat(path)
        except OSError:
            records[relative] = None
            continue
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            records[relative] = None
            continue
        records[relative] = {
            "mtime_ns": info.st_mtime_ns,
            "size": info.st_size,
            "sha256": digest.hexdigest(),
        }
    return {
        "version": CORE_GENERATOR_SIGNATURE_VERSION,
        "files": records,
    }


def core_build_ingredients():
    """Everything besides the sources that the core's compile commands use.

    Kept as the ingredients rather than the assembled command lines because
    assembling those needs the generated pyjt sources, which is one of the
    steps a current build gets to skip.
    """
    return {
        "version": __version__,
        "cc_path": cc_path,
        "cc_type": cc_type,
        "cc_flags": core_cc_flags,
        "opt_flags": opt_flags,
        "lto_flags": lto_flags,
        "nvcc_path": nvcc_path,
        "nvcc_flags": build_env("nvcc_flags", ""),
        "extension_suffix": extension_suffix,
        "lib_suffix": lib_suffix,
        "has_cuda": int(bool(has_cuda)),
        "has_accelerator": int(bool(has_accelerator)),
        "has_rocm": int(bool(has_rocm)),
        "is_cuda": int(bool(is_cuda)),
        "extra_core_files": list(extra_core_files),
        "backend_sources": [vars(source) for source in build_config.backend_sources],
        "backend_link_flags": build_config.backend_link_flags,
        "kernel_compiler": build_config.kernel_compiler,
        "kernel_language": build_config.kernel_language,
        "kernel_compile_flags": build_config.kernel_compile_flags,
        "jit_utils_core_files": list(jit_utils_core_files),
        "os_name": os.name,
        "generators": core_generator_signature(),
    }


def _core_output_signature():
    try:
        info = os.stat(core_output_path)
    except OSError:
        return None
    return [info.st_mtime_ns, info.st_size]


def core_build_is_current(signature=None, ingredients=None):
    """True when ``jittor_core`` is already built from exactly these inputs.

    The point of asking is that finding out the expensive way costs about
    0.9 s of every warm import: regenerating headers that come out
    byte-identical, then handing all ~180 compile commands to a 16-process
    pool so each worker can hash a dependency closure and report that there
    is nothing to do.

    Deliberately conservative in one direction only. A mismatch means "build",
    and building re-runs the full per-file check that was always there, so a
    stamp that is wrong about being stale costs time and nothing else. The one
    way to be wrong in the other direction is to edit a file without changing
    either its size or its nanosecond mtime, which no editor, compiler or
    ``git checkout`` does.
    """
    stamp = None
    try:
        with open(core_build_stamp_path(), "r", encoding="utf8") as handle:
            stamp = json.load(handle)
    except (OSError, ValueError):
        return False
    if not isinstance(stamp, dict):
        return False
    if stamp.get("stamp_version") != CORE_BUILD_STAMP_VERSION:
        return False
    if stamp.get("output") != _core_output_signature():
        return False
    if stamp.get("ingredients") != (ingredients or core_build_ingredients()):
        return False
    if stamp.get("sources") != (signature or core_source_signature()):
        return False
    if not isinstance(stamp.get("files"), list):
        return False
    return stamp


def _write_core_build_stamp(signature, ingredients, compile_order):
    """Record what the core was just built from.

    Written to a temporary name and renamed into place: a reader that arrives
    mid-write must see either the old stamp or the new one, never a truncated
    file that json would reject and that would silently cost the next import
    a full rebuild.
    """
    stamp = {
        "stamp_version": CORE_BUILD_STAMP_VERSION,
        "output": _core_output_signature(),
        "ingredients": ingredients,
        "sources": signature,
        "files": list(compile_order),
    }
    if stamp["output"] is None:
        return
    temporary = core_build_stamp_path() + ".tmp." + str(os.getpid())
    try:
        with open(temporary, "w", encoding="utf8") as handle:
            json.dump(stamp, handle)
        os.replace(temporary, core_build_stamp_path())
    except OSError as error:
        # A read-only or full cache directory must not stop an import that
        # otherwise succeeded; the cost is that the next import checks the
        # expensive way again.
        LOG.v("could not write core build stamp: %s" % error)
        try:
            os.remove(temporary)
        except OSError:
            pass


class BuildNotAllowed(Exception):
    """Raised instead of compiling when builds are not allowed.

    ``import jittor`` is supposed to load a build, not produce one, but for
    most of Jittor's history the difference was invisible: a cache that did
    not match compiled for forty seconds to several minutes and said so only
    through progress lines. There was no way for a deployment to state "this
    import must not build" and be told when it was about to.

    ``JITTOR_NO_BUILD=1`` is that statement. Under it, anything that would
    have to compile raises this instead, naming the explicit entry point that
    is allowed to build. Offline and read-only installations are the reason it
    exists: there the compile does not merely cost a minute, it fails a minute
    later for a reason that has nothing to do with the actual problem.
    """


def build_is_allowed():
    """False when ``JITTOR_NO_BUILD`` forbids compiling during this process."""
    return os.environ.get("JITTOR_NO_BUILD", "0") in ("0", "")


def _refuse_build(what):
    raise BuildNotAllowed(
        "JITTOR_NO_BUILD=1 is set, and %s is not built for this "
        "configuration -- importing jittor would have to compile it.\n"
        "Build it once with the explicit entry point:\n"
        "    python -m jittor_utils.bootstrap\n"
        "then import with the same JITTOR_HOME and the same build flags "
        "(nvcc_path, cc_flags, nvcc_flags), because each combination gets "
        "its own cache directory.\n"
        "cache_path: %s" % (what, cache_path))


#: Stamp version for build products other than the core. Independent of the
#: core's, so either can change meaning without invalidating the other's.
PRODUCT_BUILD_STAMP_VERSION = 1

_SOURCE_EXTENSIONS = (".h", ".hpp", ".cuh", ".inc", ".cc", ".cpp", ".cu", ".c")


def product_build_stamp_path(product):
    """Where the stamp for a build product lives: right next to it.

    Beside the product rather than in an index, so that deleting the product
    -- by hand, or with ``jittor_utils.clean_cache`` -- cannot leave a stamp
    behind that claims it is still there.
    """
    return product + ".build_stamp.json"


def _stat_signature(paths):
    """``{path: [mtime_ns, size]}``, with ``None`` for a path that is gone.

    A missing path is recorded explicitly rather than skipped: dropping it
    would make "the file was deleted" indistinguishable from "the file was
    never listed", and the second one must not be treated as up to date.
    """
    record = {}
    for path in paths:
        try:
            info = os.stat(path)
        except OSError:
            record[path] = None
            continue
        record[path] = [info.st_mtime_ns, info.st_size]
    return record


def _is_within(path, root):
    return path == root or path.startswith(root + os.sep)


def _include_tree_signature(directories):
    """Stat every source and header under each of the caller's include dirs.

    ``filenames`` names the op sources and their paired headers; it does not
    name the sibling headers those include. The per-file dependency scan
    inside :func:`compile` does find them -- and that scan is exactly what the
    stamp exists to skip, so the stamp has to cover them another way.

    Two kinds of directory are deliberately not walked, because walking them
    costs more than it can ever detect:

    * anything inside the Jittor tree -- :func:`core_source_signature` already
      covers it, including ``backends/*/libraries/*/include``;
    * the CUDA SDK's include directories. ``cache_path`` is partitioned by the
      cuda key, so a different toolkit builds into a different directory and
      gets a cold build rather than a stale one. This is the same argument
      :func:`core_source_signature` makes for external headers -- and it is
      what keeps this off the import path: those trees hold ~1400 headers
      each, and walking six of them cost 90 ms of every warm CUDA import.

    What is left is the caller's own directories, which is the case that
    matters for a custom op built outside this repository.
    """
    signature = {}
    tree = os.path.realpath(jittor_path)
    toolkit = [os.path.realpath(path) for path in cuda_include_dirs]
    for directory in sorted(set(directories)):
        if not directory:
            continue
        real = os.path.realpath(directory)
        if _is_within(real, tree):
            continue
        if any(_is_within(real, path) or _is_within(path, real)
               for path in toolkit):
            continue
        for current, _, names in os.walk(real):
            for name in names:
                if not name.endswith(_SOURCE_EXTENSIONS):
                    continue
                path = os.path.join(current, name)
                try:
                    info = os.stat(path)
                except OSError:
                    continue
                signature[os.path.relpath(path, real)] = \
                    [info.st_mtime_ns, info.st_size]
    return signature


def _custom_op_include_dirs(filenames, include_dirs, extra_flags):
    """Every directory the custom-op compile can reach a header through.

    The auto-derived ones, the directories the named files live in, and the
    ``-I`` the caller passed by hand in ``extra_flags`` -- that last one is
    the only way a public caller can point the compile at a tree Jittor knows
    nothing about, so parsing it out is what keeps the stamp honest for code
    outside this repository.
    """
    directories = list(include_dirs)
    directories += [os.path.dirname(os.path.realpath(name))
                    for name in filenames]
    for match in re.finditer(r'-I\s*(?:"([^"]*)"|(\S+))', extra_flags or ""):
        directories.append(match.group(1) or match.group(2))
    return directories


def custom_op_build_ingredients(gen_name, extra_flags, include_flags, backend):
    """Everything besides the sources that the compile commands use.

    ``cc_flags`` is safe to record verbatim here, unlike for the core: by the
    time any custom op is built the core is linked and ``-ljittor_core`` is
    already on it, so re-reading it after the import yields the same string
    the build used.
    """
    return {
        "version": __version__,
        "gen_name": gen_name,
        "cc_path": cc_path,
        "cc_type": cc_type,
        "cc_flags": cc_flags,
        "opt_flags": opt_flags,
        "extra_flags": extra_flags,
        "include_flags": include_flags,
        "backend": backend,
        "nvcc_path": nvcc_path,
        "nvcc_flags": nvcc_flags,
        "extension_suffix": extension_suffix,
        "lib_suffix": lib_suffix,
        "has_cuda": int(bool(has_cuda)),
        "has_accelerator": int(bool(has_accelerator)),
        "core_output": _core_output_signature(),
        "generators": core_generator_signature(),
    }


def product_build_is_current(product, sources, ingredients):
    """True when ``product`` was already built from exactly these inputs.

    Same bargain as the core's stamp, for the same reason: finding out the
    expensive way cost about 0.35 s of every warm CUDA import, because the
    bundled cuDNN/cuBLAS/cuRAND/cuFFT/cuSPARSE/CUB op libraries handed ~50
    compile commands to the pool so each worker could hash a dependency
    closure and report that there was nothing to do.

    Conservative in one direction only. A mismatch means "build", and
    building re-runs the full per-file dependency check that was always there,
    so a stamp that wrongly reports "stale" costs time and nothing else.

    The callers decide what goes in ``sources`` and ``ingredients``; this only
    promises that a build is current when *all* of it matches. Getting those
    two right is the whole safety argument, because the direction that is not
    safe -- claiming current when it is not -- does not fail, it silently
    keeps running the previous build.
    """
    try:
        with open(product_build_stamp_path(product), "r",
                  encoding="utf8") as handle:
            stamp = json.load(handle)
    except (OSError, ValueError):
        return False
    if not isinstance(stamp, dict):
        return False
    if stamp.get("stamp_version") != PRODUCT_BUILD_STAMP_VERSION:
        return False
    if stamp.get("output") != _stat_signature([product]).get(product):
        return False
    if stamp.get("ingredients") != ingredients:
        return False
    if stamp.get("sources") != sources:
        return False
    return True


def _write_product_build_stamp(product, sources, ingredients):
    """Record what a build product was just built from.

    Written to a temporary name and renamed, so a reader arriving mid-write
    sees either the old stamp or the new one, never a truncated file.
    """
    stamp = {
        "stamp_version": PRODUCT_BUILD_STAMP_VERSION,
        "output": _stat_signature([product]).get(product),
        "ingredients": ingredients,
        "sources": sources,
    }
    if stamp["output"] is None:
        return
    path = product_build_stamp_path(product)
    temporary = path + ".tmp." + str(os.getpid())
    try:
        with open(temporary, "w", encoding="utf8") as handle:
            json.dump(stamp, handle)
        os.replace(temporary, path)
    except OSError as error:
        # A read-only or full cache directory must not fail a build that
        # otherwise succeeded; the cost is one expensive check next time.
        LOG.v("could not write build stamp for %s: %s" % (path, error))
        try:
            os.remove(temporary)
        except OSError:
            pass


def compile_if_stale(what, compiler_path, flags, sources, output,
                     extra_ingredients=None):
    """:func:`compile`, skipped when the stamp says ``output`` is current.

    For the products that are neither the core nor a custom op library but
    still sit on the import path. ``libcuda_extern`` is the whole reason this
    exists: two translation units, nothing to do on a warm import, and 0.073 s
    of every one of them spent proving it -- the last compile fan-out left on
    the import path once the core and the op libraries had stamps.

    Returns True if it built.
    """
    stamp_sources = {
        "files": _stat_signature([os.path.realpath(name) for name in sources]),
        "core_sources": core_source_signature(),
    }
    ingredients = {
        "version": __version__,
        "what": what,
        "cc_path": compiler_path,
        "flags": flags,
        "opt_flags": opt_flags,
        "extension_suffix": extension_suffix,
        "generators": core_generator_signature(),
    }
    if extra_ingredients:
        ingredients.update(extra_ingredients)
    if product_build_is_current(output, stamp_sources, ingredients):
        LOG.v("%s is current, skipping build" % what)
        return False
    if not build_is_allowed():
        _refuse_build(what)
    compile(compiler_path, flags, sources, output)
    _write_product_build_stamp(output, stamp_sources, ingredients)
    return True


def build_core(force=False):
    """Generate the core's sources and compile ``jittor_core``.

    The single entry point for turning ``src/**`` into the core extension:
    the flag and test scanners, the op-maker header, the pyjt bindings, the
    compile order, and the compile itself. It used to be nine hundred lines
    of module body with no name, which is why "does importing jittor build
    anything" had no answer short of reading all of it.

    Returns True if it built, False if the stamp said the build was already
    current -- in which case nothing was generated, no compiler ran, and the
    only cost was stat()ing the source tree.
    """
    global files
    signature = core_source_signature()
    ingredients = core_build_ingredients()

    if not force:
        stamp = core_build_is_current(signature, ingredients)
        if stamp:
            files = stamp["files"]
            LOG.v("core build is current, skipping generation and compile")
            return False
        if not build_is_allowed():
            _refuse_build("jittor_core")

    gen_jit_flags()
    gen_jit_tests()
    op_headers = glob.glob(core_root(jittor_path)+"/ops/**/*op.h", recursive=True)
    jit_src = gen_jit_op_maker(op_headers)
    LOG.vvvv(jit_src)
    with open(os.path.join(cache_path, "gen", "jit_op_maker.h"), 'w', encoding='utf8') as f:
        f.write(jit_src)
    # gen pyjt
    pyjt_gen_src = pyjt_compiler.compile(cache_path, jittor_path)

    # initialize order:
    # 1. registers
    # 2. generate source
    # 3. op_utils
    # 4. other
    files2 = pyjt_gen_src
    files4 = glob.glob(core_root(jittor_path)+"/**/*.cc", recursive=True)
    # Keep the historical "src/..." spelling: at_beginning/at_last and
    # jit_utils_core_files below are matched with list.remove, which compares
    # strings. 4.15 moved the core out of jittor_path, so the old
    # f[len(jittor_path)+1:] slice would produce garbage; relpath against
    # core_root reproduces exactly the same names, and _source_path maps them
    # back to wherever the core now lives.
    files4 = [os.path.join("src", os.path.relpath(f, core_root(jittor_path)))
              for f in files4]
    indexing_schedule_source = os.path.join(
        backend_root(jittor_path, "cuda"), "kernels", "core", "indexing_schedule_codegen.cc")
    files4.append(indexing_schedule_source)
    if has_accelerator:
        files4 += [path for path in sorted(glob.glob(os.path.join(
            backend_root(jittor_path, "cuda"), "kernels", "core", "*_codegen.cc")))
            if path != indexing_schedule_source]
    at_beginning = [
        "src/ops/op_utils.cc",
        "src/ops/op_register.cc",
        "src/runtime/init.cc",
        "src/core/event_queue.cc",
        "src/mem/allocator/sfrl_allocator.cc",
        "src/mem/allocator.cc",
        "src/type/nano_string.cc",
    ]
    at_last = [
        "src/runtime/profiler/profiler.cc",
        "src/core/executor.cc",
    ]
    if os.name == 'nt':
        at_beginning = [ x.replace('/','\\') for x in at_beginning ]
        at_last = [ x.replace('/','\\') for x in at_last ]
    for i in range(len(at_beginning)):
        files4.remove(at_beginning[i])
        files4.insert(i, at_beginning[i])
    for v in at_last:
        files4.remove(v)
        files4.append(v)
    registers = [ name for name in files4 if "register" in name ]
    for name in registers: files4.remove(name)
    files = registers + files2 + files4
    files += extra_core_files
    for file in jit_utils_core_files:
        files.remove(file)
    LOG.vv("compile order:", files)

    # Everything a build needs, checked once and reported together -- but only
    # when there is actually a build to do. These preconditions used to be
    # checked in whatever order the module-level code ran in, so a user
    # missing three of them found out one `pip install` at a time, paying a
    # cold build between each.
    if not os.path.isfile(core_output_path):
        from jittor_utils import preflight as _preflight
        _preflight.assert_ready(config=build_config)

    try:
        provider_objects = compile_backend_sources(build_config, core_cc_flags+opt_flags+lto_flags)
        compile(cc_path, core_cc_flags+opt_flags+build_config.backend_link_flags,
                files + provider_objects, core_output_name)
    except RuntimeError as error:
        # A build that fails here usually fails for a reason the preconditions
        # above can name. Say all of them, rather than leaving the user with a
        # compiler diagnostic and nothing to act on.
        from jittor_utils import preflight as _preflight
        try:
            _report = _preflight.format_report(_preflight.run_all(config=build_config),
                                               only_problems=True)
        except Exception:
            raise error
        if _report and _report != "all build preconditions satisfied":
            raise RuntimeError("%s\n\nBuild preconditions on this machine:\n%s"
                               % (error, _report))
        raise

    # After the compile, so the stamp records the product that exists now. The
    # signature is the pre-build one on purpose: it is the state the sources
    # were in when this build was decided on, and a source edited *during* the
    # build must leave the stamp stale rather than claim to cover the edit.
    files += [source.path for source in build_config.backend_sources]
    _write_core_build_stamp(signature, ingredients, files)
    return True


if platform.system() == 'Linux':
    libname = {"clang":"omp", "icc":"iomp5", "g++":"gomp"}[cc_type]
    openmp_name = libname
    libname = ctypes.util.find_library(libname)
    if libname is None:
        raise RuntimeError(
            f"the OpenMP runtime lib{openmp_name} was not found, and every "
            f"CPU kernel Jittor compiles links against it. Install it "
            f"(libgomp for g++, libomp for clang) or set cc_path to a "
            f"compiler whose runtime is installed.")
    ctypes.CDLL(libname, os.RTLD_NOW | os.RTLD_GLOBAL)

# NOTE: sw_64 used to disable TLS certificate verification for the whole
# process here. See the note in jittor_utils/__init__.py: use SSL_CERT_FILE or
# certifi instead; nothing jittor does justifies turning it off for the
# application it is imported into.

build_core()
cc_flags += f" -l\"jittor_core{lib_suffix}\" "

with jit_utils.import_scope(import_flags):
    import jittor_core as core

flags = core.Flags()

if has_cuda and is_cuda and not hasattr(flags, "cuda_archs"):
    # The core just imported has no CUDA support while this process found nvcc
    # and is about to configure CUDA. The usual cause is a stale CPU-only
    # jittor_core one directory above the CUDA one -- any run that imports
    # Jittor without nvcc builds it there, and it then shadows the CUDA build
    # for every later run that shares the cache. Left unreported, the process
    # keeps going on CPU and every CUDA op fails with "Op ... doesn't have cuda
    # version"; this line would otherwise raise a bare AttributeError. Name the
    # file that actually got imported, which is the one to remove.
    raise RuntimeError(
        "jittor_core was imported from {}, which is built without CUDA "
        "support, but this process found nvcc at {} and expected the CUDA "
        "build in {}. Remove the file named first to stop it shadowing the "
        "CUDA build, or unset nvcc_path to keep running on CPU.".format(
            getattr(core, "__file__", "<unknown>"), nvcc_path, cache_path)
    )

if has_cuda and is_cuda:
    nvcc_flags = " " + build_env("nvcc_flags", "") + " "
    nvcc_flags += convert_nvcc_flags(cc_flags)
    nvcc_version = list(jit_utils.get_int_version(nvcc_path))
    supported_archs = query_nvcc_archs(nvcc_path)
    if supported_archs:
        max_arch = max(supported_archs)
    else:
        # nvcc too old for --list-gpu-arch (before CUDA 11.1), so the ceiling
        # has to come from its version. This table only has to stay correct
        # for toolkits that cannot answer for themselves, which is why it is
        # no longer the source of the answer for the ones that can.
        max_arch = 86
        if nvcc_version < [11,]:
            max_arch = 75
        elif nvcc_version < [11,1]:
            max_arch = 80
    if len(flags.cuda_archs):
        archs = select_cuda_archs(flags.cuda_archs, max_arch)
        flags.cuda_archs = archs
        nvcc_flags += cuda_arch_flags(archs)

flags.cc_path = cc_path
flags.cc_type = cc_type
flags.cc_flags = cc_flags + kernel_opt_flags
flags.nvcc_path = nvcc_path
flags.nvcc_flags = nvcc_flags
flags.python_path = python_path
flags.cache_path = cache_path
flags.jittor_path = jittor_path
flags.gdb_path = gdb_path
flags.addr2line_path = addr2line_path
flags.has_pybt = has_pybt

build_config = build_config.evolve(
    cc_flags=cc_flags, nvcc_flags=nvcc_flags, is_cuda=bool(is_cuda),
    kernel_flags=kernel_opt_flags,
)

# The accelerator JIT compiler is configured, not inferred from flags: the
# core's `configure_accelerator_compiler` is the only way it learns which
# compiler builds a device kernel. Without this call every accelerator JIT op
# fails with "Accelerator compiler is not configured", which is what an
# unconfigured accelerator looks like from the inside.
if build_config.has_accelerator and build_config.kernel_compiler:
    core.configure_accelerator_compiler(
        build_config.kernel_compiler,
        build_config.kernel_compile_flags + nvcc_flags
        if build_config.kernel_language == "cuda" else build_config.kernel_compile_flags,
        build_config.kernel_language,
        build_config.kernel_source_suffix,
        list(build_config.kernel_flag_filter),
        build_config.kernel_device_link,
    )
jit_utils.configure_module_build(_module_build_services(build_config))

# Hand the one lock descriptor over to C++. Both sides now take flock() on
# this single open file description; before this line the C++ side took a
# POSIX record lock on a descriptor of its own, which excluded neither the
# Python side nor anything else, and whose release was tied to whichever
# descriptor happened to be closed first.
lock.jittor_lock.bind_core(core)
