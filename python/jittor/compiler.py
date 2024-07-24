# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import subprocess as sp
import os
import re
import sys
import glob
import inspect
import datetime
import threading
import platform
import ctypes
import platform
from ctypes import cdll
from ctypes.util import find_library

import jittor_utils as jit_utils
from jittor_utils import LOG, run_cmd, find_exe, cc_path, cc_type, cache_path
from . import pyjt_compiler
from jittor_utils import lock
from jittor_utils import install_cuda
from jittor import __version__
import hashlib

def find_jittor_path():
    return os.path.dirname(__file__)

def make_cache_dir(cache_path):
    if not os.path.isdir(cache_path):
        LOG.i(f"Create cache dir: {cache_path}")
        os.mkdir(cache_path)

def shsplit(s):
    s1 = s.split(' ')
    s2 = []
    count = 0
    for s in s1:
        nc = s.count('"') + s.count('\'')
        if count&1:
            count += nc
            s2[-1] += " "
            s2[-1] += s
        else:
            count = nc
            s2.append(s)
    return s2


def remove_flags(flags, rm_flags):
    flags = shsplit(flags)
    output = []
    for s in flags:
        ss = s.replace("\"", "")
        for rm in rm_flags:
            if ss.startswith(rm) or ss.endswith(rm):
                break
        else:
            output.append(s)
    return " ".join(output)

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

def compile(compiler, flags, inputs, output, combind_build=False, cuda_flags="", obj_dirname="obj_files"):
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
        if cc_type != 'cl':
            # initialize order in windows seems reversed
            inputs = list(inputs[::-1])
            link = link + f' -Wl,--export-all-symbols,--out-implib,"{afile}" '
            
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
            new_inputs.append(os.path.join(jittor_path, name))
            obj_files.append(os.path.join(
                obj_dir, os.path.basename(name)+".o"))
    inputs = new_inputs
    cm = lambda s: f"\"{s}\""
    cms = lambda arr: [f"\"{s}\"" for s in arr ]

    if len(inputs) == 1 or combind_build:
        cmd = f"\"{compiler}\" {' '.join(cms(inputs))} {flags} -o {cm(output)}"
        return do_compile(fix_cl_flags(cmd))
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
        dumpdef_path = os.path.join(jittor_path, "utils", "dumpdef.py")
        cmd = f"\"{sys.executable}\" \"{dumpdef_path}\" {' '.join(cms(obj_files))} -Fo: \"{output}.def\""
        do_compile(fix_cl_flags(cmd))
    cmd = f"\"{compiler}\" {' '.join(cms(obj_files))} -o {cm(output)} {flags} {lto_flags}"
    return do_compile(fix_cl_flags(cmd))

def gen_jit_tests():
    all_src = glob.glob(jittor_path+"/src/**/*.cc", recursive=True)
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
    #include "common.h"

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

def gen_jit_flags():
    all_src = glob.glob(jittor_path+"/src/**/*.cc", recursive=True)
    jit_declares = []
    re_def = re.compile("DEFINE_FLAG(_WITH_SETTER)?\\((.*?)\\);", re.DOTALL)

    flags_defs = []
    visit = {}
    
    for src_name in all_src:
        with open(src_name, 'rb') as f:
            src = f.read().decode("utf8")
        defs = re_def.findall(src)
        for _, args in defs:
            args = args.split(",")
            type = args[0].strip()
            name = args[1].strip()
            if not has_cuda and "cuda" in name and name!="use_cuda":
                if name != "use_cuda_host_allocator":
                    continue
            default = args[2].strip()
            doc = ",".join(args[3:])
            doc = eval(f"({doc})")
            LOG.vv(f"Find define {name} from {src_name}")
            if name in visit:
                continue
            visit[name] = 1
            jit_declares.append(f"DECLARE_FLAG({type}, {name});")
            alias = []
            if name == "use_cuda":
                alias = ["use_device", "use_acl", "use_rocm", "use_corex"]
            elif name == "auto_mixed_precision_level":
                alias = ["amp_level"]
            get_names = ",".join(["__get__"+a for a in [name]+alias])
            set_names = ",".join(["__set__"+a for a in [name]+alias])
            flags_defs.append(f"""
                /* {name}(type:{type}, default:{default}): {doc} */
                // @pyjt({get_names})
                {type} _get_{name}() {{ return {name}; }}
                // @pyjt({set_names})
                void _set_{name}({type} v) {{ set_{name}(v); }}
                {f'''// @pyjt({set_names})
                void _set_{name}(bool v) {{ set_{name}(v); }}
                ''' if type=="int" else ""}
            """)
    
    jit_declares = "\n    ".join(jit_declares)
    jit_src = f"""
    #include "utils/flags.h"

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

def gen_jit_op_maker(op_headers, export=False, extra_flags=""):
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
                if (_op->flags.get(NodeFlags::_forwarded)) {{
                    VarPtr _out(move(_op->outputs_holder[0]));
                    delete _op;
                    return _out;
                }}
                _op->outputs_holder[0]->set_inputs({{_op}});
                VarPtr _out(move(_op->outputs_holder[0]));
                {src.replace("->var","")};
                _op->init();
                return _out;
            }}
            """)
        else:
            jit_cc_src.append(f"""
            vector<VarPtr> make_{cc_func_name}({", ".join(cc_make_args)}) {{
                auto _op = new {op_name}({", ".join(op_make_args)});
                if (_op->flags.get(NodeFlags::_forwarded)) {{
                    vector<VarPtr> _outs = move(_op->outputs_holder);
                    delete _op;
                    return _outs;
                }}
                vector<VarPtr> _outs = move(_op->outputs_holder);
                for (uint i=0; i<_outs.size(); i++)
                    _outs[i]->set_inputs({{_op}});
                {src.replace("->var","")};
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
            constructors.append(f"{{ &typeid(&{name}), (void*)&{name} }}")
        constructors = ",".join(constructors)
        var_member_reg = r"\n\s*Var\b(.*);"
        var_member_match = re.findall(var_member_reg, src)
        var_member_match = " ".join(var_member_match)
        for c in "*,": var_member_match = var_member_match.replace(c, " ")
        var_member = var_member_match.split()
        LOG.vv("var_member_match "+var_member_match)
        LOG.vv("var_member "+str(var_member))
        var_member_src = [ f"VAR_MEMBER_NAME_AND_OFFSET({name}, {name2})" for name in var_member ]
        var_member_src = ",".join(var_member_src)
        initer.append(f'\n        op_registe({{ "{func_name}", R"({cc_name})", extra_flags, {{{constructors}}}, {{{var_member_src}}} }});')
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
                with open(os.path.join(jittor_path, f"src/ops/{func_name}_op.cc"), encoding="utf-8") as f:
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
    #include "pyjt/py_obj_holder.h"
    #include "var.h"
    #include "var_holder.h"
    #include "ops/op_register.h"
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
        #include "op.h"
        #include "var.h"
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
    gen_name_ = ""):
    """Compile custom ops
    filenames: path of op source files, filenames must be
        pairs of xxx_xxx_op.cc and xxx_xxx_op.h, and the 
        type name of op must be XxxXxxOp.
    extra_flags: extra compile flags
    return_module: return module rather than ops(default: False)
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

    includes = sorted(list(set(includes)))
    includes = "".join(map(lambda x: f" -I\"{x}\" ", includes))
    LOG.vvvv(f"Include flags:{includes}")

    op_extra_flags = includes + extra_flags

    lib_path = os.path.join(cache_path, "custom_ops")
    make_cache_dir(lib_path)
    gen_src_fname = os.path.join(lib_path, gen_name+".cc")
    gen_head_fname = os.path.join(lib_path, gen_name+".h")
    gen_lib = os.path.join(lib_path, gen_name+extension_suffix)
    libname = gen_name + lib_suffix
    op_extra_flags += f" -L\"{lib_path}\" -l\"{libname}\" "

    gen_src = gen_jit_op_maker(headers.values(), export=gen_name, extra_flags=op_extra_flags)
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

    # add python path and import
    LOG.vvv(f"Import custum ops lib:{gen_lib}")
    lib_path = os.path.join(cache_path, "custom_ops")
    if lib_path not in os.sys.path:
        os.sys.path.append(lib_path)
    # unlock scope when initialize
    with lock.unlock_scope():
        with jit_utils.import_scope(dlopen_flags):
            exec(f"import {gen_name}")
    mod = locals()[gen_name]
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

def compile_extern():
    # compile llvm passes
    if cc_type != "clang" or platform.system() != 'Linux':
        return
    global kernel_opt_flags
    cache_path_llvm = os.path.join(cache_path, "llvm")
    jittor_path_llvm = os.path.join(jittor_path, "extern", "llvm")
    clang_dir = os.path.dirname(get_full_path_of_executable(cc_path))
    assert clang_dir.endswith("bin") and "llvm" in clang_dir, f"Wrong clang_dir: {clang_dir}"
    llvm_include = os.path.abspath(os.path.join(clang_dir, "..", "include"))
    assert os.path.isdir(llvm_include), "LLVM include path not found"
    make_cache_dir(cache_path_llvm)
    files = os.listdir(jittor_path_llvm)
    # test_pass.cc is used for test link problem of llvm pass plugin
    test_pass_path = os.path.join(cache_path_llvm, "test_pass.cc")
    with open(test_pass_path, 'w', encoding='utf8') as f:
        f.write("int main() {return 0;}")
    
    # -fno-rtti fix link error

    # -Wl,-znodelete fix segfault
    # https://github.com/sampsyo/llvm-pass-skeleton/issues/7#issuecomment-401834287

    # -D_GLIBCXX_USE_CXX11_ABI=0 fix undefined symbol: createPrinterPass
    # https://stackoverflow.com/questions/37366291/undefined-symbol-for-self-built-llvm-opt

    # try different flags
    try_flags = [
        " -Wl,-znodelete -D_GLIBCXX_USE_CXX11_ABI=0 ",
        " -Wl,-znodelete ",
    ]
    found_flags_id = -1
    for fname in files:
        for i, flag in enumerate(try_flags):
            if found_flags_id != -1 and found_flags_id != i:
                continue
            so_name = os.path.join(cache_path_llvm, os.path.splitext(fname)[0]+f".{i}.so")
            compile(
                cc_path,
                f"{cc_flags} {opt_flags} {flag} -I'{llvm_include}'",
                [os.path.join(jittor_path_llvm, fname)],
                so_name
            )
            # if not found available flags, we test it.
            if found_flags_id == -1:
                try:
                    s = run_cmd(
                        f"{cc_path} {cc_flags} -Xclang -load -Xclang '{so_name}' {test_pass_path}",
                        cache_path_llvm,
                        print_error=False
                    )
                except Exception as e:
                    LOG.v(f"Try flag {flag} failed: {e}")
                    continue
                found_flags_id = i
            kernel_opt_flags += f" -Xclang -load -Xclang '{so_name}' "
            break
        else:
            LOG.w("Clang is used, but LLVM pass plugin is unable to link.")
            break
    LOG.vv(f"Compile extern llvm passes: {str(files)}")

def check_cuda():
    if not nvcc_path:
        return
    global cc_flags, has_cuda, is_cuda, core_link_flags, cuda_dir, cuda_lib, cuda_include, cuda_home, cuda_bin
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
    cuda_include2 = os.path.join(jittor_path, "extern","cuda","inc")
    cc_flags += f" -DHAS_CUDA -DIS_CUDA -I\"{cuda_include}\" -I\"{cuda_include2}\" "
    if os.name == 'nt':
        cuda_lib = os.path.abspath(os.path.join(cuda_dir, "..", "lib", "x64"))
        # cc_flags += f" \"{cuda_lib}\\cudart.lib\" "
        cuda_lib_path = glob.glob(cuda_bin+"/cudart64*")[0]
        cc_flags += f" -lcudart -L\"{cuda_lib}\" -L\"{cuda_bin}\" "
        dll = ctypes.CDLL(cuda_lib_path, dlopen_flags)
        ret = dll.cudaDeviceSynchronize()
        assert ret == 0
    else:
        cc_flags += f" -lcudart -L\"{cuda_lib}\" "
        # ctypes.CDLL(cuda_lib+"/libcudart.so", import_flags)
        ctypes.CDLL(cuda_lib+"/libcudart.so", dlopen_flags)
    is_cuda = has_cuda = 1

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
    recompile = compile(cc_path, cc_flags+f" {opt_flags} ", files, jit_utils.cache_path+'/jit_utils_core'+extension_suffix, True)
    if recompile and jit_utils.cc:
        LOG.e("jit_utils updated, please rerun your command.")
        sys.exit(0)
    if not jit_utils.cc:
        with jit_utils.import_scope(import_flags):
            jit_utils.try_import_jit_utils_core()
        assert jit_utils.cc
        # recompile, generate cache key
        compile(cc_path, cc_flags+f" {opt_flags} ", files, jit_utils.cache_path+'/jit_utils_core'+extension_suffix, True)

def env_or_try_find(name, bname):
    if name in os.environ:
        path = os.environ[name]
        if path != "":
            version = jit_utils.get_version(path)
            LOG.i(f"Found {bname}{version} at {path}")
        return path
    return try_find_exe(bname)

def try_find_exe(*args):
    try:
        return find_exe(*args)
    except:
        LOG.v(f"{args[0]} not found.")
        return ""

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
    if os.environ.get("debug")=="1":
        is_debug = 1
        global cc_flags
        cc_flags += " -g -DNODE_MEMCHECK "

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

# if jtcuda is already installed
nvcc_path = None
if install_cuda.has_installation() or os.name == 'nt':
    nvcc_path = install_cuda.install_cuda()
    if nvcc_path:
        nvcc_path = try_find_exe(nvcc_path)
# check system installed cuda
if not nvcc_path:
    nvcc_path = env_or_try_find('nvcc_path', 'nvcc') or \
        try_find_exe('/usr/local/cuda/bin/nvcc') or \
        try_find_exe('/usr/bin/nvcc') or \
        try_find_exe('/opt/cuda/bin/nvcc')
# if system has no cuda, install jtcuda
if not nvcc_path:
    nvcc_path = install_cuda.install_cuda()
    if nvcc_path:
        nvcc_path = try_find_exe(nvcc_path)
if nvcc_path is None:
    nvcc_path = ""
if "nvcc_path" in os.environ:
    nvcc_path = os.environ["nvcc_path"]
gdb_path = env_or_try_find('gdb_path', 'gdb')
addr2line_path = try_find_exe('addr2line')
has_pybt = check_pybt(gdb_path, python_path)

if nvcc_path:
    # gen cuda key for cache_path
    cu = "cu"
    v = jit_utils.get_version(nvcc_path)[1:-1]
    nvcc_version = list(map(int,v.split('.')))
    cu += v
    try:
        r, s = sp.getstatusoutput(f"log_v=0 {sys.executable} -m jittor_utils.query_cuda_cc")
        if r==0:
            s = sorted(list(set(s.strip().split())))
            if len(s)==0:
                LOG.e("No GPU Device Found!")
            cu += "_sm_" + "_".join(s)
            if "cuda_arch" not in os.environ:
                os.environ["cuda_arch"] = " ".join(cu)
            cu = cu.replace(":", "").replace(" ", "")
    except:
        pass
    LOG.i("cuda key:", cu)
    cache_path = os.path.join(cache_path, cu)
    sys.path.append(cache_path)


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
if "cc_flags" in os.environ:
    cc_flags += os.environ["cc_flags"] + ' '

cc_flags += " -lstdc++ -ldl -shared "

opt_flags = ""

py_include = jit_utils.get_py3_include_path()
LOG.v(f"py_include: {py_include}")
extension_suffix = jit_utils.get_py3_extension_suffix()
lib_suffix = extension_suffix.rsplit(".", 1)[0]
LOG.v(f"extension_suffix: {extension_suffix}")
so = ".so" if os.name != 'nt' else ".dll"


kernel_opt_flags = os.environ.get("kernel_flags", "") + opt_flags
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
        if s.startswith("-l") and ("cpython" in s or "lib" in s):
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
    if os.environ.get("debug", "0") == "1":
        opt_flags += " -O0 "
    else:
        opt_flags += " -O2 "
    kernel_opt_flags += " -Ofast "
lto_flags = ""
if os.environ.get("enable_lto") == "1":
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
cc_flags += f" -I\"{os.path.join(jittor_path, 'src')}\" "
cc_flags += py_include
check_cache_compile()
LOG.v(f"Get cache_compile: {jit_utils.cc}")

# check cuda
is_cuda = has_cuda = 0
check_cuda()
nvcc_flags = os.environ.get("nvcc_flags", "")
if has_cuda:
    nvcc_flags += cc_flags
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
        nvcc_flags += f" -I\"{os.path.join(jittor_path, 'extern/cuda/inc')}\" "
        if os.environ.get("cuda_debug", "0") == "1":
            nvcc_flags += " -G "
        return nvcc_flags
    nvcc_flags = convert_nvcc_flags(nvcc_flags)

extra_core_files = []
setup_fake_cuda_lib = False
# from .acl_compiler import check_acl
from .extern.acl import acl_compiler
jit_utils.add_backend(acl_compiler)
from .extern.rocm import rocm_compiler
jit_utils.add_backend(rocm_compiler)
from .extern.corex import corex_compiler
jit_utils.add_backend(corex_compiler)

for mod in jit_utils.backends:
    if mod.check():
        break

if not os.name == 'nt':
    is_cuda = os.path.basename(nvcc_path) == "nvcc"
else:
    is_cuda = os.path.basename(nvcc_path) == "nvcc.exe"

# build core
gen_jit_flags()
gen_jit_tests()
op_headers = glob.glob(jittor_path+"/src/ops/**/*op.h", recursive=True)
jit_src = gen_jit_op_maker(op_headers)
LOG.vvvv(jit_src)
with open(os.path.join(cache_path, "gen", "jit_op_maker.h"), 'w', encoding='utf8') as f:
    f.write(jit_src)
cc_flags += f' -I\"{cache_path}\" -L\"{cache_path}\" -L\"{jit_utils.cache_path}\" '
# gen pyjt
pyjt_gen_src = pyjt_compiler.compile(cache_path, jittor_path)

# initialize order:
# 1. registers
# 2. generate source
# 3. op_utils
# 4. other
files2 = pyjt_gen_src
ext_args = 'c[cu]' if has_cuda or has_rocm else 'cc'
files4 = glob.glob(jittor_path+"/src/**/*."+ext_args, recursive=True)
files4 = [ f[len(jittor_path)+1:] for f in files4 ]
# files4 = run_cmd('find -L src | grep '+grep_args, jittor_path).splitlines()
at_beginning = [
    "src/ops/op_utils.cc",
    "src/ops/op_register.cc",
    "src/init.cc",
    "src/event_queue.cc",
    "src/mem/allocator/sfrl_allocator.cc",
    "src/mem/allocator.cc",
    "src/misc/nano_string.cc",
]
at_last = [
    "src/profiler/profiler.cc",
    "src/executor.cc",
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

if platform.system() == 'Linux':
    libname = {"clang":"omp", "icc":"iomp5", "g++":"gomp"}[cc_type]
    libname = ctypes.util.find_library(libname)
    assert libname is not None, "openmp library not found"
    ctypes.CDLL(libname, os.RTLD_NOW | os.RTLD_GLOBAL)

if platform.machine()=='sw_64':
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

data_gz_path = os.path.join(jittor_path, "utils", "data.gz")
use_data_gz = os.path.isfile(data_gz_path)
if os.environ.get("use_data_gz", "1") == "0":
    use_data_gz = False
if use_data_gz:
    import gzip
    with gzip.open(data_gz_path, 'rb') as f:
        data = f.read()
        md5 = hashlib.md5(data).hexdigest()
    target_md5 = None
    data_gz_md5_path = os.path.join(cache_path, "data.md5")
    if os.path.isfile(data_gz_md5_path):
        with open(data_gz_md5_path, 'r') as f:
            target_md5 = f.read()
    data_o_path = os.path.join(cache_path, "data.o")
    if target_md5 != md5:
        data_s_path = os.path.join(cache_path, "data.cc")
        with open(data_s_path, "w") as f:
            f.write(data.decode("utf8"))
        dflags = (cc_flags+opt_flags)\
            .replace("-Wall", "") \
            .replace("-Werror", "") \
            .replace("-shared", "")
        vdp = os.path.join(jittor_path, "src", "utils", "vdp")
        run_cmd(fix_cl_flags(f"\"{cc_path}\" {dflags} -include \"{vdp}\" \"{data_s_path}\" -c -o \"{data_o_path}\""))
        os.remove(data_s_path)
        with open(data_gz_md5_path, 'w') as f:
            f.write(md5)
    files.append(data_o_path)
    files = [f for f in files if "__data__" not in f]
else:
    files = [f for f in files 
        if "__data__" not in f or "src" in f.split("__data__")[1]]

cc_flags += f" -l\"jit_utils_core{lib_suffix}\" "
compile(cc_path, cc_flags+opt_flags, files, 'jittor_core'+extension_suffix)
cc_flags += f" -l\"jittor_core{lib_suffix}\" "

# TODO: move to compile_extern.py
# compile_extern()

with jit_utils.import_scope(import_flags):
    import jittor_core as core

flags = core.Flags()

if has_cuda and is_cuda:
    nvcc_flags = " " + os.environ.get("nvcc_flags", "") + " "
    nvcc_flags += convert_nvcc_flags(cc_flags)
    nvcc_version = list(jit_utils.get_int_version(nvcc_path))
    max_arch = 89
    if nvcc_version < [11,]:
        max_arch = 75
    elif nvcc_version < [11,1]:
        max_arch = 80
    elif nvcc_version < [11,8]:
        max_arch = 86
    if len(flags.cuda_archs):
        min_arch = 30
        archs = []
        for arch in flags.cuda_archs:
            if arch<min_arch:
                LOG.w(f"CUDA arch({arch})<{min_arch} is not supported")
                continue
            if arch>max_arch:
                LOG.w(f"CUDA arch({arch})>{max_arch} will be backward-compatible")
                arch = max_arch
            archs.append(arch)
        flags.cuda_archs = archs
        nvcc_flags += f" -arch=compute_{min(archs)} "
        nvcc_flags += ''.join(map(lambda x:f' -code=sm_{x} ', archs))

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

core.set_lock_path(lock.lock_path)
