"""Native binding and operator source generation."""
import re

# The Var* members of an op struct. Their names and byte offsets enter the op
# registry, so a declaration this parser cannot read must fail during the build
# instead of leaving the relayed member pointing at an old value.
_VAR_MEMBER_DECL = re.compile(r"^[ \t]*Var[ \t]*\*(?P<names>[^;=(){}]*);", re.M)
# Also recognize noncanonical spellings to report them explicitly.
_VAR_MEMBER_LOOSE = re.compile(r"^[ \t]*(?:const[ \t]+)?(?:\w+::)*Var[ \t]*\*", re.M)


def gen_jit_tests():
    from . import compiler as _compiler_state
    all_src = _compiler_state.glob.glob(_compiler_state.core_root(_compiler_state.jittor_path)+"/**/*.cc", recursive=True)
    jit_declares = []
    re_def = _compiler_state.re.compile("JIT_TEST\\((.*?)\\)")
    names = set()
    test_defs = []

    for src_name in all_src:
        with open(src_name, 'rb') as f:
            src = f.read().decode('utf8')
        defs = re_def.findall(src)
        for name in defs:
            _compiler_state.LOG.vv(f"Find test {name} from {src_name}")
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
    _compiler_state.LOG.vvvv(jit_src)
    with open(_compiler_state.os.path.join(_compiler_state.cache_path, "gen", "jit_tests.h"), 'w', encoding='utf8') as f:
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
    from . import compiler as _compiler_state
    all_src = _compiler_state.glob.glob(_compiler_state.core_root(_compiler_state.jittor_path)+"/**/*.cc", recursive=True)
    jit_declares = []
    re_def = _compiler_state.re.compile("DEFINE_(RUNTIME_)?FLAG(_WITH_SETTER)?\\((.*?)\\);", _compiler_state.re.DOTALL)

    flags_defs = []
    visit = {}

    for src_name in all_src:
        with open(src_name, 'rb') as f:
            src = f.read().decode("utf8")
        defs = re_def.findall(_compiler_state.strip_cxx_comments(src))
        for runtime, _, args in defs:
            args = args.split(",")
            type = args[0].strip()
            name = args[1].strip()
            if not _compiler_state.has_cuda and "cuda" in name and name not in ("use_cuda", "cuda_kernel_math"):
                if name != "use_cuda_host_allocator":
                    continue
            default = args[2].strip()
            doc = ",".join(args[3:])
            doc = eval(f"({doc})")
            _compiler_state.LOG.vv(f"Find define {name} from {src_name}")
            if name in visit:
                continue
            visit[name] = 1
            category = _compiler_state.flag_category(name)
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
    _compiler_state.LOG.vvvv(jit_src)
    with open(_compiler_state.os.path.join(_compiler_state.cache_path, "gen", "jit_flags.h"), 'w', encoding='utf8') as f:
        f.write(jit_src)


def parse_var_members(src, header="<src>"):
    """Names of the Var* members declared in an op header, in declaration order."""
    from . import compiler as _compiler_state
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
    for m in _compiler_state._VAR_MEMBER_DECL.finditer(src):
        text = m.group("names")
        for c in "*,":
            text = text.replace(c, " ")
        names.extend(text.split())
    return names


def gen_jit_op_maker(op_headers, export=False, extra_flags="", backend=None):
    from . import compiler as _compiler_state
    def compose_backend_source(operator, segments):
        import tempfile

        parts = []
        for source_path in segments:
            with open(source_path, encoding="utf-8") as source_file:
                text = source_file.read()
            parts.append('#line 1 ' + _compiler_state.json.dumps(source_path) + '\n' + text + '\n')
        content = "".join(parts)
        directory = _compiler_state.os.path.join(_compiler_state.cache_path, "backend_sources")
        _compiler_state.os.makedirs(directory, exist_ok=True)
        destination = _compiler_state.os.path.join(directory, operator + "_cuda.cc")
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
            _compiler_state.os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                _compiler_state.os.unlink(temporary)
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
        name = _compiler_state.os.path.basename(header)
        name = _compiler_state.os.path.splitext(name)[0]
        # xxx_xxx
        assert name.endswith("_op")
        func_name = name[:-3]
        # XxxXxxOp
        name2 = map(lambda s:s[:1].upper() + s[1:], name.split('_'))
        name2 = "".join(name2)
        with open(header, encoding='utf8') as f:
            src = f.read()
        # XxxXxxOp(args)
        res = _compiler_state.re.findall(pybind_attrs_reg + '[^~]('+name2+"\\([^\\n]*\\))", src, _compiler_state.re.S)
        assert len(res) >= 1, "Wrong op args in " + header
        # registe op
        cc_name = header[:-2] + ".cc"
        constructors = []
        for i in range(len(res)):
            name = 'make_'+func_name+'_'*i
            constructors.append(f"op_constructor_entry(&{name})")
        constructors = ",".join(constructors)
        var_member = _compiler_state.parse_var_members(src, header)
        _compiler_state.LOG.vv("var_member "+str(var_member))
        var_member_src = [ f"VAR_MEMBER_NAME_AND_OFFSET({name}, {name2})" for name in var_member ]
        var_member_src = ",".join(var_member_src)
        mask_arg = f", {backend_mask}" if backend_mask is not None else ""
        # Optional backend sources are composed into the first definition, not
        # registered as a replacement (which would invalidate persistent JIT keys).
        builtin_directories = (_compiler_state.os.path.join(_compiler_state.core_root(_compiler_state.jittor_path), "ops"),
                               _compiler_state.os.path.join(_compiler_state.core_root(_compiler_state.jittor_path), "ops", "composite"))
        if _compiler_state.os.path.realpath(_compiler_state.os.path.dirname(header)) in tuple(map(_compiler_state.os.path.realpath, builtin_directories)):
            kernel_directory = _compiler_state.os.path.join(_compiler_state.backend_root(_compiler_state.jittor_path, "cuda"), "kernels", "core")
            accelerator_source = _compiler_state.os.path.join(kernel_directory, func_name + "_op.cc")
            prefix_source = _compiler_state.os.path.join(kernel_directory, func_name + "_prefix.cc")
            if _compiler_state.os.path.isfile(prefix_source):
                accelerator_source = compose_backend_source(func_name, [prefix_source, cc_name])
            if _compiler_state.os.path.isfile(accelerator_source):
                mask = backend_mask if backend_mask is not None else name2 + "::backend_mask"
                accelerator_flags = ' -I"' + kernel_directory + '" '
                mask_arg = ", " + mask + ", " + _compiler_state.json.dumps(accelerator_source) + ", " + _compiler_state.json.dumps(accelerator_flags)
        initer.append(f'\n        register_op_definition<{name2}>({{ "{func_name}", R"({cc_name})", extra_flags, {{{constructors}}}, {{{var_member_src}}} }}{mask_arg});')
        for hid, h_def in enumerate(res):
            h_def = list(h_def)
            # // @attrs(...)
            attrs = {}
            if h_def[4] != "":
                attrs = _compiler_state.pyjt_compiler.parse_attrs(h_def[5])
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
            _compiler_state.LOG.vvvv(f"Find op: {name2} args: {new_args}")
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
                with open(_compiler_state.os.path.join(_compiler_state.core_root(_compiler_state.jittor_path), f"ops/{func_name}_op.cc"), encoding="utf-8") as f:
                    src = f.read()
                src = src.split(f"unordered_set<string> {func_name}_ops = ""{")[1].split("};")[0]
                match_result = _compiler_state.re.findall(pybind_reg + "\"([a-z_A-Z0-9]*)\"", src, _compiler_state.re.S)
                # remove /* doc_string */ pattern
                res2 = [ (_[3], _[4]) for _ in match_result ]
                _compiler_state.LOG.vvvv(f"All supported {func_name} ops: {res2}")
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
