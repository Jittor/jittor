# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import re
import os
from jittor_utils import LOG, run_cmd, simple_timer
import json
from collections import OrderedDict
import glob

def parse_attrs(s):
    '''parse @attrs(..., x=y) syntax'''
    attrs = {}
    if s is None: return attrs
    for a in s.split(','):
        a = a.strip()
        if len(a)==0: continue
        if '=' in a:
            k, v = a.split('=')
            attrs[k] = v
        else:
            attrs[a] = 1
    return attrs


pytype_map = {
    "const char*": ["PyUnicode_AsUTF8", "PyUnicode_FromString", "PyUnicode_CheckExact"],
    "int": ["PyLong_AsLong", "PyLong_FromLong", "PyLong_CheckExact"],
    "int64": ["PyLong_AsLongLong", "PyLong_FromLongLong", "PyLong_CheckExact"],
    "uint": ["PyLong_AsUnsignedLong", "PyLong_FromUnsignedLong", "PyLong_CheckExact"],
    "uint8": ["PyLong_AsUnsignedLong", "PyLong_FromUnsignedLong", "PyLong_CheckExact"],
    "uint16": ["PyLong_AsUnsignedLong", "PyLong_FromUnsignedLong", "PyLong_CheckExact"],
    "uint64": ["PyLong_AsUnsignedLongLong", "PyLong_FromUnsignedLongLong", "PyLong_CheckExact"],
    "void": ["...", "GET_PY_NONE", "..."],
    "PyObject*": ["","",""],
}
def get_pytype_map(T, i):
    assert T != ""
    if T in pytype_map:
        return pytype_map[T][i]
    return ["from_py_object", "to_py_object", "is_type"][i]+"<"+T+">"

binary_number_slots = {
    "__add__": "nb_add",
    "__sub__": "nb_subtract",
    "__mul__": "nb_multiply",
    "__mod__": "nb_remainder",
    "__divmod__": "nb_divmod",
    "__pow__": "nb_power",
    "__lshift__": "nb_lshift",
    "__rshift__": "nb_rshift",
    "__and__": "nb_and",
    "__xor__": "nb_xor",
    "__or__": "nb_or",
    "__floordiv__": "nb_floor_divide",
    "__truediv__": "nb_true_divide",
    "__matmul__": "nb_matrix_multiply",
}

for k,v in list(binary_number_slots.items()):
    # __add__: nb_add ----> __iadd: nb_inplace_add
    binary_number_slots["__i"+k[2:]] = "nb_inplace"+v[2:]

unary_number_slots = {
    "__neg__": "nb_negative",
    "__abs__": "nb_absolute",
}

def split_args(s):
    # split args xxx,xxx, xx<xx,xx>, xx
    s = s.strip()
    if s=="": return []
    prev = -1
    presum = 0
    args = []
    for i in range(len(s)):
        if s[i]=='<':
            presum += 1
        elif s[i]=='>':
            presum -= 1
        if presum==0 and s[i]==',':
            args.append(s[prev+1:i])
            prev = i
    args.append(s[prev+1:])
    return args

def get_def_code(df, scope_name, pyname, self_as_arg0=False):
    is_fast_call = not pyname.startswith("__")
    no_need_convert = pyname == "__getitem__"
    args = df["args"]
    # n==1 && PyXXX__CheckExact(args[0]) && ...
    max_args = len(args)
    min_args = max_args
    for tid, a in enumerate(args):
        if a[2] != "":
            min_args = tid
            break
    arg_names = [ f"args[{i}]" for i in range(len(args))]
    if self_as_arg0:
        max_args -= 1
        min_args -= 1
        arg_names = ["self"] + arg_names[:-1]
    kw_args_id = []
    for aid, arg in enumerate(args):
        if "VarHolder*" != arg[0] and is_fast_call:
            kw_args_id.append(aid)
    func_quick_check_runable = ""
    func_quick_check_size = f"n<={max_args} && n>={min_args}"
    if len(kw_args_id):
        func_quick_check_size = f"n+(kw?Py_SIZE(kw):0)<={max_args} && n+(kw?Py_SIZE(kw):0)>={min_args}"
    fill_with_default = ""
    func_args_convert = ""
    func_call = df["func_name"]+"("
    pytypes = [ get_pytype_map(a[0],0) for a in args ]
    holder_dec_array = []
    holder_set_array = []
    for tid, tpc in enumerate(pytypes):
        check = get_pytype_map(args[tid][0],2)
        default_arg = args[tid][2]
        jtp = args[tid][0]
        holder_dec = ""
        holder_set = ""
        if jtp == "VarHolder*":
            holder_dec = f"unique_ptr<VarHolder> arg{tid}_holder"
            holder_set = f", arg{tid}_holder"
        if jtp == "VarSlices":
            holder_dec = f"vector<unique_ptr<VarHolder>> arg{tid}_holder"
            holder_set = f", arg{tid}_holder"
        holder_dec_array.append(holder_dec)
        holder_set_array.append(holder_set)
        if len(default_arg):
            func_args_convert += f"""
                        {holder_dec};
                        {jtp} arg{tid};
                        if (n>{tid-self_as_arg0}) {{
                            CHECK(({check}({arg_names[tid]})));
                            arg{tid} = {tpc}({arg_names[tid]}{holder_set});
                            arg_filled |= 1ull << {tid};
                        }}
            """
            fill_with_default += f"""
                        if (!(arg_filled & (1ull<<{tid}))) {{
                            arg{tid} = {default_arg};
                        }}
            """
        else:
            func_quick_check_runable += f" && {check}({arg_names[tid]})"
            func_args_convert += f"""
                        {holder_dec};
                        {jtp} arg{tid} = {tpc}({arg_names[tid]}{holder_set});
            """
        if tid: func_call += ","
        if args[tid][3].endswith("&&"):
            func_call += f"move(arg{tid})"
        else:
            func_call += f"arg{tid}"
    if pyname == "__richcmp__":
        for rname in [ "__lt__", "__le__", "__gt__",
            "__ge__", "__eq__", "__ne__"]:
            if rname in df["attrs"]:
                func_quick_check_runable += " && op==Py_"+rname[2:-2].upper()
    # fill args with keyword arguments
    fill_with_kw = ""
    if is_fast_call and len(kw_args_id):
        fill_with_kw = f"""
                        if (kw) {{
                            auto kw_n = Py_SIZE(kw);
                            for (int i=0; i<kw_n; i++) {{
                                auto ko = PyTuple_GET_ITEM(kw, i);
                                auto vo = args[i+n];
                                auto ks = PyUnicode_AsUTF8(ko);
                                uint khash = hash(ks);
                                {"".join([
                                f'''
                                if (khash == {get_hash(args[aid][1])}u) {{
                                    // hash match {args[aid][1]}
                                    CHECK(({get_pytype_map(args[aid][0],2)}(vo)));
                                    arg{aid} = {pytypes[aid]}(vo{holder_set_array[aid]});
                                    arg_filled |= 1ull << {aid};
                                    continue;
                                }}
                                '''
                                for aid in kw_args_id
                                ])}
                                LOGf << "Not a valid keyword:" << ks;
                            }}
                        }}
        """

    if len(args):
        func_args_convert += """
                        CHECK(!PyErr_Occurred());
        """
    func_call += ")"
    if df["is_property"]:
        if pyname.startswith("__get__"):
            func_call = df["func_name"]
        else:
            assert pyname.startswith("__set__"), pyname
            func_call = df["func_name"] + "= arg0"
    has_return = df["return_t"]!="void" and df["return_t"]!=""
    # add XXX::xxx or XXX->xxx if is class def
    if df["is_scope_def"]:
        if df["is_static"]:
            func_call = f"{scope_name}::" + func_call
        else:
            func_call = f"(GET_RAW_PTR({scope_name},self))->" + func_call
    if pyname == "__init__":
        # XXX->xxx(...) ---> new XXX xxx(...)
        assert "->" in func_call, func_call
        func_call = "new " + func_call.replace("->", " ")
    if no_need_convert:
        func_quick_check_runable = ""
        func_args_convert = ""
        fill_with_kw = fill_with_default = ""
    return (
        func_quick_check_size + func_quick_check_runable, 
        func_args_convert, 
        fill_with_kw+fill_with_default, 
        func_call, 
        has_return
    )

hash_to_key_map = {}

def get_hash(s):
    mask = (1<<32)-1
    v=0
    mul = 1
    for c in s:
        v += mul * ord(c)
        mul *= 55
        v &= mask
        mul &= mask
    if v in hash_to_key_map:
        assert hash_to_key_map[v] == s, \
            f"hash conflict {hash_to_key_map[v]} {s} {hash_to_key_map}"
    hash_to_key_map[v] = s
    return v


reg = re.compile(
    '(/\\*(.*?)\\*/\\s*)?(//\\s*@pyjt\\(([^\\n]*)\\)\\s*)'
    # ^^^^^^^^^^^^^^^^^          ^^^^    ^^^^
    # doc string $1              pyjt    args $3
    +
    '(//\\s*@attrs\\(([^\\n]*)\\)\\s*)?'
    #        ^^^^^   ^^^^^^^
    #        attrs    args $5
, re.DOTALL)

def generate_error_code_from_func_header(func_head, target_scope_name, name, dfs, basename, h, class_info):
    # func_head is a string like:
    # (PyObject* self, PyObject** args, int64 n, PyObject* kw) -> PyObject*
    lib_name = os.path.basename(h).split("_")[0]
    # TODO: fix/add var help
    if target_scope_name == "Var": target_scope_name = None
    if target_scope_name:
        if target_scope_name == "flags":
            help_name = "flags"
        else:
            help_name = ""+target_scope_name+'.'+name
    else:
        help_name = name
    if lib_name in ["mpi", "nccl", "cudnn", "curand" "cufft", "cublas", "mkl"]:
        help_name = lib_name+'.'+help_name
    help_cmd = f"help(jt.{help_name})"

    LOG.vvv("gen err from func_head", func_head)
    args = func_head[1:].split(")")[0].split(",")
    error_code = f" << \"Wrong inputs arguments, Please refer to examples({help_cmd}).\""
    error_code += r' << "\n\nTypes of your inputs are:\n"'
    for arg in args:
        arg = arg.strip()
        if arg.startswith("PyObject* "):
            t, n = arg.split(' ')
            if n == "args" or n == "_args":
                error_code += f" << PyTupleArgPrinter{{{n}, \"args\"}} "
            elif n == "kw":
                error_code += f" << PyKwArgPrinter{{{n}}} "
            else:
                error_code += f" << PyArgPrinter{{{n}, \"{n}\"}} "
        elif arg.startswith("PyObject** "):
            t, n = arg.split(' ')
            error_code += f" << PyFastCallArgPrinter{{{n}, n, kw}} "
            break
        else:
            LOG.vvv("Unhandled arg", arg)
    LOG.vvv("gen err from func_head", func_head, " -> ", error_code)
    return error_code

def compile_src(src, h, basename):
    res = list(reg.finditer(src, re.S))
    if len(res)==0: return
    class_ranges = None
    class_name = None
    class_info = None
    submodule_name = None
    submodule_ranges = None
    submodule_info = None
    defs = []
    LOG.vv(("find in", h))
    for x in res:
        LOG.vvv((x, x.groups()))
        g = x.groups()
        doc = g[1]
        pyjt = g[3]
        attrs = g[5]
        esplit = lambda x: [] if x==None else \
            [ a.strip() for a in x.split(",") if len(a.strip()) ]
        attrs = parse_attrs(attrs)
        pynames = esplit(pyjt)
        end = x.end()
        def find_bc(i):
            while src[i] not in "({;":
                i += 1
            j = i+1
            if src[i]==';':
                return i, j
            presum = 1
            while True:
                if src[j] in "({[":
                    presum += 1
                elif src[j] in ")}]":
                    presum -= 1
                    if presum==0:
                        s = src[i]+src[j]
                        assert s in ("()","{}","()"), "braces not match "+s
                        return i, j
                j += 1
        # // @pyjt(DType)
        # struct DType {
        #              ^ --> a
        #     .....
        # } <--- b
        # or
        # // @pyjt(hash)
        # inline uint hash(const char* input)
        #                 ^ --> a           ^ --> b
        a, b = find_bc(end)
        is_property = 0
        if src[a] == ';':
            # This case
            # class XXX {
            #     // @pyjt(property)
            #     T property;
            # }
            is_property = 1
        if src[a] == '{':
            assert len(pynames)==1
            if "submodule" in attrs:
                assert submodule_ranges==None
                submodule_ranges = (a, b)
                submodule_name = src[end:a-1].strip().split()[-1]
                submodule_info = {
                    "pynames": pynames,
                    "attrs": attrs
                }
                continue
            assert class_ranges==None
            class_ranges = (a, b)
            class_name = src[end:a-1].strip().split()[-1]
            class_info = {
                "pynames": pynames,
                "attrs": attrs
            }
            continue
        is_scope_def = False
        is_static = False
        scope_name = ""
        if class_ranges != None:
            if class_ranges[0] < a and a < class_ranges[1]:
                is_scope_def = True
                scope_name = class_name
        if submodule_ranges != None:
            if submodule_ranges[0] < a and a < submodule_ranges[1]:
                is_scope_def = True
                scope_name = submodule_name
                is_static = True
        dec = src[end:b+1].strip()
        arr = src[end:a].strip().split()
        func_name = arr[-1]

        is_constructor = False
        if is_scope_def and func_name==class_name:
            is_constructor = True

        args = []
        for arg in split_args(src[a+1:b]):
            if arg=="": continue
            default = ""
            if "=" in arg:
                arg, default = arg.split('=')
                default = default
            arg = arg.strip()
            name = arg.split(' ')[-1]
            tp = arg[:-len(name)]
            tp = tp.strip()
            prev_tp = tp
            # const string& ----> string
            if tp.startswith("const") and tp.endswith("&"):
                tp = tp[5:-1].strip()
            # T&& -> T
            if tp.endswith("&&"):
                tp = tp[:-2].strip()
            # ArrayArgs& -> ArrayArgs
            if tp.endswith("&"):
                tp = tp[:-1].strip()
            args.append((tp, name.strip(), default.strip(), prev_tp))
        return_t = ""
        for a in arr[:-1]:
            if a in ["", "inline", "constexpr"]: continue
            if a == "static":
                is_static = True
                continue
            if return_t != "": return_t += " "
            return_t += a

        if is_scope_def and class_info and "submodule" in class_info["attrs"]:
            is_static = True

        for pid, pyname in enumerate(pynames):
            for rname in [ "__lt__", "__le__", "__gt__",
                "__ge__", "__eq__", "__ne__"]:
                if pyname.endswith(rname):
                    attrs[rname] = 1
                    pynames[pid] = pyname.replace(rname, "__richcmp__")

        def_info = {
            "is_scope_def": is_scope_def,
            "is_constructor": is_constructor,
            "is_static": is_static,
            "is_property": is_property,
            "func_name": func_name,
            "args": args, # [(type,name,defaut), ...]
            "return_t": return_t, # return type
            "dec": dec, # full string of xxx(A a, B b)
            "pynames": pynames, # names in @pyjt(...)
            "attrs": attrs, # attrs in @attrs(...)
            "doc": doc,
            "scope_name": scope_name,
        }
        if is_property:
            # This case
            # class XXX {
            #     // @pyjt(property)
            #     T property;
            # }
            assert is_scope_def and not is_static
            def_info["is_property"] = 1
            def_info["pynames"] = ["__get__"+n for n in pynames]
            assert return_t != "void"
            defs.append(dict(def_info))
            def_info["pynames"] = ["__set__"+n for n in pynames]
            assert len(args) == 0
            def_info["args"] = [(def_info["return_t"], func_name, "", "")]
            def_info["return_t"] = "void"
            defs.append(dict(def_info))
            continue
        else:
            defs.append(def_info)
        LOG.vvv(lambda: json.dumps(def_info, indent=4))
    # deal with defs
    if len(defs) == 0: return
    # include_name = h[4:] # remove "src/" prefix
    include_name = h
    code = []
    class_defs_code = []
    class_getsets_code = []
    class_gets = OrderedDict()
    class_sets = OrderedDict()
    class_slots_code = []
    submodule_defs_code = []
    def_targets = OrderedDict()
    for df in defs:
        for name in df["pynames"]:
            if df["is_scope_def"] and '.' not in name:
                if df["scope_name"] == class_name:
                    name = class_info["pynames"][0] + '.' + name
                else:
                    name = submodule_info["pynames"][0] + '.' + name
            if name not in def_targets:
                def_targets[name] = []
            def_targets[name].append(df)
    for name in def_targets:
        dfs = def_targets[name]
        target_scope_name = None
        LOG.vv(name)
        if "." in name:
            target_scope_name, name = name.split(".")
        # array for each df:
        arr_func_quick_check_runable = []
        arr_func_args_convert = []
        arr_fill_with_default = []
        arr_func_call = []
        arr_has_return = []
        self_as_arg0 = False
        for df in dfs:
            self_as_arg0 = class_info and \
                target_scope_name == class_info["pynames"][0] and \
                df["scope_name"] == submodule_name \
                and not name.startswith("__")
            res = get_def_code(df, df["scope_name"], name, bool(self_as_arg0))
            arr_func_quick_check_runable.append(res[0])
            arr_func_args_convert.append(res[1])
            arr_fill_with_default.append(res[2])
            arr_func_call.append(res[3])
            arr_has_return.append(res[4])
            
        slot_name = None
        func_cast = ""
        func_fill = ""
        before_return = ""
        if name == "__init__":
            slot_name = "tp_init"
            func_head = "(PyObject* self, PyObject* _args, PyObject* kw) -> int"
            func_fill = """
                int64 n = Py_SIZE(_args);
                auto args = (PyObject**)&PyTuple_GET_ITEM(_args, 0);
                (void)n, (void)args;
                // TODO: support kw
                CHECK(kw==0);
            """

        elif name == "__repr__":
            slot_name = "tp_repr"
            func_head = "(PyObject* self) -> PyObject*"
            func_fill = "int64 n = 0; (void)n;"

        elif name.startswith("__get__"):
            slot_name = "tp_gets"
            name = name[len("__get__"):]
            func_head = "(PyObject* self, void*) -> PyObject*"
            func_fill = "int64 n = 0; (void)n;"

        elif name.startswith("__set__"):
            slot_name = "tp_sets"
            name = name[len("__set__"):]
            func_head = "(PyObject* self, PyObject* arg, void*) -> int"
            func_fill = """
                int64 n=1;
                PyObject** args = &arg;
                (void)n, (void)args;
            """

        elif name == "__call__":
            slot_name = "tp_call"
            func_head = "(PyObject* self, PyObject* _args, PyObject* kw) -> PyObject*"
            func_fill = """
                int64 n = Py_SIZE(_args);
                auto args = (PyObject**)&PyTuple_GET_ITEM(_args, 0);
                (void)n, (void)args;
                // TODO: support kw
                CHECK(kw==0);
            """

        elif name == "__dealloc__":
            slot_name = "tp_dealloc"
            func_head = "(PyObject* self) -> void"
            func_fill = "int64 n = 0"
            before_return = "Py_TYPE(self)->tp_free((PyObject *) self);"
        
        elif name in binary_number_slots:
            slot_name = "tp_as_number->"+binary_number_slots[name]
            func_head = "(PyObject* self, PyObject* b) -> PyObject*"
            if name.endswith("pow__"):
                func_head = "(PyObject* self, PyObject* b, PyObject*) -> PyObject*"
            func_fill = """
                int64 n = 2;
                PyObject* args[] = {self, b};
                (void)n, (void)args;
            """
        
        elif name in unary_number_slots:
            slot_name = "tp_as_number->"+unary_number_slots[name]
            func_head = "(PyObject* self) -> PyObject*"
            func_fill = """
                int64 n = 1;
                PyObject* args[] = {self};
                (void)n, (void)args;
            """
        
        elif name == "__richcmp__":
            slot_name = "tp_richcompare"
            func_head = "(PyObject* self, PyObject* b, int op) -> PyObject*"
            func_fill = """
                int64 n = 2;
                PyObject* args[] = {self, b};
                (void)n, (void)args;
            """

        elif name == "__len__":
            slot_name = "tp_as_sequence->sq_length"
            func_head = "(PyObject* self) -> Py_ssize_t"
            func_fill = """
                int64 n = 0;  
                (void)n;
            """

        elif name == "__map_len__":
            slot_name = "tp_as_mapping->mp_length"
            func_head = "(PyObject* self) -> Py_ssize_t"
            func_fill = """
                int64 n = 0;  
                (void)n;
            """

        elif name == "__getitem__":
            slot_name = "tp_as_sequence->sq_item"
            func_head = "(PyObject* self, Py_ssize_t arg0) -> PyObject*"
            func_fill = f"""
                int64 n = 1;
                (void)n;
                if (arg0 >= GET_RAW_PTR({dfs[0]["scope_name"]},self)->size()) {{
                    PyErr_SetString(PyExc_IndexError, "");
                    return (PyObject*)nullptr;
                }}
            """

        elif name == "__map_getitem__":
            slot_name = "tp_as_mapping->mp_subscript"
            func_head = "(PyObject* self, PyObject* arg0) -> PyObject*"
            func_fill = f"""
                int64 n = 1;
                PyObject* args[] = {{arg0}};
                (void)n;
            """

        elif name.startswith("__"):
            LOG.f(f"Not support slot {name}")
            continue

        else:
            func_head = "(PyObject* self, PyObject** args, int64 n, PyObject* kw) -> PyObject*"
            func_cast = f"(PyCFunction)(PyObject* (*)(PyObject*,PyObject**,int64,PyObject*))"
            # if not return, return py_none
            arr_has_return = [ True for _ in arr_has_return ]

        arr_func_return = []
        doc_all = ""
        decs = "The function declarations are:\n"
        for did, has_return in enumerate(arr_has_return):
            df = dfs[did]
            func_call = arr_func_call[did]
            if df["doc"] and not (did > 0 and df["doc"] == dfs[did - 1]["doc"]):
                doc_all += "Document:\n"
                doc_all += df["doc"]+'\n'
            doc_all += "Declaration:\n"
            doc_all += df["dec"]+'\n\n'
            decs += " " + df["dec"]+'\n'
            if has_return:
                assert "-> int" not in func_head
                if "-> PyObject*" in func_head:
                    if "return_self" in df["attrs"]:
                        arr_func_return.append(
                            f"return (({func_call}), Py_INCREF(self), self)")
                    else:
                        arr_func_return.append(
                            f"return {get_pytype_map(df['return_t'],1)}(({func_call}))")
                    func_return_failed = "return nullptr"
                else:
                    arr_func_return.append(
                        f"return ({func_call});")
                    func_return_failed = "return -1"
            else:
                if "-> int" in func_head:
                    arr_func_return.append(f"return ({func_call},0)")
                    func_return_failed = "return -1"
                else:
                    assert "-> void" in func_head, func_head
                    arr_func_return.append(f"{func_call};{before_return}return")
                    func_return_failed = "return"
        # generate error msg when not a valid call
        error_log_code = generate_error_code_from_func_header(func_head, target_scope_name, name, dfs, basename ,h, class_info)
        func = f"""
        {func_cast}[]{func_head} {{
            try {{
                {func_fill};
                uint64 arg_filled=0;
                (void)arg_filled;
                {"".join([f'''
                if ({arr_func_quick_check_runable[did]}) {{
                    {arr_func_args_convert[did]};
                    {arr_fill_with_default[did]};
                    {arr_func_return[did]};
                }}
                '''
                for did in range(len(arr_func_return))
                ])}
                LOGf << "Not a valid call.";
            }} catch (const std::exception& e) {{
                if (!PyErr_Occurred()) {{
                    std::stringstream ss;
                    if (check_async_executor_error(e, ss)) {{
                        PyErr_Format(PyExc_RuntimeError, 
                            "%s",
                            ss.str().c_str()
                        );
                    }} else {{
                        ss {error_log_code};
                        PyErr_Format(PyExc_RuntimeError, 
                            "%s\\n%s\\nFailed reason:%s",
                            ss.str().c_str(),
                            R""({decs})"",
                            e.what()
                        );
                    }}
                }}
            }}
            {func_return_failed};
        }}
        """

        if slot_name:
            if slot_name=="tp_gets":
                class_gets[name] = {
                    "func": func,
                    "doc": doc_all
                }
                continue
            if slot_name=="tp_sets":
                class_sets[name] = {
                    "func": func,
                    "doc": ""
                }
                continue
            class_slots_code.append(f"""
            tp.{slot_name} = {func};
            """)
            continue
        need_static = ""
        if df["is_scope_def"] and df["is_static"] and \
            df["scope_name"] == class_name and \
            "submodule" not in class_info["attrs"]:
            need_static = " | METH_STATIC"
        func = (f"""
        {{ R""({name})"",
        {func},
        METH_FASTCALL | METH_KEYWORDS{need_static},
        R""({doc_all})""
        }}""")
        if df["is_scope_def"]:
            if df["scope_name"] == class_name or \
                (class_info and \
                    target_scope_name == class_info["pynames"][0]):
                class_defs_code.append(func)
            else:
                submodule_defs_code.append(func)
        else:
            code.append(func)
    prop_names = list(set(class_gets.keys()).union(class_sets.keys()))
    prop_names = sorted(prop_names)
    for prop_name in prop_names:
        get_func = "NULL"
        set_func = "NULL"
        doc = ""
        if prop_name in class_gets:
            get_func = class_gets[prop_name]["func"]
            if class_gets[prop_name]["doc"]:
                doc += class_gets[prop_name]["doc"]
        if prop_name in class_sets:
            set_func = class_sets[prop_name]["func"]
            if class_sets[prop_name]["doc"]:
                doc += class_sets[prop_name]["doc"]
        class_getsets_code.append(f"""
            {{"{prop_name}", {get_func}, {set_func}, R""({doc})""}}
        """)
    code.append("{0,0,0,0}")
    class_defs_code.append("{0,0,0,0}")
    class_getsets_code.append("{0,0,0,0}")
    submodule_defs_code.append("{0,0,0,0}")
    core_name = "jittor_core"
    if class_info and "attrs" in class_info and "core_name" in class_info["attrs"]:
        core_name = class_info["attrs"]["core_name"]
    if submodule_info and "attrs" in submodule_info and "core_name" in submodule_info["attrs"]:
        core_name = submodule_info["attrs"]["core_name"]
    has_map = class_name in ["VarHolder", "NanoVector"]
    has_seq = class_name == "NanoVector"
    # add extra include to avoid compile error
    src_code = ""
    if include_name.endswith("var_slices.h"):
        src_code += '#include "var_holder.h"\n' 
    src_code += f"""
    #include "utils/seh.h"
    #include "pyjt/py_converter.h"
    #include "pyjt/py_arg_printer.h"
    #include "common.h"
    #include "{include_name}"

    namespace jittor {{

    {
    "" if class_name is None else
    f"PyHeapTypeObject Pyjt{class_name};" if "heaptype" in class_info["attrs"] else
    f"PyTypeObject Pyjt{class_name};"
    }
    
    void pyjt_def_{basename}(PyObject* m) {{
        static PyMethodDef defs[] = {{
            {",".join(code)}
        }};
        ASSERT(PyModule_AddFunctions(m, defs)==0);
        {
        f'''
        static PyMethodDef class_defs[] = {{
            {",".join(class_defs_code)}
        }};
        static PyGetSetDef class_getsets[] = {{
            {",".join(class_getsets_code)}
        }};

        static PyNumberMethods number_methods = {{0}};
        {f"auto& htp =Pyjt{class_name}; auto& tp = htp.ht_type;"
        if "heaptype" in class_info["attrs"] else
        f"auto& tp = Pyjt{class_name};"}
        tp.tp_as_number = &number_methods;

        {f"static PyMappingMethods class_map_defs = {{0}};" if has_map else ""}
        {f"tp.tp_as_mapping = &class_map_defs;" if has_map else ""}

        {f"static PySequenceMethods class_seq_defs = {{0}};" if has_seq else ""}
        {f"tp.tp_as_sequence = &class_seq_defs;" if has_seq else ""}
        
        tp.tp_name = "{core_name}.{class_info["pynames"][0]}";
        tp.tp_basicsize = GET_OBJ_SIZE({class_name});
        tp.tp_new = PyType_GenericNew;
        tp.tp_flags = Py_TPFLAGS_DEFAULT;
        {"tp.tp_flags |= Py_TPFLAGS_HEAPTYPE; htp.ht_name = htp.ht_qualname = to_py_object<string>(tp.tp_name);"
        if "heaptype" in class_info["attrs"] else ""}
        tp.tp_methods = &class_defs[0];
        tp.tp_getset = &class_getsets[0];
        {"".join(class_slots_code)};
        ASSERT(0==PyType_Ready(&tp)) << (PyErr_Print(), 0);
        Py_INCREF(&tp);
        ASSERT(0==PyModule_AddObject(m, "{class_info["pynames"][0]}", (PyObject*)&tp));
        ''' if class_name is not None else ""
        }
        {f'''

        // sub module def
        static PyMethodDef submodule_defs[] = {{
            {",".join(submodule_defs_code)}
        }};
        auto sub = PyImport_AddModule("{core_name}.{submodule_info["pynames"][0]}");
        ASSERT(PyModule_AddFunctions(sub, submodule_defs)==0);
        ASSERT(sub);
        ASSERT(0==PyModule_AddObject(m, "{submodule_info["pynames"][0]}", sub));
        ''' if submodule_name is not None else ""
        }

    }}

    }}
    """
    return src_code

def compile_single(head_file_name, src_file_name, src=None):
    basename = os.path.basename(head_file_name).split(".")[0]
    if src==None:
        with open(head_file_name, 'r', encoding='utf8') as f:
            src = f.read()
    code = compile_src(src, head_file_name, basename)
    if not code: return False
    LOG.vvv("write to", src_file_name)
    LOG.vvvv(code)
    with open(src_file_name, 'w', encoding='utf8') as f:
        f.write(code)
    return True

def compile(cache_path, jittor_path):
    headers1 = glob.glob(jittor_path+"/src/**/*.h", recursive=True)
    headers2 = glob.glob(cache_path+"/gen/**/*.h", recursive=True)
    headers = headers1 + headers2
    basenames = []
    pyjt_names = []
    for h in headers:
        with open(h, 'r', encoding='utf8') as f:
            src = f.read()

        bh = os.path.basename(h)
        # jit_op_maker.h merge compile with var_holder.h
        if bh == "var_holder.h": continue
        if bh == "jit_op_maker.h":
            with open(os.path.join(jittor_path, "src", "var_holder.h"), "r", encoding='utf8') as f:
                src = f.read() + src
        basename = bh.split(".")[0]
        fname = "pyjt_"+basename+".cc"
        fname = os.path.join(cache_path, "gen", fname)
        check = compile_single(h, fname, src)

        if not check: continue

        basenames.append(basename)
        pyjt_names.append(fname)
    
    code = f"""
    #include "pyjt/numpy.h"
    #include "pyjt/py_converter.h"
    #include "common.h"

    namespace jittor {{

    { " ".join([f"extern void pyjt_def_{n}(PyObject* m);" for n in basenames])}

    void pyjt_def_all(PyObject* m) {{
        numpy_init();
        { " ".join([f"pyjt_def_{n}(m);" for n in basenames])}
    }}

    }}
    """
    fname = os.path.join(cache_path, "gen", "pyjt_all.cc")
    LOG.vvv(("write to", fname))
    LOG.vvvv(code)
    with open(fname, "w", encoding='utf8') as f:
        f.write(code)
    pyjt_names.append(fname)
    return pyjt_names
