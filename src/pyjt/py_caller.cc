// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "pyjt/py_obj_holder.h"
#include "pyjt/py_converter.h"
#include "pyjt/py_caller.h"

namespace jittor {

string py_caller(const string& mod_func, const vector<string>& args, const map<string,string>& kw) {
    PyObjHolder mod(PyImport_ImportModule("jittor"));
    PyObjHolder func(PyObject_GetAttrString(mod.obj, "python_pass_warper"));
    PyObjHolder py_name(to_py_object<string>(mod_func));
    PyObjHolder py_args(to_py_tuple(args));
    PyObjHolder py_kw(to_py_object(kw));
    PyObjHolder ret(PyObject_CallFunctionObjArgs(func.obj, py_name.obj, py_args.obj, py_kw.obj, nullptr));
    CHECK(is_type<string>(ret.obj)) << "expect return type string.";
    return from_py_object<string>(ret.obj);
}

}
