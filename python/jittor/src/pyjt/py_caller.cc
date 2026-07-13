// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
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

// RAII guard for the Python GIL.
//
// py_caller is invoked from OpCompiler::precompile while handling an
// `@python(...)` JIT directive. Under the parallel op compiler
// (use_parallel_op_compiler>0, the default) this runs on JIT compile
// *worker* threads (named C0/C1/...) that were never created by Python
// and therefore hold no GIL. Calling raw CPython C-API
// (PyImport_ImportModule / PyObject_GetAttrString /
// PyObject_CallFunctionObjArgs -> importlib / getattr / eval) from
// several such threads concurrently, with no GIL, corrupts the
// interpreter (e.g. faults like PyUnicode_New @ 0x1). The old docs
// blamed an "inf/nan ternary codegen segfault", but the real cause is
// this missing GIL synchronization: a large nested ternary just happens
// to be big enough to trigger auto_parallel, which emits the @python
// directive that lands here on worker threads.
//
// PyGILState_Ensure/Release is reentrant per-thread, so this is correct
// for both paths:
//   * worker threads        -> create/attach a thread state and take the GIL;
//   * the rare serial path  -> the calling thread already holds the GIL,
//                              Ensure simply nests and Release is a no-op.
// It serializes the @python compile passes behind the single GIL,
// eliminating the corruption while leaving generated-kernel semantics,
// fusion and meta-ops untouched.
//
// NOTE: for the worker threads to actually be able to take the GIL, the
// main thread must *release* it while it spin-waits for the parallel
// compile to finish (it otherwise holds the GIL the whole time, since it
// entered C++ from a pybind call). That release is done in
// parallel_compile_all_ops (parallel_compiler.cc); without it
// PyGILState_Ensure here would block forever -> deadlock.
struct GILScope {
    PyGILState_STATE state;
    inline GILScope() { state = PyGILState_Ensure(); }
    inline ~GILScope() { PyGILState_Release(state); }
};

string py_caller(const string& mod_func, const vector<string>& args, const map<string,string>& kw) {
    // Take the GIL for the whole CPython interaction below; released on
    // any return path, including the LOGf-throw inside PyObjHolder when a
    // CPython call returns nullptr (RAII => exception safe).
    GILScope gil;
    PyObjHolder mod(PyImport_ImportModule("jittor"));
    PyObjHolder func(PyObject_GetAttrString(mod.obj, "python_pass_wrapper"));
    PyObjHolder py_name(to_py_object<string>(mod_func));
    PyObjHolder py_args(to_py_tuple(args));
    PyObjHolder py_kw(to_py_object(kw));
    PyObjHolder ret(PyObject_CallFunctionObjArgs(func.obj, py_name.obj, py_args.obj, py_kw.obj, nullptr));
    CHECK(is_type<string>(ret.obj)) << "expect return type string.";
    return from_py_object<string>(ret.obj);
}

}
