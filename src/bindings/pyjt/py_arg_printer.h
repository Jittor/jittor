// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <Python.h>
#include "core/common.h"

namespace jittor {

struct PyArgPrinter {
    PyObject* obj;
    const char* name;
};
std::ostream& operator<<(std::ostream& os, const PyArgPrinter& arg);

struct PyTupleArgPrinter {
    PyObject* obj;
    const char* name;
};
std::ostream& operator<<(std::ostream& os, const PyTupleArgPrinter& args);

struct PyKwArgPrinter {
    PyObject* obj;
};
std::ostream& operator<<(std::ostream& os, const PyKwArgPrinter& args);

struct PyFastCallArgPrinter {
    PyObject** obj;
    int64 n;
    PyObject* kw;
};
std::ostream& operator<<(std::ostream& os, const PyFastCallArgPrinter& args);

// Names, among the string arguments of a call that matched no signature, the
// first that looks like a dtype or operator name but is not one. Generated
// only for functions taking a NanoString, where a misspelt dtype such as
// 'float99' otherwise leaves only the full signature listing to go on.
// `fast`/`n`/`kwnames` describe a vectorcall; `tuple`/`dict` the other forms.
struct PyUnknownNameHint {
    PyObject** fast;
    int64 n;
    PyObject* kwnames;
    PyObject* tuple;
    PyObject* dict;
};
std::ostream& operator<<(std::ostream& os, const PyUnknownNameHint& args);

}
