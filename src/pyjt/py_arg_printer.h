// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <Python.h>
#include "common.h"

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

}
