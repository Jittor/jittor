// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "bindings/pyjt/py_arg_printer.h"
#include "bindings/pyjt/py_obj_holder.h"
#include "bindings/pyjt/py_converter.h"

namespace jittor {

// The type, and for a str or a number its value too: "str 'float99'" says
// which argument was wrong where "str" alone does not.
static void describe(std::ostream& os, PyObject* obj) {
    os << _PyType_Name(Py_TYPE(obj));
    // Exact types only, so no user-defined __repr__ runs on the error path.
    if (!(PyUnicode_CheckExact(obj) || PyLong_CheckExact(obj) || PyBool_Check(obj)
          || PyFloat_CheckExact(obj)))
        return;
    PyObject* text = PyObject_Repr(obj);
    const char* utf8 = text ? PyUnicode_AsUTF8(text) : nullptr;
    if (utf8) {
        string value(utf8);
        if (value.size() > 40) value = value.substr(0, 37) + "...";
        os << ' ' << value;
    } else {
        PyErr_Clear();
    }
    Py_XDECREF(text);
}

std::ostream& operator<<(std::ostream& os, const PyArgPrinter& arg) {
    // A module-level function is called with its module as `self`.
    if (arg.obj && PyModule_Check(arg.obj)) return os;
    os << " " << arg.name << "\t= ";
    if (!arg.obj) return os << "null,";
    describe(os, arg.obj);
    return os << ",\n";
}

std::ostream& operator<<(std::ostream& os, const PyTupleArgPrinter& args) {
    os << " " << args.name << "\t= (";
    auto size = Py_SIZE(args.obj);
    auto arr = PySequence_Fast_ITEMS(args.obj);
    for (int i=0; i<size; i++) {
        describe(os, arr[i]);
        os << ", ";
    }
    return os << "),\n";
}

std::ostream& operator<<(std::ostream& os, const PyKwArgPrinter& args) {
    auto obj = args.obj;
    if (!obj) return os;

    // auto size = Py_SIZE(obj);
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    os << " " << "kwargs\t= {";
    while (PyDict_Next(obj, &pos, &key, &value)) {
        os << from_py_object<std::string>(key) << "=";
        describe(os, value);
        os << ", ";
    }
    return os << "},\n";
}

std::ostream& operator<<(std::ostream& os, const PyFastCallArgPrinter& args) {
    os << " args\t= (";
    auto size = args.n;
    auto arr = args.obj;
    for (int i=0; i<size; i++) {
        describe(os, arr[i]);
        os << ", ";
    }
    os << "),\n";
    auto kw = args.kw;
    if (!kw) return os;
    os << " kwargs\t= {";
    auto kw_n = Py_SIZE(kw);
    for (int i=0; i<kw_n; i++) {
        auto ko = PyTuple_GET_ITEM(kw, i);
        auto ks = PyUnicode_AsUTF8(ko);
        os << ks << "=";
        describe(os, arr[i+size]);
        os << ", ";
    }
    return os << "},\n";
}

}
