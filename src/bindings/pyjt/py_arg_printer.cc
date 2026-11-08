// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "bindings/pyjt/py_arg_printer.h"
#include "bindings/pyjt/py_obj_holder.h"
#include "bindings/pyjt/py_converter.h"
#include "type/nano_string.h"
#include <cctype>
#include <cstring>

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

// A short identifier-like string: the shape of a name, not of source code or
// a path, which other `str` parameters of the same function may carry.
static bool looks_like_a_name(PyObject* obj, const char*& text) {
    if (!obj || !PyUnicode_Check(obj)) return false;
    text = PyUnicode_AsUTF8(obj);
    if (!text) { PyErr_Clear(); return false; }
    size_t length = strlen(text);
    if (!length || length > 24) return false;
    if (!(isalpha((unsigned char)text[0]) || text[0] == '_')) return false;
    for (size_t i = 0; i < length; i++)
        if (!(isalnum((unsigned char)text[i]) || text[i] == '_' || text[i] == '.'))
            return false;
    return true;
}

std::ostream& operator<<(std::ostream& os, const PyUnknownNameHint& args) {
    vector<PyObject*> values;
    for (int64 i = 0; args.fast && i < args.n + (args.kwnames ? Py_SIZE(args.kwnames) : 0); i++)
        values.push_back(args.fast[i]);
    if (args.tuple && PyTuple_Check(args.tuple))
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args.tuple); i++)
            values.push_back(PyTuple_GET_ITEM(args.tuple, i));
    if (args.dict && PyDict_Check(args.dict)) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(args.dict, &pos, &key, &value)) values.push_back(value);
    }
    for (auto value : values) {
        const char* text = nullptr;
        if (looks_like_a_name(value, text) && !ns_valid_name(text))
            return os << "\n'" << text << "' is not a dtype or operator name jittor "
                "knows; dtypes are spelled like 'float32', 'int64' or 'bool'.";
    }
    return os;
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
