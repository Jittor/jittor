// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "pyjt/py_arg_printer.h"
#include "pyjt/py_obj_holder.h"
#include "pyjt/py_converter.h"

namespace jittor {

std::ostream& operator<<(std::ostream& os, const PyArgPrinter& arg) {
    os << " " << arg.name << "\t= ";
    if (!arg.obj) return os << "null,";
    return os << _PyType_Name(Py_TYPE(arg.obj)) << ",\n";
}

std::ostream& operator<<(std::ostream& os, const PyTupleArgPrinter& args) {
    os << " " << args.name << "\t= (";
    auto size = Py_SIZE(args.obj);
    auto arr = PySequence_Fast_ITEMS(args.obj);
    for (int i=0; i<size; i++) {
        os << _PyType_Name(Py_TYPE(arr[i])) << ", ";
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
        os << from_py_object<std::string>(key) << "=" << 
            _PyType_Name(Py_TYPE(value)) << ", ";
    }
    return os << "},\n";
}

std::ostream& operator<<(std::ostream& os, const PyFastCallArgPrinter& args) {
    os << " args\t= (";
    auto size = args.n;
    auto arr = args.obj;
    for (int i=0; i<size; i++) {
        os << _PyType_Name(Py_TYPE(arr[i])) << ", ";
    }
    os << "),\n";
    auto kw = args.kw;
    if (!kw) return os;
    os << " kwargs\t= {";
    auto kw_n = Py_SIZE(kw);
    for (int i=0; i<kw_n; i++) {
        auto ko = PyTuple_GET_ITEM(kw, i);
        auto ks = PyUnicode_AsUTF8(ko);
        os << ks << "=" << _PyType_Name(Py_TYPE(arr[i+size])) << ", ";
    }
    return os << "},\n";
}

}
