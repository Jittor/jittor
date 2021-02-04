// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <Python.h>
#include "common.h"

namespace jittor {

struct PyObjHolder {
    PyObject* obj;
    inline PyObjHolder() : obj(nullptr) {
    }
    inline void assign(PyObject* obj) {
        if (!obj) {
            LOGf << "Python error occur";
        }
        this->obj = obj;
    }
    inline PyObjHolder(PyObject* obj) : obj(obj) {
        if (!obj) {
            LOGf << "Python error occur";
        }
    }
    inline ~PyObjHolder() {
        if (obj) Py_DECREF(obj);
    }
    inline PyObject* release() {
        auto tmp = obj;
        obj = nullptr;
        return tmp;
    }
};


inline Log& operator<<(Log& os, PyObject* objp) {
    PyObjHolder repr_obj(PyObject_Repr(objp));
    
    if (PyUnicode_CheckExact(repr_obj.obj)) {
        return os << Py_TYPE(objp)->tp_name <<
             PyUnicode_AsUTF8(repr_obj.obj);
    } else {
        return os << "unknown(" >> (void*)objp >> ")";
    }
}

}

#define PYJF_MODULE_INIT(name) \
PyMODINIT_FUNC PyInit_##name() { \
    PyObject *m; \
    try { \
        PyModuleDef *def = new PyModuleDef(); \
        memset(def, 0, sizeof(PyModuleDef)); \
        def->m_name = #name; \
        def->m_doc = ""; \
        def->m_size = -1; \
        Py_INCREF(def); \
        jittor::PyObjHolder holder(m = PyModule_Create(def)); \
        init_module(def, m); \
        holder.release(); \
    } catch(const std::exception& e) { \
        PyErr_SetString(PyExc_RuntimeError, e.what()); \
        return nullptr; \
    } \
    return m; \
}

