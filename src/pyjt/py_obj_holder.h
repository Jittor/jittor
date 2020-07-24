// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <Python.h>
#include "common.h"

namespace jittor {

struct PyObjHolder {
    PyObject* obj;
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

