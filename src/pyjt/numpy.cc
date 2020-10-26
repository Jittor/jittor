// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "pyjt/numpy.h"

namespace jittor {


void** PyArray_API;
PyTypeObject *PyArray_Type;
PyTypeObject *PyNumberArrType_Type;
PyTypeObject *PyArrayDescr_Type;
PyObject* (*PyArray_New)(PyTypeObject *, int, npy_intp const *, int, npy_intp const *, void *, int, int, PyObject *);
PyObject* (*PyArray_FromAny)(PyObject *, PyArrayDescr_Proxy *, int, int, int, PyObject *);
unsigned int (*PyArray_GetNDArrayCFeatureVersion)();
int (*PyArray_SetBaseObject)(PyObject *arr, PyObject *obj);
PyObject* (*PyArray_NewCopy)(PyObject *, int);

tmp_data_t tmp_data;

void numpy_init() {
    PyObjHolder np(PyImport_ImportModule("numpy.core.multiarray"));
    PyObjHolder api(PyObject_GetAttrString(np.obj, "_ARRAY_API"));
    PyArray_API = (void **) PyCapsule_GetPointer(api.obj, NULL);

    #define fill(name, i) name = (decltype(name))PyArray_API[i]
    fill(PyArray_Type, 2);
    fill(PyArrayDescr_Type, 3);
    fill(PyNumberArrType_Type, 11);
    fill(PyArray_FromAny, 69);
    fill(PyArray_New, 93);
    fill(PyArray_GetNDArrayCFeatureVersion, 211);
    fill(PyArray_SetBaseObject, 282);
    fill(PyArray_NewCopy, 85);

    ASSERT(PyArray_GetNDArrayCFeatureVersion()>=7);
}

} // jittor
