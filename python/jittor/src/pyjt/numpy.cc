// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "pyjt/numpy.h"

namespace jittor {

NanoString npy2ns[] = {
    ns_bool, 
    ns_int8, ns_uint8,
    ns_int16, ns_uint16,
    ns_int32, ns_uint32,
    #ifdef _WIN32
    ns_int32, ns_uint32,
    #else
    ns_int64, ns_uint64,
    #endif
    ns_int64, ns_uint64,
    ns_float32, ns_float64, ns_float64,
    ns_void, ns_void, ns_void, 
    ns_void, // 17
    ns_void, ns_void, ns_void, ns_void, ns_void, // 22
    ns_float16, // 23
};

NPY_TYPES ns2npy[] = {
    NPY_OBJECT, // placeholder for ns_void
    NPY_BOOL,
    #ifdef _WIN32
    NPY_BYTE, NPY_SHORT, NPY_LONG, NPY_LONGLONG,
    NPY_UBYTE, NPY_USHORT, NPY_ULONG, NPY_ULONGLONG,
    #else
    NPY_BYTE, NPY_SHORT, NPY_INT, NPY_LONGLONG,
    NPY_UBYTE, NPY_USHORT, NPY_UINT, NPY_ULONGLONG,
    #endif
    NPY_HALF, NPY_FLOAT, NPY_DOUBLE
};

void** PyArray_API;
PyTypeObject *PyArray_Type;
PyTypeObject *PyNumberArrType_Type;
PyTypeObject *PyArrayDescr_Type;
PyObject* (*PyArray_New)(PyTypeObject *, int, npy_intp const *, int, npy_intp const *, void *, int, int, PyObject *);
PyObject* (*PyArray_FromAny)(PyObject *, PyArrayDescr_Proxy *, int, int, int, PyObject *);
unsigned int (*PyArray_GetNDArrayCFeatureVersion)();
int (*PyArray_SetBaseObject)(PyObject *arr, PyObject *obj);
PyObject* (*PyArray_NewCopy)(PyObject *, int);
int (*PyArray_CopyInto)(PyObject *, PyObject *);
void (*PyArray_CastScalarToCtype)(PyObject* scalar, void* ctypeptr, PyArrayDescr_Proxy* outcode);

tmp_data_t tmp_data;

void numpy_init() {
    PyObjHolder np(PyImport_ImportModule("numpy.core.multiarray"), "numpy is not installed");
    PyObjHolder api(PyObject_GetAttrString(np.obj, "_ARRAY_API"), "numpy _ARRAY_API not found, you may need to reinstall numpy");
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
    fill(PyArray_CopyInto, 82);
    fill(PyArray_CastScalarToCtype, 63);

    ASSERT(PyArray_GetNDArrayCFeatureVersion()>=7);
}

} // jittor
