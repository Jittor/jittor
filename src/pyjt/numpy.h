// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "pyjt/py_obj_holder.h"
#include "common.h"
#include "misc/nano_string.h"
#include "ops/array_op.h"

namespace jittor {

struct PyArrayDescr_Proxy {
    PyObject_HEAD
    PyObject* typeobj;
    char kind;
    char type;
    char byteorder;
    char flags;
    int type_num;
    int elsize;
    int alignment;
    char* subarray;
    PyObject *fields;
    PyObject *names;
};

struct PyArray_Proxy {
    PyObject_HEAD
    char* data;
    int nd;
    ssize_t* dimensions;
    ssize_t* strides;
    PyObject *base;
    PyArrayDescr_Proxy *descr;
    int flags;
};

#define NPY_ARRAY_C_CONTIGUOUS    0x0001
#define NPY_ARRAY_ALIGNED         0x0100
#define NPY_ARRAY_WRITEABLE       0x0400
// NPY_ARRAY_C_CONTIGUOUS=1
inline bool is_c_style(PyArray_Proxy* obj) { return obj->flags & 1; }
inline NanoString get_type_str(PyArray_Proxy* obj) {
    auto type = obj->descr->type;
    // bool ?
    if (type=='?') return ns_bool;
    // int8 b
    if (type=='b') return ns_int8;
    // int16 h
    if (type=='h') return ns_int16;
    // int32 i
    if (type=='i') return ns_int32;
    // int64 l
    if (type=='l') return ns_int64;
    // uint8 B
    if (type=='B') return ns_uint8;
    // uint16 H
    if (type=='H') return ns_uint16;
    // uint32 I
    if (type=='I') return ns_uint32;
    // uint64 L
    if (type=='L') return ns_uint64;
    // float32 f
    if (type=='f') return ns_float32;
    // float64 d
    if (type=='d') return ns_float64;
    LOGf << "Unsupport numpy type char:'" << type << '\'';
    return ns_float64;
}

inline int get_typenum(NanoString ns) {
    if (ns == ns_bool) return 0;
    if (ns == ns_int8) return 1;
    if (ns == ns_uint8) return 2;
    if (ns == ns_int16) return 3;
    if (ns == ns_uint16) return 4;
    if (ns == ns_int32) return 5;
    if (ns == ns_uint32) return 6;
    if (ns == ns_int64) return 7;
    if (ns == ns_uint64) return 8;
    if (ns == ns_float32) return 11;
    if (ns == ns_float64) return 12;
    LOGf << ns;
    return -1;
}

typedef Py_intptr_t npy_intp;

extern unordered_map<string, int> np_typenum_map;

extern void** PyArray_API;
extern PyTypeObject *PyArray_Type;
extern PyTypeObject *PyNumberArrType_Type;
extern PyTypeObject *PyArrayDescr_Type;
extern PyObject* (*PyArray_New)(PyTypeObject *, int, npy_intp const *, int, npy_intp const *, void *, int, int, PyObject *);
extern PyObject* (*PyArray_FromAny)(PyObject *, PyArrayDescr_Proxy *, int, int, int, PyObject *);
extern unsigned int (*PyArray_GetNDArrayCFeatureVersion)();
extern int (*PyArray_SetBaseObject)(PyObject *arr, PyObject *obj);
extern PyObject* (*PyArray_NewCopy)(PyObject *, int);
extern int (*PyArray_CopyInto)(PyObject *, PyObject *);
extern void (*PyArray_CastScalarToCtype)(PyObject* scalar, void* ctypeptr, PyArrayDescr_Proxy* outcode);

#define PyArray_Copy(obj) PyArray_NewCopy(obj, 0)

#define NPY_ARRAY_ALIGNED         0x0100
#define NPY_ARRAY_WRITEABLE       0x0400
#define NPY_ARRAY_BEHAVED      (NPY_ARRAY_ALIGNED | \
                                NPY_ARRAY_WRITEABLE)

#define NPY_ARRAY_CARRAY       (NPY_ARRAY_C_CONTIGUOUS | \
                                NPY_ARRAY_BEHAVED)

#define PyArray_SimpleNew(nd, dims, typenum) \
        PyArray_New(PyArray_Type, nd, dims, typenum, NULL, NULL, 0, 0, NULL)

#define PyArray_SimpleNewFromData(nd, dims, typenum, data) \
        PyArray_New(&PyArray_Type, nd, dims, typenum, NULL, \
                    data, 0, NPY_ARRAY_CARRAY, NULL)

#define PyArray_FROM_O(m) PyArray_FromAny(m, NULL, 0, 0, 0, NULL)

inline int64 PyArray_Size(PyArray_Proxy* arr) {
    int64 size = 1;
    for (int i=0; i<arr->nd; i++)
        size *= arr->dimensions[i];
    size *= arr->descr->elsize;
    return size;
}

union tmp_data_t {
    int32 i32;
    float32 f32;
    int8 i8;
};

extern tmp_data_t tmp_data;

void numpy_init();

} // jittor
