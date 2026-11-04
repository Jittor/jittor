// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
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

// Flag to indicate whether we are running with NumPy 2.x
// In NumPy 2.x, PyArray_Descr layout changed:
//   - after type_num, a new npy_uint64 flags field was inserted
//   - elsize changed from int to npy_intp
//   - alignment changed from int to npy_intp
EXTERN_LIB int numpy_is_v2;

// Compatible struct that only includes fields with stable offsets across
// NumPy 1.x and 2.x (up to type_num). elsize must be accessed via
// the helper function get_descr_elsize() below.
struct PyArrayDescr_Proxy {
    PyObject_HEAD
    PyObject* typeobj;
    char kind;
    char type;
    char byteorder;
    char flags;
    int type_num;
    // WARNING: fields after type_num have DIFFERENT offsets in NumPy 1.x vs 2.x!
    // NumPy 1.x: int elsize (offset = offsetof(type_num) + 4)
    // NumPy 2.x: npy_uint64 flags_new (8 bytes), then npy_intp elsize (8 bytes)
    // Do NOT access elsize directly! Use get_descr_elsize() instead.
    int elsize;  // Only valid for NumPy 1.x
    int alignment;
    char* subarray;
    PyObject *fields;
    PyObject *names;
};

// Get the element size from a PyArrayDescr_Proxy, compatible with both
// NumPy 1.x and NumPy 2.x
inline int64 get_descr_elsize(PyArrayDescr_Proxy* descr) {
    if (numpy_is_v2) {
        // NumPy 2.x layout after type_num(offset 28 from PyObject_HEAD start):
        //   npy_uint64 flags (8 bytes) at offset +4 from type_num
        //   npy_intp elsize (8 bytes) at offset +12 from type_num
        char* base = (char*)&descr->type_num;
        // Skip type_num(4) + flags(8) = 12 bytes to reach elsize
        return (int64)(*(Py_ssize_t*)(base + 4 + 8));
    } else {
        return (int64)descr->elsize;
    }
}

struct PyArray_Proxy {
    PyObject_HEAD
    char* data;
    int nd;
    Py_ssize_t* dimensions;
    Py_ssize_t* strides;
    PyObject *base;
    PyArrayDescr_Proxy *descr;
    int flags;
};

enum NPY_TYPES {    
    NPY_BOOL=0,
    NPY_BYTE, NPY_UBYTE,
    NPY_SHORT, NPY_USHORT,
    NPY_INT, NPY_UINT,
    NPY_LONG, NPY_ULONG,
    NPY_LONGLONG, NPY_ULONGLONG,
    NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
    NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
    NPY_OBJECT=17,
    NPY_HALF=23,
    NPY_END=24,
};

EXTERN_LIB NanoString npy2ns[];
EXTERN_LIB NPY_TYPES ns2npy[];

#define NPY_ARRAY_C_CONTIGUOUS    0x0001
#define NPY_ARRAY_ALIGNED         0x0100
#define NPY_ARRAY_WRITEABLE       0x0400
// NPY_ARRAY_C_CONTIGUOUS=1
inline bool is_c_style(PyArray_Proxy* obj) { return obj->flags & 1; }
inline NanoString get_type_str(PyArray_Proxy* obj) {
    NanoString type = ns_void;
    if (obj->descr->type_num < NPY_END)
        type = npy2ns[obj->descr->type_num];
    CHECK(type != ns_void) << "Numpy type not support, type_num:"
        << obj->descr->type_num 
        << "type_char:" << obj->descr->type << NPY_END << npy2ns[obj->descr->type_num];
    return type;
}

inline int get_typenum(NanoString ns) {
    return ns2npy[ns.index()];
}

typedef Py_intptr_t npy_intp;

EXTERN_LIB unordered_map<string, int> np_typenum_map;

EXTERN_LIB void** PyArray_API;
EXTERN_LIB PyTypeObject *PyArray_Type;
EXTERN_LIB PyTypeObject *PyNumberArrType_Type;
EXTERN_LIB PyTypeObject *PyArrayDescr_Type;
EXTERN_LIB PyObject* (*PyArray_New)(PyTypeObject *, int, npy_intp const *, int, npy_intp const *, void *, int, int, PyObject *);
EXTERN_LIB PyObject* (*PyArray_FromAny)(PyObject *, PyArrayDescr_Proxy *, int, int, int, PyObject *);
EXTERN_LIB unsigned int (*PyArray_GetNDArrayCFeatureVersion)();
EXTERN_LIB int (*PyArray_SetBaseObject)(PyObject *arr, PyObject *obj);
EXTERN_LIB PyObject* (*PyArray_NewCopy)(PyObject *, int);
EXTERN_LIB int (*PyArray_CopyInto)(PyObject *, PyObject *);
EXTERN_LIB void (*PyArray_CastScalarToCtype)(PyObject* scalar, void* ctypeptr, PyArrayDescr_Proxy* outcode);

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
    size *= get_descr_elsize(arr->descr);
    return size;
}

union tmp_data_t {
    int32 i32;
    float32 f32;
    int8 i8;
};

EXTERN_LIB tmp_data_t tmp_data;

void numpy_init();

} // jittor
