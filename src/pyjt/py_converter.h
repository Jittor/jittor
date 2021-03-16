// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
//     Guowei Yang <471184555@qq.com>
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "pyjt/py_obj_holder.h"
#include "pyjt/numpy.h"
#include "common.h"
#include "misc/hash.h"
#include "misc/nano_string.h"
#include "misc/fast_shared_ptr.h"
#include "profiler/simple_profiler.h"
#ifdef HAS_CUDA
#include "misc/cuda_flags.h"
#endif

namespace jittor {

#define DEF_IS(check_type, return_type) \
    template<class T> \
    typename std::enable_if<std::is_same<T, check_type>::value, return_type>::type

#define GET_PY_NONE(code) ((code), Py_INCREF(Py_None), Py_None)

// string
DEF_IS(string, bool) is_type(PyObject* obj) {
    return PyUnicode_CheckExact(obj);
}

DEF_IS(string, PyObject*) to_py_object(const string& a) {
    return PyUnicode_FromStringAndSize(a.c_str(), a.size());
}

DEF_IS(string, string) from_py_object(PyObject* obj) {
    Py_ssize_t size;
    const char* s = PyUnicode_AsUTF8AndSize(obj, &size);
    CHECK(s);
    return string(s, size);
}

// bool
DEF_IS(bool, bool) is_type(PyObject* obj) {
    return PyBool_Check(obj) || PyLong_CheckExact(obj);
}

DEF_IS(bool, PyObject*) to_py_object(const T& a) {
    if (a) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

DEF_IS(bool, T) from_py_object(PyObject* obj) {
    if (PyBool_Check(obj))
        return obj == Py_True;
    return PyLong_AsLong(obj);
}

// int
DEF_IS(int, bool) is_type(PyObject* obj) {
    return PyLong_CheckExact(obj);
}

DEF_IS(int, PyObject*) to_py_object(const T& a) {
    return PyLong_FromLong(a);
}

DEF_IS(int, T) from_py_object(PyObject* obj) {
    return PyLong_AsLong(obj);
}

// size_t
DEF_IS(size_t, bool) is_type(PyObject* obj) {
    return PyLong_CheckExact(obj);
}

DEF_IS(size_t, PyObject*) to_py_object(const T& a) {
    return PyLong_FromUnsignedLongLong(a);
}

DEF_IS(size_t, T) from_py_object(PyObject* obj) {
    return PyLong_AsUnsignedLongLong(obj);
}

// int64
DEF_IS(int64, bool) is_type(PyObject* obj) {
    return PyLong_CheckExact(obj);
}

DEF_IS(int64, PyObject*) to_py_object(const T& a) {
    return PyLong_FromLongLong(a);
}

DEF_IS(int64, T) from_py_object(PyObject* obj) {
    return PyLong_AsLongLong(obj);
}

// float64
DEF_IS(float64, bool) is_type(PyObject* obj) {
    return PyFloat_CheckExact(obj) || PyLong_CheckExact(obj);
}

DEF_IS(float64, PyObject*) to_py_object(const T& a) {
    return PyFloat_FromDouble(a);
}

DEF_IS(float64, T) from_py_object(PyObject* obj) {
    if (PyFloat_CheckExact(obj))
        return PyFloat_AS_DOUBLE(obj);
    return PyLong_AsDouble(obj);
}

struct Slice;
// Slice
DEF_IS(Slice, bool) is_type(PyObject* obj) {
    return PySlice_Check(obj);
}
DEF_IS(Slice, T) from_py_object(PyObject* obj) {
    Py_ssize_t start, stop, step;
    auto slice = (PySliceObject*)obj;

    PySlice_Unpack(obj, &start, &stop, &step);
    return {start, stop, step, 
        (slice->start == Py_None) |
        ((slice->stop == Py_None) << 1) |
        ((slice->step == Py_None) << 2)};
}

#define GET_RAW_PTR(T, obj) ((T*)(((char*)obj) + sizeof(PyObject)))
#define GET_OBJ_FROM_RAW_PTR(obj) ((PyObject*)(((char*)obj) - sizeof(PyObject)))
#define GET_OBJ_SIZE(T) (sizeof(PyObject)+sizeof(T))

// DumpGraphs
struct DumpGraphs;
extern PyTypeObject PyjtDumpGraphs;
DEF_IS(DumpGraphs, bool) is_type(PyObject* obj) {
    return Py_TYPE(obj) == &PyjtDumpGraphs;
}


DEF_IS(DumpGraphs, PyObject*) to_py_object(T&& a) {
    PyObjHolder obj(_PyObject_New(&PyjtDumpGraphs));
    auto ptr = GET_RAW_PTR(T, obj.obj);
    new (ptr) T();
    ptr->nodes_info = std::move(a.nodes_info);
    ptr->inputs = std::move(a.inputs);
    ptr->outputs = std::move(a.outputs);
    return obj.release();
}

DEF_IS(DumpGraphs, const T&) from_py_object(PyObject* obj) {
    return GET_RAW_PTR(T, obj);
}

// MemInfo
struct MemInfo;
extern PyTypeObject PyjtMemInfo;
DEF_IS(MemInfo, bool) is_type(PyObject* obj) {
    return Py_TYPE(obj) == &PyjtMemInfo;
}


DEF_IS(MemInfo, PyObject*) to_py_object(const T& a) {
    PyObjHolder obj(_PyObject_New(&PyjtMemInfo));
    auto ptr = GET_RAW_PTR(T, obj.obj);
    new (ptr) T(a);
    return obj.release();
}

DEF_IS(MemInfo, const T&) from_py_object(PyObject* obj) {
    return GET_RAW_PTR(T, obj);
}


// NanoString
struct NanoString;
extern PyTypeObject PyjtNanoString;
DEF_IS(NanoString, bool) is_type(PyObject* obj) {
    return Py_TYPE(obj) == &PyjtNanoString ||
        PyUnicode_CheckExact(obj) ||
        PyType_CheckExact(obj) ||
        // jt.float.__name__
        PyCallable_Check(obj) ||
        // numpy.dtype.type
        PyObject_HasAttrString(obj, "type");
}

DEF_IS(NanoString, PyObject*) to_py_object(T a) {
    PyObjHolder obj(_PyObject_New(&PyjtNanoString));
    auto ptr = GET_RAW_PTR(T, obj.obj);
    new (ptr) T(a);
    return obj.release();
}

DEF_IS(NanoString, T) from_py_object(PyObject* obj) {
    if (Py_TYPE(obj) == &PyjtNanoString)
        return *GET_RAW_PTR(T, obj);
    if (PyUnicode_CheckExact(obj))
        return T(PyUnicode_AsUTF8(obj));
    // PyType
    if (PyType_CheckExact(obj))
        return T(_PyType_Name((PyTypeObject *)obj));
    // jt.float.__name__
    if (PyCallable_Check(obj)) {
        PyObjHolder t(PyObject_GetAttrString(obj, "__name__"));
        return T(PyUnicode_AsUTF8(t.obj));
    }
    PyObjHolder t(PyObject_GetAttrString(obj, "type"));
    CHECK(PyType_CheckExact(t.obj)) << "Not a valid type:" << t.obj;
    return T(_PyType_Name((PyTypeObject *)t.obj));
}

// NanoVector
struct NanoVector;
extern PyTypeObject PyjtNanoVector;
DEF_IS(NanoVector, bool) is_type(PyObject* obj) {
    return Py_TYPE(obj) == &PyjtNanoVector ||
        PyList_CheckExact(obj) || PyTuple_CheckExact(obj);
}
DEF_IS(NanoVector*, bool) is_type(PyObject* obj) {
    return Py_TYPE(obj) == &PyjtNanoVector;
}

DEF_IS(NanoVector, PyObject*) to_py_object(T a) {
    PyObjHolder obj(_PyObject_New(&PyjtNanoVector));
    auto ptr = GET_RAW_PTR(T, obj.obj);
    new (ptr) T(a);
    return obj.release();
}

DEF_IS(NanoVector*, T) from_py_object(PyObject* obj) {
    return GET_RAW_PTR(typename std::remove_pointer<T>::type, obj);
}

DEF_IS(NanoVector, T) from_py_object(PyObject* obj) {
    if (Py_TYPE(obj) == &PyjtNanoVector)
        return *GET_RAW_PTR(T, obj);
    auto size = Py_SIZE(obj);
    T a;
    auto arr = PySequence_Fast_ITEMS(obj);
    for (int64 i=0; i<size; i++) {
        auto oi = arr[i]; 
        CHECK(is_type<int64>(oi));
        a.push_back_check_overflow(from_py_object<int64>(oi));
    }
    return a;
}

// ArrayArgs
struct ArrayArgs;
struct VarHolder;
vector<ArrayArgs> fetch_sync(const vector<VarHolder*>& vh);
extern PyHeapTypeObject PyjtVarHolder;
DEF_IS(ArrayArgs, bool) is_type(PyObject* obj) {
    return 
        Py_TYPE(obj) == &PyjtVarHolder.ht_type ||
        Py_TYPE(obj) == PyArray_Type || 
        PyFloat_CheckExact(obj) ||
        PyLong_CheckExact(obj) ||
        PyBool_Check(obj) ||
        PyList_CheckExact(obj) ||
        PyObject_TypeCheck(obj, PyNumberArrType_Type);
}

DEF_IS(ArrayArgs, PyObject*) to_py_object(const T& a) {
    int64 dims[a.shape.size()];
    for (int i=0; i<a.shape.size(); i++)
        dims[i] = a.shape[i];
    PyObjHolder obj(PyArray_SimpleNew(
        a.shape.size(),
        dims,
        get_typenum(a.dtype)
    ));
    auto arr = (PyArray_Proxy*)(obj.obj);
    int64 size = PyArray_Size(arr);
    memcpy((void*)arr->data, (void*)a.ptr, size);
    return obj.release();
}

DEF_IS(ArrayArgs, T) from_py_object(PyObject* obj) {
    if (PyFloat_CheckExact(obj)) {
        tmp_data.f32 = PyFloat_AS_DOUBLE(obj);
        return {&tmp_data, 1, ns_float32};
    }
    if (PyLong_CheckExact(obj)) {
        tmp_data.i32 = PyLong_AsLong(obj);
        return {&tmp_data, 1, ns_int32};
    }
    if (PyBool_Check(obj)) {
        tmp_data.i8 = obj == Py_True;
        return {&tmp_data, 1, ns_bool};
    }
    if (Py_TYPE(obj) == &PyjtVarHolder.ht_type) {
        auto ptr = GET_RAW_PTR(VarHolder, obj);
        return move(fetch_sync({ptr}).at(0));
    }
    // PyArray_Type
    auto arr = (PyArray_Proxy*)obj;
    if (Py_TYPE(obj) != PyArray_Type || !is_c_style(arr)) {
        PyObjHolder holder(
            Py_TYPE(obj) != PyArray_Type ? 
                PyArray_FROM_O(obj) :
                PyArray_Copy(obj));
        auto arr = (PyArray_Proxy*)holder.obj;
        int64 size = PyArray_Size(arr);
        T args;
        if (arr->nd)
            args.shape = NanoVector::make(arr->dimensions, arr->nd);
        else
            args.shape.push_back(1);
        args.dtype = get_type_str(arr);
        args.buffer.reset(new char[size]);
        args.ptr = (void*)args.buffer.get();
        memcpy((void*)args.buffer.get(), (void*)arr->data, size);
        if (Py_TYPE(obj) != PyArray_Type && args.dtype.dsize()==8) {
            // convert to 32bit
            auto num = size/8;
            if (args.dtype.is_int()) {
                auto* __restrict__ i64 = (int64*)args.ptr;
                auto* __restrict__ i32 = (int32*)args.ptr;
                for (int i=0; i<num; i++)
                    i32[i] = (int32)i64[i];
                args.dtype = ns_int32;
            } else if (args.dtype.is_float()) {
                auto* __restrict__ f64 = (float64*)args.ptr;
                auto* __restrict__ f32 = (float32*)args.ptr;
                for (int i=0; i<num; i++)
                    f32[i] = (float32)f64[i];
                args.dtype = ns_float32;
            }
        }
        return args;
    }
    T args;
    args.ptr = arr->data;
    if (arr->dimensions)
        for (int i=0; i<arr->nd; i++)
            args.shape.push_back(arr->dimensions[i]);
    else
        args.shape = 1;
    args.dtype = get_type_str(arr);
    return args;
}

// VarHolder
struct VarHolder;
extern PyHeapTypeObject PyjtVarHolder;
namespace jit_op_maker { extern VarHolder* array_(ArrayArgs&& args); }
DEF_IS(VarHolder*, bool) is_type(PyObject* obj) {
    return Py_TYPE(obj) == &PyjtVarHolder.ht_type ||
        is_type<ArrayArgs>(obj);
}

DEF_IS(VarHolder*, PyObject*) to_py_object(T a) {
    PyObjHolder obj(_PyObject_New(&PyjtVarHolder.ht_type));
    auto ptr = GET_RAW_PTR(T, obj.obj);
    // will move and delete a
    new (ptr) typename std::remove_pointer<T>::type (a);
    return obj.release();
}


DEF_IS(VarHolder*, T) from_py_object(PyObject* obj) {
    CHECK(Py_TYPE(obj) == &PyjtVarHolder.ht_type);
    return GET_RAW_PTR(VarHolder, obj);
}

DEF_IS(VarHolder*, T) from_py_object(PyObject* obj, unique_ptr<VarHolder>& holder) {
    if (Py_TYPE(obj) == &PyjtVarHolder.ht_type)
        return GET_RAW_PTR(VarHolder, obj);
    auto args = from_py_object<ArrayArgs>(obj);
    holder.reset(jit_op_maker::array_(move(args)));
    return holder.get();
}

struct DataView;
DEF_IS(DataView, PyObject*) to_py_object(T a) {
    int64 dims[a.shape.size()];
    for (int i=0; i<a.shape.size(); i++)
        dims[i] = a.shape[i];
    PyObjHolder oh(PyArray_New(
        PyArray_Type, // subtype
        a.shape.size(), // nd
        dims, // dims
        get_typenum(a.dtype), // type_num
        NULL, // strides
        a.ptr, // data
        0, // itemsize
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, // flags
        NULL // obj
    ));
    if (a.vh) {
        auto obj = GET_OBJ_FROM_RAW_PTR(a.vh);
        PyObjHolder oh2(obj);
        Py_INCREF(obj);
        ASSERT(PyArray_SetBaseObject(oh.obj, oh2.obj)==0);
        oh2.release();
    }
    return oh.release();
}


#pragma GCC diagnostic ignored "-Wstrict-aliasing"
struct ItemData;
DEF_IS(ItemData, PyObject*) to_py_object(T a) {
    if (a.dtype == ns_bool) {
        if (*((bool*)(&a.data))) Py_RETURN_TRUE;
        Py_RETURN_FALSE;
    }
    if (a.dtype == ns_int32)
        return PyLong_FromLongLong((int64)*(int*)&a.data);
    if (a.dtype == ns_float32)
        return PyFloat_FromDouble((float64)*(float32*)&a.data);
    if (a.dtype == ns_int64)
        return PyLong_FromLongLong(a.data);
    if (a.dtype == ns_float64)
        return PyFloat_FromDouble(*(float64*)&a.data);
    if (a.dtype == ns_int16)
        return PyLong_FromLongLong((int64)*(int16*)&a.data);
    if (a.dtype == ns_int8)
        return PyLong_FromLongLong((int64)*(int8*)&a.data);
    return PyLong_FromLongLong(a.data);
}

struct NumpyFunc;

DEF_IS(NumpyFunc, bool) is_type(PyObject* obj) {
    return PyCallable_Check(obj);
}

DEF_IS(NumpyFunc, T) from_py_object(PyObject* obj);

#define CHECK_IS_1(check_type) \
    template<typename T> struct is_##check_type : public std::false_type {}; \
    template<typename T> \
    struct is_##check_type<check_type<T>> : public std::true_type {};

#define DEF_IS_1(check_type, return_type) \
    template<class T> \
    typename std::enable_if<is_##check_type<T>::value, return_type>::type

CHECK_IS_1(vector);

DEF_IS_1(vector, bool) is_type(PyObject* obj) {
    if (!(PyList_CheckExact(obj) || PyTuple_CheckExact(obj)))
        return false;
    auto size = Py_SIZE(obj);
    if (!size)
        return true;
    auto arr = PySequence_Fast_ITEMS(obj);
    return is_type<typename T::value_type>(arr[0]);
}

DEF_IS_1(vector, PyObject*) to_py_object(const T& a) {
    PyObjHolder list(PyList_New(a.size()));
    for (uint i=0; i<a.size(); i++) {
        PyObject* o = to_py_object<typename T::value_type>(a[i]);
        CHECK(o);
        // PyList_SET_ITEM borrow ownership, we do not hold this
        PyList_SET_ITEM(list.obj, i, o);
    }
    return list.release();
}

DEF_IS_1(vector, PyObject*) to_py_tuple(const T& a) {
    PyObjHolder list(PyTuple_New(a.size()));
    for (uint i=0; i<a.size(); i++) {
        PyObject* o = to_py_object<typename T::value_type>(a[i]);
        CHECK(o);
        // PyTuple_SET_ITEM borrow ownership, we do not hold this
        PyTuple_SET_ITEM(list.obj, i, o);
    }
    return list.release();
}

DEF_IS_1(vector, PyObject*) to_py_object(T&& a) {
    PyObjHolder list(PyList_New(a.size()));
    for (uint i=0; i<a.size(); i++) {
        PyObject* o = to_py_object<typename T::value_type>(std::move(a[i]));
        CHECK(o);
        // PyList_SET_ITEM borrow ownership, we do not hold this
        PyList_SET_ITEM(list.obj, i, o);
    }
    return list.release();
}

DEF_IS_1(vector, T) from_py_object(PyObject* obj) {
    auto size = Py_SIZE(obj);
    T a(size);
    auto arr = PySequence_Fast_ITEMS(obj);
    for (int64 i=0; i<size; i++) {
        auto oi = arr[i]; 
        CHECK(is_type<typename T::value_type>(oi));
        a[i] = from_py_object<typename T::value_type>(oi);
    }
    return a;
}

struct FetchFunc;

DEF_IS(FetchFunc, bool) is_type(PyObject* obj) {
    return PyCallable_Check(obj);
}

DEF_IS(FetchFunc, T) from_py_object(PyObject* obj) {
    // PyObject_Call
    Py_INCREF(obj);
    T func(
        // callback
        [obj](typename T::R* result) {
            PyObjHolder arrays(to_py_tuple<vector<ArrayArgs>>(result->arrays));
            PyObjHolder ret(PyObject_Call(obj, arrays.obj, nullptr));
        },
        // deleter
        [obj]() { Py_DECREF(obj); }
    );
    return func;
}


#define CHECK_IS_2(check_type) \
    template<typename T> struct is_##check_type : public std::false_type {}; \
    template<typename Ta, typename Tb> \
    struct is_##check_type<check_type<Ta, Tb>> : public std::true_type {};

#define DEF_IS_2(check_type, return_type) \
    template<class T> \
    typename std::enable_if<is_##check_type<T>::value, return_type>::type

CHECK_IS_2(unordered_map);

DEF_IS_2(unordered_map, bool) is_type(PyObject* obj) {
    return PyDict_CheckExact(obj);
}

DEF_IS_2(unordered_map, PyObject*) to_py_object(const T& a) {
    PyObjHolder dict(PyDict_New());
    for (const auto& kv : a) {
        PyObjHolder key(to_py_object<typename T::key_type>(kv.first));
        PyObjHolder value(to_py_object<typename T::mapped_type>(kv.second));
        PyDict_SetItem(dict.obj, key.obj, value.obj);
    }
    return dict.release();
}

DEF_IS_2(unordered_map, T) from_py_object(PyObject* obj) {
    auto size = Py_SIZE(obj);
    T a;
    a.reserve(size);
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(obj, &pos, &key, &value)) {
        CHECK(is_type<typename T::key_type>(key)
            && is_type<typename T::mapped_type>(value));
        a.emplace(
            from_py_object<typename T::key_type>(key), 
            from_py_object<typename T::mapped_type>(value)
        );
    }
    return a;
}

// copy from unordered_map
CHECK_IS_2(map);

DEF_IS_2(map, bool) is_type(PyObject* obj) {
    return PyDict_CheckExact(obj);
}

DEF_IS_2(map, PyObject*) to_py_object(const T& a) {
    PyObjHolder dict(PyDict_New());
    for (const auto& kv : a) {
        PyObjHolder key(to_py_object<typename T::key_type>(kv.first));
        PyObjHolder value(to_py_object<typename T::mapped_type>(kv.second));
        PyDict_SetItem(dict.obj, key.obj, value.obj);
    }
    return dict.release();
}

DEF_IS_2(map, T) from_py_object(PyObject* obj) {
    T a;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(obj, &pos, &key, &value)) {
        CHECK(is_type<typename T::key_type>(key)
            && is_type<typename T::mapped_type>(value));
        a.emplace(
            from_py_object<typename T::key_type>(key), 
            from_py_object<typename T::mapped_type>(value)
        );
    }
    return a;
}


CHECK_IS_1(fast_shared_ptr);

DEF_IS_1(fast_shared_ptr, bool) is_type(PyObject* obj) {
    return is_type<typename T::value_type>(obj);
}

DEF_IS_1(fast_shared_ptr, PyObject*) to_py_object(const T& a) {
    if (a)
        return to_py_object<typename T::value_type>(a.data());
    return to_py_object<typename T::value_type>(a);
}

DEF_IS_1(fast_shared_ptr, T) from_py_object(PyObject* obj) {
    return from_py_object<typename T::value_type>(obj);
}

DEF_IS(NumpyFunc, T) from_py_object(PyObject* obj) {
    // PyObject_Call
    Py_INCREF(obj);
    T func(
        // callback
        [obj](typename T::R* result) {
            // import numpy
            string npstr="numpy";
            #ifdef HAS_CUDA
            if (use_cuda) npstr="cupy";
            #endif

            PyObjHolder np(PyImport_ImportModule(npstr.data()));
            // data = {}
            PyObjHolder data(to_py_object(result->varrays));
            PyObjHolder data2(to_py_object(result->ints));
            PyObjHolder data3(to_py_object(result->arrays));
            PyDict_Update(data.obj, data2.obj);
            PyDict_Update(data.obj, data3.obj);

            // args = []
            PyObjHolder args(PyTuple_New(2));
            PyTuple_SET_ITEM(args.obj, 0, np.release());
            PyTuple_SET_ITEM(args.obj, 1, data.release());

            #ifdef HAS_CUDA
            if (npstr=="cupy") {
                PyObjHolder jt(PyImport_ImportModule("jittor"));
                PyObjHolder pFunc(PyObject_GetAttrString(jt.obj,"numpy2cupy"));
                PyObjHolder ret1(PyObject_Call(pFunc.obj, args.obj, nullptr));
            }
            #endif

            PyObjHolder ret2(PyObject_Call(obj, args.obj, nullptr));
        },
        // deleter
        [obj]() { Py_DECREF(obj); },
        // inc_ref
        [obj]() { Py_INCREF(obj); }
    );
    return func;
}


struct GradCallback;

DEF_IS(GradCallback, bool) is_type(PyObject* obj) {
    return PyCallable_Check(obj);
}

DEF_IS(GradCallback, T) from_py_object(PyObject* obj) {
    // PyObject_Call
    Py_INCREF(obj);
    T func(
        // callback
        [obj](int n_o, typename T::Var** douts, int n_i, typename T::VarPtr* dins) {
            PyObjHolder list(PyTuple_New(n_o));
            for (int i=0; i<n_o; i++) {
                if (douts[i]) {
                    PyTuple_SET_ITEM(list.obj, i, 
                        to_py_object(new typename T::VarHolder(douts[i])));
                } else {
                    Py_INCREF(Py_None);
                    PyTuple_SET_ITEM(list.obj, i, Py_None);
                }
            }

            PyObjHolder ret(PyObject_Call(obj, list.obj, nullptr));
            auto is_seq = PyList_CheckExact(ret.obj) || PyTuple_CheckExact(ret.obj);
            auto check = [&](int i, PyObject* obj) {
                if (obj == Py_None) {
                    dins[i] = nullptr;
                } else {
                    CHECK(Py_TYPE(obj) == &PyjtVarHolder.ht_type) << "returned grad("<<Py_TYPE(obj)->tp_name<<") is not jittor variable";
                    auto vh = from_py_object<typename T::VarHolderPtr>(obj);
                    dins[i] = vh->var;
                }
            };
            if (!is_seq) {
                CHECKop(n_i,==,1) << n_i >> " returned grad required, but 1 given.";
                check(0, ret.obj);
            } else {
                auto size = Py_SIZE(ret.obj);
                CHECKop(n_i,==,size) << n_i >> " returned grad required, but " >> size >> " given.";
                auto arr = PySequence_Fast_ITEMS(ret.obj);
                for (int i=0; i<size; i++) {
                    auto oi = arr[i]; 
                    check(i, oi);
                }
            }
        },
        // deleter
        [obj]() { 
            Py_DECREF(obj); 
        }
    );
    return func;
}

struct VarSlices;
// Slice
DEF_IS(VarSlices, bool) is_type(PyObject* obj) {
    return PyTuple_CheckExact(obj) || 
        PyLong_CheckExact(obj) || 
        PySlice_Check(obj) || 
        (Py_TYPE(obj) == &PyEllipsis_Type) ||
        obj == Py_None ||
        PyUnicode_CheckExact(obj) || 
        is_type<VarHolder*>(obj);
}

template<class T>
void load_var_slice(PyObject* obj, T* var_slice, vector<unique_ptr<VarHolder>>& holders) {
    if (PyLong_CheckExact(obj)) {
        var_slice->set_int(PyLong_AsLong(obj));
    } else
    if (PySlice_Check(obj)) {
        var_slice->slice = from_py_object<decltype(var_slice->slice)>(obj);
    } else
    if (Py_TYPE(obj) == &PyEllipsis_Type) {
        var_slice->set_ellipsis();
    } else 
    if (PyUnicode_CheckExact(obj)) {
        var_slice->set_str(from_py_object<string>(obj));
    } else 
    if (obj == Py_None) {
        var_slice->set_none();
    } else
    if (PyObject_TypeCheck(obj, PyNumberArrType_Type)) {
        PyArrayDescr_Proxy array_descr;
        array_descr.type_num = 5; // 5: int32
        int value;
        PyArray_CastScalarToCtype(obj, &value, &array_descr);
        var_slice->set_int(value);
    } else {
        holders.emplace_back();
        auto* vh = from_py_object<VarHolder*>(obj, holders.back());
        auto vv = (decltype(var_slice->var)*)vh;
        CHECK(vv[0]->dtype() != ns_bool) << "Please convert bool slice into jt.array, example:\n"
            "a[[True,False,False]] ---> a[jt.array([True,False,False])";
        var_slice->set_var(vv[0]);
    }
}

DEF_IS(VarSlices, T) from_py_object(PyObject* obj, vector<unique_ptr<VarHolder>>& holders) {
    if (PyTuple_CheckExact(obj)) {
        auto size = Py_SIZE(obj);
        T vs(size);
        auto arr = PySequence_Fast_ITEMS(obj);
        for (int i=0; i<size; i++) {
            auto oi = arr[i]; 
            load_var_slice(oi, vs.slices+i, holders);
        }
        return vs;
    } else {
        T vs(1);
        load_var_slice(obj, vs.slices, holders);
        return vs;
    }
}


} // jittor
