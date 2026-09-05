// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
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
#include "type/nano_string.h"
#include "misc/fast_shared_ptr.h"
#include "profiler/simple_profiler.h"
#ifdef IS_CUDA
#include "runtime/device.h"
#endif

namespace jittor {

template<class T>
struct vector_to_tuple {
    typedef T value_type;
    vector<T> x;
    vector_to_tuple(vector<T>&& _) :x(move(_)) {}
};

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

#ifdef __linux__
// int64_t
DEF_IS(int64_t, bool) is_type(PyObject* obj) {
    return PyLong_CheckExact(obj);
}

DEF_IS(int64_t, PyObject*) to_py_object(const T& a) {
    return PyLong_FromLongLong(a);
}

DEF_IS(int64_t, T) from_py_object(PyObject* obj) {
    return PyLong_AsLongLong(obj);
}
#endif

#ifdef __APPLE__
// uint64
DEF_IS(uint64, bool) is_type(PyObject* obj) {
    return PyLong_CheckExact(obj);
}

DEF_IS(uint64, PyObject*) to_py_object(const T& a) {
    return PyLong_FromUnsignedLongLong(a);
}

DEF_IS(uint64, T) from_py_object(PyObject* obj) {
    return PyLong_AsUnsignedLongLong(obj);
}
#endif

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
    // PySlice_Unpack returns -1 and leaves all three outputs UNWRITTEN when the
    // slice cannot be used -- step == 0, or a bound whose __index__ raises.  The
    // return value therefore has to be checked before the values are read, or
    // uninitialised stack becomes the slice bounds handed to getitem/setitem.
    Py_ssize_t start = 0, stop = 0, step = 1;
    auto slice = (PySliceObject*)obj;

    if (PySlice_Unpack(obj, &start, &stop, &step) < 0) {
        // The Python exception (ValueError / whatever __index__ raised) is
        // already set; the generated binding's catch block forwards an
        // existing Python error unchanged, so it reaches the caller as-is.
        throw std::runtime_error("invalid slice");
    }
    return {start, stop, step, 
        (slice->start == Py_None) |
        ((slice->stop == Py_None) << 1) |
        ((slice->step == Py_None) << 2)};
}

// PyLong_AsLong returns a 64-bit long on LP64, and the generated bindings
// assigned it straight into whatever width the C++ parameter has.  A value
// that does not fit was therefore truncated in silence: `x.sum(dim=2**40)`
// became `dim=0` and reduced over the wrong axis.  These conversions report
// the overflow the way CPython does, and the binding's PyErr_Occurred() check
// turns it into an OverflowError.
inline int32 pyjt_as_int32(PyObject* obj) {
    int64 v = PyLong_AsLongLong(obj);
    if (v == -1 && PyErr_Occurred()) return -1;
    if (v < (int64)(-2147483647-1) || v > (int64)2147483647) {
        PyErr_SetString(PyExc_OverflowError,
            "Python int too large to convert to C int32");
        return -1;
    }
    return (int32)v;
}

inline uint32 pyjt_as_uint32(PyObject* obj) {
    uint64 v = PyLong_AsUnsignedLongLong(obj);
    if (v == (uint64)-1 && PyErr_Occurred()) return 0;
    if (v > (uint64)4294967295u) {
        PyErr_SetString(PyExc_OverflowError,
            "Python int too large to convert to C uint32");
        return 0;
    }
    return (uint32)v;
}

inline uint16 pyjt_as_uint16(PyObject* obj) {
    uint64 v = PyLong_AsUnsignedLongLong(obj);
    if (v == (uint64)-1 && PyErr_Occurred()) return 0;
    if (v > (uint64)65535u) {
        PyErr_SetString(PyExc_OverflowError,
            "Python int too large to convert to C uint16");
        return 0;
    }
    return (uint16)v;
}

inline uint8 pyjt_as_uint8(PyObject* obj) {
    uint64 v = PyLong_AsUnsignedLongLong(obj);
    if (v == (uint64)-1 && PyErr_Occurred()) return 0;
    if (v > (uint64)255u) {
        PyErr_SetString(PyExc_OverflowError,
            "Python int too large to convert to C uint8");
        return 0;
    }
    return (uint8)v;
}

#define GET_RAW_PTR(T, obj) ((T*)(((char*)obj) + sizeof(PyObject)))
#define GET_OBJ_FROM_RAW_PTR(obj) ((PyObject*)(((char*)obj) - sizeof(PyObject)))
#define GET_OBJ_SIZE(T) (sizeof(PyObject)+sizeof(T))

// Instance layout of a pyjt-generated type:
//     [PyObject header][T][instance dict pointer, if the type has one][inited]
// The trailing "inited" word exists only for types that declare
// `@pyjt(__dealloc__)`; pyjt_compiler.py grows tp_basicsize for it.
//
// It is needed because tp_new is PyType_GenericNew, which only zeroes the
// instance: the C++ object is constructed later, by tp_init.  When tp_init
// finds no matching overload it returns -1 and CPython goes straight to
// tp_dealloc, so without this flag the generated dealloc would run a C++
// destructor over storage on which no constructor ever ran -- one line of
// `jittor_core.RingBuffer()` used to segfault exactly that way.  tp_init sets
// the flag once construction succeeded, tp_dealloc checks it first.
//
// Handwritten to_py_object paths below build their instance with
// _PyObject_New, whose memory is NOT zeroed, so they must set the flag
// themselves right after the placement new.
#define GET_INITED_FLAG(T, has_dict, obj) \
    (*(uint64*)(((char*)(obj)) + sizeof(PyObject) + sizeof(T) + \
        ((has_dict) ? sizeof(PyObject*) : 0)))

// DumpGraphs
struct DumpGraphs;
EXTERN_LIB PyTypeObject PyjtDumpGraphs;
DEF_IS(DumpGraphs, bool) is_type(PyObject* obj) {
    return Py_TYPE(obj) == &PyjtDumpGraphs;
}


DEF_IS(DumpGraphs, PyObject*) to_py_object(T&& a) {
    PyObjHolder obj(_PyObject_New(&PyjtDumpGraphs));
    auto ptr = GET_RAW_PTR(T, obj.obj);
    new (ptr) T();
    ptr->hold_vars = std::move(a.hold_vars);
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
EXTERN_LIB PyTypeObject PyjtMemInfo;
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

// MemInfo
struct ZipFile;
EXTERN_LIB PyTypeObject PyjtZipFile;
DEF_IS(ZipFile, bool) is_type(PyObject* obj) {
    return Py_TYPE(obj) == &PyjtZipFile;
}


DEF_IS(ZipFile, PyObject*) to_py_object(const T& a) {
    PyObjHolder obj(_PyObject_New(&PyjtZipFile));
    auto ptr = GET_RAW_PTR(T, obj.obj);
    new (ptr) T(a);
    GET_INITED_FLAG(T, 0, obj.obj) = 1;
    return obj.release();
}

DEF_IS(ZipFile, const T&) from_py_object(PyObject* obj) {
    return GET_RAW_PTR(T, obj);
}


// NanoString
struct NanoString;
EXTERN_LIB PyTypeObject PyjtNanoString;
// Every branch here ends in ns_valid_name(): a NanoString parameter only
// matches an object that actually names a dtype or an operator.
//
// The check used to be "is it a str, a type, ANY callable, or does it have a
// `.type` attribute", which matched almost everything.  Two consequences:
// an arbitrary function or class won the NanoString overload and then failed
// inside the conversion, and `PyObject_HasAttrString` ran (and swallowed the
// exceptions of) a user-defined `__getattr__` on every overload probe.  With
// PyTorch's `Tensor.type()` present in the shim, every tensor has a `.type`
// attribute and would be a candidate dtype.
DEF_IS(NanoString, bool) is_type(PyObject* obj) {
    if (Py_TYPE(obj) == &PyjtNanoString) return true;
    // PyUnicode_Check (not CheckExact) so str SUBCLASSES are accepted too:
    // torch_compat's `dtype` is a str subclass whose underlying value is the
    // bare jittor name ("float32"), fed back into NanoString params by
    // jittor's own python (contrib.concat/linalg/nn do str(var.dtype)).
    if (PyUnicode_Check(obj)) {
        auto s = PyUnicode_AsUTF8(obj);
        if (!s) { PyErr_Clear(); return false; }
        return ns_valid_name(s);
    }
    // numpy scalar types (np.float32) and the python builtins (float, int,
    // bool) are spelled as type objects whose name is the dtype.  PyType_Check
    // rather than CheckExact so a class built by a custom metaclass still
    // arrives here instead of falling through to the callable branch, which is
    // how it used to be accepted.
    if (PyType_Check(obj))
        return ns_valid_name(_PyType_Name((PyTypeObject *)obj));
    // jt.float and friends are builtin functions named after the dtype.  Only
    // functions are probed: both kinds carry a real __name__ slot, so no user
    // __getattr__ runs here.
    if (PyFunction_Check(obj) || PyCFunction_Check(obj)) {
        PyObject* n = PyObject_GetAttrString(obj, "__name__");
        if (!n) { PyErr_Clear(); return false; }
        auto s = PyUnicode_Check(n) ? PyUnicode_AsUTF8(n) : nullptr;
        bool ok = s && ns_valid_name(s);
        if (!s) PyErr_Clear();
        Py_DECREF(n);
        return ok;
    }
    // numpy.dtype keeps the scalar type in `.type`.  Restricted to real dtype
    // instances rather than "anything with that attribute name".
    if (PyArrayDescr_Type && PyObject_TypeCheck(obj, PyArrayDescr_Type)) {
        PyObject* t = PyObject_GetAttrString(obj, "type");
        if (!t) { PyErr_Clear(); return false; }
        bool ok = PyType_CheckExact(t) &&
            ns_valid_name(_PyType_Name((PyTypeObject *)t));
        Py_DECREF(t);
        return ok;
    }
    return false;
}

DEF_IS(NanoString, PyObject*) to_py_object(T a) {
    PyObjHolder obj(_PyObject_New(&PyjtNanoString));
    auto ptr = GET_RAW_PTR(T, obj.obj);
    new (ptr) T(a);
    return obj.release();
}

// Kept branch-for-branch in step with is_type above.
DEF_IS(NanoString, T) from_py_object(PyObject* obj) {
    if (Py_TYPE(obj) == &PyjtNanoString)
        return *GET_RAW_PTR(T, obj);
    if (PyUnicode_Check(obj))   // str or str subclass (e.g. torch_compat dtype)
        return T(PyUnicode_AsUTF8(obj));
    // PyType
    if (PyType_Check(obj))
        return T(_PyType_Name((PyTypeObject *)obj));
    // jt.float.__name__
    if (PyFunction_Check(obj) || PyCFunction_Check(obj)) {
        PyObjHolder t(PyObject_GetAttrString(obj, "__name__"));
        return T(PyUnicode_AsUTF8(t.obj));
    }
    PyObjHolder t(PyObject_GetAttrString(obj, "type"));
    CHECK(PyType_CheckExact(t.obj)) << "Not a valid type:" << t.obj;
    return T(_PyType_Name((PyTypeObject *)t.obj));
}

// NanoVector
struct NanoVector;
EXTERN_LIB PyTypeObject PyjtNanoVector;
DEF_IS(NanoVector, bool) is_type(PyObject* obj) {
    // Accept tuple/list SUBCLASSES too (e.g. torch.Size from the torch-compat
    // shim is `class Size(tuple)`). Exact-only checks raised on shape compares
    // (`var.shape != torch.Size(...)`) and rejected torch.Size as a shape arg.
    // Subclasses share the tuple/list C layout, so PySequence_Fast_ITEMS works.
    return Py_TYPE(obj) == &PyjtNanoVector ||
        PyList_Check(obj) || PyTuple_Check(obj);
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
EXTERN_LIB PyHeapTypeObject PyjtVarHolder;
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
#if defined(__linux__) || defined(_WIN32)
    STACK_ALLOC(int64_t, dims, a.shape.size());
#elif defined(__APPLE__)
    long dims[a.shape.size()];
#endif
    for (int i=0; i<a.shape.size(); i++)
        dims[i] = a.shape[i];
    PyObjHolder obj(PyArray_SimpleNew(
        a.shape.size(),
        dims,
        get_typenum(a.dtype == ns_bfloat16 ? ns_float32 : a.dtype)
    ));
    auto arr = (PyArray_Proxy*)(obj.obj);
    int64 size = PyArray_Size(arr);
    if (a.dtype == ns_bfloat16) {
        // simple cast bfloat16 to float32
        auto ptr = (uint16*)a.ptr;
        auto ptr2 = (uint32*)arr->data;
        int64 num = size/4;
        for (int64 i=0; i<num; i++) {
            ptr2[i] = ptr[i]<<16;
        }
    } else {
        memcpy((void*)arr->data, (void*)a.ptr, size);
    }
    return obj.release();
}

// A scalar converted here may be held by the caller across arbitrary work --
// `Var.data = 2.0` syncs the graph before it copies, and a numpy_code/fetch
// callback running in that sync can convert scalars of its own.  So the
// ArrayArgs owns its buffer instead of pointing at shared storage.
template<class T2>
inline void _fill_scalar_array_args(ArrayArgs& args, T2 value, NanoString dtype) {
    args.buffer.reset(new char[sizeof(T2)]);
    *(T2*)args.buffer.get() = value;
    args.ptr = args.buffer.get();
    args.shape.push_back(1);
    args.dtype = dtype;
}

DEF_IS(ArrayArgs, T) from_py_object(PyObject* obj) {
    if (PyFloat_CheckExact(obj)) {
        T args;
        _fill_scalar_array_args(args, (float32)PyFloat_AS_DOUBLE(obj), ns_float32);
        return args;
    }
    if (PyLong_CheckExact(obj)) {
        T args;
        _fill_scalar_array_args(args, (int32)PyLong_AsLong(obj), ns_int32);
        return args;
    }
    if (PyBool_Check(obj)) {
        T args;
        _fill_scalar_array_args(args, (int8)(obj == Py_True), ns_bool);
        return args;
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
EXTERN_LIB PyHeapTypeObject PyjtVarHolder;
void schedule_pending_from_python(VarHolder* holder);
namespace jit_op_maker { 
EXTERN_LIB VarHolder* array_(ArrayArgs&&);
EXTERN_LIB VarHolder* array__(PyObject* obj);
}
DEF_IS(VarHolder*, bool) is_type(PyObject* obj) {
    return Py_TYPE(obj) == &PyjtVarHolder.ht_type ||
        is_type<ArrayArgs>(obj);
}

DEF_IS(VarHolder*, PyObject*) to_py_object(T a) {
    // tp_alloc, not _PyObject_New: VarHolder is a GC type (its instance dict
    // can close a reference cycle), and GC instances need the collector's
    // header in front of them plus a place on the tracked list.  tp_alloc also
    // zeroes the storage, so the dict slot and the inited flag below start out
    // null even if a collection runs before they are filled in.
    auto vh_type = &PyjtVarHolder.ht_type;
    PyObjHolder obj(vh_type->tp_alloc(vh_type, 0));
    auto ptr = GET_RAW_PTR(T, obj.obj);
    ((PyObject**)(((char*)obj.obj) + sizeof(PyObject) + sizeof(typename std::remove_pointer<T>::type)))[0] = PyDict_New();
    // new attr_dict
    // will move and delete a
    new (ptr) typename std::remove_pointer<T>::type (a);
    GET_INITED_FLAG(typename std::remove_pointer<T>::type, 1, obj.obj) = 1;
    schedule_pending_from_python(
        reinterpret_cast<typename std::remove_pointer<T>::type*>(ptr));
    return obj.release();
}


DEF_IS(VarHolder*, T) from_py_object(PyObject* obj) {
    CHECK(Py_TYPE(obj) == &PyjtVarHolder.ht_type);
    return GET_RAW_PTR(VarHolder, obj);
}

DEF_IS(VarHolder*, T) from_py_object(PyObject* obj, unique_ptr<VarHolder>& holder) {
    if (Py_TYPE(obj) == &PyjtVarHolder.ht_type)
        return GET_RAW_PTR(VarHolder, obj);
    holder.reset(jit_op_maker::array__(obj));
    return holder.get();
}

struct DataView;
struct VarHolder;
EXTERN_LIB PyObject* new_var_data_owner(VarHolder* vh);
DEF_IS(DataView, PyObject*) to_py_object(T a) {
#if defined(__linux__) || defined(_WIN32)
    STACK_ALLOC(int64_t, dims, a.shape.size());
#elif defined(__APPLE__)
    long dims[a.shape.size()];
#endif
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
        // The base must own the *allocation*, not only the python wrapper:
        // see new_var_data_owner in var_holder.h.
        PyObjHolder oh2(new_var_data_owner(a.vh));
        ASSERT(PyArray_SetBaseObject(oh.obj, oh2.obj)==0);
        oh2.release();
    }
    return oh.release();
}

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
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
    // Unsigned dtypes used to fall through to the int64 branch below, which
    // reads all 8 bytes of the payload while item() only wrote dsize of them.
    if (a.dtype == ns_uint8)
        return PyLong_FromUnsignedLongLong((uint64)*(uint8*)&a.data);
    if (a.dtype == ns_uint16)
        return PyLong_FromUnsignedLongLong((uint64)*(uint16*)&a.data);
    if (a.dtype == ns_uint32)
        return PyLong_FromUnsignedLongLong((uint64)*(uint32*)&a.data);
    if (a.dtype == ns_uint64)
        return PyLong_FromUnsignedLongLong(*(uint64*)&a.data);
    ASSERT(a.dtype == ns_int64) << "Unhandled dtype in item():" << a.dtype;
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


#define CHECK_IS_2(check_type) \
    template<typename T> struct is_##check_type : public std::false_type {}; \
    template<typename Ta, typename Tb> \
    struct is_##check_type<check_type<Ta, Tb>> : public std::true_type {};

#define DEF_IS_2(check_type, return_type) \
    template<class T> \
    typename std::enable_if<is_##check_type<T>::value, return_type>::type

CHECK_IS_1(vector);
CHECK_IS_1(vector_to_tuple);

CHECK_IS_2(map);
DEF_IS_2(map, bool) is_type(PyObject* obj);
DEF_IS_2(map, PyObject*) to_py_object(const T& a);

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

DEF_IS_1(vector_to_tuple, PyObject*) to_py_object(const T& a) {
    PyObjHolder list(PyTuple_New(a.x.size()));
    for (uint i=0; i<a.x.size(); i++) {
        PyObject* o = to_py_object<typename T::value_type>(a.x[i]);
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

struct SimpleFunc;

DEF_IS(SimpleFunc, bool) is_type(PyObject* obj) {
    return PyCallable_Check(obj);
}

DEF_IS(SimpleFunc, T) from_py_object(PyObject* obj) {
    // PyObject_Call
    Py_INCREF(obj);
    T func(
        // callback
        [obj](int64 result) {
            // check python version macro >= 3.9
            #if PY_VERSION_HEX >= 0x03090000
            PyObjHolder args(to_py_object(result));
            PyObjHolder ret(PyObject_CallOneArg(obj, args.obj));
            #else
            LOGf << "Not supported python version";
            #endif
        },
        // deleter
        [obj]() { Py_DECREF(obj); }
    );
    return func;
}

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
// CHECK_IS_2(map);

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

CHECK_IS_1(Maybe);

DEF_IS_1(Maybe, bool) is_type(PyObject* obj) {
    return obj == Py_None || 
        is_type<typename T::value_type*>(obj);
}

DEF_IS_1(Maybe, PyObject*) to_py_object(T a) {
    if (a)
        return to_py_object<typename T::value_type*>(a.ptr);
    Py_INCREF(Py_None);
    return Py_None;
}

DEF_IS_1(Maybe, T) from_py_object(PyObject* obj) {
    if (obj == Py_None) return T();
    return T(from_py_object<typename T::value_type*>(obj));
}

DEF_IS(NumpyFunc, T) from_py_object(PyObject* obj) {
    // PyObject_Call
    Py_INCREF(obj);
    T func(
        // callback
        [obj](typename T::R* result) {
            // import numpy
            string npstr="numpy";
            #ifdef IS_CUDA
            if (runtime_use_cuda()) npstr="cupy";
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

            #ifdef IS_CUDA
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
        // numpy scalar index (np.int64/np.int32/np.float64/np.bool_ ...).
        // Old code fabricated a numpy-1.x PyArrayDescr_Proxy on the stack and
        // called PyArray_CastScalarToCtype, which faults on numpy>=2 because the
        // PyArray_Descr layout changed. All numpy number scalars support
        // PyNumber_Long, which matches the old int32-cast semantics (float
        // scalars truncate toward zero like numpy-1.x did), without depending on
        // numpy's internal Descr ABI.
        PyObjHolder l(PyNumber_Long(obj));
        var_slice->set_int(PyLong_AsLong(l.obj));
    } else {
        holders.emplace_back();
        auto* vh = from_py_object<VarHolder*>(obj, holders.back());
        auto vv = (decltype(var_slice->var)*)vh;
        USER_CHECK(vv[0]->dtype() != ns_bool) << "Please convert bool slice into jt.array, example:\n"
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

EXTERN_LIB bool check_async_executor_error(const std::exception& e, std::ostream& os);

} // jittor
