// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <Python.h>
#include <dlfcn.h>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <iostream>

namespace jittor {

typedef int8_t int8;
typedef int16_t int16;
typedef int int32;
typedef int64_t int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef float float32;
typedef double float64;
typedef uint32_t uint;

using string = std::string;
using std::move;
template <class T> using vector = std::vector<T>;
template <class T> using list = std::list<T>;
template <class T> using set = std::set<T>;
template <class T> using shared_ptr = std::shared_ptr<T>;
template <class T> using unique_ptr = std::unique_ptr<T>;
template <class T> using unordered_set = std::unordered_set<T>;
template <class Ta, class Tb> using pair = std::pair<Ta,Tb>;
template <class Ta, class Tb> using map = std::map<Ta,Tb>;
template <class Ta, class Tb> using unordered_map = std::unordered_map<Ta,Tb>;

#define JT_CHECK(cond) \
    if (!(cond)) throw std::runtime_error("JT_CHECK failed: " #cond " "); 

struct PyObjHolder {

PyObject* obj;
inline PyObjHolder() : obj(nullptr) {
}
inline void assign(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw std::runtime_error("Python Error Occurred.");
    }
    this->obj = obj;
}
inline PyObjHolder(PyObject* obj) : obj(obj) {
    if (!obj) {
        PyErr_Print();
        throw std::runtime_error("Python Error Occurred.");
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

inline void free() {
    if (obj) Py_DECREF(obj);
    obj = nullptr;
}

};

inline std::ostream& operator<<(std::ostream& os, PyObjHolder& objp) {
    PyObjHolder repr_obj(PyObject_Repr(objp.obj));
    
    if (PyUnicode_CheckExact(repr_obj.obj)) {
        return os << Py_TYPE(objp.obj)->tp_name << ' ' <<
             PyUnicode_AsUTF8(repr_obj.obj);
    } else {
        return os << "unknown(" << (void*)objp.obj << ")";
    }
}


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
    JT_CHECK(s);
    return string(s, size);
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

DEF_IS(uint32, bool) is_type(PyObject* obj) {
    return PyLong_CheckExact(obj);
}

DEF_IS(uint32, PyObject*) to_py_object(const T& a) {
    return PyLong_FromUnsignedLong(a);
}

DEF_IS(uint32, T) from_py_object(PyObject* obj) {
    return PyLong_AsUnsignedLong(obj);
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
DEF_IS(int32, bool) is_type(PyObject* obj) {
    return PyLong_CheckExact(obj);
}

DEF_IS(int32, PyObject*) to_py_object(const T& a) {
    return PyLong_FromLong(a);
}

DEF_IS(int32, T) from_py_object(PyObject* obj) {
    return PyLong_AsLong(obj);
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
DEF_IS(float32, bool) is_type(PyObject* obj) {
    return PyFloat_CheckExact(obj) || PyLong_CheckExact(obj);
}

DEF_IS(float32, PyObject*) to_py_object(const T& a) {
    return PyFloat_FromFloat(a);
}

DEF_IS(float32, T) from_py_object(PyObject* obj) {
    if (PyFloat_CheckExact(obj))
        return PyFloat_AS_DOUBLE(obj);
    return PyFloat_AS_DOUBLE(obj);
}


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
        JT_CHECK(o);
        // PyList_SET_ITEM borrow ownership, we do not hold this
        PyList_SET_ITEM(list.obj, i, o);
    }
    return list.release();
}

DEF_IS_1(vector, PyObject*) to_py_tuple(const T& a) {
    PyObjHolder list(PyTuple_New(a.size()));
    for (uint i=0; i<a.size(); i++) {
        PyObject* o = to_py_object<typename T::value_type>(a[i]);
        JT_CHECK(o);
        // PyTuple_SET_ITEM borrow ownership, we do not hold this
        PyTuple_SET_ITEM(list.obj, i, o);
    }
    return list.release();
}

DEF_IS_1(vector, PyObject*) to_py_object(T&& a) {
    PyObjHolder list(PyList_New(a.size()));
    for (uint i=0; i<a.size(); i++) {
        PyObject* o = to_py_object<typename T::value_type>(std::move(a[i]));
        JT_CHECK(o);
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
        JT_CHECK(is_type<typename T::value_type>(oi));
        a[i] = from_py_object<typename T::value_type>(oi);
    }
    return a;
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
        JT_CHECK(is_type<typename T::key_type>(key)
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
        JT_CHECK(is_type<typename T::key_type>(key)
            && is_type<typename T::mapped_type>(value));
        a.emplace(
            from_py_object<typename T::key_type>(key), 
            from_py_object<typename T::mapped_type>(value)
        );
    }
    return a;
}

template<class T, int N>
struct array {

typedef  T _type;
static constexpr int _ndim = N;

int64 shape[N];
unique_ptr<T[]> data;

inline bool is_float() const { return std::is_floating_point<T>::value; }
inline bool is_unsigned() const { return std::is_unsigned<T>::value; }
inline int64 size() const {
    int64 s=1;
    for (auto x : shape) s *= x;
    return s;
}
inline int64 nbyte() const { return size()*sizeof(T); }
inline string dtype() const {
    return DTYPE();
}
inline int ndim() const { return N; }

inline static string DTYPE() {
    string dtype(std::is_floating_point<T>::value ? "float" : 
        std::is_unsigned<T>::value ? "uint" : "int");
    if (sizeof(T)==1) dtype += "8"; else
    if (sizeof(T)==2) dtype += "16"; else
    if (sizeof(T)==4) dtype += "32"; else
    if (sizeof(T)==8) dtype += "64"; else
        throw std::runtime_error("Not support type");
    return dtype;
} 

inline array(const vector<int64>& shape) {
    if (shape.size() != N) throw std::runtime_error("Dim not match");
    for (int i=0; i<N; i++) this->shape[i] = shape[i];
    data.reset(new T[size()]);
}

inline array(const vector<int64>& shape, const T* data) : array(shape) {
    memcpy(this->data.get(), data, nbyte());
}

inline array(const vector<int64>& shape, const vector<T>& data) : array(shape, &data[0]) {
}

template<int I, class Ti, typename... Targs>
inline int64 get_offset(int64 offset, Ti i, Targs... Fargs) {
    if constexpr (I+1==N)
        return offset*shape[I]+i;
    else
        return get_offset<I+1>(offset*shape[I]+i, Fargs...);
}

template<typename... Targs>
T& operator()(Targs... Fargs) {
    return data[get_offset<0>(0, Fargs...)];
}

};

struct Console {

PyObjHolder globals, locals;
PyObject* (*make_pyjt_array)(const vector<int64>& shape, const string& dtype, const void* data);
void (*get_pyjt_array)(PyObject* obj, vector<int64>& shape, string& dtype, void*& data);

inline Console() {
    Py_Initialize();
    globals.assign(PyDict_New());
    locals.assign(PyDict_New());

    #if PY_VERSION_HEX < 0x03080000
    PyObjHolder builtins(PyImport_ImportModule("builtins"));
    PyDict_SetItemString(globals.obj, "__builtins__", builtins.obj);
    #endif

    run("import jittor as jt");
    make_pyjt_array = (PyObject* (*)(const vector<int64>& shape, const string& dtype, const void* data))dlsym(RTLD_DEFAULT, "_ZN6jittor15make_pyjt_arrayERKSt6vectorIlSaIlEERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKv");
    get_pyjt_array = (void (*)(PyObject* obj, vector<int64>& shape, string& dtype, void*& data))dlsym(RTLD_DEFAULT, "_ZN6jittor14get_pyjt_arrayEP7_objectRSt6vectorIlSaIlEERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERPv");
}

inline ~Console() {
    globals.free();
    locals.free();
    Py_FinalizeEx();
}

inline void run(const char* src) {
    PyObjHolder ret(PyRun_String(src, Py_file_input, globals.obj, nullptr));
}

inline void run(const string& src) { run(src.c_str()); }

template<class T>
inline void set(const char* s, const T& data) {
    PyObjHolder py_data(to_py_object<T>(data));
    PyDict_SetItemString(globals.obj, s, py_data.obj);
}

template<class T>
inline void set(const string& s, const T& data) {
    set(s.c_str(), data);
}

template<class T>
inline T get(const char* s) {
    auto obj = PyDict_GetItemString(globals.obj, s);
    if (!obj) obj = PyDict_GetItemString(globals.obj, s);
    if (!obj) throw std::runtime_error(string("KeyError: ")+s);
    if (!is_type<T>(obj)) throw std::runtime_error(string("TypeError: key<")+s+"> is "+Py_TYPE(obj)->tp_name);
    return from_py_object<T>(obj);
};

template<class T>
inline T get(const string& s) {
    return get<T>(s.c_str());
}



template<class T, int N>
inline void set_array(const string& s, const array<T,N>& data) {
    PyObjHolder obj(make_pyjt_array(
        vector<int64>(data.shape, data.shape+N),
        data.dtype(),
        data.data.get()));
    PyDict_SetItemString(globals.obj, s.c_str(), obj.obj);
}

template<class T, int N>
inline array<T,N> get_array(const string& s) {
    auto obj = PyDict_GetItemString(globals.obj, s.c_str());
    if (!obj) obj = PyDict_GetItemString(globals.obj, s.c_str());
    if (!obj) throw std::runtime_error(string("KeyError: ")+s);
    vector<int64> shape;
    string dtype;
    void* data;
    get_pyjt_array(obj, shape, dtype, data);
    string dtype2 = array<T,N>::DTYPE();
    if (dtype2 != dtype)
        throw new std::runtime_error(string("array dtype not match: ")+dtype+"!="+dtype2);
    if (shape.size() != N)
        throw new std::runtime_error(string("array ndim not match: ")+std::to_string(shape.size())+"!="+std::to_string(N));
    return array<T, N>(shape, (T*)data);
}

};

}