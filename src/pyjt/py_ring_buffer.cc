// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "pyjt/py_ring_buffer.h"
#include "pyjt/py_obj_holder.h"
#include "pyjt/py_converter.h"
#include "ops/array_op.h"
#include "var_holder.h"

namespace jittor {

static void push_py_object_pickle(RingBuffer* rb, PyObject* obj, uint64& __restrict__ offset) {
    PyObjHolder pickle(PyImport_ImportModule("pickle"));
    PyObjHolder dumps(PyObject_GetAttrString(pickle.obj, "dumps"));
    PyObjHolder proto(PyObject_GetAttrString(pickle.obj, "HIGHEST_PROTOCOL"));
    rb->push_t<uint8>(6, offset);
    PyObjHolder ret(PyObject_CallFunctionObjArgs(dumps.obj, obj, proto.obj, nullptr));
    obj = ret.obj;
    Py_ssize_t size;
    char* s;
    ASSERT(0 == PyBytes_AsStringAndSize(ret.obj, &s, &size));
    rb->push_t<int64>(size, offset);
    rb->push(size, offset);
    // LOGir << string(rb->get_ptr(size, offset), size);
    std::memcpy(rb->get_ptr(size, offset), s, size);
    return;
}

static PyObject* pop_py_object_pickle(RingBuffer* rb, uint64& __restrict__ offset) {
    PyObjHolder pickle(PyImport_ImportModule("pickle"));
    PyObjHolder loads(PyObject_GetAttrString(pickle.obj, "loads"));

    auto size = rb->pop_t<int64>(offset);
    rb->pop(size, offset);
    PyObjHolder s(PyBytes_FromStringAndSize(rb->get_ptr(size, offset), size));

    PyObjHolder ret(PyObject_CallFunctionObjArgs(loads.obj, s.obj, nullptr));
    return ret.release();
}


static void push_py_object(RingBuffer* rb, PyObject* obj, uint64& __restrict__ offset) {
    if (PyLong_CheckExact(obj)) {
        int64 x = PyLong_AsLongLong(obj);
        rb->push_t<uint8>(0, offset);
        rb->push_t<int64>(x, offset);
        return;
    }
    if (PyFloat_CheckExact(obj)) {
        float64 x = PyFloat_AS_DOUBLE(obj);
        rb->push_t<uint8>(1, offset);
        rb->push_t<float64>(x, offset);
        return;
    }
    if (PyUnicode_CheckExact(obj)) {
        Py_ssize_t size;
        const char* s = PyUnicode_AsUTF8AndSize(obj, &size);
        rb->push_t<uint8>(2, offset);
        rb->push_t<int64>(size, offset);
        rb->push(size, offset);
        std::memcpy(rb->get_ptr(size, offset), s, size);
        return;
    }
    if (PyList_CheckExact(obj) || PyTuple_CheckExact(obj)) {
        rb->push_t<uint8>(3, offset);
        auto size = Py_SIZE(obj);
        auto arr = PySequence_Fast_ITEMS(obj);
        rb->push_t<int64>(size, offset);
        for (int64 i=0; i<size; i++) {
            auto oi = arr[i];
            push_py_object(rb, oi, offset);
        }
        return;
    }
    if (PyDict_CheckExact(obj)) {
        rb->push_t<uint8>(4, offset);
        auto size = Py_SIZE(obj);
        rb->push_t<int64>(size, offset);
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(obj, &pos, &key, &value)) {
            push_py_object(rb, key, offset);
            push_py_object(rb, value, offset);
        }
        return;
    }
    if (Py_TYPE(obj) == &PyjtVarHolder.ht_type ||
        Py_TYPE(obj) == PyArray_Type) {
        ArrayArgs args;
        int64 size=0;
        rb->push_t<uint8>(5, offset);
        if (Py_TYPE(obj) == &PyjtVarHolder.ht_type) {
            auto ptr = GET_RAW_PTR(VarHolder, obj);
            args = move(fetch_sync({ptr}).at(0));
            size = ptr->var->size;
        } else {
            auto arr = (PyArray_Proxy*)obj;
            if (arr->nd)
                args.shape = NanoVector::make(arr->dimensions, arr->nd);
            else
                args.shape.push_back(1);
            args.dtype = get_type_str(arr);
            size = PyArray_Size(arr);
            if (!is_c_style(arr)) {
                rb->push_t<NanoVector>(args.shape, offset);
                rb->push_t<NanoString>(args.dtype, offset);
                rb->push(size, offset);
                args.ptr = rb->get_ptr(size, offset);
                int64 dims[args.shape.size()];
                for (int i=0; i<args.shape.size(); i++)
                    dims[i] = args.shape[i];
                PyObjHolder oh(PyArray_New(
                    PyArray_Type, // subtype
                    args.shape.size(), // nd
                    dims, // dims
                    get_typenum(args.dtype), // type_num
                    NULL, // strides
                    (void*)args.ptr, // data
                    0, // itemsize
                    NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, // flags
                    NULL // obj
                ));
                ASSERT(0==PyArray_CopyInto(oh.obj, obj));
                return;
            } else {
                args.ptr = arr->data;
            }
        }
        rb->push_t<NanoVector>(args.shape, offset);
        rb->push_t<NanoString>(args.dtype, offset);
        rb->push(size, offset);
        std::memcpy(rb->get_ptr(size, offset), args.ptr, size);
        return;
    }
    push_py_object_pickle(rb, obj, offset);
}


static PyObject* to_py_object3(ArrayArgs&& a) {
    return to_py_object(jit_op_maker::array_(move(a)));
}

static PyObject* pop_py_object(RingBuffer* rb, uint64& __restrict__ offset, bool keep_numpy_array) {
    auto t = rb->pop_t<uint8>(offset);
    if (t==0) {
        auto x = rb->pop_t<int64>(offset);
        return PyLong_FromLongLong(x);
    }
    if (t==1) {
        auto x = rb->pop_t<float64>(offset);
        return PyFloat_FromDouble(x);
    }
    if (t==2) {
        auto size = rb->pop_t<int64>(offset);
        rb->pop(size, offset);
        return PyUnicode_FromStringAndSize(rb->get_ptr(size, offset), size);
    }
    if (t==3) {
        auto size = rb->pop_t<int64>(offset);
        PyObjHolder list(PyList_New(size));
        for (uint i=0; i<size; i++) {
            PyObject* o = pop_py_object(rb, offset, keep_numpy_array);
            PyList_SET_ITEM(list.obj, i, o);
        }
        return list.release();
    }
    if (t==4) {
        auto size = rb->pop_t<int64>(offset);
        PyObjHolder dict(PyDict_New());
        for (int64 i=0; i<size; i++) {
            PyObjHolder key(pop_py_object(rb, offset, keep_numpy_array));
            PyObjHolder value(pop_py_object(rb, offset, keep_numpy_array));
            PyDict_SetItem(dict.obj, key.obj, value.obj);
        }
        return dict.release();
    }
    if (t==5) {
        ArrayArgs args;
        args.shape = rb->pop_t<NanoVector>(offset);
        args.dtype = rb->pop_t<NanoString>(offset);
        int64 size = args.dtype.dsize();
        for (int i=0; i<args.shape.size(); i++)
            size *= args.shape[i];
        rb->pop(size, offset);
        args.ptr = rb->get_ptr(size, offset);
        if (!keep_numpy_array)
            return to_py_object3(move(args));
        else
            return to_py_object<ArrayArgs>(args);
    }
    if (t==6) {
        return pop_py_object_pickle(rb, offset);
    }
    if (t == 255) {
        LOGf << "WorkerError:" << rb->pop_string(offset);
    } else
        LOGf << "unsupport type:" << (int)t;
    return nullptr;
}

void PyMultiprocessRingBuffer::push(PyObject* obj) {
    auto offset = rb->r;
    auto offset_bk = offset;
    try {
        push_py_object(rb, obj, offset);
    } catch (const std::exception& e) {
        offset = offset_bk;
        rb->push_t<uint8>(255, offset);
        rb->push_string(string(e.what()), offset);
    }
    rb->commit_push(offset);
}

PyObject* PyMultiprocessRingBuffer::pop() {
    auto offset = rb->l;
    auto obj = pop_py_object(rb, offset, _keep_numpy_array);
    rb->commit_pop(offset);
    return obj;
}

PyMultiprocessRingBuffer::PyMultiprocessRingBuffer(uint64 size) {
    rb = RingBuffer::make_ring_buffer(size, 1);
}

PyMultiprocessRingBuffer::~PyMultiprocessRingBuffer() {
    RingBuffer::free_ring_buffer(rb);
}

}
