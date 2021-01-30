// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "mem/allocator.h"
#include "mem/allocator/cuda_dual_allocator.h"
#include "event_queue.h"
#endif
#include <Python.h>
#include "pyjt/py_obj_holder.h"
#include "pyjt/py_converter.h"
#include "pyjt/numpy.h"
#include "ops/array_op.h"
#include "var.h"
#include "ops/op_register.h"
#include "var_holder.h"

namespace jittor {

static auto make_array = get_op_info("array")
    .get_constructor<VarPtr, const void*, NanoVector, NanoString>();

PyObject* make_pyjt_array(const vector<int64>& shape, const string& dtype, const void* data) {
    // return nullptr;
    auto vh = new VarHolder(make_array(data, shape, dtype));
    return to_py_object<VarHolder*>(vh);
}

void get_pyjt_array(PyObject* obj, vector<int64>& shape, string& dtype, void*& data) {
    CHECK(Py_TYPE(obj) == &PyjtVarHolder.ht_type) << "Not a jittor array" << Py_TYPE(obj);
    auto vh = GET_RAW_PTR(VarHolder, obj);
    if (!vh->var->mem_ptr)
        vh->sync();
    ASSERT(vh->var->mem_ptr);
    shape = vh->shape().to_vector();
    dtype = vh->dtype().to_cstring();
    data = vh->var->mem_ptr;
}

ArrayOp::ArrayOp(PyObject* obj) {
    ArrayArgs args;
    PyObjHolder holder;
    args.ptr = nullptr;
    allocation.ptr = nullptr;
    if (PyFloat_CheckExact(obj)) {
        tmp_data.f32 = PyFloat_AS_DOUBLE(obj);
        args = {&tmp_data, 1, ns_float32};
    } else
    if (PyLong_CheckExact(obj)) {
        tmp_data.i32 = PyLong_AsLong(obj);
        args = {&tmp_data, 1, ns_int32};
    } else
    if (PyBool_Check(obj)) {
        tmp_data.i8 = obj == Py_True;
        args = {&tmp_data, 1, ns_bool};
    } else
    if (Py_TYPE(obj) == &PyjtVarHolder.ht_type) {
        auto ptr = GET_RAW_PTR(VarHolder, obj);
        args = move(fetch_sync({ptr}).at(0));
    } else
    if (Py_TYPE(obj) == PyArray_Type ||
        PyList_CheckExact(obj) || PyTuple_CheckExact(obj) ||
        PyObject_TypeCheck(obj, PyNumberArrType_Type)
    ) {
        if (Py_TYPE(obj) != PyArray_Type) {
            holder.assign(PyArray_FROM_O(obj));
            obj = holder.obj;
        }
        auto arr = (PyArray_Proxy*)obj;
        if (arr->nd)
            args.shape = NanoVector::make(arr->dimensions, arr->nd);
        else
            args.shape.push_back(1);
        args.dtype = get_type_str(arr);
        if (is_c_style(arr))
            args.ptr = arr->data;

        // use 32-bit by default
        if (holder.obj && args.dtype.dsize() == 8) {
            auto num = PyArray_Size(arr)/8;
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
    } else {
        LOGf << "type <" >> Py_TYPE(obj)->tp_name >> "> not support for jittor array";
    }
    NanoVector shape = args.shape;
    output = create_output(shape, args.dtype);
    int64 size = output->size;
    if (shape.size() == 1 && shape[0] == 1) {
        output->flags.set(NodeFlags::_force_fuse);
        set_type(OpType::element);
    }
    void* host_ptr = nullptr;
    #ifdef HAS_CUDA
    if (use_cuda) {
        flags.set(NodeFlags::_cpu, 0);
        flags.set(NodeFlags::_cuda, 1);
        if (!output->flags.get(NodeFlags::_force_fuse)) {
            // free prev allocation first
            event_queue.flush();
            // alloc new allocation
            auto size = output->size;
            new (&allocation) Allocation(&cuda_dual_allocator, size);
            host_ptr = cuda_dual_allocator.get_dual_allocation(allocation.allocation).host_ptr;
        }
    }
    #endif
    if (!host_ptr) {
        new (&allocation) Allocation(cpu_allocator, output->size);
        host_ptr = allocation.ptr;
    }

    if (args.ptr) {
        // if has ptr, copy from ptr
        std::memcpy(host_ptr, args.ptr, size);
    } else {
        // this is non-continue numpy array
        int64 dims[args.shape.size()];
        for (int i=0; i<args.shape.size(); i++)
            dims[i] = args.shape[i];
        holder.assign(PyArray_New(
            PyArray_Type, // subtype
            args.shape.size(), // nd
            dims, // dims
            get_typenum(args.dtype), // type_num
            NULL, // strides
            (void*)host_ptr, // data
            0, // itemsize
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, // flags
            NULL // obj
        ));
        ASSERT(0==PyArray_CopyInto(holder.obj, obj));
    }
}

}
