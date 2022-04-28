// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cmath>
#include <pybind11/pybind11.h>
#include <iostream>
#include "var.h"
#include "var_holder.h"
#include "executor.h"
#include "ops/getitem_op.h"
#include "ops/op_register.h"
#include "pyjt/py_converter.h"
#include "misc/cuda_flags.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"

namespace py = pybind11;

namespace jittor {

    void AT_CUDA_CHECK(cudaError_t status) { 
                cudaError_t error = status;                                           
        if (error != cudaSuccess) {                                           
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) 
                    << " at line: " << __LINE__ << std::endl;               
            exit(EXIT_FAILURE);                                               
        }    
    }

    #define TORCH_CHECK(a, b) assert(a) 

    #define AT_PRIVATE_CASE_TYPE(enum_type, type, ...) \
        case enum_type: {                                \
            using scalar_t = type;                         \
            return __VA_ARGS__();                          \
        }

    #define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)                 \
    [&] {                                                                      \
        int _st = TYPE; \
        switch (_st) {                                                           \
        AT_PRIVATE_CASE_TYPE(2, double, __VA_ARGS__)      \
        AT_PRIVATE_CASE_TYPE(1, float, __VA_ARGS__)        \
        default:                                                               \
        break; \
        }                                                                        \
    }()

    static auto make_empty = get_op_info("empty")
    .get_constructor<VarPtr, NanoVector, NanoString>();

    namespace at {
        namespace cuda {
            cudaStream_t getCurrentCUDAStream() {
                return (cudaStream_t)0;
            }

            struct OptionalCUDAGuard {
                // todo: supoort device guard.
                OptionalCUDAGuard() {}
                OptionalCUDAGuard(int x) {}
            };
        }
        enum MemoryFormat {
            Contiguous,
            Incontiguous,
            NCHW,
            NHWC,
            CHWN
        };
    }

    namespace torch {
        
        // definiton of torch kTtypes
        int kUInt8 = 0;
        int kFloat = 1;
        int kDouble = 2;
        int kHalf = 3;
        int kInt32 = 4;

        struct Option {
            int dtype_;
            Option dtype(int dt) {
                return Option(dt);
            }
            Option(int dt):dtype_(dt) {}
            Option() {}
        };

        struct Device {
            int index() {
                return 0;
            }
            bool operator ==(const Device& b) const { return true; }
        };

        struct Tensor {
            Tensor():format(at::MemoryFormat::Contiguous),jtptr(nullptr),ndim(0) {}

            Tensor(VarPtr& ptr) {
                jtptr = new VarHolder(ptr.ptr);
                ndim = jtptr->shape().size();
            }

            Tensor(const Tensor& b) {
                if(!b.jtptr) jtptr = nullptr;
                else
                    jtptr = new VarHolder(b.jtptr->var);
            }
            
            NanoVector size() {
                if(!jtptr) return 0;
                return jtptr->shape();
            }

            int size(int i) {
                if(!jtptr) {
                    LOGir << "Tensor is None.";
                    return -1;
                }
                if(i == -1) 
                    i = jtptr->shape().size() - 1;
                return jtptr->shape()[i];
            }

            int numel() {
                if(!jtptr)
                    return 0;
                return jtptr->numel();
            }

            void init_stride() {
                if(!jtptr) return;
                int64 prod = 1;
                for(int i=jtptr->shape().size()-1;i>=0;i--) {
                    strides.push_back(prod);
                    prod *= jtptr->shape()[i];
                }
            }

            NanoVector stride(){
                if(strides.size() == 0) init_stride();
                return strides;
            } 

            int stride(int i) {
                if(!jtptr) {
                    LOGir << "Tensor is None.";
                    return -1;
                }
                if(strides.size() == 0) init_stride();
                if(i == -1)
                    i = 0;
                return strides[strides.size() - i - 1];
            }

            int dtype() {
                if(!jtptr) return -1; // nullptr
                NanoString dtype_ = jtptr->dtype();
                if(dtype_ == "uint8") 
                    return 0;
                if(dtype_ == "float16")
                    return 3;
                if(dtype_ == "float32")
                    return 1;
                if(dtype_ == "float64")
                    return 2;
                if(dtype_ == "int32")
                    return 4;
                return -1; // non-exist types
            }

            int scalar_type() {
                return dtype();
            }

            template<typename T=float>
            T* data_ptr() {
                if(!jtptr)
                    return nullptr;
                jtptr->sync(true);
                return jtptr->var->ptr<T>();
            }

            at::MemoryFormat suggest_memory_format() {
                return format;
            }
        
            void sync(bool device_sync=true) {
                if(jtptr)
                    jtptr->sync(device_sync);
            }

            int dim() {
                if(!jtptr)
                    return 0;
                return jtptr->shape().size();
            }

            bool is_contiguous() {
                return true;
            }

            Device device() { // device is controlled by jittor 
                return Device(); // so all tensors are on the same device.
            }

            void cuda() {
                return;
            }
            
            bool is_cuda() {
                return use_cuda;
            }

            Option options() { // assume that jtptr is not nullptr
                return Option(dtype());
            }

            VarHolder* jtptr;
            int64 ndim;
            at::MemoryFormat format;
            NanoVector strides;
        };

        Tensor empty(NanoVector shape, Option option, at::MemoryFormat format) {
            // todo: add special support for different formats. For now all outputs are contiguous format.
            VarPtr ptr;
            switch(option.dtype_) {
                case 0:
                    ptr = make_empty(shape, "uint8"); 
                    break;
                case 1:
                    ptr = make_empty(shape, "float32");
                    break;
                case 2:
                    ptr = make_empty(shape, "float64");
                    break;
                case 4:
                    ptr = make_empty(shape, "int32");
                    break;
                default:
                    ptr = make_empty(shape, "float32");
                    break;  
            }
            return Tensor(ptr);
        }

        // void test_tensor(Tensor a) {
        //     // todo: add special support for different formats.
        //     std::cout << "Success!!" << std::endl;
        // }

        // Tensor test_ret_tensor(Tensor a) {
        //     return a;
        // }
    }
    
    int device_of(torch::Tensor a) {
        if(use_cuda) 
            return 1;
        else return 0;
    }

}

namespace pybind11 { namespace detail {
    template <> struct type_caster<jittor::torch::Tensor> {
    public:
 
        PYBIND11_TYPE_CASTER(jittor::torch::Tensor, const_name("Tensor"));

        bool load(handle src, bool) {
            PyObject *source = src.ptr();
            if(source != Py_None) {
                jittor::VarHolder* var_holder = jittor::from_py_object<jittor::VarHolder*>(source);
                if (!var_holder)
                    return false;
                value.jtptr = var_holder;
            } else {
                value.jtptr = nullptr;
            }
            return !PyErr_Occurred();
        }

        static handle cast(jittor::torch::Tensor src, return_value_policy, handle) {
            jittor::PyObjHolder obj(_PyObject_New(&jittor::PyjtVarHolder.ht_type));
            auto ptr = GET_RAW_PTR(jittor::VarHolder*, obj.obj);
            new (ptr) jittor::VarHolder (src.jtptr->var);
            return obj.release();
        }
    };
}} 

