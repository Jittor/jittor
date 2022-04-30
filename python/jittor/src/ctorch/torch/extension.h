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

    void AT_CUDA_CHECK(cudaError_t status);

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

    namespace at {
        namespace cuda {
            cudaStream_t getCurrentCUDAStream();

            struct OptionalCUDAGuard {
                // todo: supoort device guard.
                OptionalCUDAGuard();
                OptionalCUDAGuard(int x);
            };
        }

        typedef enum MemoryFormat {
            Contiguous,
            Incontiguous,
            NCHW,
            NHWC,
            CHWN
        } MemoryFormat; 
    }

    namespace torch {
        
        // definiton of torch kTypes
        extern int kUInt8;
        extern int kFloat;
        extern int kDouble;
        extern int kHalf;
        extern int kFloat16;
        extern int kFloat32;
        extern int kInt32;
        extern int kCUDA;

        struct Option {
            int dtype_;
            int device_;
            int devid_;

            Option dtype(int dt);

            Option device(int device, int devid);

            Option(int dt);
            Option();
        };

        struct Device {
            int index();
            int type();
            bool operator ==(const Device& b) const;
        };



        struct Tensor {
            Tensor();

            Tensor(VarPtr& ptr);

            Tensor(const Tensor& b);
            
            NanoVector size();

            int size(int i);

            int numel();

            void init_stride();

            NanoVector stride();

            int stride(int i);

            int dtype();

            int scalar_type();

            template<typename T=float>
            T* data_ptr() {
                if(!jtptr)
                    return nullptr;
                jtptr->sync(true);
                return jtptr->var->ptr<T>();
            }

            at::MemoryFormat suggest_memory_format();
        
            void sync(bool device_sync=true);

            int dim();

            bool is_contiguous();

            Device device();

            void cuda();
            
            bool is_cuda();

            Option options();

            VarHolder* jtptr;
            int64 ndim;
            at::MemoryFormat format;
            NanoVector strides;
        };

        Tensor empty(NanoVector shape, Option option, at::MemoryFormat format=at::MemoryFormat::Contiguous);

        Option TensorOptions();

        // void test_tensor(Tensor a) {
        //     // todo: add special support for different formats.
        //     std::cout << "Success!!" << std::endl;
        // }

        // Tensor test_ret_tensor(Tensor a) {
        //     return a;
        // }
    }
    
    int device_of(torch::Tensor a);

}


namespace pybind11 { namespace detail {
    template <> struct type_caster<jittor::torch::Tensor> {
    public:
        PYBIND11_TYPE_CASTER(jittor::torch::Tensor, const_name("Tensor"));

        bool load(handle src, bool);

        static handle cast(jittor::torch::Tensor src, return_value_policy, handle);
    };
}} 

using namespace jittor;
