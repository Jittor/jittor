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
#include <vector>
#include "var.h"
#include "var_holder.h"
#include "executor.h"
#include "ops/getitem_op.h"
#include "ops/op_register.h"
#include "pyjt/py_converter.h"
#include "misc/cuda_flags.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "type/fp16_compute.h"

namespace py = pybind11;

namespace jittor {
    void AT_CUDA_CHECK(cudaError_t status);

    #define TORCH_CHECK(a, ...) assert(a) 

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
        extern int kBFloat16;
        extern int kLong;
        extern int kInt64;
        // TODO: support BFloat16 in jittor
        using BFloat16 = jittor::float16; 

        struct Option {
            int dtype_;
            int device_;
            int devid_;
            int type();

            Option dtype(int dt) const;

            Option device(int device, int devid=0) const;

            Option(int dt);
            Option();
        };

        struct Device {
            int index();
            int type();
            bool operator ==(const Device& b) const;
        };


        Option dtype(int);
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
            int dtype() const;
            int scalar_type();
            Tensor contiguous();
            bool defined();
            int get_device();
            NanoVector sizes(); // TODO: WHAT torch sizes really do?
            Tensor clone();
            Tensor detach();
            template<typename T=float>
            T* data_ptr() {
                if(!jtptr)
                    return nullptr;
                jtptr->sync(true);
                return jtptr->var->ptr<T>();
            }

            long long nbytes() {
                if(!jtptr || !jtptr->var) return 0;
                return jtptr->var->dsize() * (long long) jtptr->numel();
            }

            at::MemoryFormat suggest_memory_format();
        
            void sync(bool device_sync=true);

            int dim();

            bool is_contiguous() const;

            Device device() const;

            void cuda();
            
            bool is_cuda();

            Option options();

            VarHolder* jtptr;
            int64 ndim;
            at::MemoryFormat format;
            NanoVector strides;
        };

        Tensor empty(NanoVector shape, Option option, at::MemoryFormat format=at::MemoryFormat::Contiguous);
        Tensor zeros(NanoVector shape, Option option);
        Tensor zeros_like(Tensor& refer_tensor);
        Tensor empty_like(Tensor& refer_tensor);
        Option TensorOptions();
        // void test_tensor(Tensor a) {
        //     // todo: add special support for different formats.
        //     std::cout << "Success!!" << std::endl;
        // }

        // Tensor test_ret_tensor(Tensor a) {
        //     return a;
        // }
    }
    
    namespace at {
        // in face at::Tensor is not differentiable. TODO: set requires_grad to false for it.
        using Tensor = torch::Tensor; 
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
