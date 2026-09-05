// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
// Ownership for cuDNN descriptors and for scratch memory.
//
// Every cuDNN op used to Create its descriptors, do the work, and Destroy them
// at the bottom of the function. Any throw in between -- and every cuDNN call
// in these ops throws on failure -- skipped every Destroy, so the run that hit
// a cuDNN error also leaked four descriptors and a workspace per attempt. An
// error path is exactly the path that gets retried.
//
// Destructors report and never raise (6.B17): a destructor is noexcept, so a
// Destroy that fails during unwinding would replace the original error with
// std::terminate.
//
// Header-only: only ops/*op.cc are compiled into the cuDNN op library.
#include <cudnn.h>
#include <vector>
#include "cudnn_wrapper.h"
#include "executor.h"
#include "mem/allocator.h"

namespace jittor {

#define JT_CUDNN_DESCRIPTOR(NAME, TYPE, CREATE, DESTROY)                    \
struct NAME {                                                               \
    TYPE desc = nullptr;                                                    \
    NAME() { checkCudaErrors(CREATE(&desc)); }                              \
    ~NAME() { if (desc) peekCudaErrorsAlways(DESTROY(desc)); }              \
    NAME(const NAME&) = delete;                                             \
    NAME& operator=(const NAME&) = delete;                                  \
    operator TYPE() const { return desc; }                                  \
}

JT_CUDNN_DESCRIPTOR(CudnnTensorDescriptor, cudnnTensorDescriptor_t,
    cudnnCreateTensorDescriptor, cudnnDestroyTensorDescriptor);
JT_CUDNN_DESCRIPTOR(CudnnFilterDescriptor, cudnnFilterDescriptor_t,
    cudnnCreateFilterDescriptor, cudnnDestroyFilterDescriptor);
JT_CUDNN_DESCRIPTOR(CudnnConvolutionDescriptor, cudnnConvolutionDescriptor_t,
    cudnnCreateConvolutionDescriptor, cudnnDestroyConvolutionDescriptor);

#undef JT_CUDNN_DESCRIPTOR

/** A run of `n` tensor descriptors, as the RNN ops need.

    They pass `.data()` to cuDNN, so the array has to stay a plain
    `cudnnTensorDescriptor_t[]`; only the ownership is added. Partial
    construction is handled: if the k-th Create throws, the k-1 already made
    are still destroyed.
 */
struct CudnnTensorDescriptorArray {
    std::vector<cudnnTensorDescriptor_t> descs;

    explicit CudnnTensorDescriptorArray(int n) : descs(n, nullptr) {
        for (int i = 0; i < n; i++)
            checkCudaErrors(cudnnCreateTensorDescriptor(&descs[i]));
    }
    ~CudnnTensorDescriptorArray() {
        for (auto d : descs)
            if (d) peekCudaErrorsAlways(cudnnDestroyTensorDescriptor(d));
    }
    CudnnTensorDescriptorArray(const CudnnTensorDescriptorArray&) = delete;
    CudnnTensorDescriptorArray& operator=(const CudnnTensorDescriptorArray&) = delete;

    cudnnTensorDescriptor_t* data() { return descs.data(); }
    cudnnTensorDescriptor_t operator[](int i) const { return descs[i]; }
};

/** Scratch memory from an allocator, released however the scope is left.

    Zero bytes means no allocation and a null pointer, which is what cuDNN
    wants for "no workspace" -- the hand-written version of this was a bare
    `void* ws` left uninitialized when the size came back zero, then freed
    unconditionally.
 */
struct CudnnWorkspace {
    void* ptr = nullptr;
    size_t size = 0;
    size_t allocation = 0;
    Allocator* allocator = nullptr;

    CudnnWorkspace() = default;
    explicit CudnnWorkspace(size_t bytes) : size(bytes) {
        if (!size) return;
        allocator = runtime_executor().temp_allocator;
        ptr = allocator->alloc(size, allocation);
    }
    ~CudnnWorkspace() {
        if (ptr) allocator->free(ptr, size, allocation);
    }
    CudnnWorkspace(const CudnnWorkspace&) = delete;
    CudnnWorkspace& operator=(const CudnnWorkspace&) = delete;
};

} // jittor
