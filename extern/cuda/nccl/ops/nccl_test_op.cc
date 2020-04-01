// ***************************************************************
// Copyright (c) 2019 Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "nccl_test_op.h"
#include "misc/str_utils.h"

#include <nccl.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#ifndef JIT
const char *_cudaGetErrorEnum(ncclResult_t error) {
    return ncclGetErrorString(error);
}
#endif

namespace jittor {

#ifndef JIT
NcclTestOp::NcclTestOp(string cmd) : cmd(cmd) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    output = create_output(1, ns_float32);
}

void NcclTestOp::jit_prepare() {
    add_jit_define("T", ns_float32);
}

#else // JIT
#ifdef JIT_cuda

void NcclTestOp::jit_run() {
    auto args = split(cmd, " ");
    if (!cmd.size()) args.clear();
    vector<char*> v(args.size());
    for (uint i=0; i<args.size(); i++)
        v[i] = &args[i][0];
    output->ptr<T>()[0] = 123;
    


    //managing 4 devices
    int nDev;
    checkCudaErrors(cudaGetDeviceCount(&nDev));
    nDev = std::min(nDev, 2);

    ncclComm_t comms[nDev];
    int size = 32*1024*1024;
    int devs[4] = { 0, 1, 2, 3 };


    //allocating and initializing device buffers
    float** sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** recvbuff = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


    for (int i = 0; i < nDev; ++i) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaMalloc(sendbuff + i, size * sizeof(float)));
        checkCudaErrors(cudaMalloc(recvbuff + i, size * sizeof(float)));
        checkCudaErrors(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        checkCudaErrors(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        checkCudaErrors(cudaStreamCreate(s+i));
    }


    //initializing NCCL
    checkCudaErrors(ncclCommInitAll(comms, nDev, devs));


    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
    checkCudaErrors(ncclGroupStart());
    for (int i = 0; i < nDev; ++i)
        checkCudaErrors(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
    comms[i], s[i]));
    checkCudaErrors(ncclGroupEnd());


    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaStreamSynchronize(s[i]));
    }


    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaFree(sendbuff[i]));
        checkCudaErrors(cudaFree(recvbuff[i]));
    }


    //finalizing NCCL
    for(int i = 0; i < nDev; ++i)
    ncclCommDestroy(comms[i]);
}

#endif
#endif // JIT

} // jittor
