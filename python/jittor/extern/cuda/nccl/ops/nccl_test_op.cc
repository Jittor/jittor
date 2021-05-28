// ***************************************************************
// Copyright (c) 2019 Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "nccl_test_op.h"
#include "utils/str_utils.h"

#include "nccl_warper.h"


namespace jittor {

#ifndef JIT
NcclTestOp::NcclTestOp(string cmd) : cmd(cmd) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    output = create_output(1, ns_float32);
}

void NcclTestOp::jit_prepare(JK& jk) {
    jk << _CS("[T:float32]");
}

#else // JIT
#ifdef JIT_cuda

static void test_with_mpi() {
    int size = 32*1024*1024;
    int myRank = mpi_world_rank;
    int nRanks = mpi_world_size;
    int localRank = mpi_local_rank;

    float *sendbuff, *recvbuff;
    cudaStream_t s;
    checkCudaErrors(cudaMalloc(&sendbuff, size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&recvbuff, size * sizeof(float)));
    checkCudaErrors(cudaStreamCreate(&s));

    //communicating using NCCL
    checkCudaErrors(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
        comm, s));

    //completing NCCL operation by synchronizing on the CUDA stream
    checkCudaErrors(cudaStreamSynchronize(s));

    //free device buffers
    checkCudaErrors(cudaFree(sendbuff));
    checkCudaErrors(cudaFree(recvbuff));
    checkCudaErrors(cudaStreamDestroy(s));

    LOGi << "MPI rank" << myRank << "Success";
}

void NcclTestOp::jit_run() {
    output->ptr<T>()[0] = 123;
    if (cmd == "test_with_mpi") {
        test_with_mpi();
        return;
    }

    
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
    checkCudaErrors(cudaSetDevice(0));
}

#endif
#endif // JIT

} // jittor
