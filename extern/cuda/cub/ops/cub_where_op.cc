// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Xiangli Li <1905692338@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cub_where_op.h"
#ifdef JIT_cuda
#include "executor.h"
#include <assert.h>
#include <executor.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#endif

namespace jittor {

#ifndef JIT
CubWhereOp::CubWhereOp(Var* cond, NanoString dtype) : cond(cond) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    flags.set(NodeFlags::_vary_shape);
    auto ndim = cond->shape.size();
    outs.reset(new Var*[ndim]);
    for (uint i=0; i<ndim; i++)
        outs[i] = create_output(nullptr, dtype);
}

void CubWhereOp::infer_shape() {
    auto ndim = cond->shape.size();
    auto num = cond->num;
    if (num>0) num = -num;
    for (uint i=0; i<ndim; i++)
        outs[i]->set_shape({num});
}

void CubWhereOp::jit_prepare() {
    add_jit_define("Ti", cond->dtype());
    add_jit_define("To", outs[0]->dtype());
    add_jit_define("NDIM", JK::hex1(cond->shape.size()));
}

#else // JIT
#ifdef JIT_cuda

template<typename T>
struct NonZeroOp
{
    __host__ __device__ __forceinline__ bool operator()(const T& a) const {
      return (a!=T(0));
    }
};

template<typename T>
struct ConvertOp 
{   
    const T div;
    const T dim_size;
    ConvertOp(T _div,T dim_size): div(_div),dim_size(dim_size){} 
    __host__ __device__ __forceinline__ T operator()(const T& val) const {
        return (val/div) % dim_size;
    }
};

void CubWhereOp::jit_run(){
    int N = cond->num;
    size_t temp_storage_bytes=0;
    size_t num_nonzeros_allocation;
    auto num_nonzeros = exe.allocator->alloc(sizeof(int), num_nonzeros_allocation);
    cub::TransformInputIterator<bool, NonZeroOp<Ti>, Ti*> itr(cond->ptr<Ti>(), NonZeroOp<Ti>());
    cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, itr, (int *)num_nonzeros, N);
    
    size_t temp_storage_allocation;
    auto temp_storage = exe.allocator->alloc(temp_storage_bytes, temp_storage_allocation);

    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, itr, (int *)num_nonzeros, N);
    exe.allocator->free(temp_storage, temp_storage_bytes, temp_storage_allocation);



    int num_nonzeros_h;
    checkCudaErrors(cudaMemcpyAsync(&num_nonzeros_h, num_nonzeros, sizeof(int), cudaMemcpyDeviceToHost, 0));
    //need to synchronize to make sure data is available on the host
    checkCudaErrors(cudaStreamSynchronize(0));
    
    size_t out_temp_allocation;
    To * out_temp = (To *) exe.allocator->alloc(num_nonzeros_h*sizeof(To), out_temp_allocation);

    @for(i, 0, NDIM, outs[@i]->set_shape({num_nonzeros_h});)
    if (NDIM > 0) {
        cub::CountingInputIterator<To> counting_itr(0);
        temp_storage_bytes = 0;
        cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, counting_itr, itr,out_temp, (int*)num_nonzeros, N);
        temp_storage = exe.allocator->alloc(temp_storage_bytes, temp_storage_allocation);
        cub::DeviceSelect::Flagged(temp_storage, temp_storage_bytes, counting_itr, itr,out_temp, (int*)num_nonzeros, N);
        exe.allocator->free(temp_storage, temp_storage_bytes, temp_storage_allocation);

        if (num_nonzeros_h > 0 && NDIM > 1){
            To div = 1;
            for (int dim = NDIM-1; dim >= 0; dim--){
                To dim_size = cond->shape[dim];
                thrust::transform(
                    thrust::device_ptr<To>(out_temp),
                    thrust::device_ptr<To>(out_temp) + num_nonzeros_h,
                    thrust::device_ptr<To>(outs[dim]->ptr<To>()),
                    ConvertOp<To>(div,dim_size)
                );
                div *= dim_size;
           }
        }else if (num_nonzeros_h>0 && NDIM==1){
            checkCudaErrors(cudaMemcpyAsync(outs[0]->ptr<To>(), out_temp, num_nonzeros_h*sizeof(To), cudaMemcpyDeviceToDevice, 0));
        }
    }
    exe.allocator->free(num_nonzeros, sizeof(int), num_nonzeros_allocation);
    exe.allocator->free(out_temp, NDIM*num_nonzeros_h*sizeof(To), out_temp_allocation);
    
}
#endif
#endif // JIT

} // jittor
