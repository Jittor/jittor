// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>
//     Guowei Yang <471184555@qq.com>
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mem/allocator.h"
#include "var.h"
#include "cudnn_conv3d_backward_x_op.h"
#include "cudnn_wrapper.h"
#include "executor.h"
#include "ops/op_register.h"

using namespace std;

namespace jittor {

extern int use_tensorcore;

#pragma GCC diagnostic ignored "-Wunused-variable"

#ifndef JIT

CudnnConv3dBackwardXOp::CudnnConv3dBackwardXOp(Var* w, Var* dy, int depth, int height, int width, int strided, int strideh, int stridew, int paddingd, int paddingh, int paddingw, int dilationd, int dilationh, int dilationw, int groups, string xformat) 
        : w(w), dy(dy), xd(depth), xh(height), xw(width), strided(strided), strideh(strideh), stridew(stridew), paddingd(paddingd), paddingh(paddingh), paddingw(paddingw), dilationd(dilationd), dilationh(dilationh), dilationw(dilationw), groups(groups),
      xformat(move(xformat)) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_manual_set_vnbb);
    w->flags.set(NodeFlags::_needed_by_backward);
    dy->flags.set(NodeFlags::_needed_by_backward);
    dx = create_output(nullptr, dtype_infer(dy->ns, w->ns));
}

void CudnnConv3dBackwardXOp::infer_shape() {
    ASSERTop(w->shape.size(),==,5);
    ASSERTop(dy->shape.size(),==,5);
    int xn, xc, wd, wh, ww, wci, wco, yn, yc, yd, yh, yw;
    w->shape.unpack(wco, wci, wd, wh, ww);
    if (xformat == "ncdhw")
        dy->shape.unpack(yn, yc, yd, yh, yw);
    else
        dy->shape.unpack(yn, yd, yh, yw, yc);
    xn = yn, xc = wci * groups;
    if (xformat == "ncdhw")
        dx->set_shape(NanoVector(xn, xc, xd, xh, xw));
    else
        dx->set_shape(NanoVector(xn, xd, xh, xw, xc));
}

void CudnnConv3dBackwardXOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << dx->dtype();
    jk << "«Ty:" << dy->dtype();
    jk << "«Tw:" << w->dtype();
}


static auto make_conv3d = get_op_info("cudnn_conv3d")
    .get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, int, string>();
static auto make_backwardw = get_op_info("cudnn_conv3d_backward_w")
    .get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, int, int, int, int, string>();


VarPtr CudnnConv3dBackwardXOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    int xn, xc, wd, wh, ww, wci, wco, yn, yc, yd, yh, yw;
    w->shape.unpack(wco, wci, wd, wh, ww);
    
    if (v_index == 0) {
        return make_backwardw(dout, dy, wd, wh, ww, strided, strideh, stridew, paddingd, paddingh, paddingw, dilationd, dilationh, dilationw, groups, xformat);
    } else {
        return make_conv3d(dout, w, strided, strideh, stridew, paddingd, paddingh, paddingw, dilationd, dilationh, dilationw, groups, xformat);
    }
}
// unordered_map<string, cudnnConvolutionBwdDataAlgo_t> bwdx_algo_cache;

#else // JIT
#ifdef JIT_cuda

#pragma clang diagnostic ignored "-Wtautological-compare"

EXTERN_LIB unordered_map<string, cudnnConvolutionBwdDataAlgo_t> bwdx_algo_cache;

template <typename T_ELEM> __inline__  cudnnDataType_t getDataType();
template <> __inline__ cudnnDataType_t getDataType<half1>() { return CUDNN_DATA_HALF;   }
template <> __inline__ cudnnDataType_t getDataType<float>() { return CUDNN_DATA_FLOAT;  }

void CudnnConv3dBackwardXOp::jit_run() {
    auto x = dx;
    auto y = dy;        
    cudnnHandle_t& handle_ = cudnn_handle;

    cudnnTensorDescriptor_t cudnnIdesc;
    cudnnFilterDescriptor_t cudnnFdesc;
    cudnnTensorDescriptor_t cudnnOdesc;
    cudnnConvolutionDescriptor_t cudnnConvDesc;
    
    checkCudaErrors(cudnnCreateTensorDescriptor( &cudnnIdesc ));
    checkCudaErrors(cudnnCreateFilterDescriptor( &cudnnFdesc ));
    checkCudaErrors(cudnnCreateTensorDescriptor( &cudnnOdesc ));
    checkCudaErrors(cudnnCreateConvolutionDescriptor( &cudnnConvDesc ));
    checkCudaErrors(cudnnSetConvolutionGroupCount( cudnnConvDesc, groups ));


    int xn, xc, xd, xh, xw, wd, wh, ww, wci, wco, yn, yc, yd, yh, yw;
    int sx[] = {0,0,0,0,1};
    for (int i=3; i>=0; i--) sx[i] = sx[i+1] * x->shape[i+1];
    int strideX[5];
    if (xformat == "ncdhw") {
        x->shape.unpack(xn, xc, xd, xh, xw);
        int tmp[5] = {sx[0],sx[1],sx[2],sx[3],sx[4]};
        memcpy(strideX, tmp, sizeof(tmp));
    } else {
        x->shape.unpack(xn, xd, xh, xw, xc);
        int tmp[5] = {sx[0],sx[2],sx[3],sx[4],sx[1]};
        memcpy(strideX, tmp, sizeof(tmp));
    }
    int dimX[] = {xn, xc, xd, xh, xw};
    // dimX: ncdhw
    checkCudaErrors(cudnnSetTensorNdDescriptor(
        cudnnIdesc, getDataType<Tx>(),
        5, dimX, strideX
    ));

    auto ws = w->shape;
    int dimW[] = {(int)ws[0],(int)ws[1],(int)ws[2],(int)ws[3],(int)ws[4]};
    // cudnn only support this two format
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnSetFilterNdDescriptor
    #define filterFormat_oihw CUDNN_TENSOR_NCHW
    #define filterFormat_ohwi CUDNN_TENSOR_NHWC

    // dimW: KCRS(oihw)
    checkCudaErrors(cudnnSetFilterNdDescriptor(
        cudnnFdesc, getDataType<Tw>(),
        // filterFormat_@WFORMAT, 5, dimW
        filterFormat_oihw, 5, dimW
    ));

    int padA[] = {paddingd, paddingh, paddingw};
    int convstrideA[] = {strided, strideh, stridew};
    int dilationA[] = {dilationd, dilationh, dilationw};
    // difference between
    // CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION
    // is the kernel rc order
    // currently, No perf difference is observed between
    // this two mode
    checkCudaErrors(cudnnSetConvolutionNdDescriptor(
        cudnnConvDesc, 3,
        padA, convstrideA, dilationA,
        CUDNN_CROSS_CORRELATION, getDataType<Ty>()
    ));

    // using tensor core
    if(use_tensorcore){
        checkCudaErrors( cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) );
    }


    int sy[] = {0,0,0,0,1};
    for (int i=3; i>=0; i--) sy[i] = sy[i+1] * y->shape[i+1];
    int strideY[5];
    if (xformat == "ncdhw") {
        y->shape.unpack(yn, yc, yd, yh, yw);
        int tmp[5] = {sy[0],sy[1],sy[2],sy[3],sy[4]};
        memcpy(strideY, tmp, sizeof(tmp));
    } else {
        y->shape.unpack(yn, yd, yh, yw, yc);
        int tmp[5] = {sy[0],sy[2],sy[3],sy[4],sy[1]};
        memcpy(strideY, tmp, sizeof(tmp));
    }
    int dimY[] = {yn, yc, yd, yh, yw};

    checkCudaErrors(cudnnSetTensorNdDescriptor(
        cudnnOdesc, getDataType<Ty>(),
        5, dimY, strideY
    ));

    cudnnConvolutionBwdDataAlgo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
    };
    int num_algos = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    int perf_count;
    STACK_ALLOC(cudnnConvolutionBwdDataAlgoPerf_t,perf_results,num_algos);
    cudnnConvolutionBwdDataAlgo_t algo;
    bool benchmark=true;

    JK& jk = get_jk();
    jk.clear();
    jk << dimX[0] << "," << dimX[1] << "," << dimX[2] << "," << dimX[3] << "," << dimX[4] << ",";
    jk << dimW[0] << "," << dimW[1] << "," << dimW[2] << "," << dimW[3] << "," << dimW[4] << ",";
    jk << paddingd << paddingh << paddingw << "," << strided << strideh <<stridew << "," << dilationd << dilationh << dilationw << "," << groups << ".";
    auto iter = bwdx_algo_cache.find(jk.to_string());
    
    if (iter!=bwdx_algo_cache.end()) algo = iter->second;
    else {
        if (bwdx_algo_cache.size()>=max_cache_size) benchmark = false;
        if (benchmark) {
            size_t max_ws_size = 0;
            for (int i = 0; i < num_algos; i++) {
                size_t sz;
                cudnnStatus_t ret = cudnnGetConvolutionBackwardDataWorkspaceSize(handle_, cudnnFdesc, cudnnOdesc, cudnnConvDesc, cudnnIdesc, algos[i], &sz);
                // continue if use too much workspace
                if (sz > mem_info.total_cuda_ram * max_workspace_ratio) continue;
                if (CUDNN_STATUS_SUCCESS == ret && sz > max_ws_size) max_ws_size = sz;
            } 
            size_t allocation;
            void* ws = exe.temp_allocator->alloc(max_ws_size, allocation);
            checkCudaErrors(cudnnFindConvolutionBackwardDataAlgorithmEx(
                handle_,
                cudnnFdesc, w->ptr<Tw>(),
                cudnnOdesc, y->ptr<Ty>(),
                cudnnConvDesc,
                cudnnIdesc, x->ptr<Tx>(),
                num_algos,
                &perf_count,
                perf_results,
                ws,
                max_ws_size));
            exe.temp_allocator->free(ws, max_ws_size, allocation);
        } else {
            checkCudaErrors(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                handle_,
                cudnnFdesc,
                cudnnOdesc,
                cudnnConvDesc,
                cudnnIdesc,
                num_algos,
                &perf_count,
                perf_results));
        }
        int best_algo_idx=-1;
        for (int i = 0; i < perf_count; i++) 
            if (perf_results[i].status == CUDNN_STATUS_SUCCESS){
                best_algo_idx=i;
                break;
            }
        ASSERT(best_algo_idx!=-1);
        algo=perf_results[best_algo_idx].algo;
        if (benchmark) {
            bwdx_algo_cache[jk.to_string()] = algo;
            if (bwdx_algo_cache.size()==max_cache_size)
                LOGw << "backward x algorithm cache is full";
        }
    }

    // TODO: warp work space
    void *workSpace = 0;
    size_t workSpaceSize;
    checkCudaErrors (cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle_, cudnnFdesc, cudnnOdesc, cudnnConvDesc, 
        cudnnIdesc, algo, &workSpaceSize));
    size_t allocation;
    if (workSpaceSize > 0) {
        workSpace = exe.temp_allocator->alloc(workSpaceSize, allocation);
    }
    float alpha=1, beta=0;
    checkCudaErrors(cudnnConvolutionBackwardData(
        handle_,
        (void*)(&alpha),
        cudnnFdesc, w->ptr<Tw>(),
        cudnnOdesc, y->ptr<Ty>(),
        cudnnConvDesc,
        algo,
        workSpace, workSpaceSize,
        (void*)(&beta),
        cudnnIdesc, x->ptr<Tx>())
    );
    if (workSpace)
        exe.temp_allocator->free(workSpace, workSpaceSize, allocation);
        
    checkCudaErrors(cudnnDestroyTensorDescriptor( cudnnIdesc ));
    checkCudaErrors(cudnnDestroyFilterDescriptor( cudnnFdesc ));
    checkCudaErrors(cudnnDestroyTensorDescriptor( cudnnOdesc ));
    checkCudaErrors(cudnnDestroyConvolutionDescriptor( cudnnConvDesc ));
}
#endif
#endif // JIT

} // jittor
