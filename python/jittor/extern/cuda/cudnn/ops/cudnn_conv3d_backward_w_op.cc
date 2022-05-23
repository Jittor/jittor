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
#include "cudnn_conv3d_backward_w_op.h"
#include "cudnn_wrapper.h"
#include "executor.h"
#include "ops/op_register.h"

using namespace std;

namespace jittor {

extern int use_tensorcore;

#pragma GCC diagnostic ignored "-Wunused-variable"

#ifndef JIT

CudnnConv3dBackwardWOp::CudnnConv3dBackwardWOp(Var* x, Var* dy, int kd, int kh, int kw, int strided, int strideh, int stridew, int paddingd, int paddingh, int paddingw, int dilationd, int dilationh, int dilationw, int groups, string xformat)
        : x(x), dy(dy), kd(kd), kh(kh), kw(kw), strided(strided), strideh(strideh), stridew(stridew), paddingd(paddingd), paddingh(paddingh), paddingw(paddingw), dilationd(dilationd), dilationh(dilationh), dilationw(dilationw), groups(groups),
      xformat(move(xformat)) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_manual_set_vnbb);
    x->flags.set(NodeFlags::_needed_by_backward);
    dy->flags.set(NodeFlags::_needed_by_backward);
    dw = create_output(nullptr, dtype_infer(dy->ns, x->ns));
}

void CudnnConv3dBackwardWOp::infer_shape() {
    ASSERTop(x->shape.size(),==,5);
    ASSERTop(dy->shape.size(),==,5);
    int xn, xc, xd, xh, xw, wd, wh, ww, wci, wco, yn, yc, yd, yh, yw;

    if (xformat == "ncdhw") {
        x->shape.unpack(xn, xc, xd, xh, xw);
        dy->shape.unpack(yn, yc, yd, yh, yw);
    } else {
        x->shape.unpack(xn, xd, xh, xw, xc);
        dy->shape.unpack(yn, yd, yh, yw, yc);
    }
    wco = yc, wci = xc / groups;
    wh = kh;
    ww = kw;
    wd = kd;
    dw->set_shape(NanoVector(wco, wci, wd, wh, ww));
}

void CudnnConv3dBackwardWOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«Ty:" << dy->dtype();
    jk << "«Tw:" << dw->dtype();
}

static auto make_conv3d = get_op_info("cudnn_conv3d")
    .get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, int, string>();
static auto make_backwardx = get_op_info("cudnn_conv3d_backward_x")
    .get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, int, int, int, int, string>();


VarPtr CudnnConv3dBackwardWOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    int xn, xc, xd, xh, xw, wd, wh, ww, wci, wco, yn, yc, yd, yh, yw;

    if (xformat == "ncdhw") {
        x->shape.unpack(xn, xc, xd, xh, xw);
        dy->shape.unpack(yn, yc, yd, yh, yw);
    } else {
        x->shape.unpack(xn, xd, xh, xw, xc);
        dy->shape.unpack(yn, yd, yh, yw, yc);
    }

    if (v_index == 0) {
        return make_backwardx(dout, dy, xd, xh, xw, strided, strideh, stridew, paddingd, paddingh, paddingw, dilationd, dilationh, dilationw, groups, xformat);
    } else {
        return make_conv3d(x, dout, strided, strideh, stridew, paddingd, paddingh, paddingw, dilationd, dilationh, dilationw, groups, xformat);
    }
}

// unordered_map<string, cudnnConvolutionBwdFilterAlgo_t> bwdw_algo_cache;

#else // JIT
#ifdef JIT_cuda

#pragma clang diagnostic ignored "-Wtautological-compare"

EXTERN_LIB unordered_map<string, cudnnConvolutionBwdFilterAlgo_t> bwdw_algo_cache;

template <typename T_ELEM> __inline__  cudnnDataType_t getDataType();
template <> __inline__ cudnnDataType_t getDataType<half1>() { return CUDNN_DATA_HALF;   }
template <> __inline__ cudnnDataType_t getDataType<float>() { return CUDNN_DATA_FLOAT;  }

void CudnnConv3dBackwardWOp::jit_run() {
    auto w = dw;
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

    cudnnConvolutionBwdFilterAlgo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
    };
    int num_algos = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
    int perf_count;
    STACK_ALLOC(cudnnConvolutionBwdFilterAlgoPerf_t,perf_results,num_algos);
    cudnnConvolutionBwdFilterAlgo_t algo;
    bool benchmark=true;

    JK& jk = get_jk();
    jk.clear();
    jk << dimX[0] << "," << dimX[1] << "," << dimX[2] << "," << dimX[3] << "," << dimX[4] << ",";
    jk << dimW[0] << "," << dimW[1] << "," << dimW[2] << "," << dimW[3] << "," << dimW[4] << ",";
    jk << paddingd << paddingh << paddingw << "," << strided << strideh <<stridew << "," << dilationd << dilationh << dilationw << "," << groups << ".";
    auto iter = bwdw_algo_cache.find(jk.to_string());
    
    if (iter!=bwdw_algo_cache.end()) algo = iter->second;
    else {
        if (bwdw_algo_cache.size()>=max_cache_size) benchmark = false;
        if (benchmark) {
            size_t max_ws_size = 0;
            for (int i = 0; i < num_algos; i++) {
                size_t sz;
                cudnnStatus_t ret = cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_, cudnnIdesc, cudnnOdesc, cudnnConvDesc, cudnnFdesc, algos[i], &sz);
                // continue if use too much workspace
                if (sz > mem_info.total_cuda_ram * max_workspace_ratio) continue;
                if (CUDNN_STATUS_SUCCESS == ret && sz > max_ws_size) max_ws_size = sz;
            } 
            size_t allocation;
            void* ws = exe.temp_allocator->alloc(max_ws_size, allocation);
            checkCudaErrors(cudnnFindConvolutionBackwardFilterAlgorithmEx(
                handle_,
                cudnnIdesc, x->ptr<Tx>(),
                cudnnOdesc, y->ptr<Ty>(),
                cudnnConvDesc,
                cudnnFdesc, w->ptr<Tw>(),
                num_algos,
                &perf_count,
                perf_results,
                ws,
                max_ws_size));
            exe.temp_allocator->free(ws, max_ws_size, allocation);
        } else {
            checkCudaErrors(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                handle_,
                cudnnIdesc,
                cudnnOdesc,
                cudnnConvDesc,
                cudnnFdesc,
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
            bwdw_algo_cache[jk.to_string()] = algo;
            if (bwdw_algo_cache.size()==max_cache_size)
                LOGw << "backward w algorithm cache is full";
        }
    }

    // TODO: warp work space
    void *workSpace = 0;
    size_t workSpaceSize;
    checkCudaErrors (cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle_, cudnnIdesc, cudnnOdesc, cudnnConvDesc, 
        cudnnFdesc, algo, &workSpaceSize));
    size_t allocation;
    if (workSpaceSize > 0) {
        workSpace = exe.temp_allocator->alloc(workSpaceSize, allocation);
    }
    float alpha=1, beta=0;
    checkCudaErrors(cudnnConvolutionBackwardFilter(
        handle_,
        (void*)(&alpha),
        cudnnIdesc,  x->ptr<Tx>(),
        cudnnOdesc, y->ptr<Ty>(),
        cudnnConvDesc,
        algo,
        workSpace, workSpaceSize,
        (void*)(&beta),
        cudnnFdesc,  w->ptr<Tw>())
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
