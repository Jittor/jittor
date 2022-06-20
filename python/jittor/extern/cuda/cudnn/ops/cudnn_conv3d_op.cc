// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cudnn_conv3d_op.h"
#include "cudnn_wrapper.h"
#include "executor.h"
#include "ops/op_register.h"

using namespace std;

namespace jittor {

extern int use_tensorcore;

#pragma GCC diagnostic ignored "-Wunused-variable"

#ifndef JIT

CudnnConv3dOp::CudnnConv3dOp(Var* x, Var* w, int strided, int strideh, int stridew, int paddingd, int paddingh, int paddingw, int dilationd, int dilationh, int dilationw, int groups, string xformat)
    : x(x), w(w), strided(strided), strideh(strideh), stridew(stridew), paddingd(paddingd), paddingh(paddingh), paddingw(paddingw), dilationd(dilationd), dilationh(dilationh), dilationw(dilationw), groups(groups),
      xformat(move(xformat)) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_manual_set_vnbb);
    x->flags.set(NodeFlags::_needed_by_backward);
    w->flags.set(NodeFlags::_needed_by_backward);
    y = create_output(nullptr, dtype_infer(x->ns, w->ns));
}

void CudnnConv3dOp::infer_shape() {
    ASSERTop(x->shape.size(),==,5);
    ASSERTop(w->shape.size(),==,5);
    int xn, xc, xd, xh, xw, wd, wh, ww, wci, wco, yn, yc, yd, yh, yw;
    if (xformat == "ncdhw")
        x->shape.unpack(xn, xc, xd, xh, xw);
    else
        x->shape.unpack(xn, xd, xh, xw, xc);
    w->shape.unpack(wco, wci, wd, wh, ww);
    ASSERTop(wci * groups,==,xc);
    yn = xn, yc = wco;
    yd = (xd+paddingd*2-wd*dilationd+dilationd-1)/strided+1;
    yh = (xh+paddingh*2-wh*dilationh+dilationh-1)/strideh+1;
    yw = (xw+paddingw*2-ww*dilationw+dilationw-1)/stridew+1;
    if (xformat == "ncdhw")
        y->set_shape(NanoVector(yn, yc, yd, yh, yw));
    else
        y->set_shape(NanoVector(yn, yd, yh, yw, yc));
}

void CudnnConv3dOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«Ty:" << y->dtype();
    jk << "«Tw:" << w->dtype();
}

static auto make_backwardx = get_op_info("cudnn_conv3d_backward_x")
    .get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, int, int, int, int, string>();
static auto make_backwardw = get_op_info("cudnn_conv3d_backward_w")
    .get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, int, int, int, int, string>();

VarPtr CudnnConv3dOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    int xn, xc, xd, xh, xw, wd, wh, ww, wci, wco, yn, yc, yd, yh, yw;
    if (xformat == "ncdhw")
        x->shape.unpack(xn, xc, xd, xh, xw);
    else
        x->shape.unpack(xn, xd, xh, xw, xc);
    w->shape.unpack(wco, wci, wd, wh, ww);
    if (v_index == 0) {
        return make_backwardx(w, dout, xd, xh, xw, strided, strideh, stridew, paddingd, paddingh, paddingw, dilationd, dilationh, dilationw, groups, xformat);
    } else {
        return make_backwardw(x, dout, wd, wh, ww, strided, strideh, stridew, paddingd, paddingh, paddingw, dilationd, dilationh, dilationw, groups, xformat);
    }
}

// unordered_map<string, cudnnConvolutionFwdAlgo_t> fwd_algo_cache;

#else // JIT
#ifdef JIT_cuda

#pragma clang diagnostic ignored "-Wtautological-compare"

EXTERN_LIB unordered_map<string, cudnnConvolutionFwdAlgo_t> fwd_algo_cache;

template <typename T_ELEM> __inline__  cudnnDataType_t getDataType();
template <> __inline__ cudnnDataType_t getDataType<half1>() { return CUDNN_DATA_HALF;   }
template <> __inline__ cudnnDataType_t getDataType<float>() { return CUDNN_DATA_FLOAT;  }

void CudnnConv3dOp::jit_run() {
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

    cudnnConvolutionFwdAlgo_t algos[] = {
         CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };
    int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int perf_count;
    STACK_ALLOC(cudnnConvolutionFwdAlgoPerf_t,perf_results,num_algos);
    cudnnConvolutionFwdAlgo_t algo;
    bool benchmark=true;

    JK& jk = get_jk();
    jk.clear();
    jk << dimX[0] << "," << dimX[1] << "," << dimX[2] << "," << dimX[3] << "," << dimX[4] << ",";
    jk << dimW[0] << "," << dimW[1] << "," << dimW[2] << "," << dimW[3] << "," << dimW[4] << ",";
    jk << paddingd << paddingh << paddingw << "," << strided << strideh <<stridew << "," << dilationd << dilationh << dilationw << "," << groups << ".";
    auto iter = fwd_algo_cache.find(jk.to_string());
    
    if (iter!=fwd_algo_cache.end()) algo = iter->second;
    else {
        if (fwd_algo_cache.size()>=max_cache_size) benchmark = false;
        if (benchmark) {
            size_t max_ws_size = 0;
            for (int i = 0; i < num_algos; i++) {
                size_t sz;
                cudnnStatus_t ret = cudnnGetConvolutionForwardWorkspaceSize(
                    handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc, 
                    cudnnOdesc, algos[i], &sz);
                // continue if use too much workspace
                if (sz > mem_info.total_cuda_ram * max_workspace_ratio) continue;
                if (CUDNN_STATUS_SUCCESS == ret && sz > max_ws_size) max_ws_size = sz;
            } 
            size_t allocation;
            void* ws = exe.temp_allocator->alloc(max_ws_size, allocation);
            checkCudaErrors(cudnnFindConvolutionForwardAlgorithmEx(
                handle_,
                cudnnIdesc, x->ptr<Tx>(),
                cudnnFdesc, w->ptr<Tw>(),
                cudnnConvDesc,
                cudnnOdesc, y->ptr<Ty>(),
                num_algos,
                &perf_count,
                perf_results,
                ws,
                max_ws_size));
            exe.temp_allocator->free(ws, max_ws_size, allocation);
        } else {
            checkCudaErrors(cudnnGetConvolutionForwardAlgorithm_v7(
                handle_,
                cudnnIdesc,
                cudnnFdesc,
                cudnnConvDesc,
                cudnnOdesc,
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
            fwd_algo_cache[jk.to_string()] = algo;
            if (fwd_algo_cache.size()==max_cache_size)
                LOGw << "forward_ algorithm cache is full";
        }
    }

    // TODO: warp work space
    void *workSpace = 0;
    size_t workSpaceSize;
    checkCudaErrors (cudnnGetConvolutionForwardWorkspaceSize(
        handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc, 
        cudnnOdesc, algo, &workSpaceSize) );
    size_t allocation;
    if (workSpaceSize > 0) {
        workSpace = exe.temp_allocator->alloc(workSpaceSize, allocation);
    }
    float alpha=1, beta=0;
    checkCudaErrors(cudnnConvolutionForward(
        handle_,
        (void*)(&alpha),
        cudnnIdesc, x->ptr<Tx>(),
        cudnnFdesc, w->ptr<Tw>(),
        cudnnConvDesc,
        algo,
        workSpace, workSpaceSize,
        (void*)(&beta),
        cudnnOdesc, y->ptr<Ty>())
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
