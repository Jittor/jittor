// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Dun Liang <randonlang@gmail.com>
//     Guowei Yang <471184555@qq.com>
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mem/allocator.h"
#include "var.h"
#include "cudnn_conv_backward_w_op.h"
#include "cudnn_warper.h"
#include "executor.h"

using namespace std;

namespace jittor {
static inline int findc(const string& format, const char& c) {
    if (c==format[0]) return 0;
    if (c==format[1]) return 1;
    if (c==format[2]) return 2;
    ASSERT(c==format[3]) << "Not a valid format" << format << c;
    return 3;
}

#ifndef JIT
static inline void get_shape(Var* x, const char* f, const string& format, int& a, int& b, int &c, int& d) {
    auto& shape = x->shape;
    a = shape[findc(format, f[0])];
    b = shape[findc(format, f[1])];
    c = shape[findc(format, f[2])];
    d = shape[findc(format, f[3])];
}

static inline void set_shape(Var* x, const char* f, const string& format, int a, int b, int c, int d) {
    int64 shape[4];
    shape[findc(format, f[0])] = a;
    shape[findc(format, f[1])] = b;
    shape[findc(format, f[2])] = c;
    shape[findc(format, f[3])] = d;
    x->set_shape(NanoVector(
        shape[0], shape[1], shape[2], shape[3]));
}

CudnnConvBackwardWOp::CudnnConvBackwardWOp(Var* x, Var* dy, int kh, int kw, int stride, int padding, int dilation, int groups, string xformat, string wformat, string yformat)
        : x(x), dy(dy), kh(kh), kw(kw), stride(stride), padding(padding), dilation(dilation), groups(groups),
      xformat(move(xformat)), wformat(move(wformat)), yformat(move(yformat)) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0);
    dw = create_output(nullptr, dtype_infer(dy->ns, x->ns));
}

void CudnnConvBackwardWOp::infer_shape() {
    ASSERTop(x->shape.size(),==,4);
    ASSERTop(dy->shape.size(),==,4);
    int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
    get_shape(x, "abcd", xformat, xn, xc, xh, xw);
    get_shape(dy, "abcd", yformat, yn, yc, yh, yw);
    wco = yc, wci = xc / groups;
    wh = kh;
    ww = kw;
    set_shape(dw, "oihw", wformat, wco, wci, wh, ww);
}

void CudnnConvBackwardWOp::jit_prepare() {
    add_jit_define("Tx", x->dtype());
    add_jit_define("Ty", dy->dtype());
    add_jit_define("Tw", dw->dtype());
    add_jit_define("XFORMAT", xformat);
    add_jit_define("WFORMAT", wformat);
    add_jit_define("YFORMAT", yformat);
}
unordered_map<string, cudnnConvolutionBwdFilterAlgo_t> bwdw_algo_cache;

#else // JIT
#ifdef JIT_cuda

#pragma clang diagnostic ignored "-Wtautological-compare"

extern unordered_map<string, cudnnConvolutionBwdFilterAlgo_t> bwdw_algo_cache;

template <typename T_ELEM> __inline__  cudnnDataType_t getDataType();
template <> __inline__ cudnnDataType_t getDataType<half1>() { return CUDNN_DATA_HALF;   }
template <> __inline__ cudnnDataType_t getDataType<float>() { return CUDNN_DATA_FLOAT;  }

void CudnnConvBackwardWOp::jit_run() {
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

    int dimX[] = {
        (int)x->shape[findc("@XFORMAT", 'a')], // n
        (int)x->shape[findc("@XFORMAT", 'b')], // c
        (int)x->shape[findc("@XFORMAT", 'c')], // h
        (int)x->shape[findc("@XFORMAT", 'd')], // w
    };
    int _strideX[] = {0,0,0,1};
    for (int i=2; i>=0; i--) _strideX[i] = _strideX[i+1] * x->shape[i+1];
    int strideX[] = {
        _strideX[findc("@XFORMAT", 'a')], // n
        _strideX[findc("@XFORMAT", 'b')], // c
        _strideX[findc("@XFORMAT", 'c')], // h
        _strideX[findc("@XFORMAT", 'd')], // w
    };
    // dimX: nchw
    checkCudaErrors(cudnnSetTensorNdDescriptor(
        cudnnIdesc, getDataType<Tx>(),
        4, dimX, strideX
    ));

    auto ws = w->shape;
    int dimW[] = {(int)ws[0],(int)ws[1],(int)ws[2],(int)ws[3]};
    // cudnn only support this two format
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnSetFilterNdDescriptor
    #define filterFormat_oihw CUDNN_TENSOR_NCHW
    #define filterFormat_ohwi CUDNN_TENSOR_NHWC

    // dimW: KCRS(oihw)
    checkCudaErrors(cudnnSetFilterNdDescriptor(
        cudnnFdesc, getDataType<Tw>(),
        filterFormat_@WFORMAT, 4, dimW
    ));

    int padA[] = {padding, padding};
    int convstrideA[] = {stride, stride};
    int dilationA[] = {dilation, dilation};
    // difference between
    // CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION
    // is the kernel rc order
    // currently, No perf difference is observed between
    // this two mode
    checkCudaErrors(cudnnSetConvolutionNdDescriptor(
        cudnnConvDesc, /*convDim=*/2,
        padA, convstrideA, dilationA,
        CUDNN_CROSS_CORRELATION, getDataType<Ty>()
    ));

    // using tensor core
    // checkCudaErrors( cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH) );

    int dimY[] = {
        (int)y->shape[findc("@YFORMAT", 'a')], // n
        (int)y->shape[findc("@YFORMAT", 'b')], // c
        (int)y->shape[findc("@YFORMAT", 'c')], // h
        (int)y->shape[findc("@YFORMAT", 'd')], // w
    };
    int _strideY[] = {0,0,0,1};
    for (int i=2; i>=0; i--) _strideY[i] = _strideY[i+1] * y->shape[i+1];
    int strideY[] = {
        _strideY[findc("@YFORMAT", 'a')], // n
        _strideY[findc("@YFORMAT", 'b')], // c
        _strideY[findc("@YFORMAT", 'c')], // h
        _strideY[findc("@YFORMAT", 'd')], // w
    };
    checkCudaErrors(cudnnSetTensorNdDescriptor(
        cudnnOdesc, getDataType<Ty>(),
        4, dimY, strideY
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
    cudnnConvolutionBwdFilterAlgoPerf_t perf_results[num_algos];
    cudnnConvolutionBwdFilterAlgo_t algo;
    bool benchmark=true;

    jk.clear();
    jk << dimX[0] << "," << dimX[1] << "," << dimX[2] << "," << dimX[3] << ",";
    jk << dimW[0] << "," << dimW[1] << "," << dimW[2] << "," << dimW[3] << ",";
    jk << padding << "," <<stride << "," << dilation << "," << groups << ".";
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
                if (sz*4 > mem_info.total_cuda_ram) continue;
                if (CUDNN_STATUS_SUCCESS == ret && sz > max_ws_size) max_ws_size = sz;
            } 
            size_t allocation;
            void* ws = exe.allocator->alloc(max_ws_size, allocation);
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
            exe.allocator->free(ws, max_ws_size, allocation);
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
        workSpace = exe.allocator->alloc(workSpaceSize, allocation);
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
        exe.allocator->free(workSpace, workSpaceSize, allocation);
        
    checkCudaErrors(cudnnDestroyTensorDescriptor( cudnnIdesc ));
    checkCudaErrors(cudnnDestroyFilterDescriptor( cudnnFdesc ));
    checkCudaErrors(cudnnDestroyTensorDescriptor( cudnnOdesc ));
    checkCudaErrors(cudnnDestroyConvolutionDescriptor( cudnnConvDesc ));
}
#endif
#endif // JIT

} // jittor
