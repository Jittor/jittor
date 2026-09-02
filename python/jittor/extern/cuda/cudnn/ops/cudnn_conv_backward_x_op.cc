// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>
//     Guowei Yang <471184555@qq.com>
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mem/allocator.h"
#include "var.h"
#include "cudnn_conv_backward_x_op.h"
#include "cudnn_descriptor.h"
#include "cudnn_wrapper.h"
#include "cudnn_conv_plan.h"
#include "executor.h"
#include "ops/op_register.h"
#include "mem/mem_info.h"

using namespace std;

static inline int findc(const char* format, const char& c) {
    if (c==format[0]) return 0;
    if (c==format[1]) return 1;
    if (c==format[2]) return 2;
    ASSERT(c==format[3]) << "Not a valid format" << format << c;
    return 3;
}

namespace jittor {

extern int use_tensorcore;
extern int cuda_allow_cudnn_tf32;

#ifndef JIT

static inline void get_shape(Var* x, const char* f, const string& format, int& a, int& b, int &c, int& d) {
    auto& shape = x->shape;
    a = shape[findc(format.c_str(), f[0])];
    b = shape[findc(format.c_str(), f[1])];
    c = shape[findc(format.c_str(), f[2])];
    d = shape[findc(format.c_str(), f[3])];
}

static inline void set_shape(Var* x, const char* f, const string& format, int a, int b, int c, int d) {
    int64 shape[4];
    shape[findc(format.c_str(), f[0])] = a;
    shape[findc(format.c_str(), f[1])] = b;
    shape[findc(format.c_str(), f[2])] = c;
    shape[findc(format.c_str(), f[3])] = d;
    x->set_shape(NanoVector(
        shape[0], shape[1], shape[2], shape[3]));
}

CudnnConvBackwardXOp::CudnnConvBackwardXOp(Var* w, Var* dy, int height, int width, int strideh, int stridew, int paddingh, int paddingw, int dilationh, int dilationw, int groups, string xformat, string wformat, string yformat) 
        : w(w), dy(dy), xh(height), xw(width), strideh(strideh), stridew(stridew), paddingh(paddingh), paddingw(paddingw), dilationh(dilationh), dilationw(dilationw), groups(groups),
      xformat(move(xformat)), wformat(move(wformat)), yformat(move(yformat)) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_manual_set_vnbb);
    w->flags.set(NodeFlags::_needed_by_backward);
    dy->flags.set(NodeFlags::_needed_by_backward);
    dx = create_output(nullptr, dtype_infer(dy->ns, w->ns));
}

void CudnnConvBackwardXOp::infer_shape() {
    ASSERTop(w->shape.size(),==,4);
    ASSERTop(dy->shape.size(),==,4);
    int xn, xc, wh, ww, wci, wco, yn, yc, yh, yw;
    get_shape(w, "oihw", wformat, wco, wci, wh, ww);
    get_shape(dy, "abcd", yformat, yn, yc, yh, yw);
    xn = yn, xc = wci * groups;
    set_shape(dx, "abcd", xformat, xn, xc, xh, xw);
}

void CudnnConvBackwardXOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << dx->dtype();
    jk << "«Ty:" << dy->dtype();
    jk << "«Tw:" << w->dtype();
    jk << "«XFORMAT:" << xformat;
    jk << "«WFORMAT:" << wformat;
    jk << "«YFORMAT:" << yformat;
}

static auto make_conv = get_op_info("cudnn_conv")
    .get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, string, string, string>();
static auto make_backwardw = get_op_info("cudnn_conv_backward_w")
    .get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, string, string, string>();

VarPtr CudnnConvBackwardXOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    int xn, xc, wh, ww, wci, wco, yn, yc, yd, yh, yw;
    w->shape.unpack(wco, wci, wh, ww);
    
    if (v_index == 0) {
        return make_backwardw(dout, dy, wh, ww, strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups, xformat, wformat, yformat);
    } else {
        return make_conv(dout, w, strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups, xformat, wformat, yformat);
    }
}
unordered_map<string, cudnnConvolutionBwdDataAlgo_t> bwdx_algo_cache;

#else // JIT
#ifdef JIT_cuda

#pragma clang diagnostic ignored "-Wtautological-compare"

EXTERN_LIB unordered_map<string, cudnnConvolutionBwdDataAlgo_t> bwdx_algo_cache;

void CudnnConvBackwardXOp::jit_run() {
    auto x = dx;
    auto y = dy;        
    cudnnHandle_t& handle_ = cudnn_handle;

    // Integer arithmetic on Var shapes down to the backend fast path; the
    // legacy descriptors are built below, only if the fallback needs them.
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
    auto ws = w->shape;
    // cuDNN takes filter dimensions in KCRS order whatever the memory layout;
    // read them through the layout string, as infer_shape does.
    int dimW[] = {
        (int)ws[findc("@WFORMAT", 'o')], (int)ws[findc("@WFORMAT", 'i')],
        (int)ws[findc("@WFORMAT", 'h')], (int)ws[findc("@WFORMAT", 'w')],
    };
    // cudnn only support this two format
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnSetFilterNdDescriptor
    #define filterFormat_oihw CUDNN_TENSOR_NCHW
    #define filterFormat_iohw CUDNN_TENSOR_NCHW
    #define filterFormat_ohwi CUDNN_TENSOR_NHWC

    int padA[] = {paddingh, paddingw};
    int convstrideA[] = {strideh, stridew};
    int dilationA[] = {dilationh, dilationw};
    // difference between
    // CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION
    // is the kernel rc order
    // currently, No perf difference is observed between
    // this two mode
    int conv_math_key = 0;
#ifndef IS_ROCM
    bool fp32_conv = x->dtype() == ns_float32
        && y->dtype() == ns_float32 && w->dtype() == ns_float32;
    cudnnMathType_t conv_math_type = CUDNN_DEFAULT_MATH;
    if (use_tensorcore || (fp32_conv && cuda_allow_cudnn_tf32)) {
        conv_math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
#if CUDNN_VERSION >= 8000
    } else if (fp32_conv) {
        conv_math_type = CUDNN_FMA_MATH;
#endif
    }
    conv_math_key = static_cast<int>(conv_math_type);
#endif

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

#ifndef IS_ROCM
    {
        // Backend-API plan cache; falls through to the legacy path below when
        // no plan can be built for this configuration.
        ConvPlanRequest req; memset(&req, 0, sizeof(req));
        req.kind = CONV_PLAN_BWD_DATA;
        req.dtype_x = getDataType<Tx>(); req.dtype_w = getDataType<Tw>(); req.dtype_y = getDataType<Ty>();
        for (int i=0; i<4; i++) {
            req.xdim[i] = dimX[i]; req.xstride[i] = strideX[i];
            req.ydim[i] = dimY[i]; req.ystride[i] = strideY[i];
            req.wdim[i] = dimW[i];
        }
        conv_plan_filter_strides(req.wstride, dimW, filterFormat_@WFORMAT == CUDNN_TENSOR_NHWC);
        req.pad[0] = paddingh; req.pad[1] = paddingw;
        req.stride[0] = strideh; req.stride[1] = stridew;
        req.dilation[0] = dilationh; req.dilation[1] = dilationw;
        req.allow_tf32 = conv_math_type == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
        req.benchmark = cudnn_benchmark != 0;
        if (cudnn_conv_backend_run(req, x->ptr<Tx>(), w->ptr<Tw>(), y->ptr<Ty>())) {
            return;
        }
    }
#endif

    // ---- legacy cuDNN API fallback: from here on descriptors are needed ----
    CudnnTensorDescriptor cudnnIdesc;
    CudnnFilterDescriptor cudnnFdesc;
    CudnnTensorDescriptor cudnnOdesc;
    CudnnConvolutionDescriptor cudnnConvDesc;

    checkCudaErrors(cudnnSetTensorNdDescriptor(
        cudnnIdesc, getDataType<Tx>(), 4, dimX, strideX));
    checkCudaErrors(cudnnSetFilterNdDescriptor(
        cudnnFdesc, getDataType<Tw>(), filterFormat_@WFORMAT, 4, dimW));
    checkCudaErrors(cudnnSetTensorNdDescriptor(
        cudnnOdesc, getDataType<Ty>(), 4, dimY, strideY));
    checkCudaErrors(cudnnSetConvolutionNdDescriptor(
        cudnnConvDesc, 2, padA, convstrideA, dilationA,
        CUDNN_CROSS_CORRELATION, getDataType<Ty>()));
    // MIOpen requires groups to be set after descriptor initialization
    checkCudaErrors(cudnnSetConvolutionGroupCount( cudnnConvDesc, groups ));
#ifndef IS_ROCM
    checkCudaErrors(cudnnSetConvolutionMathType(cudnnConvDesc, conv_math_type));
#endif
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
    jk << dimX[0] << "," << dimX[1] << "," << dimX[2] << "," << dimX[3] << ",";
    jk << dimW[0] << "," << dimW[1] << "," << dimW[2] << "," << dimW[3] << ",";
    jk << paddingh << paddingw << "," <<strideh <<stridew << "," << dilationh << dilationw << "," << groups << ".";
    jk << "math=" << conv_math_key << ".";
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
            CudnnWorkspace search_ws(max_ws_size);
            checkCudaErrors(cudnnFindConvolutionBackwardDataAlgorithmEx(
                handle_,
                cudnnFdesc, w->ptr<Tw>(),
                cudnnOdesc, y->ptr<Ty>(),
                cudnnConvDesc,
                cudnnIdesc, x->ptr<Tx>(),
                num_algos,
                &perf_count,
                perf_results,
                search_ws.ptr,
                search_ws.size));
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
        // Cache the choice whether it was measured or merely predicted.
        // cudnnGet*Algorithm_v7 is deterministic for a descriptor set, so its
        // answer is as reusable as a benchmarked one -- and it is a cuDNN call
        // per convolution, not a free lookup. Caching only the benchmarked
        // answer left the default path re-querying on every invocation: a
        // diffusers UNet step made 663 of these queries where 27 would do, and
        // that query, not the algorithm it returned, was the cost.
        if (bwdx_algo_cache.size() < (size_t)max_cache_size) {
            bwdx_algo_cache[jk.to_string()] = algo;
            if (bwdx_algo_cache.size()==(size_t)max_cache_size)
                LOGw << "backward x algorithm cache is full";
        }
    }

    // TODO: warp work space
    size_t workSpaceSize;
    checkCudaErrors (cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle_, cudnnFdesc, cudnnOdesc, cudnnConvDesc, 
        cudnnIdesc, algo, &workSpaceSize));
    CudnnWorkspace workSpace(workSpaceSize);
    float alpha=1, beta=0;
    checkCudaErrors(cudnnConvolutionBackwardData(
        handle_,
        (void*)(&alpha),
        cudnnFdesc, w->ptr<Tw>(),
        cudnnOdesc, y->ptr<Ty>(),
        cudnnConvDesc,
        algo,
        workSpace.ptr, workSpace.size,
        (void*)(&beta),
        cudnnIdesc, x->ptr<Tx>())
    );
}
#endif
#endif // JIT

} // jittor
