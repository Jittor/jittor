// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cudnn_conv_op.h"
#include "cudnn_descriptor.h"
#include "cudnn_wrapper.h"
#include "cudnn_conv_plan.h"
#include "executor.h"
#include "ops/op_register.h"
#include "mem/mem_info.h"

using namespace std;

namespace jittor {

static inline int findc(const char* format, const char& c) {
    if (c==format[0]) return 0;
    if (c==format[1]) return 1;
    if (c==format[2]) return 2;
    ASSERT(c==format[3]) << "Not a valid format" << format << c;
    return 3;
}

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

CudnnConvOp::CudnnConvOp(Var* x, Var* w, int strideh, int stridew, int paddingh, int paddingw, int dilationh, int dilationw, int groups, string xformat, string wformat, string yformat)
    : x(x), w(w), strideh(strideh), stridew(stridew), paddingh(paddingh), paddingw(paddingw), dilationh(dilationh), dilationw(dilationw), groups(groups),
      xformat(move(xformat)), wformat(move(wformat)), yformat(move(yformat)) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    x->set_flag(VarFlags::_needed_by_backward);
    w->set_flag(VarFlags::_needed_by_backward);
    y = create_output(nullptr, dtype_infer(x->ns, w->ns));
    if (!this->yformat.size())
        this->yformat = this->xformat;
}

void CudnnConvOp::infer_shape() {
    ASSERTop(x->shape.size(),==,4);
    ASSERTop(w->shape.size(),==,4);
    int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
    get_shape(x, "abcd", xformat, xn, xc, xh, xw);
    get_shape(w, "oihw", wformat, wco, wci, wh, ww);
    ASSERTop(wci * groups,==,xc);
    yn = xn, yc = wco;
    yh = (xh+paddingh*2-wh*dilationh+dilationh-1)/strideh+1;
    yw = (xw+paddingw*2-ww*dilationw+dilationw-1)/stridew+1;
    set_shape(y, "abcd", yformat, yn, yc, yh, yw);
}

void CudnnConvOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«Ty:" << y->dtype();
    jk << "«Tw:" << w->dtype();
    jk << "«XFORMAT:" << xformat;
    jk << "«WFORMAT:" << wformat;
    jk << "«YFORMAT:" << yformat;
}
static auto make_backwardx = op_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, string, string, string>("cudnn_conv_backward_x");
static auto make_backwardw = op_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, string, string, string>("cudnn_conv_backward_w");
VarPtr CudnnConvOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    int xn, xc, xh, xw, wh, ww, wci, wco;
    // Read the spatial sizes through the layout strings ("abcd" is NCHW,
    // "acdb" NHWC; "oihw"/"ohwi" for the filter), as infer_shape does.
    // Comparing against a layout name the op never receives handed the
    // backward ops NHWC positions for an NCHW input.
    get_shape(x, "abcd", xformat, xn, xc, xh, xw);
    get_shape(w, "oihw", wformat, wco, wci, wh, ww);
    if (v_index == 0) {
        return make_backwardx(w, dout, xh, xw, strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups, xformat, wformat, yformat);
    } else {
        return make_backwardw(x, dout, wh, ww, strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups, xformat, wformat, yformat);
    }
}

unordered_map<string, cudnnConvolutionFwdAlgo_t> fwd_algo_cache;

#else // JIT
#ifdef JIT_cuda

#pragma clang diagnostic ignored "-Wtautological-compare"

EXTERN_LIB unordered_map<string, cudnnConvolutionFwdAlgo_t> fwd_algo_cache;
EXTERN_LIB int cudnn_benchmark;

void CudnnConvOp::jit_run() {
    cudnnHandle_t handle_ = cudnn_bind_stream();

    // Everything down to the backend fast path below is integer arithmetic on
    // Var shapes. The four legacy descriptors used to be created up here and
    // destroyed again the moment the fast path succeeded -- four Creates,
    // seven Sets and four Destroys per call, for a path that never looks at
    // them. They are built further down, only when the fallback needs them.
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
    #define filterFormat_ohwi CUDNN_TENSOR_NHWC

    int padA[] = {paddingh, paddingw};
    int convstrideA[] = {strideh, stridew};
    int dilationA[] = {dilationh, dilationw};
    // difference between
    // CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION
    // is the kernel rc order
    // currently, No perf difference is observed between
    // this two mode
    bool has_fp16_or_bf16 = x->dtype() == ns_float16
        || y->dtype() == ns_float16 || w->dtype() == ns_float16
        || x->dtype() == ns_bfloat16
        || y->dtype() == ns_bfloat16 || w->dtype() == ns_bfloat16;
    cudnnDataType_t conv_compute_type =
        cudnn_conv_compute_type(has_fp16_or_bf16, getDataType<Ty>());

    int conv_math_key = 0;
#ifndef IS_ROCM
    bool fp32_conv = x->dtype() == ns_float32
        && y->dtype() == ns_float32 && w->dtype() == ns_float32;
    cudnnMathType_t conv_math_type =
        cudnn_conv_math_type(has_fp16_or_bf16, fp32_conv);
    conv_math_key = static_cast<int>(conv_math_type);
#endif
    LOGvvv << "cudnn_conv precision select:"
        << "precision=" >> float32_precision_tier_name(float32_cudnn_tier())
        << "computeType=" >> cudnn_data_type_name(conv_compute_type)
        << "mathType=" >> cudnn_math_type_name(conv_math_key);

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
        ConvPlanRequest req = conv_plan_request(
            CONV_PLAN_FWD,
            getDataType<Tx>(), getDataType<Tw>(), getDataType<Ty>(),
            dimX, strideX, dimW,
            filterFormat_@WFORMAT == CUDNN_TENSOR_NHWC,
            dimY, strideY, padA, convstrideA, dilationA, conv_math_type);
        if (cudnn_conv_backend_run(req, x->ptr<Tx>(), w->ptr<Tw>(), y->ptr<Ty>()))
            return;
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
        CUDNN_CROSS_CORRELATION, conv_compute_type));
    // MIOpen requires groups to be set after descriptor initialization
    checkCudaErrors(cudnnSetConvolutionGroupCount( cudnnConvDesc, groups ));
#ifndef IS_ROCM
    checkCudaErrors(cudnnSetConvolutionMathType(cudnnConvDesc, conv_math_type));
#endif
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
    // Measure rather than predict, as both backward passes already do.
    // cuDNN's heuristic is not merely imprecise here: on ordinary fp32 3x3
    // convolutions it picks algorithms 2-3x slower than the measured best
    // (4x256x32x32 and 4x384x16x16 each ran 2.2x faster once measured), and it
    // was the only reason forward was the most expensive of the three passes in
    // a diffusers UNet step. The measurement is paid once per shape -- the
    // cache below keeps it -- and stops once the cache is full, which bounds
    // the cost for a workload whose shapes never repeat. Explicit
    // benchmark=0 still forces the heuristic.
    bool benchmark = cudnn_benchmark != 0;

    JK& jk = get_jk();
    jk.clear();
    jk << "x=" << x->dtype() << ":";
    jk << dimX[0] << "," << dimX[1] << "," << dimX[2] << "," << dimX[3] << ":";
    jk << strideX[0] << "," << strideX[1] << "," << strideX[2] << "," << strideX[3] << ";";
    jk << "w=" << w->dtype() << ":";
    jk << dimW[0] << "," << dimW[1] << "," << dimW[2] << "," << dimW[3] << ":";
    jk << static_cast<int>(filterFormat_@WFORMAT) << ";";
    jk << "y=" << y->dtype() << ":";
    jk << dimY[0] << "," << dimY[1] << "," << dimY[2] << "," << dimY[3] << ":";
    jk << strideY[0] << "," << strideY[1] << "," << strideY[2] << "," << strideY[3] << ";";
    jk << "conv=" << paddingh << "," << paddingw << ":";
    jk << strideh << "," << stridew << ":";
    jk << dilationh << "," << dilationw << ":" << groups << ";";
    jk << "compute=" << static_cast<int>(conv_compute_type) << ":";
    jk << "math=" << conv_math_key << ":";
    // A cached algorithm must still honor a workspace limit changed at runtime.
    jk << "workspace_ratio=" << max_workspace_ratio << ".";
    auto iter = fwd_algo_cache.find(jk.to_string());
    
    if (iter!=fwd_algo_cache.end()) algo = iter->second;
    else {
        if (fwd_algo_cache.size()>=max_cache_size) benchmark = false;
        if (benchmark) {
            size_t max_ws_size = 0;
            for (int i = 0; i < num_algos; i++) {
                size_t sz = 0;
                cudnnStatus_t ret = cudnnGetConvolutionForwardWorkspaceSize(
                    handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc, 
                    cudnnOdesc, algos[i], &sz);
                if (ret != CUDNN_STATUS_SUCCESS) continue;
                // continue if use too much workspace
                if (sz > mem_info.total_cuda_ram * max_workspace_ratio) continue;
                if (sz > max_ws_size) max_ws_size = sz;
            } 
            CudnnWorkspace search_ws(max_ws_size);
            checkCudaErrors(cudnnFindConvolutionForwardAlgorithmEx(
                handle_,
                cudnnIdesc, x->ptr<Tx>(),
                cudnnFdesc, w->ptr<Tw>(),
                cudnnConvDesc,
                cudnnOdesc, y->ptr<Ty>(),
                num_algos,
                &perf_count,
                perf_results,
                search_ws.ptr,
                search_ws.size));
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
        // Cache the choice whether it was measured or merely predicted.
        // cudnnGet*Algorithm_v7 is deterministic for a descriptor set, so its
        // answer is as reusable as a benchmarked one -- and it is a cuDNN call
        // per convolution, not a free lookup. Caching only the benchmarked
        // answer left the default path re-querying on every invocation: a
        // diffusers UNet step made 663 of these queries where 27 would do, and
        // that query, not the algorithm it returned, was the cost.
        if (fwd_algo_cache.size() < (size_t)max_cache_size) {
            fwd_algo_cache[jk.to_string()] = algo;
            if (fwd_algo_cache.size()==(size_t)max_cache_size)
                LOGw << "forward_ algorithm cache is full";
        }
    }

    size_t workSpaceSize;
    checkCudaErrors (cudnnGetConvolutionForwardWorkspaceSize(
        handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc, 
        cudnnOdesc, algo, &workSpaceSize) );
    CudnnWorkspace workSpace(workSpaceSize);
    float alpha=1, beta=0;
    checkCudaErrors(cudnnConvolutionForward(
        handle_,
        (void*)(&alpha),
        cudnnIdesc, x->ptr<Tx>(),
        cudnnFdesc, w->ptr<Tw>(),
        cudnnConvDesc,
        algo,
        workSpace.ptr, workSpace.size,
        (void*)(&beta),
        cudnnOdesc, y->ptr<Ty>())
    );
}
#endif
#endif // JIT

} // jittor
