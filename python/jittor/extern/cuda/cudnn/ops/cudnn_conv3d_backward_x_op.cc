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
#include "cudnn_conv3d_backward_x_op.h"
#include "cudnn_descriptor.h"
#include "cudnn_wrapper.h"
#include "executor.h"
#include "ops/op_register.h"
#include "mem/mem_info.h"

using namespace std;

namespace jittor {

#pragma GCC diagnostic ignored "-Wunused-variable"

#ifndef JIT

CudnnConv3dBackwardXOp::CudnnConv3dBackwardXOp(Var* w, Var* dy, int depth, int height, int width, int strided, int strideh, int stridew, int paddingd, int paddingh, int paddingw, int dilationd, int dilationh, int dilationw, int groups, string xformat) 
        : w(w), dy(dy), xd(depth), xh(height), xw(width), strided(strided), strideh(strideh), stridew(stridew), paddingd(paddingd), paddingh(paddingh), paddingw(paddingw), dilationd(dilationd), dilationh(dilationh), dilationw(dilationw), groups(groups),
      xformat(move(xformat)) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    w->set_flag(VarFlags::_needed_by_backward);
    dy->set_flag(VarFlags::_needed_by_backward);
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


static auto make_conv3d = op_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, int, string>("cudnn_conv3d");
static auto make_backwardw = op_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, int, int, int, int, string>("cudnn_conv3d_backward_w");


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
EXTERN_LIB int cudnn_benchmark;

template <typename T_ELEM> __inline__  cudnnDataType_t getDataType();
// template <> __inline__ cudnnDataType_t getDataType<half1>() { return CUDNN_DATA_HALF;   }
// template <> __inline__ cudnnDataType_t getDataType<float>() { return CUDNN_DATA_FLOAT;  }

void CudnnConv3dBackwardXOp::jit_run() {
    auto x = dx;
    auto y = dy;        
    cudnnHandle_t handle_ = cudnn_bind_stream();

    // Owned, so a throw anywhere below releases them. There is no backend
    // fast path for 3-D yet (6.B14 left these on the legacy API), so unlike
    // the 2-D ops they are built unconditionally.
    CudnnTensorDescriptor cudnnIdesc;
    CudnnFilterDescriptor cudnnFdesc;
    CudnnTensorDescriptor cudnnOdesc;
    CudnnConvolutionDescriptor cudnnConvDesc;

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
    bool has_fp16_or_bf16 = x->dtype() == ns_float16
        || y->dtype() == ns_float16 || w->dtype() == ns_float16
        || x->dtype() == ns_bfloat16
        || y->dtype() == ns_bfloat16 || w->dtype() == ns_bfloat16;
    cudnnDataType_t conv_compute_type =
        cudnn_conv_compute_type(has_fp16_or_bf16, getDataType<Ty>());
    checkCudaErrors(cudnnSetConvolutionNdDescriptor(
        cudnnConvDesc, 3,
        padA, convstrideA, dilationA,
        CUDNN_CROSS_CORRELATION, conv_compute_type
    ));
    // MIOpen requires groups to be set after descriptor initialization
    checkCudaErrors(cudnnSetConvolutionGroupCount( cudnnConvDesc, groups ));

    int conv_math_key = 0;
#ifndef IS_ROCM
    bool fp32_conv = x->dtype() == ns_float32
        && y->dtype() == ns_float32 && w->dtype() == ns_float32;
    cudnnMathType_t conv_math_type =
        cudnn_conv_math_type(has_fp16_or_bf16, fp32_conv);
    checkCudaErrors(cudnnSetConvolutionMathType(cudnnConvDesc, conv_math_type));
    conv_math_key = static_cast<int>(conv_math_type);
#endif
    LOGvvv << "cudnn_conv3d_backward_x precision select:"
        << "precision=" >> float32_precision_tier_name(float32_cudnn_tier())
        << "computeType=" >> cudnn_data_type_name(conv_compute_type)
        << "mathType=" >> cudnn_math_type_name(conv_math_key);


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
    // Same criterion as conv2d: cudnn_benchmark alone decides.
    // Half precision used to be excluded here, which meant a 3D half
    // convolution was never measured and the flag did nothing in 3D
    // backward at all.
    bool benchmark = cudnn_benchmark != 0;

    JK& jk = get_jk();
    jk.clear();
    // conv3d shares this cache with conv2d, so the key needs a namespace of
    // its own; and it has to carry the dtypes, the output extent, the compute
    // type and the workspace budget, or an fp32 and an fp16 convolution of the
    // same shape get one another's algorithm and a changed
    // max_workspace_ratio never invalidates anything.
    jk << "conv3d.bwdx;";
    jk << "x=" << x->dtype() << ":";
    jk << dimX[0] << "," << dimX[1] << "," << dimX[2] << "," << dimX[3] << "," << dimX[4] << ";";
    jk << "w=" << w->dtype() << ":";
    jk << dimW[0] << "," << dimW[1] << "," << dimW[2] << "," << dimW[3] << "," << dimW[4] << ";";
    jk << "y=" << y->dtype() << ":";
    jk << dimY[0] << "," << dimY[1] << "," << dimY[2] << "," << dimY[3] << "," << dimY[4] << ";";
    jk << "conv=" << paddingd << paddingh << paddingw << "," << strided << strideh <<stridew << "," << dilationd << dilationh << dilationw << "," << groups << ";";
    jk << "compute=" << static_cast<int>(conv_compute_type) << ":";
    jk << "math=" << conv_math_key << ":";
    jk << "workspace_ratio=" << max_workspace_ratio << ".";
    LOGvvv << "cudnn_conv3d bwdx algo cache key:" << jk.to_string();
    auto iter = bwdx_algo_cache.find(jk.to_string());
    
    if (iter!=bwdx_algo_cache.end()) algo = iter->second;
    else {
        bool cache_algo = bwdx_algo_cache.size() < max_cache_size;
        if (!cache_algo) benchmark = false;
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
        if (cache_algo) {
            bwdx_algo_cache[jk.to_string()] = algo;
            if (bwdx_algo_cache.size()==max_cache_size)
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
