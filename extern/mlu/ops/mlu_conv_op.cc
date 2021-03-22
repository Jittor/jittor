// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Wenyang Zhou <576825820@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "mlu_conv_op.h"

#include <string.h>
#include <iostream>
#include <sstream>
#include "./cnml.h"
#include "./cnrt.h"
#include "mlu_warper.h"
#include <chrono>

using namespace std;

namespace jittor {

static inline int findc(const string& format, const char& c) {
    if (c==format[0]) return 0;
    if (c==format[1]) return 1;
    if (c==format[2]) return 2;
    ASSERT(c==format[3]) << "Not a valid format" << format << c;
    return 3;
}

static inline void get_shape(Var* x, const char* f, const string& format, int& a, int& b, int &c, int& d) {
    auto& shape = x->shape;
    a = shape[findc(format, f[0])];
    b = shape[findc(format, f[1])];
    c = shape[findc(format, f[2])];
    d = shape[findc(format, f[3])];
}

#ifndef JIT

static inline void set_shape(Var* x, const char* f, const string& format, int a, int b, int c, int d) {
    int64 shape[4];
    shape[findc(format, f[0])] = a;
    shape[findc(format, f[1])] = b;
    shape[findc(format, f[2])] = c;
    shape[findc(format, f[3])] = d;
    x->set_shape(NanoVector(
        shape[0], shape[1], shape[2], shape[3]));
}

MluConvOp::MluConvOp(Var* x, Var* w, int strideh, int stridew, int paddingh, int paddingw, int dilationh, int dilationw, int groups, string xformat, string wformat, string yformat)
    : x(x), w(w), strideh(strideh), stridew(stridew), paddingh(paddingh), paddingw(paddingw), dilationh(dilationh), dilationw(dilationw), groups(groups),
      xformat(move(xformat)), wformat(move(wformat)), yformat(move(yformat)) {
    y = create_output(nullptr, dtype_infer(x->ns, w->ns));
    if (!this->yformat.size())
        this->yformat = this->xformat;
}

void MluConvOp::infer_shape() {
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

static const char* short_type(Var* x) {
    if (x->is_float()) {
        if (x->dsize()==4) return "f32";
        if (x->dsize()==8) return "f64";
        if (x->dsize()==2) return "f16";
        return "f8";
    } else {
        if (x->dsize()==4) return "s32";
        if (x->dsize()==8) return "s64";
        if (x->dsize()==2) return "s16";
        return "s8";
    }
}

void MluConvOp::jit_prepare(JK& jk) {
    jk << _CS("[Txd:") << x->dtype();
    jk << _CS("][Tyd:") << y->dtype();
    jk << _CS("][Twd:") << w->dtype();
    jk << _CS("][Tx:") << short_type(x);
    jk << _CS("][Tw:") << short_type(w);
    jk << _CS("][Ty:") << short_type(y);
    jk << _CS("][XFORMAT:") << xformat;
    jk << _CS("][WFORMAT:") << wformat;
    jk << _CS("][YFORMAT:") << yformat;
    jk << ']';
}

#else // JIT
#ifdef JIT_cpu
#pragma clang diagnostic ignored "-Wtautological-compare"
void MluConvOp::jit_run() {
    // cnrtInit(0);
    // cnmlInit(0);
    const int coreNum = 4;
    const int dimNum = 4;

    int ni, ci, hi, wi, no, co, cw, ho, wo, kh, kw;
    get_shape(x, "abcd", xformat, ni, ci, hi, wi);
    get_shape(w, "oihw", wformat, co, cw, kh, kw);
    get_shape(y, "abcd", yformat, no, co, ho, wo);

    /*
    CNN计算量
    FLOPs (乘法) =  co*  H * W * (ci * kw * kh)   其中H， W代表输出特征的宽和高
    FLOPs (加法(w)) =  co*  H * W * (ci * kw * kh - 1)   其中H， W代表输出特征的宽和高
    FLOPs (加法(b)) =  co*  H * W * (1)   其中H， W代表输出特征的宽和高
    */

    // count input, filter, bias, output nums
    int input_count = ni * hi * wi * ci;
    int filter_count = co * kh * kw * cw;
    int output_count = no * ho * wo * co;
    // printf("%d %d %d %d\n", ni, hi, wi, ci);
    // printf("%d %d %d %d\n", co, kh, kw, cw);
    // printf("%d %d %d %d\n", no, ho, wo, co);
    // printf("%d %d %d %d %d %d\n", strideh, stridew, dilationh, dilationw, paddingh * 2, paddingw * 2);

    float *input_cpu_data = (float*)x->mem_ptr;
    float *filter_cpu_data = (float*)w->mem_ptr;
    float *output_cpu_data = (float*)y->mem_ptr;

    // prepare buffer to store the converted data after calling cnrt-cast function
    int16_t *input_cpu_ptr = (int16_t *)malloc(input_count * sizeof(int16_t));
    int16_t *output_cpu_ptr = (int16_t *)malloc(output_count * sizeof(int16_t));
  #ifdef FILTER_DATA_FP32
    float *filter_cpu_ptr = (float *)malloc(filter_count * sizeof(float));
  #else
    int8_t *filter_cpu_ptr = (int8_t *)malloc(filter_count * sizeof(int8_t));
  #endif

    // converts data type for mlu computing
    cnrtCastDataType(input_cpu_data, CNRT_FLOAT32, input_cpu_ptr, CNRT_FLOAT16, input_count, NULL);
    // u should set value depending op the data or your own needs
    int filter_position = -6;
    float filter_scale = 1, filter_offset = 0;
  #ifdef FILTER_DATA_FP32
    memcpy(filter_cpu_ptr, filter_cpu_data, filter_count * sizeof(float));
  #else
    // prepare filter tensor quant param for filter data
    cnrtQuantizedParam_t filter_quant_param;
    cnrtCreateQuantizedParam(&filter_quant_param, filter_position, filter_scale, filter_offset);
    cnrtCastDataType(filter_cpu_data, CNRT_FLOAT32, filter_cpu_ptr, CNRT_INT8, filter_count,
                    filter_quant_param);
  #endif

    // set tensor shapes
    int input_shape[] = {ni, ci, hi, wi};
    int filter_shape[] = {co, cw, kh, kw};
    int output_shape[] = {no, co, ho, wo};

    // prepare input tensor
    cnmlTensor_t input_tensor = NULL;
    cnmlCreateTensor_V2(&input_tensor, CNML_TENSOR);
    cnmlSetTensorShape_V2(input_tensor, dimNum, input_shape, NULL);
    cnmlSetTensorDataType(input_tensor, CNML_DATA_FLOAT16);

    // prepare filter tensor
    cnmlTensor_t filter_tensor = NULL;
    cnmlCreateTensor_V2(&filter_tensor, CNML_FILTER);
    cnmlSetTensorShape_V2(filter_tensor, dimNum, filter_shape, NULL);
  #ifdef FILTER_DATA_FP32
    cnmlSetTensorDataType(filter_tensor, CNML_DATA_FLOAT32);
  #else
    cnmlSetTensorDataType(filter_tensor, CNML_DATA_INT8);
    // set filter tensor quant scale and position
    cnmlSetQuantizedPosition(filter_tensor, filter_position);
    cnmlSetQuantizedScale(filter_tensor, filter_scale);
  #endif

    // prepare output tensor
    cnmlTensor_t output_tensor = NULL;
    cnmlCreateTensor_V2(&output_tensor, CNML_TENSOR);
    cnmlSetTensorShape_V2(output_tensor, dimNum, output_shape, NULL);
    cnmlSetTensorDataType(output_tensor, CNML_DATA_FLOAT16);

    // bind cpu filter to cnml const tensor
    cnmlBindConstData_V2(filter_tensor, filter_cpu_ptr, false);

    // create conv op ptr and conv_param
    cnmlBaseOp_t conv_op = NULL;
    cnmlConvOpParam_t conv_param;

    // set conv op
    cnmlCreateConvOpParam(&conv_param, strideh, stridew, dilationh, dilationw, paddingh * 2, paddingw * 2);
    if (groups > 1) {
      cnmlCreateConvGroupOp(&conv_op, conv_param, input_tensor, output_tensor, filter_tensor, NULL, groups);
    }
    else {
      cnmlCreateConvOp(&conv_op, conv_param, input_tensor, output_tensor, filter_tensor, NULL);
    }

    // u should set value depending op the data or your own needs
    int input_position = -6;
    float input_scale = 1, input_offset = 0;
    // prepare input tensor quant param for conv op
    cnmlQuantizedParam_t input_quant_param;
    // create quant-param when setting computing datatype for conv op, please set offset 0 here
    cnmlCreateQuantizedParam(&input_quant_param, input_position, input_scale, input_offset);
    // setup conv op computing datatype
    cnmlSetOperationComputingDataType(conv_op, input_tensor, CNML_DATA_INT8, input_quant_param);

  #ifdef FILTER_DATA_FP32
    // prepare filter tensor quant param for conv op
    cnmlQuantizedParam_t filter_compute_quant;
    cnmlCreateQuantizedParam(&filter_compute_quant, filter_position, filter_scale, filter_offset);
    // setup conv op computing datatype
    cnmlSetOperationComputingDataType(conv_op, filter_tensor, CNML_DATA_INT8, filter_compute_quant);
  #endif

    // setup conv op computing layout
    cnmlSetOperationComputingLayout(conv_op, CNML_NCHW);

    const cnmlCoreVersion_t coreVersion = CNML_MLU270;

    // compile op
    cnmlSetBaseOpCoreVersion(conv_op, coreVersion);
    cnmlSetBaseOpCorenum(conv_op, coreNum);
    cnmlCompileBaseOp_V2(conv_op);

    // mlu buffer ptr
    void *input_mlu_ptr = NULL;
    void *output_mlu_ptr = NULL;

    // malloc cnml tensor
    cnrtMalloc(&input_mlu_ptr, input_count * sizeof(int16_t));
    cnrtMalloc(&output_mlu_ptr, output_count * sizeof(int16_t));
    // copy input to cnml buffer
    cnrtMemcpy(input_mlu_ptr, input_cpu_ptr, input_count * sizeof(int16_t),
              CNRT_MEM_TRANS_DIR_HOST2DEV);

    // compute on mlu
    // set cnrt queue
    // cnrtQueue_t queue;
    // cnrtCreateQueue(&queue);

    if (0) {
      int operations = co * ho * wo * (2 * ci * kw * kh - 1);
      int n = 1000;
      for (int i=0; i<100; i++) {
          cnmlComputeConvOpForward_V4(conv_op, NULL, input_mlu_ptr, NULL, output_mlu_ptr, queue, NULL);
      }
      JT_MLU_CHECK(cnrtSyncQueue(queue));
      auto start = std::chrono::high_resolution_clock::now();
      for (int i=0; i<n; i++) {
          cnmlComputeConvOpForward_V4(conv_op, NULL, input_mlu_ptr, NULL, output_mlu_ptr, queue, NULL);
      }
      JT_MLU_CHECK(cnrtSyncQueue(queue));
      auto finish = std::chrono::high_resolution_clock::now();
      auto total_ns =  (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() / 1000. / 1000.;
      cout << "FLOPS: " << 1.*no*operations*n/total_ns << " || Operations: " << operations << " || total_ns: " << total_ns << endl;
    }

    // compute conv op on MLU
    if (groups > 1) {
      cnmlComputeConvGroupOpForward_V4(conv_op, NULL, input_mlu_ptr, NULL, output_mlu_ptr, queue, NULL);
    }
    else {
      cnmlComputeConvOpForward_V4(conv_op, NULL, input_mlu_ptr, NULL, output_mlu_ptr, queue, NULL);
    }

    // wait for computing task over
    cnrtSyncQueue(queue);
    // end of queue life cycle
    // cnrtDestroyQueue(queue);

    // copy output to cpu
    cnrtMemcpy(output_cpu_ptr, output_mlu_ptr, output_count * sizeof(int16_t),
              CNRT_MEM_TRANS_DIR_DEV2HOST);
    
    // cast datatype to float
    cnrtCastDataType(output_cpu_ptr, CNRT_FLOAT16, output_cpu_data, CNRT_FLOAT32, output_count, NULL);

    // dump mlu result to file mlu_output
    // printf("dumping mlu result to file mlu_output...\n");
    // cnmlDumpTensor2File_V2("mlu_output", output_tensor, output_cpu_data, false);
    // printf("%f\n", output_cpu_data[0]);

    // delete conv param, op
    cnmlDestroyConvOpParam(&conv_param);
    cnmlDestroyBaseOp(&conv_op);

    // delete cnml buffer
    cnrtFree(input_mlu_ptr);
    cnrtFree(output_mlu_ptr);

  #ifdef FILTER_DATA_FP32
    // destory filter compute quant-param
    cnmlDestroyQuantizedParam(&filter_compute_quant);
  #else
    // destroy filter cast quant-param
    cnrtDestroyQuantizedParam(filter_quant_param);
  #endif

    // destory computing quant-param
    cnmlDestroyQuantizedParam(&input_quant_param);

    // delete cnml tensors
    cnmlDestroyTensor(&input_tensor);
    cnmlDestroyTensor(&filter_tensor);
    cnmlDestroyTensor(&output_tensor);
    return;
}
#endif
#endif // JIT

} // jittor
