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
    y = create_output(nullptr, ns_float32);
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

unordered_map<string, MluConv_t> mlu_conv_cache;

#else // JIT
#ifdef JIT_cpu
#pragma clang diagnostic ignored "-Wtautological-compare"

extern unordered_map<string, MluConv_t> mlu_conv_cache;

void MluConvOp::jit_run() {
    // auto start = std::chrono::high_resolution_clock::now();
    if (mlu_conv_cache.size() == 0) {
      cnrtInit(0);
      cnmlInit(0);
    }

    int ni, ci, hi, wi, no, co, cw, ho, wo, kh, kw;
    get_shape(x, "abcd", xformat, ni, ci, hi, wi);
    get_shape(w, "oihw", wformat, co, cw, kh, kw);
    get_shape(y, "abcd", yformat, no, co, ho, wo);

    int input_count = ni * hi * wi * ci;
    int filter_count = co * cw * kh * kw;
    int output_count = no * ho * wo * co;

    int8_t* input_cpu_ptr = (int8_t*)x->mem_ptr;
    int8_t* filter_cpu_ptr = (int8_t*)w->mem_ptr;
    float *output_cpu_data = (float*)y->mem_ptr;
    int16_t* output_cpu_ptr = (int16_t *)malloc(output_count * sizeof(int16_t));

    void *input_mlu_ptr_ygw = NULL;
    void *filter_mlu_ptr_ygw = NULL;
    cnrtMalloc(&input_mlu_ptr_ygw, input_count * sizeof(int8_t));
    cnrtMalloc(&filter_mlu_ptr_ygw, filter_count * sizeof(int8_t));
    cnrtMemcpy(input_mlu_ptr_ygw, input_cpu_ptr, input_count * sizeof(int8_t),
              CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(filter_mlu_ptr_ygw, filter_cpu_ptr, filter_count * sizeof(int8_t),
              CNRT_MEM_TRANS_DIR_HOST2DEV);

    const int coreNum = 16;
    const int dimNum = 4;
    MluConv_t mlu_conv;

    jk.clear();
    jk << ni << "," << ci << "," << hi << "," << wi << ",";
    jk << no << "," << co << "," << ho << "," << wo << ",";
    jk << paddingh << paddingw << "," <<strideh <<stridew << "," << dilationh << dilationw << "," << groups << ".";
    auto iter = mlu_conv_cache.find(jk.to_string());

    cnmlBaseOp_t conv_op = NULL;
    cnmlConvOpParam_t conv_param;
    cnmlTensor_t input_tensor = NULL;
    cnmlTensor_t filter_tensor = NULL;
    cnmlTensor_t output_tensor = NULL;
    void *input_mlu_ptr = NULL;
    void *output_mlu_ptr = NULL;
    if (iter != mlu_conv_cache.end()) {
      // LOGw << "find key";
      mlu_conv = iter->second;
      conv_op = mlu_conv.conv_op_cache;
      conv_param = mlu_conv.conv_param_cache;
      input_tensor = mlu_conv.input_tensor_cache;
      filter_tensor = mlu_conv.filter_tensor_cache;
      output_tensor = mlu_conv.output_tensor_cache;
      input_mlu_ptr = mlu_conv.input_mlu_ptr_cache;
      output_mlu_ptr = mlu_conv.output_mlu_ptr_cache;
    }
    else {
      // LOGw << "not find key";
      int input_shape[] = {ni, ci, hi, wi};
      int filter_shape[] = {co, cw, kh, kw};
      int output_shape[] = {no, co, ho, wo};

      cnmlCreateTensor_V2(&input_tensor, CNML_TENSOR);
      cnmlSetTensorShape_V2(input_tensor, dimNum, input_shape, NULL);
      cnmlCreateTensor_V2(&filter_tensor, CNML_FILTER);
      cnmlSetTensorShape_V2(filter_tensor, dimNum, filter_shape, NULL);
      cnmlCreateTensor_V2(&output_tensor, CNML_TENSOR);
      cnmlSetTensorShape_V2(output_tensor, dimNum, output_shape, NULL);

      cnmlSetTensorDataType(input_tensor, CNML_DATA_INT8);
      cnmlSetTensorDataType(filter_tensor, CNML_DATA_INT8);
      cnmlSetTensorDataType(output_tensor, CNML_DATA_FLOAT16);

      cnmlCreateConvOpParam(&conv_param, strideh, stridew, dilationh, dilationw, paddingh * 2, paddingw * 2);
      if (groups > 1) {
        cnmlCreateConvGroupOp(&conv_op, conv_param, input_tensor, output_tensor, filter_tensor, NULL, groups);
      }
      else {
        cnmlCreateConvOp(&conv_op, conv_param, input_tensor, output_tensor, filter_tensor, NULL);
      }
      cnmlSetOperationComputingLayout(conv_op, CNML_NCHW);
      cnrtMemcpy(filter_cpu_ptr, filter_mlu_ptr_ygw, filter_count * sizeof(int8_t),
              CNRT_MEM_TRANS_DIR_DEV2HOST);
      cnmlBindConstData_V2(filter_tensor, filter_cpu_ptr, false);
      
      const cnmlCoreVersion_t coreVersion = CNML_MLU270;
      cnmlSetBaseOpCoreVersion(conv_op, coreVersion);
      cnmlSetBaseOpCorenum(conv_op, coreNum);
      cnmlCompileBaseOp_V2(conv_op);

      // cnrtMalloc(&input_mlu_ptr, input_count * sizeof(int8_t));
      cnrtMalloc(&output_mlu_ptr, output_count * sizeof(int16_t));

      mlu_conv.conv_op_cache = conv_op;
      mlu_conv.conv_param_cache = conv_param;
      mlu_conv.input_tensor_cache = input_tensor;
      mlu_conv.filter_tensor_cache = filter_tensor;
      mlu_conv.output_tensor_cache = output_tensor;
      mlu_conv.input_mlu_ptr_cache = input_mlu_ptr;
      mlu_conv.output_mlu_ptr_cache = output_mlu_ptr;
      mlu_conv_cache[jk.to_string()] = mlu_conv;
    }

    // cnrtMemcpy(input_mlu_ptr, input_cpu_ptr, input_count * sizeof(int8_t),
    //           CNRT_MEM_TRANS_DIR_HOST2DEV);

    cnrtQueue_t queue;
    cnrtCreateQueue(&queue);

    if (groups > 1) {
      cnmlComputeConvGroupOpForward_V4(conv_op, NULL, input_mlu_ptr_ygw, NULL, output_mlu_ptr, queue, NULL);
    }
    else {
      cnmlComputeConvOpForward_V4(conv_op, NULL, input_mlu_ptr_ygw, NULL, output_mlu_ptr, queue, NULL) ;
      CNRT_CHECK(cnrtSyncQueue(queue));
    }

    cnrtSyncQueue(queue);
    cnrtDestroyQueue(queue);

    cnrtMemcpy(output_cpu_ptr, output_mlu_ptr, output_count * sizeof(int16_t),
              CNRT_MEM_TRANS_DIR_DEV2HOST);
    
    cnrtCastDataType(output_cpu_ptr, CNRT_FLOAT16, output_cpu_data, CNRT_FLOAT32, output_count, NULL);

    /*
    计算结果存放在output_mlu_ptr，然后copy到本地的output_cpu_ptr，output_cpu_ptr再转成float32拷贝到output_cpu_data，所以output_cpu_data应该再传回y，现在的问题可能是y和output的shape不匹配
    */

    // auto finish = std::chrono::high_resolution_clock::now();
    // auto total_ns =  (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() / 1000. / 1000.;
    // LOGw << total_ns;

    return;
}
#endif
#endif // JIT

} // jittor
