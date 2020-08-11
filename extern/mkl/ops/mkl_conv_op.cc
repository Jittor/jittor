// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <mkldnn.hpp>

#include "var.h"
#include "mkl_conv_op.h"

using namespace mkldnn;
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

MklConvOp::MklConvOp(Var* x, Var* w, int stride, int padding, int dilation, int groups, string xformat, string wformat, string yformat)
    : x(x), w(w), stride(stride), padding(padding), dilation(dilation), groups(groups),
      xformat(move(xformat)), wformat(move(wformat)), yformat(move(yformat)) {
    y = create_output(nullptr, dtype_infer(x->ns, w->ns));
    if (!this->yformat.size())
        this->yformat = this->xformat;
}

void MklConvOp::infer_shape() {
    ASSERTop(x->shape.size(),==,4);
    ASSERTop(w->shape.size(),==,4);
    int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
    get_shape(x, "abcd", xformat, xn, xc, xh, xw);
    get_shape(w, "oihw", wformat, wco, wci, wh, ww);
    ASSERTop(wci * groups,==,xc);
    yn = xn, yc = wco;
    yh = (xh+padding*2-wh*dilation+dilation-1)/stride+1;
    yw = (xw+padding*2-ww*dilation+dilation-1)/stride+1;
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

void MklConvOp::jit_prepare() {
    add_jit_define("Tx", short_type(x));
    add_jit_define("Tw", short_type(w));
    add_jit_define("Ty", short_type(y));
    add_jit_define("XFORMAT", xformat);
    add_jit_define("WFORMAT", wformat);
    add_jit_define("YFORMAT", yformat);
}

#else // JIT
#ifdef JIT_cpu
#pragma clang diagnostic ignored "-Wtautological-compare"
void MklConvOp::jit_run() {
    const auto& xs = x->shape;
    const auto& ws = w->shape;

    using tag = memory::format_tag;
    using dt = memory::data_type;

    if (tag::@XFORMAT==tag::nhwc && tag::@YFORMAT==tag::nhwc && tag::@WFORMAT==tag::hwio
        && stride==1 && padding==0  && dilation==1 && ws[0]==1 && ws[1]==1
        && dt::@Tx==dt::f32 && dt::@Ty==dt::f32 && dt::@Tw==dt::f32) {
        auto m = xs[0]*xs[1]*xs[2];
        auto n = ws[3];
        auto k = xs[3];
        // x: [m,k], w: [k,n], y: [m,n]
        ASSERTop(0,==,mkldnn_sgemm('N', 'N', m, n, k,
            1.f, x->ptr<float32>(), k,
            w->ptr<float32>(), n,
            0.f, y->ptr<float32>(), n));
        return;
    }

    engine eng(engine::kind::cpu, 0);
    stream s(eng);
    
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
    get_shape(x, "abcd", xformat, xn, xc, xh, xw);
    get_shape(w, "oihw", wformat, wco, wci, wh, ww);
    get_shape(y, "abcd", yformat, yn, yc, yh, yw);

    memory::dims conv1_src_tz = {xn, xc, xh, xw};
    memory::dims conv1_weights_tz = groups>1
        ? memory::dims{groups, wco/groups, wci, wh, ww} 
        : memory::dims{wco, wci, wh, ww};
    memory::dims conv1_dst_tz = {yn, yc, yh, yw};
    memory::dims conv1_strides = { stride, stride };
    memory::dims conv1_padding = { padding, padding };
    memory::dims conv1_dilation = { dilation-1, dilation-1 };

    if (groups>1) ASSERT(tag::@WFORMAT == tag::oihw);

    auto user_src_memory = memory(
            { { conv1_src_tz }, dt::@Tx, tag::@XFORMAT }, eng, x->mem_ptr);
    auto user_dst_memory = memory(
            { { conv1_dst_tz }, dt::@Ty, tag::@YFORMAT }, eng, y->mem_ptr);
    auto user_weights_memory = memory(
        { { conv1_weights_tz }, dt::@Tw, groups>1 ? tag::goihw : tag::@WFORMAT }, eng, w->mem_ptr);
            
    auto conv1_src_md = memory::desc({ conv1_src_tz }, dt::@Tx, tag::any);
    auto conv1_weights_md
            = memory::desc({ conv1_weights_tz }, dt::@Tw, tag::any);
    auto conv1_dst_md = memory::desc({ conv1_dst_tz }, dt::@Ty, tag::any);
    
    auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_auto, conv1_src_md, conv1_weights_md, conv1_dst_md, conv1_strides, conv1_dilation, conv1_padding, conv1_padding);
            
    auto conv1_prim_desc = convolution_forward::primitive_desc(conv1_desc, eng);

        net.clear();
        net_args.clear();
    auto conv1_src_memory = user_src_memory;
    if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
        conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
        net.push_back(reorder(user_src_memory, conv1_src_memory));
        net_args.push_back({ { MKLDNN_ARG_FROM, user_src_memory },
                { MKLDNN_ARG_TO, conv1_src_memory } });
    }

    auto conv1_weights_memory = user_weights_memory;
    if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
        net.push_back(reorder(user_weights_memory, conv1_weights_memory));
        net_args.push_back({ { MKLDNN_ARG_FROM, user_weights_memory }, { MKLDNN_ARG_TO, conv1_weights_memory } });
    }
    
    auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);
    
    net.push_back(convolution_forward(conv1_prim_desc));
    net_args.push_back({ { MKLDNN_ARG_SRC, conv1_src_memory },
            { MKLDNN_ARG_WEIGHTS, conv1_weights_memory },
            { MKLDNN_ARG_DST, conv1_dst_memory } });

    if (conv1_dst_memory != user_dst_memory) {
        net.push_back(reorder(conv1_dst_memory, user_dst_memory));
        net_args.push_back({ { MKLDNN_ARG_FROM, conv1_dst_memory },{ MKLDNN_ARG_TO, user_dst_memory } });
    }

    ASSERTop(net.size(),==,net_args.size());
    for (size_t i = 0; i < net.size(); ++i)
        net.at(i).execute(s, net_args.at(i));
}
#endif
#endif // JIT

} // jittor
