// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guowei Yang <471184555@qq.com>
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "var.h"
#include "mkl_conv_backward_w_op.h"

#include <mkldnn.hpp>

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

MklConvBackwardWOp::MklConvBackwardWOp(Var* x, Var* dy, int kh, int kw, int stride, int padding, int dilation, int groups, string xformat, string wformat, string yformat)
        : x(x), dy(dy), kh(kh), kw(kw), stride(stride), padding(padding), dilation(dilation), groups(groups), 
      xformat(move(xformat)), wformat(move(wformat)), yformat(move(yformat)) {
    dw = create_output(nullptr, dtype_infer(dy->ns, x->ns));
}

void MklConvBackwardWOp::infer_shape() {
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

void MklConvBackwardWOp::jit_prepare() {
    add_jit_define("Txd", x->dtype());
    add_jit_define("Tyd", dy->dtype());
    add_jit_define("Twd", dw->dtype());
    add_jit_define("Tx", short_type(x));
    add_jit_define("Tw", short_type(dw));
    add_jit_define("Ty", short_type(dy));
    add_jit_define("XFORMAT", xformat);
    add_jit_define("WFORMAT", wformat);
    add_jit_define("YFORMAT", yformat);
}

#else // JIT
#ifdef JIT_cpu
void MklConvBackwardWOp::jit_run() {
    int batch = x->shape[findc("@XFORMAT",'a')];
    int ch_in = x->shape[findc("@XFORMAT",'b')];
    int height = x->shape[findc("@XFORMAT",'c')];
    int width = x->shape[findc("@XFORMAT",'d')];
    int ch_out = dw->shape[findc("@WFORMAT",'o')];
    int kh = dw->shape[findc("@WFORMAT",'h')];
    int kw = dw->shape[findc("@WFORMAT",'w')];
    
    auto* __restrict__ net_src = x->ptr<Txd>();
    auto* __restrict__ net_diff_dst = dy->ptr<Tyd>();
    auto* __restrict__ conv_user_diff_weights_buffer = dw->ptr<Twd>();
    
    using tag = memory::format_tag;
    using dt = memory::data_type;

    auto eng = engine(engine::kind::cpu, 0);
    stream s(eng);

    std::vector<primitive> net_bwd;
    std::vector<std::unordered_map<int, memory>> net_bwd_args;

    memory::dims conv_src_tz = {batch, ch_in, height, width};
    memory::dims conv_weights_tz = groups>1
        ? memory::dims{groups, ch_out/groups, ch_in/groups, kh, kw} 
        : memory::dims{ch_out, ch_in, kh, kw};
    memory::dims conv_dst_tz = {batch, ch_out, (height+padding*2-kh*dilation+dilation-1)/stride+1, (width+padding*2-kw*dilation+dilation-1)/stride+1};
    memory::dims conv_strides = {stride, stride};
    memory::dims conv_padding = {padding, padding};
    memory::dims conv_dilation = {dilation-1, dilation-1};

    if (groups>1) ASSERT(tag::@WFORMAT == tag::oihw);

    auto conv_user_src_memory
            = memory({{conv_src_tz}, dt::@Tx, tag::@XFORMAT}, eng, net_src);
            
    auto conv_src_md = memory::desc({conv_src_tz}, dt::@Tx, tag::any);
    auto conv_weights_md = memory::desc({conv_weights_tz}, dt::@Tw, tag::any);
    auto conv_dst_md = memory::desc({conv_dst_tz}, dt::@Ty, tag::any);

    auto conv_desc = convolution_forward::desc(prop_kind::forward,
            algorithm::convolution_direct, conv_src_md, conv_weights_md,
            conv_dst_md, conv_strides, conv_dilation, conv_padding,
            conv_padding);
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, eng);

    auto conv_src_memory = conv_user_src_memory;
    if (conv_pd.src_desc() != conv_user_src_memory.get_desc()) {
        conv_src_memory = memory(conv_pd.src_desc(), eng);
        net_bwd.push_back(reorder(conv_user_src_memory, conv_src_memory));
        net_bwd_args.push_back({{MKLDNN_ARG_FROM, conv_user_src_memory},
                {MKLDNN_ARG_TO, conv_src_memory}});
    }
    
    auto conv_user_diff_dst_memory
            = memory({{conv_dst_tz}, dt::@Ty, tag::YFORMAT}, eng, net_diff_dst);

    auto conv_user_diff_weights_memory
            = memory({{conv_weights_tz}, dt::@Tw, groups>1 ? tag::goihw : tag::@WFORMAT}, eng, conv_user_diff_weights_buffer);

    auto conv_bwd_src_md = memory::desc({conv_src_tz}, dt::@Tx, tag::any);
    auto conv_diff_weights_md
            = memory::desc({conv_weights_tz}, dt::@Tw, tag::any);
    auto conv_diff_dst_md = memory::desc({conv_dst_tz}, dt::@Ty, tag::any); 

    auto conv_bwd_weights_desc
            = convolution_backward_weights::desc(algorithm::convolution_direct,
                    conv_bwd_src_md, conv_diff_weights_md,
                    conv_diff_dst_md, conv_strides, conv_dilation, conv_padding, conv_padding);
    auto conv_bwd_weights_pd = convolution_backward_weights::primitive_desc(
            conv_bwd_weights_desc, eng, conv_pd);

    auto conv_bwd_src_memory = conv_src_memory;
    if (conv_bwd_weights_pd.src_desc() != conv_src_memory.get_desc()) {
        conv_bwd_src_memory = memory(conv_bwd_weights_pd.src_desc(), eng);
        net_bwd.push_back(reorder(conv_src_memory, conv_bwd_src_memory));
        net_bwd_args.push_back({{MKLDNN_ARG_FROM, conv_src_memory},
                {MKLDNN_ARG_TO, conv_bwd_src_memory}});
    }

    auto conv_diff_dst_memory = conv_user_diff_dst_memory;
    if (conv_bwd_weights_pd.diff_dst_desc()
            != conv_user_diff_dst_memory.get_desc()) {
        conv_diff_dst_memory = memory(conv_bwd_weights_pd.diff_dst_desc(), eng);
        net_bwd.push_back(reorder(conv_user_diff_dst_memory, conv_diff_dst_memory));
        net_bwd_args.push_back({{MKLDNN_ARG_FROM, conv_user_diff_dst_memory},
                {MKLDNN_ARG_TO, conv_diff_dst_memory}});
    }

    net_bwd.push_back(convolution_backward_weights(conv_bwd_weights_pd));
    net_bwd_args.push_back({{MKLDNN_ARG_SRC, conv_bwd_src_memory},
            {MKLDNN_ARG_DIFF_DST, conv_diff_dst_memory}});

    auto conv_diff_weights_memory = conv_user_diff_weights_memory;
    if (conv_bwd_weights_pd.diff_weights_desc()
            != conv_user_diff_weights_memory.get_desc()) {
        conv_diff_weights_memory
                = memory(conv_bwd_weights_pd.diff_weights_desc(), eng);
        net_bwd_args.back().insert(
                {MKLDNN_ARG_DIFF_WEIGHTS, conv_diff_weights_memory});

        net_bwd.push_back(reorder(
                conv_diff_weights_memory, conv_user_diff_weights_memory));
        net_bwd_args.push_back({{MKLDNN_ARG_FROM, conv_diff_weights_memory},
                {MKLDNN_ARG_TO, conv_user_diff_weights_memory}});
    } else {
        net_bwd_args.back().insert(
                {MKLDNN_ARG_DIFF_WEIGHTS, conv_diff_weights_memory});
    }

    ASSERTop(net_bwd.size(),==,net_bwd_args.size());

    for (size_t i = 0; i < net_bwd.size(); ++i)
        net_bwd.at(i).execute(s, net_bwd_args.at(i));

    s.wait();
}
#endif
#endif // JIT

} // jittor
