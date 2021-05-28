// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "var.h"
#include "mkl_conv_backward_x_op.h"

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

MklConvBackwardXOp::MklConvBackwardXOp(Var* w, Var* dy, int height, int width, int strideh, int stridew, int paddingh, int paddingw, int dilationh, int dilationw, int groups, string xformat, string wformat, string yformat) 
        : w(w), dy(dy), xh(height), xw(width), strideh(strideh), stridew(stridew), paddingh(paddingh), paddingw(paddingw), dilationh(dilationh), dilationw(dilationw), groups(groups),
      xformat(move(xformat)), wformat(move(wformat)), yformat(move(yformat)) {
    dx = create_output(nullptr, dtype_infer(dy->ns, w->ns));
}

void MklConvBackwardXOp::infer_shape() {
    ASSERTop(w->shape.size(),==,4);
    ASSERTop(dy->shape.size(),==,4);
    int xn, xc, wh, ww, wci, wco, yn, yc, yh, yw;
    get_shape(w, "oihw", wformat, wco, wci, wh, ww);
    get_shape(dy, "abcd", yformat, yn, yc, yh, yw);
    xn = yn, xc = wci * groups;
    set_shape(dx, "abcd", xformat, xn, xc, xh, xw);
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

void MklConvBackwardXOp::jit_prepare(JK& jk) {
    jk << _CS("[Tyd:") << dy->dtype();
    jk << _CS("][Twd:") << w->dtype();
    jk << _CS("][Txd:") << dx->dtype();
    jk << _CS("][Tx:") << short_type(dx);
    jk << _CS("][Tw:") << short_type(w);
    jk << _CS("][Ty:") << short_type(dy);
    jk << _CS("][XFORMAT:") << xformat;
    jk << _CS("][WFORMAT:") << wformat;
    jk << _CS("][YFORMAT:") << yformat;
    jk << ']';
}

#else // JIT
#ifdef JIT_cpu
void MklConvBackwardXOp::jit_run() {
    int batch = dx->shape[findc("@XFORMAT",'a')];
    int ch_in = dx->shape[findc("@XFORMAT",'b')];
    int height = dx->shape[findc("@XFORMAT",'c')];
    int width = dx->shape[findc("@XFORMAT",'d')];
    int ch_out = w->shape[findc("@WFORMAT",'o')];
    int kernel_sizeh = w->shape[findc("@WFORMAT",'h')];
    int kernel_sizew = w->shape[findc("@WFORMAT",'w')];
    
    auto* __restrict__ conv_weights = w->ptr<Twd>();
    auto* __restrict__ net_diff_dst = dy->ptr<Tyd>();
    auto* __restrict__ conv_user_diff_src_buffer = dx->ptr<Txd>();
    
    using tag = memory::format_tag;
    using dt = memory::data_type;

    auto eng = engine(engine::kind::cpu, 0);
    stream s(eng);

    std::vector<primitive> net_bwd;
    std::vector<std::unordered_map<int, memory>> net_bwd_args;
    
    memory::dims conv_src_tz = {batch, ch_in, height, width};
    memory::dims conv_weights_tz = groups>1
        ? memory::dims{groups, ch_out/groups, ch_in/groups, kernel_sizeh, kernel_sizew} 
        : memory::dims{ch_out, ch_in, kernel_sizeh, kernel_sizew};
    memory::dims conv_dst_tz = {batch, ch_out, (height+paddingh*2-kernel_sizeh*dilationh+dilationh-1)/strideh+1, (width+paddingw*2-kernel_sizew*dilationw+dilationw-1)/stridew+1};
    memory::dims conv_strides = {strideh, stridew};
    memory::dims conv_padding = {paddingh, paddingw};
    memory::dims conv_dilation = {dilationh-1, dilationw-1};

    if (groups>1) ASSERT(tag::@WFORMAT == tag::oihw);

    auto conv_user_weights_memory
            = memory({{conv_weights_tz}, dt::@Tw, groups>1 ? tag::goihw : tag::@WFORMAT}, eng, conv_weights);

    auto conv_src_md = memory::desc({conv_src_tz}, dt::@Tx, tag::any);
    auto conv_weights_md = memory::desc({conv_weights_tz}, dt::@Tw, tag::any);
    auto conv_dst_md = memory::desc({conv_dst_tz}, dt::@Ty, tag::any);

    auto conv_desc = convolution_forward::desc(prop_kind::forward,
            algorithm::convolution_direct, conv_src_md, conv_weights_md,
            conv_dst_md, conv_strides, conv_dilation, conv_padding,
            conv_padding);
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, eng);

    auto conv_weights_memory = conv_user_weights_memory;
    if (conv_pd.weights_desc() != conv_user_weights_memory.get_desc()) {
        conv_weights_memory = memory(conv_pd.weights_desc(), eng);
        net_bwd.push_back(
                reorder(conv_user_weights_memory, conv_weights_memory));
        net_bwd_args.push_back({{MKLDNN_ARG_FROM, conv_user_weights_memory},
                {MKLDNN_ARG_TO, conv_weights_memory}});
    }
    
    auto conv_user_diff_dst_memory
            = memory({{conv_dst_tz}, dt::@Ty, tag::@YFORMAT}, eng, net_diff_dst);

    auto conv_user_diff_src_memory
            = memory({{conv_src_tz}, dt::@Tx, tag::@XFORMAT}, eng, conv_user_diff_src_buffer);
            
    auto conv_bwd_weights_md
            = memory::desc({conv_weights_tz}, dt::@Tw, tag::any);
    auto conv_diff_src_md = memory::desc({conv_src_tz}, dt::@Tx, tag::any);
    auto conv_diff_dst_md = memory::desc({conv_dst_tz}, dt::@Ty, tag::any); 

    auto conv_bwd_data_desc
            = convolution_backward_data::desc(algorithm::convolution_direct,
                    conv_diff_src_md, conv_bwd_weights_md, conv_diff_dst_md, conv_strides, conv_dilation, conv_padding, conv_padding);
    auto conv_bwd_data_pd = convolution_backward_data::primitive_desc(
            conv_bwd_data_desc, eng, conv_pd);

    auto conv_diff_dst_memory = conv_user_diff_dst_memory;
    if (conv_bwd_data_pd.diff_dst_desc()
            != conv_user_diff_dst_memory.get_desc()) {
        conv_diff_dst_memory = memory(conv_bwd_data_pd.diff_dst_desc(), eng);
        net_bwd.push_back(reorder(conv_user_diff_dst_memory, conv_diff_dst_memory));
        net_bwd_args.push_back({{MKLDNN_ARG_FROM, conv_user_diff_dst_memory},
                {MKLDNN_ARG_TO, conv_diff_dst_memory}});
    }

    auto conv_bwd_weights_memory = conv_weights_memory;
    if (conv_bwd_data_pd.weights_desc() != conv_weights_memory.get_desc()) {
        conv_bwd_weights_memory = memory(conv_bwd_data_pd.weights_desc(), eng);
        net_bwd.push_back(reorder(conv_weights_memory, conv_bwd_weights_memory));
        net_bwd_args.push_back({{MKLDNN_ARG_FROM, conv_weights_memory},
                {MKLDNN_ARG_TO, conv_bwd_weights_memory}});
    }

    net_bwd.push_back(convolution_backward_data(conv_bwd_data_pd));
    net_bwd_args.push_back({{MKLDNN_ARG_WEIGHTS, conv_bwd_weights_memory},
        {MKLDNN_ARG_DIFF_DST, conv_diff_dst_memory}});
            
    auto conv_diff_src_memory = conv_user_diff_src_memory;
    if (conv_bwd_data_pd.diff_src_desc()
            != conv_user_diff_src_memory.get_desc()) {
        conv_diff_src_memory
                = memory(conv_bwd_data_pd.diff_src_desc(), eng);
        net_bwd_args.back().insert(
                {MKLDNN_ARG_DIFF_SRC, conv_diff_src_memory});
                
        net_bwd.push_back(reorder(
                conv_diff_src_memory, conv_user_diff_src_memory));
        net_bwd_args.push_back({{MKLDNN_ARG_FROM, conv_diff_src_memory},
                {MKLDNN_ARG_TO, conv_user_diff_src_memory}});
    } else {
        net_bwd_args.back().insert(
                {MKLDNN_ARG_DIFF_SRC, conv_diff_src_memory});
    }

    ASSERTop(net_bwd.size(),==,net_bwd_args.size());

    for (size_t i = 0; i < net_bwd.size(); ++i)
        net_bwd.at(i).execute(s, net_bwd_args.at(i));
}
#endif
#endif // JIT

} // jittor
