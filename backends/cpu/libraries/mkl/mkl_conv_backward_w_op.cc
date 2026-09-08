// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "core/var.h"
#include "mkl_conv_backward_w_op.h"

#include "onednn_runtime.h"

using namespace std;

namespace jittor {
static inline int findc(const string& format, const char& c) {
    auto position = format.find(c);
    USER_CHECK(format.size() == 4 && position != string::npos) << "Not a valid format" << format;
    return int(position);
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

MklConvBackwardWOp::MklConvBackwardWOp(Var* x, Var* dy, int kh, int kw, int strideh, int stridew, int paddingh, int paddingw, int dilationh, int dilationw, int groups, string xformat, string wformat, string yformat)
        : x(x), dy(dy), kh(kh), kw(kw), strideh(strideh), stridew(stridew), paddingh(paddingh), paddingw(paddingw), dilationh(dilationh), dilationw(dilationw), groups(groups), 
      xformat(move(xformat)), wformat(move(wformat)), yformat(move(yformat)) {
    check_onednn_conv_args(x, dy, strideh, stridew, paddingh, paddingw,
                          dilationh, dilationw, groups, this->xformat, this->wformat, this->yformat);
    USER_CHECK(kh > 0 && kw > 0) << "oneDNN backward weight requires positive kernel size";
    dw = create_output(nullptr, dtype_infer(dy->ns, x->ns));
}

void MklConvBackwardWOp::infer_shape() {
    USER_CHECKop(x->shape.size(),==,4);
    USER_CHECKop(dy->shape.size(),==,4);
    int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
    get_shape(x, "abcd", xformat, xn, xc, xh, xw);
    get_shape(dy, "abcd", yformat, yn, yc, yh, yw);
    USER_CHECK(xn == yn && xc % groups == 0 && yc % groups == 0)
        << "oneDNN backward weight invalid batch or grouped channels";
    USER_CHECK(xh+paddingh*2 >= (kh-1)*dilationh+1 && xw+paddingw*2 >= (kw-1)*dilationw+1
        && yh == (xh+paddingh*2-(kh-1)*dilationh-1)/strideh+1
        && yw == (xw+paddingw*2-(kw-1)*dilationw-1)/stridew+1)
        << "oneDNN backward weight gradient shape does not match convolution";
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

void MklConvBackwardWOp::jit_prepare(JK& jk) {
    jk << "«Txd:" << x->dtype();
    jk << "«Tyd:" << dy->dtype();
    jk << "«Twd:" << dw->dtype();
    jk << "«Tx:" << short_type(x);
    jk << "«Tw:" << short_type(dw);
    jk << "«Ty:" << short_type(dy);
    jk << "«XFORMAT:" << xformat;
    jk << "«WFORMAT:" << wformat;
    jk << "«YFORMAT:" << yformat;
}

#else // JIT
#ifdef JIT_cpu
void MklConvBackwardWOp::jit_run() {
    auto spec = onednn_conv_spec(2, x, dw, dy, strideh, stridew,
        paddingh, paddingw, dilationh, dilationw, groups, xformat, wformat, yformat);
    onednn_conv_execute(spec, x->mem_ptr, dw->mem_ptr, dy->mem_ptr);
}
#endif
#endif // JIT

} // jittor
