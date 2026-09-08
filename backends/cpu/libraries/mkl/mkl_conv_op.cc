// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "onednn_runtime.h"

#include "core/var.h"
#include "mkl_conv_op.h"

using namespace std;

namespace jittor {

static inline int findc(const string& format, const char& c) {
    auto position = format.find(c);
    USER_CHECK(format.size() == 4 && position != string::npos) << "Not a valid format" << format;
    return int(position);
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

MklConvOp::MklConvOp(Var* x, Var* w, int strideh, int stridew, int paddingh, int paddingw, int dilationh, int dilationw, int groups, string xformat, string wformat, string yformat)
    : x(x), w(w), strideh(strideh), stridew(stridew), paddingh(paddingh), paddingw(paddingw), dilationh(dilationh), dilationw(dilationw), groups(groups),
      xformat(move(xformat)), wformat(move(wformat)), yformat(move(yformat)) {
    y = create_output(nullptr, dtype_infer(x->ns, w->ns));
    if (!this->yformat.size())
        this->yformat = this->xformat;
    check_onednn_conv_args(x, w, strideh, stridew, paddingh, paddingw,
                          dilationh, dilationw, groups, this->xformat, this->wformat, this->yformat);
}

void MklConvOp::infer_shape() {
    USER_CHECKop(x->shape.size(),==,4);
    USER_CHECKop(w->shape.size(),==,4);
    int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
    get_shape(x, "abcd", xformat, xn, xc, xh, xw);
    get_shape(w, "oihw", wformat, wco, wci, wh, ww);
    USER_CHECKop(wci * groups,==,xc);
    USER_CHECK(wco % groups == 0 && wh > 0 && ww > 0) << "oneDNN invalid grouped channels or kernel shape";
    yn = xn, yc = wco;
    yh = (xh+paddingh*2-wh*dilationh+dilationh-1)/strideh+1;
    yw = (xw+paddingw*2-ww*dilationw+dilationw-1)/stridew+1;
    USER_CHECK(xh+paddingh*2 >= (wh-1)*dilationh+1 && xw+paddingw*2 >= (ww-1)*dilationw+1)
        << "oneDNN convolution kernel exceeds padded input";
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

void MklConvOp::jit_prepare(JK& jk) {
    jk << "«Txd:" << x->dtype();
    jk << "«Tyd:" << y->dtype();
    jk << "«Twd:" << w->dtype();
    jk << "«Tx:" << short_type(x);
    jk << "«Tw:" << short_type(w);
    jk << "«Ty:" << short_type(y);
    jk << "«XFORMAT:" << xformat;
    jk << "«WFORMAT:" << wformat;
    jk << "«YFORMAT:" << yformat;
}

#else // JIT
#ifdef JIT_cpu
void MklConvOp::jit_run() {
    auto spec = onednn_conv_spec(0, x, w, y, strideh, stridew,
        paddingh, paddingw, dilationh, dilationw, groups, xformat, wformat, yformat);
    onednn_conv_execute(spec, x->mem_ptr, w->mem_ptr, y->mem_ptr);
}
#endif
#endif // JIT

} // jittor
