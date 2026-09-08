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
#include "mkl_conv_backward_x_op.h"

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

MklConvBackwardXOp::MklConvBackwardXOp(Var* w, Var* dy, int height, int width, int strideh, int stridew, int paddingh, int paddingw, int dilationh, int dilationw, int groups, string xformat, string wformat, string yformat) 
        : w(w), dy(dy), xh(height), xw(width), strideh(strideh), stridew(stridew), paddingh(paddingh), paddingw(paddingw), dilationh(dilationh), dilationw(dilationw), groups(groups),
      xformat(move(xformat)), wformat(move(wformat)), yformat(move(yformat)) {
    check_onednn_conv_args(w, dy, strideh, stridew, paddingh, paddingw,
                          dilationh, dilationw, groups, this->xformat, this->wformat, this->yformat);
    USER_CHECK(height > 0 && width > 0) << "oneDNN backward input requires positive height/width";
    dx = create_output(nullptr, dtype_infer(dy->ns, w->ns));
}

void MklConvBackwardXOp::infer_shape() {
    USER_CHECKop(w->shape.size(),==,4);
    USER_CHECKop(dy->shape.size(),==,4);
    int xn, xc, wh, ww, wci, wco, yn, yc, yh, yw;
    get_shape(w, "oihw", wformat, wco, wci, wh, ww);
    get_shape(dy, "abcd", yformat, yn, yc, yh, yw);
    USER_CHECK(wco == yc && wco % groups == 0 && wh > 0 && ww > 0)
        << "oneDNN backward input invalid channels or kernel shape";
    USER_CHECK(xh+paddingh*2 >= (wh-1)*dilationh+1 && xw+paddingw*2 >= (ww-1)*dilationw+1
        && yh == (xh+paddingh*2-(wh-1)*dilationh-1)/strideh+1
        && yw == (xw+paddingw*2-(ww-1)*dilationw-1)/stridew+1)
        << "oneDNN backward input gradient shape does not match convolution";
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
    jk << "«Tyd:" << dy->dtype();
    jk << "«Twd:" << w->dtype();
    jk << "«Txd:" << dx->dtype();
    jk << "«Tx:" << short_type(dx);
    jk << "«Tw:" << short_type(w);
    jk << "«Ty:" << short_type(dy);
    jk << "«XFORMAT:" << xformat;
    jk << "«WFORMAT:" << wformat;
    jk << "«YFORMAT:" << yformat;
}

#else // JIT
#ifdef JIT_cpu
void MklConvBackwardXOp::jit_run() {
    auto spec = onednn_conv_spec(1, dx, w, dy, strideh, stridew,
        paddingh, paddingw, dilationh, dilationw, groups, xformat, wformat, yformat);
    onednn_conv_execute(spec, dx->mem_ptr, w->mem_ptr, dy->mem_ptr);
}
#endif
#endif // JIT

} // jittor
