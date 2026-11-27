#include "miopen_conv2d_op.h"
#include "miopen_conv.h"
#include "core/var.h"
#include "ops/op_register.h"
#include "runtime/backend.h"
#include <climits>

namespace jittor {

MiopenConv2dOp::MiopenConv2dOp(Var* a, Var* b, NanoVector output_shape,
        NanoVector stride, NanoVector padding, NanoVector dilation, int direction)
    : a(a), b(b), output_shape(output_shape), stride(stride), padding(padding),
      dilation(dilation), direction(direction) {
    USER_CHECK(direction >= 0 && direction <= 2) << "invalid MIOpen convolution direction";
    USER_CHECK(a->dtype() == b->dtype()
        && (a->dtype() == ns_float32 || a->dtype() == ns_float16 || a->dtype() == ns_bfloat16))
        << "MIOpen convolution requires matching floating point operands";
    USER_CHECK(a->shape.size() == 4 && b->shape.size() == 4 && output_shape.size() == 4
        && stride.size() == 2 && padding.size() == 2 && dilation.size() == 2)
        << "MIOpen convolution requires rank-4 tensors and paired geometry";

    const auto& xs = direction == 0 ? a->shape : (direction == 1 ? output_shape : b->shape);
    const auto& ws = direction == 2 ? output_shape : b->shape;
    const auto& ys = direction == 0 ? output_shape : a->shape;
    for (int i=0; i<4; ++i)
        USER_CHECK(xs[i] > 0 && ws[i] > 0 && ys[i] > 0
            && xs[i] <= INT_MAX && ws[i] <= INT_MAX && ys[i] <= INT_MAX)
            << "MIOpen convolution dimensions must fit positive int32";
    USER_CHECK(xs[0] == ys[0] && xs[1] == ws[1] && ys[1] == ws[0])
        << "MIOpen convolution channel or batch mismatch";
    for (int i=0; i<2; ++i) {
        USER_CHECK(stride[i] > 0 && padding[i] >= 0 && dilation[i] > 0
            && stride[i] <= INT_MAX && padding[i] <= INT_MAX && dilation[i] <= INT_MAX)
            << "invalid MIOpen convolution spatial parameters";
        int64 extent = xs[i+2]+2*padding[i]-dilation[i]*(ws[i+2]-1)-1;
        USER_CHECK(extent >= 0 && extent/stride[i]+1 == ys[i+2])
            << "MIOpen convolution output geometry mismatch";
    }
    int device = a->device_id;
    if (device < 0) device = backend_ops(BackendId::Rocm).current_device();
    USER_CHECK(miopen_training_available(xs,ws,ys,a->dtype(),device,stride,padding,dilation))
        << "MIOpen training convolution has no bounded solution";

    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_manual_set_vnbb);
    a->set_flag(VarFlags::_needed_by_backward);
    b->set_flag(VarFlags::_needed_by_backward);
    result = create_output(nullptr, a->dtype());
}

void MiopenConv2dOp::infer_shape() {
    result->set_shape(output_shape);
}

void MiopenConv2dOp::run() {
    Var* x = direction == 0 ? a : (direction == 1 ? result : b);
    Var* w = direction == 2 ? result : b;
    Var* y = direction == 0 ? result : a;
    CHECK(try_miopen_conv(x,w,y,MiopenDirection(direction),stride,padding,dilation))
        << "MIOpen training convolution plan unavailable during execution";
}

VarPtr MiopenConv2dOp::grad(Var*, Var* dout, Var*, int v_index) {
    auto make_conv = op_constructor<VarPtr, Var*, Var*, NanoVector,
        NanoVector, NanoVector, NanoVector, int>("miopen_conv2d");
    VarPtr narrow;
    if (dout->dtype() != a->dtype()) {
        auto cast = op_constructor<VarPtr, Var*, NanoString>("unary");
        narrow = cast(dout, a->dtype());
        dout = narrow.ptr;
    }
    // The adjoints of the three bilinear convolution directions are the
    // other two directions. They remain graph operators for higher gradients.
    if (direction == 0) {
        if (v_index == 0)
            return make_conv(dout,b,a->shape,stride,padding,dilation,1);
        return make_conv(dout,a,b->shape,stride,padding,dilation,2);
    }
    if (direction == 1) {
        if (v_index == 0)
            return make_conv(dout,b,a->shape,stride,padding,dilation,0);
        return make_conv(a,dout,b->shape,stride,padding,dilation,2);
    }
    if (v_index == 0)
        return make_conv(b,dout,a->shape,stride,padding,dilation,0);
    return make_conv(a,dout,b->shape,stride,padding,dilation,1);
}

} // namespace jittor
