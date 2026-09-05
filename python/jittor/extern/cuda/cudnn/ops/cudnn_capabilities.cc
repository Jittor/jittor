#include "var.h"
#include "ops/op_capability.h"

#ifndef JIT
namespace jittor {
namespace {
bool supports_conv_layout(Var* a, Var* b, int groups, const string& wformat) {
    // cuDNN filter descriptors accept NCHW/NHWC, while the grouped path
    // currently encodes the group dimension in OIHW order only.
    return a->dtype().is_float() && b->dtype().is_float()
        && (wformat == "oihw" || wformat == "ohwi")
        && (groups <= 1 || wformat == "oihw");
}

bool supports_conv(Var* x, Var* w, int, int, int, int, int, int, int groups,
                   string, string wformat, string) {
    return supports_conv_layout(x, w, groups, wformat);
}

bool supports_conv_backward(Var* a, Var* b, int, int, int, int, int, int, int, int,
                            int groups, string, string wformat, string) {
    return supports_conv_layout(a, b, groups, wformat);
}

RegisterOpCapability<VarPtr, Var*, Var*, int, int, int, int, int, int, int, string, string, string> conv(
    accelerator_backend_id(), OpCapability::Conv2d, "cudnn_conv", supports_conv);
RegisterOpCapability<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, string, string, string> conv_x(
    accelerator_backend_id(), OpCapability::Conv2dBackwardInput, "cudnn_conv_backward_x", supports_conv_backward);
RegisterOpCapability<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, string, string, string> conv_w(
    accelerator_backend_id(), OpCapability::Conv2dBackwardWeight, "cudnn_conv_backward_w", supports_conv_backward);
}
} // namespace jittor
#endif
