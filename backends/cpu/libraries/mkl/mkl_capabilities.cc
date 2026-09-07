#include "core/var.h"
#include "ops/composite/op_capability.h"

#ifndef JIT
namespace jittor {
namespace {
bool supports_matmul(Var* a, Var* b, bool, bool) {
    return a->dtype() == ns_float32 && b->dtype() == ns_float32;
}

bool supports_conv_layout(Var* a, Var* b, int groups, const string& xformat,
                          const string& wformat, const string& yformat) {
    return a->dtype() == ns_float32 && b->dtype() == ns_float32
        && yformat != "cdab" && xformat != "bacd"
        && (groups <= 1 || wformat == "oihw");
}

bool supports_conv(Var* x, Var* w, int, int, int, int, int, int, int groups,
                   string xformat, string wformat, string yformat) {
    return supports_conv_layout(x, w, groups, xformat, wformat, yformat);
}

bool supports_conv_backward(Var* a, Var* b, int, int, int, int, int, int, int, int,
                            int groups, string xformat, string wformat, string yformat) {
    return supports_conv_layout(a, b, groups, xformat, wformat, yformat);
}

// oneDNN itself covers f64/f16/bf16 for both matmul and convolution, but
// Jittor's operators here do not: `mkl_matmul_op.cc` calls `dnnl_sgemm`, whose
// signature is float-only, and the convolution operators are only exercised
// and only asserted for f32. So the declaration says f32 -- what this code
// does -- rather than what the library could do. Widening it is a code change
// (a `dnnl::matmul` primitive in place of `dnnl_sgemm`), not a declaration
// change; until then a CPU f64/f16/bf16 matmul runs as the generic reindex
// kernel, and now says so when asked.
const vector<string> f32_only = {"float32"};

RegisterOpCapability<VarPtr, Var*, Var*, bool, bool> matmul(
    BackendId::Cpu, OpCapability::Matmul, "mkl_matmul", supports_matmul, f32_only);
RegisterOpCapability<VarPtr, Var*, Var*, int, int, int, int, int, int, int, string, string, string> conv(
    BackendId::Cpu, OpCapability::Conv2d, "mkl_conv", supports_conv, f32_only);
RegisterOpCapability<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, string, string, string> conv_x(
    BackendId::Cpu, OpCapability::Conv2dBackwardInput, "mkl_conv_backward_x", supports_conv_backward, f32_only);
RegisterOpCapability<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, string, string, string> conv_w(
    BackendId::Cpu, OpCapability::Conv2dBackwardWeight, "mkl_conv_backward_w", supports_conv_backward, f32_only);
}
} // namespace jittor
#endif
