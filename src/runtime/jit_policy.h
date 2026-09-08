#pragma once
#include "core/common.h"
#include "codegen/jit_key.h"

namespace jittor {

struct RuntimeJitPolicy {
    string cuda_kernel_math = "default";
    int float32_matmul_precision_tier = 0;
    int float32_cudnn_precision_tier = 0;
};

EXTERN_LIB RuntimeJitPolicy& runtime_jit_policy();
DECLARE_RUNTIME_FLAG(string, cuda_kernel_math);

EXTERN_LIB void add_cuda_math_jit_define(JK& jk);
EXTERN_LIB string cuda_math_flags_for_key(const string& flags, const string& jit_key);

} // namespace jittor
