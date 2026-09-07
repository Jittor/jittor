#pragma once
#include "common.h"
#include "jit_key.h"

namespace jittor {

struct RuntimeJitPolicy {
    string cuda_kernel_math = "default";
};

EXTERN_LIB RuntimeJitPolicy& runtime_jit_policy();
DECLARE_RUNTIME_FLAG(string, cuda_kernel_math);

EXTERN_LIB void add_cuda_math_jit_define(JK& jk);
EXTERN_LIB string cuda_math_flags_for_key(const string& flags, const string& jit_key);

} // namespace jittor
