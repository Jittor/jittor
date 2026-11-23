#include <algorithm>
#include "runtime/jit_policy.h"
#include "runtime/runtime.h"
#include "utils/flags.h"

namespace jittor {
vector<string> shsplit(const string& command);

// Kernels are compiled to IEEE behaviour by default: no --use_fast_math (so no
// flush of subnormals, approximate sinf/expf or reciprocal division), and no
// multiply-add contraction. A workload that has measured fast math and accepts
// its error opts in with "fast" -- a quantum-circuit simulation whose gates are
// built from __sinf/__cosf, for instance, drifts its state norm by 2e-5 over
// 400 gates (energy 6x further from the reference than PyTorch or JAX). See
// docs/notes/numerics/numerics-contract.md.
DEFINE_RUNTIME_FLAG_WITH_SETTER(string, cuda_kernel_math, "strict",
    "CUDA kernel math policy. strict (the default) compiles without fast math, "
    "with IEEE division and square root and no multiply-add contraction. fast "
    "keeps the startup NVCC flags (--use_fast_math). backend removes the "
    "CUDA-specific strict flags. The old name of fast, default, is accepted.");

EXTERN_LIB void sync_all(bool device_sync);

void setter_cuda_kernel_math(const string& old_value, const string& requested) {
    USER_CHECK(requested == "strict" || requested == "fast" || requested == "backend"
               || requested == "default")
        << "cuda_kernel_math must be strict, fast, or backend";
    // "default" named fast math while it was the default; it keeps meaning it.
    const string policy = requested == "default" ? "fast" : requested;
    if (old_value == policy) {
        runtime_jit_policy().cuda_kernel_math = policy;
        return;
    }
    // Pending graphs belong to the old policy; publish the new policy only
    // after their submission. The flag setter rolls back if submission fails.
    runtime_jit_policy().cuda_kernel_math = old_value;
    // Environment initialization can precede construction of other core
    // globals. With no roots and no initialized executor there is no graph.
    if (!runtime_holder_state().holders().empty() || runtime_executor().allocator)
        sync_all(false);
    runtime_jit_policy().cuda_kernel_math = policy;
}

void add_cuda_math_jit_define(JK& jk) {
    add_jit_define(jk, "JIT_cuda_math", runtime_flag_cuda_kernel_math());
}

string cuda_math_flags_for_key(const string& flags, const string& jit_key) {
    // A key without the define predates the policy and kept the startup flags.
    string policy = "fast";
    for (const auto& entry : parse_jit_keys(jit_key)) {
        if (entry.first == "JIT_cuda_math") policy = entry.second;
    }
    USER_CHECK(policy == "fast" || policy == "default" || policy == "strict"
               || policy == "backend")
        << "invalid CUDA kernel math policy in JIT key: " << policy;
    if (policy == "fast" || policy == "default") return flags;

    const vector<string> strict_flags = {
        "--fmad=false", "--prec-div=true", "--prec-sqrt=true"};
    auto tokens = shsplit(flags);
    string result = " ";
    for (const auto& token : tokens) {
        if (policy == "strict" && token == "--use_fast_math") continue;
        if (std::find(strict_flags.begin(), strict_flags.end(), token) != strict_flags.end())
            continue;
        result += token + " ";
    }
    if (policy == "strict") {
        for (const auto& token : strict_flags) result += token + " ";
    }
    return result;
}

} // namespace jittor
