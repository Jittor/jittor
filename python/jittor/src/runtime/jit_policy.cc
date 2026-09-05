#include <algorithm>
#include "runtime/jit_policy.h"
#include "runtime/runtime.h"
#include "utils/flags.h"

namespace jittor {
vector<string> shsplit(const string& command);

DEFINE_RUNTIME_FLAG_WITH_SETTER(string, cuda_kernel_math, "default",
    "CUDA kernel math policy: default preserves startup flags, strict disables fast math, backend removes CUDA-specific strict flags.");

EXTERN_LIB void sync_all(bool device_sync);

void setter_cuda_kernel_math(const string& old_value, const string& requested) {
    USER_CHECK(requested == "default" || requested == "strict" || requested == "backend")
        << "cuda_kernel_math must be default, strict, or backend";
    if (old_value == requested) return;
    // Pending graphs belong to the old policy; publish the new policy only
    // after their submission. The flag setter rolls back if submission fails.
    runtime_jit_policy().cuda_kernel_math = old_value;
    // Environment initialization can precede construction of other core
    // globals. With no roots and no initialized executor there is no graph.
    if (!runtime_holder_state().holders().empty() || runtime_executor().allocator)
        sync_all(false);
    runtime_jit_policy().cuda_kernel_math = requested;
}

void add_cuda_math_jit_define(JK& jk) {
    add_jit_define(jk, "JIT_cuda_math", runtime_flag_cuda_kernel_math());
}

string cuda_math_flags_for_key(const string& flags, const string& jit_key) {
    string policy = "default";
    for (const auto& entry : parse_jit_keys(jit_key)) {
        if (entry.first == "JIT_cuda_math") policy = entry.second;
    }
    USER_CHECK(policy == "default" || policy == "strict" || policy == "backend")
        << "invalid CUDA kernel math policy in JIT key: " << policy;
    if (policy == "default") return flags;

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
