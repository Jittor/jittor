#include "runtime/float32_precision.h"

namespace jittor {
namespace {
thread_local Float32PrecisionPolicy active_precision;
}

Float32PrecisionPolicy current_float32_precision_policy() { return active_precision; }
void set_float32_precision_policy(Float32PrecisionPolicy policy) { active_precision = policy; }

vector<string> float32_precision_state() {
    return {float32_precision_tier_name(float32_matmul_tier()),
            float32_precision_tier_name(float32_cudnn_tier())};
}
} // namespace jittor
