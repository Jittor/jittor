#include "runtime/backend_fallback.h"
#include "runtime/runtime.h"

namespace jittor {

JIT_TEST(backend_fallback_has_one_runtime_owner) {
    CHECK(&runtime_backend_fallback() == &native_runtime().fallbacks());
    CHECK(&runtime_flag_backend_fallback() == &native_runtime().fallbacks().policy);
    BackendFallbackState isolated;
    CHECK(isolated.policy == "warn");
    CHECK(isolated.count == 0);
}

JIT_TEST(backend_fallback_checks_current_policy) {
    const auto before = backend_fallback_count();
    check_backend_fallback("same_backend_probe", BackendId::Cpu, BackendId::Cpu, "none");
    CHECK(backend_fallback_count() == before);
    check_backend_fallback("fallback_policy_probe", accelerator_backend_id(),
                           BackendId::Cpu, "test kernel is unavailable");
    CHECK(backend_fallback_count() == before + 1);
}

} // namespace jittor
