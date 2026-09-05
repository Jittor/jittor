#pragma once
#include "ops/op_register.h"
#include "runtime/backend.h"

namespace jittor {

enum class OpCapability : uint32 {
    SegmentedArgReduce,
    SegmentedArgsort,
    Where,
    Random,
    Transpose,
    Matmul,
    Conv2d,
    Conv2dBackwardInput,
    Conv2dBackwardWeight,
};

struct OpCapabilityEntry {
    virtual ~OpCapabilityEntry() = default;
};

template<class To, class... Ts>
struct TypedOpCapabilityEntry final : OpCapabilityEntry {
    using Predicate = bool (*)(Ts...);
    Predicate supports;
    explicit TypedOpCapabilityEntry(Predicate supports) : supports(supports) {}
};

struct OpCapabilityRegistration {
    BackendId backend = BackendId::Cpu;
    OpCapability capability = OpCapability::Matmul;
    string implementation;
    shared_ptr<OpCapabilityEntry> entry;
};

EXTERN_LIB void register_op_capability(const OpCapabilityRegistration& registration);
EXTERN_LIB OpCapabilityRegistration find_op_capability_registration(
    BackendId backend, OpCapability capability);
EXTERN_LIB vector<OpCapabilityRegistration> op_capability_registrations(BackendId backend);
EXTERN_LIB const char* op_capability_name(OpCapability capability);
EXTERN_LIB bool has_op_capability(BackendId backend, OpCapability capability);

// Registrations are published by their implementing libraries, independently
// of generated OpInfo initializers. Resolve only at the operation boundary.
template<class To, class... Ts>
class RegisterOpCapability {
public:
    RegisterOpCapability(BackendId backend, OpCapability capability,
                         const char* implementation,
                         bool (*supports)(Ts...) = nullptr) {
        register_op_capability({backend, capability, implementation,
            std::make_shared<TypedOpCapabilityEntry<To, Ts...>>(supports)});
    }
};

template<class To, class... Ts>
To (*find_op_capability(BackendId backend, OpCapability capability, Ts... args))(Ts...) {
    auto registration = find_op_capability_registration(backend, capability);
    if (!registration.entry || !has_op(registration.implementation)) return nullptr;
    auto typed = std::dynamic_pointer_cast<TypedOpCapabilityEntry<To, Ts...>>(registration.entry);
    USER_CHECK(typed) << "Capability constructor signature mismatch:"
        << op_capability_name(capability) << registration.implementation;
    if (typed->supports && !typed->supports(args...)) return nullptr;
    return get_op_info(registration.implementation).template get_constructor<To, Ts...>();
}

// @pyjt(backend_supported_capabilities)
vector<string> backend_supported_capabilities(const string& backend);

} // namespace jittor
