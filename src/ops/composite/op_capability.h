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
    // Element types this implementation accepts, as dtype spellings.
    //
    // A backend's dtype coverage was previously stated only inside its
    // `supports` predicate -- a function pointer -- so "which dtypes does CPU
    // matmul do?" had no answer short of reading C++ and no answer at all from
    // Python. oneDNN's matmul is fp32-only while cuBLAS takes every float
    // width, and that difference decides whether a model's fp64 or bf16
    // matmul silently runs as the generic reindex kernel instead. Declaring it
    // here makes the difference queryable (`backend_capability_dtypes`).
    //
    // The predicate stays authoritative for *selection*: this list does not
    // gate anything, because capability signatures are not uniform (Random
    // takes a shape and two dtype strings, Conv2d takes two Vars and eleven
    // scalars) and no generic dtype extraction covers them. What keeps the two
    // from drifting is a test that compares this declaration against which
    // implementation actually runs, per dtype.
    vector<string> dtypes;
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
                         bool (*supports)(Ts...) = nullptr,
                         vector<string> dtypes = {}) {
        register_op_capability({backend, capability, implementation,
            std::make_shared<TypedOpCapabilityEntry<To, Ts...>>(supports),
            move(dtypes)});
    }
};

// Whether a backend executes this op through a native callback of its own,
// as opposed to a generated-JIT entry or a placeholder that can only fall
// back. A backend may implement a core op directly instead of publishing a
// separate capability op for it; requiring the callback keeps an op whose
// accelerator JIT body is an unimplemented stub from claiming the device.
inline bool backend_runs_op_natively(BackendId backend, const string& name) {
    auto definition = get_op_definition(name, false);
    if (!definition) return false;
    auto implementation = definition->implementations.find(backend);
    return implementation != definition->implementations.end()
        && implementation->second.kernel.native != nullptr
        && !implementation->second.kernel.fallback_only;
}

inline shared_ptr<const OpDef> op_capability_definition(const OpCapabilityRegistration& registration) {
    if (!registration.entry) return nullptr;
    auto definition = get_op_definition(registration.implementation, false);
    if (!definition) return nullptr;
    auto implementation = definition->implementations.find(registration.backend);
    if (implementation == definition->implementations.end() || implementation->second.kernel.fallback_only)
        return nullptr;
    return definition;
}

template<class To, class... Ts>
To (*find_op_capability(BackendId backend, OpCapability capability, Ts... args))(Ts...) {
    auto registration = find_op_capability_registration(backend, capability);
    auto definition = op_capability_definition(registration);
    if (!definition) return nullptr;
    auto typed = std::dynamic_pointer_cast<TypedOpCapabilityEntry<To, Ts...>>(registration.entry);
    USER_CHECK(typed) << "Capability constructor signature mismatch:"
        << op_capability_name(capability) << registration.implementation;
    if (typed->supports && !typed->supports(args...)) return nullptr;
    return definition->template get_constructor<To, Ts...>();
}

// @pyjt(backend_supported_capabilities)
vector<string> backend_supported_capabilities(const string& backend);

// Declared element types for one (backend, capability), or an empty list when
// the implementation has not declared any. An empty list is "undeclared", not
// "supports nothing": the caller must not read it as a negative answer.
// @pyjt(backend_capability_dtypes)
vector<string> backend_capability_dtypes(const string& backend,
                                         const string& capability);

} // namespace jittor
