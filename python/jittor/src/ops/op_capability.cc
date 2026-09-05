#include "ops/op_capability.h"
#include <map>
#include <mutex>

namespace jittor {
namespace {
struct CapabilityRegistry {
    std::mutex mutex;
    std::map<pair<BackendId, OpCapability>, OpCapabilityRegistration> entries;
};

CapabilityRegistry& capabilities() {
    static auto* registry = new CapabilityRegistry;
    return *registry;
}
}

const char* op_capability_name(OpCapability capability) {
    switch (capability) {
        case OpCapability::SegmentedArgReduce: return "segmented_arg_reduce";
        case OpCapability::SegmentedArgsort: return "segmented_argsort";
        case OpCapability::Where: return "where";
        case OpCapability::Random: return "random";
        case OpCapability::Transpose: return "transpose";
        case OpCapability::Matmul: return "matmul";
        case OpCapability::Conv2d: return "conv2d";
        case OpCapability::Conv2dBackwardInput: return "conv2d_backward_input";
        case OpCapability::Conv2dBackwardWeight: return "conv2d_backward_weight";
    }
    USER_CHECK(false) << "Unknown operator capability" << uint32(capability);
    return "unknown";
}

void register_op_capability(const OpCapabilityRegistration& registration) {
    USER_CHECK(registration.entry && !registration.implementation.empty())
        << "Operator capability requires a typed implementation";
    op_capability_name(registration.capability);
    auto& registry = capabilities();
    std::lock_guard<std::mutex> guard(registry.mutex);
    auto key = std::make_pair(registration.backend, registration.capability);
    USER_CHECK(!registry.entries.count(key))
        << "Duplicate operator capability:" << op_capability_name(registration.capability)
        << "backend" << uint32(registration.backend);
    registry.entries.emplace(key, registration);
}

OpCapabilityRegistration find_op_capability_registration(BackendId backend, OpCapability capability) {
    auto& registry = capabilities();
    std::lock_guard<std::mutex> guard(registry.mutex);
    auto it = registry.entries.find({backend, capability});
    return it == registry.entries.end() ? OpCapabilityRegistration{} : it->second;
}

vector<OpCapabilityRegistration> op_capability_registrations(BackendId backend) {
    auto& registry = capabilities();
    std::lock_guard<std::mutex> guard(registry.mutex);
    vector<OpCapabilityRegistration> result;
    for (const auto& entry : registry.entries)
        if (entry.first.first == backend) result.push_back(entry.second);
    return result;
}

bool has_op_capability(BackendId backend, OpCapability capability) {
    auto registration = find_op_capability_registration(backend, capability);
    return registration.entry && has_op(registration.implementation);
}

vector<string> backend_supported_capabilities(const string& backend) {
    vector<string> result;
    for (const auto& registration : op_capability_registrations(backend_registry().get(backend).id))
        if (has_op(registration.implementation))
            result.emplace_back(op_capability_name(registration.capability));
    return result;
}
} // namespace jittor
