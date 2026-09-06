#include "runtime/backend.h"
#include <stdexcept>

namespace jittor {

void BackendRegistry::register_backend(const BackendOps& backend) {
    if (backend.abi_version != 2 || backend.struct_size != sizeof(BackendOps))
        throw std::invalid_argument("Backend callback table ABI or size mismatch");
    if (!backend.name || !backend.name[0])
        throw std::invalid_argument("Backend name must not be empty");
    if (!backend.device_count || !backend.current_device || !backend.set_device
            || !backend.allocator || !backend.copy || !backend.copy_async
            || !backend.synchronize || !backend.stream || !backend.enable_peer)
        throw std::invalid_argument("Backend callback table is incomplete");
    if (backends_.count(backend.id))
        throw std::invalid_argument("Backend id is already registered");
    for (const auto& entry : names_)
        if (entry.second == backend.name)
            throw std::invalid_argument("Backend name is already registered");
    names_.emplace(backend.id, backend.name);
    try {
        auto entry = backends_.emplace(backend.id, backend).first;
        entry->second.name = names_.at(backend.id).c_str();
    } catch (...) {
        names_.erase(backend.id);
        throw;
    }
}

const BackendOps& BackendRegistry::get(BackendId id) const {
    auto found = backends_.find(id);
    if (found == backends_.end())
        throw std::out_of_range("Backend is not registered");
    return found->second;
}

vector<string> BackendRegistry::names() const {
    vector<string> result;
    for (const auto& entry : names_) result.push_back(entry.second);
    return result;
}

const BackendOps& BackendRegistry::get(const string& name) const {
    for (const auto& entry : names_)
        if (entry.second == name) return get(entry.first);
    throw std::out_of_range("Unknown backend: " + name);
}

} // namespace jittor
