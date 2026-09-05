// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "op.h"
#include "ops/op_register.h"

namespace jittor {

//: The one key the map is read and written by.
//:
//: `op_registe` used to insert under `op_info.name` while `has_op` and
//: `get_op_info` looked up `op_name_to_file_name(name)`, which truncates at the
//: first '.'. For every name without a dot the two agree and nothing showed;
//: register a name *with* a dot and it went in under the full name and could
//: never be found again, by any spelling. One key, computed in one place.
static inline string op_key(const string& name) {
    return Op::op_name_to_file_name(name);
}

string NativeOpRegistry::key(const string& name) {
    return op_key(name);
}

NativeOpRegistry& op_registry() {
    static auto* registry = new NativeOpRegistry();
    return *registry;
}

void NativeOpRegistry::register_op(const OpInfo& op_info) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    string op_file_name = key(op_info.name);
    auto iter = entries.find(op_file_name);
    if (iter != entries.end()) {
        if (iter->second.source_path == op_info.source_path) {
            LOGvv << "replace duplicated op registration" << op_info.name
                << "\nsource_path:" << op_info.source_path
                << "\nextra_flags:" << op_info.extra_flags
                << "\nold_extra_flags:" << iter->second.extra_flags;
            OpInfo replacement = op_info;
            replacement.id = iter->second.id;
            iter->second = move(replacement);
            return;
        }
        ASSERT(false) << "Op" << op_info.name << "is already registed, "
            << "source_path:" << op_info.source_path << "extra_flags" << op_info.extra_flags;
    }
    LOGvv << "registe op" << op_info.name
        << "\nsource_path:" << op_info.source_path
        << "\nextra_flags:" << op_info.extra_flags
        << "\nconstructors:" << op_info.constructors.size()
        << "\nvar_members:" << op_info.var_members;
    OpInfo registered = op_info;
    registered.id = next_id++;
    const auto registered_id = registered.id;
    entries[op_file_name] = move(registered);
    op_keys_by_id.emplace(registered_id, op_file_name);
}

bool NativeOpRegistry::has(const string& name) const {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    return entries.count(key(name));
}

OpInfo NativeOpRegistry::get(const string& name) const {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    auto iter = entries.find(key(name));
    ASSERT(iter != entries.end()) << "Op" << name << "not found.";
    return iter->second;
}

OpId NativeOpRegistry::id(const string& name) const {
    return get(name).id;
}

vector<string> NativeOpRegistry::names() const {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    vector<string> names;
    names.reserve(entries.size());
    for (const auto& item : entries) names.push_back(item.first);
    return names;
}

bool NativeOpRegistry::unregister(const string& name) {
    vector<NativeOpDispatchKey> unbound;
    vector<NativeProviderLifecycleEvent> lifecycle_events;
    NativeProviderLifecycleObserver* observer = nullptr;
    bool removed = false;
    {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    string op_file_name = key(name);
    auto op_iter = entries.find(op_file_name);
    if (op_iter == entries.end())
        return false;
    for (auto& item : provider_bindings) {
        if (!item.second.erase(op_file_name))
            continue;
        auto id_iter = provider_ids.find(item.first);
        auto registration_iter = provider_registrations.find(item.first);
        ASSERT(id_iter != provider_ids.end());
        ASSERT(registration_iter != provider_registrations.end());
        NativeOpDispatchKey dispatch_key = {op_iter->second.id, item.first,
            id_iter->second, registration_iter->second.abi_version};
        unbound.push_back(dispatch_key);
        lifecycle_events.push_back(NativeProviderLifecycleEvent::op_unbound(
            NativeProviderMetadata(registration_iter->second, id_iter->second),
            dispatch_key));
    }
    op_keys_by_id.erase(op_iter->second.id);
    entries.erase(op_iter);
    observer = lifecycle_observer;
    removed = true;
    }
    if (observer)
        for (size_t i = 0; i < unbound.size(); ++i) {
            const auto& key = unbound[i];
            observer->on_provider_op_unbound(key);
            observer->on_provider_lifecycle_event(lifecycle_events[i]);
        }
    return removed;
}

void NativeOpRegistry::register_provider(
        const NativeProviderRegistration& registration, bool replace) {
    ASSERT(registration.valid())
        << "invalid native provider registration for" << registration.name
        << "abi_version:" << registration.abi_version
        << "struct_size:" << registration.struct_size;
    const string& provider = registration.name;
    NativeProviderLifecycleObserver* observer = nullptr;
    uint32 provider_instance = 0;
    uint32 old_provider_instance = 0;
    NativeProviderRegistration old_registration;
    vector<NativeOpDispatchKey> unbound;
    vector<NativeProviderLifecycleEvent> lifecycle_events;
    bool replaced = false;
    {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    auto iter = provider_bindings.find(provider);
    if (iter != provider_bindings.end()) {
        if (!replace)
            ASSERT(false) << "provider" << provider << "is already registered";
        auto old_id = provider_ids.at(provider);
        old_provider_instance = old_id;
        old_registration = provider_registrations.at(provider);
        for (const auto& op_name : iter->second) {
            auto op_iter = entries.find(op_name);
            if (op_iter == entries.end())
                continue;
            NativeOpDispatchKey dispatch_key = {op_iter->second.id, provider,
                old_id, old_registration.abi_version};
            unbound.push_back(dispatch_key);
            lifecycle_events.push_back(NativeProviderLifecycleEvent::op_unbound(
                NativeProviderMetadata(old_registration, old_id), dispatch_key));
        }
        iter->second.clear();
        // A replacement is a new provider instance.  Never let a cached
        // backend handle accidentally address the new instance.
        provider_ids[provider] = next_provider_id++;
        provider_registrations[provider] = registration;
        replaced = true;
    } else {
        provider_bindings.emplace(provider, unordered_set<string>());
        provider_ids.emplace(provider, next_provider_id++);
        provider_registrations.emplace(provider, registration);
    }
    provider_instance = provider_ids.at(provider);
    observer = lifecycle_observer;
    }
    if (observer) {
        for (size_t i = 0; i < unbound.size(); ++i) {
            const auto& key = unbound[i];
            observer->on_provider_op_unbound(key);
            observer->on_provider_lifecycle_event(lifecycle_events[i]);
        }
        if (replaced)
            observer->on_provider_unregistered(old_registration,
                                               old_provider_instance);
            observer->on_provider_lifecycle_event(
                NativeProviderLifecycleEvent::provider_unregistered(
                    old_registration, old_provider_instance));
        observer->on_provider_registered(registration, provider_instance);
        observer->on_provider_lifecycle_event(
            NativeProviderLifecycleEvent::provider_registered(
                registration, provider_instance));
    }
}

void NativeOpRegistry::register_provider(const string& provider, bool replace) {
    register_provider(NativeProviderRegistration(provider), replace);
}

bool NativeOpRegistry::has_provider(const string& provider) const {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    return provider_bindings.count(provider) != 0;
}

vector<string> NativeOpRegistry::providers() const {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    vector<string> result;
    result.reserve(provider_bindings.size());
    for (const auto& item : provider_bindings)
        result.push_back(item.first);
    return result;
}

uint32 NativeOpRegistry::provider_id(const string& provider) const {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    auto iter = provider_ids.find(provider);
    ASSERT(iter != provider_ids.end())
        << "provider" << provider << "is not registered";
    return iter->second;
}

NativeProviderRegistration NativeOpRegistry::provider_registration(
        const string& provider) const {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    auto iter = provider_registrations.find(provider);
    ASSERT(iter != provider_registrations.end())
        << "provider" << provider << "is not registered";
    return iter->second;
}

NativeProviderMetadata NativeOpRegistry::provider_metadata(
        const string& provider) const {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    auto registration_iter = provider_registrations.find(provider);
    auto id_iter = provider_ids.find(provider);
    ASSERT(registration_iter != provider_registrations.end())
        << "provider" << provider << "is not registered";
    ASSERT(id_iter != provider_ids.end())
        << "provider" << provider << "has no identity";
    // Construct a value while holding the registry lock.  The returned
    // object remains valid after replacement or teardown of this provider.
    return NativeProviderMetadata(registration_iter->second, id_iter->second);
}

NativeProviderConsumerDispatch NativeOpRegistry::provider_consumer_dispatch(
        const string& name, const string& provider) const {
    NativeProviderConsumerDispatch result;
    ASSERT(try_provider_consumer_dispatch(name, provider, result))
        << "Op" << name << "has no current provider dispatch for" << provider;
    return result;
}

NativeProviderConsumerDispatch NativeOpRegistry::provider_consumer_dispatch(
        OpId op_id, const string& provider) const {
    NativeProviderConsumerDispatch result;
    ASSERT(try_provider_consumer_dispatch(op_id, provider, result))
        << "Op id" << op_id << "has no current provider dispatch for"
        << provider;
    return result;
}

bool NativeOpRegistry::try_provider_consumer_dispatch(
        const string& name, const string& provider,
        NativeProviderConsumerDispatch& dispatch) const {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    string op_file_name = key(name);
    auto op_iter = entries.find(op_file_name);
    if (op_iter == entries.end())
        return false;
    return try_provider_consumer_dispatch_locked(
        op_iter->second.id, provider, dispatch);
}

bool NativeOpRegistry::try_provider_consumer_dispatch(
        OpId op_id, const string& provider,
        NativeProviderConsumerDispatch& dispatch) const {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    return try_provider_consumer_dispatch_locked(op_id, provider, dispatch);
}

bool NativeOpRegistry::try_provider_consumer_dispatch_locked(
        OpId op_id, const string& provider,
        NativeProviderConsumerDispatch& dispatch) const {
    if (!op_id)
        return false;
    auto provider_iter = provider_bindings.find(provider);
    if (provider_iter == provider_bindings.end())
        return false;
    auto op_key_iter = op_keys_by_id.find(op_id);
    if (op_key_iter == op_keys_by_id.end())
        return false;
    auto op_iter = entries.find(op_key_iter->second);
    if (op_iter == entries.end() ||
            !provider_iter->second.count(op_iter->first))
        return false;
    auto id_iter = provider_ids.find(provider);
    if (id_iter == provider_ids.end())
        return false;
    auto registration_iter = provider_registrations.find(provider);
    if (registration_iter == provider_registrations.end())
        return false;
    NativeProviderConsumerDispatch result;
    result.metadata = NativeProviderMetadata(registration_iter->second,
                                             id_iter->second);
    result.dispatch_key = {op_id, provider, id_iter->second,
                           registration_iter->second.abi_version};
    if (!result.valid())
        return false;
    dispatch = move(result);
    return true;
}

NativeProviderLifecycleObserver* NativeOpRegistry::set_lifecycle_observer(
        NativeProviderLifecycleObserver* observer) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    auto previous = lifecycle_observer;
    lifecycle_observer = observer;
    return previous;
}

bool NativeOpRegistry::clear_lifecycle_observer(
        NativeProviderLifecycleObserver* observer) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    if (lifecycle_observer != observer)
        return false;
    lifecycle_observer = nullptr;
    return true;
}

void NativeOpRegistry::bind_provider(const string& name, const string& provider) {
    NativeOpDispatchKey dispatch_key;
    NativeProviderMetadata metadata;
    NativeProviderLifecycleObserver* observer = nullptr;
    {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    string op_file_name = key(name);
    ASSERT(entries.count(op_file_name)) << "Op" << name << "not found.";
    auto iter = provider_bindings.find(provider);
    ASSERT(iter != provider_bindings.end())
        << "provider" << provider << "is not registered";
    auto inserted = iter->second.insert(op_file_name);
    if (!inserted.second)
        return;
    auto op_iter = entries.find(op_file_name);
    auto provider_iter = provider_ids.find(provider);
    auto registration_iter = provider_registrations.find(provider);
    ASSERT(provider_iter != provider_ids.end());
    ASSERT(registration_iter != provider_registrations.end());
    dispatch_key = {op_iter->second.id, provider, provider_iter->second,
                    registration_iter->second.abi_version};
    metadata = NativeProviderMetadata(registration_iter->second,
                                      provider_iter->second);
    observer = lifecycle_observer;
    }
    if (observer) {
        observer->on_provider_op_bound(dispatch_key);
        observer->on_provider_lifecycle_event(
            NativeProviderLifecycleEvent::op_bound(metadata, dispatch_key));
    }
}

bool NativeOpRegistry::unbind_provider_if_current(
        const NativeOpDispatchKey& dispatch_key) {
    NativeProviderLifecycleObserver* observer = nullptr;
    NativeProviderMetadata metadata;
    bool removed = false;
    {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    if (!dispatch_key.valid())
        return false;
    auto provider_iter = provider_bindings.find(dispatch_key.provider);
    auto id_iter = provider_ids.find(dispatch_key.provider);
    auto registration_iter = provider_registrations.find(dispatch_key.provider);
    if (provider_iter == provider_bindings.end() ||
            id_iter == provider_ids.end() ||
            registration_iter == provider_registrations.end() ||
            id_iter->second != dispatch_key.provider_id ||
            registration_iter->second.abi_version != dispatch_key.abi_version)
        return false;
    string op_file_name;
    for (const auto& item : entries) {
        if (item.second.id == dispatch_key.op_id) {
            op_file_name = item.first;
            break;
        }
    }
    if (op_file_name.empty())
        return false;
    auto binding = provider_iter->second.find(op_file_name);
    if (binding == provider_iter->second.end())
        return false;
    provider_iter->second.erase(binding);
    metadata = NativeProviderMetadata(registration_iter->second,
                                      id_iter->second);
    observer = lifecycle_observer;
    removed = true;
    }
    if (removed && observer) {
        observer->on_provider_op_unbound(dispatch_key);
        observer->on_provider_lifecycle_event(
            NativeProviderLifecycleEvent::op_unbound(metadata, dispatch_key));
    }
    return removed;
}

NativeOpDispatchKey NativeOpRegistry::resolve_provider(
        const string& name, const string& provider) const {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    string op_file_name = key(name);
    auto provider_iter = provider_bindings.find(provider);
    ASSERT(provider_iter != provider_bindings.end())
        << "provider" << provider << "is not registered";
    ASSERT(provider_iter->second.count(op_file_name))
        << "Op" << name << "has no provider binding for" << provider;
    auto op_iter = entries.find(op_file_name);
    ASSERT(op_iter != entries.end()) << "Op" << name << "not found.";
    auto id_iter = provider_ids.find(provider);
    ASSERT(id_iter != provider_ids.end())
        << "provider" << provider << "has no identity";
    auto registration_iter = provider_registrations.find(provider);
    ASSERT(registration_iter != provider_registrations.end())
        << "provider" << provider << "has no ABI registration";
    return {op_iter->second.id, provider, id_iter->second,
            registration_iter->second.abi_version};
}

bool NativeOpRegistry::is_current(
        const NativeOpDispatchKey& dispatch_key) const {
    if (!dispatch_key.valid())
        return false;
    std::lock_guard<std::recursive_mutex> guard(mutex);
    auto provider_iter = provider_bindings.find(dispatch_key.provider);
    if (provider_iter == provider_bindings.end() ||
            !provider_iter->second.size())
        return false;
    auto id_iter = provider_ids.find(dispatch_key.provider);
    auto registration_iter = provider_registrations.find(dispatch_key.provider);
    if (id_iter == provider_ids.end() || registration_iter == provider_registrations.end() ||
            id_iter->second != dispatch_key.provider_id ||
            registration_iter->second.abi_version != dispatch_key.abi_version)
        return false;
    for (const auto& op_name : provider_iter->second) {
        auto op_iter = entries.find(op_name);
        if (op_iter != entries.end() && op_iter->second.id == dispatch_key.op_id)
            return true;
    }
    return false;
}

bool NativeOpRegistry::unregister_provider(const string& provider) {
    return unregister_provider_if_current(provider, 0);
}

bool NativeOpRegistry::unregister_provider_if_current(
        const string& provider, uint32 expected_provider_id) {
    NativeProviderLifecycleObserver* observer = nullptr;
    NativeProviderRegistration registration;
    uint32 provider_instance = 0;
    vector<NativeOpDispatchKey> unbound;
    vector<NativeProviderLifecycleEvent> lifecycle_events;
    bool removed = false;
    {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    auto bindings = provider_bindings.find(provider);
    if (bindings == provider_bindings.end())
        return false;
    auto ids = provider_ids.find(provider);
    auto registrations = provider_registrations.find(provider);
    ASSERT(ids != provider_ids.end());
    ASSERT(registrations != provider_registrations.end());
    if (expected_provider_id && ids->second != expected_provider_id)
        return false;
    provider_instance = ids->second;
    registration = registrations->second;
    for (const auto& op_name : bindings->second) {
        auto op_iter = entries.find(op_name);
        if (op_iter != entries.end())
            {
            NativeOpDispatchKey dispatch_key = {
                op_iter->second.id, provider, provider_instance,
                registration.abi_version};
            unbound.push_back(dispatch_key);
            lifecycle_events.push_back(NativeProviderLifecycleEvent::op_unbound(
                NativeProviderMetadata(registration, provider_instance),
                dispatch_key));
            }
    }
    provider_bindings.erase(bindings);
    provider_ids.erase(ids);
    provider_registrations.erase(registrations);
    observer = lifecycle_observer;
    removed = true;
    }
    if (removed && observer) {
        for (size_t i = 0; i < unbound.size(); ++i) {
            const auto& key = unbound[i];
            observer->on_provider_op_unbound(key);
            observer->on_provider_lifecycle_event(lifecycle_events[i]);
        }
        observer->on_provider_unregistered(registration, provider_instance);
        observer->on_provider_lifecycle_event(
            NativeProviderLifecycleEvent::provider_unregistered(
                registration, provider_instance));
    }
    return removed;
}

void op_registe(const OpInfo& op_info) {
    op_registry().register_op(op_info);
}

bool has_op(const string& name) {
    return op_registry().has(name);
}

OpInfo get_op_info(const string& name) {
    return op_registry().get(name);
}

OpId get_op_id(const string& name) {
    return op_registry().id(name);
}

vector<string> registered_op_names() {
    return op_registry().names();
}

bool unregister_op(const string& name) {
    return op_registry().unregister(name);
}

#define DEFINE_BUILTIN_OP_ID(name) \
    OpId op_ids::name() { \
        static OpId id = get_op_id(#name); \
        return id; \
    }

DEFINE_BUILTIN_OP_ID(array)
DEFINE_BUILTIN_OP_ID(binary)
DEFINE_BUILTIN_OP_ID(broadcast_to)
DEFINE_BUILTIN_OP_ID(empty)
DEFINE_BUILTIN_OP_ID(fused)
DEFINE_BUILTIN_OP_ID(getitem)
DEFINE_BUILTIN_OP_ID(index)
DEFINE_BUILTIN_OP_ID(reduce)
DEFINE_BUILTIN_OP_ID(reindex)
DEFINE_BUILTIN_OP_ID(reindex_reduce)
DEFINE_BUILTIN_OP_ID(safe_clip)
DEFINE_BUILTIN_OP_ID(setitem)

#undef DEFINE_BUILTIN_OP_ID

vector<OpByType*>& get_op_types() {
    static auto* types = new vector<OpByType*>();
    return *types;
}

int registe_op_type(OpByType* op_type) {
    get_op_types().push_back(op_type);
    return 0;
}


} // jittor
