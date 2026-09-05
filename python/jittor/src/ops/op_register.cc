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
    entries[op_file_name] = move(registered);
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
    std::lock_guard<std::recursive_mutex> guard(mutex);
    string op_file_name = key(name);
    bool removed = entries.erase(op_file_name);
    if (removed) {
        for (auto& item : provider_bindings)
            item.second.erase(op_file_name);
    }
    return removed;
}

void NativeOpRegistry::register_provider(const string& provider, bool replace) {
    ASSERT(!provider.empty()) << "provider name must not be empty";
    std::lock_guard<std::recursive_mutex> guard(mutex);
    auto iter = provider_bindings.find(provider);
    if (iter != provider_bindings.end()) {
        if (!replace)
            ASSERT(false) << "provider" << provider << "is already registered";
        iter->second.clear();
        return;
    }
    provider_bindings.emplace(provider, unordered_set<string>());
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

void NativeOpRegistry::bind_provider(const string& name, const string& provider) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    string op_file_name = key(name);
    ASSERT(entries.count(op_file_name)) << "Op" << name << "not found.";
    auto iter = provider_bindings.find(provider);
    ASSERT(iter != provider_bindings.end())
        << "provider" << provider << "is not registered";
    iter->second.insert(op_file_name);
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
    return {op_iter->second.id, provider};
}

bool NativeOpRegistry::unregister_provider(const string& provider) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    return provider_bindings.erase(provider) != 0;
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
