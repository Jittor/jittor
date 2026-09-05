// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <mutex>
#include <type_traits>
#include <utility>
#include "common.h"

namespace jittor {

struct OpConstructorEntry {
    virtual ~OpConstructorEntry() = default;
};

template<class Func>
struct TypedOpConstructorEntry : OpConstructorEntry {
    Func function;

    explicit TypedOpConstructorEntry(Func function) : function(function) {}
};

template<class Func>
shared_ptr<OpConstructorEntry> op_constructor_entry(Func function) {
    static_assert(std::is_pointer<Func>::value &&
        std::is_function<typename std::remove_pointer<Func>::type>::value,
        "an op constructor entry must be a function pointer");
    return std::make_shared<TypedOpConstructorEntry<Func>>(function);
}

struct OpInfo {
    string name, source_path, extra_flags;
    vector<shared_ptr<OpConstructorEntry>> constructors;
    // string: var member name, uint64: var member offset
    vector<pair<string, uint64>> var_members;
    // Zero is reserved for an Op instance that has not resolved its
    // registration yet. Every registered base op receives one process-local id.
    OpId id = 0;

    template<class To, class ...Ts> auto get_constructor() {
        typedef To (*func_t)(Ts...);
        for (const auto& constructor : constructors) {
            auto typed = std::dynamic_pointer_cast<TypedOpConstructorEntry<func_t>>(constructor);
            if (typed) return typed->function;
        }
        LOGf << "constructor" << name << "with requested signature not found.";
        return func_t(nullptr);
    }
};

/**
 * ABI-neutral identity of one provider-backed operator implementation.
 *
 * The native registry owns operator metadata, while a backend owns the
 * callable kernel and any device-specific handle.  Keeping this boundary to
 * an OpId and provider name lets CUDA/ACL providers evolve independently of
 * the core ABI; no backend library type crosses this header.
 */
struct NativeOpDispatchKey {
    OpId op_id = 0;
    string provider;
};

/**
 * Native owner for operator registration state.
 *
 * The public free functions below are kept as a compatibility boundary for
 * generated operators and old callers.  Storage and lifecycle now have one
 * owner, however: a lazily-created registry object owns the name map, id
 * allocator, and lock together.  This is the native counterpart of the
 * Python Backend/OpRegistry seam and gives future backend providers one
 * explicit place to attach registration teardown.
 */
class NativeOpRegistry {
public:
    void register_op(const OpInfo& op_info);
    bool has(const string& name) const;
    OpInfo get(const string& name) const;
    OpId id(const string& name) const;
    vector<string> names() const;
    bool unregister(const string& name);

    // Provider lifecycle and dispatch ownership are intentionally separate
    // from OpInfo registration.  A provider may bind several operators and
    // teardown removes all of its bindings atomically with its identity.
    void register_provider(const string& provider, bool replace = false);
    bool has_provider(const string& provider) const;
    vector<string> providers() const;
    void bind_provider(const string& name, const string& provider);
    NativeOpDispatchKey resolve_provider(const string& name,
                                         const string& provider) const;
    bool unregister_provider(const string& provider);

private:
    static string key(const string& name);

    mutable std::recursive_mutex mutex;
    unordered_map<string, OpInfo> entries;
    unordered_map<string, unordered_set<string>> provider_bindings;
    OpId next_id = 1;
};

// Intentionally process-lived: registration may happen from static
// initializers in many translation units, while destruction order is not
// specified across those units.
NativeOpRegistry& op_registry();

void op_registe(const OpInfo& op_info);
bool has_op(const string& name);
OpInfo get_op_info(const string& name);
OpId get_op_id(const string& name);
vector<string> registered_op_names();
bool unregister_op(const string& name);

// Canonical ids used by core correctness and optimization decisions. Each
// function resolves the registry once on first use, after static registration.
namespace op_ids {
OpId array();
OpId binary();
OpId broadcast_to();
OpId empty();
OpId fused();
OpId getitem();
OpId index();
OpId reduce();
OpId reindex();
OpId reindex_reduce();
OpId safe_clip();
OpId setitem();
}

/** An op constructor resolved on first call instead of at load time.
 *
 * The spelling this replaces was, at *namespace* scope, in 113 places:
 *
 *     static auto make_binary = get_op_info("binary")
 *         .get_constructor<VarPtr, Var*, Var*, NanoString>();
 *
 * `get_op_info` asserts the op is registered. Registration is itself a static
 * initialiser -- `gen_ops.cc`'s `int caller = (initer(), 0)` and
 * `op_utils.cc`'s `init()` -- in a *different* translation unit, and the
 * relative order of static initialisers across translation units is
 * unspecified by the standard. So "registered by the time this line runs" was
 * never a guarantee; it was a link order that happened to hold.
 *
 * When it does not hold, the failure is the worst available shape: the ASSERT
 * throws out of a static initialiser, before main, where no catch exists, so
 * the process calls std::terminate with no message naming the op, the file, or
 * the ordering. Deferring the lookup to the first *call* removes the question:
 * by then main is running and every registration has happened.
 *
 * (Function-local `static auto x = get_op_info(...)` inside an op constructor
 * was always fine -- it is already lazy -- and is left alone.)
 */
template<class To, class ...Ts>
struct OpConstructor {
    typedef To (*func_t)(Ts...);
    const char* name;
    //: Resolved once, then reused. Two threads racing here compute the same
    //: pointer from a map that is never written after static initialisation,
    //: so the duplicated work is the whole cost of the race.
    mutable func_t cached = nullptr;

    explicit OpConstructor(const char* name) : name(name) {}

    inline To operator()(Ts... args) const {
        if (!cached)
            cached = get_op_info(name).template get_constructor<To, Ts...>();
        return cached(std::forward<Ts>(args)...);
    }

    //: "is this op available at all" -- for the optional backend ops.
    inline explicit operator bool() const { return has_op(name); }
};

template<class To, class ...Ts>
inline OpConstructor<To, Ts...> op_constructor(const char* name) {
    return OpConstructor<To, Ts...>(name);
}

struct OpCompiler;
struct OpByType {
    unordered_set<string> types;
    virtual string expand_op(const vector<string>& args) = 0;
    virtual void post_pass(OpCompiler*) = 0;
};

vector<OpByType*>& get_op_types();
int registe_op_type(OpByType*);

} // jittor
