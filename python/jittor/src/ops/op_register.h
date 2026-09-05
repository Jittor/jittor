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

// Versioned, backend-neutral registration record. Backend libraries may be
// built independently of the core, so registration carries an explicit ABI
// contract instead of relying on a backend-specific object layout.
static const uint32 NATIVE_PROVIDER_ABI_VERSION = 1;

struct NativeProviderRegistration {
    string name;
    uint32 abi_version;
    uint32 struct_size;

    NativeProviderRegistration()
        : abi_version(NATIVE_PROVIDER_ABI_VERSION),
          struct_size(sizeof(NativeProviderRegistration)) {}

    explicit NativeProviderRegistration(const string& name,
                                        uint32 abi_version = NATIVE_PROVIDER_ABI_VERSION,
                                        uint32 struct_size = 0)
        : name(name), abi_version(abi_version),
          struct_size(struct_size ? struct_size : sizeof(NativeProviderRegistration)) {}

    bool valid() const {
        return !name.empty() &&
            abi_version == NATIVE_PROVIDER_ABI_VERSION &&
            struct_size >= sizeof(NativeProviderRegistration);
    }
};

/**
 * Value-only metadata published to a native provider consumer.
 *
 * A consumer must not retain a reference into NativeOpRegistry: provider
 * replacement can invalidate the registry entry while a device backend is
 * still draining work.  This snapshot is therefore deliberately composed of
 * strings and scalar ABI fields.  Consumers may cache it for diagnostics and
 * compare ``provider_id`` before using a backend handle.
 */
struct NativeProviderMetadata {
    string name;
    uint32 provider_id;
    uint32 abi_version;
    uint32 struct_size;

    NativeProviderMetadata()
        : provider_id(0), abi_version(0), struct_size(0) {}

    NativeProviderMetadata(const NativeProviderRegistration& registration,
                           uint32 provider_id)
        : name(registration.name), provider_id(provider_id),
          abi_version(registration.abi_version),
          struct_size(registration.struct_size) {}

    bool valid() const {
        return !name.empty() && provider_id != 0 &&
            abi_version == NATIVE_PROVIDER_ABI_VERSION &&
            struct_size >= sizeof(NativeProviderRegistration);
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
    // Registry-owned identity for the provider instance.  The spelling is
    // retained for diagnostics; native backend handles should cache this
    // numeric field so a provider replacement cannot reuse an old key.
    uint32 provider_id = 0;
    // Contract version used to create this key. Callers can reject stale
    // keys before handing them to a backend ABI.
    uint32 abi_version = 0;

    bool valid() const {
        return op_id != 0 && provider_id != 0 && !provider.empty() &&
            abi_version == NATIVE_PROVIDER_ABI_VERSION;
    }
};

/**
 * Value-only checks a backend consumer performs before using a published
 * provider handle.  ``valid()`` on each value is necessary but insufficient:
 * a cached metadata snapshot and dispatch key can each be valid while
 * belonging to different provider instances after replacement.  Keeping this
 * check in the ABI header gives CUDA/ACL consumers one identical fail-closed
 * boundary without exposing registry storage or backend handle types.
 */
struct NativeProviderConsumerContract {
    static bool accepts(const NativeProviderMetadata& metadata) {
        return metadata.valid();
    }

    static bool accepts(const NativeProviderMetadata& metadata,
                        const NativeOpDispatchKey& dispatch_key) {
        return accepts(metadata) && dispatch_key.valid() &&
            metadata.name == dispatch_key.provider &&
            metadata.provider_id == dispatch_key.provider_id &&
            metadata.abi_version == dispatch_key.abi_version;
    }
};

/**
 * Coherent value-only input for a native provider consumer.
 *
 * A consumer that reads provider metadata and an operator dispatch key in
 * separate calls can observe two different provider generations.  The
 * registry publishes this pair while holding one lock, so a CUDA/ACL
 * consumer can validate the pair first and only then consult its own handle
 * table.  The values remain safe to retain after registry teardown.
 */
struct NativeProviderConsumerDispatch {
    NativeProviderMetadata metadata;
    NativeOpDispatchKey dispatch_key;

    bool valid() const {
        return NativeProviderConsumerContract::accepts(metadata, dispatch_key);
    }
};

/**
 * Non-owning lifecycle sink for a native provider.
 *
 * Providers keep their device handles and kernel callables on their side of
 * this boundary.  The registry only publishes value objects (registration
 * metadata and dispatch keys), so a CUDA/ACL consumer can invalidate its
 * handles when a provider is replaced or removed without depending on the
 * registry's private containers.  Notifications are delivered after the
 * corresponding registry mutation; consumers must not retain references to
 * the callback arguments and must unregister before destruction.
 */
struct NativeProviderLifecycleObserver {
    virtual ~NativeProviderLifecycleObserver() = default;
    virtual void on_provider_registered(
        const NativeProviderRegistration& registration, uint32 provider_id) = 0;
    virtual void on_provider_unregistered(
        const NativeProviderRegistration& registration, uint32 provider_id) = 0;
    virtual void on_provider_op_bound(const NativeOpDispatchKey& key) = 0;
    virtual void on_provider_op_unbound(const NativeOpDispatchKey& key) = 0;
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
    void register_provider(const NativeProviderRegistration& registration,
                           bool replace = false);
    void register_provider(const string& provider, bool replace = false);
    bool has_provider(const string& provider) const;
    vector<string> providers() const;
    uint32 provider_id(const string& provider) const;
    NativeProviderRegistration provider_registration(const string& provider) const;
    // Return a value snapshot for host-side provider consumers.  No registry
    // storage or backend handle is exposed through this API.
    NativeProviderMetadata provider_metadata(const string& provider) const;
    // Atomically publish provider metadata and one bound operator key.  This
    // closes the replacement race between separate metadata/key lookups.
    NativeProviderConsumerDispatch provider_consumer_dispatch(
        const string& name, const string& provider) const;
    // Non-throwing consumer boundary for optional/backend probing.  On
    // failure ``dispatch`` is left untouched; successful publication writes
    // both metadata and dispatch key as one value snapshot.
    bool try_provider_consumer_dispatch(
        const string& name, const string& provider,
        NativeProviderConsumerDispatch& dispatch) const;
    // The observer is non-owning and receives future transitions only.  The
    // returned pointer is the previous observer, mirroring the node lifecycle
    // observer API and making scoped installation straightforward.
    NativeProviderLifecycleObserver* set_lifecycle_observer(
        NativeProviderLifecycleObserver* observer);
    // Clear only the observer that the caller previously installed. A stale
    // provider consumer must not detach a replacement observer during teardown.
    bool clear_lifecycle_observer(NativeProviderLifecycleObserver* observer);
    void bind_provider(const string& name, const string& provider);
    NativeOpDispatchKey resolve_provider(const string& name,
                                         const string& provider) const;
    // Consumer-side stale-key guard.  A key becomes invalid after provider
    // replacement, unbinding, or operator removal; this query never throws
    // and does not expose registry-owned storage to backend code.
    bool is_current(const NativeOpDispatchKey& dispatch_key) const;
    bool unregister_provider(const string& provider);
    // Identity-checked teardown for a provider owner.  A stale owner must
    // never remove a replacement that reused the same provider spelling.
    bool unregister_provider_if_current(const string& provider,
                                        uint32 provider_id);

private:
    static string key(const string& name);

    mutable std::recursive_mutex mutex;
    unordered_map<string, OpInfo> entries;
    unordered_map<string, unordered_set<string>> provider_bindings;
    unordered_map<string, uint32> provider_ids;
    unordered_map<string, NativeProviderRegistration> provider_registrations;
    NativeProviderLifecycleObserver* lifecycle_observer = nullptr;
    OpId next_id = 1;
    uint32 next_provider_id = 1;
};

/**
 * Scoped, non-owning installation of a provider lifecycle observer.
 *
 * Provider consumers commonly install their observer while constructing a
 * backend and must restore the previous observer on every exit path.  The
 * scope deliberately uses the identity-checked clear operation: if another
 * consumer replaced the observer before this scope is destroyed, teardown
 * leaves that replacement untouched instead of detaching it accidentally.
 */
class NativeProviderLifecycleObserverScope {
public:
    NativeProviderLifecycleObserverScope(
            NativeOpRegistry& registry,
            NativeProviderLifecycleObserver* observer)
        : registry(&registry), observer(observer), previous(nullptr) {
        ASSERT(observer) << "a lifecycle observer scope requires an observer";
        previous = registry.set_lifecycle_observer(observer);
    }

    ~NativeProviderLifecycleObserverScope() {
        if (registry && registry->clear_lifecycle_observer(observer))
            registry->set_lifecycle_observer(previous);
    }

    NativeProviderLifecycleObserverScope(
            const NativeProviderLifecycleObserverScope&) = delete;
    NativeProviderLifecycleObserverScope& operator=(
            const NativeProviderLifecycleObserverScope&) = delete;

private:
    NativeOpRegistry* registry;
    NativeProviderLifecycleObserver* observer;
    NativeProviderLifecycleObserver* previous;
};

/**
 * Own one provider registration for the lifetime of a backend object.
 *
 * The registry still owns the registration record; this scope only carries
 * the provider spelling and instance id needed for safe teardown.  Its
 * destructor is intentionally non-throwing and leaves a newer replacement
 * untouched when the original provider was already superseded.
 */
class NativeProviderRegistrationScope {
public:
    NativeProviderRegistrationScope(
            NativeOpRegistry& registry,
            const NativeProviderRegistration& registration,
            bool replace = false)
        : registry(&registry), provider(registration.name), provider_id(0) {
        registry.register_provider(registration, replace);
        provider_id = registry.provider_id(provider);
    }

    ~NativeProviderRegistrationScope() {
        if (registry && provider_id)
            registry->unregister_provider_if_current(provider, provider_id);
    }

    NativeProviderRegistrationScope(
            const NativeProviderRegistrationScope&) = delete;
    NativeProviderRegistrationScope& operator=(
            const NativeProviderRegistrationScope&) = delete;

    uint32 id() const { return provider_id; }

private:
    NativeOpRegistry* registry;
    string provider;
    uint32 provider_id;
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
