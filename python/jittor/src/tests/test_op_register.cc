// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "op.h"
#include "var.h"
#include "ops/op_register.h"

namespace jittor {

// The registry is read by the name truncated at the first '.' -- that is how
// `name_ex()` spellings like "binary.add" resolve to the "binary" op. It used
// to be *written* by the untruncated name, so the two agreed only as long as
// no registered name contained a dot. One that did went in under a key nothing
// could ever look up, and every spelling of it reported "Op not found".
JIT_TEST(op_register_reads_and_writes_the_same_key) {
    const char* dotted = "jit_test_op.variant";
    op_registe({dotted, "", ""});

    ASSERT(has_op(dotted)) << "registered under one key, looked up under another";
    ASSERT(has_op("jit_test_op")) << "the truncated spelling must resolve too";
    ASSERTop(get_op_info(dotted).name,==,string(dotted));
    ASSERTop(get_op_info("jit_test_op").name,==,string(dotted));
    CHECKop(get_op_id(dotted),==,get_op_id("jit_test_op"));
    CHECKop(get_op_id(dotted),!=,(OpId)0);

    // and the ordinary case is unchanged: "binary.add" finds "binary"
    ASSERT(has_op("binary.add"));
    ASSERTop(get_op_info("binary.add").name,==,string("binary"));
    CHECKop(get_op_id("binary.add"),==,op_ids::binary());
    CHECKop(op_ids::binary(),!=,op_ids::array());
}

// NativeOpRegistry is the ownership boundary for storage and lifecycle.  A
// local instance is useful in tests and for a future backend provider: it can
// publish and tear down its registrations without touching the process-wide
// compatibility facade above.
JIT_TEST(native_op_registry_owns_lifecycle) {
    NativeOpRegistry registry;
    registry.register_op({"jit_test_native_registry", "test", ""});
    ASSERT(registry.has("jit_test_native_registry"));
    ASSERT(registry.get("jit_test_native_registry").id != (OpId)0);
    ASSERT(registry.names().size() == 1);
    ASSERT(registry.unregister("jit_test_native_registry"));
    ASSERT(!registry.has("jit_test_native_registry"));
    ASSERT(!registry.unregister("jit_test_native_registry"));
}

// Provider bindings carry only the process-local OpId and provider spelling.
// This is the native ABI seam: backend-specific kernel handles stay owned by
// the provider and do not leak into op_register.h.
JIT_TEST(native_op_registry_provider_dispatch_boundary) {
    NativeOpRegistry registry;
    registry.register_op({"jit_test_provider_dispatch", "test", ""});
    NativeProviderRegistration cpu_registration("cpu");
    registry.register_provider(cpu_registration);
    registry.register_provider("cuda");
    registry.bind_provider("jit_test_provider_dispatch", "cpu");

    auto key = registry.resolve_provider("jit_test_provider_dispatch", "cpu");
    ASSERT(key.op_id != (OpId)0);
    ASSERT(key.provider == "cpu");
    ASSERT(key.valid());
    ASSERT(key.provider_id == registry.provider_id("cpu"));
    ASSERT(key.abi_version == NATIVE_PROVIDER_ABI_VERSION);
    ASSERT(registry.is_current(key));
    auto published = registry.provider_registration("cpu");
    ASSERT(published.name == cpu_registration.name);
    ASSERT(published.abi_version == cpu_registration.abi_version);
    ASSERT(published.struct_size == cpu_registration.struct_size);
    auto metadata = registry.provider_metadata("cpu");
    ASSERT(metadata.valid());
    ASSERT(metadata.name == "cpu");
    ASSERT(metadata.provider_id == key.provider_id);
    ASSERT(metadata.abi_version == key.abi_version);
    ASSERT(metadata.struct_size == sizeof(NativeProviderRegistration));
    ASSERT(NativeProviderConsumerContract::accepts(metadata));
    ASSERT(NativeProviderConsumerContract::accepts(metadata, key));
    auto consumer_dispatch = registry.provider_consumer_dispatch(
        "jit_test_provider_dispatch", "cpu");
    ASSERT(consumer_dispatch.valid());
    ASSERT(consumer_dispatch.metadata.provider_id == key.provider_id);
    ASSERT(consumer_dispatch.dispatch_key.op_id == key.op_id);
    ASSERT(consumer_dispatch.dispatch_key.provider_id == key.provider_id);
    ASSERT(registry.is_current(consumer_dispatch.dispatch_key));
    // Generated/JIT consumers already hold the registered OpId.  The ID
    // boundary must publish the same coherent value pair without a name
    // lookup, and unknown ids must fail closed.
    auto id_dispatch = registry.provider_consumer_dispatch(
        get_op_id("jit_test_provider_dispatch"), "cpu");
    ASSERT(id_dispatch.valid());
    ASSERT(id_dispatch.metadata.provider_id == consumer_dispatch.metadata.provider_id);
    ASSERT(id_dispatch.dispatch_key.op_id == consumer_dispatch.dispatch_key.op_id);
    NativeProviderConsumerDispatch missing_id_dispatch;
    ASSERT(!registry.try_provider_consumer_dispatch(
        (OpId)0x7fffffff, "cpu", missing_id_dispatch));
    ASSERT(!registry.try_provider_consumer_dispatch(
        consumer_dispatch.dispatch_key.op_id, "missing_provider", missing_id_dispatch));
    NativeProviderConsumerDispatch optional_dispatch;
    ASSERT(registry.try_provider_consumer_dispatch(
        "jit_test_provider_dispatch", "cpu", optional_dispatch));
    ASSERT(optional_dispatch.valid());
    ASSERT(!registry.try_provider_consumer_dispatch(
        "missing_op", "cpu", optional_dispatch));
    ASSERT(!registry.try_provider_consumer_dispatch(
        "jit_test_provider_dispatch", "missing_provider", optional_dispatch));
    ASSERT(registry.providers().size() == 2);

    // Replacing a provider is a teardown boundary; stale bindings cannot
    // survive into a new provider instance.
    registry.register_provider("cpu", true);
    ASSERT(registry.has_provider("cpu"));
    auto replacement = registry.provider_id("cpu");
    ASSERT(replacement != key.provider_id);
    ASSERT(!registry.is_current(key));
    ASSERT(!registry.is_current(consumer_dispatch.dispatch_key));
    // The value remains structurally valid, but its key is stale after
    // replacement and must be rejected before a backend handle lookup.
    ASSERT(consumer_dispatch.valid());
    // The old value snapshot is safe to retain while the provider is
    // replaced; it must never alias registry-owned storage.
    ASSERT(metadata.provider_id != replacement);
    auto replacement_metadata = registry.provider_metadata("cpu");
    ASSERT(replacement_metadata.valid());
    ASSERT(replacement_metadata.provider_id == replacement);
    ASSERT(replacement_metadata.name == metadata.name);
    registry.bind_provider("jit_test_provider_dispatch", "cpu");
    auto replacement_key = registry.resolve_provider(
        "jit_test_provider_dispatch", "cpu");
    ASSERT(registry.is_current(replacement_key));
    // Each value remains individually well-formed, but a consumer must not
    // combine snapshots across provider instances.
    ASSERT(NativeProviderConsumerContract::accepts(replacement_metadata));
    ASSERT(!NativeProviderConsumerContract::accepts(metadata, replacement_key));
    ASSERT(!NativeProviderConsumerContract::accepts(replacement_metadata, key));
    ASSERT(NativeProviderConsumerContract::accepts(replacement_metadata,
                                                   replacement_key));
    ASSERT(!registry.unregister_provider("missing"));
    ASSERT(registry.unregister_provider("cpu"));
    ASSERT(!registry.has_provider("cpu"));
    ASSERT(!registry.is_current(replacement_key));
    ASSERT(registry.unregister("jit_test_provider_dispatch"));
}

// A host consumer keeps a value snapshot, but liveness is checked against the
// provider generation before a backend handle is used.  This is deliberately
// non-owning: replacement and teardown only invalidate the view.
JIT_TEST(native_op_registry_consumer_lease_is_generation_checked) {
    NativeOpRegistry registry;
    registry.register_op({"jit_test_provider_lease", "test", ""});
    registry.register_provider("cuda");
    registry.bind_provider("jit_test_provider_lease", "cuda");
    auto dispatch = registry.provider_consumer_dispatch(
        "jit_test_provider_lease", "cuda");
    NativeProviderConsumerLease lease(registry, dispatch);
    ASSERT(lease.valid());
    ASSERT((bool)lease);
    ASSERT(lease.get().dispatch_key.op_id == dispatch.dispatch_key.op_id);
    NativeProviderConsumerDispatch copied;
    ASSERT(lease.try_get(copied));
    ASSERT(copied.valid());

    // A replacement invalidates the old generation even though the value
    // snapshot remains well-formed and safe to retain.
    registry.register_provider("cuda", true);
    ASSERT(!lease.valid());
    ASSERT(!lease.try_get(copied));
    expect_error([&]() { lease.get(); });

    lease.reset();
    ASSERT(!lease.valid());
    ASSERT(!lease.try_get(copied));
    ASSERT(registry.unregister_provider("cuda"));
    ASSERT(registry.unregister("jit_test_provider_lease"));
}

JIT_TEST(native_op_registry_rejects_incompatible_provider_abi) {
    ASSERT(NativeProviderAbiContract::version_matches(
        NATIVE_PROVIDER_ABI_VERSION));
    ASSERT(!NativeProviderAbiContract::version_matches(
        NATIVE_PROVIDER_ABI_VERSION + 1));
    ASSERT(NativeProviderAbiContract::size_supported(
        sizeof(NativeProviderRegistration), sizeof(NativeProviderRegistration)));
    ASSERT(NativeProviderAbiContract::size_supported(
        sizeof(NativeProviderRegistration) + 8,
        sizeof(NativeProviderRegistration)));
    ASSERT(!NativeProviderAbiContract::size_supported(
        sizeof(NativeProviderRegistration) - 1,
        sizeof(NativeProviderRegistration)));
    ASSERT(NativeProviderAbiContract::accepts(
        NATIVE_PROVIDER_ABI_VERSION,
        sizeof(NativeProviderRegistration),
        sizeof(NativeProviderRegistration)));

    NativeOpRegistry registry;
    expect_error([&]() {
        registry.register_provider(NativeProviderRegistration(
            "bad_abi", NATIVE_PROVIDER_ABI_VERSION + 1));
    });
    expect_error([&]() {
        registry.register_provider(NativeProviderRegistration(
            "truncated", NATIVE_PROVIDER_ABI_VERSION,
            sizeof(NativeProviderRegistration) - 1));
    });
    ASSERT(!registry.has_provider("bad_abi"));
    ASSERT(!registry.has_provider("truncated"));
}

JIT_TEST(native_provider_lifecycle_event_is_value_only_and_fail_closed) {
    NativeProviderRegistration registration("jit_test_event_provider");
    const uint32 provider_id = 17;
    const NativeProviderMetadata metadata(registration, provider_id);
    const NativeOpDispatchKey key = {42, registration.name, provider_id,
                                     registration.abi_version};

    auto registered = NativeProviderLifecycleEvent::provider_registered(
        registration, provider_id);
    ASSERT(registered.valid());
    ASSERT(registered.kind == NATIVE_PROVIDER_EVENT_REGISTERED);
    ASSERT(registered.metadata.provider_id == provider_id);
    ASSERT(!registered.dispatch_key.valid());

    auto unregistered = NativeProviderLifecycleEvent::provider_unregistered(
        registration, provider_id);
    ASSERT(unregistered.valid());
    ASSERT(unregistered.kind == NATIVE_PROVIDER_EVENT_UNREGISTERED);

    auto bound = NativeProviderLifecycleEvent::op_bound(metadata, key);
    ASSERT(bound.valid());
    ASSERT(bound.kind == NATIVE_PROVIDER_EVENT_OP_BOUND);
    ASSERT(bound.dispatch_key.op_id == key.op_id);
    ASSERT(bound.metadata.provider_id == key.provider_id);

    auto unbound = NativeProviderLifecycleEvent::op_unbound(metadata, key);
    ASSERT(unbound.valid());
    ASSERT(unbound.kind == NATIVE_PROVIDER_EVENT_OP_UNBOUND);

    // A value can retain well-formed fields but must fail closed when its
    // event kind or generation payload is inconsistent.
    auto mixed = bound;
    mixed.kind = NATIVE_PROVIDER_EVENT_REGISTERED;
    ASSERT(!mixed.valid());
    mixed = bound;
    mixed.metadata.provider_id += 1;
    ASSERT(!mixed.valid());
    mixed = bound;
    mixed.abi_version = NATIVE_PROVIDER_LIFECYCLE_ABI_VERSION + 1;
    ASSERT(!mixed.valid());
    mixed = bound;
    mixed.struct_size = sizeof(NativeProviderLifecycleEvent) - 1;
    ASSERT(!mixed.valid());
}

// Provider implementations consume lifecycle events instead of reaching
// into registry storage.  The observer is deliberately non-owning and keys
// remain value objects, so replacement/unregister can invalidate stale device
// handles before a new provider instance is published.
struct NativeProviderLifecycleProbe : NativeProviderLifecycleObserver {
    vector<string> events;
    vector<NativeOpDispatchKey> keys;

    void on_provider_registered(const NativeProviderRegistration& registration,
                                uint32 provider_id) override {
        events.push_back("registered:" + registration.name);
        ASSERT(provider_id != 0);
    }
    void on_provider_unregistered(const NativeProviderRegistration& registration,
                                  uint32 provider_id) override {
        events.push_back("unregistered:" + registration.name);
        ASSERT(provider_id != 0);
    }
    void on_provider_op_bound(const NativeOpDispatchKey& key) override {
        events.push_back("bound:" + key.provider);
        ASSERT(key.valid());
        keys.push_back(key);
    }
    void on_provider_op_unbound(const NativeOpDispatchKey& key) override {
        events.push_back("unbound:" + key.provider);
        ASSERT(key.valid());
        keys.push_back(key);
    }
};

JIT_TEST(native_op_registry_lifecycle_consumer_boundary) {
    NativeOpRegistry registry;
    NativeProviderLifecycleProbe probe;
    registry.register_op({"jit_test_provider_consumer", "test", ""});
    ASSERT(registry.set_lifecycle_observer(&probe) == nullptr);
    registry.register_provider("cuda");
    registry.bind_provider("jit_test_provider_consumer", "cuda");
    // Binding the same operator twice is idempotent and must not publish a
    // duplicate consumer event.
    registry.bind_provider("jit_test_provider_consumer", "cuda");
    ASSERT(probe.events.size() == 2);
    ASSERT(probe.events[0] == "registered:cuda");
    ASSERT(probe.events[1] == "bound:cuda");

    registry.register_provider("cuda", true);
    ASSERT(probe.events.size() == 5);
    ASSERT(probe.events[2] == "unbound:cuda");
    ASSERT(probe.events[3] == "unregistered:cuda");
    ASSERT(probe.events[4] == "registered:cuda");

    registry.bind_provider("jit_test_provider_consumer", "cuda");
    ASSERT(registry.unregister("jit_test_provider_consumer"));
    ASSERT(probe.events.size() == 7);
    ASSERT(probe.events[5] == "bound:cuda");
    ASSERT(probe.events[6] == "unbound:cuda");

    // Single-op teardown is generation checked: an old provider key cannot
    // remove a binding created by a replacement instance.
    registry.register_op({"jit_test_provider_unbind", "test", ""});
    registry.bind_provider("jit_test_provider_unbind", "cuda");
    auto unbind_key = registry.resolve_provider(
        "jit_test_provider_unbind", "cuda");
    ASSERT(registry.unbind_provider_if_current(unbind_key));
    ASSERT(!registry.is_current(unbind_key));
    ASSERT(!registry.unbind_provider_if_current(unbind_key));
    registry.bind_provider("jit_test_provider_unbind", "cuda");
    auto replacement_key = registry.resolve_provider(
        "jit_test_provider_unbind", "cuda");
    ASSERT(!registry.unbind_provider_if_current(unbind_key));
    ASSERT(registry.unbind_provider_if_current(replacement_key));

    ASSERT(registry.unregister_provider("cuda"));
    ASSERT(probe.events.size() == 8);
    ASSERT(probe.events[7] == "unregistered:cuda");
    ASSERT(registry.clear_lifecycle_observer(&probe));
    ASSERT(!registry.clear_lifecycle_observer(&probe));
}

JIT_TEST(native_op_registry_observer_teardown_is_identity_checked) {
    NativeOpRegistry registry;
    NativeProviderLifecycleProbe first;
    NativeProviderLifecycleProbe replacement;
    ASSERT(registry.set_lifecycle_observer(&first) == nullptr);
    ASSERT(registry.set_lifecycle_observer(&replacement) == &first);
    // A stale consumer must not detach the observer that replaced it.
    ASSERT(!registry.clear_lifecycle_observer(&first));
    ASSERT(registry.clear_lifecycle_observer(&replacement));
    ASSERT(registry.set_lifecycle_observer(nullptr) == nullptr);
}

JIT_TEST(native_op_registry_observer_scope_restores_identity) {
    NativeOpRegistry registry;
    NativeProviderLifecycleProbe first;
    NativeProviderLifecycleProbe replacement;
    registry.set_lifecycle_observer(&replacement);
    {
        NativeProviderLifecycleObserverScope scope(registry, &first);
        ASSERT(registry.set_lifecycle_observer(&first) == &first);
        // A scope must restore the observer that was active before it was
        // created, even when the consumer exits through normal teardown.
    }
    ASSERT(registry.set_lifecycle_observer(nullptr) == &replacement);

    // If another consumer takes ownership before scope teardown, the stale
    // scope must not detach that replacement observer.
    {
        NativeProviderLifecycleObserverScope scope(registry, &first);
        ASSERT(registry.set_lifecycle_observer(&replacement) == &first);
    }
    ASSERT(registry.set_lifecycle_observer(nullptr) == &replacement);
}

JIT_TEST(native_op_registry_registration_scope_is_identity_checked) {
    NativeOpRegistry registry;
    {
        NativeProviderRegistrationScope scope(
            registry, NativeProviderRegistration("jit_test_scoped_provider"));
        ASSERT(scope.id() != 0);
        ASSERT(registry.has_provider("jit_test_scoped_provider"));
        ASSERT(registry.provider_id("jit_test_scoped_provider") == scope.id());
    }
    ASSERT(!registry.has_provider("jit_test_scoped_provider"));

    // Replacing the provider invalidates the old scope's owner token.  Its
    // destructor must leave the replacement alive rather than unregistering
    // by spelling alone.
    {
        NativeProviderRegistrationScope scope(
            registry, NativeProviderRegistration("jit_test_scoped_provider"));
        const auto old_id = scope.id();
        registry.register_provider("jit_test_scoped_provider", true);
        const auto replacement_id = registry.provider_id(
            "jit_test_scoped_provider");
        ASSERT(replacement_id != old_id);
        ASSERT(!registry.unregister_provider_if_current(
            "jit_test_scoped_provider", old_id));
        ASSERT(registry.has_provider("jit_test_scoped_provider"));
    }
    ASSERT(!registry.has_provider("jit_test_scoped_provider"));
}

static NativeProviderRegistrationScope make_scoped_provider(
        NativeOpRegistry& registry) {
    return NativeProviderRegistrationScope(
        registry, NativeProviderRegistration("jit_test_movable_provider"));
}

JIT_TEST(native_op_registry_scopes_transfer_teardown_ownership) {
    NativeOpRegistry registry;
    NativeProviderLifecycleProbe replacement;
    registry.set_lifecycle_observer(&replacement);
    {
        auto registration = make_scoped_provider(registry);
        ASSERT(registration.id() != 0);
        ASSERT(registry.has_provider("jit_test_movable_provider"));

        NativeProviderLifecycleProbe observer;
        {
            NativeProviderLifecycleObserverScope first(registry, &observer);
            NativeProviderLifecycleObserverScope second(move(first));
            // The moved-from scope must not clear the active observer when it
            // leaves its inner lifetime; the destination owns restoration.
        }
        ASSERT(registry.set_lifecycle_observer(nullptr) == &replacement);
    }
    ASSERT(!registry.has_provider("jit_test_movable_provider"));
}

// A constructor resolved on first call, not at load time. The point is what
// does NOT happen at construction: no registry lookup, so no dependency on
// this translation unit's static initialiser running after the registry's.
JIT_TEST(op_constructor_resolves_lazily) {
    auto missing = op_constructor<VarPtr, Var*>("jit_test_no_such_op");
    // constructing it asked the registry nothing
    ASSERT(!(bool)missing);
    // and only calling it fails, where there is someone to catch it
    expect_error([&]() { missing(nullptr); });

    auto make_unary = op_constructor<VarPtr, Var*, NanoString>("unary");
    ASSERT((bool)make_unary);
    VarPtr a({4}, "float32");
    auto b = make_unary(a, ns_float64);
    ASSERT(b->dtype() == ns_float64);
    CHECK(b->input()->is_op(get_op_id("unary")));
}

}
