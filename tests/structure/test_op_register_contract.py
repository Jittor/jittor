"""No op constructor may be resolved at load time.

``get_op_info`` asserts the op is registered, and registration is itself a
static initialiser in another translation unit (``gen_ops.cc``'s
``int caller = (initer(), 0)``, ``op_utils.cc``'s ``init()``). C++ does not
order static initialisers across translation units, so a namespace-scope

    static auto make_binary = get_op_info("binary")
        .get_constructor<VarPtr, Var*, Var*, NanoString>();

is a lookup that *may* run before the thing it looks up exists. There were 113
of them. When the order does not hold the ASSERT throws out of a static
initialiser -- before ``main``, with no catch anywhere -- so the process
terminates with no message naming the op, the file, or the reason.

``op_constructor<...>("name")`` stores the name and resolves on first call, by
which time ``main`` is running. This test keeps the old spelling from coming
back, which matters because reintroducing it costs nothing today: the link
order that makes it work is stable until someone adds a file.

Function-local statics are fine and are not flagged -- they are already lazy.
The rule is therefore about *indentation*: a call at column 0 is at namespace
scope, and one inside a function is not.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (
    REPO_ROOT / "python" / "jittor" / "src",
    REPO_ROOT / "python" / "jittor" / "extern",
)

#: The registry's own implementation and its unit test name these on purpose.
ALLOWED = {
    Path("python/jittor/src/ops/op_register.cc"),
    Path("python/jittor/src/ops/op_register.h"),
    Path("python/jittor/src/tests/test_op_register.cc"),
}


def _sources():
    for root in SOURCE_ROOTS:
        if not root.is_dir():
            continue
        for suffix in ("*.cc", "*.h"):
            yield from sorted(root.rglob(suffix))


def test_no_op_is_looked_up_at_namespace_scope():
    offenders = []
    for path in _sources():
        relative = path.relative_to(REPO_ROOT)
        if relative in ALLOWED:
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if "get_op_info(" not in line:
                continue
            if line[:1].isspace():
                continue          # inside a function: already lazy
            if line.lstrip().startswith(("//", "*", "/*")):
                continue
            offenders.append("%s:%d: %s" % (relative, number, line.strip()))
    assert not offenders, (
        "these resolve an op at load time, which depends on an unspecified "
        "static-initialisation order; use op_constructor<...>(\"name\") "
        "instead:\n  " + "\n  ".join(offenders))


def test_op_registry_storage_is_lazily_initialized():
    source = (REPO_ROOT / "python/jittor/src/ops/op_register.cc").read_text(
        encoding="utf-8")
    header = (REPO_ROOT / "python/jittor/src/ops/op_register.h").read_text(
        encoding="utf-8")
    acl = (REPO_ROOT / "python/jittor/extern/acl/acl_op_exec.cc").read_text(
        encoding="utf-8")
    assert "unordered_map<string, OpInfo> op_info_map;" not in source
    assert "static OpId next_op_id = 1;" not in source
    assert "vector<OpByType*> op_types;" not in source
    assert "extern vector<OpByType*> op_types" not in header
    assert "extern unordered_map<string, OpInfo> op_info_map" not in acl


def test_op_constructors_are_not_erased_through_void_pointer_rtti():
    header = (REPO_ROOT / "python/jittor/src/ops/op_register.h").read_text(
        encoding="utf-8")
    compiler = (REPO_ROOT / "python/jittor/compiler.py").read_text(
        encoding="utf-8")
    utils = (REPO_ROOT / "python/jittor/src/ops/op_utils.cc").read_text(
        encoding="utf-8")
    assert "pair<const std::type_info*, void*>" not in header
    assert "typeid(func_t)" not in header
    assert "(void*)&{name}" not in compiler
    assert "(void*)&make_number" not in utils


def test_native_provider_lifecycle_consumer_is_value_only_and_non_owning():
    header = (REPO_ROOT / "python/jittor/src/ops/op_register.h").read_text(
        encoding="utf-8")
    source = (REPO_ROOT / "python/jittor/src/ops/op_register.cc").read_text(
        encoding="utf-8")
    jit_test = (REPO_ROOT / "python/jittor/src/tests/test_op_register.cc").read_text(
        encoding="utf-8")
    assert "struct NativeProviderLifecycleObserver" in header
    for method in (
            "on_provider_registered", "on_provider_unregistered",
            "on_provider_op_bound", "on_provider_op_unbound"):
        assert method in header
        assert method in source
        assert method in jit_test
    assert "NativeProviderLifecycleObserver* lifecycle_observer" in header
    assert "set_lifecycle_observer" in header
    assert "clear_lifecycle_observer" in header
    assert "NativeOpRegistry::clear_lifecycle_observer" in source
    assert "bool is_current(const NativeOpDispatchKey& dispatch_key) const" in header
    assert "NativeOpRegistry::is_current" in source
    assert "unbind_provider_if_current" in header
    assert "NativeOpRegistry::unbind_provider_if_current" in source
    # The lifecycle seam may not own a backend object or expose one through
    # the ABI header.  Providers retain their handles on their own side.
    assert "shared_ptr<NativeProviderLifecycleObserver>" not in header
    assert "unique_ptr<NativeProviderLifecycleObserver>" not in header
    assert "void*" not in header[header.index("struct NativeProviderLifecycleObserver"):
                                  header.index("class NativeOpRegistry")]


def test_native_provider_observer_scope_is_identity_checked_and_non_owning():
    header = (REPO_ROOT / "python/jittor/src/ops/op_register.h").read_text(
        encoding="utf-8")
    jit_test = (REPO_ROOT / "python/jittor/src/tests/test_op_register.cc").read_text(
        encoding="utf-8")
    scope = header[header.index("class NativeProviderLifecycleObserverScope"):
                   header.index("// Intentionally process-lived")]
    assert "class NativeProviderLifecycleObserverScope" in header
    assert "registry->clear_lifecycle_observer(observer)" in scope
    assert "registry->set_lifecycle_observer(previous)" in scope
    assert "shared_ptr" not in scope
    assert "unique_ptr" not in scope
    assert "void*" not in scope
    assert "NativeProviderLifecycleObserverScope scope" in jit_test
    assert "set_lifecycle_observer(&replacement)" in jit_test


def test_native_provider_registration_scope_teardown_is_identity_checked():
    header = (REPO_ROOT / "python/jittor/src/ops/op_register.h").read_text(
        encoding="utf-8")
    source = (REPO_ROOT / "python/jittor/src/ops/op_register.cc").read_text(
        encoding="utf-8")
    jit_test = (REPO_ROOT / "python/jittor/src/tests/test_op_register.cc").read_text(
        encoding="utf-8")
    assert "class NativeProviderRegistrationScope" in header
    assert "unregister_provider_if_current" in header
    assert "NativeOpRegistry::unregister_provider_if_current" in source
    scope = header[header.index("class NativeProviderRegistrationScope"):
                  header.index("// Intentionally process-lived")]
    assert "registry->unregister_provider_if_current(provider, provider_id)" in scope
    assert "NativeProviderRegistrationScope scope" in jit_test
    assert "!registry.unregister_provider_if_current" in jit_test


def test_native_provider_metadata_is_a_value_only_host_consumer_contract():
    header = (REPO_ROOT / "python/jittor/src/ops/op_register.h").read_text(
        encoding="utf-8")
    source = (REPO_ROOT / "python/jittor/src/ops/op_register.cc").read_text(
        encoding="utf-8")
    jit_test = (REPO_ROOT / "python/jittor/src/tests/test_op_register.cc").read_text(
        encoding="utf-8")
    assert "struct NativeProviderMetadata" in header
    assert "NativeProviderMetadata provider_metadata(const string& provider) const" in header
    assert "NativeOpRegistry::provider_metadata" in source
    assert "auto metadata = registry.provider_metadata" in jit_test
    metadata = header[header.index("struct NativeProviderMetadata"):
                     header.index("struct NativeOpDispatchKey")]
    assert "shared_ptr" not in metadata
    assert "unique_ptr" not in metadata
    assert "void*" not in metadata
    assert "provider_id" in metadata
    assert "abi_version" in metadata
    assert "struct_size" in metadata


def test_native_provider_consumer_contract_is_value_only_and_fail_closed():
    header = (REPO_ROOT / "python/jittor/src/ops/op_register.h").read_text(
        encoding="utf-8")
    jit_test = (REPO_ROOT / "python/jittor/src/tests/test_op_register.cc").read_text(
        encoding="utf-8")
    contract = header[header.index("struct NativeProviderConsumerContract"):
                      header.index("struct NativeProviderLifecycleObserver")]
    assert "struct NativeProviderConsumerContract" in header
    assert "static bool accepts(const NativeProviderMetadata& metadata)" in contract
    assert "static bool accepts(const NativeProviderMetadata& metadata," in contract
    assert "metadata.provider_id == dispatch_key.provider_id" in contract
    assert "metadata.abi_version == dispatch_key.abi_version" in contract
    assert "shared_ptr" not in contract
    assert "unique_ptr" not in contract
    assert "void*" not in contract
    assert "NativeProviderConsumerContract::accepts(metadata, key)" in jit_test
    assert "!NativeProviderConsumerContract::accepts(metadata, replacement_key)" in jit_test


def test_native_provider_consumer_dispatch_is_atomic_and_value_only():
    header = (REPO_ROOT / "python/jittor/src/ops/op_register.h").read_text(
        encoding="utf-8")
    source = (REPO_ROOT / "python/jittor/src/ops/op_register.cc").read_text(
        encoding="utf-8")
    jit_test = (REPO_ROOT / "python/jittor/src/tests/test_op_register.cc").read_text(
        encoding="utf-8")
    assert "struct NativeProviderConsumerDispatch" in header
    assert "NativeProviderConsumerDispatch provider_consumer_dispatch(" in header
    assert "NativeProviderConsumerDispatch provider_consumer_dispatch(\n        OpId op_id" in header
    assert "bool try_provider_consumer_dispatch(" in header
    assert "bool try_provider_consumer_dispatch(\n        OpId op_id" in header
    assert "NativeOpRegistry::provider_consumer_dispatch" in source
    assert "NativeOpRegistry::try_provider_consumer_dispatch" in source
    contract = header[header.index("struct NativeProviderConsumerDispatch"):
                      header.index("struct NativeProviderLifecycleObserver")]
    assert "NativeProviderMetadata metadata" in contract
    assert "NativeOpDispatchKey dispatch_key" in contract
    assert "NativeProviderConsumerContract::accepts(metadata, dispatch_key)" in contract
    assert "shared_ptr" not in contract
    assert "unique_ptr" not in contract
    assert "void*" not in contract
    assert "registry.provider_consumer_dispatch" in jit_test
    assert "get_op_id(\"jit_test_provider_dispatch\")" in jit_test
    assert "missing_id_dispatch" in jit_test
    assert "registry.try_provider_consumer_dispatch" in jit_test
    assert "!registry.is_current(consumer_dispatch.dispatch_key)" in jit_test


def test_native_provider_consumer_lease_is_generation_checked_and_non_owning():
    header = (REPO_ROOT / "python/jittor/src/ops/op_register.h").read_text(
        encoding="utf-8")
    jit_test = (REPO_ROOT / "python/jittor/src/tests/test_op_register.cc").read_text(
        encoding="utf-8")
    view = header[header.index("class NativeProviderConsumerLease"):
                  header.index("class NativeProviderLifecycleObserverScope")]
    assert "class NativeProviderConsumerLease" in header
    assert "registry->is_current(dispatch.dispatch_key)" in view
    assert "bool try_get(NativeProviderConsumerDispatch& result) const" in view
    assert "void reset()" in view
    assert "shared_ptr" not in view
    assert "unique_ptr" not in view
    assert "void*" not in view
    assert "NativeProviderConsumerLease lease(registry, dispatch)" in jit_test
    assert "!lease.valid()" in jit_test
    assert "!lease.try_get(copied)" in jit_test


def test_native_provider_abi_admission_has_one_host_contract():
    header = (REPO_ROOT / "python/jittor/src/ops/op_register.h").read_text(
        encoding="utf-8")
    jit_test = (REPO_ROOT / "python/jittor/src/tests/test_op_register.cc").read_text(
        encoding="utf-8")
    contract = header[header.index("struct NativeProviderAbiContract"):
                      header.index("struct NativeProviderRegistration")]
    assert "struct NativeProviderAbiContract" in contract
    assert "static bool version_matches(uint32 abi_version)" in contract
    assert "static bool size_supported(uint32 struct_size, uint32 minimum_size)" in contract
    assert "static bool accepts(uint32 abi_version, uint32 struct_size" in contract
    assert "NativeProviderAbiContract::accepts" in header
    assert "NativeProviderAbiContract::version_matches" in header
    assert "NativeProviderAbiContract::accepts" in jit_test


def test_native_provider_lifecycle_event_descriptor_is_host_jit_compatible():
    header = (REPO_ROOT / "python/jittor/src/ops/op_register.h").read_text(
        encoding="utf-8")
    jit_test = (REPO_ROOT / "python/jittor/src/tests/test_op_register.cc").read_text(
        encoding="utf-8")
    event = header[header.index("enum NativeProviderLifecycleEventKind"):
                   header.index("struct NativeProviderLifecycleObserver")]
    assert "NATIVE_PROVIDER_LIFECYCLE_ABI_VERSION" in event
    assert "struct NativeProviderLifecycleAbiContract" in event
    assert "struct NativeProviderLifecycleEvent" in event
    assert "provider_registered" in event
    assert "provider_unregistered" in event
    assert "op_bound" in event
    assert "op_unbound" in event
    assert "NativeProviderConsumerContract::accepts" in event
    assert "event.metadata.valid() && !event.dispatch_key.valid()" in event
    assert "native_provider_lifecycle_event_is_value_only_and_fail_closed" in jit_test
    assert "NATIVE_PROVIDER_EVENT_REGISTERED" in jit_test
    assert "NATIVE_PROVIDER_EVENT_OP_BOUND" in jit_test
    assert "mixed.metadata.provider_id += 1" in jit_test
