"""Identity and ownership contract for the independent Torch namespace seam."""

import types

from jittor.compat.torch.namespace import (
    TorchNamespace, bind_published_namespace, independent_torch_namespace,
    namespace_owner,
)


def test_namespace_has_independent_module_identity_and_delegates_public_api():
    owner = types.ModuleType("jittor")
    owner.answer = object()
    namespace = independent_torch_namespace(owner)

    assert isinstance(namespace, TorchNamespace)
    assert isinstance(namespace, types.ModuleType)
    assert namespace is not owner
    assert namespace.__name__ == "torch"
    assert namespace.answer is owner.answer


def test_namespace_writes_public_values_to_explicit_owner_only():
    owner = types.ModuleType("jittor")
    namespace = TorchNamespace(owner)

    namespace.new_api = 42
    namespace._install_marker = "torch-only"

    assert owner.new_api == 42
    assert not hasattr(owner, "_install_marker")
    assert namespace._install_marker == "torch-only"
    assert namespace.owner is owner


def test_namespace_owner_is_explicit_and_does_not_misidentify_plain_modules():
    owner = types.ModuleType("jittor")
    namespace = independent_torch_namespace(owner)

    assert namespace_owner(namespace) is owner
    assert namespace_owner(owner) is None


def test_published_children_bind_to_independent_root_and_nested_parent():
    owner = types.ModuleType("jittor")
    namespace = independent_torch_namespace(owner)
    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")

    bind_published_namespace(namespace, {
        "torch.nn": nn,
        "torch.nn.functional": functional,
    })

    assert namespace.nn is nn
    assert nn.functional is functional
    assert not hasattr(owner, "functional")


def test_published_root_entry_is_ignored_when_binding_children():
    owner = types.ModuleType("jittor")
    namespace = independent_torch_namespace(owner)
    nn = types.ModuleType("torch.nn")

    bind_published_namespace(namespace, {
        "torch": owner,
        "torch.nn": nn,
    })

    assert namespace.nn is nn
    assert namespace is not owner


def test_published_children_rollback_restores_owner_bindings():
    owner = types.ModuleType("jittor")
    owner.nn = types.SimpleNamespace(functional="old")
    namespace = independent_torch_namespace(owner)
    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")

    from jittor.compat.transaction import ActivationTransaction
    transaction = ActivationTransaction("namespace-test")
    transaction.acquire()
    try:
        bind_published_namespace(namespace, {
            "torch.nn": nn,
            "torch.nn.functional": functional,
        }, transaction=transaction)
        assert namespace.nn is nn
        assert nn.functional is functional
        transaction.rollback()
    finally:
        transaction.release()

    assert owner.nn is not None
    assert owner.nn.functional == "old"
    assert namespace.nn is owner.nn


def test_independent_root_registry_binding_rolls_back_with_import_identity():
    from jittor.compat.shim.runtime import _publish_registry_root
    from jittor.compat.transaction import ActivationTransaction

    owner = types.ModuleType("jittor")
    namespace = independent_torch_namespace(owner)
    registry = types.SimpleNamespace(_published={"torch": owner})
    transaction = ActivationTransaction("namespace-registry-test")
    transaction.acquire()
    try:
        _publish_registry_root(transaction, registry, namespace)
        assert registry._published["torch"] is namespace
        transaction.rollback()
    finally:
        transaction.release()

    assert registry._published["torch"] is owner


def test_independent_root_and_children_restore_import_identity_on_rollback():
    from jittor.compat.shim.runtime import _publish_registry_root
    from jittor.compat.transaction import ActivationTransaction

    owner = types.ModuleType("jittor")
    old_root = types.ModuleType("torch")
    old_nn = types.ModuleType("torch.nn")
    old_root.nn = old_nn
    namespace = independent_torch_namespace(owner)
    new_nn = types.ModuleType("torch.nn")
    registry = types.SimpleNamespace(
        _published={"torch": owner, "torch.nn": new_nn}
    )
    modules = {"torch": old_root, "torch.nn": old_nn}
    transaction = ActivationTransaction("namespace-import-identity")
    transaction.acquire()
    try:
        bind_published_namespace(namespace, registry._published, transaction=transaction)
        _publish_registry_root(transaction, registry, namespace)
        transaction.record(modules, "torch", old_root, namespace)
        modules["torch"] = namespace
        assert modules["torch"] is namespace
        assert modules["torch.nn"] is old_nn
        assert namespace.nn is new_nn
        transaction.rollback()
    finally:
        transaction.release()

    assert modules == {"torch": old_root, "torch.nn": old_nn}
    assert registry._published == {"torch": owner, "torch.nn": new_nn}
    assert old_root.nn is old_nn
