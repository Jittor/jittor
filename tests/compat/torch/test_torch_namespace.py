"""Identity and ownership contract for the independent Torch namespace seam."""

import sys
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


def test_publication_boundary_is_importable_without_runtime_or_installer():
    """The standalone distribution boundary must stay CUDA/NPU agnostic."""
    import importlib

    publication = importlib.import_module("jittor.compat.torch.publication")
    assert publication.__name__ == "jittor.compat.torch.publication"


def test_namespace_has_importable_package_spec():
    """A published detached root must remain a valid importlib package."""
    import importlib.util

    namespace = independent_torch_namespace(types.ModuleType("jittor"))
    previous = sys.modules.get("torch")
    sys.modules["torch"] = namespace
    try:
        spec = importlib.util.find_spec("torch")
    finally:
        if previous is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous

    assert spec is namespace.__spec__
    assert spec.name == "torch"
    assert spec.submodule_search_locations == []


def test_namespace_keeps_import_metadata_off_the_owner():
    owner = types.ModuleType("jittor")
    namespace = independent_torch_namespace(owner)
    marker = object()

    namespace.__spec__ = marker
    namespace.__loader__ = marker
    namespace.__file__ = "detached-torch"

    assert namespace.__spec__ is marker
    assert namespace.__loader__ is marker
    assert namespace.__file__ == "detached-torch"
    assert owner.__spec__ is None
    assert owner.__loader__ is None
    assert not hasattr(owner, "__file__")


def test_namespace_writes_public_values_to_explicit_owner_only():
    owner = types.ModuleType("jittor")
    namespace = TorchNamespace(owner)

    namespace.new_api = 42
    namespace._install_marker = "torch-only"

    assert owner.new_api == 42
    assert not hasattr(owner, "_install_marker")
    assert namespace._install_marker == "torch-only"
    assert namespace.owner is owner


def test_namespace_deletes_public_values_from_explicit_owner_only():
    owner = types.ModuleType("jittor")
    owner.optional_api = object()
    namespace = TorchNamespace(owner)

    del namespace.optional_api

    assert not hasattr(owner, "optional_api")
    assert not hasattr(namespace, "optional_api")


def test_namespace_deletes_metadata_locally_without_touching_owner():
    owner = types.ModuleType("jittor")
    owner.__file__ = "owner-jittor"
    namespace = TorchNamespace(owner)
    namespace.__file__ = "detached-torch"

    del namespace.__file__

    assert not hasattr(namespace, "__file__")
    assert owner.__file__ == "owner-jittor"


def test_namespace_deleted_metadata_does_not_fall_back_to_owner_identity():
    owner = types.ModuleType("jittor")
    owner.__name__ = "jittor-owner"
    namespace = TorchNamespace(owner)

    del namespace.__name__

    assert not hasattr(namespace, "__name__")
    assert owner.__name__ == "jittor-owner"


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


def test_owner_root_alias_is_rebound_to_independent_namespace():
    """The core installer's ``torch.torch`` alias must not leak the owner."""
    owner = types.ModuleType("jittor")
    namespace = independent_torch_namespace(owner)
    published = {"torch": owner, "torch.torch": owner}

    bind_published_namespace(namespace, published)

    assert published["torch.torch"] is namespace
    assert namespace.torch is namespace
    assert namespace.torch is not owner


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


def test_missing_published_parent_fails_closed_without_partial_binding():
    owner = types.ModuleType("jittor")
    namespace = independent_torch_namespace(owner)
    child = types.ModuleType("torch.nn.functional")
    from jittor.compat.transaction import ActivationTransaction

    transaction = ActivationTransaction("namespace-missing-parent")
    transaction.acquire()
    try:
        try:
            bind_published_namespace(
                namespace, {"torch.nn.functional": child}, transaction=transaction
            )
        except RuntimeError as error:
            assert "parent 'torch.nn' is not published" in str(error)
        else:
            raise AssertionError("missing namespace parent was accepted")
        transaction.rollback()
    finally:
        transaction.release()

    assert not hasattr(namespace, "nn")
    assert not hasattr(owner, "nn")


def test_missing_later_parent_fails_closed_without_transaction():
    """Preflight all parents before binding the first valid sibling."""
    owner = types.ModuleType("jittor")
    namespace = independent_torch_namespace(owner)
    nn = types.ModuleType("torch.nn")
    orphan = types.ModuleType("torch.optim.sgd")

    try:
        bind_published_namespace(
            namespace,
            {"torch.nn": nn, "torch.optim.sgd": orphan},
        )
    except RuntimeError as error:
        assert "parent 'torch.optim' is not published" in str(error)
    else:
        raise AssertionError("missing later parent was accepted")

    assert not hasattr(namespace, "nn")
    assert not hasattr(owner, "nn")


def test_missing_parent_does_not_leave_root_alias_rebound():
    """Preflight failure must not mutate the published root alias."""
    owner = types.ModuleType("jittor")
    namespace = independent_torch_namespace(owner)
    orphan = types.ModuleType("torch.optim.sgd")
    published = {
        "torch": owner,
        "torch.torch": owner,
        "torch.optim.sgd": orphan,
    }

    try:
        bind_published_namespace(namespace, published)
    except RuntimeError as error:
        assert "parent 'torch.optim' is not published" in str(error)
    else:
        raise AssertionError("missing parent was accepted")

    assert published["torch.torch"] is owner
    assert not hasattr(namespace, "optim")
