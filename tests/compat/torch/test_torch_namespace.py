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


def test_distribution_manifest_is_importable_and_parent_complete():
    """Standalone package metadata must not depend on a selected backend."""
    from jittor.compat.torch.distribution import (
        DISTRIBUTION_MODULES, DISTRIBUTION_PACKAGE_ALIASES,
        distribution_manifest, distribution_package_names,
        validate_distribution_aliases, validate_distribution_graph,
        validate_distribution_manifest,
    )

    assert DISTRIBUTION_MODULES[0] == "torch.distributed"
    assert "torch.distributed.fsdp" in distribution_package_names()
    assert validate_distribution_graph(DISTRIBUTION_MODULES)
    assert validate_distribution_aliases(DISTRIBUTION_MODULES)
    manifest = distribution_manifest()
    assert validate_distribution_manifest(manifest)
    assert all(left in DISTRIBUTION_MODULES for left, _ in DISTRIBUTION_PACKAGE_ALIASES)

    assert manifest["root"] == "torch.distributed"
    assert manifest["modules"] is DISTRIBUTION_MODULES
    assert manifest["aliases"] is DISTRIBUTION_PACKAGE_ALIASES


def test_distribution_boundary_facade_validates_manifest_and_publication():
    from jittor.compat.shim import bootstrap
    from jittor.compat.torch.distribution import validate_distribution_boundary

    assert bootstrap.DISTRIBUTION_ROOT == "torch.distributed"
    assert bootstrap.validate_distribution_boundary is validate_distribution_boundary
    assert bootstrap.validate_distribution_boundary()

    published, manifest = _small_distribution_fixture()
    assert bootstrap.validate_distribution_boundary(published, manifest)


def test_shim_facade_exports_publication_owner_and_binding_boundaries():
    from jittor.compat.shim import bootstrap
    from jittor.compat.shim import bind_published_namespace, namespace_owner
    from jittor.compat.torch.publication import (
        bind_published_namespace as direct_bind,
        namespace_owner as direct_owner,
    )

    assert bootstrap.bind_published_namespace is direct_bind
    assert bootstrap.namespace_owner is direct_owner
    assert bind_published_namespace is direct_bind
    assert namespace_owner is direct_owner


def test_distribution_alias_validation_rejects_missing_or_duplicate_endpoints():
    from jittor.compat.torch.distribution import validate_distribution_aliases

    try:
        validate_distribution_aliases(("torch.distributed",),
                                      (("torch.distributed", "torch.missing"),))
    except ValueError as error:
        assert "endpoint is missing" in str(error)
    else:
        raise AssertionError("missing alias endpoint was accepted")

    try:
        validate_distribution_aliases(
            ("a", "b"), (("a", "b"), ("a", "b")))
    except ValueError as error:
        assert "declared more than once" in str(error)
    else:
        raise AssertionError("duplicate alias source was accepted")


def test_distribution_alias_validation_rejects_malformed_entries():
    from jittor.compat.torch.distribution import validate_distribution_aliases

    for aliases in (
        (("torch.distributed",),),
        (("torch.distributed", "torch.distributed.tensor", "extra"),),
    ):
        try:
            validate_distribution_aliases(
                ("torch.distributed", "torch.distributed.tensor"), aliases
            )
        except ValueError as error:
            assert "must be a pair" in str(error)
        else:
            raise AssertionError("malformed alias entry was accepted")

    try:
        validate_distribution_aliases(
            ("torch.distributed", "torch.distributed.tensor"),
            (("torch.distributed", 7),),
        )
    except ValueError as error:
        assert "endpoints must be strings" in str(error)
    else:
        raise AssertionError("non-string alias endpoint was accepted")

    try:
        validate_distribution_aliases(
            ("torch.distributed",), aliases=None
        )
    except ValueError as error:
        assert "iterable of pairs" in str(error)
    else:
        raise AssertionError("non-iterable alias metadata was accepted")


def test_distribution_manifest_validation_rejects_inconsistent_package_closure():
    from jittor.compat.torch.distribution import validate_distribution_manifest

    malformed = {
        "root": "torch.distributed",
        "modules": ("torch.distributed", "torch.distributed.tensor._api"),
        "packages": ("torch.distributed",),
        "aliases": (),
    }
    try:
        validate_distribution_manifest(malformed)
    except ValueError as error:
        assert "package closure mismatch" in str(error)
    else:
        raise AssertionError("incomplete package closure was accepted")


def test_distribution_manifest_validation_rejects_missing_alias_endpoints():
    from jittor.compat.torch.distribution import validate_distribution_manifest

    malformed = {
        "root": "torch.distributed",
        "modules": (
            "torch.distributed",
            "torch.distributed.tensor",
            "torch.distributed.tensor._api",
        ),
        "packages": ("torch.distributed", "torch.distributed.tensor"),
        "aliases": (("torch.distributed.tensor._missing", "torch.distributed.tensor"),),
    }
    try:
        validate_distribution_manifest(malformed)
    except ValueError as error:
        assert "endpoint is missing" in str(error)
    else:
        raise AssertionError("missing alias endpoint was accepted")


def _small_distribution_fixture():
    """Build a backend-free live publication graph for validator tests."""
    import types

    names = (
        "torch.distributed",
        "torch.distributed.tensor",
        "torch.distributed.tensor._api",
        "torch.distributed._tensor",
    )
    packages = ("torch.distributed", "torch.distributed.tensor")
    aliases = (("torch.distributed._tensor", "torch.distributed.tensor"),)
    modules = {name: types.ModuleType(name) for name in names}
    for name in packages:
        modules[name].__path__ = []
    modules["torch.distributed"].tensor = modules["torch.distributed.tensor"]
    modules["torch.distributed.tensor"]._api = modules[
        "torch.distributed.tensor._api"
    ]
    modules["torch.distributed"]._tensor = modules["torch.distributed._tensor"]
    manifest = {
        "root": "torch.distributed",
        "modules": names,
        "packages": packages,
        "aliases": aliases,
    }
    return modules, manifest


def test_distribution_publication_validator_checks_live_parent_bindings():
    from jittor.compat.torch.distribution import validate_distribution_publication

    published, manifest = _small_distribution_fixture()
    assert validate_distribution_publication(published, manifest)


def test_distribution_publication_validator_rejects_unbound_or_malformed_nodes():
    from jittor.compat.torch.distribution import validate_distribution_publication

    published, manifest = _small_distribution_fixture()
    del published["torch.distributed"].tensor
    try:
        validate_distribution_publication(published, manifest)
    except ValueError as error:
        assert "does not bind child" in str(error)
    else:
        raise AssertionError("unbound publication child was accepted")

    published, manifest = _small_distribution_fixture()
    published["torch.distributed.tensor"].__path__ = None
    try:
        validate_distribution_publication(published, manifest)
    except ValueError as error:
        assert "has no __path__" in str(error)
    else:
        raise AssertionError("malformed package node was accepted")


def test_distribution_boundary_rejects_alias_object_collapse():
    from jittor.compat.torch.distribution import validate_distribution_boundary

    published, manifest = _small_distribution_fixture()
    published["torch.distributed._tensor"] = published["torch.distributed.tensor"]
    try:
        validate_distribution_boundary(published, manifest)
    except ValueError as error:
        # Identity collapse is rejected at the module identity check before
        # the alias-specific guard because a ModuleType can expose only one
        # ``__name__`` value.
        assert "wrong __name__" in str(error)
    else:
        raise AssertionError("distribution alias object collapse was accepted")


def test_distribution_manifest_has_no_backend_import_dependency():
    import ast
    import pathlib

    source = pathlib.Path(__file__).resolve().parents[3] / "python/jittor/compat/torch/distribution.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        or (isinstance(node, ast.ImportFrom) and node.module != "__future__")
    ]
    assert not imports


def test_bootstrap_and_lazy_shim_expose_the_same_distribution_boundary():
    from jittor.compat.shim import bootstrap, distribution_manifest
    from jittor.compat.torch.distribution import validate_distribution_bootstrap

    assert bootstrap.distribution_manifest is distribution_manifest
    manifest = distribution_manifest()
    assert manifest["root"] == "torch.distributed"
    assert "torch.distributed.fsdp" in manifest["modules"]
    assert bootstrap.validate_distribution_bootstrap is validate_distribution_bootstrap
    assert validate_distribution_bootstrap(bootstrap)


def test_distribution_bootstrap_rejects_copied_or_wrapped_exports():
    from jittor.compat.shim import bootstrap
    from jittor.compat.torch.distribution import validate_distribution_bootstrap

    class Wrapper:
        pass

    wrapper = Wrapper()
    for name in (
        "DISTRIBUTION_ROOT",
        "DISTRIBUTION_MODULES",
        "DISTRIBUTION_PACKAGE_ALIASES",
        "distribution_manifest",
        "distribution_module_names",
        "distribution_package_names",
        "validate_distribution_aliases",
        "validate_distribution_manifest",
        "validate_distribution_graph",
        "validate_distribution_publication",
        "validate_distribution_boundary",
        "validate_distribution_bootstrap",
    ):
        setattr(wrapper, name, getattr(bootstrap, name))
    wrapper.distribution_manifest = lambda: bootstrap.distribution_manifest()
    try:
        validate_distribution_bootstrap(wrapper)
    except ValueError as error:
        assert "canonical object" in str(error)
    else:
        raise AssertionError("wrapped bootstrap export was accepted")

    del wrapper.validate_distribution_bootstrap
    wrapper.distribution_manifest = bootstrap.distribution_manifest
    try:
        validate_distribution_bootstrap(wrapper)
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("incomplete bootstrap surface was accepted")


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


def test_publication_helper_binds_registry_and_root_as_one_operation():
    from jittor.compat.torch.publication import publish_independent_namespace
    from jittor.compat.transaction import ActivationTransaction

    owner = types.ModuleType("jittor")
    namespace = independent_torch_namespace(owner)
    child = types.ModuleType("torch.nn")
    registry = types.SimpleNamespace(_published={"torch": owner, "torch.nn": child})
    transaction = ActivationTransaction("namespace-publication-boundary")
    transaction.acquire()
    try:
        publish_independent_namespace(namespace, registry, transaction=transaction)
        assert registry._published["torch"] is namespace
        assert namespace.nn is child
        transaction.rollback()
    finally:
        transaction.release()

    assert registry._published["torch"] is owner
    assert not hasattr(namespace, "nn")


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
