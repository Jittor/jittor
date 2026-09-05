"""Identity and ownership contract for the independent Torch namespace seam."""

import types

from jittor.compat.torch.namespace import (
    TorchNamespace, independent_torch_namespace, namespace_owner,
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
