"""Metadata-only repr regression: no Tensor construction or model execution."""
import pytest
import torch
from torch.nn.modules.module import Module as PublicModule
from jittor._core.module import Module as NativeModule
from jittor.compat.torch.installers.nn import module_methods
from jittor.compat.torch.nn_frontend import LayerInitializer
from jittor.compat.torch.fidelity import Fidelity, fidelity_of


def test_default_repr_owner_and_public_alias():
    assert PublicModule is torch.nn.Module
    assert torch.nn.Module.extra_repr is module_methods._module_extra_repr
    assert NativeModule.extra_repr is not module_methods._module_extra_repr
    assert torch.nn.Module().extra_repr() == ""
    assert fidelity_of("torch.nn.Module.extra_repr").level == Fidelity.EXACT


def test_callable_initializer_without_code_repr_and_nested_child():
    class MetadataLeaf(NativeModule):
        def __init__(self):
            super().__init__()
            self.label = "metadata only"

    native_extra_repr = MetadataLeaf.extra_repr
    owner = torch.nn.Module._nn_frontend_owner
    adapted = owner.adapt_class(MetadataLeaf)
    initializer = adapted.__dict__["__init__"]
    assert isinstance(initializer, LayerInitializer)
    assert not hasattr(initializer, "__code__")
    leaf = adapted()
    assert leaf.extra_repr() == ""
    assert repr(leaf) == "MetadataLeaf()"
    container = torch.nn.Module()
    container.child = leaf
    assert "MetadataLeaf()" in repr(container)
    assert MetadataLeaf.extra_repr is native_extra_repr
    assert NativeModule.extra_repr is native_extra_repr


def test_explicit_extra_repr_override_is_preserved_and_errors_propagate():
    class Custom(torch.nn.Module):
        def extra_repr(self):
            return "custom=kept"

    class Broken(torch.nn.Module):
        def extra_repr(self):
            raise ValueError("explicit repr error")

    assert Custom().extra_repr() == "custom=kept"
    assert "custom=kept" in repr(Custom())
    with pytest.raises(ValueError, match="explicit repr error"):
        repr(Broken())
