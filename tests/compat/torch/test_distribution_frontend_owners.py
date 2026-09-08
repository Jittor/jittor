"""Distribution type configuration delegates to module-owned implementations."""
import ast
import inspect
import pickle
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import jittor as jt
from jittor.compat.torch.tensor_state import compatibility_owner
from jittor.compat.torch import distribution_frontend
from jittor.compat.torch.distribution_adapters import (
    DistributionAdapterState, DistributionConstructor, DistributionMethod,
)


def test_type_factory_only_configures_and_members_have_real_owners():
    for function in (distribution_frontend.make_distribution_frontend, DistributionAdapterState.adapt):
        node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        assert not [child for child in ast.walk(node) if child is not node and
                    isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.Lambda))]
    torch = compatibility_owner(jt)
    cls = torch.distributions.Normal
    assert isinstance(cls.__init__, DistributionConstructor)
    assert isinstance(cls.log_prob, DistributionMethod)
    assert cls.log_prob.__module__ == "jittor.compat.torch.distribution_adapters"
    assert "<locals>" not in DistributionMethod.__call__.__qualname__
    assert inspect.signature(cls.log_prob) == inspect.signature(jt.distributions.Normal.log_prob)
    from jittor.compat.torch.fidelity import fidelity_of
    assert fidelity_of("torch.distributions.Normal.log_prob").implementation is cls.log_prob


def test_constructor_aliases_validation_and_sampling_policy():
    torch = compatibility_owner(jt)
    constant = torch.distributions.Normal(loc=0., scale=1.)
    assert type(constant.loc) is torch.Tensor
    assert not constant.loc.requires_grad
    assert not constant.rsample((2,)).requires_grad
    with pytest.raises(TypeError, match="both loc and mu"):
        torch.distributions.Normal(mu=0., loc=1., scale=1.)
    with pytest.raises(NotImplementedError, match="validate_args"):
        torch.distributions.Normal(0., 1., validate_args=True)
    with pytest.raises(ValueError, match="high > low"):
        torch.distributions.Uniform(2., 1.)
    mean = torch.tensor(0., requires_grad=True)
    differentiable = torch.distributions.Normal(mean, 1.)
    assert differentiable.loc is mean
    sample = differentiable.rsample((2,))
    assert type(sample) is torch.Tensor and sample.requires_grad
    sample.sum().backward()
    np.testing.assert_allclose(mean.grad.numpy(), 2.)
    mean.requires_grad_(False)
    from jittor.compat.torch.nested import _torch_prune_leaf_registry
    _torch_prune_leaf_registry()


def test_distribution_class_instance_and_member_pickle_identity():
    torch = compatibility_owner(jt)
    cls = torch.distributions.Normal
    assert pickle.loads(pickle.dumps(cls)) is cls
    assert pickle.loads(pickle.dumps(cls.log_prob)) is cls.log_prob
    assert pickle.loads(pickle.dumps(cls.__init__)) is cls.__init__
    original = cls(1., 2.)
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is cls
    assert isinstance(restored, torch.distributions.Distribution)
    value = torch.tensor(0.)
    np.testing.assert_allclose(restored.log_prob(value).numpy(), original.log_prob(value).numpy())
    new_mean = torch.tensor(3.)
    restored.loc = new_mean
    assert restored.mu is new_mean


def test_adapter_state_is_per_frontend_and_native_classes_are_untouched():
    class TensorA:
        def __init__(self, value):
            self.value = value
            self.requires_grad = False
        def stop_grad(self):
            self.requires_grad = False
            return self
    class TensorB(TensorA):
        pass
    class Base:
        pass
    class Normal(Base):
        def __init__(self, mu, sigma):
            self.mu, self.sigma = mu, sigma
        @property
        def mean(self):
            return self.mu
    class Uniform(Base):
        pass
    native = SimpleNamespace(Distribution=Base, Normal=Normal, Uniform=Uniform)
    def owner(tensor_type):
        return SimpleNamespace(Var=tensor_type, get_default_dtype=lambda: "float32",
                               as_tensor=lambda value, dtype=None: tensor_type(value))
    before = vars(Normal).copy()
    first = DistributionAdapterState(native, owner(TensorA))
    second = DistributionAdapterState(native, owner(TensorB))
    a, b = first.adapt(Normal), second.adapt(Normal)
    assert first.adapt(Normal) is a and a is not b
    left, right = a(0., 1.), b(0., 1.)
    assert type(left.mean) is TensorA and type(right.mean) is TensorB
    assert isinstance(left, first.adapt(Base)) and not isinstance(left, second.adapt(Base))
    assert vars(Normal) == before
