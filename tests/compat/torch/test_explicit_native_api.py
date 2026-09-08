"""Independent namespace publication and native mathematical delegation."""
import importlib
import pickle

import numpy as np
import pytest
import jittor as jt

from jittor.compat.torch.context import get_install_context
from jittor.compat.torch.tensor_state import compatibility_owner


@pytest.fixture
def frontend():
    target = compatibility_owner(jt)
    if target is jt:
        pytest.skip("independent frontend contract")
    return target


def test_completed_namespace_does_not_inherit_backend_attributes(frontend, monkeypatch):
    assert frontend._sealed
    for name in ("flags", "core", "runtime", "config", "clean_graph", "op_compiler"):
        assert not hasattr(frontend, name)
        assert name not in dir(frontend)
    marker = object()
    monkeypatch.setattr(jt, "native_only_new_attribute", marker, raising=False)
    assert not hasattr(frontend, "native_only_new_attribute")
    monkeypatch.setattr(frontend, "native_only_new_attribute", "local", raising=False)
    assert jt.native_only_new_attribute is marker


def test_native_operations_have_stable_objects_and_frontend_results(frontend):
    owner = importlib.import_module("jittor.compat.torch.native_api")
    from jittor.compat.torch.fidelity import fidelity_of
    for name in ("sin", "cos", "exp", "matmul"):
        function = getattr(frontend, name)
        assert function is getattr(owner, name)
        assert pickle.loads(pickle.dumps(function)) is function
        assert fidelity_of("torch." + name).implementation is function
    value = frontend.tensor([0., 1.], requires_grad=True)
    result = frontend.sin(value)
    assert type(result) is frontend.Tensor
    result.sum().backward()
    np.testing.assert_allclose(value.grad.numpy(), np.cos([0., 1.]), rtol=1e-6)
    assert type(jt.array([1.])) is jt.Var


def test_native_module_facades_do_not_publish_imported_modules(frontend, monkeypatch):
    assert frontend.fft is not jt.fft
    assert frontend.linalg is not jt.linalg
    assert not hasattr(frontend.fft, "jt")
    for name in ("decompositions", "solving", "norms"):
        assert not hasattr(frontend.linalg, name)
    marker = object()
    monkeypatch.setattr(frontend.fft, "local_setting", marker, raising=False)
    assert not hasattr(jt.fft, "local_setting")
    assert type(frontend.fft.fftfreq(4)) is frontend.Tensor
    assert frontend.jit.ScriptModule is frontend.nn.Module
    assert frontend.jit.ScriptModule is not jt.Module


def test_native_rebinding_preserves_delegates_and_explicit_overrides(frontend, monkeypatch):
    from jittor.compat.torch.native_api import install
    context = get_install_context(jt)
    before = dict(context.state["native_torch_operations"])
    original = frontend.sin
    sentinel = lambda value: value
    monkeypatch.setattr(frontend.linalg, "svd", sentinel)
    install(context)
    assert frontend.sin is original
    assert frontend.linalg.svd is sentinel
    assert all(context.state["native_torch_operations"][name] is value
               for name, value in before.items())
    np.testing.assert_allclose(frontend.sin(frontend.tensor([0.])).numpy(), [0.])


def test_vmap_owner_is_stable_and_uses_context_local_scope(frontend):
    owner = importlib.import_module("jittor.compat.torch.installers.numerical.batching")
    assert frontend.vmap is owner.vmap
    value = frontend.tensor([[1., 2.], [3., 4.]])
    result = frontend.vmap(lambda item: item * 2)(value)
    np.testing.assert_array_equal(result.numpy(), [[2., 4.], [6., 8.]])


def test_vmap_preserves_real_singleton_dimensions_and_scalar_rank(frontend):
    values = np.arange(6, dtype=np.float32).reshape(3, 2, 1)
    tensor = frontend.tensor(values)
    result = frontend.vmap(lambda item: item)(tensor)
    assert tuple(result.shape) == values.shape
    np.testing.assert_array_equal(result.numpy(), values)
    scalars = frontend.vmap(lambda item: item.sum())(tensor)
    assert tuple(scalars.shape) == (3,)
    np.testing.assert_array_equal(scalars.numpy(), values.sum(axis=(1, 2)))


def test_manifest_describes_final_bindings_without_changing_native_functions(frontend):
    from jittor.compat.torch.api_manifest import API_PATHS, UNIMPLEMENTED_PATHS, resolve
    from jittor.compat.torch.fidelity import fidelity_of, register_fidelity, Fidelity
    for namespace, names in [*API_PATHS.items(), *UNIMPLEMENTED_PATHS.items()]:
        owner = resolve(frontend, namespace)
        if owner is None:
            continue
        for name in names:
            implementation = getattr(owner, name, None)
            if isinstance(implementation, property):
                implementation = implementation.fget
            if callable(implementation):
                assert fidelity_of(namespace + "." + name).implementation is implementation
    source = jt.nn.functional.linear
    before = vars(source).copy()
    from jittor.compat.transaction import runtime_hook, release_runtime_hooks
    with runtime_hook("test-native-fidelity"):
        register_fidelity("torch.test_native_metadata", source, Fidelity.APPROXIMATE, "test")
        assert vars(source) == before
    release_runtime_hooks("test-native-fidelity")
