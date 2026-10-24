"""Execute source-owned frontend metadata contracts without importing Jittor."""
import ast
import pickle
from pathlib import Path
import sys
import types
from types import MappingProxyType

import pytest


ROOT = Path(__file__).resolve().parents[3]
FRONTEND = ROOT / "compat/torch/frontend.py"


class _UnpicklableNativeShape:
    def __init__(self, dimensions):
        self.dimensions = tuple(dimensions)

    def __iter__(self):
        return iter(self.dimensions)

    def __reduce__(self):
        raise TypeError("native shape metadata cannot be pickled")


class _NativeVar:
    __slots__ = ("dimensions", "shape_reads")

    @property
    def shape(self):
        self.shape_reads += 1
        return self.dimensions


class _OwnedRoot:
    __slots__ = ("dimensions", "shape_reads")

    @property
    def shape(self):
        self.shape_reads += 1
        return self.dimensions


def _backend(factory=None):
    core = types.SimpleNamespace(Var=_NativeVar)
    if factory is not None:
        core._create_owned_tensor_base_type = factory
    return types.SimpleNamespace(Var=_NativeVar, core=core)


def _factory(monkeypatch):
    package = "_frontend_shape_host_fixture"
    package_module = types.ModuleType(package)
    package_module.__path__ = []
    nested = types.ModuleType(package + ".nested")
    shape = next(node for node in ast.parse((ROOT / "compat/torch/nested.py").read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name == "_TorchSize")
    exec(compile(ast.Module(body=[shape], type_ignores=[]), str(ROOT / "compat/torch/nested.py"), "exec"),
         nested.__dict__)
    properties = types.ModuleType(package + ".tensor_object_state")
    properties.tensor_object_properties = lambda: {}
    monkeypatch.setitem(sys.modules, package, package_module)
    monkeypatch.setitem(sys.modules, nested.__name__, nested)
    monkeypatch.setitem(sys.modules, properties.__name__, properties)
    # Inert metadata-only stand-ins: never import or execute the real backend.
    native_package = types.ModuleType("jittor")
    native_package.__path__ = []
    core_package = types.ModuleType("jittor._core")
    core_package.__path__ = []
    protocol = types.ModuleType("jittor._core.carrier_protocol")
    legacy = MappingProxyType({name: object() for name in
                              ("clone", "copy", "detach", "item",
                               "inplace_version", "to_device")})
    protocol.capture_native_api = lambda root, backend: MappingProxyType(
        dict(legacy, shape=vars(root)["shape"]))
    protocol.bind_tensor_protocol = lambda tensor, backend: setattr(
        tensor, "_frontend_legacy_native_api", legacy)
    monkeypatch.setitem(sys.modules, "jittor", native_package)
    monkeypatch.setitem(sys.modules, "jittor._core", core_package)
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)
    functions = [node for node in ast.parse(FRONTEND.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == "make_tensor_type"]
    assert len(functions) == 1
    namespace = {"__package__": package, "__name__": package + ".frontend", "_TensorMeta": type,
                 "clone": lambda value: value, "_frontend_precision_policy": lambda cls: None}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(FRONTEND), "exec"), namespace)
    return namespace["make_tensor_type"], nested._TorchSize


def _value(tensor_type, dimensions):
    value = object.__new__(tensor_type)
    value.dimensions = _UnpicklableNativeShape(dimensions)
    value.shape_reads = 0
    return value


@pytest.mark.parametrize("dimensions", ((), (1, 2), (0, 3), (2, 3, 4)))
def test_tensor_shape_is_existing_pickleable_size_not_native_descriptor(monkeypatch, dimensions):
    make_tensor_type, size_type = _factory(monkeypatch)
    descriptor = vars(_NativeVar)["shape"]
    backend = _backend()
    tensor_type = make_tensor_type(backend)
    value = _value(tensor_type, dimensions)
    assert isinstance(value.shape, size_type), "public Tensor.shape leaked native shape metadata"
    assert value.shape == tuple(dimensions)
    assert pickle.loads(pickle.dumps(value.shape)) == tuple(dimensions)
    assert vars(_NativeVar)["shape"] is descriptor
    native = _value(_NativeVar, dimensions)
    assert isinstance(native.shape, _UnpicklableNativeShape)
    with pytest.raises(TypeError, match="native shape"):
        pickle.dumps(native.shape)


def test_public_shape_is_readonly_and_does_not_change_native_dimensions(monkeypatch):
    make_tensor_type, size_type = _factory(monkeypatch)
    tensor_type = make_tensor_type(_backend())
    assert isinstance(vars(tensor_type).get("shape"), property)
    value = _value(tensor_type, (1, 2))
    before = value.dimensions
    with pytest.raises(AttributeError):
        value.shape = (5,)
    assert value.dimensions is before
    assert value.shape == (1, 2)
    parameter_type = type("Parameter", (tensor_type,), {})
    assert isinstance(_value(parameter_type, (3,)).shape, size_type)


def test_shape_uses_captured_native_descriptor_even_if_backend_var_is_projected(monkeypatch):
    make_tensor_type, size_type = _factory(monkeypatch)
    backend = _backend()
    tensor_type = make_tensor_type(backend)
    # Projection/install order must not turn the getter into self.shape recursion.
    backend.Var = tensor_type
    value = _value(tensor_type, (2, 3))
    assert isinstance(value.shape, size_type)
    assert value.shape_reads == 1


def test_shape_is_live_metadata_not_a_stale_cached_size(monkeypatch):
    make_tensor_type, _ = _factory(monkeypatch)
    tensor_type = make_tensor_type(_backend())
    value = _value(tensor_type, (1, 2))
    assert value.shape == (1, 2)
    value.dimensions = _UnpicklableNativeShape((2, 1))
    assert value.shape == (2, 1)


def test_optional_owned_root_unavailable_keeps_complete_legacy_delegate_table(monkeypatch):
    make_tensor_type, _ = _factory(monkeypatch)
    calls = []

    def unavailable():
        calls.append(True)
        raise NotImplementedError("unsupported interpreter")

    before = dict(vars(_NativeVar))
    tensor = make_tensor_type(_backend(unavailable))
    assert calls == [True]
    assert tensor.__bases__ == (_NativeVar,)
    assert tensor._frontend_tensor_base is _NativeVar
    assert dict(vars(_NativeVar)) == before
    for name, delegate in tensor._frontend_legacy_native_api.items():
        assert tensor._frontend_native_api[name] is delegate
    assert tensor._frontend_native_api["shape"] is vars(_NativeVar)["shape"]
    with pytest.raises(TypeError):
        tensor._frontend_native_api["copy"] = None


def test_supported_owned_root_retains_its_real_owner(monkeypatch):
    make_tensor_type, size_type = _factory(monkeypatch)
    tensor = make_tensor_type(_backend(lambda: _OwnedRoot))
    assert tensor.__bases__ == (_OwnedRoot,)
    assert tensor._frontend_tensor_base is _OwnedRoot
    assert not issubclass(tensor, _NativeVar)
    assert tensor._frontend_native_api["shape"] is vars(_OwnedRoot)["shape"]
    assert isinstance(_value(tensor, (2, 3)).shape, size_type)


@pytest.mark.parametrize("error", (RuntimeError, ValueError))
def test_owned_root_unrelated_failures_are_not_silenced(monkeypatch, error):
    make_tensor_type, _ = _factory(monkeypatch)

    def broken():
        raise error("actual factory failure")

    with pytest.raises(error, match="actual factory failure"):
        make_tensor_type(_backend(broken))
