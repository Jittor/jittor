"""NN owner routing and graph isolation without a native build."""
import ast
from contextlib import nullcontext
from collections.abc import MutableMapping
import importlib.util
from pathlib import Path
import sys
import types
import unittest
import weakref
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[3] / "compat/torch"


class NativeVar:
    dtype = "float32"


class Tensor(NativeVar):
    pass


class Parameter(Tensor):
    def __init__(self, source=None, requires_grad=True):
        self.source, self.requires_grad = source, requires_grad


class NativeModule:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    def _var_roles(self):
        for name, value in vars(self).items():
            if isinstance(value, NativeVar):
                role = "buffer" if name in vars(self).get("_buffer_names", ()) else "parameter"
                yield name, value, role


class SpatialConv1d(NativeModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        if kernel_size == 0:
            raise ValueError('native constructor rejection')
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride = (kernel_size, 1), (stride, 1)
        self.padding = padding if isinstance(padding, str) else (padding, 0)
        self.dilation = (dilation, 1)
        self.groups, self.padding_mode = groups, padding_mode
        self.weight = object()
        self._conv = [types.SimpleNamespace(kernel_size=self.kernel_size,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=groups, padding_mode=padding_mode,
            _reversed_padding_repeated_twice=('original', 'cached'))]

    def execute(self, value):
        inner = self._conv[0]
        return (value, self.weight, inner.kernel_size, inner.stride, inner.padding,
                inner.dilation, inner.groups, inner.padding_mode,
                inner._reversed_padding_repeated_twice)


class TestNNFrontendOwners(unittest.TestCase):
    def setUp(self):
        replaced = patch.dict(sys.modules)
        replaced.start()
        self.addCleanup(replaced.stop)
        # Exercise the real native child-view descriptor without importing
        # the tensor runtime; the minimal stand-in must expose that protocol.
        native_source = SOURCE.parents[1] / "python/jittor/_core/module.py"
        tree = ast.parse(native_source.read_text())
        children = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == "_ModuleChildren")
        module = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "Module")
        descriptor = next(node for node in module.body
                          if isinstance(node, ast.FunctionDef) and node.name == "_modules")
        namespace = {"Module": NativeModule, "MutableMapping": MutableMapping, "weakref": weakref}
        exec(compile(ast.Module(body=[children, descriptor], type_ignores=[]),
                     str(native_source), "exec"), namespace)
        NativeModule._modules = namespace["_modules"]
        self.addCleanup(delattr, NativeModule, "_modules")
        package = types.ModuleType("_nn_owner_probe")
        package.__path__ = [str(SOURCE)]
        sys.modules[package.__name__] = package
        dtypes = types.ModuleType("jittor._core.dtypes")
        dtypes.dtype_name = str
        sys.modules[dtypes.__name__] = dtypes
        frontend = types.ModuleType("_nn_owner_probe.frontend")
        frontend.tensor_frontend = lambda tensor_type, **scope: nullcontext()
        frontend._default_tensor_dtype = lambda backend: "float32"
        frontend.make_parameter_type = lambda backend, tensor_type: Parameter
        lookup = next(node for node in ast.parse((SOURCE / "frontend.py").read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == "first_tensor")
        lookup_namespace = {}
        exec(compile(ast.Module(body=[lookup], type_ignores=[]), "frontend.py", "exec"),
             lookup_namespace)
        frontend.first_tensor = lookup_namespace["first_tensor"]
        sys.modules[frontend.__name__] = frontend
        self.frontend = self.load("nn_frontend")
        self.containers = sys.modules["_nn_owner_probe.parameter_containers"]
        self.backend = types.SimpleNamespace(
            Var=NativeVar, nn=types.SimpleNamespace(Module=NativeModule, Parameter=Parameter))
        self.owner = self.frontend.NNFrontendOwner(self.backend, Tensor)

    def load(self, name):
        spec = importlib.util.spec_from_file_location("_nn_owner_probe." + name, SOURCE / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_layer_initialization_preserves_native_types_and_parameter_aliases(self):
        class Layer(NativeModule):
            def __init__(self):
                self.weight = self.alias = Tensor()
                self.buffer = Tensor()
                self._buffer_names = {"buffer"}

            def execute(self, value):
                return value
        original = dict(vars(Layer))
        adapted = self.owner.adapt_class(Layer)
        self.assertIs(self.owner.adapt_class(Layer), adapted)
        self.assertIsInstance(vars(adapted)["__init__"], self.frontend.LayerInitializer)
        module = adapted()
        self.assertIsInstance(module.weight, Parameter)
        self.assertIs(module.weight, module.alias)
        self.assertNotIsInstance(module.buffer, Parameter)
        self.assertNotIn("_native_parameter_construction", vars(module))
        self.assertEqual(original, dict(vars(Layer)))
        self.assertIs(self.owner.Module.__setattr__, self.frontend.module_setattr)
        self.assertIs(self.owner.Module.__call__, self.frontend.module_call)
        marker = object()
        self.assertIs(module(marker), marker)
        class Derived(adapted):
            def __init__(self):
                super().__init__()
                self.tag = "derived"
        self.assertEqual(Derived().tag, "derived")

    def test_conv1d_public_axes_and_original_native_math_owner(self):
        self.backend.nn.Conv1d = SpatialConv1d
        before = dict(vars(SpatialConv1d))
        adapted = self.owner.adapt_class(SpatialConv1d)
        self.assertIs(adapted.execute, self.frontend._conv1d_execute)
        for size in (4, (4,), [4]):
            layer = adapted(2, 2, size, stride=[2], padding=(3,), dilation=[2])
            self.assertEqual((layer.kernel_size, layer.stride, layer.padding, layer.dilation,
                              layer.output_padding, layer.transposed),
                             ((4,), (2,), (3,), (2,), (0,), False))
            self.assertEqual(layer._conv[0].kernel_size, (4, 1))
            layer.stride, layer.padding, layer.dilation = (3,), [1], (2,)
            layer.groups, layer.padding_mode = 2, "reflect"
            layer.kernel_size = (9,)
            weight = layer.weight
            self.assertEqual(layer.execute(None), (None, weight, (4, 1), (3, 1),
                (1, 0), (2, 1), 2, "reflect", ("original", "cached")))
        for padding in ("same", "valid"):
            self.assertEqual(adapted(2, 2, [4], padding=padding).execute(None)[4], padding)
        self.assertEqual(dict(vars(SpatialConv1d)), before)
        self.assertEqual(SpatialConv1d(2, 2, 4).kernel_size, (4, 1))
        for name in ("kernel_size", "stride", "padding", "dilation"):
            values = dict(kernel_size=4)
            values[name] = [1, 2]
            with self.assertRaises(ValueError):
                adapted(2, 2, **values)
        with self.assertRaises(TypeError):
            adapted(2, 2, 4, kernel_size=4)

    def test_external_and_global_children_keep_source_objects_unchanged(self):
        class Child(NativeModule):
            def __init__(self):
                self.weight = Tensor()
        Child.__module__ = "jittor.nn.modules.fake"
        shared = Child()
        original_view = shared._modules
        external = [Tensor()]
        class Parent(NativeModule):
            def __init__(self, supplied):
                self.supplied = supplied
                self.child = shared
                self.alias = shared.weight
        module = self.owner.adapt_class(Parent)(external)
        self.assertIs(module.supplied, external)
        self.assertIs(type(shared), Child)
        self.assertIsNot(module.child, shared)
        self.assertIs(module.child.weight, shared.weight)
        self.assertIs(module.alias, shared.weight)
        self.assertIs(type(shared.weight), Tensor)
        self.assertIs(type(external[0]), Tensor)
        self.assertIs(original_view._owner(), shared)
        self.assertIs(module.child._modules._owner(), module.child)

    def test_embedding_freeze_and_initializer_failure_restore_construction_scope(self):
        class Embedding(NativeModule):
            def __init__(self, _freeze=False):
                self.weight = Tensor()
        frozen = self.owner.adapt_class(Embedding)(_freeze=True)
        self.assertFalse(frozen.weight.requires_grad)
        class Broken(NativeModule):
            def __init__(self):
                raise ValueError("constructor failure")
        adapted = self.owner.adapt_class(Broken)
        module = object.__new__(adapted)
        with self.assertRaisesRegex(ValueError, "constructor failure"):
            adapted.__init__(module)
        self.assertNotIn("_native_parameter_construction", vars(module))

    def test_parameter_containers_use_stable_methods_and_do_not_mutate_tensor(self):
        list_type, dict_type = self.containers.make_parameter_containers(
            self.owner.Module, Parameter, NativeVar)
        source = Tensor()
        parameter = Parameter(source)
        values = list_type([source, parameter])
        self.assertIs(type(source), Tensor)
        self.assertIsInstance(values[0], Parameter)
        self.assertIs(values[1], parameter)
        self.assertIs(list_type.append, self.containers.ParameterListAdapter.append)
        self.assertIs(dict_type.update, self.containers.ParameterDictAdapter.update)
        values.append(source)
        self.assertIs(values[1:][0], parameter)
        setattr(values, "0", parameter)
        self.assertIs(values[0], parameter)
        mapping = dict_type({"weight": source, "alias": parameter})
        self.assertIsInstance(mapping["weight"], Parameter)
        mapping.weight = parameter
        self.assertIs(mapping["weight"], parameter)
        self.assertIs(mapping.copy()["alias"], parameter)
        with self.assertRaises(TypeError):
            values.extend(source)

    def test_factories_have_no_nested_behavior_definitions(self):
        for filename in ("nn_frontend.py", "parameter_containers.py", "nn_adoption.py"):
            tree = ast.parse((SOURCE / filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.assertFalse(any(isinstance(child, (ast.FunctionDef, ast.ClassDef))
                                         and child is not node for child in ast.walk(node)),
                                     (filename, node.name))

    def test_module_input_lookup_handles_nested_keywords_and_cycles(self):
        value = Tensor()
        cyclic = []
        cyclic.append(cyclic)
        with patch.object(Tensor, "_frontend_backend", self.backend, create=True):
            self.assertIs(self.frontend._first_tensor(self.owner, (cyclic,), {"inputs": [value]}), value)
            self.assertIsNone(self.frontend._first_tensor(self.owner, (cyclic,), {"count": 2}))

    def test_input_lookup_preserves_positional_then_keyword_order(self):
        first, second = Tensor(), Tensor()
        with patch.object(Tensor, "_frontend_backend", self.backend, create=True):
            self.assertIs(self.frontend._first_tensor(self.owner, ([first], second), {"x": second}), first)
            self.assertIs(self.frontend._first_tensor(self.owner, (), {"a": {"x": first}, "b": second}), first)

    def test_input_lookup_requires_tensor_type_backend_and_its_var_identity(self):
        value = Tensor()
        self.assertIsNone(self.frontend._first_tensor(self.owner, (value,), {}))
        class ForeignVar:
            pass
        class ForeignTensor(ForeignVar):
            _frontend_backend = types.SimpleNamespace(Var=ForeignVar)
        foreign = ForeignTensor()
        owner = types.SimpleNamespace(backend=self.backend, tensor_type=ForeignTensor)
        self.assertIs(self.frontend._first_tensor(owner, ([value, foreign],), {}), foreign)
        self.assertIsNone(self.frontend._first_tensor(owner, (value,), {}))

    def test_module_call_passes_input_placement_to_native_scratch_scope(self):
        value = Tensor()
        class Echo(NativeModule):
            def execute(self, *, batch):
                return batch["x"][0]
        module = self.owner.adapt_class(Echo)()
        with patch.object(Tensor, "_frontend_backend", self.backend, create=True):
            with patch.object(self.frontend, "tensor_frontend", return_value=nullcontext()) as scope:
                self.assertIs(module(batch={"x": [value]}), value)
        scope.assert_called_once_with(Tensor, like=value, default_placement=False)


if __name__ == "__main__":
    unittest.main()
