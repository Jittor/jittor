"""NN owner routing and graph isolation without a native build."""
import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import types
import unittest
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


class TestNNFrontendOwners(unittest.TestCase):
    def setUp(self):
        replaced = patch.dict(sys.modules)
        replaced.start()
        self.addCleanup(replaced.stop)
        package = types.ModuleType("_nn_owner_probe")
        package.__path__ = [str(SOURCE)]
        sys.modules[package.__name__] = package
        dtypes = types.ModuleType("jittor._core.dtypes")
        dtypes.dtype_name = str
        sys.modules[dtypes.__name__] = dtypes
        frontend = types.ModuleType("_nn_owner_probe.frontend")
        frontend.tensor_frontend = lambda tensor_type: nullcontext()
        frontend.make_parameter_type = lambda backend, tensor_type: Parameter
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

    def test_external_and_global_children_keep_source_objects_unchanged(self):
        class Child(NativeModule):
            def __init__(self):
                self.weight = Tensor()
        Child.__module__ = "jittor.nn.modules.fake"
        shared = Child()
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


if __name__ == "__main__":
    unittest.main()
