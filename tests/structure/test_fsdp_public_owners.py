"""FSDP publication/forward routing contracts without native compilation."""
import ast
import enum
import importlib.util
from pathlib import Path
import pickle
import sys
import types
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[2] / "compat/fsdp2"


class _Owner(types.ModuleType):
    def __getattr__(self, name):
        value = type(name, (), {})
        setattr(self, name, value)
        return value


class _Registry:
    def __init__(self):
        self.modules = {}

    def publish(self, name, module):
        self.modules[name] = module
        if "." in name:
            parent, _, child = name.rpartition(".")
            if parent in self.modules:
                setattr(self.modules[parent], child, module)
        return module

    def ensure(self, name):
        if name not in self.modules:
            self.publish(name, types.ModuleType(name))
        return self.modules[name]


class TestFSDPPublicOwners(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        for name in ("_fsdp_owner_probe", "_fsdp_owner_probe.fsdp2", "_fsdp_owner_probe.torch"):
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        self.owners = {}
        for name in ("api", "common", "compat_types", "config", "dtensor", "grad_sync", "optimizer", "shard"):
            module = _Owner("_fsdp_owner_probe.fsdp2." + name)
            sys.modules[module.__name__] = module
            self.owners[name] = module
        self.records = {}
        fidelity = types.ModuleType("_fsdp_owner_probe.torch.fidelity")
        fidelity.Fidelity = enum.Enum("Fidelity", "EXACT APPROXIMATE UNIMPLEMENTED")
        fidelity.register_fidelity = lambda name, obj, level, detail: self.records.update(
            {name: (obj, level, detail)})
        sys.modules[fidelity.__name__] = fidelity
        self.helpers = self.load("public_helpers")
        self.installer = self.load("installer")

    def load(self, name):
        spec = importlib.util.spec_from_file_location("_fsdp_owner_probe.fsdp2." + name, SOURCE / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_two_installations_publish_same_real_function_and_class_owners(self):
        graphs = []
        for _ in range(2):
            registry = _Registry()
            self.installer._registry_for = lambda *args: registry
            dist = types.ModuleType("torch.distributed")
            self.installer.install_with_registry(dist, registry=registry)
            graphs.append(registry.modules)
            self.installer.install_with_registry(dist, registry=registry)
        for path, name in (
            ("torch.distributed.fsdp.wrap", "enable_wrap"),
            ("torch.distributed.fsdp.wrap", "lambda_auto_wrap_policy"),
            ("torch.distributed.tensor.parallel", "ColwiseParallel"),
            ("torch.distributed._functional_collectives", "AsyncCollectiveTensor"),
            ("torch.distributed.algorithms._checkpoint.checkpoint_wrapper", "CheckpointImpl"),
        ):
            value = getattr(graphs[0][path], name)
            self.assertIs(value, getattr(graphs[1][path], name))
            self.assertIs(value, getattr(self.helpers, name))
            self.assertIs(pickle.loads(pickle.dumps(value)), value)
        self.assertIn("no activation offload", self.records[
            "torch.distributed.algorithms._checkpoint.checkpoint_wrapper.offload_wrapper"][2])

    def test_wrap_and_collective_routing_preserve_original_owner_arguments(self):
        seen = []
        self.owners["api"].fully_shard = lambda module, **kwargs: seen.append((module, kwargs)) or module
        module, tensor = object(), object()
        self.assertIs(self.helpers.wrap(module, mesh="mesh", ignored=True), module)
        self.assertEqual(seen, [(module, {"mesh": "mesh"})])
        common = self.owners["common"]
        common._in_true_distributed = lambda: False
        common._all_gather_shards = lambda tensor: ("gather", tensor)
        common._reduce_scatter_padded = lambda tensor: ("scatter", tensor)
        self.assertIs(self.helpers.all_gather(tensor), tensor)
        self.assertIs(self.helpers.reduce_scatter(tensor), tensor)
        common._in_true_distributed = lambda: True
        self.assertEqual(self.helpers.all_gather(tensor), ("gather", tensor))
        self.assertEqual(self.helpers.reduce_scatter(tensor), ("scatter", tensor))
        self.assertIs(self.helpers.AsyncCollectiveTensor(tensor).wait(), tensor)

    def test_execute_method_has_stable_owner_and_restores_depth_after_failure(self):
        tree = ast.parse((SOURCE / "shard.py").read_text())
        names = {"_wrapped_execute", "_install_true_fsdp_execute", "_execute_with_true_fsdp"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        events = []
        scope = {"types": types, "_EXECUTE_DEPTH_ATTR": "_depth",
                 "_unshard_module_params": lambda module: events.append("unshard"),
                 "_reshard_module_params": lambda module: events.append("reshard")}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "shard.py", "exec"), scope)
        class Module:
            def execute(self, value):
                events.append(("execute", value))
                if value == "failure":
                    raise ValueError(value)
                return value
        module = Module()
        module._fsdp_state = types.SimpleNamespace(true_fsdp_initialized=True, reshard_after_forward=True)
        install = scope["_install_true_fsdp_execute"]
        install(module)
        original, bound = module._fsdp_orig_execute, module.execute
        install(module)
        self.assertIs(module.execute, bound)
        self.assertIs(module.execute.__func__, scope["_wrapped_execute"])
        self.assertIs(module._fsdp_orig_execute, original)
        self.assertEqual(module.execute(3), 3)
        with self.assertRaisesRegex(ValueError, "failure"):
            module.execute("failure")
        self.assertEqual(module._fsdp_state._depth, 0)
        self.assertEqual(events, ["unshard", ("execute", 3), "reshard",
                                  "unshard", ("execute", "failure"), "reshard"])

    def test_installer_has_no_public_lambda_or_nested_definitions(self):
        tree = ast.parse((SOURCE / "installer.py").read_text())
        self.assertFalse(any(isinstance(node, ast.Lambda) for node in ast.walk(tree)))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self.assertFalse(any(isinstance(child, (ast.FunctionDef, ast.ClassDef))
                                     and child is not node for child in ast.walk(node)))


if __name__ == "__main__":
    unittest.main()
