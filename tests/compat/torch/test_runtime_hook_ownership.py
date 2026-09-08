"""Standard-library probes: no Jittor import, compilation, or accelerator."""
import importlib
import ast
import pathlib
import sys
import tempfile
import threading
import types
import unittest
import uuid
from unittest import mock


_PACKAGE = "_jittor_runtime_hook_tests"
package = types.ModuleType(_PACKAGE)
package.__path__ = [str(pathlib.Path(__file__).resolve().parents[3] / "compat")]
sys.modules[_PACKAGE] = package
tx = importlib.import_module(_PACKAGE + ".transaction")
backend = importlib.import_module(_PACKAGE + ".external_backend")
patcher = importlib.import_module(_PACKAGE + ".module_patcher")


class RuntimeHookOwnership(unittest.TestCase):
    def source_function(self, relative, name, namespace):
        source = pathlib.Path(package.__path__[0]) / relative
        tree = ast.parse(source.read_text())
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        return namespace[name]

    def name(self):
        return "_owned_" + uuid.uuid4().hex

    def assert_lock_available(self, lock):
        acquired = []
        def attempt():
            ok = lock.acquire(timeout=1)
            acquired.append(ok)
            if ok:
                lock.release()
        thread = threading.Thread(target=attempt)
        thread.start()
        thread.join(2)
        self.assertEqual(acquired, [True])

    def test_optional_vllm_discovery_is_named_and_absence_is_not_failure(self):
        context = object()
        install = self.source_function(
            "torch/__init__.py", "_install_optional_vllm",
            dict(__package__=_PACKAGE + ".torch", active_transaction=lambda value: None))
        unrelated = types.SimpleNamespace(name="unrelated", value="foreign:register")
        unrelated.load = mock.Mock(side_effect=AssertionError("unselected plugin loaded"))
        original_meta = list(sys.meta_path)
        original_finder = patcher._FINDER
        original_registry = {key: list(value) for key, value in patcher._REGISTRY.items()}
        original_loaded = set(patcher._ENTRY_POINTS_LOADED)
        try:
            patcher._REGISTRY.clear()
            patcher._ENTRY_POINTS_LOADED.clear()
            with mock.patch.object(patcher, "_entry_points", return_value=[unrelated]):
                report = install(context)
            self.assertTrue(report.ok)
            self.assertTrue(any(item.name == "jittor_vllm" and item.status == "unavailable"
                                for item in report.results))
            unrelated.load.assert_not_called()
        finally:
            sys.meta_path[:] = original_meta
            patcher._FINDER = original_finder
            patcher._REGISTRY.clear()
            patcher._REGISTRY.update(original_registry)
            patcher._ENTRY_POINTS_LOADED.clear()
            patcher._ENTRY_POINTS_LOADED.update(original_loaded)

    def test_failed_hook_restores_descriptor_module_and_lock(self):
        class Base:
            value = staticmethod(lambda: 1)
        original = vars(Base)["value"]
        modules = {}
        with self.assertRaisesRegex(ValueError, "failure"):
            with tx.runtime_hook(self.name()) as hook:
                hook.mutate_attr(Base, "value", staticmethod(lambda: 2))
                hook.replace_module(modules, "owned", types.ModuleType("owned"))
                raise ValueError("failure")
        self.assertIs(vars(Base)["value"], original)
        self.assertEqual(modules, {})
        self.assert_lock_available(tx.InstallTransaction._lock)

    def test_active_hook_release_preserves_foreign_replacement_and_restores_others(self):
        owner = self.name()
        modules = {}
        target = types.SimpleNamespace(value=object())
        original = target.value
        with tx.runtime_hook(owner) as hook:
            hook.mutate_attr(target, "value", object())
            hook.replace_module(modules, "owned", types.ModuleType("owned"))
        foreign = types.ModuleType("foreign")
        modules["owned"] = foreign
        with self.assertRaises(tx.TransactionConflict):
            tx.release_runtime_hooks(owner)
        self.assertIs(modules["owned"], foreign)
        self.assertIs(target.value, original)
        self.assert_lock_available(tx.InstallTransaction._lock)

    def test_child_commit_is_reverted_by_parent_and_retry_can_start(self):
        parent = tx.InstallTransaction("outer")
        child = tx.InstallTransaction("inner")
        markers = {}
        child.record(markers, "complete", tx._MISSING, True)
        markers["complete"] = True
        parent.adopt(child)
        child.commit()
        parent.rollback()
        self.assertEqual(markers, {})
        self.assertEqual(parent.retry().state, "open")

    def test_object_diff_does_not_undo_explicit_or_child_hook_slots_twice(self):
        class Namespace:
            def __init__(self):
                self.version = object()
            def _binding_state(self, name):
                return name in vars(self), vars(self).get(name), False
            def _restore_binding(self, name, state):
                if state[0]:
                    setattr(self, name, state[1])
                elif name in vars(self):
                    delattr(self, name)
        for through_hook in (False, True):
            target = Namespace()
            before = vars(target).copy()
            parent = tx.InstallTransaction("deduplicated-parent")
            if through_hook:
                with tx.runtime_hook((self.name(), through_hook), parent_transaction=parent) as hook:
                    hook.mutate_attr(target, "version", object())
            else:
                parent.mutate_attr(target, "version", object())
            parent.record_object_diffs(target, before)
            parent.rollback()
            self.assertIs(target.version, before["version"])

    def test_actual_install_adopts_complete_markers_classes_and_namespace(self):
        target = types.ModuleType(self.name())
        target.Var = type("Tensor", (), {})
        target.runtime = types.SimpleNamespace(service_state=lambda name: {})
        published_name = "torch." + self.name()
        calls = []
        class Context:
            COMPLETE_ATTR = "_complete"
            def __init__(self):
                self.state, self.markers = {}, {}
                self.registry = types.SimpleNamespace(_published={})
                self.native_backend = self.target_namespace = target
            @property
            def complete(self):
                return getattr(target, self.COMPLETE_ATTR, False)
            @classmethod
            def for_module(cls, module, strict=True):
                return context
            def run_required(self, step, installer):
                installer(self)
                self.markers[step] = "complete"
            def mark_complete(self):
                setattr(target, self.COMPLETE_ATTR, True)
        context = Context()
        def install_step(context):
            calls.append(1)
            target.Var.installed_method = object()
            target.installed_attribute = object()
            sys.modules[published_name] = target
        tensor_state = types.ModuleType(_PACKAGE + ".torch.tensor_state")
        tensor_state.compatibility_owner = lambda module: module
        tensor_state.bind_tensor_state = lambda *args, **kwargs: {}
        tensor_state.snapshot_tensor_state = lambda state: {}
        tensor_state.record_tensor_state_changes = lambda *args: None
        namespace = dict(__package__=_PACKAGE + ".torch", sys=sys,
                         InstallTransaction=tx.InstallTransaction, InstallContext=Context,
                         _MISSING=tx._MISSING, _NAMESPACE_TRANSACTION="_namespace",
                         _REQUIRED_STEPS=(("one", install_step),), _OPTIONAL_STEPS=(),
                         EXPECTED=(Exception,), swallowed=lambda *args: None,
                         _torch_namespace_snapshot=lambda: {
                             name: value for name, value in sys.modules.items()
                             if name == "torch" or name.startswith("torch.")})
        for name in ("_restore_namespace", "_abandon", "install"):
            self.source_function("torch/__init__.py", name, namespace)
        with mock.patch.dict(sys.modules, {tensor_state.__name__: tensor_state}):
            for _ in range(2):
                parent = tx.InstallTransaction("activation")
                namespace["install"](target, parent_transaction=parent)
                self.assertTrue(context.complete)
                parent.rollback()
                self.assertFalse(context.complete)
                self.assertEqual(context.markers, {})
                self.assertNotIn("_tensor_state", context.state)
                self.assertNotIn(published_name, sys.modules)
                self.assertFalse(hasattr(target.Var, "installed_method"))
                self.assertFalse(hasattr(target, "installed_attribute"))
        self.assertEqual(len(calls), 2)
        self.assert_lock_available(tx.InstallTransaction._lock)

    def test_nested_hook_is_reverted_when_parent_fails(self):
        modules = {}
        with self.assertRaises(ValueError):
            with tx.runtime_hook(self.name()):
                with tx.runtime_hook(self.name()) as inner:
                    inner.replace_module(modules, "owned", types.ModuleType("owned"))
                raise ValueError("later activation failed")
        self.assertEqual(modules, {})

    def test_fidelity_registration_uses_current_scope_without_native_import(self):
        name = _PACKAGE + ".torch.fidelity"
        spec = importlib.util.spec_from_file_location(name, pathlib.Path(package.__path__[0]) / "torch/fidelity.py")
        fidelity = importlib.util.module_from_spec(spec)
        sys.modules[name] = fidelity
        spec.loader.exec_module(fidelity)
        function = lambda: None
        transaction = tx.InstallTransaction("installer")
        transaction.acquire()
        try:
            fidelity.register_fidelity("torch.probe", function, "approximate", "test")
            self.assertIn("torch.probe", fidelity._REGISTRY)
            transaction.rollback()
        finally:
            transaction.release()
        self.assertNotIn("torch.probe", fidelity._REGISTRY)
        self.assertFalse(hasattr(function, "__torch_fidelity__"))
        self.assertIsNone(tx.current_transaction())

    def test_deferred_class_patch_has_runtime_lifetime(self):
        name = self.name()
        module = types.ModuleType(name)
        class Layer:
            value = staticmethod(lambda: 1)
        original = vars(Layer)["value"]
        module.Layer = Layer
        def callback(module):
            patcher.patch_method(module.Layer, "value", staticmethod(lambda: 2))
        patcher.register_module_patch(name, callback)
        patcher._apply_module_patches(module)
        self.assertEqual(Layer.value(), 2)
        tx.release_runtime_hooks(("module_patch", name, callback))
        self.assertIs(vars(Layer)["value"], original)

    def test_source_miss_preserves_concurrent_module_and_path_insertions(self):
        name, foreign_name = self.name(), self.name()
        foreign = types.ModuleType(foreign_name)
        extra_path = self.name()
        with tempfile.TemporaryDirectory() as root:
            resolver = backend.ExternalBackend(backend.ExternalBackendSpec("probe", ("run",)))
            def load(source):
                resolver._add_source_to_sys_path(pathlib.Path(source))
                module = types.ModuleType(name)
                module.__file__ = str(pathlib.Path(source) / "probe.py")
                backend.publish_source_module(name, module)
                thread = threading.Thread(target=lambda: (sys.modules.__setitem__(foreign_name, foreign),
                                                          sys.path.insert(0, extra_path)))
                thread.start()
                thread.join()
                return None
            resolver.load_source_root = load
            try:
                self.assertEqual(resolver._load_candidate(root, None), (None, None))
                self.assertNotIn(name, sys.modules)
                self.assertIs(sys.modules[foreign_name], foreign)
                self.assertIn(extra_path, sys.path)
            finally:
                sys.modules.pop(name, None)
                sys.modules.pop(foreign_name, None)
                sys.path.remove(extra_path)

    def test_source_module_conflict_is_hard_and_releases_lock(self):
        name = self.name()
        foreign = types.ModuleType(name)
        with tempfile.TemporaryDirectory() as root:
            def miss(module, key):
                sys.modules[name] = foreign
                return "unsupported"
            resolver = backend.ExternalBackend(backend.ExternalBackendSpec("probe", ("run",)),
                                               capability_miss=miss)
            def load(source):
                module = types.ModuleType(name)
                module.__file__ = str(pathlib.Path(source) / "probe.py")
                backend.publish_source_module(name, module)
                return module
            resolver.load_source_root = load
            try:
                with self.assertRaises(tx.TransactionConflict):
                    resolver._load_candidate(root, "capability")
                self.assertIs(sys.modules[name], foreign)
                self.assert_lock_available(backend._SOURCE_IMPORT_LOCK)
            finally:
                sys.modules.pop(name, None)

    def test_regular_source_import_restores_displaced_installed_module(self):
        name = self.name()
        original = types.ModuleType(name)
        original.__file__ = "/installed/" + name + ".py"
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / name
            path.mkdir()
            (path / "__init__.py").write_text("def run(): return 1\n")
            resolver = backend.ExternalBackend(
                backend.ExternalBackendSpec("probe", ("run",), module_names=(name,)),
                capability_miss=lambda module, key: "unsupported")
            with mock.patch.dict(sys.modules, {name: original}):
                result, miss = resolver._load_candidate(root, "probe")
                self.assertEqual(miss, "unsupported")
                self.assertEqual(result.run(), 1)
                self.assertIs(sys.modules[name], original)

    def test_untracked_same_source_publication_is_preserved_and_fails_closed(self):
        name = self.name()
        with tempfile.TemporaryDirectory() as root:
            foreign = types.ModuleType(name)
            foreign.__file__ = str(pathlib.Path(root) / "foreign.py")
            resolver = backend.ExternalBackend(backend.ExternalBackendSpec("probe", ("run",)))
            def load(source):
                thread = threading.Thread(target=lambda: sys.modules.__setitem__(name, foreign))
                thread.start()
                thread.join()
                return None
            resolver.load_source_root = load
            try:
                with self.assertRaisesRegex(tx.TransactionConflict, "untracked source publication"):
                    resolver._load_candidate(root, None)
                self.assertIs(sys.modules[name], foreign)
            finally:
                sys.modules.pop(name, None)

    def test_extension_publication_rolls_back_failed_loader(self):
        name = self.name()
        with tempfile.TemporaryDirectory() as root:
            builder = types.ModuleType("jittor.compat.shim.cpp_extension")
            builder.cfg = lambda: {"ext_suffix": ".py"}
            builder.build = lambda **kwargs: pathlib.Path(kwargs["output_path"]).write_text("raise ValueError('loader failure')\n")
            shim = types.ModuleType("jittor.compat.shim")
            shim.cpp_extension = builder
            old = types.ModuleType(name)
            old.__file__ = str(pathlib.Path(root) / (name + ".py"))
            namespace = dict(importlib=importlib, os=__import__("os"), sys=sys,
                             runtime_hook=tx.runtime_hook, TransactionConflict=tx.TransactionConflict,
                             _extension_import_name=lambda value, identity: value)
            load = self.source_function("shim/cpp_extension/torch_utils.py", "load", namespace)
            with mock.patch.dict(sys.modules, {"jittor.compat.shim": shim, name: old}):
                with self.assertRaisesRegex(ValueError, "loader failure"):
                    load(name, [], build_directory=root, force=True)
                self.assertIs(sys.modules[name], old)
            self.assert_lock_available(tx.InstallTransaction._lock)




if __name__ == "__main__":
    unittest.main()
