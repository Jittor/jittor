"""Pinned HF loading-policy contracts; fake tensors are not device evidence.

The prerequisites run before the opt-in adapter is implemented. Adapter tests
intentionally require its real entry points: absence is not a successful skip.
"""
import ast
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ADAPTERS = Path(__file__).resolve().parents[3]
ROOT = ADAPTERS.parent
sys.path.insert(0, str(ADAPTERS))
sys.path.insert(0, str(ADAPTERS / "tests"))
from _adapter_boundary import boundary_errors
from jittor_adapters._common import UnsupportedAdapterVersion

compat = types.ModuleType("_transformers_loading_host_compat")
compat.__path__ = [str(ROOT / "compat")]
sys.modules[compat.__name__] = compat
patcher = importlib.import_module(compat.__name__ + ".module_patcher")
transaction = importlib.import_module(compat.__name__ + ".transaction")
FIXTURE = Path(__file__).with_name("_hf_v590_fixture.py")
ENV = "JITTOR_TRANSFORMERS_MATERIALIZED_INIT"
GET_CONTEXT_SHA = "85673c5340632d9d83e0bfd513a728a7f2a7347dacba87d412202471a24cc0e2"
NO_INIT_SHA = "1efb0d0d1fa2d202924ca22dc92bad70566ebb1f1b10f93d422e9c4e983c92b1"


def guarded_init_type():
    path = ROOT / "compat/torch/installers/nn_init.py"
    node, = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.ClassDef) and node.name == "_GuardedInit"]
    namespace = {"_types_init": types}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_GuardedInit"]


class LoadingFixture:
    def __init__(self):
        self.events = []
        self.active = []
        self.root = types.ModuleType("transformers")
        self.root.__version__, self.root.__path__ = "5.9.0", []
        self.module = types.ModuleType("transformers.modeling_utils")
        self.module.__file__, self.module.__package__ = str(FIXTURE), "transformers"
        self.init = types.ModuleType("transformers.initialization")
        self.init.__file__ = str(FIXTURE)
        self.torch = types.ModuleType("torch")
        self.torch_init = guarded_init_type()("torch.nn.init")

        def native_normal(tensor, *args, **kwargs):
            tensor.initialized = True
            return tensor

        self.native_normal = native_normal
        self.torch_init.normal_ = native_normal
        self.torch_init._protected = {"normal_"}
        self.torch.nn = types.SimpleNamespace(init=self.torch_init)
        self.torch.Tensor, self.torch.dtype = object, object
        self.torch.float32 = "float32"
        state = self

        class Device:
            def __init__(self, kind):
                self.type, self.index = kind, None

            def __enter__(self):
                state.events.append("enter:" + self.type)
                state.active.append(self.type)
                if self.type == "meta":
                    raise AssertionError("materialized strategy must never enter meta")
                return self

            def __exit__(self, *exc):
                state.events.append("exit:" + self.type)
                state.active.pop()

        self.torch.device = Device
        self.module.__dict__.update(
            torch=self.torch, init=self.init,
            local_torch_dtype=lambda *args: self.context("dtype"),
            apply_patches=lambda: self.context("patches"),
            allow_all_hub_kernels=lambda: self.context("kernels"),
            is_deepspeed_zero3_enabled=lambda: False,
            TORCH_INIT_FUNCTIONS={"normal_": native_normal},
            TORCH_MODULES_TO_PATCH=("torch.nn.init",),
        )
        exec(compile(FIXTURE.read_text(), str(FIXTURE), "exec"), vars(self.module))
        self.base = self.module.PreTrainedModel
        self.base.is_remote_code = classmethod(lambda cls: False)
        self.init.__package__ = "transformers"
        for name in ("contextmanager", "defaultdict", "sys", "TORCH_INIT_FUNCTIONS", "TORCH_MODULES_TO_PATCH"):
            vars(self.init)[name] = vars(self.module)[name]
        no_init_node, = [node for node in ast.parse(FIXTURE.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "no_init_weights"]
        exec(compile(ast.Module(body=[no_init_node], type_ignores=[]), str(FIXTURE), "exec"), vars(self.init))
        self.init.no_tie_weights = lambda: self.context("no_tie")
        self.init.meta_device_safe_creation_ops = lambda: self.context("safe_creation")
        self.original_descriptor = vars(self.base)["get_init_context"]
        self.original_initializer = self.base.init_weights

    @contextmanager
    def context(self, label):
        self.events.append("enter:" + label)
        self.active.append(label)
        try:
            yield
        finally:
            self.events.append("exit:" + label)
            self.active.pop()

    def modules(self):
        return {"transformers": self.root, "transformers.modeling_utils": self.module,
                "transformers.initialization": self.init, "torch": self.torch,
                "torch.nn.init": self.torch_init, "jittor.compat.module_patcher": patcher}


class TestLoadingStrategyPrerequisites(unittest.TestCase):
    def setUp(self):
        self.fixture = LoadingFixture()
        self.patches = mock.patch.dict(sys.modules, self.fixture.modules())
        self.patches.start()
        self.addCleanup(self.patches.stop)

    def test_exact_upstream_function_excerpts(self):
        fixture = self.fixture
        for function, expected in ((fixture.original_descriptor.__func__, GET_CONTEXT_SHA),
                                   (fixture.init.no_init_weights, NO_INIT_SHA)):
            self.assertEqual(hashlib.sha256(inspect.getsource(function).encode()).hexdigest(), expected)

    def test_upstream_no_init_is_accepted_by_real_guard_and_restored(self):
        fixture = self.fixture
        with fixture.init.no_init_weights():
            self.assertEqual(fixture.torch_init.normal_.__name__, "empty_func")
            tensor = types.SimpleNamespace()
            fixture.torch_init.normal_(tensor)
            model = fixture.base()
            model.init_weights()
            self.assertFalse(hasattr(tensor, "initialized"))
            self.assertFalse(hasattr(model, "initialized"))
        self.assertIs(fixture.torch_init.normal_, fixture.native_normal)
        self.assertIs(fixture.base.init_weights, fixture.original_initializer)
        model.init_weights()
        fixture.torch_init.normal_(tensor)
        self.assertTrue(model.initialized and tensor.initialized)

    def test_upstream_no_init_restores_after_constructor_failure(self):
        fixture = self.fixture
        error = RuntimeError("constructor failed")
        with self.assertRaises(RuntimeError) as caught:
            with fixture.init.no_init_weights():
                raise error
        self.assertIs(caught.exception, error)
        self.assertIs(fixture.torch_init.normal_, fixture.native_normal)
        self.assertIs(fixture.base.init_weights, fixture.original_initializer)

    def test_public_patch_method_restores_raw_classmethod_descriptor(self):
        fixture = self.fixture
        replacement = classmethod(lambda cls, *args: cls)
        with transaction.runtime_hook("loading_descriptor_contract"):
            patcher.patch_method(fixture.base, "get_init_context", replacement)
        try:
            class Child(fixture.base):
                pass
            self.assertIs(Child.get_init_context(), Child)
        finally:
            transaction.release_runtime_hooks("loading_descriptor_contract")
        self.assertIs(vars(fixture.base)["get_init_context"], fixture.original_descriptor)


class TestMaterializedLoadingAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = importlib.import_module("jittor_adapters.transformers_loading")
        self.assertTrue(callable(getattr(self.adapter, "register", None)), "adapter not implemented yet")
        self.fixture = LoadingFixture()
        self.saved_registry = dict(patcher._REGISTRY)
        patcher._REGISTRY.clear()
        self.modules_patch = mock.patch.dict(sys.modules, self.fixture.modules())
        self.modules_patch.start()
        self.env_patch = mock.patch.dict(os.environ, {ENV: "1"})
        self.env_patch.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        patcher.uninstall_module_patches()
        patcher._REGISTRY.clear()
        patcher._REGISTRY.update(self.saved_registry)
        self.env_patch.stop()
        self.modules_patch.stop()

    def install(self):
        self.adapter.register(patcher.register_module_patch)
        return patcher.install_module_patches(load_entry_points=False)

    def contexts(self, **changes):
        kwargs = dict(dtype=self.fixture.torch.float32, is_quantized=False,
                      _is_ds_init_called=False, allow_all_kernels=False)
        kwargs.update(changes)
        return self.fixture.base.get_init_context(**kwargs)

    def test_disabled_registers_nothing_and_leaves_descriptor_untouched(self):
        for value in (None, "0"):
            with self.subTest(value=value), mock.patch.dict(os.environ):
                if value is None:
                    os.environ.pop(ENV, None)
                else:
                    os.environ[ENV] = value
                self.adapter.register(patcher.register_module_patch)
                self.assertEqual(patcher.registered_module_patches(), {})
                self.assertIs(vars(self.fixture.base)["get_init_context"], self.fixture.original_descriptor)

    def test_invalid_optin_values_fail_closed_before_and_after_install(self):
        for value in ("", "true", "yes", "2", " 1"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {ENV: value}):
                with self.assertRaisesRegex(ValueError, ENV):
                    self.adapter.register(patcher.register_module_patch)
                self.assertEqual(patcher.registered_module_patches(), {})
        self.install()
        for value in ("", "true", "2"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {ENV: value}):
                with self.assertRaisesRegex(ValueError, ENV):
                    self.contexts()

    def test_context_is_cpu_materialized_and_init_guard_ends_before_loading(self):
        self.install()
        fixture = self.fixture
        with ExitStack() as stack:
            for context in self.contexts():
                stack.enter_context(context)
            self.assertIn("cpu", fixture.active)
            self.assertNotIn("meta", fixture.active)
            model = fixture.base()
            model.init_weights()
            self.assertFalse(hasattr(model, "initialized"))
        self.assertEqual(fixture.active, [])
        self.assertIs(fixture.torch_init.normal_, fixture.native_normal)
        self.assertIs(fixture.base.init_weights, fixture.original_initializer)
        model.init_weights()
        self.assertTrue(model.initialized)
        for label in ("dtype", "no_tie", "patches", "safe_creation", "cpu"):
            self.assertIn("enter:" + label, fixture.events)
            self.assertIn("exit:" + label, fixture.events)

    def test_constructor_failure_restores_original_initializers_and_contexts(self):
        self.install()
        failure = RuntimeError("model constructor failed")
        with self.assertRaises(RuntimeError) as caught:
            with ExitStack() as stack:
                for context in self.contexts():
                    stack.enter_context(context)
                raise failure
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.fixture.active, [])
        self.assertIs(self.fixture.torch_init.normal_, self.fixture.native_normal)
        self.assertIs(self.fixture.base.init_weights, self.fixture.original_initializer)

    def test_observable_marker_and_original_reference_are_on_raw_descriptor(self):
        self.install()
        wrapper = vars(self.fixture.base)["get_init_context"].__func__
        self.assertIs(wrapper._jittor_transformers_materialized_init, True)
        self.assertIs(wrapper._jittor_original, self.fixture.original_descriptor.__func__)
        self.assertEqual(self.adapter.SUPPORTED_VERSIONS, {"5.9.0"})
        self.assertEqual(self.adapter.SOURCE_SHA256,
                         {"get_init_context": GET_CONTEXT_SHA, "no_init_weights": NO_INIT_SHA})

    def test_disabling_optin_after_install_returns_original_meta_policy(self):
        self.install()
        with mock.patch.dict(os.environ, {ENV: "0"}):
            contexts = self.contexts()
        self.assertEqual(len(contexts), 5)
        self.assertEqual(contexts[3].type, "meta")
        self.assertEqual(self.fixture.events, [])
        self.assertIs(self.fixture.base.init_weights, self.fixture.original_initializer)

    def test_inherited_binding_works_but_super_call_from_override_is_rejected(self):
        self.install()
        class Inherited(self.fixture.base):
            pass
        contexts = Inherited.get_init_context("float32", False, False, False)
        self.assertEqual(contexts[4].type, "cpu")

        class Custom(self.fixture.base):
            @classmethod
            def get_init_context(cls, *args):
                return super().get_init_context(*args)

        with self.assertRaises(UnsupportedAdapterVersion):
            Custom.get_init_context("float32", False, False, False)

    def test_invalid_returned_context_or_later_factory_override_is_rejected(self):
        with mock.patch.object(self.fixture.module, "apply_patches", lambda: object()):
            self.install()
            with self.assertRaises(UnsupportedAdapterVersion):
                self.contexts()
        self.assertEqual(self.fixture.events, [])
        with self.assertRaises(UnsupportedAdapterVersion):
            self.contexts()

    def test_quantization_deepspeed_and_kernels_fail_before_context_entry(self):
        self.install()
        for kwargs in ({"is_quantized": True}, {"_is_ds_init_called": True}, {"allow_all_kernels": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, RuntimeError, UnsupportedAdapterVersion)):
                self.contexts(**kwargs)
        with mock.patch.object(self.fixture.module, "is_deepspeed_zero3_enabled", return_value=True):
            with self.assertRaises((ValueError, RuntimeError, UnsupportedAdapterVersion)):
                self.contexts()
        self.assertEqual(self.fixture.events, [])

    def test_pinned_version_and_modified_dependency_are_rejected_without_patch(self):
        with mock.patch.object(self.fixture.root, "__version__", "5.8.0"):
            with self.assertRaises(UnsupportedAdapterVersion):
                self.install()
        self.assertIs(vars(self.fixture.base)["get_init_context"], self.fixture.original_descriptor)
        with mock.patch.object(self.fixture.init, "no_init_weights", lambda: None):
            with self.assertRaises(UnsupportedAdapterVersion):
                self.install()
        self.assertIs(vars(self.fixture.base)["get_init_context"], self.fixture.original_descriptor)

    def test_changed_selector_and_remote_code_are_rejected(self):
        with mock.patch.object(self.fixture.base, "get_init_context", classmethod(lambda cls, *args: [])):
            with self.assertRaises(UnsupportedAdapterVersion):
                self.install()
        self.install()
        with mock.patch.object(self.fixture.base, "is_remote_code", classmethod(lambda cls: True)):
            with self.assertRaises((ValueError, RuntimeError, UnsupportedAdapterVersion)):
                self.contexts()

    def test_functools_wrapped_impostor_cannot_bypass_source_hash(self):
        from functools import wraps
        original = self.fixture.init.no_init_weights
        @wraps(original)
        def impostor():
            return original()
        with mock.patch.object(self.fixture.init, "no_init_weights", impostor):
            with self.assertRaises(UnsupportedAdapterVersion):
                self.install()
        self.assertIs(vars(self.fixture.base)["get_init_context"], self.fixture.original_descriptor)

    def test_repeat_install_and_uninstall_preserve_exact_descriptor(self):
        self.install()
        replacement = vars(self.fixture.base)["get_init_context"]
        self.install()
        self.assertIs(vars(self.fixture.base)["get_init_context"], replacement)
        patcher.uninstall_module_patches()
        self.assertIs(vars(self.fixture.base)["get_init_context"], self.fixture.original_descriptor)

    def test_registration_never_imports_transformers(self):
        import builtins
        original_import = builtins.__import__
        seen = []
        def guarded_import(name, *args, **kwargs):
            self.assertFalse(name == "transformers" or name.startswith("transformers."))
            return original_import(name, *args, **kwargs)
        with mock.patch.object(builtins, "__import__", guarded_import):
            self.adapter.register(lambda path, callback: seen.append((path, callback)))
        self.assertEqual([path for path, _ in seen], ["transformers.modeling_utils"])

    def test_callback_failure_rolls_back_classmethod(self):
        self.adapter.register(patcher.register_module_patch)
        callback, = patcher.registered_module_patches()["transformers.modeling_utils"]
        with self.assertRaisesRegex(RuntimeError, "abort installation"):
            with transaction.runtime_hook("loading_failed_install"):
                callback(self.fixture.module)
                raise RuntimeError("abort installation")
        self.assertIs(vars(self.fixture.base)["get_init_context"], self.fixture.original_descriptor)

    def test_external_replacement_is_preserved_on_uninstall_conflict(self):
        self.install()
        owned = vars(self.fixture.base)["get_init_context"]
        external = classmethod(lambda cls, *args: [])
        self.fixture.base.get_init_context = external
        try:
            with self.assertRaises(transaction.TransactionConflict):
                patcher.uninstall_module_patches()
            self.assertIs(vars(self.fixture.base)["get_init_context"], external)
        finally:
            self.fixture.base.get_init_context = owned

    def test_relocatable_contract_uses_shared_guard_with_negative_controls(self):
        package = Path(self.adapter.__file__).parent
        for path in package.rglob("*.py"):
            if "tests" not in path.relative_to(package).parts:
                self.assertEqual(boundary_errors(path.read_text()), [], str(path))
        for forbidden in ("from jittor._core import Var", "torch.is_meta = True", "setattr(torch, 'device', object)"):
            self.assertTrue(boundary_errors(forbidden))


if __name__ == "__main__":
    unittest.main()
