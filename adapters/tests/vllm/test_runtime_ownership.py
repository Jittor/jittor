import ast
import pathlib
import sys
import threading
import types
import unittest
import uuid
from unittest import mock
import jittor_adapters.vllm as jittor_vllm
from jittor.compat import transaction as tx


class RuntimeHookOwnership(unittest.TestCase):
    def source_function(self, relative, name, namespace):
        source = pathlib.Path(jittor_vllm.__file__).resolve().parent / relative
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


    def test_flash_bundle_conflict_rolls_back_first_publication(self):
        bundle, interface = self.name(), self.name()
        foreign = types.ModuleType(interface)
        namespace = dict(sys=sys, PermissiveModule=types.ModuleType,
                         _BUNDLE=bundle, _INTERFACE=interface,
                         flash_attn_varlen_func=object(), flash_attn_with_kvcache=object(),
                         _no_scheduler_metadata=object(), owned_runtime_hook=tx.owned_runtime_hook,
                         active_transaction=tx.active_transaction,
                         runtime_owns_module=tx.runtime_owns_module,
                         install_permissive_package=lambda *a, **kw: None)
        install = self.source_function("flash_attn.py", "install", namespace)
        with mock.patch.dict(sys.modules, {interface: foreign}):
            with self.assertRaises(tx.TransactionConflict):
                install()
            self.assertNotIn(bundle, sys.modules)
            self.assertIs(sys.modules[interface], foreign)
        self.assert_lock_available(tx.InstallTransaction._lock)


    def test_vllm_operator_failure_rolls_back_partial_registry(self):
        class Dispatcher:
            def __init__(self):
                self._namespaces = {}
            def __getattr__(self, name):
                return self._namespaces.setdefault(name, types.SimpleNamespace(_ops={}))
        dispatcher = Dispatcher()
        class Library:
            def __init__(self, namespace, kind):
                self.namespace = getattr(dispatcher, namespace)
            def define(self, schema):
                name = schema.split("(")[0]
                self.namespace._ops[name] = types.SimpleNamespace(schema=schema, impls={})
            def impl(self, name, function):
                self.namespace._ops[name].impls["default"] = function
                if name == "second":
                    raise ValueError("registration failed")
        namespace = dict(__name__="jittor_adapters.vllm.custom_ops",
                         __package__="jittor_adapters.vllm",
                         _OPERATORS=(("first", "first()"), ("second", "second()")),
                         _IMPLEMENTATIONS={"first": lambda: 1, "second": lambda: 2},
                         _CAPABILITY_PROBES=())
        register = self.source_function("custom_ops.py", "register", namespace)
        target = types.SimpleNamespace(ops=dispatcher, library=types.SimpleNamespace(Library=Library))
        with self.assertRaisesRegex(ValueError, "registration failed"):
            with tx.runtime_hook(self.name()):
                register(target)
        self.assertEqual(dispatcher._namespaces, {})


