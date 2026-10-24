"""Host-only NPU facade routing contracts; no Jittor import or hardware claim.

The initial facade is built from the actual generic publication statements.
The NPU-specific installer must replace its placeholders with native queries.
"""

import ast
from pathlib import Path
import types
import unittest


ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "compat/torch/installers/cuda"
NAMES = ("is_available", "device_count", "current_device", "set_device")


def without_imports(path):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if not isinstance(node, ast.ImportFrom)
                 and not (isinstance(node, ast.Import)
                          and any(alias.name == "jittor" for alias in node.names))]
    return compile(tree, str(path), "exec")


class TestNPUDeviceControlContract(unittest.TestCase):
    def setUp(self):
        self.selected = 1
        self.visible = 2
        self.calls = []
        self.core = types.SimpleNamespace(current_device=self.current,
                                          get_device_count=self.count,
                                          set_device=self.select)
        self.jt = types.SimpleNamespace(core=self.core,
            compiler=types.SimpleNamespace(has_acl=True, has_cuda=True),
            flags=types.SimpleNamespace(use_cuda=0, device_id=1))
        memory = {"jt": self.jt}
        exec(without_imports(DIRECTORY / "npu_memory.py"), memory)
        self.api = {"jt": self.jt, "_device_index": memory["_device_index"]}
        exec(without_imports(DIRECTORY / "npu_device.py"), self.api)
        # Execute the actual generic defaults and four publication statements,
        # so the old facade fails behaviorally, not merely on a missing symbol.
        api_tree = ast.parse((DIRECTORY / "api.py").read_text())
        defaults = [node for node in api_tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in {"_api_mod_" + name for name in NAMES}]
        self.npu = types.SimpleNamespace()
        self.native_name = lambda device=None: "native-name-preserved"
        self.memory_allocated = lambda device=None: 123
        self.npu.get_device_name = self.native_name
        self.npu.memory_allocated = self.memory_allocated
        namespace = {"_mod": self.npu}
        exec(compile(ast.Module(body=defaults, type_ignores=[]), "generic_npu_defaults", "exec"), namespace)
        bindings = ast.parse((DIRECTORY / "bindings.py").read_text())
        assignments = [node for node in ast.walk(bindings)
                       if isinstance(node, ast.Assign) and len(node.targets) == 1
                       and isinstance(node.targets[0], ast.Attribute)
                       and isinstance(node.targets[0].value, ast.Name)
                       and node.targets[0].value.id == "_mod"
                       and node.targets[0].attr in NAMES]
        self.assertEqual(len(assignments), 4)
        exec(compile(ast.Module(body=assignments, type_ignores=[]), "npu_publication", "exec"), namespace)
        installer = self.api.get("install")
        if installer is not None:
            installer(self.npu)

    def count(self):
        self.calls.append(("count",))
        return self.visible

    def current(self):
        self.calls.append(("current",))
        return self.selected

    def select(self, index):
        self.calls.append(("set", index))
        self.selected = index

    def test_acl_availability_uses_real_visible_count_not_use_cuda(self):
        self.assertTrue(self.npu.is_available())
        self.assertEqual(self.npu.device_count(), 2)
        self.visible = 0
        self.assertFalse(self.npu.is_available())
        self.assertEqual(self.npu.device_count(), 0)
        self.assertEqual(self.jt.flags.use_cuda, 0)

    def test_no_acl_cannot_report_an_available_other_accelerator(self):
        self.jt.compiler.has_acl = False
        self.assertFalse(self.npu.is_available())
        self.assertEqual(self.npu.device_count(), 0)
        self.assertEqual(self.calls, [])
        for function, args in ((self.npu.current_device, ()), (self.npu.set_device, (0,))):
            with self.subTest(function=function.__name__), self.assertRaisesRegex(RuntimeError, "ACL"):
                function(*args)
        self.assertEqual(self.calls, [])

    def test_current_device_tracks_native_nonzero_ordinal(self):
        self.assertEqual(self.npu.current_device(), 1)
        self.selected = 0
        self.assertEqual(self.npu.current_device(), 0)
        self.assertGreaterEqual(self.calls.count(("current",)), 2)

    def test_set_device_routes_native_without_touching_factory_policy(self):
        for device, expected in ((0, 0), ("npu:1", 1),
                                 (types.SimpleNamespace(type="npu", index=0), 0)):
            self.npu.set_device(device)
            self.assertEqual(self.selected, expected)
            self.assertIn(("set", expected), self.calls)
            self.assertEqual(self.npu.current_device(), expected)
            self.assertEqual(self.jt.flags.use_cuda, 0)

    def test_invalid_device_is_not_a_successful_noop(self):
        for device in (True, False, 0.5, "cpu", "cuda:0", "npu:2", 2):
            with self.subTest(device=device):
                before = self.selected
                with self.assertRaises((ValueError, TypeError, RuntimeError)):
                    self.npu.set_device(device)
                self.assertEqual(self.selected, before)
        self.assertFalse(any(call[0] == "set" for call in self.calls))

    def test_negative_integer_matches_official_npu_noop_without_native_calls(self):
        class IntegerIndex:
            def __index__(self):
                return -2
        for acl_available in (True, False):
            self.jt.compiler.has_acl = acl_available
            for device in (-1, -10, IntegerIndex()):
                self.assertIsNone(self.npu.set_device(device))
                self.assertEqual(self.selected, 1)
                self.assertEqual(self.calls, [])

    def test_native_query_and_switch_failures_propagate(self):
        def fail(*args):
            raise RuntimeError("native device failure")
        for method, public, args in (("get_device_count", "is_available", ()),
                                     ("get_device_count", "device_count", ()),
                                     ("current_device", "current_device", ()),
                                     ("set_device", "set_device", (0,))):
            original = getattr(self.core, method)
            setattr(self.core, method, fail)
            try:
                with self.subTest(public=public), self.assertRaisesRegex(RuntimeError, "native device failure"):
                    getattr(self.npu, public)(*args)
            finally:
                setattr(self.core, method, original)

    def test_uninitialized_current_device_is_not_fabricated_zero(self):
        self.selected = -1
        with self.assertRaises((RuntimeError, ValueError)):
            self.npu.current_device()

    def test_installer_preserves_name_memory_and_is_repeatable(self):
        installer = self.api.get("install")
        self.assertTrue(callable(installer), "NPU device control requires an explicit native installer")
        installer(self.npu)
        self.assertIs(self.npu.get_device_name, self.native_name)
        self.assertIs(self.npu.memory_allocated, self.memory_allocated)
        for name in NAMES:
            self.assertIs(getattr(self.npu, name), self.api[name])

    def test_publication_calls_native_installer_after_generic_defaults(self):
        tree = ast.parse((DIRECTORY / "bindings.py").read_text())
        installation = next(node for node in tree.body
                            if isinstance(node, ast.FunctionDef) and node.name == "_install_cuda")
        imports = [(index, node) for index, node in enumerate(installation.body)
                   if isinstance(node, ast.ImportFrom) and node.module == "npu_device"
                   and any(alias.name == "install" for alias in node.names)]
        self.assertEqual(len(imports), 1)
        position, statement = imports[0]
        alias = next(alias.asname or alias.name for alias in statement.names if alias.name == "install")
        following = installation.body[position + 1]
        self.assertIsInstance(following, ast.Expr)
        self.assertIsInstance(following.value, ast.Call)
        self.assertEqual(ast.unparse(following.value.func), alias)
        self.assertEqual([ast.unparse(argument) for argument in following.value.args], ["g.npu"])
        self.assertTrue(any(isinstance(node, ast.For) and "_dev_ns" in ast.dump(node)
                            for node in installation.body[:position]))


if __name__ == "__main__":
    unittest.main()
