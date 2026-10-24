"""Canonical explicit placement override, with default factory rules intact."""
import ast
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


class TestInternalPlacementOverride(unittest.TestCase):
    def setUp(self):
        self.placement, self.events = (0, 0), []
        self.initial_construction = self.construction = (True, 1, 7)
        def select(backend, device):
            old = self.placement
            self.placement = (backend, device)
            self.events.append(self.placement)
            return old
        def restore(previous):
            self.placement = previous
        def select_construction(backend, device):
            previous = self.construction
            self.construction = (True, backend, 0 if backend == 0 else device)
            return previous
        def restore_construction(previous):
            self.construction = previous
        self.runtime = SimpleNamespace(core=SimpleNamespace(
            _current_tensor_placement=lambda: self.placement,
            _set_tensor_placement=select, _reset_tensor_placement=restore,
            _set_construction_placement=select_construction,
            _restore_construction_placement=restore_construction))
        tree = ast.parse((ROOT / "python/jittor/_core/var.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "placement_scope_like")
        namespace = {"contextmanager": contextmanager, "ori_int": int}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "placement_owner", "exec"), namespace)
        self.scope = namespace["placement_scope_like"]
        self.patch = mock.patch.dict(sys.modules, jittor=self.runtime)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_default_reference_never_overrides_callers_explicit_request(self):
        with self.scope(SimpleNamespace(placement_backend=2, device_id=1)):
            self.assertEqual(self.placement, (0, 0))
            self.assertEqual(self.construction, self.initial_construction)
        self.assertEqual(self.events, [])
        self.placement = None
        with self.scope(SimpleNamespace(placement_backend=2, device_id=1)):
            self.assertEqual(self.placement, (2, 1))
            self.assertEqual(self.construction, (True, 2, 1))
        self.assertIsNone(self.placement)
        self.assertEqual(self.construction, self.initial_construction)

    def test_internal_override_is_exact_and_restores_on_exception(self):
        for backend, device in ((0, 0), (1, 1), (2, 1)):
            with self.assertRaisesRegex(ValueError, "allocation failed"):
                with self.scope(SimpleNamespace(placement_backend=backend, device_id=device), override=True):
                    self.assertEqual(self.placement, (backend, device))
                    self.assertEqual(self.construction, (True, backend, device))
                    with self.scope(SimpleNamespace(placement_backend=2, device_id=3), override=True):
                        self.assertEqual(self.construction, (True, 2, 3))
                    self.assertEqual(self.placement, (backend, device))
                    self.assertEqual(self.construction, (True, backend, device))
                    raise ValueError("allocation failed")
            self.assertEqual(self.placement, (0, 0))
            self.assertEqual(self.construction, self.initial_construction)

    def test_unplaced_reference_does_not_invent_a_backend(self):
        with self.scope(SimpleNamespace(placement_backend=-1, device_id=-1), override=True):
            self.assertEqual(self.placement, (0, 0))
        self.assertEqual(self.events, [])

    def test_fsdp_internal_zeros_use_canonical_explicit_override(self):
        tree = ast.parse((ROOT / "compat/fsdp2/common.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_zeros_like_shape")
        calls = []
        def zeros(shape, dtype):
            calls.append((shape, dtype, self.placement, self.construction))
            return "native-allocation"
        namespace = {"_factory_scope_like": self.scope, "jt": SimpleNamespace(zeros=zeros)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "fsdp_internal_zero", "exec"), namespace)
        reference = SimpleNamespace(placement_backend=2, device_id=1, dtype="float32")
        self.assertEqual(namespace["_zeros_like_shape"](reference, (3,)), "native-allocation")
        self.assertEqual(calls, [((3,), "float32", (2, 1), (True, 2, 1))])
        self.assertEqual(self.placement, (0, 0))
        self.assertEqual(self.construction, self.initial_construction)


if __name__ == "__main__":
    unittest.main()
