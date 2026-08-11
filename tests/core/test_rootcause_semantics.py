"""Regressions for core semantics that replace downstream compatibility patches."""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import jittor as jt
from jittor import nn
from jittor.compat import external_backend, module_patcher


class TestRequiresGradSemantics(unittest.TestCase):
    def test_temporary_freeze_preserves_preexisting_policy_graph(self):
        weight = nn.Parameter(jt.array([1.0, -2.0, 3.0]))
        inputs = jt.array([2.0, 4.0, -1.0]).stop_grad()
        policy_loss = (weight * inputs).sum()

        self.assertIs(weight.requires_grad_(False), weight)
        self.assertFalse(weight.requires_grad)
        frozen_loss = (weight * inputs).sum()
        self.assertFalse(frozen_loss.requires_grad)

        # This mirrors PEFT disable_adapter(): the reference pass is frozen, then
        # the same Parameter is restored before the policy graph is differentiated.
        self.assertIs(weight.requires_grad_(True), weight)
        self.assertTrue(weight.requires_grad)
        grad = jt.grad(policy_loss, [weight])[0]
        np.testing.assert_allclose(grad.numpy(), inputs.numpy(), atol=1e-6)

        # Operations created while disabled remain detached after re-enabling.
        # sync() also drives fused execution's repeated Op::init() path.
        frozen_loss.sync()
        frozen_grad = jt.grad(frozen_loss, [weight])[0]
        np.testing.assert_allclose(
            frozen_grad.numpy(), np.zeros(3, dtype=np.float32), atol=1e-6
        )

    def test_frozen_edge_does_not_block_an_active_input(self):
        frozen = nn.Parameter(jt.array([1.0, -2.0, 3.0]))
        active = nn.Parameter(jt.array([2.0, 4.0, -1.0]))

        frozen.requires_grad_(False)
        mixed_loss = (frozen * active).sum()
        self.assertTrue(mixed_loss.requires_grad)
        frozen.requires_grad_(True)

        frozen_grad, active_grad = jt.grad(mixed_loss, [frozen, active])
        np.testing.assert_allclose(
            frozen_grad.numpy(), np.zeros(3, dtype=np.float32), atol=1e-6
        )
        np.testing.assert_allclose(
            active_grad.numpy(), frozen.numpy(), atol=1e-6
        )

    def test_explicit_stop_grad_remains_permanent(self):
        value = jt.array([2.0, 3.0])
        old_loss = (value * value).sum()
        value.stop_grad()
        value.start_grad()
        grad = jt.grad(old_loss, [value])[0]
        np.testing.assert_allclose(
            grad.numpy(), np.zeros(2, dtype=np.float32), atol=1e-6
        )

    def test_optimizer_skips_frozen_parameter_and_resumes_after_enable(self):
        frozen = nn.Parameter(jt.array([2.0]))
        active = nn.Parameter(jt.array([3.0]))
        optimizer = jt.optim.AdamW(
            [frozen, active], lr=0.1, weight_decay=0.5
        )

        frozen.requires_grad_(False)
        optimizer.step((frozen * active).sum())
        np.testing.assert_allclose(frozen.numpy(), [2.0], atol=1e-6)
        self.assertNotAlmostEqual(float(active.item()), 3.0)

        frozen.requires_grad_(True)
        before = float(frozen.item())
        optimizer.step((frozen * active).sum())
        self.assertNotAlmostEqual(float(frozen.item()), before)


class TestParameterIdentity(unittest.TestCase):
    def test_parameter_metaclass_distinguishes_raw_vars(self):
        raw = jt.array([1.0, 2.0])
        self.assertNotIsInstance(raw, nn.Parameter)

        parameter = nn.Parameter(jt.array([1.0, 2.0]))
        self.assertIsInstance(parameter, nn.Parameter)
        self.assertTrue(parameter.requires_grad)

        layer = nn.Linear(2, 3)
        self.assertIsInstance(layer.weight, nn.Parameter)
        self.assertIsInstance(layer.bias, nn.Parameter)

        layer.register_buffer("running", jt.zeros((3,)))
        self.assertNotIsInstance(layer.running, nn.Parameter)

        parameters = nn.ParameterList([jt.ones((2,))])
        self.assertIsInstance(parameters[0], nn.Parameter)

    def test_parameter_marker_survives_private_alias_assignment(self):
        module = jt.Module()
        parameter = nn.Parameter(jt.array([1.0, 2.0]))

        module.weight = parameter
        module._alias = parameter

        self.assertIs(module.weight, parameter)
        self.assertIs(module._alias, parameter)
        self.assertIsInstance(parameter, nn.Parameter)
        parameters = module.parameters()
        self.assertEqual(len(parameters), 1)
        self.assertIs(parameters[0], parameter)

    def test_buffer_and_parameter_aliases_preserve_object_roles(self):
        module = jt.Module()
        buffer = jt.array([1.0, 2.0]).stop_grad()
        module.register_buffer("running", buffer)
        module.running_alias = buffer

        self.assertNotIsInstance(module.running, nn.Parameter)
        self.assertNotIsInstance(module.running_alias, nn.Parameter)
        self.assertEqual(module.parameters(), [])
        self.assertEqual(
            [name for name, _ in module.named_buffers()], ["running"]
        )

        parameter = nn.Parameter(jt.array([3.0, 4.0]))
        module.weight = parameter
        module.register_buffer("weight_view", parameter)
        module._weight_alias = parameter

        self.assertIsInstance(module.weight, nn.Parameter)
        self.assertIsInstance(module.weight_view, nn.Parameter)
        self.assertIsInstance(module._weight_alias, nn.Parameter)
        parameters = module.parameters()
        self.assertEqual(len(parameters), 1)
        self.assertIs(parameters[0], parameter)
        self.assertEqual(
            [name for name, _ in module.named_parameters()], ["weight"]
        )
        self.assertEqual(
            [name for name, _ in module.named_buffers()],
            ["running", "weight_view"],
        )

    def test_no_grad_tensor_is_rewrapped_as_parameter(self):
        with jt.no_grad():
            fused = jt.stack([jt.ones((2,)), jt.zeros((2,))])
        self.assertNotIsInstance(fused, nn.Parameter)
        self.assertFalse(fused.requires_grad)

        parameter = nn.Parameter(fused, requires_grad=True)
        self.assertIsInstance(parameter, nn.Parameter)
        self.assertTrue(parameter.requires_grad)

    def test_canonical_and_deploy_shim_use_the_same_marker(self):
        shim_path = (
            jt.__path__[0] + "/torch_shim/torch__init__.py"
        )
        source = Path(shim_path).read_text(encoding="utf-8")
        self.assertIn("_is_torch_parameter", source)
        self.assertNotIn("return isinstance(obj, _jt.Var)\n", source)


class TestShapeHashSemantics(unittest.TestCase):
    def test_nanovector_hash_matches_equal_tuple(self):
        for values in ((), (2,), (2, 3, 5), (-1, 0, 1)):
            shape = jt.NanoVector(values)
            with self.subTest(values=values):
                self.assertEqual(shape, values)
                self.assertEqual(hash(shape), hash(values))
                grouped = {shape: "value"}
                self.assertEqual(grouped[values], "value")

    def test_var_shape_is_a_stable_dictionary_key(self):
        shape = jt.ones((2, 3, 4)).shape
        grouped = {shape[1:]: "image"}
        self.assertEqual(grouped[(3, 4)], "image")
        self.assertEqual(hash(shape), hash((2, 3, 4)))


class TestPatchReporting(unittest.TestCase):
    def test_root_runtime_report_is_readable_and_repeatable(self):
        patch_report = module_patcher.PatchReport(
            (
                module_patcher.PatchResult(
                    "entry_point", "broken", "provider:broken", "failed", "first"
                ),
                module_patcher.PatchResult(
                    "entry_point", "working", "provider:working", "loaded"
                ),
            ),
            True,
        )
        backend_report = (
            external_backend.BackendEntryPointResult(
                "broken-backend", "provider:backend", "failed", "second"
            ),
            external_backend.BackendEntryPointResult(
                "working-backend", "provider:working", "loaded"
            ),
        )
        fake_triton = types.ModuleType("jittor.triton_shim")
        with mock.patch.dict(sys.modules, {"jittor.triton_shim": fake_triton}), \
                mock.patch.object(
                    module_patcher, "install_module_patches", return_value=patch_report
                ) as patch_mock, mock.patch.object(
                    external_backend,
                    "load_external_backend_entry_points",
                    return_value=backend_report,
                ) as backend_mock:
            first = jt._apply_external_runtime_patches()
            second = jt._apply_external_runtime_patches()

        self.assertEqual(patch_mock.call_count, 2)
        self.assertEqual(backend_mock.call_count, 2)
        self.assertTrue(first["triton_shim"]["ok"])
        self.assertFalse(first["module_patches"]["ok"])
        self.assertEqual(first["module_patches"]["results"][0]["detail"], "first")
        self.assertFalse(first["external_backends"]["ok"])
        self.assertEqual(first["external_backends"]["results"][0]["detail"], "second")
        self.assertEqual(first, second)
        self.assertIs(jt._apply_external_runtime_patches.last_report, second)

    def test_obsolete_downstream_patches_are_gone(self):
        self.assertFalse((Path(jt.__file__).resolve().parent / "monkeypatch_ops.py").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
