"""Stable core misc objects and their live installation-owned state."""

import ast
from pathlib import Path
import pickle
import unittest

import numpy as np


class TestCoreMiscOwner(unittest.TestCase):
    def test_a_published_objects_have_one_owner_and_pickle_identity(self):
        import torch
        from jittor.compat.torch.installers import core
        from jittor.compat.torch.fidelity import fidelity_of

        for name, implementation in core._MISC_BINDINGS.items():
            with self.subTest(name=name):
                self.assertIs(getattr(torch, name), implementation)
                self.assertIs(fidelity_of("torch." + name).implementation, implementation)
                self.assertEqual(implementation.__module__, core.__name__)
                self.assertIs(pickle.loads(pickle.dumps(implementation)), implementation)
        for storage in core._STORAGE_TYPES:
            self.assertIs(getattr(torch, storage.__name__), storage)
            self.assertIs(getattr(torch.storage, storage.__name__), storage)
        self.assertIs(torch.random.manual_seed, core.manual_seed)
        self.assertIs(torch.random.seed, core._torch_seed)
        self.assertIs(torch.Tensor.bincount, core.bincount)
        self.assertEqual(fidelity_of("torch.set_autocast_enabled").level.value, "unimplemented")
        self.assertEqual(fidelity_of("torch.use_deterministic_algorithms").level.value, "unimplemented")

    def test_b_values_limits_and_promotion(self):
        import torch

        x = torch.tensor([1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(torch.norm(x).item()), float(np.sqrt(14)), places=5)
        np.testing.assert_array_equal(torch.where(x > 1, x, -x).numpy(), [-1, 2, 3])
        np.testing.assert_array_equal(torch.where(x > 1)[0].numpy(), [1, 2])
        indices = torch.tensor([0, 1, 1, 3], dtype=torch.int64)
        np.testing.assert_array_equal(torch.bincount(indices).numpy(), [1, 2, 0, 1])
        np.testing.assert_array_equal(indices.bincount().numpy(), [1, 2, 0, 1])
        data = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        result = torch.segment_reduce(data, "sum", lengths=[2, 1])
        np.testing.assert_array_equal(result.numpy(), [[4, 6], [5, 6]])
        self.assertIs(torch.promote_types(torch.float16, torch.bfloat16), torch.float32)
        self.assertIs(torch.result_type(x, 0.5), torch.float32)
        self.assertFalse(torch.can_cast(torch.float64, torch.float32))
        self.assertEqual(torch.finfo(torch.float32).eps, np.finfo(np.float32).eps)
        self.assertEqual(torch.iinfo(torch.int64).min, np.iinfo(np.int64).min)
        self.assertTrue(torch.is_tensor(x))
        self.assertTrue(torch.is_floating_point(x))
        self.assertEqual(torch.numel(x), 3)

    def test_c_seed_default_dtype_and_grad_scopes(self):
        import torch

        dtype, seed = torch.get_default_dtype(), torch.initial_seed()
        try:
            torch.set_default_dtype(torch.float64)
            self.assertIs(torch.get_default_dtype(), torch.float64)
            self.assertIs(torch.manual_seed(1729), torch)
            self.assertEqual(torch.initial_seed(), 1729)
            self.assertEqual(int(torch.get_rng_state().numpy()[0]), 1729)
            torch.set_rng_state(torch.tensor([42], dtype=torch.int64))
            self.assertEqual(torch.random.initial_seed(), 42)
            self.assertEqual(torch.get_default_device().type, "cpu")
            with torch.set_grad_enabled(False):
                self.assertFalse(torch.is_grad_enabled())
        finally:
            torch.set_default_dtype(dtype)
            torch.manual_seed(seed)

    def test_d_rebinding_retains_objects_and_owned_policy(self):
        import jittor as jt
        import torch
        from jittor.compat.torch.context import get_install_context
        from jittor.compat.torch.installers import core

        dtype = torch.get_default_dtype()
        random = torch.random
        try:
            torch.set_default_dtype(torch.float64)
            core.install_misc(get_install_context(jt))
            self.assertIs(torch.random, random)
            self.assertIs(torch.get_default_dtype(), torch.float64)
            for name, implementation in core._MISC_BINDINGS.items():
                self.assertIs(getattr(torch, name), implementation)
        finally:
            torch.set_default_dtype(dtype)

    def test_e_installer_defines_no_api_closures_or_classes(self):
        from jittor.compat.torch.installers import core

        tree = ast.parse(Path(core.__file__).read_text())
        installer = next(node for node in tree.body
                         if isinstance(node, ast.FunctionDef) and node.name == "install_misc")
        self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Lambda))
                             for node in ast.walk(installer) if node is not installer))
