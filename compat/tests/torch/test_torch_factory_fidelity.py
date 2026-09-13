import importlib
import unittest

import numpy as np

import jittor as jt
import torch


FACTORY_NAMES = (
    "arange", "bernoulli", "empty", "empty_like", "full", "full_like",
    "linspace", "multinomial", "normal", "ones", "ones_like", "rand",
    "rand_like", "randint", "randn", "randn_like", "randperm", "tril",
    "triu", "zeros", "zeros_like",
)


class TestTorchFactoryFidelity(unittest.TestCase):

    def test_triu_indices_rectangular_offset_dtype_and_device(self):
        actual = torch.triu_indices(3, 5, offset=1, dtype=torch.int64)
        rows, cols = np.triu_indices(3, k=1, m=5)
        np.testing.assert_array_equal(actual.cpu().numpy(), np.stack((rows, cols)))
        self.assertEqual(actual.dtype, torch.int64)

    def test_triu_indices_empty_and_extreme_offsets(self):
        for shape, offset in (((0, 4), 0), ((3, 0), 0), ((2, 3), 4), ((2, 3), -3)):
            row, col = shape
            actual = torch.triu_indices(row, col, offset=offset, dtype=torch.int32)
            rows, cols = np.triu_indices(row, k=offset, m=col)
            np.testing.assert_array_equal(actual.cpu().numpy(), np.stack((rows, cols)))
            self.assertEqual(actual.dtype, torch.int32)

    def test_normal_accepts_keyword_mean_std_and_size(self):
        actual = torch.normal(mean=0.0, std=0.02, size=(2, 3, 1))
        self.assertEqual(tuple(actual.shape), (2, 3, 1))
        self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_cpu_generator_randperm_matches_pytorch_26_and_advances(self):
        generator = torch.Generator().manual_seed(777)
        first = torch.randperm(256, generator=generator)
        second = torch.randperm(128, generator=generator)
        np.testing.assert_array_equal(first[:4].numpy(), [103, 15, 197, 254])
        np.testing.assert_array_equal(second[:4].numpy(), [111, 108, 121, 2])
        replay = torch.Generator().manual_seed(777)
        np.testing.assert_array_equal(torch.randperm(256, generator=replay).numpy(), first.numpy())
        state = generator.get_state()
        expected = torch.randperm(16, generator=generator)
        restored = torch.Generator().set_state(state)
        np.testing.assert_array_equal(torch.randperm(16, generator=restored).numpy(), expected.numpy())

    def test_generator_randperm_lazy_reverse_evaluation_keeps_reserved_states(self):
        forward_generator = torch.Generator().manual_seed(91)
        expected_first = torch.randperm(32, generator=forward_generator).numpy()
        expected_second = torch.randperm(32, generator=forward_generator).numpy()
        reverse_generator = torch.Generator().manual_seed(91)
        actual_first = torch.randperm(32, generator=reverse_generator)
        actual_second = torch.randperm(32, generator=reverse_generator)
        np.testing.assert_array_equal(actual_second.numpy(), expected_second)
        np.testing.assert_array_equal(actual_first.numpy(), expected_first)

    def test_unimplemented_generator_distribution_fails_closed(self):
        with self.assertRaisesRegex(NotImplementedError, "Generator"):
            torch.normal(0.0, 1.0, size=(2,), generator=torch.Generator())

    def test_manual_seed_materializes_live_random_graph_before_reset(self):
        torch.manual_seed(41)
        expected_old = torch.rand(8).numpy()
        torch.manual_seed(99)
        expected_new = torch.rand(8).numpy()

        torch.manual_seed(41)
        pending = torch.rand(8)
        torch.manual_seed(99)
        after_reset = torch.rand(8)
        np.testing.assert_array_equal(pending.numpy(), expected_old)
        np.testing.assert_array_equal(after_reset.numpy(), expected_new)

        torch.manual_seed(123)
        state = torch.get_rng_state()
        pending = torch.rand(8)
        torch.set_rng_state(state)
        replay = torch.rand(8)
        np.testing.assert_array_equal(pending.numpy(), replay.numpy())

    def test_factory_objects_are_module_level_and_keep_public_identity(self):
        factories = importlib.import_module(
            "jittor.compat.torch.installers.factories")
        for name in FACTORY_NAMES:
            with self.subTest(name=name):
                implementation = getattr(factories, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertEqual(implementation.__name__, name)

    def test_fidelity_report_is_complete_deterministic_and_conservative(self):
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        report = fidelity.fidelity_report(prefix="torch.")
        factory_records = tuple(
            record for record in report
            if record.api in {"torch." + name for name in FACTORY_NAMES}
        )
        self.assertEqual(
            tuple(record.api for record in factory_records),
            tuple("torch." + name for name in FACTORY_NAMES),
        )
        for record in factory_records:
            self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
            self.assertIs(
                fidelity.fidelity_of(record.api).implementation,
                record.implementation,
            )
            self.assertTrue(record.detail)

    def test_independently_imported_factory_executes_on_cpu(self):
        factories = importlib.import_module(
            "jittor.compat.torch.installers.factories")
        with jt.flag_scope(use_cuda=0):
            value = factories.zeros((2, 3), dtype=torch.float32)
        np.testing.assert_array_equal(value.numpy(), np.zeros((2, 3)))

    def test_empty_like_implementation_is_family_owned_and_runs_on_cpu(self):
        factories = importlib.import_module(
            "jittor.compat.torch.installers.factories")
        self.assertEqual(factories.empty_like.__module__, factories.__name__)
        self.assertIs(getattr(torch, "empty_like"), factories.empty_like)
        record = importlib.import_module(
            "jittor.compat.torch.fidelity").fidelity_of("torch.empty_like")
        self.assertIn("device", record.detail)
        with jt.flag_scope(use_cuda=0):
            source = torch.ones((2, 3), dtype=torch.float64)
            value = factories.empty_like(source)
        self.assertEqual(tuple(value.shape), (2, 3))
        self.assertEqual(value.dtype, source.dtype)


if __name__ == "__main__":
    unittest.main()
