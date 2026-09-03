import importlib
import unittest

import numpy as np

import jittor as torch


FACTORY_NAMES = (
    "arange", "bernoulli", "empty", "full", "full_like", "linspace",
    "multinomial", "normal", "ones", "ones_like", "rand", "rand_like",
    "randint", "randn", "randn_like", "randperm", "tril", "triu", "zeros",
    "zeros_like",
)


class TestTorchFactoryFidelity(unittest.TestCase):

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
            if record.api.split(".")[-1] in FACTORY_NAMES
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
        with torch.flag_scope(use_cuda=0):
            value = factories.zeros((2, 3), dtype=torch.float32)
        np.testing.assert_array_equal(value.numpy(), np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
