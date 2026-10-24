"""Module keyword device moves parameters and both persistent buffer kinds.

Run the same file in a separate native Torch interpreter for the oracle.
Default CPU; explicitly request npu/cuda and an ordinal for accelerator proof.
"""
from contextlib import ExitStack
import os
import unittest

import numpy as np
import torch

ORACLE = os.environ.get("JITTOR_MODULE_DEVICE_ORACLE") == "1"
if ORACLE:
    if os.environ.get("JITTOR_TEST_DEVICES") == "npu":
        import torch_npu  # noqa: F401
    jt = None
else:
    import jittor as jt


class TestModuleKeywordDevice(unittest.TestCase):
    def setUp(self):
        self.kind = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
        self.assertIn(self.kind, ("cpu", "cuda", "npu"))
        self.ordinal = int(os.environ.get("JITTOR_TEST_DEVICE_INDEX", "0"))
        stack = ExitStack()
        self.addCleanup(stack.close)
        if self.kind != "cpu":
            backend = getattr(torch, self.kind)
            self.assertTrue(backend.is_available(), "requested accelerator unavailable")
            backend.set_device(self.ordinal)
        if jt is not None:
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(jt.runtime.scope(use_cuda=int(self.kind != "cpu")))
            stack.enter_context(forbid_backend_fallbacks())
            before = jt.core.backend_fallback_count()
            self.addCleanup(lambda: self.assertEqual(jt.core.backend_fallback_count() - before, 0))
        else:
            self.assertFalse(hasattr(torch, "_torch_compat_install_context"))

    def assert_placement(self, value):
        if jt is not None:
            value.sync()
            self.assertEqual(value.location(), "cpu" if self.kind == "cpu" else "device")
            if self.kind != "cpu":
                self.assertEqual(value.device_id, self.ordinal)
        self.assertEqual(value.device.type, self.kind)
        if self.kind != "cpu":
            self.assertEqual(value.device.index, self.ordinal)

    @staticmethod
    def model():
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([2., 3.], device="cpu"))
                self.register_buffer("persistent", torch.tensor([4., 5.], device="cpu"))
                self.register_buffer("temporary", torch.tensor([6., 7.], device="cpu"), persistent=False)

            def forward(self, x):
                return (x * self.weight + self.persistent + self.temporary).sum()
        return Model()

    def targets(self):
        if self.kind == "cpu":
            return ["cpu", torch.device("cpu")]
        return [self.ordinal, np.int64(self.ordinal), self.kind + ":" + str(self.ordinal),
                torch.device(self.kind, self.ordinal)]

    def test_keyword_to_moves_all_values_and_preserves_parameter_identity(self):
        for target in self.targets():
            with self.subTest(target=target):
                model = self.model()
                parameter = model.weight
                self.assertIs(model.to(device=target, non_blocking=True), model)
                self.assertIs(model.weight, parameter)
                for value in (model.weight, model.persistent, model.temporary):
                    self.assert_placement(value)
                self.assertEqual(set(model.state_dict()), {"weight", "persistent"})
                x = torch.tensor([1., 2.], device=model.weight.device, requires_grad=True)
                loss = model(x)
                loss.backward()
                for value in (loss, x.grad, model.weight.grad):
                    self.assert_placement(value)
                np.testing.assert_array_equal(model.weight.detach().cpu().numpy(), [2., 3.])
                np.testing.assert_array_equal(x.grad.detach().cpu().numpy(), [2., 3.])
                np.testing.assert_array_equal(model.weight.grad.detach().cpu().numpy(), [1., 2.])
                self.assertEqual(float(loss.detach().cpu()), 30.)

    def test_to_empty_keyword_moves_buffers_without_asserting_undefined_contents(self):
        for target in self.targets():
            with self.subTest(target=target):
                model = self.model()
                self.assertIs(model.to_empty(device=target), model)
                for value in (model.weight, model.persistent, model.temporary):
                    self.assert_placement(value)
                self.assertEqual(set(model.state_dict()), {"weight", "persistent"})

    def test_invalid_keyword_ordinal_is_not_silently_ignored(self):
        for target in (True, False, -1, np.int64(-2)):
            with self.subTest(target=target), self.assertRaises((TypeError, RuntimeError)):
                self.model().to(device=target)


if __name__ == "__main__":
    unittest.main()
