"""Real TensorDict keeps jagged payloads and their differentiable graph."""
import importlib.util
import os
import unittest
import numpy as np
import jittor as jt
import torch
from jittor._runtime.fallback import forbid_backend_fallbacks


_HAS_TENSORDICT = importlib.util.find_spec("tensordict") is not None
if not _HAS_TENSORDICT and (os.environ.get("JITTOR_REQUIRE_TENSORDICT") == "1"
                            or os.environ.get("JITTOR_REQUIRE_OPTIONAL_DEPS") == "1"):
    raise ModuleNotFoundError("tensordict is required for nested integration verification")
if _HAS_TENSORDICT:
    from tensordict import TensorDict


@unittest.skipUnless(_HAS_TENSORDICT, "tensordict is not installed")
class TestTensorDictNested(unittest.TestCase):
    def exercise(self, device):
        raw = np.arange(10, dtype=np.float32).reshape(5, 2)
        with forbid_backend_fallbacks():
            source = torch.tensor(raw, device=device, requires_grad=True)
            nested = torch.nested.nested_tensor_from_jagged(
                source, torch.tensor([0, 2, 5], device=device), jagged_dim=1)
            data = TensorDict({"value": nested}, batch_size=[2])
            self.assertIs(data["value"], nested)
            self.assertTrue(data["value"].is_nested)
            cloned = data.clone()
            self.assertIsNot(cloned["value"], nested)
            moved = cloned.to(device)
            self.assertTrue(moved["value"].is_nested)
            self.assertIsInstance(moved["value"].offsets(), torch.Tensor)
            np.testing.assert_array_equal(moved["value"].offsets().diff().detach().cpu().numpy(), [2, 3])
            np.testing.assert_array_equal(moved["value"].offsets().detach().cpu().numpy(), [0, 2, 5])
            chunks = moved.chunk(2)
            self.assertEqual(len(chunks), 2)
            joined = torch.cat([chunk["value"].values() for chunk in chunks])
            joined.sync()
            self.assertEqual(joined.location(), "device" if device == "npu" else "cpu")
            gradient, = torch.autograd.grad((joined * torch.tensor(raw + 1, device=device)).sum(), (source,))
            np.testing.assert_array_equal(joined.detach().cpu().numpy(), raw)
            np.testing.assert_array_equal(gradient.detach().cpu().numpy(), raw + 1)
            host = moved.cpu()
            self.assertTrue(host["value"].is_nested)
            self.assertEqual(host["value"].device.type, "cpu")
            self.assertEqual(host["value"].offsets().device.type, "cpu")

    def test_cpu_identity_chunk_clone_and_gradient(self):
        with jt.runtime.scope(use_cuda=0):
            self.exercise("cpu")

    def test_npu_identity_chunk_clone_and_gradient(self):
        if not getattr(jt.compiler, "has_acl", False):
            self.skipTest("requires real ACL NPU")
        self.exercise("npu")

    def test_cpu_to_npu_moves_every_leaf(self):
        if not getattr(jt.compiler, "has_acl", False):
            self.skipTest("requires real ACL NPU")
        with forbid_backend_fallbacks():
            source = torch.tensor([1., 2., 3.], device="cpu", requires_grad=True)
            nested = torch.nested.nested_tensor_from_jagged(
                source, torch.tensor([0, 1, 3], device="cpu", dtype=torch.int64))
            data = TensorDict({"ragged": nested,
                               "dense": torch.tensor([7, 8], device="cpu")}, batch_size=[2])
            for target in ("npu:0", 0):
                moved = data.to(target)
                for value in (moved["ragged"].values(), moved["ragged"].offsets(),
                              moved["dense"], moved["ragged"].to_padded_tensor(0)):
                    value.sync()
                    self.assertEqual(value.location(), "device", (target, value.device))
                self.assertEqual(source.device.type, "cpu")
                self.assertEqual(nested.device.type, "cpu")
                np.testing.assert_array_equal(moved["ragged"].values().detach().cpu().numpy(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
