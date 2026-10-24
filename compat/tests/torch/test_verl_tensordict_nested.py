"""Real verl 0.9.1 five-helper integration, independently on CPU and NPU.

Requires the normal optional verl/TensorDict adapters to be installed by the
Torch frontend. This test never patches helpers, pretends Nested is Tensor, or
wraps payloads as metadata. Fixed NumPy arrays are observations/oracles only.
Run this file directly with unittest to avoid unrelated pytest collection.
"""
import importlib.metadata
import importlib.util
import os
import unittest

import numpy as np


_HAS_VERL = importlib.util.find_spec("verl") is not None
if os.environ.get("JITTOR_REQUIRE_VERL") == "1" and not _HAS_VERL:
    raise ModuleNotFoundError("JITTOR_REQUIRE_VERL=1 but verl is not installed")


@unittest.skipUnless(_HAS_VERL, "verl is not installed")
class TestVerlTensorDictNested(unittest.TestCase):
    def setUp(self):
        import jittor as jt
        import torch
        import verl
        from jittor.compat.shim.runtime import activation_status
        from jittor.compat.torch import nested_tensor_type
        from jittor.compat.torch.publication import namespace_owner
        from verl.utils import tensordict_utils as tu

        self.assertIs(namespace_owner(torch), jt)
        self.assertTrue(activation_status(jt).active)
        self.assertIsNot(torch.Tensor, jt.Var)
        self.assertEqual(verl.__version__, "0.9.1")
        self.assertEqual(importlib.metadata.version("verl"), "0.9.1")
        self.jt, self.torch, self.tu = jt, torch, tu
        self.nested_type = nested_tensor_type()

    def _placement(self, value, device, index):
        value.sync()
        self.assertEqual(value.location(), "device" if device == "npu" else "cpu")
        self.assertEqual(value.device.type, device)
        if device == "npu":
            self.assertEqual(value.get_device(), index)

    @staticmethod
    def _snapshot(value):
        return np.array(value.detach().cpu().numpy(), copy=True)

    def _nested(self, value, expected_values, expected_offsets, device, index):
        self.assertIsInstance(value, self.nested_type)
        self.assertNotIsInstance(value, self.torch.Tensor)
        self.assertTrue(value.is_nested)
        self.assertTrue(value.is_contiguous())
        self.assertTrue(value.requires_grad)
        self.assertEqual(value.layout, self.torch.jagged)
        self.assertEqual(value.device.type, device)
        values, offsets = value.values(), value.offsets()
        self._placement(values, device, index)
        self._placement(offsets, device, index)
        self.assertEqual(values.dtype, self.torch.float32)
        self.assertEqual(offsets.dtype, self.torch.int64)
        np.testing.assert_array_equal(self._snapshot(values), np.asarray(expected_values, dtype=np.float32))
        np.testing.assert_array_equal(self._snapshot(offsets), np.asarray(expected_offsets, dtype=np.int64))

    def _exercise(self, device):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        jt, torch, tu = self.jt, self.torch, self.tu
        index = jt.core.current_device() if device == "npu" else None
        target = "npu:%d" % index if device == "npu" else "cpu"
        fallback_before = jt.core.backend_fallback_count()
        with forbid_backend_fallbacks():
            source = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.], device=target, requires_grad=True)
            offsets = torch.tensor([0, 2, 3, 6, 8], device=target, dtype=torch.int64)
            nested = torch.nested.nested_tensor_from_jagged(source, offsets, jagged_dim=1)
            dense = torch.tensor([10, 20, 30, 40], device=target, dtype=torch.int64)

            batch = tu.get_tensordict(
                {"tokens": nested, "sample_id": dense, "labels": ["a", "b", "c", "d"]},
                {"fixture": "five-helper-gradient"},
            )
            self.assertEqual(tuple(batch.batch_size), (4,))
            self.assertIs(tu.get(batch, "tokens"), nested)
            self.assertEqual(tu.get(batch, "labels"), ["a", "b", "c", "d"])
            self.assertEqual(tu.get(batch, "fixture"), "five-helper-gradient")
            self._nested(tu.get(batch, "tokens"), [1, 2, 3, 4, 5, 6, 7, 8],
                         [0, 2, 3, 6, 8], device, index)

            chunks = tu.chunk_tensordict(batch, 2)
            self.assertEqual(len(chunks), 2)
            self.assertEqual([tuple(part.batch_size) for part in chunks], [(2,), (2,)])
            self._nested(tu.get(chunks[0], "tokens"), [1, 2, 3], [0, 2, 3], device, index)
            self._nested(tu.get(chunks[1], "tokens"), [4, 5, 6, 7, 8], [0, 3, 5], device, index)

            # Reverse the two chunks, then select/repeat rows. This exercises
            # ragged lengths, order and accumulation rather than an identity.
            joined = tu.concat_tensordict([chunks[1], chunks[0]])
            self.assertEqual(tuple(joined.batch_size), (4,))
            self._nested(tu.get(joined, "tokens"), [4, 5, 6, 7, 8, 1, 2, 3],
                         [0, 3, 5, 7, 8], device, index)
            self._placement(tu.get(joined, "sample_id"), device, index)
            np.testing.assert_array_equal(self._snapshot(tu.get(joined, "sample_id")), [30, 40, 10, 20])
            self.assertEqual(tu.get(joined, "labels"), ["c", "d", "a", "b"])

            indices = torch.tensor([2, 0, 3, 1, 0], device=target, dtype=torch.int64)
            selected = tu.index_select_tensor_dict(joined, indices)
            self.assertEqual(tuple(selected.batch_size), (5,))
            selected_nested = tu.get(selected, "tokens")
            self._nested(selected_nested, [1, 2, 4, 5, 6, 3, 7, 8, 4, 5, 6],
                         [0, 2, 5, 6, 8, 11], device, index)
            self._placement(tu.get(selected, "sample_id"), device, index)
            np.testing.assert_array_equal(self._snapshot(tu.get(selected, "sample_id")), [10, 30, 20, 40, 30])
            self.assertEqual(tu.get(selected, "labels"), ["a", "c", "b", "d", "c"])
            self.assertEqual(tu.get(selected, "fixture"), "five-helper-gradient")

            weights = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.], device=target)
            loss = (selected_nested.values() * weights).sum()
            self._placement(loss, device, index)
            gradients = torch.autograd.grad(loss, (source,))
            self.assertEqual(len(gradients), 1)
            gradient = gradients[0]
            self.assertIsNotNone(gradient)
            self.assertEqual(tuple(gradient.shape), (8,))
            self._placement(source, device, index)
            self._placement(gradient, device, index)
            # Source row [4,5,6] appears twice, so its cotangents add.
            expected = np.asarray([1., 2., 6., 12., 14., 16., 7., 8.], dtype=np.float32)
            actual = self._snapshot(gradient)
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            jt.sync_all(True)
            self.assertEqual(jt.core.backend_fallback_count(), fallback_before)

    def test_cpu_five_helpers_preserve_nested_and_packed_gradient(self):
        with self.jt.runtime.scope(use_cuda=0):
            self._exercise("cpu")

    def test_npu_five_helpers_preserve_nested_and_packed_gradient(self):
        if not getattr(self.jt.compiler, "has_acl", False):
            self.skipTest("requires the ACL backend and real NPU")
        # An installed but broken ACL backend must error, not become a skip.
        self.assertGreater(self.jt.core.get_device_count(), 0)
        with self.jt.runtime.scope(use_acl=1, use_cuda=1):
            self._exercise("npu")


if __name__ == "__main__":
    unittest.main()
