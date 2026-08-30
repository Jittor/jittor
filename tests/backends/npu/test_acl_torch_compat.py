import unittest

import numpy as np

import jittor as torch
import jittor as jt


@unittest.skipIf(not jt.compiler.has_acl, "No ACL found")
class TestACLTorchCompat(unittest.TestCase):
    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_empty_native_shapes_stay_on_device(self):
        native_shape = torch.ones((2, 3)).shape
        for shape in ((2, 3), [2, 3], native_shape):
            value = torch.empty(shape)
            value.sync()
            self.assertEqual(tuple(value.shape), (2, 3))
            self.assertTrue(value.is_cuda)
            self.assertEqual(value.location(), "device")

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_empty_cuda_tensor(self):
        device = torch.device("cuda")
        empty = torch.tensor([], dtype=torch.float32, device=device)
        self.assertEqual(empty.numel(), 0)
        self.assertTrue(empty.is_cuda)

        value = torch.tensor([1.0], dtype=torch.float32, device=device)
        joined = torch.cat((empty, value))
        np.testing.assert_array_equal(joined.cpu().numpy(), [1.0])

    def test_default_device_follows_execution_flag(self):
        with jt.flag_scope(use_acl=0, use_cuda=0):
            self.assertEqual(torch.get_default_device().type, "cpu")
        with jt.flag_scope(use_acl=1, use_cuda=1):
            self.assertEqual(torch.get_default_device().type, "cuda")

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_constant_pad_forward_backward_stays_on_acl(self):
        acl_pad = jt.nn._acl_constant_pad
        calls = []

        def record_acl_pad(x, amounts, value):
            calls.append((tuple(amounts), value))
            return acl_pad(x, amounts, value)

        jt.nn._acl_constant_pad = record_acl_pad
        try:
            with jt.log_capture_scope(
                log_v=0, log_vprefix="acl_op_exec.cc=100"
            ) as logs:
                labels = torch.tensor([[1, 2, 3]], dtype=torch.int64)
                shifted = torch.nn.functional.pad(
                    labels, (0, 1), value=-100
                )[:, 1:]

                source = torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
                )
                source.requires_grad_(True)
                padded = torch.nn.functional.pad(
                    source, (1, 2, 2, 1), value=3.5
                )
                weight = torch.arange(
                    padded.numel(), dtype=torch.float32
                ).reshape(padded.shape)
                gradient = torch.autograd.grad(
                    (padded * weight).sum(), source
                )[0]
                self.assertTrue(shifted.is_cuda)
                self.assertTrue(padded.is_cuda)
                self.assertTrue(gradient.is_cuda)
                shifted, padded, gradient = jt.fetch_sync(
                    [shifted, padded, gradient]
                )
        finally:
            jt.nn._acl_constant_pad = acl_pad

        np.testing.assert_array_equal(shifted, [[2, 3, -100]])
        np.testing.assert_array_equal(
            padded,
            [
                [3.5, 3.5, 3.5, 3.5, 3.5],
                [3.5, 3.5, 3.5, 3.5, 3.5],
                [3.5, 1.0, 2.0, 3.5, 3.5],
                [3.5, 3.0, 4.0, 3.5, 3.5],
                [3.5, 3.5, 3.5, 3.5, 3.5],
            ],
        )
        np.testing.assert_array_equal(gradient, [[11.0, 12.0], [16.0, 17.0]])

        self.assertEqual(calls, [((0, 1), -100), ((1, 2, 2, 1), 3.5)])
        messages = [entry["msg"].lower() for entry in logs]
        self.assertFalse(any("compile cpu" in message for message in messages))
        self.assertFalse(any("fallback cpu" in message for message in messages))
