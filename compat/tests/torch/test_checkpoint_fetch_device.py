"""Checkpoint host snapshots must not route nonzero-rank tensors through card zero."""
import io
import os
import unittest

import numpy as np
import jittor as jt
import torch

from jittor._runtime.fallback import forbid_backend_fallbacks


class TestCheckpointFetchDevice(unittest.TestCase):
    def test_fetch_and_save_preserve_source_device_and_values(self):
        kind = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
        ordinal = int(os.environ.get("JITTOR_TEST_DEVICE_INDEX", "0"))
        self.assertIn(kind, ("cpu", "npu", "cuda"))
        target = kind if kind == "cpu" else "%s:%d" % (kind, ordinal)
        if kind != "cpu":
            self.assertTrue(getattr(torch, kind).is_available())
            getattr(torch, kind).set_device(ordinal)
        with forbid_backend_fallbacks():
            before = jt.core.backend_fallback_count()
            x = torch.tensor([1., 2., 3.], device=target, requires_grad=True)
            y = x * 2
            y.sync()
            calls = []
            jt.fetch(y, lambda value: calls.append(np.array(value, copy=True)))
            jt.sync_all(True)
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(calls[0], [2., 4., 6.])
            payload = {"value": y, "empty": torch.empty(0, dtype=torch.float32, device=target),
                       "integer": torch.tensor([2, 5], dtype=torch.int64, device=target)}
            stream = io.BytesIO()
            torch.save(payload, stream)
            stream.seek(0)
            loaded = torch.load(stream, map_location="cpu", weights_only=False)
            for name, expected in (("value", [2., 4., 6.]), ("empty", []), ("integer", [2, 5])):
                np.testing.assert_array_equal(loaded[name].numpy(), expected)
                self.assertEqual(loaded[name].dtype, payload[name].dtype)
            y.sum().backward()
            np.testing.assert_array_equal(x.grad.detach().cpu().numpy(), [2., 2., 2.])
            for value in (x, y, payload["integer"], x.grad):
                value.sync()
                self.assertEqual(value.location(), "cpu" if kind == "cpu" else "device")
                if kind != "cpu":
                    self.assertEqual(value.device_id, ordinal)
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)


if __name__ == "__main__":
    unittest.main()
