"""``torch.cuda.memory._record_memory_history`` / ``_snapshot`` /
``_dump_snapshot`` and ``torch.cuda.memory_snapshot`` write PyTorch's snapshot.

They used to be missing, so a training script that dumps a snapshot for
https://pytorch.org/memory_viz stopped with an AttributeError. The keys below
are the ones PyTorch 2.11 writes; memory_viz reads nothing else.
"""

import os
import pickle
import tempfile
import unittest

import torch

from _helpers import capability as _test_capability

import jittor as jt

HAS_CUDA = bool(_test_capability.check_accelerator("cuda", backend=jt).enabled)

SNAPSHOT_KEYS = ["segments", "device_traces", "allocator_settings", "external_annotations"]
TRACE_KEYS = ["action", "addr", "size", "stream", "time_us", "compile_context", "user_metadata",
              "frames"]


@unittest.skipIf(not HAS_CUDA, "needs a CUDA device: the pools under test are the device's")
class TestTorchMemorySnapshot(unittest.TestCase):
    def tearDown(self):
        torch.cuda.memory._record_memory_history(enabled=None)

    def _train(self):
        model = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(),
                                    torch.nn.Linear(512, 10)).cuda()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(2):
            loss = model(torch.randn(64, 256, device="cuda")).square().mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()
        return model

    def test_a_training_loop_dumps_a_snapshot_memory_viz_can_read(self):
        torch.cuda.memory._record_memory_history(max_entries=100000)
        model = self._train()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snapshot.pickle")
            torch.cuda.memory._dump_snapshot(path)
            with open(path, "rb") as f:
                data = pickle.load(f)
        self.assertEqual(list(data), SNAPSHOT_KEYS)
        trace = data["device_traces"][torch.cuda.current_device()]
        self.assertEqual(list(trace[0]), TRACE_KEYS)
        # segment_alloc too, unless earlier work left enough cached segments
        # (tests/runtime/test_memory_history.py checks segments directly).
        self.assertTrue({"alloc", "free_requested", "free_completed"} <= {e["action"] for e in trace})
        # The weights' blocks are named after the line that built the model.
        weight = model[0].weight.numel() * 4
        blocks = [b for s in data["segments"] for b in s["blocks"]
                  if b["state"] == "active_allocated" and b["requested_size"] == weight]
        self.assertTrue(blocks)
        self.assertTrue(any(f["filename"] == __file__ and f["name"] == "_train"
                            for f in blocks[0]["frames"]), blocks[0]["frames"][:8])

    def test_memory_snapshot_is_the_segments(self):
        torch.cuda.memory._record_memory_history()
        x = torch.randn(1000, 1000, device="cuda")
        segments = torch.cuda.memory_snapshot()
        self.assertIsInstance(segments, list)
        self.assertEqual(sum(s["allocated_size"] for s in segments), torch.cuda.memory_allocated())
        self.assertIs(torch.cuda.memory.memory_snapshot, torch.cuda.memory_snapshot)
        del x

    def test_the_boolean_form_still_records(self):
        torch.cuda.memory._record_memory_history(True, trace_alloc_max_entries=1000,
                                                 trace_alloc_record_context=True)
        torch.randn(100, device="cuda").sum().item()
        trace = torch.cuda.memory._snapshot()["device_traces"][torch.cuda.current_device()]
        self.assertTrue(trace)
        self.assertTrue(any(e["frames"] for e in trace))
        torch.cuda.memory._record_memory_history(False)
        self.assertEqual(torch.cuda.memory._snapshot()["device_traces"][torch.cuda.current_device()], [])


if __name__ == "__main__":
    unittest.main()
