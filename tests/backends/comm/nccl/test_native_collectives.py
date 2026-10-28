"""Pure native rapid NCCL sequence regression; run with ``nox -s nccl``."""

import os
import sys
import unittest

import numpy as np
import jittor as jt
from _helpers import capability


@unittest.skipUnless(
    capability.check_accelerator("cuda", backend=jt).enabled and int(jt.world_size) >= 2,
    "requires the multi-rank NCCL nox gate",
)
class TestNativeCollectives(unittest.TestCase):
    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_native_rapid_mixed_collectives_preserve_every_result(self):
        from jittor import distributed as dist

        self.assertNotEqual(os.environ.get("JITTOR_TORCH_SHIM"), "1")
        torch = sys.modules.get("torch")
        self.assertNotIn("_torch_compat_install_context", vars(torch) if torch is not None else {})
        self.assertEqual(dist.get_backend(), "nccl")
        rank, world = int(jt.rank), int(jt.world_size)
        total = world * (world + 1) // 2
        before = jt.core.backend_fallback_count()
        # Rotate the operation at position one, including bf16 and broadcast.
        # Reading each result is part of the failing readback/reuse pattern;
        # no device-wide synchronization is inserted between collectives.
        operations = ("reduce", "bf16", "broadcast", "gather", "scatter")
        for rotation in range(len(operations)):
            order = operations[rotation:] + operations[:rotation]
            for position, operation in enumerate(order * 3):
                offset = rotation + position % 3
                value = jt.array(
                    np.full((8,), rank + 1 + offset, dtype="float32")
                ).cuda(0).stop_grad()
                if operation == "bf16":
                    value = value.cast("bfloat16")
                if operation in ("reduce", "bf16"):
                    output = dist.all_reduce(value, op="sum")
                    expected = np.full((8,), total + world * offset)
                elif operation == "broadcast":
                    source = (rotation + position) % world
                    output = dist.broadcast(value, src=source)
                    expected = np.full((8,), source + 1 + offset)
                elif operation == "gather":
                    output = dist.all_gather(value)
                    expected = np.concatenate([np.full((8,), peer + 1 + offset)
                                               for peer in range(world)])
                else:
                    value = jt.concat([value] * world)
                    output = dist.reduce_scatter(value)
                    expected = np.full((8,), total + world * offset)
                with self.subTest(rank=rank, rotation=rotation,
                                  position=position, operation=operation):
                    # NCCL has no CPU kernel; fallback="error" guards actual
                    # device execution. numpy readback may move host residency.
                    np.testing.assert_array_equal(output.float32().numpy(), expected)
        self.assertEqual(jt.core.backend_fallback_count(), before)
