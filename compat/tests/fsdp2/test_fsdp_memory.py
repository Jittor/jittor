"""Same-model CUDA peak/trajectory probe; run full and shard in fresh ranks.

JITTOR_FSDP2_MEMORY_MODE selects full or shard. Compare the printed native
profiler peaks from identical invocations; CUDA facade peak sampling is not
an allocator high-water mark and is intentionally not used here.
"""

from _helpers import capability as _test_capability

import gc
import json
import os
import unittest

import jittor as jt


@unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled and int(jt.world_size) >= 2,
                     "requires a multi-rank CUDA launch")
class TestFsdpMemory(unittest.TestCase):
    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, profile_memory_enable=2)
    def test_same_model_memory_trajectory(self):
        import torch
        from jittor.compat import fsdp2

        mode = os.environ.get("JITTOR_FSDP2_MEMORY_MODE", "shard")
        self.assertIn(mode, ("full", "shard"))
        jt.seed(912)
        width, depth = 512, 4
        model = torch.nn.Sequential(*[torch.nn.Linear(width, width) for _ in range(depth)])
        if mode == "shard":
            for layer in model.children():
                fsdp2.fully_shard(layer)
            fsdp2.fully_shard(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs = torch.ones((2, width), dtype=torch.float32, device="cuda")
        trajectory = []
        for step in range(5):
            optimizer.zero_grad()
            output = model(inputs)
            loss = (output ** 2).mean()
            loss.backward()
            optimizer.step()
            jt.sync_all(True)
            del loss, output
            # Do not use gc.collect inside steps: orphaned shard views must
            # drop promptly, not only under forced Python cyclic collection.
            trajectory.append(int(jt.get_mem_info().total_cuda_used))
        memory_info = jt.get_max_memory_info()
        peak = int(memory_info.split("[!@#div1!@#]", 1)[0])
        used_peak = int(jt.core.get_peak_allocator_used_memory())
        parameter_bytes = depth * (width * width + width) * 4
        self.assertLessEqual(max(trajectory[2:]) - min(trajectory[2:]), parameter_bytes // 8)
        self.assertGreater(peak, 0)
        print("FSDP_MEMORY " + json.dumps({
            "mode": mode, "rank": int(jt.rank), "width": width, "depth": depth,
            "world_size": int(jt.world_size), "peak_cuda_var_bytes": peak,
            "peak_allocator_used_bytes": used_peak,
            "reserved_cuda_bytes": trajectory,
        }), flush=True)
        if os.environ.get("JITTOR_FSDP2_MEMORY_DETAILS") == "1":
            print("FSDP_ALLOCATIONS " + json.dumps({
                "mode": mode, "rank": int(jt.rank),
                "allocations": [item.split("[!@#div2!@#]")[:2]
                                for item in memory_info.split("[!@#div1!@#]")[1:]],
            }), flush=True)
        reference_peak = os.environ.get("JITTOR_FSDP2_REFERENCE_PEAK_BYTES")
        if mode == "shard" and reference_peak is not None:
            self.assertLess(used_peak, int(reference_peak),
                            "FSDP allocator peak must be below the same-model full reference")
        del optimizer, model, inputs
        gc.collect()


if __name__ == "__main__":
    unittest.main()
