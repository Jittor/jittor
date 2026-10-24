"""Two-rank real-HCCL probe for the FSDP all-reduce/split fallback."""

from datetime import timedelta
import os

import numpy as np

import jittor as jt
import torch
import torch.distributed as dist


rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size == 2

torch.compat_report_torch_api_version(True)
torch.npu.set_device(local_rank)
dist.init_process_group(
    "hccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=120))

from jittor.compat.fsdp2 import common

value = torch.full(
    (1026,), rank + 1, dtype=torch.bfloat16,
    device=torch.device("npu", local_rank))
native = getattr(value, "_var", value)
shard = common._reduce_scatter_padded(native, dist.group.WORLD)
shard.sync()
actual = np.asarray(shard.cast("float32").numpy())
np.testing.assert_array_equal(actual, np.full((513,), 3, dtype=np.float32))
backend, device_id = jt.core.dispatch_context([shard])
assert backend in ("acl", "acl_legacy") and int(device_id) == local_rank
print(
    "HCCL_BF16_PARTITION_OK rank=%d device=%d shape=%s value=%s" %
    (rank, local_rank, tuple(shard.shape), float(actual[0])),
    flush=True,
)
dist.barrier()
dist.destroy_process_group()
