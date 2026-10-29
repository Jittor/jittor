"""Actual NCCL subgroup identity, independent of the launcher agents' disks.

Run under the public launcher with two real CUDA ranks. The same numerical
test also guards the retained single-agent/file subgroup transport.
"""

import os
import sys

import jittor as jt
from jittor import distributed as dist
import numpy as np
import pytest


_READY = bool(jt.compiler.has_cuda and jt.flags.use_cuda and dist.get_world_size() == 2)
_EXTERNAL = os.environ.get('JITTOR_LAUNCHER_EXTERNAL_STORE') == '1'
if os.environ.get('JITTOR_REQUIRE_MULTINODE_CONTROL') == '1' and not (_READY and _EXTERNAL):
    raise RuntimeError('required launcher Store gate needs two real CUDA ranks and external agent Store')
pytestmark = pytest.mark.skipif(not _READY, reason='requires two actual CUDA NCCL ranks')


@jt.flag_scope(use_cuda=1, backend_fallback='error')
def test_reordered_distinct_singleton_groups_keep_world_intact():
    assert 'torch' not in sys.modules
    rank = dist.get_rank()
    before = jt.core.backend_fallback_count()
    groups = [dist.new_group([1, 0]), dist.new_group([1, 0])]
    assert groups[0]._backend_handle != groups[1]._backend_handle
    for group in groups:
        value = jt.array(np.full((32,), rank + 10, dtype='float32')).cuda(0)
        np.testing.assert_array_equal(group.broadcast(value, src=1).numpy(), np.full((32,), 11))
        value = jt.array(np.full((32,), rank + 1, dtype='float32')).cuda(0)
        np.testing.assert_array_equal(group.all_reduce(value, 'sum').numpy(), np.full((32,), 3))
    for peer in range(2):
        group = dist.new_group([peer])
        assert group._backend_handle not in (0, None)
        if rank == peer:
            value = jt.array(np.full((32,), rank + 100, dtype='float32')).cuda(0)
            # Execute NCCL itself: the public singleton shortcut alone cannot
            # prove that an independent native communicator was created.
            output = jt.compile_extern.nccl_ops.nccl_all_reduce(value, group._backend_handle)
            np.testing.assert_array_equal(output.numpy(), np.full((32,), rank + 100))
    value = jt.array(np.full((32,), rank + 1, dtype='float32')).cuda(0)
    np.testing.assert_array_equal(dist.all_reduce(value, op='sum').numpy(), np.full((32,), 3))
    assert jt.core.backend_fallback_count() == before


@pytest.mark.skipif(not _EXTERNAL, reason='requires multi-rank CUDA launcher-owned Store subgroup transport')
def test_group_order_mismatch_refuses_before_nccl_init():
    ranks = [0, 1] if dist.get_rank() == 0 else [1, 0]
    with pytest.raises(ValueError, match='rank order differs'):
        dist.new_group(ranks)
