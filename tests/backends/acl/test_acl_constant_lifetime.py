"""Queued ACL copies must retain force-fused rank-1 constants' host bytes."""

import numpy as np
import pytest
import jittor as jt

from _helpers import capability as _test_capability
from jittor._runtime.fallback import forbid_backend_fallbacks


def test_rank_one_advanced_indices_survive_a_queued_stream_and_host_allocation_reuse():
    if not _test_capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required for queued constant lifetime")
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_acl=1, use_cuda=1), jt.no_grad(), forbid_backend_fallbacks():
        values = np.broadcast_to(
            np.arange(6, dtype=np.float32).reshape(1, 1, 6, 1),
            (1, 800, 6, 3072),
        ).copy()
        source = jt.array(values)
        matrix = jt.array(np.full((3072, 3072), 1 / 3072, dtype=np.float16))
        jt.sync([source, matrix], True)
        for repeat in range(3):
            # Keep the stream occupied while the executor enqueues tiny H2D
            # copies and releases their ArrayOps. No sleep or timing assertion:
            # the following host allocations expose an invalid source lifetime.
            blocker = matrix
            for _ in range(12):
                blocker = jt.matmul(blocker, matrix)
            blocker.sync()
            # Keep explicit rank-1 advanced indices here even when chunk uses
            # basic views: the constant lifetime regression is independent of
            # which partition implementation a model selects.
            parts = [source[:, :, [index], :] for index in range(6)]
            squeezed = [part.squeeze(2) for part in parts]
            result = (1 + squeezed[1]) * 0.5 + squeezed[0]
            result.sync()
            churn = [jt.array(np.array([100 + index], dtype=np.int32)) for index in range(256)]
            assert result.location() == "device"
            actual = result.numpy()
            np.testing.assert_array_equal(actual, np.ones((1, 800, 3072), dtype=np.float32),
                                          err_msg="rank-1 chunk index changed during queued H2D copy")
            assert len(churn) == 256
            del churn, result, squeezed, parts, blocker
    assert jt.core.backend_fallback_count() == before
