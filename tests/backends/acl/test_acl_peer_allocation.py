"""Run in separate processes with JT_ACL_DISABLE_P2P unset and set to 1."""
import numpy as np
import pytest
import jittor as jt
from _helpers import capability


@pytest.mark.parametrize('mib', [64, 256])
def test_peer_copy_after_device_allocation(mib):
    if not capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip('real Ascend devices required')
    if capability.device_count('acl', backend=jt) < 2:
        pytest.skip('two real Ascend devices required')
    with jt.flag_scope(use_acl=1, use_cuda=1, device_id=0), jt.no_grad():
        # Activate the peer before allocating the bulk source, so the normal
        # run exercises P2P policy and the forced run ordinary HBM policy.
        jt.ones(1).to_device(1).sync()
        for iteration in range(2):
            source = jt.ones(mib * 1024 * 1024 // 4) * (iteration + 1.5)
            source.sync()
            remote = source.to_device(1).sync()
            returned = remote.to_device(0).sync()
            assert source.location() == remote.location() == returned.location() == 'device'
            assert remote.device_id == 1 and returned.device_id == 0
            np.testing.assert_array_equal(remote.numpy(), np.full(source.shape, iteration + 1.5))
            np.testing.assert_array_equal(returned.numpy(), np.full(source.shape, iteration + 1.5))
            del source, remote, returned
            jt.sync_all(True)
            jt.gc()
