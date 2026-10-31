"""Regression for copy_ across different abstract placements on the CPU."""

import numpy as np
import pytest
import torch


def test_no_grad_copy_follow_runtime_to_explicit_cpu_preserves_alias():
    if not hasattr(torch, "_torch_compat_install_context"):
        pytest.skip("requires the Jittor Torch compatibility runtime")

    import jittor as jt
    from jittor._runtime.fallback import forbid_backend_fallbacks

    with jt.flag_scope(use_cuda=0):
        with forbid_backend_fallbacks():
            native = jt.array(np.asarray([1.0, 2.0], dtype=np.float32))
            target = torch.Tensor(native)
            source = torch.tensor([3.0, 4.0], dtype=torch.float32, device="cpu")
            assert target.device.type == source.device.type == "cpu"
            assert target.placement_backend == -1
            assert source.placement_backend == 0
            pointer = target.untyped_storage().data_ptr()

            with torch.no_grad():
                assert target.copy_(source) is target

            assert target.placement_backend == -1
            assert target.untyped_storage().data_ptr() == pointer
            np.testing.assert_array_equal(target.detach().cpu().numpy(), [3.0, 4.0])
            with torch.no_grad():
                source.add_(10.0)
            np.testing.assert_array_equal(target.detach().cpu().numpy(), [3.0, 4.0])
