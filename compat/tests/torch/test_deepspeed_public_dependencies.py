"""Public names required by DeepSpeed must advertise their actual capability."""
import numpy as np
import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestDeepSpeedPublicDependencies:
    def test_adam_foreach_option_preserves_group_policy_and_updates(self, device):
        for algorithm in (torch.optim.Adam, torch.optim.AdamW):
            parameter = torch.nn.Parameter(torch.tensor([2.], device=device))
            other = torch.nn.Parameter(torch.tensor([4.], device=device))
            optimizer = algorithm([{"params": [parameter]}, {"params": [other], "foreach": False}],
                                  lr=.1, foreach=None, weight_decay=0)
            assert optimizer.param_groups[0]["foreach"] is None
            assert optimizer.param_groups[1]["foreach"] is False
            (parameter.square() + other.square()).sum().backward()
            optimizer.step()
            np.testing.assert_allclose(parameter.detach().cpu().numpy(), [1.9], rtol=1e-6)
            np.testing.assert_allclose(other.detach().cpu().numpy(), [3.9], rtol=1e-6)
            dense = parameter.grad
            assert dense.is_sparse is False

    def test_missing_native_algorithms_reject_construction(self, device):
        # This is an explicit limitation of the compatibility backend, not an
        # optimizer numerical test. Merely importing these names is no support.
        with pytest.raises(NotImplementedError, match="GradBucket"):
            torch.distributed.GradBucket()


instantiate_device_type_tests(TestDeepSpeedPublicDependencies, globals(), only_for=("cpu", "cuda"))
