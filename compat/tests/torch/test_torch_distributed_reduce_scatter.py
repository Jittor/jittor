"""The Torch reduce-scatter facade writes into its caller-owned output."""

import numpy as np
import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestReduceScatter:
    def test_single_rank_sum_writes_output(self, device):
        assert torch.distributed.get_world_size() == 1
        source = torch.tensor([2., 4., 6.], device=device)
        output = torch.zeros(3, device=device)
        assert torch.distributed.reduce_scatter(output, [source]) is None
        np.testing.assert_array_equal(output.cpu().numpy(), [2., 4., 6.])
        output.zero_()
        assert torch.distributed.reduce_scatter_tensor(output, source) is None
        np.testing.assert_array_equal(output.cpu().numpy(), [2., 4., 6.])

    def test_bad_partition_shape_rejected(self, device):
        output = torch.zeros(2, device=device)
        with pytest.raises(ValueError, match="one input"):
            torch.distributed.reduce_scatter(output, [])
        with pytest.raises(ValueError, match="input size"):
            torch.distributed.reduce_scatter_tensor(
                output, torch.ones(3, device=device))


instantiate_device_type_tests(TestReduceScatter, globals(), only_for=("cpu", "cuda"))
