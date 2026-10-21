"""Public DTensor factories preserve explicit dtype on a singleton CPU mesh."""

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.tensor as dt
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate


@pytest.fixture
def cpu_mesh(tmp_path):
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(
            "gloo", init_method="file://" + str(tmp_path / "rendezvous"),
            rank=0, world_size=1)
    try:
        assert dist.get_world_size() == 1
        yield init_device_mesh("cpu", (1,))
    finally:
        if initialized_here:
            dist.destroy_process_group()


@pytest.mark.parametrize("factory,dtype,expected", [
    ("ones", torch.float64, 1.),
    ("zeros", torch.int64, 0),
    ("full", torch.float16, 1.25),
    ("empty", torch.bfloat16, None),
])
def test_explicit_dtype_factory(factory, dtype, expected, cpu_mesh):
    kwargs = dict(device_mesh=cpu_mesh, placements=[Replicate()], dtype=dtype)
    value = (dt.full((2, 3), expected, **kwargs) if factory == "full"
             else getattr(dt, factory)((2, 3), **kwargs))
    local = value.to_local()
    assert local.dtype == dtype
    assert tuple(local.shape) == (2, 3)
    assert isinstance(local, torch.Tensor)
    assert local.device.type == "cpu"
    if expected is not None:
        np.testing.assert_array_equal(local.numpy(), np.full((2, 3), expected))
