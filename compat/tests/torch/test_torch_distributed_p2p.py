"""torch.distributed point-to-point communication through the Jittor shim."""

import os

import numpy as np
import pytest


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    os.environ.get("JT_NCCL_WORLD_SIZE") != "2",
    reason="requires real two-rank NCCL",
)]


def _payload(rank):
    return np.arange(4, dtype="float32") + 10 * rank


def _device():
    import torch
    return torch.device("cuda:0")


def test_blocking_send_recv_matches_the_upstream_example():
    """The exact call shape of transformers' distributed_training.py."""
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    assert dist.get_backend() == "nccl"
    tensor = torch.zeros(4).to(_device())

    if rank == 0:
        value = torch.tensor(_payload(0)).to(_device())
        assert dist.send(tensor=value, dst=1) is None
    else:
        assert dist.recv(tensor=tensor, src=0) is None
        np.testing.assert_array_equal(tensor.cpu().numpy(), _payload(0))
    dist.barrier()

    if rank == 1:
        value = torch.tensor(_payload(1)).to(_device())
        assert dist.send(tensor=value, dst=0) is None
    else:
        assert dist.recv(tensor=tensor, src=1) is None
        np.testing.assert_array_equal(tensor.cpu().numpy(), _payload(1))
    dist.barrier()


def test_isend_irecv_return_waitable_work():
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    for source, destination in ((0, 1), (1, 0)):
        if rank == source:
            value = torch.tensor(_payload(source)).to(_device())
            work = dist.isend(value, dst=destination)
            work.wait()
        elif rank == destination:
            buffer = torch.zeros(4).to(_device())
            work = dist.irecv(buffer, src=source)
            work.wait()
            np.testing.assert_array_equal(buffer.cpu().numpy(), _payload(source))
        dist.barrier()


def test_batch_isend_irecv_exchanges_both_directions_at_once():
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    peer = 1 - rank
    outgoing = torch.tensor(_payload(rank)).to(_device())
    incoming = torch.zeros(4).to(_device())
    works = dist.batch_isend_irecv([
        dist.P2POp(dist.isend, outgoing, peer),
        dist.P2POp(dist.irecv, incoming, peer),
    ])
    assert len(works) == 2
    for work in works:
        work.wait()
    np.testing.assert_array_equal(incoming.cpu().numpy(), _payload(peer))
    dist.barrier()


def test_point_to_point_rejects_host_tensors():
    import torch
    import torch.distributed as dist

    with pytest.raises(ValueError, match="CUDA"):
        dist.send(torch.zeros(4), dst=1 - dist.get_rank())
