"""Explicit CPU process groups must preserve their requested backend."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import socket

import pytest

from _helpers.child_process import run_child_script


_CPU_ENV = {
    "JITTOR_TORCH_SHIM": "1",
    "JITTOR_TEST_DEVICES": "cpu",
    "CUDA_VISIBLE_DEVICES": "",
    "use_cuda": "0",
    "use_nccl": "0",
    "use_mpi": "0",
    "JITTOR_TORCH_DISTRIBUTED_AUTO_INIT": "1",
}


def test_host_group_backend_is_advertised():
    result = run_child_script(
        "import torch.distributed as d; assert d.is_gloo_available(); "
        "assert d.is_backend_available('gloo')",
        env=_CPU_ENV, text=True, merge_stderr=True,
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("subgroup", [False, True], ids=["world", "subgroup"])
def test_explicit_gloo_group_keeps_cpu_backend(subgroup):
    """Use a real initialized group, without a mocked backend or communicator."""
    result = run_child_script(
        """
import torch
import torch.distributed as dist

dist.init_process_group(backend="gloo", rank=0, world_size=1,
                        store=dist.TCPStore())
try:
    group = dist.new_group([0], backend="gloo") if SUBGROUP else None
    assert dist.get_backend(group) == "gloo", dist.get_backend(group)
    value = torch.tensor([3, 7], dtype=torch.int64, device="cpu")
    dist.all_reduce(value, group=group)
    assert value.tolist() == [3, 7]
    assert value.device.type == "cpu"
    objects = [None]
    dist.all_gather_object(objects, {"rank": 0}, group=group)
    assert objects == [{"rank": 0}]
    dist.barrier(group=group)
finally:
    dist.destroy_process_group()
""".replace("SUBGROUP", repr(subgroup)),
        env=_CPU_ENV, text=True, merge_stderr=True,
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("subgroup", [False, True], ids=["world", "subgroup"])
def test_host_autograd_all_reduce_rejects_missing_backward(subgroup):
    """A host control exchange must not silently detach a training tensor."""
    result = run_child_script(
        """
import jittor as jt
import pytest
import torch
import torch.distributed as dist

def wrong_native_route(*args, **kwargs):
    raise AssertionError("CPU autograd all_reduce reached the native MPI route")

jt.core.Var.mpi_all_reduce = wrong_native_route
# WORLD needs size > 1 to expose the unqualified native route. An explicit
# singleton subgroup needs no peer to demonstrate the detached host result.
dist.init_process_group(backend="gloo", rank=0, world_size=2,
                        store=dist.TCPStore())
try:
    group = dist.new_group([0], backend="gloo") if SUBGROUP else None
    value = torch.tensor([3.0], device="cpu", requires_grad=True)
    with pytest.raises(NotImplementedError, match="CPU.*autograd"):
        dist.nn.all_reduce(value, group=group)
finally:
    dist.destroy_process_group()
""".replace("SUBGROUP", repr(subgroup)),
        env=_CPU_ENV, text=True, merge_stderr=True,
    )
    assert result.returncode == 0, result.stdout


_TWO_RANK_CPU = r'''
import datetime
import os
import torch
import torch.distributed as dist

def forbidden_accelerator_route(*args, **kwargs):
    raise AssertionError("CPU group attempted a native accelerator collective")

if hasattr(torch, '_torch_compat_install_context'):
    import jittor as jt
    from jittor.compat.torch.installers import distributed as implementation
    implementation._native_all_gather_flat = forbidden_accelerator_route
    jt.core.Var.mpi_all_reduce = forbidden_accelerator_route
    jt.core.Var.mpi_broadcast = forbidden_accelerator_route
rank = int(os.environ["HOST_GROUP_RANK"])
dist.init_process_group(
    backend="gloo", rank=rank, world_size=2,
    init_method=os.environ["HOST_GROUP_INIT_METHOD"],
    timeout=datetime.timedelta(seconds=15),
)
try:
    assert dist.get_backend() == "gloo"
    assert dist.get_rank() == rank
    assert dist.get_world_size() == 2
    group = dist.new_group([0, 1], backend="gloo")
    assert dist.get_backend(group) == "gloo"
    for op, expected in ((dist.ReduceOp.SUM, [3, 7]),
                         (dist.ReduceOp.MAX, [2, 4]),
                         (dist.ReduceOp.MIN, [1, 3])):
        value = torch.tensor([rank + 1, rank + 3], dtype=torch.int64,
                             device="cpu")
        work = dist.all_reduce(value, op=op, group=group, async_op=True)
        work.wait()
        assert value.tolist() == expected, (rank, value.tolist(), expected)
        assert value.device.type == "cpu"

    value = torch.tensor([rank, rank + 2], dtype=torch.int32, device="cpu")
    gathered = [torch.empty((2,), dtype=torch.int32, device="cpu")
                for _ in range(2)]
    dist.all_gather(gathered, value, group=group)
    assert [item.tolist() for item in gathered] == [[0, 2], [1, 3]]
    dist.broadcast(value, src=1, group=group)
    assert value.tolist() == [1, 3]

    objects = [None, None]
    dist.all_gather_object(objects, {"rank": rank, "text": "x" * (rank + 1)},
                           group=group)
    assert objects == [{"rank": 0, "text": "x"}, {"rank": 1, "text": "xx"}]
    broadcast = [{"source": 1}, [4, 5]] if rank == 1 else [None, None]
    dist.broadcast_object_list(broadcast, src=1, group=group)
    assert broadcast == [{"source": 1}, [4, 5]]
    dist.barrier(group=group)

    # Destroying one subgroup must not close WORLD's shared store. Repeating
    # the same ranks must not consume stale collective keys from the old group.
    dist.destroy_process_group(group)
    assert dist.is_initialized()
    world_value = torch.tensor([rank + 5], dtype=torch.int64, device="cpu")
    dist.all_reduce(world_value)
    assert world_value.tolist() == [11]
    replacement = dist.new_group([0, 1], backend="gloo")
    new_value = torch.tensor([rank + 10], dtype=torch.int64, device="cpu")
    dist.all_reduce(new_value, group=replacement)
    assert new_value.tolist() == [21]
    dist.barrier()
    if rank == 0 and hasattr(torch, '_torch_compat_install_context'):
        # Completed exchanges must not retain every serialized request forever.
        assert dist.distributed_c10d._get_default_store().num_keys() == 0
    print("CPU_GROUP_DONE", rank, flush=True)
finally:
    dist.destroy_process_group()
'''


@pytest.mark.parametrize("store_kind", ["file", "tcp"])
def test_two_rank_cpu_transport_and_subgroup_lifetime(tmp_path, store_kind):
    """Exercise actual cross-process CPU data exchange, not a backend label."""
    configured_homes = os.environ.get("JITTOR_HOST_GROUP_TEST_HOMES")
    homes = (json.loads(configured_homes) if configured_homes else
             [str(tmp_path / "rank-0-home"), str(tmp_path / "rank-1-home")])
    assert len(homes) == 2 and homes[0] != homes[1]
    if store_kind == "tcp":
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        init_method = "tcp://127.0.0.1:{}".format(port)
    else:
        init_method = "file://" + str(tmp_path / "rendezvous.sqlite")
    environments = []
    for rank in range(2):
        env = dict(_CPU_ENV, JITTOR_HOME=homes[rank],
                   HOST_GROUP_RANK=str(rank),
                   HOST_GROUP_INIT_METHOD=init_method)
        environments.append(env)
        # Compile each independent rank cache serially before rendezvous.
        warm = run_child_script(
            "import torch\n"
            "x = torch.tensor([1, 2], dtype=torch.int64, device='cpu')\n"
            "assert x.tolist() == [1, 2]\n",
            env=env, text=True, merge_stderr=True,
        )
        assert warm.returncode == 0, warm.stdout

    def run_rank(env):
        return run_child_script(_TWO_RANK_CPU, env=env, text=True,
                                merge_stderr=True, timeout=90)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run_rank, environments))
    failures = ["rank {}:\n{}".format(rank, result.stdout)
                for rank, result in enumerate(results) if result.returncode != 0]
    assert not failures, "\n".join(failures)
    for rank, result in enumerate(results):
        assert "CPU_GROUP_DONE {}".format(rank) in result.stdout
