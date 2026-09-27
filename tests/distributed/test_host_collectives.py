"""CPU control transport lifetime, without importing or compiling Jittor."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import socket

import pytest

from _helpers.child_process import run_child_script, source_python_dir


_SOURCE = (Path(source_python_dir() or str(Path(__file__).resolve().parents[2] / "python"))
           / "jittor" / "distributed")


def _load_source(name):
    spec = importlib.util.spec_from_file_location(
        "host_transport_test_" + name, _SOURCE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PAIR = r'''
import importlib.util
import json
import os
from pathlib import Path
import time

def load(name):
    path = Path(os.environ["HOST_TRANSPORT_SOURCE"]) / (name + ".py")
    spec = importlib.util.spec_from_file_location("host_transport_child_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

stores = load("store")
hosts = load("host_collectives")
rank = int(os.environ["HOST_TRANSPORT_RANK"])
server_rank = int(os.environ["HOST_TRANSPORT_SERVER"])
ranks = json.loads(os.environ["HOST_TRANSPORT_RANKS"])
store = stores.TCPStore(
    "127.0.0.1", int(os.environ["HOST_TRANSPORT_PORT"]), 2,
    is_master=rank == server_rank, timeout=10)
try:
    # Nested prefixes exercise discovery of the actual server owner.
    group = hosts.HostCollectives(
        stores.PrefixStore("outer/", stores.PrefixStore("inner/", store)),
        ranks, rank)
    if os.environ["HOST_TRANSPORT_CASE"] == "lifetime":
        for step in range(16):
            payload = {"rank": rank, "step": step, "data": "x" * (4096 + rank)}
            observed = group.all_gather_object(payload)
            assert observed == [
                {"rank": r, "step": step, "data": "x" * (4096 + r)}
                for r in ranks]
            if rank == server_rank:
                # The next call may already have its peer's payload, but
                # completed rounds must not accumulate retained store keys.
                assert store.num_keys() <= len(ranks)
    else:
        try:
            group.exchange("broadcast" if rank else "barrier", None)
        except RuntimeError as error:
            assert "order differs between ranks" in str(error), str(error)
        else:
            raise AssertionError("mismatched collective order silently passed")

    original_wait = group.store.wait
    def delayed_wait(keys, timeout=None):
        # A legal scheduling delay in final cleanup. Before the fix, a store
        # owner that was not group-local rank zero returned and exited here,
        # while the delayed cleanup coordinator still needed its TCP server.
        time.sleep(.25)
        return original_wait(keys, timeout)
    group.store.wait = delayed_wait
    group.barrier()
    if rank == server_rank:
        assert store.num_keys() == 0, store.num_keys()
    print("HOST_TRANSPORT_DONE", rank, flush=True)
finally:
    store.close()
'''


def _run_pair(server_rank, ranks, case):
    import json

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    common = {
        "HOST_TRANSPORT_SOURCE": str(_SOURCE),
        "HOST_TRANSPORT_SERVER": str(server_rank),
        "HOST_TRANSPORT_RANKS": json.dumps(ranks),
        "HOST_TRANSPORT_PORT": str(port),
        "HOST_TRANSPORT_CASE": case,
        "CUDA_VISIBLE_DEVICES": "",
        "use_cuda": "0",
        "use_nccl": "0",
        "use_mpi": "0",
    }

    def run_rank(rank):
        return run_child_script(
            _PAIR, env=dict(common, HOST_TRANSPORT_RANK=str(rank)),
            without_torch_mode=True, timeout=30, text=True, merge_stderr=True)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_rank, range(2)))
    for rank, result in enumerate(results):
        assert result.returncode == 0, "rank {}:\n{}".format(rank, result.stdout)
        assert "HOST_TRANSPORT_DONE {}".format(rank) in result.stdout


@pytest.mark.parametrize("server_rank", [0, 1])
@pytest.mark.parametrize("ranks", [(0, 1), (1, 0)], ids=["ordered", "reversed"])
def test_tcp_server_survives_final_barrier_and_releases_payloads(server_rank, ranks):
    _run_pair(server_rank, ranks, "lifetime")


@pytest.mark.parametrize("server_rank", [0, 1])
def test_collective_order_mismatch_fails_on_both_ranks_and_cleans_keys(server_rank):
    _run_pair(server_rank, (1, 0), "order")


def test_closed_group_and_nonmember_fail_before_store_access():
    stores, hosts = _load_source("store"), _load_source("host_collectives")
    store = stores.Store()
    closed = hosts.HostCollectives(store, [0], 0)
    closed.closed = True
    with pytest.raises(RuntimeError, match="has been destroyed"):
        closed.barrier()
    nonmember = hosts.HostCollectives(store, [0], 1)
    with pytest.raises(RuntimeError, match="not a member"):
        nonmember.all_gather_object("payload")
    assert store.num_keys() == 0


def test_singleton_exchange_does_not_leave_rendezvous_payloads():
    stores, hosts = _load_source("store"), _load_source("host_collectives")
    store = stores.Store()
    group = hosts.HostCollectives(stores.PrefixStore("singleton/", store), [7], 7)
    for step in range(16):
        payload = {"step": step, "values": [step, step + 1]}
        assert group.all_gather_object(payload) == [payload]
        group.barrier()
    assert store.num_keys() == 0
