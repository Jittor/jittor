"""Cross-process rendezvous stores used by torch.distributed compatibility."""

import importlib.util
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import unittest

from _helpers.child_process import (
    PYTHON,
    child_env,
    default_timeout,
    run_python_child,
    source_python_dir,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
# ``jittor/distributed/store.py`` imports only the standard library, so the
# rendezvous tests can load it straight off disk. That keeps them free of a
# JIT core build, which would otherwise dominate their runtime.
STORE_SOURCE = (
    Path(source_python_dir() or str(REPO_ROOT / "python"))
    / "jittor" / "distributed" / "store.py"
)
_BASE_ENV = {
    "JITTOR_TORCH_SHIM": "1",
    "JITTOR_TEST_DEVICES": "cpu",
    "nvcc_path": "",
    "use_cuda": "0",
    "use_mpi": "0",
    "use_nccl": "0",
}


_DIRECT_STORE = r"""
import datetime
import os
import jittor
import torch.distributed as dist

rank = int(os.environ["STORE_RANK"])
kind = os.environ["STORE_KIND"]
timeout = datetime.timedelta(seconds=10)
if kind == "tcp":
    store = dist.TCPStore(
        "127.0.0.1", int(os.environ["STORE_PORT"]), 2, rank == 0,
        timeout=timeout,
    )
else:
    store = dist.FileStore(os.environ["STORE_PATH"], 2, timeout=timeout)

if rank == 0:
    store.set("payload", b"from-rank-0")
    assert store.get("reply") == b"from-rank-1"
else:
    assert store.get("payload") == b"from-rank-0"
    store.set("reply", b"from-rank-1")
assert store.add("arrivals", 1) in (1, 2)
store.set("arrived/{}".format(rank), b"1")
store.wait(["arrived/0", "arrived/1"])
assert int(store.get("arrivals")) == 2
store.wait(["payload", "reply"])
print("DONE", rank, kind, flush=True)
"""


_INIT_METHOD_STORE = r"""
import datetime
import os
import jittor
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from jittor.distributed import process_group as implementation

rank = int(os.environ["STORE_RANK"])
implementation.is_initialized = lambda: True
implementation.get_rank = lambda: rank
implementation.get_world_size = lambda: 2
dist.init_process_group(
    backend="mpi",
    init_method=os.environ["STORE_INIT_METHOD"],
    timeout=datetime.timedelta(seconds=10),
)
store = c10d._get_default_store()
assert store is not None
if rank == 0:
    store.set("configured", b"through-init-process-group")
else:
    assert store.get("configured") == b"through-init-process-group"
    store.set("ack", b"1")
if rank == 0:
    assert store.get("ack") == b"1"
print("DONE", rank, os.environ["STORE_INIT_METHOD"], flush=True)
"""


class TestProcessGroupOwnership(unittest.TestCase):
    def test_native_and_torch_share_classes_and_legacy_pickle(self):
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            self.skipTest("the child imports the real torch.distributed")
        source = r'''
import importlib
import pickle
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

native = importlib.import_module("jittor.distributed.process_group")
legacy = importlib.import_module("jittor.compat.torch.installers.distributed")
assert dist.ProcessGroup is native.ProcessGroup
assert c10d.ProcessGroup is native.ProcessGroup
assert c10d.Work is native.Work
assert legacy._JittorProcessGroup is native.ProcessGroup
assert legacy._JittorWork is native.Work
assert native.ProcessGroup.__module__ == "jittor.distributed.process_group"
for old_name, cls in (("_JittorProcessGroup", native.ProcessGroup),
                      ("_JittorWork", native.Work)):
    payload = ("cjittor.compat.torch.installers.distributed\n" + old_name + "\n.")
    assert pickle.loads(payload.encode("ascii")) is cls
    value = cls()
    assert type(pickle.loads(pickle.dumps(value))) is cls
print("native ProcessGroup ownership and legacy pickle passed")
'''
        result = run_python_child(
            ["-c", source], env=_BASE_ENV, cwd=REPO_ROOT,
            text=True, merge_stderr=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)


class TestCrossProcessStores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Both children share one JIT cache. Warm it before the master blocks
        # waiting for the client, otherwise the client can be waiting on the
        # compile lock held by that blocked master.
        completed = run_python_child(
            ["-c", "import jittor, os; print(jittor.__file__)"],
            env=_BASE_ENV,
            cwd=REPO_ROOT,
            text=True,
            merge_stderr=True,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stdout)
        expected = os.fspath(REPO_ROOT / "python" / "jittor")
        if expected not in completed.stdout:
            raise AssertionError("child imported another checkout:\n" + completed.stdout)

    def _run_pair(self, source, rank_envs):
        # The two ranks have to be alive at the same time, so they are launched
        # by hand rather than through run_python_child() -- but the budget is
        # still that helper's, for its reasons: a child that has to compile the
        # core does not fit in a timeout tuned for a warm cache, and setUpClass'
        # warm-up does not help when something invalidates the cache mid-run.
        # A fixed 30 s turned that into ``-9 != 0`` with an empty output, which
        # names neither the compile nor the rank that was still building.
        #
        # Wall clock is not what this test asserts: both stores above carry a
        # 10 s timeout of their own, so a rendezvous that never completes still
        # fails inside the child. This budget only turns a true hang into a
        # failure instead of a hung session.
        budget = default_timeout()
        processes = []
        outputs = []
        killed = set()
        try:
            for rank, extra in enumerate(rank_envs):
                env = dict(_BASE_ENV)
                env.update(extra)
                env["STORE_RANK"] = str(rank)
                processes.append(subprocess.Popen(
                    [PYTHON, "-c", source],
                    cwd=REPO_ROOT,
                    env=child_env(env),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                ))
            for rank, process in enumerate(processes):
                try:
                    output, _ = process.communicate(timeout=budget)
                except subprocess.TimeoutExpired:
                    killed.add(rank)
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    output, _ = process.communicate(timeout=5)
                outputs.append(output)
        finally:
            for process in processes:
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait(timeout=5)
        for rank, (process, output) in enumerate(zip(processes, outputs)):
            self.assertNotIn(
                rank, killed,
                "rank {} was still running after {} s and was killed; its "
                "return code below is that kill, not its own exit:\n{}".format(
                    rank, budget, output))
            self.assertEqual(process.returncode, 0, "rank {}:\n{}".format(rank, output))
            self.assertIn("DONE", output)

    @staticmethod
    def _free_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def test_tcp_store_crosses_process_boundary(self):
        port = self._free_port()
        env = {"STORE_KIND": "tcp", "STORE_PORT": str(port)}
        self._run_pair(_DIRECT_STORE, [env, env])

    def test_file_store_crosses_process_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {
                "STORE_KIND": "file",
                "STORE_PATH": os.path.join(directory, "store.sqlite3"),
            }
            self._run_pair(_DIRECT_STORE, [env, env])

    def test_init_process_group_uses_env_and_tcp_urls(self):
        for scheme in ("env", "tcp"):
            with self.subTest(scheme=scheme):
                port = self._free_port()
                common = {
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": str(port),
                    "WORLD_SIZE": "2",
                }
                rank_envs = []
                for rank in range(2):
                    env = dict(common)
                    env["RANK"] = str(rank) if scheme == "env" else "99"
                    env["WORLD_SIZE"] = "2" if scheme == "env" else "99"
                    env["STORE_INIT_METHOD"] = (
                        "env://" if scheme == "env" else
                        "tcp://127.0.0.1:{}?rank={}&world_size=2".format(
                            port, rank)
                    )
                    rank_envs.append(env)
                self._run_pair(_INIT_METHOD_STORE, rank_envs)


class TestHostnameRendezvous(unittest.TestCase):
    """A store must be reachable at the hostname the caller passes.

    ``localhost`` is not always one address. On the host this was found,
    ``/etc/hosts`` maps it to ``::1`` and to the machine's own IPv4 address,
    and to no loopback IPv4 at all. The server used to bind whatever
    ``socket.bind`` resolved first (``127.0.0.1``) while every client dialled
    ``::1`` first, so the rendezvous could never complete: it surfaced as a
    multi-minute connect timeout in a traceback that named neither address.
    """

    @staticmethod
    def _store_module():
        spec = importlib.util.spec_from_file_location(
            "jittor_store_under_test", STORE_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _free_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def test_server_binds_an_address_the_client_dials(self):
        module = self._store_module()
        port = self._free_port()
        store = module.TCPStore("localhost", port, 1, True, timeout=10)
        try:
            bound = store._server.socket.getsockname()
            # Exactly the list ``socket.create_connection`` walks, in order.
            dialled = [
                sockaddr for _family, _socktype, _proto, _canonname, sockaddr
                in socket.getaddrinfo("localhost", port, 0, socket.SOCK_STREAM)
            ]
            self.assertIn(
                bound, dialled,
                "TCPStore bound {} but a client only dials {}".format(
                    bound, dialled),
            )
            store.set("payload", b"reachable")
            self.assertEqual(store.get("payload"), b"reachable")
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
