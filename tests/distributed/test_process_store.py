"""Cross-process rendezvous stores used by torch.distributed compatibility."""

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
    run_python_child,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
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
from jittor.compat.torch.installers import distributed as implementation

rank = int(os.environ["STORE_RANK"])
implementation._native_distributed_active = lambda: True
implementation._distributed_rank = lambda: rank
implementation._distributed_world_size = lambda: 2
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
        processes = []
        outputs = []
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
            for process in processes:
                try:
                    output, _ = process.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    output, _ = process.communicate(timeout=5)
                outputs.append(output)
        finally:
            for process in processes:
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait(timeout=5)
        for rank, (process, output) in enumerate(zip(processes, outputs)):
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


if __name__ == "__main__":
    unittest.main()
