# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The NCCL file rendezvous must fail with a diagnosis, not a riddle (8.09).

This is the minimal reproduction of the multi-card failure that is hardest to
read: **one rank does not come up, and the others are left holding nothing**.
It needs only one GPU, because a rank waiting for a peer that never arrives
never gets as far as touching a second device.

Measured on the code this replaces (NCCL 2.18.3, one RTX 4090):

* ``JT_NCCL_RANK`` != 0 polled the rootinfo file 6000 times at 20 ms -- a
  hardcoded 121 s that no environment variable could shorten or lengthen --
  and then carried on *without checking whether it had read anything*. The
  still-zero ``ncclUniqueId`` went into ``ncclCommInitRank``, which returned
  ``ncclInternalError``: "internal error - please report this issue to the
  NCCL developers". So a jittor launch misconfiguration spent two minutes
  looking like a hang and then told the operator to open a bug against NVIDIA.
  (The audit predicted a permanent hang instead. Which of the two you get
  depends on whether the NCCL build rejects an all-zero id; neither names the
  cause, and a partially written file still reaches the blocking variant.)
* ``JT_NCCL_ROOTINFO_FILE`` unset skipped the wait entirely and reached the
  same ``ncclCommInitRank`` after 0 s, with the same NCCL-internal-error text.

So the assertions are: it says *rendezvous*, it names the rank and the path, it
waits exactly as long as it was told to, and -- the half that needs the
communicator to be built by an explicit call rather than a static constructor
-- the failure arrives as a Python exception that ``import jittor``'s caller
can catch. A regression that reinstates a blocking ``ncclCommInitRank`` shows
up as this test timing out rather than failing an assert, which is why every
subprocess below carries its own timeout.

``JT_RENDEZVOUS_TIMEOUT_S`` keeps this test at a few seconds instead of the
120 s default -- and that it works at all is part of what is being tested.

Two traps that this file has to work around, both worth knowing before writing
any other test that spawns a jittor process:

1. **A child that dies by a signal kills the parent.** jittor installs a
   SIGCHLD handler (``utils/log.cc``) that reads any non-``CLD_EXITED`` child
   as "maybe out of memory" and calls ``_Exit(1)`` on itself. ``_Exit`` skips
   stdio flushing, so pytest vanishes with *no output at all* and exit code 1
   -- it does not look like a crash, it looks like nothing happened.
2. **A failed ``import jittor`` aborts at shutdown on a CUDA build.** The
   global ``EventQueue`` starts a worker thread and unregisters it from
   ``jittor_exit``/``core.cleanup()``, which is only wired up by an import that
   *completes*. When one does not, ``~std::thread`` runs on a joinable thread:
   "terminate called without an active exception", SIGABRT -- and then trap 1
   takes the test runner with it. That is a pre-existing shutdown defect,
   independent of anything here, so the child below catches the exception and
   leaves via ``os._exit`` rather than letting interpreter shutdown run. That
   the exception can be caught at all is the point of the explicit init.
"""
import json
import os
from pathlib import Path
import socket
import tempfile
import time
import unittest

import jittor as jt

from _helpers.child_process import run_python_child

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Long enough to build the NCCL op module on a cold cache, short enough that a
# rank hung in ncclCommInitRank is reported as a failure of this test rather
# than of whatever runs next.
_IMPORT_TIMEOUT_S = 900
_RENDEZVOUS_TIMEOUT_S = 5

# Exit code the child uses for "import jittor raised, and I caught it".
_RAISED = 9

_CHILD = """
import os, sys, traceback
try:
    import jittor
except BaseException:
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit({raised})
print("IMPORT OK")
""".format(raised=_RAISED)

# One plain python between pytest and the jittor process, for trap 1 above: it
# never imports jittor, so it has no SIGCHLD handler, and it reports a
# signalled grandchild as a shell-style 128+signal exit instead of letting the
# signal reach the test runner. Without it, running this file against the code
# it was written for -- where the failure aborts inside a static constructor --
# does not report a failing assertion, it makes pytest disappear.
_RUNNER = """
import subprocess, sys
p = subprocess.run([sys.executable, "-c", sys.argv[1]])
sys.exit(p.returncode if p.returncode >= 0 else 128 - p.returncode)
"""

_TWO_RANK_CHILD = """
import os
import jittor as jt
import torch.distributed as dist
dist.init_process_group(backend="nccl", init_method="env://")
jt.flags.use_cuda = 1
rank = int(os.environ["RANK"])
value = jt.array([rank + 1.0]).mpi_all_reduce("sum")
assert float(value.item()) == 3.0, (rank, value.item())
print("STORE NCCL OK", rank, flush=True)
"""

_TWO_RANK_CONDUCTOR = r"""
import json
import os
import signal
import subprocess
import sys
import time

child_source, port = sys.argv[1:]
processes = []
devices = [item.strip() for item in os.environ.get(
    "CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
for rank in range(2):
    env = dict(os.environ)
    env["WORLD_SIZE"] = "2"
    env["RANK"] = str(rank)
    if len(devices) >= 2:
        env["CUDA_VISIBLE_DEVICES"] = devices[rank]
        env["LOCAL_RANK"] = "0"
    else:
        env["LOCAL_RANK"] = str(rank)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = port
    env["JITTOR_TORCH_SHIM"] = "1"
    env["JITTOR_TORCH_DISTRIBUTED_AUTO_INIT"] = "1"
    env.pop("JT_NCCL_ROOTINFO_FILE", None)
    env.pop("JT_NCCL_WORLD_SIZE", None)
    env.pop("JT_NCCL_RANK", None)
    env.pop("JT_NCCL_LOCAL_RANK", None)
    processes.append(subprocess.Popen(
        [sys.executable, "-c", child_source], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    ))

deadline = time.monotonic() + 120
while any(process.poll() is None for process in processes):
    if any(process.poll() not in (None, 0) for process in processes):
        break
    if time.monotonic() >= deadline:
        break
    time.sleep(0.05)

for process in processes:
    if process.poll() is None:
        process.terminate()
for process in processes:
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
outputs = []
for process in processes:
    output, _ = process.communicate(timeout=5)
    outputs.append({"returncode": process.returncode, "output": output})
print(json.dumps(outputs), flush=True)
"""


def _run_import(env_overrides, timeout=_IMPORT_TIMEOUT_S, code=_CHILD):
    """Run `code` in a subprocess. Returns (returncode, output, seconds)."""
    env = dict(os.environ)
    env["use_nccl"] = "1"
    env.pop("JT_NCCL_ROOTINFO_FILE", None)
    env.update(env_overrides)
    start = time.time()
    # inherit=False: this environment had JT_NCCL_ROOTINFO_FILE *removed*, and
    # the helper merges onto os.environ, which would put it straight back.
    # The helper also pins PYTHONPATH to this checkout -- a bare `python -c`
    # picks up whatever the editable install points at instead.
    try:
        done = run_python_child(
            ["-c", _RUNNER, code], env=env, inherit=False,
            cwd=_REPO_ROOT, merge_stderr=True, timeout=timeout)
    except AssertionError as expired:
        raise AssertionError(
            "import jittor did not return within {}s -- the rendezvous is "
            "hanging, which is exactly the defect this test exists for.\n"
            "{}".format(timeout, expired))
    return done.returncode, done.stdout, time.time() - start


@unittest.skipIf(not jt.has_cuda, "no CUDA, NCCL is not built")
class TestNcclRendezvousTimeout(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _rendezvous_env(self, rank, world_size, rootinfo):
        env = {
            "JT_NCCL_WORLD_SIZE": str(world_size),
            "JT_NCCL_RANK": str(rank),
            "JT_NCCL_LOCAL_RANK": "0",
            "JT_RENDEZVOUS_TIMEOUT_S": str(_RENDEZVOUS_TIMEOUT_S),
        }
        if rootinfo is not None:
            env["JT_NCCL_ROOTINFO_FILE"] = rootinfo
        return env

    def _assert_raised(self, code, out):
        # _RAISED, not just "non-zero": it means the child reached its own
        # except branch, i.e. the failure came back as a Python exception
        # instead of unwinding out of a static constructor into the dynamic
        # loader (where it becomes std::terminate and no traceback at all).
        self.assertEqual(code, _RAISED,
                         "expected a catchable import failure, got exit {}{}:\n{}"
                         .format(code,
                                 " (killed by signal {})".format(code - 128)
                                 if code > 128 else "",
                                 out[-3000:]))

    @staticmethod
    def _free_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def test_bad_master_port_times_out_with_endpoint(self):
        port = self._free_port()
        env = self._rendezvous_env(1, 2, None)
        env.update({
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        })
        code, out, elapsed = _run_import(env)
        self._assert_raised(code, out)
        self.assertIn("127.0.0.1:{}".format(port), out)
        self.assertIn("timed out", out)
        self.assertGreater(elapsed, _RENDEZVOUS_TIMEOUT_S)
        self.assertLess(elapsed, _RENDEZVOUS_TIMEOUT_S + 20)

    @unittest.skipIf(jt.core.get_device_count() < 2, "requires two CUDA devices")
    def test_two_rank_nccl_uses_tcp_store_without_rootinfo_file(self):
        # Warm the shared cache before rank 0 enters the blocking store
        # constructor; otherwise rank 1 can be waiting for rank 0's build lock.
        warm_root = os.path.join(self.tmp.name, "warm-rootinfo.bin")
        warm_env = self._rendezvous_env(0, 1, warm_root)
        warm_env.update({
            "JITTOR_TORCH_SHIM": "1",
            "JITTOR_TORCH_DISTRIBUTED_AUTO_INIT": "1",
        })
        code, out, _ = _run_import(warm_env)
        self.assertEqual(code, 0, out[-3000:])

        port = self._free_port()
        env = dict(os.environ)
        env["JT_RENDEZVOUS_TIMEOUT_S"] = str(_RENDEZVOUS_TIMEOUT_S)
        env.pop("JT_NCCL_ROOTINFO_FILE", None)
        completed = run_python_child(
            ["-c", _TWO_RANK_CONDUCTOR, _TWO_RANK_CHILD, str(port)],
            env=env, inherit=False, cwd=_REPO_ROOT, merge_stderr=True,
            timeout=150,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout[-4000:])
        records = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertTrue(
            all(record["returncode"] == 0 for record in records),
            json.dumps(records, indent=2),
        )
        for rank, record in enumerate(records):
            self.assertIn(
                "STORE NCCL OK {}".format(rank), record["output"]
            )

    def test_single_rank_rendezvous_still_works(self):
        """The happy path: the shared helper must not break the working case.

        A world_size=1 job writes the unique id and builds a communicator. If
        only the failure cases were checked, "throw unconditionally" would pass
        all of them.
        """
        root = os.path.join(self.tmp.name, "rootinfo.bin")
        code, out, _ = _run_import(self._rendezvous_env(0, 1, root))
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("IMPORT OK", out)
        self.assertTrue(os.path.isfile(root),
                        "rank 0 did not write the rootinfo file:\n" + out[-3000:])

    def test_missing_peer_times_out_instead_of_hanging(self):
        """Rank 1 of 2, and rank 0 never publishes the unique-id key."""
        root = os.path.join(self.tmp.name, "never_written.bin")
        code, out, elapsed = _run_import(self._rendezvous_env(1, 2, root))
        self._assert_raised(code, out)
        self.assertIn("rendezvous timeout", out)
        # The message has to name the rank and the path, or the operator is left
        # guessing which of N ranks is stuck and on which file.
        self.assertIn("rank 1", out)
        self.assertIn(root, out)
        from jittor.distributed.store import FileStore
        store = FileStore(root, 2, timeout=1)
        self.addCleanup(store.close)
        self.assertFalse(store.check(["jittor/nccl/world/unique_id"]))
        # It waited the budget it was given. Asserting the *reported* wait
        # rather than the wall clock keeps this independent of how long the
        # build took, and it is the half the old 6000-iteration loop could not
        # do at all: its 121 s was not configurable by anything.
        self.assertIn("waited {} s".format(_RENDEZVOUS_TIMEOUT_S), out)
        # ...and it really did wait, rather than failing for an unrelated
        # reason before ever reaching the rendezvous.
        self.assertGreater(elapsed, _RENDEZVOUS_TIMEOUT_S)

    def test_missing_rootinfo_path_is_a_hard_error(self):
        """world_size>1 with no rootinfo path at all.

        There is nothing to wait for and nothing to read, so the old code went
        straight into ncclCommInitRank with an uninitialized id. Now it names
        the variable that is missing.
        """
        code, out, _ = _run_import(self._rendezvous_env(1, 2, None))
        self._assert_raised(code, out)
        self.assertIn("JT_NCCL_ROOTINFO_FILE", out)

    def test_rendezvous_does_not_block_holding_the_build_lock(self):
        """The rendezvous must run with jittor.lock released.

        jittor.lock is one flock over the whole cache directory and
        ``import jittor`` holds it end to end. A rank that waits for its peers
        while holding it stops them from ever arriving -- they need that same
        lock to compile. That deadlock has no symptom: one rank at 100% CPU
        inside MPI_Bcast, the others asleep on a file lock, and not one line of
        output from any of them. It is what a cold two-rank MPI run did the
        moment the communicator moved out of ``compile_custom_ops``'s dlopen
        (which drops the lock: "unlock scope when initialize") and into an
        explicit call that did not.

        This reproduces the setup in one process: hold the lock, then ask
        ``setup_nccl()`` to bring up rank 1 of 2. It must get as far as the
        rendezvous -- proving the lock was released around the call -- and fail
        there on the timeout. If the release is ever removed, the C++ guard in
        misc/file_rendezvous.h fires instead and the message names the lock,
        which is what the last assertion checks for.

        This is also the path the torch shim's NCCL installer takes: it sets
        the JT_NCCL_* variables after import and calls setup_nccl() itself.
        """
        root = os.path.join(self.tmp.name, "locked.bin")
        code = """
import os, sys, traceback
try:
    import jittor as jt
    from jittor_utils import lock
    os.environ["JT_NCCL_WORLD_SIZE"] = "2"
    os.environ["JT_NCCL_RANK"] = "1"
    os.environ["JT_NCCL_LOCAL_RANK"] = "0"
    os.environ["JT_NCCL_ROOTINFO_FILE"] = {root!r}
    with lock.lock_scope():
        assert lock.jittor_lock.is_locked, "test did not manage to take the lock"
        jt.compile_extern.setup_nccl()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit({raised})
print("SETUP RETURNED")
""".format(root=root, raised=_RAISED)
        # No JT_NCCL_* in the environment: `import jittor` has to succeed
        # first, exactly as it does under the torch installer.
        env = {"JT_RENDEZVOUS_TIMEOUT_S": str(_RENDEZVOUS_TIMEOUT_S)}
        rc, out, elapsed = _run_import(env, code=code)
        self._assert_raised(rc, out)
        self.assertIn("rendezvous timeout", out)
        self.assertGreater(elapsed, _RENDEZVOUS_TIMEOUT_S)
        self.assertNotIn("build lock", out)

    def test_rank_outside_world_is_a_hard_error(self):
        """A launcher bug (rank >= world_size) must not reach NCCL."""
        root = os.path.join(self.tmp.name, "rootinfo.bin")
        code, out, _ = _run_import(self._rendezvous_env(2, 2, root))
        self._assert_raised(code, out)
        self.assertIn("JT_NCCL_RANK", out)


if __name__ == "__main__":
    unittest.main()
