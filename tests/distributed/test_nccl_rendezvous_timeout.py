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
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import jittor as jt

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


def _run_import(env_overrides, timeout=_IMPORT_TIMEOUT_S, code=_CHILD):
    """Run `code` in a subprocess. Returns (returncode, output, seconds)."""
    env = dict(os.environ)
    # A bare `python -c` does NOT pick up this worktree: jittor is usually
    # installed editable and its .pth points at whatever tree was installed.
    env["PYTHONPATH"] = os.pathsep.join(
        [os.fspath(_REPO_ROOT / "python")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["use_nccl"] = "1"
    env.pop("JT_NCCL_ROOTINFO_FILE", None)
    env.update(env_overrides)
    start = time.time()
    try:
        done = subprocess.run(
            [sys.executable, "-c", _RUNNER, code],
            env=env, cwd=os.fspath(_REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = e.output.decode() if isinstance(e.output, bytes) else (e.output or "")
        raise AssertionError(
            "import jittor did not return within {}s -- the rendezvous is "
            "hanging, which is exactly the defect this test exists for.\n"
            "{}".format(timeout, out[-3000:]))
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
        """Rank 1 of 2, and rank 0 never starts. Nothing writes the file."""
        root = os.path.join(self.tmp.name, "never_written.bin")
        code, out, elapsed = _run_import(self._rendezvous_env(1, 2, root))
        self._assert_raised(code, out)
        self.assertIn("rendezvous timeout", out)
        # The message has to name the rank and the path, or the operator is left
        # guessing which of N ranks is stuck and on which file.
        self.assertIn("rank 1", out)
        self.assertIn(root, out)
        self.assertFalse(os.path.exists(root))
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
