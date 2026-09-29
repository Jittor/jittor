# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""jittor.distributed.launch: failure propagation and isolated per-rank JIT state.

No GPU and no jittor import in the ranks -- the launcher's own behaviour is
what is under test, so the ranks here are three-line python programs. That is
deliberate: the defect below is entirely in how the launcher waits, and a test
that needed N cards would never run.

The defect: the launcher waited on its ranks **in rank order**
(``for rank, (p, logf) in enumerate(procs): p.wait()``), with the kill only in
a ``finally``. So when rank 3 crashed, the launcher was still blocked on
rank 0 -- and rank 0 was in all likelihood hung *because* rank 3 had died, in
the collective it would now wait for forever. The job never ended, no rank
reported anything, and the launcher printed nothing until someone killed it by
hand. Every extra rank makes this more likely, which is the wrong direction.
"""
import os
import runpy
from pathlib import Path
import tempfile
import time
import unittest

from _helpers.child_process import PYTHON, run_python_child

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LAUNCH = _REPO_ROOT / "python" / "jittor" / "distributed" / "launch.py"

# Rank 1 fails immediately; every other rank would otherwise outlive the test.
# `flush` before sleeping so the "started" marker is on disk when we look.
_RANK_0_SLEEPS = """
import os, sys, time
rank = int(os.environ["JT_NCCL_RANK"])
print("rank %d up" % rank, flush=True)
if rank == 1:
    sys.exit(7)
time.sleep(600)
"""

_PRINT_CACHE_NAME = """
import os
print("cache_name=%r" % os.environ.get("JITTOR_HOME"), flush=True)
"""


def _launch(nproc, code, logdir, timeout):
    # Pin the repository path for every child. An explicit backend keeps the
    # launcher and metadata-only workers independent of the Jittor runtime;
    # no GPU, model, communicator or simulated remote host is involved.
    start = time.time()
    done = run_python_child(
        [os.fspath(_LAUNCH), "-n", str(nproc), "--backend", "nccl",
         "--logdir", logdir, "--", PYTHON, "-c", code],
        cwd=_REPO_ROOT, merge_stderr=True, timeout=timeout)
    return done, time.time() - start


class TestLaunchFailurePropagation(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_static_topology_environment_is_pure_metadata(self):
        worker_environment = runpy.run_path(str(_LAUNCH))["worker_environment"]
        kwargs = dict(nproc=2, nnodes=3, node_rank=2, backend="hccl",
                      master_addr="actual-master", master_port=29500,
                      rootinfo="/shared/fresh-run/rootinfo.bin",
                      state_root=self.tmp.name, run_id="metadata-contract")
        first = worker_environment({}, local_rank=0, **kwargs)
        second = worker_environment({}, local_rank=1, **kwargs)
        self.assertEqual((first["RANK"], second["RANK"], first["WORLD_SIZE"]), ("4", "5", "6"))
        self.assertEqual(first["LOCAL_WORLD_SIZE"], "2")
        self.assertEqual(second["JT_HCCL_LOCAL_RANK"], "1")
        self.assertEqual(first["JT_HCCL_ROOTINFO_FILE"], second["JT_HCCL_ROOTINFO_FILE"])
        for name in ("JITTOR_HOME", "TMPDIR", "XDG_CACHE_HOME", "CCACHE_DIR"):
            self.assertNotEqual(first[name], second[name])

    def test_fixed_private_cache_does_not_reuse_execution_identity(self):
        worker_environment = runpy.run_path(str(_LAUNCH))["worker_environment"]
        pool = str(Path(self.tmp.name) / "private-pool")
        kwargs = dict(nproc=2, nnodes=1, node_rank=0, local_rank=0,
                      backend="hccl", master_addr="localhost", master_port=29500,
                      state_root=self.tmp.name, cache_root=pool)
        first = worker_environment({}, run_id="fresh-first",
                                   rootinfo="/shared/fresh-first.bin", **kwargs)
        second = worker_environment({}, run_id="fresh-second",
                                    rootinfo="/shared/fresh-second.bin", **kwargs)
        for name in ("JITTOR_HOME", "CCACHE_DIR", "CCACHE_BASEDIR", "CCACHE_CONFIGPATH"):
            self.assertEqual(first[name], second[name])
        for name in ("TORCHELASTIC_RUN_ID", "JT_HCCL_ROOTINFO_FILE", "TMPDIR", "XDG_CACHE_HOME"):
            self.assertNotEqual(first[name], second[name])
        kwargs["local_rank"] = 1
        other = worker_environment({}, run_id="fresh-second",
                                   rootinfo="/shared/fresh-second.bin", **kwargs)
        self.assertNotEqual(second["JITTOR_HOME"], other["JITTOR_HOME"])
        self.assertNotEqual(second["CCACHE_DIR"], other["CCACHE_DIR"])
        self.assertEqual(other["RANK"], "1")
        # Expanding a later actual world reuses only its matching global rank.
        kwargs.update(nproc=4, local_rank=0)
        expanded = worker_environment({}, run_id="fresh-four",
                                      rootinfo="/shared/fresh-four.bin", **kwargs)
        self.assertEqual(second["JITTOR_HOME"], expanded["JITTOR_HOME"])
        self.assertEqual(expanded["WORLD_SIZE"], "4")

    def test_one_failing_rank_ends_the_job(self):
        """Rank 1 exits 7 while rank 0 sleeps for ten minutes.

        The launcher must come back in seconds with rc 7, not in ten minutes.
        Before 8.10 it blocked on rank 0 first, so this hung until the harness
        gave up -- which is why the subprocess timeout here is 120s and not the
        600s rank 0 would otherwise take.
        """
        done, elapsed = _launch(4, _RANK_0_SLEEPS, self.tmp.name, timeout=120)
        self.assertEqual(done.returncode, 7, done.stdout[-3000:])
        self.assertLess(elapsed, 60,
                        "launcher took %.0fs: it is still waiting in rank "
                        "order:\n%s" % (elapsed, done.stdout[-3000:]))
        # It says which rank failed, and points at that rank's log rather than
        # leaving the operator to guess among N of them.
        self.assertIn("rank 1", done.stdout)
        self.assertIn("rank1.log", done.stdout)
        # The sleeping ranks are gone too: _stop_all waits for them before the
        # launcher returns, so "came back in seconds" also means "did not leave
        # three ten-minute sleepers behind".

    def test_rendezvous_files_are_cleaned_up(self):
        """No rootinfo and no watchdog heartbeats left in the log directory.

        A heartbeat left behind by a killed rank would make the next job
        started on the same path see a peer that is not there.
        """
        Path(self.tmp.name).mkdir(exist_ok=True)
        _launch(2, _RANK_0_SLEEPS, self.tmp.name, timeout=120)
        left = sorted(p for p in os.listdir(self.tmp.name)
                      if "rootinfo" in p or ".hb" in p)
        self.assertEqual(left, [], "left behind: %s" % left)

    def test_all_ranks_have_distinct_jit_homes(self):
        """Concurrent workers must not write one compilation cache.

        External protocols prewarm each deterministic run/rank directory
        serially before launching model workers.
        """
        done, _ = _launch(3, _PRINT_CACHE_NAME, self.tmp.name, timeout=120)
        self.assertEqual(done.returncode, 0, done.stdout[-3000:])
        names = set()
        for rank in range(3):
            text = Path(self.tmp.name, "rank%d.log" % rank).read_text()
            self.assertIn("cache_name=", text, text)
            names.add(text.strip().split("cache_name=", 1)[1])
        self.assertEqual(len(names), 3,
                         "ranks shared JIT homes: %s" % sorted(names))


if __name__ == "__main__":
    unittest.main()
