# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""MPI-free launcher arguments, rank setup, cleanup, and failure propagation.

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
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import unittest

from _helpers.child_process import PYTHON, child_env, run_python_child

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
print("cache_name=%r" % os.environ.get("cache_name"), flush=True)
"""

_PRINT_NCCL_ENV = """
import os
print("use_nccl=%s use_mpi=%s" %
      (os.environ.get("JT_BUILD_USE_NCCL"),
       os.environ.get("JT_BUILD_USE_MPI")), flush=True)
"""

_PRINT_NCCL_P2P = """
import os
print("nccl_p2p_disable=%s" % os.environ.get("NCCL_P2P_DISABLE"), flush=True)
"""

_PRINT_RANK_ENV = """
import os
keys = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "MASTER_ADDR", "MASTER_PORT", "JT_NCCL_RANK", "JT_NCCL_LOCAL_RANK",
        "JT_NCCL_WORLD_SIZE", "JT_RENDEZVOUS_TIMEOUT_S", "CUDA_VISIBLE_DEVICES")
print(" ".join("%s=%s" % (key, os.environ.get(key)) for key in keys), flush=True)
"""


def _launch(nproc, code, logdir, timeout, rendezvous_timeout=120):
    # Through _helpers.child_process, the launcher passes its environment down
    # to every rank, so an unpinned PYTHONPATH here would put another checkout
    # in all of them. The ranks need no GPU; explicit backend setup is tested
    # without importing the runtime in the parent.
    start = time.time()
    done = run_python_child(
        [os.fspath(_LAUNCH), "--nproc-per-node", str(nproc), "--backend", "nccl",
         "--device-ids", ",".join(str(rank) for rank in range(nproc)),
         "--log-dir", logdir, "--standalone", "--timeout", str(rendezvous_timeout),
         "--", PYTHON, "-c", code],
        cwd=_REPO_ROOT, merge_stderr=True, timeout=timeout)
    return done, time.time() - start


class TestLaunchFailurePropagation(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

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

    def test_all_ranks_share_one_jit_cache(self):
        """The launcher must not give each rank a cache of its own.

        It used to set ``cache_name=<backend><rank>``, so an N-card job
        compiled the same kernels N times into N directories: minutes and
        gigabytes per extra card, for nothing. One cache is what the mpirun
        path has always used; jittor.lock serializes the builds.
        """
        done, _ = _launch(3, _PRINT_CACHE_NAME, self.tmp.name, timeout=120)
        self.assertEqual(done.returncode, 0, done.stdout[-3000:])
        names = set()
        for rank in range(3):
            text = Path(self.tmp.name, "rank%d.log" % rank).read_text()
            self.assertIn("cache_name=", text, text)
            names.add(text.strip().split("cache_name=", 1)[1])
        self.assertEqual(len(names), 1,
                         "ranks got different JIT caches: %s" % sorted(names))

    def test_explicit_nccl_launch_enables_nccl_before_rank_import(self):
        done, _ = _launch(2, _PRINT_NCCL_ENV, self.tmp.name, timeout=120)
        self.assertEqual(done.returncode, 0, done.stdout[-3000:])
        for rank in range(2):
            text = Path(self.tmp.name, "rank%d.log" % rank).read_text()
            self.assertIn("use_nccl=1 use_mpi=0", text, text)

    def test_nccl_p2p_is_frozen_disabled_for_every_rank(self):
        done, _ = _launch(3, _PRINT_NCCL_P2P, self.tmp.name, timeout=120)
        self.assertEqual(done.returncode, 0, done.stdout[-3000:])
        for rank in range(3):
            text = Path(self.tmp.name, "rank%d.log" % rank).read_text()
            self.assertIn("nccl_p2p_disable=1", text, text)

    def test_auto_cuda_detection_sets_nccl_build_flags_before_import(self):
        done = run_python_child(
            [os.fspath(_LAUNCH), "--nproc-per-node", "1", "--backend", "auto",
             "--device-ids", "0", "--log-dir", self.tmp.name, "--standalone",
             "--", PYTHON, "-c", _PRINT_NCCL_ENV],
            cwd=_REPO_ROOT, env={"JT_BACKEND": "cuda"},
            merge_stderr=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stdout[-3000:])
        text = Path(self.tmp.name, "rank0.log").read_text()
        self.assertIn("use_nccl=1 use_mpi=0", text, text)

    def test_torchrun_environment_and_per_rank_device_mapping(self):
        done, _ = _launch(2, _PRINT_RANK_ENV, self.tmp.name,
                          timeout=120, rendezvous_timeout=17)
        self.assertEqual(done.returncode, 0, done.stdout[-3000:])
        for rank in range(2):
            text = Path(self.tmp.name, "rank%d.log" % rank).read_text()
            expected = {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": "2",
                "LOCAL_WORLD_SIZE": "2",
                "MASTER_ADDR": "127.0.0.1",
                "JT_NCCL_RANK": str(rank),
                "JT_NCCL_LOCAL_RANK": "0",
                "JT_NCCL_WORLD_SIZE": "2",
                "JT_RENDEZVOUS_TIMEOUT_S": "17.0",
                "CUDA_VISIBLE_DEVICES": str(rank),
            }
            for key, value in expected.items():
                self.assertIn("%s=%s" % (key, value), text, text)
            port = text.split("MASTER_PORT=", 1)[1].split()[0]
            self.assertTrue(1 <= int(port) <= 65535, text)

    def test_invalid_device_count_and_duplicate_device_ids_fail_before_start(self):
        too_few = run_python_child(
            [os.fspath(_LAUNCH), "-n", "2", "--backend", "hccl",
             "--device-ids", "0", "--", PYTHON, "-c", "pass"],
            cwd=_REPO_ROOT, merge_stderr=True, timeout=120)
        self.assertEqual(too_few.returncode, 2, too_few.stdout)
        self.assertIn("only 1 visible device", too_few.stdout)

        duplicate = run_python_child(
            [os.fspath(_LAUNCH), "-n", "2", "--backend", "hccl",
             "--device-ids", "0,0", "--", PYTHON, "-c", "pass"],
            cwd=_REPO_ROOT, merge_stderr=True, timeout=120)
        self.assertEqual(duplicate.returncode, 2, duplicate.stdout)
        self.assertIn("must not contain duplicates", duplicate.stdout)

    def test_occupied_master_port_fails_before_rank_start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            done = run_python_child(
                [os.fspath(_LAUNCH), "-n", "1", "--backend", "hccl",
                 "--device-ids", "0", "--master-addr", "127.0.0.1",
                 "--master-port", str(port), "--", PYTHON, "-c", "pass"],
                cwd=_REPO_ROOT, merge_stderr=True, timeout=120)
        self.assertEqual(done.returncode, 2, done.stdout)
        self.assertIn("master port", done.stdout)

    def test_missing_executable_returns_127_without_rendezvous_artifacts(self):
        done = run_python_child(
            [os.fspath(_LAUNCH), "-n", "1", "--backend", "hccl",
             "--device-ids", "0", "--", "jtrun-command-that-does-not-exist"],
            cwd=_REPO_ROOT, merge_stderr=True, timeout=120)
        self.assertEqual(done.returncode, 127, done.stdout)
        self.assertIn("command not found", done.stdout)
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_missing_script_exit_is_propagated_and_rendezvous_is_cleaned(self):
        missing_script = Path(self.tmp.name, "missing-training-script.py")
        done = run_python_child(
            [os.fspath(_LAUNCH), "-n", "1", "--backend", "nccl",
             "--device-ids", "0", "--log-dir", self.tmp.name, "--standalone",
             "--", PYTHON, os.fspath(missing_script)],
            cwd=_REPO_ROOT, merge_stderr=True, timeout=120)
        self.assertEqual(done.returncode, 2, done.stdout[-3000:])
        self.assertIn("rank 0 failed", done.stdout)
        self.assertTrue(Path(self.tmp.name, "rank0.log").exists())
        self.assertFalse(any("rootinfo" in name or ".hb" in name
                             for name in os.listdir(self.tmp.name)))

    def test_parent_sigterm_stops_ranks_and_cleans_rendezvous(self):
        Path(self.tmp.name).mkdir(exist_ok=True)
        process = subprocess.Popen(
            [PYTHON, os.fspath(_LAUNCH), "-n", "2", "--backend", "nccl",
             "--device-ids", "0,1", "--log-dir", self.tmp.name, "--standalone",
             "--", PYTHON, "-c", "import time; print('rank up', flush=True); time.sleep(600)"],
            cwd=_REPO_ROOT,
            env=child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        deadline = time.time() + 60
        while time.time() < deadline:
            if all(Path(self.tmp.name, "rank%d.log" % rank).exists()
                   for rank in range(2)):
                if all("rank up" in Path(self.tmp.name, "rank%d.log" % rank).read_text()
                       for rank in range(2)):
                    break
            if process.poll() is not None:
                break
            time.sleep(0.1)
        self.assertIsNone(process.poll(), "launcher exited before interruption")
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 143, output[-3000:])
        self.assertIn("received SIGTERM", output)
        left = [name for name in os.listdir(self.tmp.name)
                if "rootinfo" in name or ".hb" in name]
        self.assertEqual(left, [], "left behind: %s" % left)

    def test_module_entry_help_and_legacy_alias(self):
        for args in (
            ["-m", "jittor.distributed.launch", "--help"],
            ["-m", "jittor.distributed.launch", "-h"],
        ):
            done = run_python_child(args, cwd=_REPO_ROOT, merge_stderr=True, timeout=120)
            self.assertEqual(done.returncode, 0, done.stdout[-3000:])
            self.assertIn("--nproc-per-node", done.stdout)
            self.assertIn("--device-ids", done.stdout)

    def test_script_and_module_targets_run_without_separator(self):
        script = Path(self.tmp.name, "train.py")
        script.write_text(
            "import json, sys\n"
            "print(json.dumps(sys.argv[1:]), flush=True)\n"
        )
        script_done = run_python_child(
            [os.fspath(_LAUNCH), "--nproc-per-node", "1", "--backend", "hccl",
             "--device-ids", "0", "--log-dir", self.tmp.name,
             os.fspath(script), "--learning-rate", "0.1"],
            cwd=_REPO_ROOT, merge_stderr=True, timeout=120)
        self.assertEqual(script_done.returncode, 0, script_done.stdout[-3000:])
        self.assertIn(json.dumps(["--learning-rate", "0.1"]),
                      Path(self.tmp.name, "rank0.log").read_text())

        module_done = run_python_child(
            [os.fspath(_LAUNCH), "--nproc-per-node", "1", "--backend", "hccl",
             "--device-ids", "0", "--log-dir", self.tmp.name, "--module", "site"],
            cwd=_REPO_ROOT, merge_stderr=True, timeout=120)
        self.assertEqual(module_done.returncode, 0, module_done.stdout[-3000:])
        self.assertIn("sys.path", Path(self.tmp.name, "rank0.log").read_text())

    def test_console_script_is_registered(self):
        project = (_REPO_ROOT / "pyproject.toml").read_text()
        self.assertIn("[project.scripts]", project)
        self.assertIn('jtrun = "jittor.distributed.launch:main"', project)


if __name__ == "__main__":
    unittest.main()
