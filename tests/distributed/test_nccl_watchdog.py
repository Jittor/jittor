# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""One rank dies; the others must find out (8.09).

Measured on two RTX 4090s before this existed: two ranks all-reducing happily,
``kill -9`` one of them, and the survivor was **still sitting there two minutes
later** -- last line "STEP 10", no error, no exit, GPU at 100%. That is the
audit's "one rank crashes, the rest hang indefinitely with no diagnosis", and
it is the single most expensive failure mode in multi-card training because
nothing about it says anything at all.

Two ranks and two GPUs are the whole point, so this file skips below that. It
is part of the ``nox -s nccl`` gate, which is where two devices are available.

Two things are asserted, and the second matters as much as the first:

1. the survivor **exits**, within the watchdog's own budget;
2. its message **names the rank that went away**. "Communication timed out" on
   every surviving rank tells an operator nothing about where to look;
   "rank 1 stopped updating its heartbeat" points straight at the log that has
   the actual cause. NCCL's own API cannot answer this -- ncclCommGetAsyncError
   returns a result code and no peer identity -- which is why the watchdog
   carries heartbeat files of its own.

Everything below runs the ranks under a plain python "conductor" rather than
from pytest directly, for the same reason
``test_nccl_rendezvous_timeout.py`` does: jittor installs a SIGCHLD handler
that reads any child killed by a signal as "maybe out of memory" and calls
``_Exit(1)`` on itself, without flushing stdio. A test that SIGKILLs a rank
from a jittor-importing process therefore does not fail -- pytest vanishes,
no output, exit 1. The conductor never imports jittor, so it absorbs that.

Also verified here: the async-error check alone is not enough. With both ranks
on one host and no peer access (this box: ``nvidia-smi topo -p2p r`` is CNS
everywhere, so NCCL_P2P_DISABLE=1), NCCL uses shared memory, there is no
socket to break, and ncclCommGetAsyncError stayed ncclSuccess for the full two
minutes while the kernel spun. The heartbeats are what catch this case; a
regression that removed them would leave this test hanging, not failing.
"""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import jittor as jt

from _helpers.child_process import run_python_child

_REPO_ROOT = Path(__file__).resolve().parents[2]

_INTERVAL_S = 3      # heartbeat/poll period
_STALE_S = 12        # a peer silent this long is gone
_GRACE_S = 20        # after the abort, before the watchdog ends the process
# Detection is bounded by _STALE_S; the abort then has to unwind a CUDA sync.
# Well under the _GRACE_S hard exit, so a pass means the abort worked and not
# that the watchdog shot the process.
_DEATH_BUDGET_S = _STALE_S + 20
_STEPS = 40
_START_STEP = 5     # both ranks must be past this before the kill
_WARMUP_TIMEOUT_S = 1800

_RANK_SCRIPT = """
import os, sys, time
import jittor as jt
jt.flags.use_cuda = 1
ops = jt.compile_extern.nccl_ops
rank, world = int(jt.rank), int(jt.world_size)
print("READY rank %d/%d pid %d" % (rank, world, os.getpid()), flush=True)
x = jt.ones((1 << 18,)) * (rank + 1.0)
x.sync()
for step in range(int(sys.argv[1])):
    total = float(ops.nccl_all_reduce(x).data[0])
    print("STEP %d rank %d total %g" % (step, rank, total), flush=True)
    time.sleep(0.25)
print("DONE rank %d" % rank, flush=True)
"""

# argv: mode("kill"|"clean") outdir steps devices... ; everything else is env.
# Never imports jittor -- see the module docstring.
_CONDUCTOR = """
import json, os, signal, subprocess, sys, time

mode, outdir, steps = sys.argv[1], sys.argv[2], sys.argv[3]
devices = sys.argv[4:]
script = os.environ["JT_TEST_RANK_SCRIPT"]
procs, logs = [], []
for rank, device in enumerate(devices):
    # os.environ here already carries the PYTHONPATH child_process pinned when
    # it started this conductor, so the ranks import the tree under test.
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = device
    env["cache_name"] = "nccl%d" % rank
    env["JT_NCCL_RANK"] = str(rank)
    env["JT_NCCL_LOCAL_RANK"] = "0"
    path = os.path.join(outdir, "rank%d.log" % rank)
    logs.append(path)
    handle = open(path, "w")
    procs.append((subprocess.Popen([sys.executable, "-c", script, steps],
                                   env=env, stdout=handle,
                                   stderr=subprocess.STDOUT), handle))

def alive(p):
    return p.poll() is None

def report(status, **extra):
    for p, h in procs:
        if alive(p):
            p.send_signal(signal.SIGKILL)
            try: p.wait(timeout=30)
            except Exception: pass
        h.close()
    extra["status"] = status
    extra["logs"] = logs
    print("SUMMARY " + json.dumps(extra), flush=True)
    sys.exit(0)

if mode == "clean":
    start = time.time()
    codes = []
    for p, _ in procs:
        try:
            codes.append(p.wait(timeout=900))
        except subprocess.TimeoutExpired:
            report("timeout")
    report("ok", codes=codes, seconds=time.time() - start)

# mode == "kill"
deadline = time.time() + 600
while time.time() < deadline:
    texts = [open(path, errors="replace").read() for path in logs]
    if all(("STEP %s " % os.environ["JT_TEST_START_STEP"]) in t for t in texts):
        break
    for p, _ in procs:
        if not alive(p):
            report("rank_exited_early")
    time.sleep(1)
else:
    report("never_started")

victim = procs[1][0]
victim.send_signal(signal.SIGKILL)
victim.wait(timeout=30)
killed_at = time.time()

survivor = procs[0][0]
budget = float(os.environ["JT_TEST_DEATH_BUDGET_S"])
while alive(survivor) and time.time() - killed_at < budget:
    time.sleep(0.5)
if alive(survivor):
    report("survivor_hung", seconds=time.time() - killed_at)
report("survivor_exited", code=survivor.returncode,
       seconds=time.time() - killed_at)
"""


def _visible_devices():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [d.strip() for d in raw.split(",") if d.strip()]


def _base_env(rootinfo, world_size):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.fspath(_REPO_ROOT / "python")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["use_nccl"] = "1"
    env["use_mpi"] = "0"
    # No GPU pair here has peer access, and NCCL treats a refused peer access
    # as fatal. compile_extern works this out when it can see the whole device
    # list; each rank below sees one device and cannot.
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env["JT_NCCL_WORLD_SIZE"] = str(world_size)
    env["JT_NCCL_ROOTINFO_FILE"] = rootinfo
    env["JT_NCCL_WATCHDOG_INTERVAL_S"] = str(_INTERVAL_S)
    env["JT_NCCL_WATCHDOG_STALE_S"] = str(_STALE_S)
    env["JT_NCCL_WATCHDOG_GRACE_S"] = str(_GRACE_S)
    env["JT_TEST_RANK_SCRIPT"] = _RANK_SCRIPT
    env["JT_TEST_START_STEP"] = str(_START_STEP)
    env["JT_TEST_DEATH_BUDGET_S"] = str(_DEATH_BUDGET_S)
    return env


@unittest.skipUnless(jt.has_cuda and len(_visible_devices()) >= 2,
                     "needs two CUDA devices in CUDA_VISIBLE_DEVICES")
class TestNcclWatchdog(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = _visible_devices()[:2]
        # Build each rank's cache on its own first. A cold two-rank start would
        # otherwise race the rendezvous timeout against a ten-minute compile.
        cls._warm = tempfile.TemporaryDirectory()
        for rank, device in enumerate(cls.devices):
            root = os.path.join(cls._warm.name, "warm%d.bin" % rank)
            env = _base_env(root, world_size=1)
            env["CUDA_VISIBLE_DEVICES"] = device
            env["cache_name"] = "nccl%d" % rank
            env["JT_NCCL_RANK"] = "0"
            env["JT_NCCL_LOCAL_RANK"] = "0"
            # inherit=False: `env` is already complete and had things set
            # per rank; the helper still pins PYTHONPATH to this checkout, so
            # the rank imports the tree under test rather than whatever the
            # editable install points at.
            done = run_python_child(
                ["-c", _RANK_SCRIPT, "1"], env=env, inherit=False,
                cwd=_REPO_ROOT, merge_stderr=True,
                timeout=_WARMUP_TIMEOUT_S)
            if done.returncode != 0:
                raise unittest.SkipTest(
                    "could not bring NCCL up on device %s:\n%s"
                    % (device, done.stdout[-3000:]))

    @classmethod
    def tearDownClass(cls):
        cls._warm.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rootinfo = os.path.join(self.tmp.name, "rootinfo.bin")

    def _conduct(self, mode, steps, timeout):
        done = run_python_child(
            ["-c", _CONDUCTOR, mode, self.tmp.name, str(steps)]
            + list(self.devices),
            env=_base_env(self.rootinfo, len(self.devices)), inherit=False,
            cwd=_REPO_ROOT, merge_stderr=True, timeout=timeout)
        line = [l for l in done.stdout.splitlines() if l.startswith("SUMMARY ")]
        self.assertTrue(line, "conductor said nothing:\n" + done.stdout[-3000:])
        summary = json.loads(line[-1][len("SUMMARY "):])
        summary["rank_logs"] = [Path(p).read_text(errors="replace")
                                for p in summary["logs"]]
        return summary

    def test_a_dead_rank_is_noticed_and_named(self):
        s = self._conduct("kill", _STEPS, timeout=1200)
        out = s["rank_logs"][0]
        self.assertEqual(
            s["status"], "survivor_exited",
            "rank 0 did not exit within %ds of rank 1 being killed (%s) -- it "
            "is waiting for a peer that will never arrive, which is the whole "
            "defect:\n%s" % (_DEATH_BUDGET_S, s["status"], out[-3000:]))
        self.assertNotEqual(s["code"], 0, out[-3000:])
        self.assertIn("NCCL watchdog", out)
        # Names the rank that went away, not just "something timed out".
        self.assertIn("rank(s) 1", out)
        # And it died through its own error path rather than being shot by the
        # watchdog's last-resort exit, which is what the abort is for.
        self.assertNotIn("still alive", out)

    def test_all_ranks_finishing_is_undisturbed(self):
        """The heartbeats must not decide a busy rank is dead.

        Without this, "abort whenever asked" would pass the test above.
        """
        s = self._conduct("clean", _STEPS, timeout=1800)
        self.assertEqual(s["status"], "ok", str(s)[:2000])
        for rank, (code, out) in enumerate(zip(s["codes"], s["rank_logs"])):
            self.assertEqual(code, 0, "rank %d:\n%s" % (rank, out[-3000:]))
            self.assertIn("DONE rank %d" % rank, out)
            self.assertNotIn("NCCL watchdog", out)
            # 1 + 2, so a rank that skipped the collective would show 1 or 2.
            self.assertIn("total 3", out)
        # Each rank removes its own heartbeat on the way out; a leftover would
        # make the next job started on this path see a peer that is not there.
        left = sorted(p for p in os.listdir(self.tmp.name) if ".hb" in p)
        self.assertEqual(left, [], "heartbeat files left behind: %s" % left)


if __name__ == "__main__":
    unittest.main()
