# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Splitting a pending graph must not change what it computes, or crash.

`auto_flush_ops` launches everything pending once that many operators have been
built, so the device computes while Python keeps building. A batch formed that
way can contain a `Tapes` op whose inputs the forward has already finished and
released -- correctly released, since `Tapes` computes nothing and marks none of
them needed. The runner applied the rule for a *compute* op to it anyway, read
`v->allocator->is_cuda()` on a freed Var, and segfaulted (KI-EXEC-001).

Five bottleneck blocks crashed at `auto_flush_ops` 1, 16, 32 and **128, the
shipping default**, while 64 and 256 happened not to. That is what made it read
as "past a graph-size threshold": the threshold was not a size, it was the
first place the cut landed on a tape.

Two things this file does deliberately
--------------------------------------
**Each setting runs in its own process.** The failure is a segfault, which
takes the interpreter with it; in-process it would end the suite instead of
failing a case, and every later test would silently not run.

**cuDNN autotuning is turned off for the comparison.** It picks convolution
algorithms by measuring them, so what is resident when it measures decides
which one wins -- and `auto_flush_ops` changes what is resident. That is a real
and separate defect ([KI-EXEC-003]) worth `3.8e-4` on the gradient norm, and it
is present whether or not this one is fixed. Leaving it on would put a
`3.8e-4` band around every comparison here and hide anything smaller. Off, the
settings agree to `2e-6`, which is reassociation from different batch
boundaries.
"""

from _helpers import capability as _test_capability
from _helpers.child_process import run_child_script

import json
import textwrap
import unittest

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


#: 0 is the baseline (never flushes). 1, 16, 32 and 128 all segfaulted before
#: the fix; 128 is the shipping default. 64 and 256 did not, and are here so a
#: regression that only ever exercised a crashing value could not pass by
#: turning the feature off.
FLUSH_SETTINGS = (0, 1, 16, 32, 64, 128, 256)

#: Five blocks. Four never crashed at any setting, so a shorter chain would
#: pass on the unfixed build and prove nothing.
BLOCKS = 5

#: Above the `2e-6` reassociation the batch boundaries produce, far below the
#: corruption a wrong fix causes -- an earlier attempt that suppressed the
#: check rather than exempting the op moved individual elements by whole
#: multiples of their neighbours.
GRADIENT_TOLERANCE = 1e-4

#: `run_child_script` writes the source and runs it with no arguments, so the
#: two parameters are formatted in rather than read from argv.
CHILD = textwrap.dedent('''
    import json, numpy as np, jittor as jt
    from jittor import nn
    jt.flags.use_cuda = 1
    jt.flags.auto_flush_ops = {flush}
    n = {blocks}

    # cuDNN picks its algorithm by measuring, and what is resident when it
    # measures depends on the flag under test (KI-EXEC-003). Off, so this
    # compares the executor and not the autotuner.
    import jittor.compile_extern as ce
    ce.cudnn.set_benchmark(0)

    jt.set_global_seed(0)
    class BK(nn.Module):
        def __init__(s, cin, mid, stride=1):
            s.c1 = nn.Conv2d(cin, mid, 1, bias=False); s.b1 = nn.BatchNorm(mid)
            s.c2 = nn.Conv2d(mid, mid, 3, stride, 1, bias=False); s.b2 = nn.BatchNorm(mid)
            s.c3 = nn.Conv2d(mid, mid*4, 1, bias=False); s.b3 = nn.BatchNorm(mid*4)
            s.ds = nn.Sequential(nn.Conv2d(cin, mid*4, 1, stride, bias=False),
                                 nn.BatchNorm(mid*4)) if (stride != 1 or cin != mid*4) else None
        def execute(s, x):
            idt = s.ds(x) if s.ds is not None else x
            y = nn.relu(s.b1(s.c1(x))); y = nn.relu(s.b2(s.c2(y))); y = s.b3(s.c3(y))
            return nn.relu(y + idt)

    model = nn.Sequential(*([BK(512, 256, stride=2)] + [BK(1024, 256) for _ in range(n-1)]))
    x = jt.array(np.random.RandomState(7).randn(1, 512, 8, 8).astype("float32"))
    loss = model(x).sum()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = jt.grad(loss, params)
    flat = np.concatenate([np.asarray(g.numpy()).ravel() for g in grads])
    print("RESULT " + json.dumps({{
        "loss": float(loss.numpy()),
        "grad_norm": float(np.linalg.norm(flat)),
        "grad_sum": float(flat.sum()),
        "nelem": int(flat.size),
    }}))
''')


def _run(flush, blocks):
    result = run_child_script(CHILD.format(flush=flush, blocks=blocks),
                              text=True, timeout=2400,
                              name="auto_flush_%d" % flush,
                              env={"JITTOR_ARGS": ""},
                              crash_isolated=True)
    stdout = result.stdout or ""
    line = next((l for l in stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        raise AssertionError(
            "auto_flush_ops=%d produced no result (return code %s). Before "
            "KI-EXEC-001 was fixed this was a segfault in run_exec_plan, "
            "reading a freed allocator on an input of a `tapes` op.\n%s"
            % (flush, result.returncode, (result.stderr or stdout)[-2000:]))
    return json.loads(line[len("RESULT "):])


@unittest.skipIf(not _has_cuda(), "no CUDA device")
class TestAutoFlushGraphSplit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for flush in FLUSH_SETTINGS:
            cls.results[flush] = _run(flush, BLOCKS)

    def test_every_setting_completes(self):
        """The crash itself. `_run` raises with the child's output if not."""
        for flush in FLUSH_SETTINGS:
            with self.subTest(auto_flush_ops=flush):
                self.assertEqual(self.results[flush]["nelem"],
                                 self.results[0]["nelem"])

    def test_the_loss_does_not_depend_on_the_flag(self):
        """Bit-identical: the forward is the same arithmetic in every batching."""
        base = self.results[0]["loss"]
        for flush in FLUSH_SETTINGS:
            with self.subTest(auto_flush_ops=flush):
                self.assertEqual(
                    self.results[flush]["loss"], base,
                    "auto_flush_ops=%d moved the loss" % flush)

    def test_the_gradient_does_not_depend_on_the_flag(self):
        """The half a crash-only check would miss.

        A fix that stops the segfault while leaving the backward reading the
        wrong bytes passes a "did it run" test and fails here, which is the
        point: the loss agrees to the last bit in that case too.
        """
        base = self.results[0]["grad_norm"]
        for flush in FLUSH_SETTINGS:
            with self.subTest(auto_flush_ops=flush):
                got = self.results[flush]["grad_norm"]
                self.assertLessEqual(
                    abs(got - base) / max(abs(base), 1e-9), GRADIENT_TOLERANCE,
                    "auto_flush_ops=%d gradient norm %s against %s at 0"
                    % (flush, got, base))


if __name__ == "__main__":
    unittest.main()
