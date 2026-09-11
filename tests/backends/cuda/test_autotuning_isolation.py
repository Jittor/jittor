# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Only the autotuner may move the gradient when a scheduling knob moves.

Convolution algorithms are chosen by *measuring* the candidates and caching the
winner per shape. The measurement depends on what else is resident, and the
cache falls back to cuDNN's heuristic once it is full -- so which algorithm a
shape gets depends on how many other shapes have been seen and what was in
memory at the time. Neither has anything to do with the arithmetic being asked
for, and both are moved by flags that read as scheduling knobs:
`auto_flush_ops` changes what is resident, `set_algorithm_cache_size` changes
when the fallback starts (KI-EXEC-003).

Measured: the same model and input, gradient norm 45839.379 at
`auto_flush_ops` 0 and 45822.207 at 64 -- `3.8e-4` apart, deterministic, while
the **loss agrees to the last bit**. Any check watching the loss sees nothing.

What this file pins is the boundary, not the behaviour:

* with the autotuner off, the gradient must not depend on the scheduling flag
  at all -- that is the assertion with teeth, and it is what would go red if
  something *other* than autotuning started moving the numbers;
* the loss must be bit-identical either way, autotuner or not.

It deliberately does **not** assert that the gradients differ with the
autotuner on. That divergence depends on the machine, the shapes and the
cuDNN version; asserting it would make this file fail on hardware where the
measurement happens to be stable, which is not a defect.
"""

from _helpers import capability as _test_capability
from _helpers.child_process import run_child_script

import json
import textwrap
import unittest

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


#: Two settings that produced different algorithm choices on the machine where
#: this was written. Any pair would do; these are the measured ones.
FLUSH_SETTINGS = (0, 64)

#: Well above the `2e-6` reassociation different batch boundaries produce, well
#: below the `3.8e-4` an algorithm change produces.
TOLERANCE = 1e-5

CHILD = textwrap.dedent('''
    import json, numpy as np, jittor as jt
    from jittor import nn
    jt.flags.use_cuda = 1
    jt.flags.auto_flush_ops = {flush}
    import jittor.compile_extern as ce
    ce.cudnn.set_benchmark({benchmark})

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

    model = nn.Sequential(*([BK(512, 256, stride=2)] + [BK(1024, 256) for _ in range(4)]))
    x = jt.array(np.random.RandomState(7).randn(1, 512, 8, 8).astype("float32"))
    loss = model(x).sum()
    grads = jt.grad(loss, [p for p in model.parameters() if p.requires_grad])
    flat = np.concatenate([np.asarray(g.numpy()).ravel() for g in grads])
    print("RESULT " + json.dumps({{
        "loss": float(loss.numpy()),
        "grad_norm": float(np.linalg.norm(flat)),
    }}))
''')


def _run(flush, benchmark):
    result = run_child_script(
        CHILD.format(flush=flush, benchmark=benchmark),
        text=True, timeout=2400, crash_isolated=True,
        name="autotune_%d_%d" % (flush, benchmark))
    line = next((l for l in (result.stdout or "").splitlines()
                 if l.startswith("RESULT ")), None)
    if line is None:
        raise AssertionError(
            "auto_flush_ops=%d benchmark=%d produced no result (%s)\n%s"
            % (flush, benchmark, result.returncode,
               (result.stderr or result.stdout or "")[-2000:]))
    return json.loads(line[len("RESULT "):])


@unittest.skipIf(not _has_cuda(), "no CUDA device")
class TestAutotuningIsolation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.off = {f: _run(f, 0) for f in FLUSH_SETTINGS}
        cls.on = {f: _run(f, 1) for f in FLUSH_SETTINGS}

    def test_without_the_autotuner_the_gradient_is_flag_independent(self):
        """The assertion with teeth.

        With algorithm selection held still, nothing else may move the
        gradient when the scheduling flag moves. If this goes red, the cause
        is no longer the autotuner and KI-EXEC-003 is not the whole story.
        """
        base = self.off[FLUSH_SETTINGS[0]]["grad_norm"]
        for flush in FLUSH_SETTINGS:
            with self.subTest(auto_flush_ops=flush):
                got = self.off[flush]["grad_norm"]
                self.assertLessEqual(
                    abs(got - base) / max(abs(base), 1e-9), TOLERANCE,
                    "with cuDNN autotuning off, auto_flush_ops=%d moved the "
                    "gradient norm to %s from %s" % (flush, got, base))

    def test_the_loss_is_bit_identical_either_way(self):
        """Why the divergence is invisible: the forward never moves."""
        base = self.off[FLUSH_SETTINGS[0]]["loss"]
        for table, label in ((self.off, "autotuner off"), (self.on, "autotuner on")):
            for flush in FLUSH_SETTINGS:
                with self.subTest(mode=label, auto_flush_ops=flush):
                    self.assertEqual(
                        table[flush]["loss"], base,
                        "%s, auto_flush_ops=%d moved the loss" % (label, flush))


if __name__ == "__main__":
    unittest.main()
