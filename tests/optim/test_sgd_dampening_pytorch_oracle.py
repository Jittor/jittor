# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# SGD momentum/dampening (jittor-core-gaps.md §3.3): the momentum buffer
# must be seeded from the raw gradient on its first touch, and dampening
# must have zero effect when momentum is off. Cross-checked here against an
# independent, real PyTorch process (not this repo's own torch-compat
# shim, which is jittor's own implementation under another name and so
# cannot serve as an independent reference for a jittor bug).
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import jittor as jt


def _oracle_env():
    """The oracle's own environment: the suite's PYTHONPATH puts Jittor's
    torch shim first, which would make the "real" interpreter import it."""
    return {k: v for k, v in os.environ.items()
            if k not in ("PYTHONPATH", "JITTOR_TORCH_SHIM")}


def _real_torch_python():
    """Locate a Python interpreter with real (non-shim) PyTorch installed,
    isolated from this repo's own jittor/jittor-torch. Mirrors
    tests/linalg/test_complex_spectral_gradients.py's helper of the same
    name/contract."""
    candidates = []
    override = os.environ.get("JITTOR_TEST_REAL_TORCH_PYTHON")
    if override:
        candidates.append(override)
    candidates.append("/opt/real-torch-venv/bin/python3")
    for python in candidates:
        if not Path(python).exists():
            continue
        probe = subprocess.run(
            [python, "-c", "import torch; assert not hasattr(torch, 'compat')"],
            capture_output=True, timeout=60, env=_oracle_env())
        if probe.returncode == 0:
            return python
    raise RuntimeError(
        "no interpreter with real (non-shim) PyTorch found; set "
        "JITTOR_TEST_REAL_TORCH_PYTHON to one, or skip this test")


def _torch_sgd_three_steps(p0, g0, lr, momentum, dampening):
    """Independent oracle: three real torch.optim.SGD steps with a FIXED
    (repeated) gradient every step, matching this repo's own repro
    convention (a stand-in for a real per-step loss/backward)."""
    python = _real_torch_python()
    script = f"""
import torch, json
p = torch.tensor([{p0}], requires_grad=True)
opt = torch.optim.SGD([p], lr={lr}, momentum={momentum}, dampening={dampening})
vals = []
for _ in range(3):
    opt.zero_grad()
    p.grad = torch.tensor([{g0}])
    opt.step()
    vals.append(float(p.detach()[0]))
print(json.dumps(vals))
"""
    # 180s, not 60: this file's own tests run several real-torch child
    # processes back to back while jittor's own CUDA JIT compiles may still
    # be finishing in the same container, and a plain `torch.optim.SGD`
    # subprocess occasionally queues behind that -- seen directly (13s in
    # isolation, timed out at 60s when run after two CUDA-compiling
    # siblings in the same file).
    out = subprocess.run([python, "-c", script], check=True,
                          capture_output=True, timeout=180, text=True,
                          env=_oracle_env())
    return json.loads(out.stdout)


class TestSGDDampeningIndependentOracle(unittest.TestCase):
    def setUp(self):
        try:
            _real_torch_python()
        except RuntimeError as e:
            self.skipTest(str(e))

    def _check(self, momentum, dev):
        p0, g0, lr, damp = 1.0, 2.0, 0.1, 0.2
        expect = _torch_sgd_three_steps(p0, g0, lr, momentum, damp)

        p = jt.array([p0])
        opt = jt.optim.SGD([p], lr=lr, momentum=momentum, dampening=damp)
        got = []
        for _ in range(3):
            g = jt.array([g0])
            loss = (p * g).sum()
            opt.step(loss)
            got.append(float(p.numpy()[0]))
        np.testing.assert_allclose(got, expect, atol=1e-5, rtol=1e-5,
                                    err_msg=f"SGD dampening vs real PyTorch, "
                                            f"momentum={momentum} [{dev}]")

    def test_momentum_zero_matches_real_pytorch(self):
        for dev in ("cpu", "cuda"):
            if dev == "cuda" and not jt.has_cuda:
                continue
            with jt.flag_scope(use_cuda=(dev == "cuda")):
                self._check(0.0, dev)

    def test_momentum_nonzero_matches_real_pytorch(self):
        for dev in ("cpu", "cuda"):
            if dev == "cuda" and not jt.has_cuda:
                continue
            with jt.flag_scope(use_cuda=(dev == "cuda")):
                self._check(0.8, dev)

    def test_cuda_fused_matches_real_pytorch(self):
        if not jt.has_cuda:
            self.skipTest("no CUDA found")
        p0, g0, lr, damp = 1.0, 2.0, 0.1, 0.2
        with jt.flag_scope(use_cuda=1):
            for momentum in (0.0, 0.8):
                expect = _torch_sgd_three_steps(p0, g0, lr, momentum, damp)
                p = jt.array([p0])
                opt = jt.optim.SGD([p], lr=lr, momentum=momentum, dampening=damp,
                                   fused=True)
                got = []
                for _ in range(3):
                    g = jt.array([g0])
                    loss = (p * g).sum()
                    opt.step(loss)
                    got.append(float(p.numpy()[0]))
                np.testing.assert_allclose(
                    got, expect, atol=1e-5, rtol=1e-5,
                    err_msg=f"CUDA fused SGD dampening vs real PyTorch, "
                            f"momentum={momentum}")


def _torch_trajectory(body):
    """Run `body` (torch API, defines `params` and `run()`) in real PyTorch.

    `run()` yields once per step; each yield records every parameter and
    its momentum buffer (None until torch creates one).
    """
    python = _real_torch_python()
    script = "import torch, json\n" + body + """
rows = []
for _ in run():
    row = []
    for q in params:
        row.append(q.detach().reshape(-1).tolist())
        buf = opt.state.get(q, {}).get("momentum_buffer")
        row.append(None if buf is None else buf.reshape(-1).tolist())
    rows.append(row)
print(json.dumps(rows))
"""
    out = subprocess.run([python, "-c", script], check=True,
                         capture_output=True, timeout=180, text=True,
                         env=_oracle_env())
    return json.loads(out.stdout)


def _jittor_row(params, opt):
    row = []
    flags = {}
    for pg in opt.param_groups:
        for q, v, initialized in zip(pg["params"], pg["values"],
                                     pg["momentum_initialized"]):
            flags[id(q)] = v.numpy().reshape(-1).tolist() if initialized else None
    for q in params:
        row.append(q.numpy().reshape(-1).tolist())
        row.append(flags[id(q)])
    return row


def _assert_rows(test, got, expect, msg):
    test.assertEqual(len(got), len(expect), msg)
    for step, (g_row, e_row) in enumerate(zip(got, expect)):
        for k, (g, e) in enumerate(zip(g_row, e_row)):
            what = f"{msg}: step {step} {'buffer' if k % 2 else 'param'} {k // 2}"
            if e is None or g is None:
                test.assertEqual(g, e, what)
            else:
                np.testing.assert_allclose(g, e, rtol=1e-5, atol=1e-6, err_msg=what)


# Momentum's first update belongs to each buffer, not to the optimizer's or
# the group's first step: these buffers all start after the group stepped.
_FIRST_TOUCH_CASES = {
    "momentum_turned_on": """
p = torch.tensor([1.0], requires_grad=True)
params = [p]
opt = torch.optim.SGD(params, lr=0.1, momentum=0.0, dampening=0.2)
def run():
    for mom in (0.0, 0.8, 0.8, 0.8):
        opt.param_groups[0]["momentum"] = mom
        opt.zero_grad(); (p * 2.0).sum().backward(); opt.step()
        yield
""",
    "late_unfreeze": """
a = torch.tensor([1.0, -1.0], requires_grad=True)
b = torch.tensor([0.5, 2.0], requires_grad=False)
params = [a, b]
opt = torch.optim.SGD(params, lr=0.1, momentum=0.9, dampening=0.3)
def run():
    for i in range(4):
        if i == 2:
            b.requires_grad_(True)
        loss = (a * torch.tensor([1.0, 3.0])).sum()
        if b.requires_grad:
            loss = loss + (b * b).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        yield
""",
}


def _jittor_first_touch(case, fused):
    if case == "momentum_turned_on":
        p = jt.array([1.0])
        params = [p]
        opt = jt.optim.SGD(params, lr=0.1, momentum=0.0, dampening=0.2, fused=fused)
        rows = []
        for mom in (0.0, 0.8, 0.8, 0.8):
            opt.param_groups[0]["momentum"] = mom
            opt.step((p * 2.0).sum())
            rows.append(_jittor_row(params, opt))
        return rows
    a = jt.array([1.0, -1.0])
    b = jt.array([0.5, 2.0]).stop_grad()
    params = [a, b]
    opt = jt.optim.SGD(params, lr=0.1, momentum=0.9, dampening=0.3, fused=fused)
    rows = []
    for i in range(4):
        if i == 2:
            b.start_grad()
        loss = (a * jt.array([1.0, 3.0])).sum()
        if i >= 2:
            loss = loss + (b * b).sum()
        opt.step(loss)
        rows.append(_jittor_row(params, opt))
    return rows


class TestSGDMomentumFirstTouchIndependentOracle(unittest.TestCase):
    def setUp(self):
        try:
            _real_torch_python()
        except RuntimeError as e:
            self.skipTest(str(e))

    def _devices(self):
        yield "cpu", False
        if jt.has_cuda:
            yield "cuda", False      # portable update on the device
            yield "cuda", None       # CUDA fused kernel (float32)

    def test_first_touch_matches_real_pytorch(self):
        for case, body in _FIRST_TOUCH_CASES.items():
            expect = _torch_trajectory(body)
            for dev, fused in self._devices():
                with jt.flag_scope(use_cuda=(dev == "cuda")):
                    got = _jittor_first_touch(case, fused)
                _assert_rows(self, got, expect, f"{case} [{dev}, fused={fused}]")

    def test_native_checkpoint_keeps_buffer_state_while_momentum_paused(self):
        expect = _torch_trajectory("""
p = torch.tensor([1.0, -2.0], requires_grad=True)
params = [p]
opt = torch.optim.SGD(params, lr=0.1, momentum=0.9, dampening=0.1)
def run():
    for mom in (0.9, 0.9, 0.0, 0.9, 0.9):
        opt.param_groups[0]["momentum"] = mom
        opt.zero_grad(); (p * p).sum().backward(); opt.step()
        yield
""")
        for dev, fused in self._devices():
            with jt.flag_scope(use_cuda=(dev == "cuda")):
                p = jt.array([1.0, -2.0])
                opt = jt.optim.SGD([p], lr=0.1, momentum=0.9, dampening=0.1, fused=fused)
                rows = []
                for mom in (0.9, 0.9, 0.0):
                    opt.param_groups[0]["momentum"] = mom
                    opt.step((p * p).sum())
                    rows.append(_jittor_row([p], opt))
                saved = opt.state_dict()
                p2 = jt.array(p.numpy())
                opt2 = jt.optim.SGD([p2], lr=0.1, momentum=0.0, dampening=0.1, fused=fused)
                opt2.load_state_dict(saved)
                for _ in range(2):
                    opt2.param_groups[0]["momentum"] = 0.9
                    opt2.step((p2 * p2).sum())
                    rows.append(_jittor_row([p2], opt2))
            _assert_rows(self, rows, expect, f"paused checkpoint [{dev}, fused={fused}]")


class TestSGDLegacyCheckpointMigration(unittest.TestCase):
    """A state written before the per-buffer flags resumes as its writer would."""

    def _legacy_state(self, n_step):
        p = jt.array([1.0])
        opt = jt.optim.SGD([p], lr=0.1, momentum=0.5, dampening=0.25)
        state = opt.state_dict()
        group = state["defaults"]["param_groups"][0]
        del group["momentum_initialized"]
        group["values"] = [jt.array([4.0])]
        if n_step is None:
            group.pop("n_step", None)
        else:
            group["n_step"] = n_step
        return state

    def _resume(self, state):
        p = jt.array([1.0])
        opt = jt.optim.SGD([p], lr=0.1, momentum=0.5, dampening=0.25)
        opt.load_state_dict(state)
        opt.step((p * 2.0).sum())                      # grad 2
        return float(opt.param_groups[0]["values"][0].numpy()[0])

    def test_group_that_had_stepped_continues_its_recurrence(self):
        # 0.5 * 4 + 0.75 * 2: the writer was past its first step.
        self.assertAlmostEqual(self._resume(self._legacy_state(3)), 3.5, places=6)

    def test_state_from_before_step_counting_continues_its_recurrence(self):
        self.assertAlmostEqual(self._resume(self._legacy_state(None)), 3.5, places=6)

    def test_group_saved_before_any_step_seeds_from_the_gradient(self):
        self.assertAlmostEqual(self._resume(self._legacy_state(0)), 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
