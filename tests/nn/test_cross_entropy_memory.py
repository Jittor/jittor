# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""`cross_entropy_loss` keeps its numbers and stops carrying [N, C] copies.

The loss used to be a composite -- a one-hot from `index(1) == target`, then
`x - max`, `exp`, `x * onehot` -- whose autodiff kept or rebuilt several
tensors the size of the logits, plus an [N, C] int32 index grid. With C a
vocabulary (Qwen3: 151936) each is over a gigabyte: a 2048-row loss peaked at
4.93 GB live against PyTorch's 4.64 GB, and a Qwen3-0.6B 4x512 training step
no longer fit a 24 GB card. It is now a Function with a closed-form backward,
peaking at the logits plus their gradient.

The reference is the float64 closed form in numpy -- log-softmax and
``softmax - onehot`` -- so nothing here compares jittor with itself.
"""

from _helpers import capability as _test_capability
from _helpers.child_process import run_child_script

import textwrap
import unittest

import numpy as np

import jittor as jt
from jittor import nn


_DEVICES = [("cpu", 0)] + (
    [("cuda", 1)] if _test_capability.check_accelerator("cuda", backend=jt).enabled
    else [])


def _reference(x, target, weight=None, ignore_index=None, reduction="mean"):
    """Loss and d(loss)/dx in float64, torch's semantics."""
    x = x.astype(np.float64)
    n, c = x.shape
    valid = (target >= 0) & (target < c)
    w = valid.astype(np.float64)
    if weight is not None:
        w = w * weight[np.clip(target, 0, c - 1)]
    if ignore_index is not None:
        w = np.where(target == ignore_index, 0.0, w)
    shifted = x - x.max(1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    safe = np.where(w != 0, target, 0)
    per_row = -logp[np.arange(n), safe] * w
    onehot = np.zeros_like(x)
    onehot[np.arange(n), safe] = 1.0
    drow = (np.exp(logp) - onehot) * w[:, None]
    if reduction == "sum":
        return per_row.sum(), drow
    if reduction == "none":
        return per_row, drow
    return per_row.sum() / w.sum(), drow / w.sum()


class TestCrossEntropyMatchesTheClosedForm(unittest.TestCase):
    def _check(self, x, target, rtol=1e-5, atol=1e-6, **kw):
        want_loss, want_grad = _reference(x, target, **kw)
        weight = kw.pop("weight", None)
        for device, use_cuda in _DEVICES:
            with self.subTest(device=device, **{k: str(v) for k, v in kw.items()}):
                with jt.flag_scope(use_cuda=use_cuda):
                    jx = jt.array(x)
                    loss = nn.cross_entropy_loss(
                        jx, jt.array(target),
                        weight=None if weight is None else jt.array(weight), **kw)
                    seed = loss if loss.ndim == 0 else loss.sum()
                    (dx,) = jt.grad(seed, [jx])
                    np.testing.assert_allclose(loss.numpy(), want_loss,
                                               rtol=rtol, atol=atol)
                    np.testing.assert_allclose(dx.numpy(), want_grad,
                                               rtol=rtol, atol=atol)

    def test_plain(self):
        rng = np.random.RandomState(0)
        self._check(rng.randn(33, 17).astype("float32"), rng.randint(0, 17, 33))

    def test_ignore_index_and_out_of_range_targets(self):
        rng = np.random.RandomState(1)
        target = rng.randint(0, 17, 33)
        target[[3, 9]] = -100
        self._check(rng.randn(33, 17).astype("float32"), target,
                    ignore_index=-100)

    def test_class_weights_and_every_reduction(self):
        rng = np.random.RandomState(2)
        x = rng.randn(33, 17).astype("float32")
        target = rng.randint(0, 17, 33)
        weight = rng.rand(17).astype("float32")
        for reduction in ("mean", "sum", "none"):
            self._check(x, target, weight=weight, reduction=reduction)

    def test_spatial_input(self):
        rng = np.random.RandomState(3)
        x = rng.randn(2, 5, 3, 4).astype("float32")
        target = rng.randint(0, 5, (2, 3, 4))
        flat = x.transpose(0, 2, 3, 1).reshape(-1, 5)
        want_loss, want_grad = _reference(flat, target.reshape(-1))
        want_grad = want_grad.reshape(2, 3, 4, 5).transpose(0, 3, 1, 2)
        for device, use_cuda in _DEVICES:
            with self.subTest(device=device):
                with jt.flag_scope(use_cuda=use_cuda):
                    jx = jt.array(x)
                    loss = nn.cross_entropy_loss(jx, jt.array(target))
                    (dx,) = jt.grad(loss, [jx])
                    np.testing.assert_allclose(loss.numpy(), want_loss, rtol=1e-5)
                    np.testing.assert_allclose(dx.numpy(), want_grad,
                                               rtol=1e-5, atol=1e-7)

    def test_half_inputs_compute_in_float32(self):
        # The reference is taken on the *rounded* input, so what is measured is
        # the loss's own arithmetic; the gradient comes back in the input dtype.
        rng = np.random.RandomState(4)
        base = rng.randn(33, 17).astype("float32")
        target = rng.randint(0, 17, 33)
        for name, tol in (("float16", 2e-3), ("bfloat16", 1.6e-2)):
            for device, use_cuda in _DEVICES:
                with self.subTest(dtype=name, device=device):
                    with jt.flag_scope(use_cuda=use_cuda):
                        jx = jt.array(base).cast(name)
                        rounded = jx.float32().numpy()
                        want_loss, want_grad = _reference(rounded, target)
                        loss = nn.cross_entropy_loss(jx, jt.array(target))
                        (dx,) = jt.grad(loss, [jx])
                        self.assertEqual(str(dx.dtype), name)
                        np.testing.assert_allclose(loss.numpy(), want_loss,
                                                   rtol=1e-5)
                        np.testing.assert_allclose(dx.float32().numpy(),
                                                   want_grad, rtol=tol, atol=tol)

    def test_a_retained_graph_backwards_twice(self):
        rng = np.random.RandomState(5)
        x = rng.randn(8, 6).astype("float32")
        target = rng.randint(0, 6, 8)
        _, want_grad = _reference(x, target)
        jx = jt.array(x)
        loss = nn.cross_entropy_loss(jx, jt.array(target))
        first = jt.grad(loss, [jx], retain_graph=True)[0].numpy()
        second = jt.grad(loss, [jx])[0].numpy()
        np.testing.assert_allclose(first, want_grad, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(second, want_grad, rtol=1e-5, atol=1e-7)


#: The peak is a process-wide maximum, so each measurement is its own process.
PEAK_CHILD = textwrap.dedent('''
    import numpy as np, jittor as jt
    from jittor import nn
    jt.flags.use_cuda = 1
    n, c = 512, 32768
    x = jt.random((n, c))
    target = jt.array(np.random.RandomState(0).randint(0, c, n))
    jt.sync_all(True)
    with jt.flag_scope(profile_memory_enable=2):
        loss = nn.cross_entropy_loss(x, target)
        (dx,) = jt.grad(loss, [x])
        dx.sync()
        jt.sync_all(True)
        peak = int(jt.get_max_memory_info().split("[!@#div1!@#]")[0])
    print("RESULT %d %d" % (peak, n * c * 4))
''')


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no CUDA device")
class TestCrossEntropyPeak(unittest.TestCase):
    def test_the_backward_holds_the_logits_and_their_gradient_only(self):
        result = run_child_script(PEAK_CHILD, text=True, timeout=1800,
                                  name="cross_entropy_peak",
                                  env={"JITTOR_ARGS": ""}, crash_isolated=True)
        line = next((row for row in (result.stdout or "").splitlines()
                     if row.startswith("RESULT ")), None)
        self.assertIsNotNone(line, (result.stderr or result.stdout)[-2000:])
        peak, logits = (int(v) for v in line.split()[1:])
        # The input and its gradient are 2x; the composite peaked at ~4.25x.
        self.assertLessEqual(
            peak, 2.5 * logits,
            "cross-entropy backward peaked at %.2fx the logits" % (peak / logits))


if __name__ == "__main__":
    unittest.main()
