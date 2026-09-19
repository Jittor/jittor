"""float16/bfloat16 numerics through ``import torch``, not the API surface.

``test_torch_amp_fidelity.py`` next door pins the ``torch.amp`` API: the names,
the signatures, the autocast state record. This module asks the other half of
the question -- whether a half tensor that goes through the shim comes back with
the right *dtype* and the right *value*.

Every reference here is independent of jittor: a float64 closed form evaluated
on the same rounded input, or a number measured from real PyTorch 2.13 and
quoted in the comment beside the assertion. Nothing compares jittor to itself.

Three things this caught, all of them visible only through values:

* ``.bfloat16()`` truncated on CPU instead of rounding to nearest even, so the
  shim disagreed with torch *and* with its own CUDA path: ``0.1`` came back
  0.099609375 on the host against torch's 0.100097656.
* ``x.sum()``, ``x @ y`` and ``F.layer_norm`` accumulated in half where ATen
  accumulates in float32. A 4096-long float16 sum of ones answered 2048.
* ``x.max()``/``.min()``/``.prod()``/``.cumprod()``/``.logsumexp()`` came back
  float32 for a half input, silently widening the graph.

Run::  JITTOR_TORCH_SHIM=1 python -m pytest compat/tests/torch/test_torch_half_precision_numerics.py
"""

from _helpers import capability as _test_capability
import unittest

import numpy as np
import torch
import jittor as jt


_DEVICES = ["cpu"] + (["cuda"] if _test_capability.any_accelerator_enabled(backend=jt) else [])

_ULP = {"float16": 2.0 ** -11, "bfloat16": 2.0 ** -8}
_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def _bf16_round(values):
    """fp32 -> bfloat16 -> fp32 with round-to-nearest-even, in numpy.

    numpy has no bfloat16 and the rounding rule is what is under test, so this
    cannot borrow the library's own conversion.
    """
    bits = np.asarray(values, dtype=np.float32).view(np.uint32).copy()
    bits += np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))
    return (bits & np.uint32(0xffff0000)).view(np.float32)


def _round_trip(values, name):
    if name == "float16":
        return np.asarray(values, np.float32).astype(np.float16).astype(np.float64)
    return _bf16_round(values).astype(np.float64)


def _np(t):
    return t.detach().float().cpu().numpy().astype(np.float64)


class TestHalfConversion(unittest.TestCase):
    """``.bfloat16()`` and ``.half()`` round the way torch rounds."""

    CASES = [1.0 + 2 ** -9, 1.0 + 3 * 2 ** -10, 0.1, 3.14159265,
             1.0 - 2 ** -10, 255.7, 1e-3, 65792.0, -0.1, -255.7]

    def test_bfloat16_matches_round_to_nearest_even(self):
        values = np.array(self.CASES, dtype=np.float32)
        expect = _bf16_round(values)
        for device in _DEVICES:
            with self.subTest(device=device):
                got = _np(torch.tensor(values, device=device).bfloat16())
                # Exact: real torch 2.13 returns exactly these values on both
                # of its devices. Truncation gives 0.099609375 for 0.1 and 255
                # for 255.7, both a full ULP low and both biased toward zero.
                np.testing.assert_array_equal(got, expect.astype(np.float64))

    def test_float16_matches_round_to_nearest_even(self):
        values = np.array(self.CASES, dtype=np.float32)
        expect = values.astype(np.float16).astype(np.float64)
        for device in _DEVICES:
            with self.subTest(device=device):
                np.testing.assert_array_equal(
                    _np(torch.tensor(values, device=device).half()), expect)

    def test_host_and_device_agree(self):
        if len(_DEVICES) < 2:
            raise unittest.SkipTest("no accelerator enabled")
        values = np.array(self.CASES, dtype=np.float32)
        for name in ("bfloat16", "float16"):
            with self.subTest(dtype=name):
                host = _np(torch.tensor(values).to(_DTYPES[name]))
                dev = _np(torch.tensor(values, device="cuda").to(_DTYPES[name]))
                np.testing.assert_array_equal(host, dev)


def _dtype_sweep():
    F = torch.nn.functional
    return [
        ("sum", lambda t: t.sum(-1)),
        ("mean", lambda t: t.mean(-1)),
        ("prod", lambda t: t.prod(-1)),
        ("max", lambda t: t.max(-1)[0]),
        ("min", lambda t: t.min(-1)[0]),
        ("std", lambda t: t.std(-1)),
        ("norm", lambda t: t.norm(dim=-1)),
        ("cumsum", lambda t: t.cumsum(-1)),
        ("cumprod", lambda t: t.cumprod(-1)),
        ("logsumexp", lambda t: t.logsumexp(-1)),
        ("softmax", lambda t: torch.softmax(t, -1)),
        ("log_softmax", lambda t: torch.log_softmax(t, -1)),
        ("layer_norm", lambda t: F.layer_norm(t, (t.shape[-1],))),
        ("matmul", lambda t: t @ t.t()),
        ("relu", lambda t: F.relu(t)),
        ("gelu", lambda t: F.gelu(t)),
        ("sigmoid", lambda t: t.sigmoid()),
        ("sort", lambda t: t.sort(-1)[0]),
        ("topk", lambda t: t.topk(2, -1)[0]),
        ("cat", lambda t: torch.cat([t, t], -1)),
    ]


class TestHalfDtypeIsPreserved(unittest.TestCase):
    """Real torch 2.13 returns the input's dtype from every one of these.

    Checked against the real library, not assumed. Five came back float32 here:
    ``max``, ``min``, ``prod`` and ``cumprod``, because jittor's
    ``reduce_dtype_infer`` widened any half float reduce that ``ReduceOp``'s
    float32-intermediate path did not intercept, and ``logsumexp``, whose
    ``exp`` is on jittor's white list.
    """

    def test_sweep(self):
        base = (np.random.RandomState(0).rand(4, 8) + 0.5).astype(np.float32)
        for device in _DEVICES:
            for name, dtype in _DTYPES.items():
                t = torch.tensor(base, device=device).to(dtype)
                for op, fn in _dtype_sweep():
                    with self.subTest(device=device, dtype=name, op=op):
                        out = fn(t)
                        self.assertEqual(
                            str(out.dtype), "torch." + name,
                            "%s on %s widened %s to %s"
                            % (op, device, name, out.dtype))


class TestHalfAccumulatesInFloat32(unittest.TestCase):
    """ATen accumulates a half reduction in float32 and rounds once.

    ``ones(n).sum()`` is the sharpest form: in float16 a running sum stops
    moving at 2048, because 2048 + 1 rounds back to 2048, so a half accumulator
    answers 2048 for every n >= 2048 -- real torch answers n. bfloat16 stops at
    256.
    """

    SATURATES = {"float16": 2048.0, "bfloat16": 256.0}

    def test_sum_of_ones(self):
        n = 4096
        for device in _DEVICES:
            for name, dtype in _DTYPES.items():
                with self.subTest(device=device, dtype=name):
                    total = float(torch.ones(n, device=device, dtype=dtype).sum())
                    self.assertNotAlmostEqual(total, self.SATURATES[name], delta=1.0)
                    self.assertEqual(total, float(n))

    def test_cumsum_of_ones(self):
        n = 4096
        for device in _DEVICES:
            for name, dtype in _DTYPES.items():
                with self.subTest(device=device, dtype=name):
                    out = torch.ones(n, device=device, dtype=dtype).cumsum(-1)
                    self.assertEqual(str(out.dtype), "torch." + name)
                    self.assertEqual(float(out[-1]), float(n))

    def test_matmul_contraction(self):
        k = 4096
        for device in _DEVICES:
            for name, dtype in _DTYPES.items():
                with self.subTest(device=device, dtype=name):
                    a = torch.ones((1, k), device=device, dtype=dtype)
                    b = torch.ones((k, 1), device=device, dtype=dtype)
                    out = a @ b
                    self.assertEqual(str(out.dtype), "torch." + name)
                    self.assertEqual(float(out[0, 0]), float(k))

    def test_layer_norm_statistics(self):
        """Measured error against the float64 closed form on the same input.

        Real torch 2.13 on this input: 9.750e-4 relative for float16 and
        3.867e-3 for bfloat16 -- both under 2 ULP of their own format, which is
        what a float32 accumulator with one final rounding gives. Computing the
        statistics at the input's width measured 3.02e-3 and 4.02e-2.
        """
        data = (np.random.RandomState(0).randn(16, 1024) * 2 + 1).astype(np.float32)
        for device in _DEVICES:
            for name, dtype in _DTYPES.items():
                with self.subTest(device=device, dtype=name):
                    exact = _round_trip(data, name)
                    truth = ((exact - exact.mean(-1, keepdims=True))
                             / np.sqrt(exact.var(-1, keepdims=True) + 1e-5))
                    out = torch.nn.functional.layer_norm(
                        torch.tensor(data, device=device).to(dtype), (1024,))
                    err = float(np.max(np.abs(_np(out) - truth)
                                       / np.maximum(np.abs(truth), 1e-30)))
                    self.assertLessEqual(
                        err, 2.0 * _ULP[name],
                        "layer_norm %s on %s: %.3e, bound %.3e (2 ULP)"
                        % (name, device, err, 2.0 * _ULP[name]))


# ---------------------------------------------------------------------------
# End-to-end: autocast + GradScaler.
# ---------------------------------------------------------------------------

STEPS = 80
LR = 0.005


def _task(n=256, d=16, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    w = rng.randn(d, 1).astype(np.float32)
    return x, (x @ w + 0.1 * rng.randn(n, 1)).astype(np.float32)


class _MLP(torch.nn.Module):
    def __init__(self, d=16, h=32):
        super().__init__()
        self.l1 = torch.nn.Linear(d, h)
        self.l2 = torch.nn.Linear(h, h)
        self.norm = torch.nn.LayerNorm(h)
        self.l3 = torch.nn.Linear(h, 1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.norm(self.l2(x))
        return self.l3(torch.relu(x))


def _seed_parameters(model, seed=1):
    rng = np.random.RandomState(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.tensor((rng.randn(*tuple(p.shape)) * 0.1).astype(np.float32)))


def _train(mode, device):
    """(losses, parameters stayed finite, fraction of steps with finite grads)."""
    torch.manual_seed(0)
    x_np, y_np = _task()
    model = _MLP().to(device)
    _seed_parameters(model)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    x = torch.tensor(x_np, device=device)
    y = torch.tensor(y_np, device=device)
    scaler = torch.amp.GradScaler(device) if mode == "fp16" else None
    losses, finite_steps, params_finite = [], 0, True
    for _ in range(STEPS):
        opt.zero_grad()
        if mode == "fp32":
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            finite_steps += int(all(
                p.grad is None or bool(torch.isfinite(p.grad).all())
                for p in model.parameters()))
            opt.step()
        else:
            dtype = torch.bfloat16 if mode == "bf16" else torch.float16
            with torch.autocast(device_type=device, dtype=dtype):
                out = model(x)
                loss = ((out.float() - y) ** 2).mean()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if all(p.grad is None or bool(torch.isfinite(p.grad).all())
                       for p in model.parameters()):
                    finite_steps += 1
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                finite_steps += int(all(
                    p.grad is None or bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()))
                opt.step()
        losses.append(float(loss.detach().float().cpu()))
        params_finite = params_finite and all(
            bool(torch.isfinite(p).all()) for p in model.parameters())
    return losses, params_finite, finite_steps / float(STEPS)


class TestAutocastTraining(unittest.TestCase):
    """A model trained under autocast reaches float32's answer.

    Real torch 2.13 on this exact task, architecture, optimiser and schedule:
    float32 0.12238, ``autocast(bfloat16)`` 0.12289 (+0.4%), and
    ``autocast(float16)`` with a ``GradScaler`` 0.12652 (+3.4%). The 25% allowed
    below is 7x torch's own spread.

    ``GradScaler`` starts at 65536 and halves on the first overflow, so a few
    early steps legitimately produce non-finite gradients and are skipped --
    which is the mechanism, not a failure. What must hold is that the
    *parameters* never go non-finite and that the great majority of steps are
    taken.
    """

    TOLERANCE = 0.25

    def _baseline(self, device):
        losses, params_finite, finite_fraction = _train("fp32", device)
        self.assertTrue(params_finite)
        self.assertEqual(finite_fraction, 1.0)
        self.assertLess(losses[-1], losses[0] / 50.0,
                        "the float32 baseline itself did not train")
        return losses

    def test_float32_baseline(self):
        for device in _DEVICES:
            with self.subTest(device=device):
                self._baseline(device)

    def test_bfloat16_autocast(self):
        for device in _DEVICES:
            with self.subTest(device=device):
                base = self._baseline(device)
                losses, params_finite, finite_fraction = _train("bf16", device)
                self.assertTrue(params_finite)
                self.assertEqual(finite_fraction, 1.0,
                                 "a bfloat16 autocast step had a non-finite gradient")
                self.assertLess(losses[-1], losses[0] / 50.0)
                self.assertLess(
                    abs(losses[-1] - base[-1]) / base[-1], self.TOLERANCE,
                    "bfloat16 autocast reached %.5f against float32's %.5f on %s"
                    % (losses[-1], base[-1], device))

    def test_float16_autocast_with_grad_scaler(self):
        for device in _DEVICES:
            with self.subTest(device=device):
                base = self._baseline(device)
                losses, params_finite, finite_fraction = _train("fp16", device)
                self.assertTrue(params_finite,
                                "a parameter went non-finite under GradScaler")
                # The scaler calibrates down from 65536; real torch skips a
                # handful of early steps on this model for the same reason.
                self.assertGreaterEqual(
                    finite_fraction, 0.8,
                    "GradScaler skipped %d%% of steps -- it is not converging on "
                    "a usable scale" % round(100 * (1 - finite_fraction)))
                self.assertLess(losses[-1], losses[0] / 50.0)
                self.assertLess(
                    abs(losses[-1] - base[-1]) / base[-1], self.TOLERANCE,
                    "float16 autocast reached %.5f against float32's %.5f on %s"
                    % (losses[-1], base[-1], device))


if __name__ == "__main__":
    unittest.main()
