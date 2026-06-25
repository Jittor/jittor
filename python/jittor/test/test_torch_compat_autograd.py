"""Torch-grade autograd-semantics regression tests for ``import jittor as torch``.

Part of the torch-grade test-suite rewrite (round 3). Like ``test_torch_compat_ops.py``
this is a structured ``unittest`` module: every check compares jittor-as-torch's autograd
against the ANALYTIC gradient (computed by hand), and runs on BOTH CPU and CUDA (when the
build has it), so it locks autograd *semantics* rather than jittor self-consistency.

Covered: ``jt.grad`` / ``torch.autograd.grad`` / ``Var.backward`` basics; analytic grads
of ``a*b``, ``exp``, ``log``, ``matmul``, ``sum``, ``mean``, chained/composed expressions;
``stop_grad`` and ``no_grad``; gradient accumulation (sum of grads); backward through
``F.relu`` and ``softmax``; and HIGHER-ORDER (2nd) gradients via both ``jt.grad`` of a
grad and ``torch.autograd.grad(create_graph=True)``.

Notes:
  * jittor's ``jt.grad(loss, targets)`` returns a LIST even for a single target (the
    torch_compat shim coerces a lone Var target into ``[var]``); we index ``[0]``.
  * jittor has no 0-d scalars -- a reduced loss is shape ``(1,)``. ``jt.grad`` needs a
    scalar-ish loss, so we ``.sum()`` outputs before differentiating, exactly as torch
    users do.

Run:  python -m jittor.test.test_torch_compat_autograd
      python -m pytest python/jittor/test/test_torch_compat_autograd.py
"""
import unittest
import numpy as np
import jittor as torch          # the whole point: jittor IS torch here
import jittor as jt

# Exercise CPU always; add CUDA when the build has it. NPU(ACL) reports has_cuda too.
_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    """Run ``fn(device_name)`` once per available device under the right flag scope."""
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def grad1(loss, x, **kw):
    """jt.grad of a single target -> the grad Var (jt.grad always returns a list)."""
    g = jt.grad(loss, [x], **kw)
    return g[0]


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)


# ------------------------------------------------------------------ analytic gradients

class TestAnalyticGrad(Base):
    def test_mul_grad(self):
        # d/da sum(a*b) = b ;  d/db sum(a*b) = a
        rng = np.random.RandomState(0)
        a0 = rng.randn(4).astype("float32"); b0 = rng.randn(4).astype("float32")
        def body(dev):
            a = jt.array(a0); b = jt.array(b0)
            ga, gb = jt.grad((a * b).sum(), [a, b])
            self.ac(ga.numpy(), b0, atol=1e-6, msg=f"d(a*b)/da {dev}")
            self.ac(gb.numpy(), a0, atol=1e-6, msg=f"d(a*b)/db {dev}")
        both_devices(body)

    def test_exp_grad(self):
        # d/dx sum(exp(x)) = exp(x)
        x0 = np.random.RandomState(1).randn(5).astype("float32")
        def body(dev):
            x = jt.array(x0)
            self.ac(grad1(jt.exp(x).sum(), x).numpy(), np.exp(x0), atol=1e-5,
                    msg=f"d exp {dev}")
        both_devices(body)

    def test_log_grad(self):
        # d/dx sum(log(x)) = 1/x
        x0 = (np.random.RandomState(2).rand(5) + 0.5).astype("float32")
        def body(dev):
            x = jt.array(x0)
            self.ac(grad1(jt.log(x).sum(), x).numpy(), 1.0 / x0, atol=1e-5,
                    msg=f"d log {dev}")
        both_devices(body)

    def test_sum_grad(self):
        # d/dx sum(x) = ones
        x0 = np.random.RandomState(3).randn(3, 4).astype("float32")
        def body(dev):
            x = jt.array(x0)
            self.ac(grad1(x.sum(), x).numpy(), np.ones_like(x0), atol=1e-6,
                    msg=f"d sum {dev}")
        both_devices(body)

    def test_mean_grad(self):
        # d/dx mean(x) = 1/N
        x0 = np.random.RandomState(4).randn(6).astype("float32")
        def body(dev):
            x = jt.array(x0)
            self.ac(grad1(x.mean(), x).numpy(), np.full(6, 1.0 / 6, "float32"),
                    atol=1e-6, msg=f"d mean {dev}")
        both_devices(body)

    def test_matmul_grad(self):
        # y = A @ B ;  d/dA sum(y) = ones(NxK) @ B^T ;  d/dB sum(y) = A^T @ ones(NxK)
        rng = np.random.RandomState(5)
        A0 = rng.randn(3, 4).astype("float32"); B0 = rng.randn(4, 5).astype("float32")
        def body(dev):
            A = jt.array(A0); B = jt.array(B0)
            gA, gB = jt.grad(jt.matmul(A, B).sum(), [A, B])
            self.ac(gA.numpy(), np.ones((3, 5)) @ B0.T, atol=1e-5, msg=f"dA {dev}")
            self.ac(gB.numpy(), A0.T @ np.ones((3, 5)), atol=1e-5, msg=f"dB {dev}")
        both_devices(body)

    def test_pow_grad(self):
        # d/dx sum(x^3) = 3 x^2
        x0 = np.random.RandomState(6).randn(4).astype("float32")
        def body(dev):
            x = jt.array(x0)
            self.ac(grad1((x * x * x).sum(), x).numpy(), 3 * x0 ** 2, atol=1e-4,
                    msg=f"d x^3 {dev}")
        both_devices(body)

    def test_chain_grad(self):
        # d/dx sum(3*x^2) = 6x
        x0 = np.random.RandomState(7).randn(4).astype("float32")
        def body(dev):
            x = jt.array(x0)
            self.ac(grad1(((x * x) * 3).sum(), x).numpy(), 6 * x0, atol=1e-4,
                    msg=f"chain 6x {dev}")
        both_devices(body)

    def test_composed_grad(self):
        # f = sum(exp(2x) + log(x^2+1));  f' = 2 exp(2x) + 2x/(x^2+1)
        x0 = np.random.RandomState(8).randn(5).astype("float32")
        ref = 2 * np.exp(2 * x0) + 2 * x0 / (x0 ** 2 + 1)
        def body(dev):
            x = jt.array(x0)
            loss = (jt.exp(2 * x) + jt.log(x * x + 1)).sum()
            self.ac(grad1(loss, x).numpy(), ref, atol=1e-4, rtol=1e-4,
                    msg=f"composed {dev}")
        both_devices(body)


# -------------------------------------------------------------- autograd.grad / backward

class TestAutogradGradAPI(Base):
    def test_autograd_grad_single(self):
        x0 = np.random.RandomState(10).randn(4).astype("float32")
        def body(dev):
            x = jt.array(x0)
            (g,) = torch.autograd.grad((x * x).sum(), x)
            self.ac(g.numpy(), 2 * x0, atol=1e-5, msg=f"autograd.grad {dev}")
        both_devices(body)

    def test_autograd_grad_multi_inputs(self):
        rng = np.random.RandomState(11)
        a0 = rng.randn(3).astype("float32"); b0 = rng.randn(3).astype("float32")
        def body(dev):
            a = jt.array(a0); b = jt.array(b0)
            ga, gb = torch.autograd.grad((a * b + a).sum(), [a, b])
            self.ac(ga.numpy(), b0 + 1, atol=1e-5, msg=f"ga {dev}")
            self.ac(gb.numpy(), a0, atol=1e-5, msg=f"gb {dev}")
        both_devices(body)

    def test_autograd_grad_with_grad_outputs(self):
        # vector-Jacobian product: grad_outputs weights each output element.
        rng = np.random.RandomState(12)
        x0 = rng.randn(4).astype("float32"); w0 = rng.randn(4).astype("float32")
        def body(dev):
            x = jt.array(x0)
            y = x * x                              # dy_i/dx_i = 2 x_i
            (g,) = torch.autograd.grad(y, x, grad_outputs=jt.array(w0))
            self.ac(g.numpy(), w0 * 2 * x0, atol=1e-5, msg=f"vjp {dev}")
        both_devices(body)

    def test_backward_sets_grad(self):
        # torch-style: loss.backward() then read leaf .grad (no optimizer leaf path).
        x0 = np.random.RandomState(13).randn(4).astype("float32")
        def body(dev):
            x = jt.array(x0); x.requires_grad = True
            (x * x).sum().backward()
            self.assertIsNotNone(getattr(x, "grad", None), f"x.grad set {dev}")
            self.ac(x.grad.numpy(), 2 * x0, atol=1e-5, msg=f"backward grad {dev}")
        both_devices(body)


# ------------------------------------------------------------------------ stop / no_grad

class TestStopNoGrad(Base):
    def test_stop_grad_blocks(self):
        # loss = sum(stop_grad(2x) * x); only the bare x factor carries grad -> grad = 2x.
        x0 = np.random.RandomState(14).randn(4).astype("float32")
        def body(dev):
            x = jt.array(x0)
            y = x * 2
            loss = (y.stop_grad() * x).sum()
            self.ac(grad1(loss, x).numpy(), 2 * x0, atol=1e-5, msg=f"stop_grad {dev}")
        both_devices(body)

    def test_no_grad_detaches(self):
        # inside torch.no_grad(), produced tensors don't require grad.
        x0 = np.random.RandomState(15).randn(3).astype("float32")
        def body(dev):
            with torch.no_grad():
                x = jt.array(x0)
                z = x * 2
            self.assertFalse(bool(getattr(z, "requires_grad", False)),
                             f"no_grad detaches {dev}")
        both_devices(body)

    def test_is_stop_grad(self):
        x0 = np.random.RandomState(16).randn(3).astype("float32")
        def body(dev):
            x = jt.array(x0)
            self.assertFalse(x.is_stop_grad(), f"trainable {dev}")
            xs = x.stop_grad()
            self.assertTrue(xs.is_stop_grad(), f"stopped {dev}")
        both_devices(body)


# ------------------------------------------------------------- accumulation / nn backward

class TestAccumAndNN(Base):
    def test_grad_accumulation(self):
        # grads of two separate losses sum: d(2x)/dx + d(3x)/dx = 2 + 3 = 5.
        x0 = np.random.RandomState(17).randn(4).astype("float32")
        def body(dev):
            x = jt.array(x0)
            g1 = grad1((x * 2).sum(), x)
            g2 = grad1((x * 3).sum(), x)
            self.ac((g1 + g2).numpy(), np.full(4, 5.0, "float32"), atol=1e-6,
                    msg=f"grad accum {dev}")
        both_devices(body)

    def test_backward_accumulates_into_grad(self):
        # two backward() calls accumulate into the same leaf .grad (torch semantics).
        x0 = np.random.RandomState(170).randn(4).astype("float32")
        def body(dev):
            x = jt.array(x0); x.requires_grad = True
            (x * 2).sum().backward()
            (x * 3).sum().backward()
            self.ac(x.grad.numpy(), np.full(4, 5.0, "float32"), atol=1e-5,
                    msg=f"backward accum {dev}")
        both_devices(body)

    def test_relu_backward(self):
        # d/dx sum(relu(x)) = 1[x>0]
        x0 = np.random.RandomState(18).randn(6).astype("float32")
        def body(dev):
            x = jt.array(x0)
            self.ac(grad1(jt.nn.relu(x).sum(), x).numpy(),
                    (x0 > 0).astype("float32"), atol=1e-6, msg=f"relu bwd {dev}")
        both_devices(body)

    def test_softmax_backward_sums_to_zero(self):
        # softmax outputs sum to 1 (constant) -> d/dx sum(softmax(x)) = 0.
        x0 = np.random.RandomState(19).randn(5).astype("float32")
        def body(dev):
            x = jt.array(x0)
            g = grad1(jt.nn.softmax(x, dim=-1).sum(), x).numpy()
            self.ac(g, np.zeros(5, "float32"), atol=1e-5, msg=f"softmax sum bwd {dev}")
        both_devices(body)

    def test_softmax_backward_jacobian(self):
        # weighted softmax backward against the analytic Jacobian-vector product:
        # for s = softmax(x), grad of sum(w*s) w.r.t x is s*(w - sum(w*s)).
        rng = np.random.RandomState(190)
        x0 = rng.randn(5).astype("float32"); w0 = rng.randn(5).astype("float32")
        e = np.exp(x0 - x0.max()); s = e / e.sum()
        ref = s * (w0 - (w0 * s).sum())
        def body(dev):
            x = jt.array(x0)
            loss = (jt.nn.softmax(x, dim=-1) * jt.array(w0)).sum()
            self.ac(grad1(loss, x).numpy(), ref, atol=1e-5, msg=f"softmax jvp {dev}")
        both_devices(body)


# ----------------------------------------------------------------------- higher order

class TestHigherOrder(Base):
    def test_second_order_pow(self):
        # d2/dx2 sum(x^3) = 6x  (jittor's grad is itself differentiable).
        x0 = np.random.RandomState(20).randn(4).astype("float32")
        def body(dev):
            x = jt.array(x0)
            g1 = grad1((x * x * x).sum(), x)       # 3 x^2
            g2 = grad1(g1.sum(), x)                # 6 x
            self.ac(g2.numpy(), 6 * x0, atol=1e-4, msg=f"2nd order x^3 {dev}")
        both_devices(body)

    def test_second_order_exp(self):
        # d2/dx2 sum(exp(x)) = exp(x)
        x0 = np.random.RandomState(21).randn(3).astype("float32")
        def body(dev):
            x = jt.array(x0)
            g1 = grad1(jt.exp(x).sum(), x)
            g2 = grad1(g1.sum(), x)
            self.ac(g2.numpy(), np.exp(x0), atol=1e-4, rtol=1e-4,
                    msg=f"2nd order exp {dev}")
        both_devices(body)

    def test_second_order_via_autograd_create_graph(self):
        # torch idiom: autograd.grad(..., create_graph=True) then differentiate again.
        x0 = np.random.RandomState(22).randn(4).astype("float32")
        def body(dev):
            x = jt.array(x0)
            (g1,) = torch.autograd.grad((x * x * x).sum(), x, create_graph=True)
            (g2,) = torch.autograd.grad(g1.sum(), x)
            self.ac(g2.numpy(), 6 * x0, atol=1e-4,
                    msg=f"create_graph 2nd order {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
