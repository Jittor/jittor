"""Torch-grade optimizer / lr-scheduler parity for ``import jittor as torch``.

One optimizer step on a known loss (loss = sum(w^2) -> grad = 2w) is checked against the
exact analytic update rule. CPU+CUDA.

Run:  python -m jittor.test.test_torch_compat_optim
"""
import unittest
import numpy as np
import jittor as torch
import jittor as jt

_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=atol, rtol=rtol,
                                   err_msg=msg)


class TestSGD(Base):
    def test_sgd_plain(self):
        w0 = np.array([1., 2., 3.], "float32")
        def body(dev):
            w = jt.array(w0)
            opt = torch.optim.SGD([w], lr=0.1)
            opt.step((w * w).sum())              # grad = 2w
            self.ac(w.numpy(), w0 - 0.1 * 2 * w0, msg=f"sgd {dev}")
        both_devices(body)

    def test_sgd_weight_decay(self):
        w0 = np.array([1., 2., 3.], "float32")
        def body(dev):
            w = jt.array(w0)
            opt = torch.optim.SGD([w], lr=0.1, weight_decay=0.5)
            opt.step((w * w).sum())              # effective grad = 2w + 0.5w = 2.5w
            self.ac(w.numpy(), w0 - 0.1 * 2.5 * w0, rtol=1e-4, msg=f"sgd wd {dev}")
        both_devices(body)

    def test_sgd_momentum_two_steps(self):
        w0 = np.array([1., 2.], "float32")
        lr, mu = 0.1, 0.9
        def body(dev):
            w = jt.array(w0)
            opt = torch.optim.SGD([w], lr=lr, momentum=mu)
            # analytic (torch): v1 = g0; v2 = mu*v1 + g1; w2 = w1 - lr*v2
            opt.step((w * w).sum())
            w1 = w.numpy().copy()
            g0 = 2 * w0
            self.ac(w1, w0 - lr * g0, rtol=1e-4, msg=f"sgd mom step1 {dev}")
            opt.step((w * w).sum())
            g1 = 2 * w1
            v2 = mu * g0 + g1
            self.ac(w.numpy(), w1 - lr * v2, rtol=1e-3, msg=f"sgd mom step2 {dev}")
        both_devices(body)


class TestAdam(Base):
    def test_adam_first_step(self):
        w0 = np.array([1., 2., 3.], "float32")
        lr, eps = 0.1, 1e-8
        def body(dev):
            w = jt.array(w0)
            opt = torch.optim.Adam([w], lr=lr, eps=eps)
            opt.step((w * w).sum())              # g = 2w > 0
            # first step: m_hat=g, v_hat=g^2 -> w -= lr * g/(sqrt(g^2)+eps) ~ lr*sign(g)
            g = 2 * w0
            ref = w0 - lr * g / (np.sqrt(g * g) + eps)
            self.ac(w.numpy(), ref, atol=1e-4, msg=f"adam {dev}")
        both_devices(body)

    def test_adam_tiny_gradient_eps_matches_torch_formula(self):
        w0 = np.array([0.0, 1.0], "float32")
        grad = np.array([1e-15, -2e-15], "float32")
        lr, eps = 1e-3, 1e-15
        b0, b1 = 0.9, 0.999

        def body(dev):
            w = jt.array(w0)
            opt = torch.optim.Adam([w], lr=lr, eps=eps, betas=(b0, b1))
            opt.step((w * jt.array(grad)).sum())
            m = (1 - b0) * grad
            v = (1 - b1) * grad * grad
            m_hat = m / (1 - b0)
            v_hat = v / (1 - b1)
            ref = w0 - lr * m_hat / (np.sqrt(v_hat) + eps)
            self.ac(w.numpy(), ref, atol=1e-7, rtol=1e-5,
                    msg=f"adam tiny-grad eps {dev}")
        both_devices(body)

    def test_adam_step_inside_no_grad_keeps_param_trainable(self):
        def body(dev):
            w = jt.array(np.array([1.0, 2.0], "float32"))
            opt = torch.optim.Adam([w], lr=0.1)
            (w * w).sum().backward()
            with torch.no_grad():
                opt.step()
                opt.zero_grad()
            self.assertFalse(w.is_stop_grad(), f"param remains trainable {dev}")
            (w * w).sum().backward()
            with torch.no_grad():
                opt.step()
                opt.zero_grad()
            self.assertEqual(opt.n_step, 2, f"step counter advances {dev}")
        both_devices(body)

    def test_adam_state_dict_is_torch_shaped(self):
        def body(dev):
            w = jt.array(np.array([1.0, 2.0], "float32"))
            opt = torch.optim.Adam([{"params": [w], "lr": 0.1, "name": "w"}], lr=0.0)
            opt.step((w * w).sum())
            sd = opt.state_dict()
            self.assertIn("state", sd, f"state key {dev}")
            self.assertIn("param_groups", sd, f"param_groups key {dev}")
            self.assertEqual(sd["param_groups"][0]["params"], [0], f"param id {dev}")
            self.assertEqual(sd["param_groups"][0]["name"], "w", f"group name {dev}")
            self.assertIn("exp_avg", sd["state"][0], f"exp_avg {dev}")
            self.assertIn("exp_avg_sq", sd["state"][0], f"exp_avg_sq {dev}")
            self.assertIn("step", sd["state"][0], f"step {dev}")
        both_devices(body)


class TestScheduler(Base):
    def test_steplr(self):
        def body(dev):
            w = jt.array(np.array([1.0], "float32"))
            opt = torch.optim.SGD([w], lr=1.0)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)
            lrs = []
            for _ in range(5):
                lrs.append(float(opt.lr))
                sched.step()
            # lr = 1.0 for epochs 0,1; 0.1 for 2,3; 0.01 for 4
            self.ac(lrs, [1.0, 1.0, 0.1, 0.1, 0.01], rtol=1e-5, msg=f"steplr {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
