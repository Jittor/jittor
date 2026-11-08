"""``torch.optim.SGD`` momentum-buffer state follows the buffer, not the knob.

torch keeps ``state[p]["momentum_buffer"]`` from the buffer's first momentum
update onward, whatever the group's *current* momentum is; a parameter whose
momentum never ran has no state at all. The compatibility layer used to derive
both ``optimizer.state`` and ``state_dict()`` from the current momentum, so a
checkpoint saved while momentum was paused lost the buffer.

Reference trajectories: real PyTorch 2.5.1 (CPU, float32), recorded with the
same torch-API script in an isolated interpreter. CPU+CUDA.

Run:  python -m pytest compat/tests/torch/test_sgd_momentum_state.py
"""

from _helpers import capability as _test_capability
import unittest
import numpy as np
import torch
import jittor as jt

_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if _test_capability.any_accelerator_enabled(backend=jt) else [])


def _buffer(opt, p):
    buf = opt.state.get(p, {}).get("momentum_buffer")
    return None if buf is None else buf.detach().cpu().numpy().tolist()


def _step(opt, p):
    opt.zero_grad()
    (p * p).sum().backward()
    opt.step()


class TestSGDMomentumState(unittest.TestCase):
    def ac(self, got, ref, msg):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref),
                                   rtol=1e-5, atol=1e-6, err_msg=msg)

    def test_buffer_stays_visible_while_momentum_is_paused(self):
        expect = [([0.62, -1.24], [2.0, -4.0]),
                  ([0.2224, -0.4448], [3.04, -6.08]),
                  ([0.17792, -0.35584], [3.04, -6.08]),
                  ([-0.1359296, 0.2718592], [3.09184, -6.18368])]
        for dev, use_cuda in _DEVICES:
            with jt.flag_scope(use_cuda=use_cuda):
                p = torch.nn.Parameter(torch.tensor([1.0, -2.0], device=dev))
                opt = torch.optim.SGD([p], lr=0.1, momentum=0.9, nesterov=True)
                for k, mom in enumerate((0.9, 0.9, 0.0, 0.9)):
                    opt.param_groups[0]["momentum"] = mom
                    opt.param_groups[0]["nesterov"] = mom > 0
                    _step(opt, p)
                    self.ac(p.detach().cpu().numpy(), expect[k][0], f"param {k} [{dev}]")
                    buf = _buffer(opt, p)
                    self.assertIsNotNone(buf, f"buffer hidden at step {k} [{dev}]")
                    self.ac(buf, expect[k][1], f"buffer {k} [{dev}]")

    def test_torch_format_checkpoint_saved_while_paused_keeps_buffer(self):
        expect = [([0.020656, -0.041312], [3.60144, -7.20288]),
                  ([-0.30719168, 0.61438336], [3.2784768, -6.5569536])]
        for dev, use_cuda in _DEVICES:
            with jt.flag_scope(use_cuda=use_cuda):
                p = torch.nn.Parameter(torch.tensor([1.0, -2.0], device=dev))
                opt = torch.optim.SGD([p], lr=0.1, momentum=0.9, dampening=0.1)
                _step(opt, p)
                _step(opt, p)
                opt.param_groups[0]["momentum"] = 0.0
                _step(opt, p)
                saved = opt.state_dict()
                self.assertIn("momentum_buffer", next(iter(saved["state"].values())),
                              f"buffer dropped from the checkpoint [{dev}]")
                self.assertNotIn("momentum_initialized", saved["param_groups"][0])
                p2 = torch.nn.Parameter(p.detach().clone())
                opt2 = torch.optim.SGD([p2], lr=0.1, momentum=0.0, dampening=0.1)
                opt2.load_state_dict(saved)
                opt2.param_groups[0]["momentum"] = 0.9
                for k in range(2):
                    _step(opt2, p2)
                    self.ac(p2.detach().cpu().numpy(), expect[k][0], f"param {k} [{dev}]")
                    self.ac(_buffer(opt2, p2), expect[k][1], f"buffer {k} [{dev}]")

    def test_no_state_until_momentum_first_runs(self):
        for dev, use_cuda in _DEVICES:
            with jt.flag_scope(use_cuda=use_cuda):
                p = torch.nn.Parameter(torch.tensor([1.0], device=dev))
                opt = torch.optim.SGD([p], lr=0.1, momentum=0.0, dampening=0.2)
                _step(opt, p)
                self.assertNotIn(p, opt.state, f"state without a buffer [{dev}]")
                self.assertEqual(opt.state_dict()["state"], {}, dev)
                opt.param_groups[0]["momentum"] = 0.8
                _step(opt, p)                              # first touch: seeded
                self.ac(_buffer(opt, p), [1.6], f"seeded buffer [{dev}]")

    def test_clearing_the_buffer_reseeds_on_the_next_update(self):
        for dev, use_cuda in _DEVICES:
            with jt.flag_scope(use_cuda=use_cuda):
                p = torch.nn.Parameter(torch.tensor([1.0], device=dev))
                opt = torch.optim.SGD([p], lr=0.1, momentum=0.8, dampening=0.5)
                _step(opt, p)
                opt.state[p] = {"momentum_buffer": None}
                self.assertNotIn(p, opt.state, dev)
                _step(opt, p)                              # p = 0.8, grad 1.6
                self.ac(_buffer(opt, p), [1.6], f"reseeded buffer [{dev}]")

    def test_a_restored_optimizer_does_not_share_state_with_its_source(self):
        # torch deep-copies a loaded state_dict. Adopting the source's Vars
        # shared one momentum buffer between both optimizers, and the CUDA
        # fused kernel updates it in place, so each stepped the other's.
        def run(steps, p0, buf=None):
            p = np.array(p0)
            for _ in range(steps):
                g = 2 * (p - 0.2)
                buf = g if buf is None else 0.8 * buf + g
                p = p - 0.06 * buf
            return p, buf

        def loss_step(opt, p):
            opt.zero_grad()
            ((p - 0.2) ** 2).sum().backward()
            opt.step()

        for dev, use_cuda in _DEVICES:
            with jt.flag_scope(use_cuda=use_cuda):
                a = torch.nn.Parameter(torch.tensor([1.0, -0.5], device=dev))
                opt_a = torch.optim.SGD([a], lr=0.06, momentum=0.8)
                for _ in range(2):
                    loss_step(opt_a, a)
                b = torch.nn.Parameter(torch.tensor([3.0, 2.0], device=dev))
                opt_b = torch.optim.SGD([b], lr=0.06, momentum=0.8)
                opt_b.load_state_dict(opt_a.state_dict())
                for _ in range(2):
                    loss_step(opt_a, a)
                    loss_step(opt_b, b)
                _, buf_a = run(2, [1.0, -0.5])
                self.ac(a.detach().cpu().numpy(), run(4, [1.0, -0.5])[0],
                        f"source optimizer [{dev}]")
                self.ac(b.detach().cpu().numpy(), run(2, [3.0, 2.0], buf_a)[0],
                        f"restored optimizer [{dev}]")


if __name__ == "__main__":
    unittest.main()
