
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import random
from _helpers.torch_runtime import import_torch_modules, modules_available

skip_this_test = not modules_available("torch")
torch = None


def setUpModule():
    global torch
    if not skip_this_test:
        (torch,) = import_torch_modules("torch")
    
def check_equal(q,k,v,tatt,jatt):
    tq=torch.from_numpy(q)
    jq=jt.array(q)
    tk=torch.from_numpy(k)
    jk=jt.array(k)
    tv=torch.from_numpy(v)
    jv=jt.array(v)

    jatt.load_parameters(tatt.state_dict())
    ty, tw = tatt(tq, tk, tv)
    jy, jw = jatt(jq, jk, jv)
    assert np.allclose(ty.detach().numpy(), jy.numpy(), rtol=1e-3)
    assert np.allclose(tw.detach().numpy(), jw.numpy(), rtol=1e-3)

@unittest.skipIf(skip_this_test, "No Torch found")
class TestAttention(unittest.TestCase):
    def test_attention(self):
        j_opt = jt.optim.SGD([jt.array([1])], 1.0)
        t_opt = torch.optim.SGD([torch.ones([1])], 1.0)
        j_scheduler = jt.lr_scheduler.ReduceLROnPlateau(j_opt)
        t_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(t_opt)
        for i in range(100):
            loss=random.random()
            j_scheduler.step(loss)
            t_scheduler.step(loss)
            assert j_opt.lr == t_opt.state_dict()['param_groups'][0]['lr']

class TestReduceLROnPlateauGroups(unittest.TestCase):
    """A reduction must lower the shared lr exactly once, not once per group."""

    def _plateau_opt(self, n_groups):
        params = [{"params": [jt.array([1.0])]} for _ in range(n_groups)]
        opt = jt.optim.SGD(params, 1.0)
        # No group carries its own "lr", so every group falls back to the
        # optimizer-wide lr -- this is jittor's default param group layout.
        # (Drop the key explicitly: a torch-mode session installs an optimizer
        # that seeds one per group, which is a different code path.)
        for pg in opt.param_groups:
            pg.pop("lr", None)
        opt.lr = 1.0
        return opt, jt.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.1, patience=1, threshold=0.0)

    def test_shared_lr_drops_once_per_reduction(self):
        for n_groups in (1, 2, 3):
            opt, sched = self._plateau_opt(n_groups)
            sched.step(1.0)
            sched.step(1.0)
            sched.step(1.0)
            np.testing.assert_allclose(opt.lr, 0.1, rtol=1e-6)

    def test_min_lr_respected_for_shared_lr(self):
        opt, sched = self._plateau_opt(3)
        sched.min_lrs = [0.5] * 3
        sched.step(1.0)
        sched.step(1.0)
        sched.step(1.0)
        np.testing.assert_allclose(opt.lr, 0.5, rtol=1e-6)

    def test_per_group_lr_still_scales_per_group(self):
        params = [{"params": [jt.array([1.0])], "lr": 1.0},
                  {"params": [jt.array([1.0])], "lr": 2.0}]
        opt = jt.optim.SGD(params, 1.0)
        sched = jt.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.1, patience=1, threshold=0.0)
        sched.step(1.0)
        sched.step(1.0)
        sched.step(1.0)
        np.testing.assert_allclose(opt.param_groups[0]["lr"], 0.1, rtol=1e-6)
        np.testing.assert_allclose(opt.param_groups[1]["lr"], 0.2, rtol=1e-6)
        np.testing.assert_allclose(opt.lr, 1.0, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()


class TestLegacySchedulerSingleLRStore(unittest.TestCase):
    """The legacy schedulers must write one lr store per group, not two.

    ``jt.optim.LRScheduler`` (the new-style base, e.g. ``LambdaLR``) stamps an
    ``"lr"`` key into every param group at construction and never removes it.
    The legacy schedulers branch on whether that key exists, so merely having
    used a LambdaLR once flipped them from "update the shared optimizer.lr" to
    "update the shared optimizer.lr *and* every group's own lr", each from a
    different base. The two stores then drift and disagree about what the
    learning rate is -- while training silently follows the per-group one.
    """

    def _opt(self, n_groups=2, lr=1.0):
        opt = jt.optim.SGD(
            [{"params": [jt.array([1.0])]} for _ in range(n_groups)], lr)
        # jittor's default layout: no group carries its own "lr", so all of
        # them fall back to the single optimizer-wide value.
        for pg in opt.param_groups:
            pg.pop("lr", None)
        opt.lr = lr
        return opt

    def _effective(self, opt):
        # exactly what Optimizer.step reads: pg.get("lr", self.lr)
        return [float(pg.get("lr", opt.lr)) for pg in opt.param_groups]

    def test_optimizer_lr_agrees_with_the_lr_training_actually_uses(self):
        # The common recipe that triggers it: a LambdaLR warmup, then a
        # legacy scheduler for the decay.
        for name, make in (
                ("StepLR",
                 lambda o: jt.lr_scheduler.StepLR(o, step_size=1, gamma=0.5)),
                ("MultiStepLR",
                 lambda o: jt.lr_scheduler.MultiStepLR(
                     o, milestones=[1, 2], gamma=0.5)),
                ("ExponentialLR",
                 lambda o: jt.lr_scheduler.ExponentialLR(o, gamma=0.5)),
        ):
            with self.subTest(scheduler=name):
                opt = self._opt(2)
                jt.optim.LambdaLR(opt, lambda epoch: 0.5)   # warmup
                sched = make(opt)
                for _ in range(3):
                    sched.step()
                for lr in self._effective(opt):
                    self.assertAlmostEqual(
                        float(opt.lr), lr, places=9,
                        msg="%s: optimizer.lr reports %r but the optimizer "
                            "trains with %r" % (name, float(opt.lr), lr))

    def test_decay_is_applied_once_per_step_not_once_per_group(self):
        # group count must not change the trajectory
        for name, make, expect in (
                ("StepLR",
                 lambda o: jt.lr_scheduler.StepLR(o, step_size=1, gamma=0.5),
                 [1.0, 0.5, 0.25]),
                ("ExponentialLR",
                 lambda o: jt.lr_scheduler.ExponentialLR(o, gamma=0.5),
                 [1.0, 0.5, 0.25]),
        ):
            for n_groups in (1, 2, 3):
                with self.subTest(scheduler=name, groups=n_groups):
                    opt = self._opt(n_groups)
                    sched = make(opt)
                    got = []
                    for _ in range(3):
                        sched.step()
                        got.append(self._effective(opt)[0])
                    np.testing.assert_allclose(got, expect, rtol=1e-9)

    def test_get_lr_returns_what_update_lr_will_set(self):
        """One get_lr contract, matching jt.optim.LRScheduler.get_lr.

        StepLR.get_lr used to return optimizer.lr with gamma never applied
        (and update_lr never called it -- it was dead code), while
        MultiStepLR.get_lr applied gamma on top of an update_lr that applied
        it again from its own separate read of optimizer.lr.

        The invariant is that ``update_lr()`` applies exactly ``get_lr()``:
        gamma lands once, and get_lr is a pure query. (This pairs get_lr with
        update_lr rather than step(), because some of these schedulers bump
        last_epoch inside step() before computing -- as torch's do too.)
        """
        for name, make in (
                ("StepLR",
                 lambda o: jt.lr_scheduler.StepLR(o, step_size=1, gamma=0.5)),
                ("MultiStepLR",
                 lambda o: jt.lr_scheduler.MultiStepLR(
                     o, milestones=[0, 1, 2], gamma=0.5)),
                ("ExponentialLR",
                 lambda o: jt.lr_scheduler.ExponentialLR(o, gamma=0.5)),
                ("CosineAnnealingLR",
                 lambda o: jt.lr_scheduler.CosineAnnealingLR(o, T_max=4)),
        ):
            with self.subTest(scheduler=name):
                opt = self._opt(2)
                sched = make(opt)
                for _ in range(3):
                    predicted = sched.get_lr()
                    self.assertEqual(len(predicted), len(opt.param_groups))
                    # a pure query: asking twice must not move the lr
                    self.assertEqual(predicted, sched.get_lr())
                    sched.update_lr()
                    np.testing.assert_allclose(
                        self._effective(opt), predicted, rtol=1e-9,
                        err_msg="%s.update_lr() did not apply exactly "
                                "get_lr()" % name)
                    sched.last_epoch = getattr(sched, "last_epoch", 0) + 1
                    if hasattr(sched, "cur_epoch"):
                        sched.cur_epoch += 1
