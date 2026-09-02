
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
