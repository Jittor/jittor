"""Historical Jittor learning-rate scheduler contracts."""

# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers:
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import jittor as jt
from jittor.optim import Optimizer
import math


def _effective_lr(optimizer, param_group):
    """The lr this param group actually trains with.

    Mirrors ``Optimizer.step``, which reads ``pg.get("lr", self.lr)``: a group
    without its own ``"lr"`` key shares the single optimizer-wide value. Every
    scheduler in this module reads through here so that "the current lr" has
    one definition instead of one per class.
    """
    lr = param_group.get("lr")
    return float(optimizer.lr if lr is None else lr)


def _apply_lrs(optimizer, new_lrs):
    """Write each group's new lr back to wherever that group reads it from.

    This is the whole fix for the two-scheduler cross-contamination. The
    schedulers here used to write **both** stores on every update: first
    ``optimizer.lr *= gamma``, then ``pg["lr"] *= gamma`` for any group that
    happened to have its own key. Whether a group had that key depended on
    history -- constructing any new-style :class:`jittor.optim.LRScheduler`
    (``LambdaLR``) stamps ``"lr"`` into every param group and never removes it.
    So after a LambdaLR warmup the two stores decayed from different bases and
    drifted apart: ``optimizer.lr`` would report 0.25 while the optimizer was
    training with 0.125.

    Writing one store per group keeps them in agreement no matter what ran
    before. Groups that share ``optimizer.lr`` collapse to a single write, so a
    reduction is applied once rather than once per group.
    """
    groups = optimizer.param_groups
    shared = [lr for pg, lr in zip(groups, new_lrs) if pg.get("lr") is None]
    for pg, lr in zip(groups, new_lrs):
        if pg.get("lr") is not None:
            pg["lr"] = lr
    if shared:
        # One shared value has to satisfy every sharing group, so keep the
        # largest candidate (this is what min_lr floors need).
        optimizer.lr = max(shared)
    elif new_lrs and all(lr == new_lrs[0] for lr in new_lrs):
        # No group reads optimizer.lr any more, but it is still the documented
        # way to ask "what is the learning rate". Keep it truthful while there
        # is one unambiguous answer, instead of letting it go stale.
        optimizer.lr = new_lrs[0]


class ReduceLROnPlateau(object):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        assert factor < 1.0, "factor should be < 1.0."
        assert isinstance(optimizer, Optimizer), '{} is not an Optimizer'.format(type(optimizer).__name__)
        assert mode in {'min', 'max'}, 'mode ' + mode + ' is unknown!'
        assert threshold_mode in {'rel', 'abs'},  'threshold mode ' + threshold_mode + ' is unknown!'

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            assert len(min_lr) == len(optimizer.param_groups), "expected {} min_lrs, got {}".format(len(optimizer.param_groups), len(min_lr))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        self.factor = factor
        self.optimizer = optimizer
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.n_cd = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.loss_best = None
        self.n_bad = 0
        self.eps = eps
        self.last_epoch = 0
        self.loss_best = math.inf if mode=="min" else -math.inf
        
    def step(self, loss, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        loss_now = float(loss)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.better(loss_now, self.loss_best):
            self.loss_best = loss_now
            self.n_bad = 0
        else:
            self.n_bad += 1

        if self.n_cd > 0:
            self.n_cd -= 1
            self.n_bad = 0

        if self.n_bad > self.patience:
            self.update_lr(epoch)
            self.n_cd = self.cooldown
            self.n_bad = 0
            
    def update_lr(self, epoch):
        opt = self.optimizer
        new_lrs = []
        for i, param_group in enumerate(opt.param_groups):
            old_lr = _effective_lr(opt, param_group)
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr <= self.eps:
                new_lr = old_lr          # change too small to be worth making
            elif self.verbose:
                print('Epoch {:5d}: reducing learning rate of group {} from {:.4e} to {:.4e}.'.format(epoch, i, old_lr, new_lr))
            new_lrs.append(new_lr)
        _apply_lrs(opt, new_lrs)

    def better(self, a, b):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            save = 1.0 - self.threshold
            return a < b * save
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < b - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            save = self.threshold + 1.0
            return a > b * save
        else:
            return a > b + self.threshold

class CosineAnnealingLR(object):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        # One base lr per group, read through the single _effective_lr rule.
        # (Was: a separate self.base_lr for optimizer.lr plus a self.base_lr_pg
        # list, which is what let the two stores decay independently.)
        self.base_lrs = [_effective_lr(optimizer, pg)
                         for pg in optimizer.param_groups]
        #TODO set last_epoch is not ready

    def _next_lr(self, base_lr, now_lr):
        if self.last_epoch == 0:
            return base_lr
        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return (now_lr + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2)
        return  ((1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (now_lr - self.eta_min) + self.eta_min)

    def get_lr(self):
        """The lr each param group will train with after the next update.

        Same contract as :meth:`jittor.optim.LRScheduler.get_lr`: no arguments,
        one value per param group, and exactly what ``step()`` goes on to set.
        """
        opt = self.optimizer
        return [self._next_lr(base, _effective_lr(opt, pg))
                for base, pg in zip(self.base_lrs, opt.param_groups)]

    def step(self):
        self.last_epoch += 1
        self.update_lr()

    def update_lr(self):
        _apply_lrs(self.optimizer, self.get_lr())


class ExponentialLR(object):
    """ learning rate is multiplied by gamma in each step.
    """
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.optimizer = optimizer
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.base_lrs = [_effective_lr(optimizer, pg)
                         for pg in optimizer.param_groups]

    def get_lr(self):
        """The lr each param group will train with after the next update."""
        if self.last_epoch == 0:
            return list(self.base_lrs)
        return [base * self.gamma ** self.last_epoch for base in self.base_lrs]

    def step(self):
        self.last_epoch += 1
        self.update_lr()

    def update_lr(self):
        _apply_lrs(self.optimizer, self.get_lr())


class StepLR(object):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.cur_epoch = 0

    def get_gamma(self):
        if self.last_epoch < 0:
            if (self.cur_epoch != 0 and (self.cur_epoch + 1) % self.step_size == 0):
                return self.gamma
        else:
            if (self.cur_epoch + 1 + self.last_epoch) % self.step_size == 0:
                return self.gamma
        return 1.

    def get_lr(self):
        """The lr each param group will train with after the next update.

        Was ``return self.optimizer.lr`` -- dead code that ignored ``gamma``
        entirely and that ``update_lr`` never called, while the same method on
        MultiStepLR *did* apply gamma. Both now mean the same thing and both
        are what ``step()`` actually applies.
        """
        gamma = self.get_gamma()
        opt = self.optimizer
        return [_effective_lr(opt, pg) * gamma for pg in opt.param_groups]

    def step(self):
        self.update_lr()
        self.cur_epoch += 1

    def update_lr(self):
        _apply_lrs(self.optimizer, self.get_lr())


class MultiStepLR(object):
    def __init__(self, optimizer, milestones=[], gamma=0.1, last_epoch=-1):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        #TODO set last_epoch is not ready

    def get_gamma(self):
        if (self.last_epoch in self.milestones):
            return self.gamma
        return 1.0

    def get_lr(self):
        """The lr each param group will train with after the next update.

        ``get_lr`` applied gamma and ``update_lr`` applied it again from a
        separate read of ``optimizer.lr``; ``update_lr`` now goes through this
        method, so gamma is applied exactly once per step.
        """
        gamma = self.get_gamma()
        opt = self.optimizer
        return [_effective_lr(opt, pg) * gamma for pg in opt.param_groups]

    def step(self):
        self.last_epoch += 1
        self.update_lr()

    def update_lr(self):
        _apply_lrs(self.optimizer, self.get_lr())
