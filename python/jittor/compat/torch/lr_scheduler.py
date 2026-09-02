"""Torch-compatible learning-rate schedulers for Jittor optimizers."""

import jittor as jt

from .context import registry_for


def _install_lr_scheduler(g, registry=None):
    """Torch-compatible torch.optim.lr_scheduler over jittor optimizers, on the
    `import jittor as torch` path (the shim reuses this same namespace). jittor reads
    lr from pg.get("lr", self.lr), so every step must update BOTH optimizer.lr and each
    param_group["lr"]. Schedulers follow torch's convention: __init__ applies the
    epoch-0 lr, last_epoch advances on step(). Covers the schedulers transformers /
    LlamaFactory / torch users actually use (the warmup helpers wrap LambdaLR)."""
    _modules = registry_for(g, registry).module_map
    import types as _types, math as _math
    from jittor import optim as _optim
    if getattr(_optim, "Optimizer", None) is None:
        raise RuntimeError("jittor.optim has no Optimizer owner")
    if getattr(_optim, "_torch_lr_installed", False):
        _modules.setdefault("torch.optim", _optim)
        if hasattr(_optim, "lr_scheduler"):
            _modules.setdefault("torch.optim.lr_scheduler", _optim.lr_scheduler)
        if hasattr(_optim, "swa_utils"):
            _modules.setdefault("torch.optim.swa_utils", _optim.swa_utils)
        return

    def _base_lrs(opt):
        out = []
        for pg in getattr(opt, "param_groups", []) or []:
            out.append(pg.get("lr", getattr(opt, "lr", 0.0)))
        return out or [getattr(opt, "lr", 0.0)]

    def _set_lrs(opt, lrs):
        for pg, lr in zip(getattr(opt, "param_groups", []) or [], lrs):
            pg["lr"] = lr
        try: opt.lr = lrs[0]
        except Exception: pass

    class LRScheduler:
        def __init__(self, optimizer, last_epoch=-1, verbose=False):
            self.optimizer = optimizer
            if not hasattr(self, "base_lrs"):
                self.base_lrs = _base_lrs(optimizer)
            self.last_epoch = last_epoch
            self._step_count = 0
            self._last_lr = list(self.base_lrs)
            self.step()                       # torch: apply epoch-0 lr at construction
        def get_lr(self):
            return list(self.base_lrs)
        def get_last_lr(self):
            return list(self._last_lr)
        def state_dict(self):
            return {k: v for k, v in self.__dict__.items()
                    if k not in ("optimizer",) and not callable(v)}
        def load_state_dict(self, sd):
            self.__dict__.update(sd)
        def step(self, epoch=None):
            self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
            self._step_count += 1
            lrs = self.get_lr()
            self._last_lr = list(lrs)
            _set_lrs(self.optimizer, lrs)

    class LambdaLR(LRScheduler):
        def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
            self.base_lrs = _base_lrs(optimizer)
            n = len(self.base_lrs)
            self.lr_lambdas = list(lr_lambda) if isinstance(lr_lambda, (list, tuple)) else [lr_lambda]*n
            super().__init__(optimizer, last_epoch, verbose)
        def get_lr(self):
            e = max(self.last_epoch, 0)
            return [b * fn(e) for b, fn in zip(self.base_lrs, self.lr_lambdas)]

    class MultiplicativeLR(LRScheduler):
        def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
            self.base_lrs = _base_lrs(optimizer)
            n = len(self.base_lrs)
            self.lr_lambdas = list(lr_lambda) if isinstance(lr_lambda, (list, tuple)) else [lr_lambda]*n
            super().__init__(optimizer, last_epoch, verbose)
        def get_lr(self):
            if self.last_epoch <= 0:
                return list(self.base_lrs)
            return [lr * fn(self.last_epoch) for lr, fn in zip(self._last_lr, self.lr_lambdas)]

    class ConstantLR(LRScheduler):
        def __init__(self, optimizer, factor=1.0/3, total_iters=5, last_epoch=-1, verbose=False):
            self.factor = factor; self.total_iters = total_iters
            self.base_lrs = _base_lrs(optimizer)
            super().__init__(optimizer, last_epoch, verbose)
        def get_lr(self):
            f = self.factor if self.last_epoch < self.total_iters else 1.0
            return [b * f for b in self.base_lrs]

    class LinearLR(LRScheduler):
        def __init__(self, optimizer, start_factor=1.0/3, end_factor=1.0, total_iters=5,
                     last_epoch=-1, verbose=False):
            self.start_factor = start_factor; self.end_factor = end_factor
            self.total_iters = total_iters; self.base_lrs = _base_lrs(optimizer)
            super().__init__(optimizer, last_epoch, verbose)
        def get_lr(self):
            t = min(max(self.last_epoch, 0), self.total_iters)
            frac = t / self.total_iters if self.total_iters else 1.0
            f = self.start_factor + (self.end_factor - self.start_factor) * frac
            return [b * f for b in self.base_lrs]

    class StepLR(LRScheduler):
        def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
            self.step_size = step_size; self.gamma = gamma
            self.base_lrs = _base_lrs(optimizer)
            super().__init__(optimizer, last_epoch, verbose)
        def get_lr(self):
            return [b * self.gamma ** (max(self.last_epoch, 0) // self.step_size) for b in self.base_lrs]

    class MultiStepLR(LRScheduler):
        def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
            self.milestones = sorted(milestones); self.gamma = gamma
            self.base_lrs = _base_lrs(optimizer)
            super().__init__(optimizer, last_epoch, verbose)
        def get_lr(self):
            n = sum(1 for m in self.milestones if m <= self.last_epoch)
            return [b * self.gamma ** n for b in self.base_lrs]

    class ExponentialLR(LRScheduler):
        def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
            self.gamma = gamma; self.base_lrs = _base_lrs(optimizer)
            super().__init__(optimizer, last_epoch, verbose)
        def get_lr(self):
            return [b * self.gamma ** max(self.last_epoch, 0) for b in self.base_lrs]

    class CosineAnnealingLR(LRScheduler):
        def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1, verbose=False):
            self.T_max = T_max; self.eta_min = eta_min
            self.base_lrs = _base_lrs(optimizer)
            super().__init__(optimizer, last_epoch, verbose)
        def get_lr(self):
            e = max(self.last_epoch, 0)
            return [self.eta_min + (b - self.eta_min) * (1 + _math.cos(_math.pi * e / self.T_max)) / 2
                    for b in self.base_lrs]

    class PolynomialLR(LRScheduler):
        def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose=False):
            self.total_iters = total_iters; self.power = power
            self.base_lrs = _base_lrs(optimizer)
            super().__init__(optimizer, last_epoch, verbose)
        def get_lr(self):
            t = min(max(self.last_epoch, 0), self.total_iters)
            f = (1 - t / self.total_iters) ** self.power if self.total_iters else 1.0
            return [b * f for b in self.base_lrs]

    def _list_for_groups(value, n, name):
        if isinstance(value, (list, tuple)):
            if len(value) != n:
                raise ValueError(f"{name} length {len(value)} does not match optimizer param groups {n}")
            return [float(v) for v in value]
        return [float(value)] * n

    class OneCycleLR(LRScheduler):
        def __init__(self, optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None,
                     pct_start=0.3, anneal_strategy="cos", cycle_momentum=True,
                     base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
                     final_div_factor=1e4, three_phase=False, last_epoch=-1, verbose=False):
            if total_steps is None:
                if epochs is None or steps_per_epoch is None:
                    raise ValueError("OneCycleLR requires total_steps or both epochs and steps_per_epoch")
                total_steps = int(epochs) * int(steps_per_epoch)
            self.total_steps = int(total_steps)
            if self.total_steps <= 0:
                raise ValueError("OneCycleLR total_steps must be positive")
            if anneal_strategy not in ("cos", "linear"):
                raise ValueError("OneCycleLR anneal_strategy must be 'cos' or 'linear'")
            self.pct_start = float(pct_start)
            self.anneal_strategy = anneal_strategy
            self.cycle_momentum = bool(cycle_momentum)
            self.three_phase = bool(three_phase)
            self.div_factor = float(div_factor)
            self.final_div_factor = float(final_div_factor)
            n = len(getattr(optimizer, "param_groups", []) or [None])
            self.max_lrs = _list_for_groups(max_lr, n, "max_lr")
            self.base_lrs = [lr / self.div_factor for lr in self.max_lrs]
            self.min_lrs = [lr / self.final_div_factor for lr in self.base_lrs]
            self.base_momentums = _list_for_groups(base_momentum, n, "base_momentum")
            self.max_momentums = _list_for_groups(max_momentum, n, "max_momentum")
            for pg, lr, max_lr_v, min_lr in zip(getattr(optimizer, "param_groups", []) or [],
                                                self.base_lrs, self.max_lrs, self.min_lrs):
                pg["initial_lr"] = lr
                pg["max_lr"] = max_lr_v
                pg["min_lr"] = min_lr
            super().__init__(optimizer, last_epoch, verbose)

        def _anneal(self, start, end, pct):
            pct = min(max(float(pct), 0.0), 1.0)
            if self.anneal_strategy == "linear":
                return start + (end - start) * pct
            return end + (start - end) / 2.0 * (1.0 + _math.cos(_math.pi * pct))

        def _phases(self):
            first_end = self.pct_start * self.total_steps - 1
            last_end = self.total_steps - 1
            if self.three_phase:
                second_end = first_end * 2 + 1
                return [
                    (first_end, self.base_lrs, self.max_lrs, self.max_momentums, self.base_momentums),
                    (second_end, self.max_lrs, self.base_lrs, self.base_momentums, self.max_momentums),
                    (last_end, self.base_lrs, self.min_lrs, self.max_momentums, self.max_momentums),
                ]
            return [
                (first_end, self.base_lrs, self.max_lrs, self.max_momentums, self.base_momentums),
                (last_end, self.max_lrs, self.min_lrs, self.base_momentums, self.max_momentums),
            ]

        def get_lr(self):
            step_num = min(max(self.last_epoch, 0), max(self.total_steps - 1, 0))
            start_step = 0.0
            selected = self._phases()[-1]
            for phase in self._phases():
                if step_num <= phase[0]:
                    selected = phase
                    break
                start_step = phase[0]
            end_step, start_lrs, end_lrs, start_moms, end_moms = selected
            denom = end_step - start_step
            pct = 1.0 if denom <= 0 else (step_num - start_step) / denom
            lrs = [self._anneal(s, e, pct) for s, e in zip(start_lrs, end_lrs)]
            if self.cycle_momentum:
                moms = [self._anneal(s, e, pct) for s, e in zip(start_moms, end_moms)]
                for pg, mom in zip(getattr(self.optimizer, "param_groups", []) or [], moms):
                    if "betas" in pg:
                        b0, b1 = pg.get("betas", (mom, 0.999))
                        pg["betas"] = (mom, b1)
                    elif hasattr(self.optimizer, "betas"):
                        b0, b1 = getattr(self.optimizer, "betas", (mom, 0.999))
                        pg["betas"] = (mom, b1)
                    else:
                        pg["momentum"] = mom
            return lrs

    class SequentialLR(LRScheduler):
        def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False):
            self.optimizer = optimizer; self._scheds = list(schedulers)
            self._milestones = list(milestones); self.last_epoch = last_epoch + 1
            # torch resets each sub-scheduler's epoch-0; apply the first one's lr
            self._scheds[0].step(0)
        def step(self, epoch=None):
            self.last_epoch += 1
            idx = sum(1 for m in self._milestones if m <= self.last_epoch)
            sch = self._scheds[idx]
            if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
                sch.step(0)
            else:
                sch.step()
            self._last_lr = sch.get_last_lr()
        def get_last_lr(self):
            return list(getattr(self, "_last_lr", _base_lrs(self.optimizer)))

    class ChainedScheduler:
        def __init__(self, schedulers, optimizer=None):
            self._scheds = list(schedulers)
            self.optimizer = optimizer or self._scheds[0].optimizer
        def step(self):
            for s in self._scheds: s.step()
        def get_last_lr(self):
            return self._scheds[-1].get_last_lr()
        def state_dict(self): return {}
        def load_state_dict(self, sd): pass

    class ReduceLROnPlateau:
        def __init__(self, optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4,
                     min_lr=0.0, **k):
            self.optimizer = optimizer; self.mode = mode; self.factor = factor
            self.patience = patience; self.threshold = threshold; self.min_lr = min_lr
            self.best = None; self.num_bad = 0
        def step(self, metric=None, epoch=None):
            if metric is None: return
            m = float(metric)
            better = (self.best is None or
                      (m < self.best - self.threshold if self.mode == "min"
                       else m > self.best + self.threshold))
            if better:
                self.best = m; self.num_bad = 0
            else:
                self.num_bad += 1
                if self.num_bad > self.patience:
                    new = max(self.min_lr, getattr(self.optimizer, "lr", 0.0) * self.factor)
                    _set_lrs(self.optimizer, [new] * len(_base_lrs(self.optimizer)))
                    self.num_bad = 0
        def get_last_lr(self):
            return [getattr(self.optimizer, "lr", 0.0)]
        def state_dict(self):
            return {k: v for k, v in self.__dict__.items() if k != "optimizer"}
        def load_state_dict(self, sd): self.__dict__.update(sd)

    ns = _types.ModuleType("torch.optim.lr_scheduler")
    for _name, _cls in [("LRScheduler", LRScheduler), ("_LRScheduler", LRScheduler),
                        ("LambdaLR", LambdaLR), ("MultiplicativeLR", MultiplicativeLR),
                        ("ConstantLR", ConstantLR), ("LinearLR", LinearLR),
                        ("StepLR", StepLR), ("MultiStepLR", MultiStepLR),
                        ("ExponentialLR", ExponentialLR), ("CosineAnnealingLR", CosineAnnealingLR),
                        ("PolynomialLR", PolynomialLR), ("OneCycleLR", OneCycleLR),
                        ("SequentialLR", SequentialLR),
                        ("ChainedScheduler", ChainedScheduler), ("ReduceLROnPlateau", ReduceLROnPlateau)]:
        setattr(ns, _name, _cls)
    _optim.lr_scheduler = ns
    _optim._torch_lr_installed = True
    _modules.setdefault("torch.optim", _optim)
    _optim.__path__ = getattr(_optim, "__path__", [])
    _modules.setdefault("torch.optim.lr_scheduler", ns)
    _modules.setdefault("jittor.optim.lr_scheduler", ns)
    swa_utils = _types.ModuleType("torch.optim.swa_utils")
    class SWALR(LRScheduler):
        def __init__(self, optimizer, swa_lr, anneal_epochs=10, anneal_strategy="cos",
                     last_epoch=-1, verbose=False):
            self.swa_lr = swa_lr
            self.anneal_epochs = anneal_epochs
            self.anneal_strategy = anneal_strategy
            base = _base_lrs(optimizer)
            if isinstance(swa_lr, (list, tuple)):
                self.swa_lrs = list(swa_lr)
            else:
                self.swa_lrs = [swa_lr] * len(base)
            self.base_lrs = base
            super().__init__(optimizer, last_epoch, verbose)
        def get_lr(self):
            if self.anneal_epochs <= 0:
                return list(self.swa_lrs)
            t = min(max(self.last_epoch, 0), self.anneal_epochs) / self.anneal_epochs
            if self.anneal_strategy == "linear":
                alpha = t
            else:
                alpha = (1 - _math.cos(_math.pi * t)) / 2
            return [b + (s - b) * alpha for b, s in zip(self.base_lrs, self.swa_lrs)]
    swa_utils.SWALR = SWALR
    def get_swa_avg_fn():
        average = lambda averaged_param, current_param, num_averaged: (
            averaged_param + (current_param - averaged_param) / (num_averaged + 1)
        )
        return average
    def get_ema_avg_fn(decay=0.999):
        average = lambda averaged_param, current_param, num_averaged: (
            decay * averaged_param + (1.0 - decay) * current_param
        )
        return average
    class AveragedModel(jt.nn.Module):
        def __init__(self, model, device=None, avg_fn=None, multi_avg_fn=None, use_buffers=False):
            super().__init__()
            import copy as _copy
            self.module = _copy.deepcopy(model)
            self.n_averaged = jt.array(0).int64()
            self.avg_fn = avg_fn or get_swa_avg_fn()
            self.multi_avg_fn = multi_avg_fn
            self.use_buffers = use_buffers
            if device is not None and hasattr(self.module, "to"):
                try:
                    self.module.to(device)
                except Exception:
                    pass
        def execute(self, *args, **kwargs):
            return self.module(*args, **kwargs)
        def update_parameters(self, model):
            avg_params = list(self.module.parameters())
            model_params = list(model.parameters())
            try:
                n = int(self.n_averaged.item())
            except Exception:
                n = 0
            if n == 0:
                for ap, mp in zip(avg_params, model_params):
                    try: ap.update(mp)
                    except Exception: pass
            elif self.multi_avg_fn is not None:
                self.multi_avg_fn(avg_params, model_params, self.n_averaged)
            else:
                for ap, mp in zip(avg_params, model_params):
                    try:
                        ap.update(self.avg_fn(ap, mp, self.n_averaged))
                    except Exception:
                        pass
            try:
                self.n_averaged.update(self.n_averaged + 1)
            except Exception:
                self.n_averaged = jt.array(n + 1).int64()
    swa_utils.AveragedModel = AveragedModel
    swa_utils.get_swa_avg_fn = get_swa_avg_fn
    swa_utils.get_ema_avg_fn = get_ema_avg_fn
    def _update_bn(loader, model, device=None):
        """Recompute BatchNorm running statistics over `loader`.

        Was `lambda *a, **k: None`. SWA averages weights across checkpoints,
        which leaves every BatchNorm holding statistics for weights that no
        longer exist; update_bn exists precisely to fix that. Skipping it costs
        accuracy silently -- the model still runs and still reports a number.
        """
        bn_modules = [m for m in model.modules()
                      if hasattr(m, "running_mean") and hasattr(m, "running_var")]
        if not bn_modules:
            return
        saved_momentum = []
        for bn in bn_modules:
            saved_momentum.append(getattr(bn, "momentum", None))
            # reset to a cumulative moving average over the whole loader
            if isinstance(getattr(bn, "running_mean", None), jt.Var):
                bn.running_mean.assign(jt.zeros_like(bn.running_mean))
            if isinstance(getattr(bn, "running_var", None), jt.Var):
                bn.running_var.assign(jt.ones_like(bn.running_var))
        was_training = getattr(model, "is_training", lambda: True)()
        model.train()
        n = 0
        try:
            for batch in loader:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                n += 1
                for bn in bn_modules:
                    # cumulative average: momentum = 1/n
                    try:
                        bn.momentum = 1.0 / n
                    except Exception:
                        pass
                model(inputs)
        finally:
            for bn, mom in zip(bn_modules, saved_momentum):
                if mom is not None:
                    try:
                        bn.momentum = mom
                    except Exception:
                        pass
            if not was_training:
                model.eval()
    swa_utils.update_bn = _update_bn
    _optim.swa_utils = swa_utils
    _modules["torch.optim.swa_utils"] = swa_utils
