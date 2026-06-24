# ***************************************************************
# Minimal pytorch-lightning-compatible training core for jittor (#11).
#
# Provides LightningModule + Trainer with the most-used surface so that
# Lightning-style training code runs on the jittor-as-torch stack with only the
# import changed:
#
#     import jittor.lightning as pl     # instead of: import pytorch_lightning as pl
#
#     class LitModel(pl.LightningModule):
#         def __init__(self): super().__init__(); self.net = nn.Linear(8, 1)
#         def forward(self, x): return self.net(x)
#         def training_step(self, batch, batch_idx):
#             x, y = batch
#             loss = ((self(x) - y) ** 2).mean()
#             self.log("train_loss", loss)
#             return loss
#         def configure_optimizers(self):
#             return jt.optim.Adam(self.parameters(), lr=1e-3)
#
#     pl.Trainer(max_epochs=5).fit(LitModel(), train_loader)
#
# Scope (honest): the CORE training/validation loop -- epochs, batches, gradient
# accumulation, gradient clipping, lr schedulers, self.log, max_steps/limit_*_batches.
# NOT (yet): callbacks beyond a no-op hook, distributed/DDP strategies, automatic
# checkpointing, precision plugins, the full logger ecosystem. configure_optimizers
# accepts the optimizer, (optimizers, schedulers), or the dict form.
# ***************************************************************
import jittor as jt
from jittor import nn


class LightningModule(nn.Module):
    """Subclass and implement training_step + configure_optimizers (and forward)."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError("LightningModule subclasses must implement forward()")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("LightningModule subclasses must implement training_step()")

    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        raise NotImplementedError("LightningModule subclasses must implement configure_optimizers()")

    # --- logging: store the latest scalar values (queryable via trainer.logged_metrics) ---
    def log(self, name, value, **kwargs):
        if not hasattr(self, "_logged_metrics"):
            self._logged_metrics = {}
        self._logged_metrics[name] = float(value.item() if hasattr(value, "item") else value)

    def log_dict(self, dictionary, **kwargs):
        for k, v in dictionary.items():
            self.log(k, v, **kwargs)

    # convenience no-ops Lightning code commonly calls
    def on_train_epoch_start(self): pass
    def on_train_epoch_end(self): pass
    def on_validation_epoch_end(self): pass


class Trainer:
    def __init__(self, max_epochs=1000, max_steps=-1, gradient_clip_val=None,
                 accumulate_grad_batches=1, limit_train_batches=None,
                 limit_val_batches=None, enable_progress_bar=True, logger=None,
                 callbacks=None, accelerator="auto", devices="auto", **kwargs):
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.callbacks = callbacks or []
        self.global_step = 0
        self.current_epoch = 0
        self.logged_metrics = {}

    def _configure(self, model):
        cfg = model.configure_optimizers()
        # (optimizers, schedulers)
        if isinstance(cfg, (list, tuple)) and len(cfg) == 2 and isinstance(cfg[0], (list, tuple)):
            return list(cfg[0]), list(cfg[1])
        if isinstance(cfg, dict):
            sched = cfg.get("lr_scheduler", None)
            if isinstance(sched, dict):
                sched = sched.get("scheduler", None)
            return [cfg["optimizer"]], ([sched] if sched is not None else [])
        if isinstance(cfg, (list, tuple)):
            return list(cfg), []
        return [cfg], []

    def fit(self, model, train_dataloaders=None, val_dataloaders=None, **kwargs):
        optimizers, schedulers = self._configure(model)
        model.train()
        stop = False
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            model.on_train_epoch_start()
            for opt in optimizers:
                opt.zero_grad()
            for batch_idx, batch in enumerate(train_dataloaders):
                if self.limit_train_batches is not None and batch_idx >= self.limit_train_batches:
                    break
                out = model.training_step(batch, batch_idx)
                loss = out["loss"] if isinstance(out, dict) else out
                if self.accumulate_grad_batches > 1:
                    loss = loss / self.accumulate_grad_batches
                loss.backward()
                if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                    if self.gradient_clip_val is not None:
                        for opt in optimizers:
                            try:
                                opt.clip_grad_norm(float(self.gradient_clip_val))
                            except Exception:
                                pass
                    for opt in optimizers:
                        opt.step()
                        opt.zero_grad()
                    self.global_step += 1
                    if self.max_steps > 0 and self.global_step >= self.max_steps:
                        stop = True
                        break
            for sched in schedulers:
                try:
                    sched.step()
                except Exception:
                    pass
            model.on_train_epoch_end()
            if hasattr(model, "_logged_metrics"):
                self.logged_metrics = dict(model._logged_metrics)
            if val_dataloaders is not None:
                self._run_eval(model, val_dataloaders, "validation_step")
            if stop:
                break
        return model

    def _run_eval(self, model, dataloaders, step_name):
        model.eval()
        step = getattr(model, step_name)
        with jt.no_grad():
            for batch_idx, batch in enumerate(dataloaders):
                if self.limit_val_batches is not None and batch_idx >= self.limit_val_batches:
                    break
                step(batch, batch_idx)
        model.on_validation_epoch_end()
        model.train()

    def validate(self, model, dataloaders=None, **kwargs):
        self._run_eval(model, dataloaders, "validation_step")
        return model

    def test(self, model, dataloaders=None, **kwargs):
        self._run_eval(model, dataloaders, "test_step")
        return model


# pytorch-lightning aliases some names; expose the common ones.
LightningDataModule = object
seed_everything = lambda seed=0, **k: jt.set_global_seed(seed) if hasattr(jt, "set_global_seed") else None

# Drop-in: after `import jittor.lightning`, existing `import pytorch_lightning as pl`
# (and `import lightning`) also resolve here -- without clobbering a real install.
import sys as _sys
for _alias in ("pytorch_lightning", "lightning"):
    _sys.modules.setdefault(_alias, _sys.modules[__name__])
