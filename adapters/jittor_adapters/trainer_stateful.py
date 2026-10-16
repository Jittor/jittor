"""Explicit, single-process Trainer checkpointing of a prepared stateful loader.

Nothing is patched on import. This integration deliberately supports a bounded
first epoch; unsupported epoch/tail/distributed schedules fail before training.
"""
import json
import os
from pathlib import Path

from ._common import require_version

SUPPORTED_VERSIONS = {
    "transformers": frozenset(("4.56.2",)),
    "accelerate": frozenset(("1.10.1",)),
}
STATE_FILE = "stateful_dataloader.pt"
STATE_VERSION = 1


def _generators(loader):
    result = {}
    generator = getattr(loader, "generator", None)
    if generator is not None:
        result["loader"] = generator
    sampler = getattr(getattr(loader, "batch_sampler", None), "sampler", None)
    if sampler is None:
        sampler = getattr(loader, "sampler", None)
    for _ in range(4):
        if sampler is None:
            break
        generator = getattr(sampler, "generator", None)
        if generator is not None:
            result["sampler"] = generator
            break
        sampler = getattr(sampler, "sampler", None)
    return result


def stateful_trainer_class(base_class=None):
    """Return an opt-in Trainer subclass, optionally extending a custom Trainer.

    Instantiate with ``stateful_dataset_id`` identifying the immutable dataset
    and transform configuration. Checkpoints contain Python/NumPy RNG objects;
    load only trusted checkpoints, just as for Trainer's own RNG checkpoint.
    """
    import accelerate
    import transformers
    import torch
    from transformers import Trainer, TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint

    versions = {name: require_version(name, supported)
                for name, supported in SUPPORTED_VERSIONS.items()}
    if base_class is None:
        base_class = Trainer
    if not issubclass(base_class, Trainer):
        raise TypeError("base_class must be a transformers.Trainer subclass")

    class LoaderCheckpoint(TrainerCallback):
        def __init__(self, owner):
            self.owner = owner

        def on_train_begin(self, args, state, control, **kwargs):
            owner = self.owner
            loader = kwargs["train_dataloader"]
            owner.stateful_loader = loader
            configuration = owner.stateful_configuration(loader)
            pending = owner.stateful_pending_checkpoint
            if pending is not None:
                if pending["configuration"] != configuration:
                    raise ValueError("stateful dataloader checkpoint configuration differs")
                if pending["global_step"] != state.global_step:
                    raise ValueError("stateful dataloader checkpoint global_step differs")
                expected_epoch = (state.global_step * args.gradient_accumulation_steps / len(loader))
                if pending["epoch"] != expected_epoch:
                    raise ValueError("stateful Trainer checkpoint epoch does not match consumed batches")
                generators = _generators(loader)
                if set(generators) != set(pending["generator_states"]):
                    raise ValueError("stateful dataloader checkpoint generators differ")
                for name, generator in generators.items():
                    generator.set_state(pending["generator_states"][name])
                loader.load_state_dict(pending["dataloader_state"])
                owner.stateful_restored_step = state.global_step
                owner.stateful_pending_checkpoint = None
            return control

        def on_step_end(self, args, state, control, **kwargs):
            # ignore_data_skip prevents a second cursor advance, but Trainer then
            # computes epoch using only newly consumed batches. Correct its public
            # state before other callbacks and the checkpoint writer observe it.
            state.epoch = (state.global_step * args.gradient_accumulation_steps
                           / len(self.owner.stateful_loader))
            return control

        def on_save(self, args, state, control, **kwargs):
            owner = self.owner
            destination = Path(args.output_dir) / ("checkpoint-%d" % state.global_step) / STATE_FILE
            # The prepared loader accounts for Accelerate's one-batch lookahead.
            # Its base loader's state would include an unconsumed batch.
            payload = {
                "version": STATE_VERSION,
                "versions": versions,
                "global_step": state.global_step,
                "epoch": state.epoch,
                "configuration": owner.stateful_configuration(owner.stateful_loader),
                "dataloader_state": owner.stateful_loader.state_dict(),
                "generator_states": {name: generator.get_state()
                                     for name, generator in _generators(owner.stateful_loader).items()},
            }
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(payload, str(temporary))
            os.replace(str(temporary), str(destination))
            owner.stateful_saved_checkpoints.append(str(destination))
            return control

    class StatefulTrainer(base_class):
        def __init__(self, *args, stateful_dataset_id, stateful_dataloader_factory=None, **kwargs):
            if not isinstance(stateful_dataset_id, str) or not stateful_dataset_id.strip():
                raise ValueError("stateful_dataset_id must identify the fixed dataset and transforms")
            self.stateful_dataset_id = stateful_dataset_id
            if stateful_dataloader_factory is not None and not callable(stateful_dataloader_factory):
                raise TypeError("stateful_dataloader_factory must be callable")
            self.stateful_dataloader_factory = stateful_dataloader_factory
            self.stateful_pending_checkpoint = None
            self.stateful_restored_step = None
            self.stateful_saved_checkpoints = []
            self.stateful_loader = None
            callbacks = list(kwargs.pop("callbacks", None) or [])
            kwargs["callbacks"] = [LoaderCheckpoint(self)] + callbacks
            super().__init__(*args, **kwargs)
            self.accelerator.dataloader_config.use_stateful_dataloader = True
            self.args.ignore_data_skip = True

        def stateful_configuration(self, loader):
            args = self.args
            if self.accelerator.num_processes != 1 or args.world_size != 1 or args.n_gpu > 1:
                raise ValueError("stateful Trainer currently requires a single process and device")
            if args.auto_find_batch_size or args.save_only_model:
                raise ValueError("stateful Trainer requires fixed batches and full checkpoints")
            if args.max_steps <= 0:
                raise ValueError("stateful Trainer requires an explicit positive max_steps")
            batches = len(loader)
            accumulation = args.gradient_accumulation_steps
            if batches == 0 or batches % accumulation:
                raise ValueError("stateful Trainer requires complete accumulation windows (no epoch tail)")
            if args.max_steps * accumulation > batches:
                raise ValueError("stateful Trainer currently supports training within the first epoch only")
            if not getattr(loader, "use_stateful_dataloader", False):
                raise ValueError("Trainer did not prepare a stateful dataloader")
            if not all(callable(getattr(loader, method, None))
                       for method in ("state_dict", "load_state_dict")):
                raise TypeError("prepared dataloader must implement state_dict/load_state_dict")
            return {
                "dataset_id": self.stateful_dataset_id,
                "dataset_size": len(self.train_dataset),
                "dataset_type": type(self.train_dataset).__qualname__,
                "batches_per_epoch": batches,
                "batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": accumulation,
                "num_workers": args.dataloader_num_workers,
                "prefetch_factor": args.dataloader_prefetch_factor,
                "persistent_workers": args.dataloader_persistent_workers,
                "drop_last": args.dataloader_drop_last,
                "seed": args.seed,
                "data_seed": args.data_seed,
                "max_steps": args.max_steps,
            }

        def get_train_dataloader(self):
            if self.stateful_dataloader_factory is None:
                loader = super().get_train_dataloader()
            else:
                # Public opt-in for caller-owned sampler/dataset state contracts.
                # The factory returns an unprepared torchdata loader.
                loader = self.accelerator.prepare(self.stateful_dataloader_factory(self))
            self.stateful_configuration(loader)
            return loader

        def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
            if trial is not None:
                raise ValueError("stateful Trainer does not support hyperparameter-search trials")
            if resume_from_checkpoint is True:
                resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
                if resume_from_checkpoint is None:
                    raise ValueError("no checkpoint found for stateful Trainer resume")
            self.stateful_pending_checkpoint = None
            if resume_from_checkpoint:
                path = Path(resume_from_checkpoint)
                state_path = path / STATE_FILE
                if not state_path.is_file():
                    raise FileNotFoundError("stateful Trainer checkpoint is missing %s" % state_path)
                payload = torch.load(str(state_path), map_location="cpu", weights_only=False)
                required = {"version", "versions", "global_step", "epoch", "configuration",
                            "dataloader_state", "generator_states"}
                if not isinstance(payload, dict) or set(payload) != required:
                    raise ValueError("invalid stateful Trainer checkpoint schema")
                if payload["version"] != STATE_VERSION or payload["versions"] != versions:
                    raise ValueError("unsupported stateful Trainer checkpoint version")
                if type(payload["global_step"]) is not int or payload["global_step"] <= 0:
                    raise ValueError("stateful Trainer checkpoint global_step must be a positive integer")
                if not isinstance(payload["generator_states"], dict):
                    raise ValueError("invalid stateful Trainer checkpoint generator states")
                trainer_state = json.loads((path / "trainer_state.json").read_text())
                if (payload["global_step"] != trainer_state["global_step"]
                        or payload["epoch"] != trainer_state["epoch"]):
                    raise ValueError("stateful dataloader and Trainer checkpoints disagree")
                if not 0 < payload["epoch"] < 1:
                    raise ValueError("stateful Trainer resume requires an unfinished first epoch")
                self.stateful_pending_checkpoint = payload
            self.args.ignore_data_skip = True
            return super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial,
                                 ignore_keys_for_eval=ignore_keys_for_eval, **kwargs)

    return StatefulTrainer
