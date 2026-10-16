"""Exercise Trainer's own checkpoint resume and data cursor in fresh processes."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import random
from pathlib import Path
import time

import numpy as np

from _hf_trainer_e2e_runner import (
    _array,
    _attach_peft,
    _load_initial,
    _make_base_model,
    _named_tensors,
    _runtime,
    _save_initial,
    _tokenizer_and_data,
    VOCAB,
)


class _RepeatedDataset:
    """Expose a larger deterministic fixture while preserving sample IDs."""

    def __init__(self, dataset, size):
        if size < 1:
            raise ValueError("dataset size must be positive")
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        item = dict(self.dataset[index % len(self.dataset)])
        item["sample_id"] = index
        return item


class _PythonNumpyRandomTransform:
    """Apply and expose worker-local Python and NumPy random draws."""

    def __init__(self, dataset, vocab_size, get_worker_info):
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.get_worker_info = get_worker_info

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        python_draw = random.random()
        numpy_draw = float(np.random.random_sample())
        input_ids = list(item["input_ids"])
        if len(input_ids) > 2:
            available_positions = len(input_ids) - 2
            position = 1 + min(int(python_draw * available_positions), available_positions - 1)
            ordinary_token_count = self.vocab_size - 5
            replacement = 5 + min(int(numpy_draw * ordinary_token_count), ordinary_token_count - 1)
            input_ids[position] = replacement
        item["input_ids"] = input_ids
        worker = self.get_worker_info()
        item["worker_id"] = -1 if worker is None else int(worker.id)
        item["transform_python_draw"] = python_draw
        item["transform_numpy_draw"] = numpy_draw
        return item


class _CheckpointableRandomTransform(_PythonNumpyRandomTransform):
    """Native torchdata requires the dataset to expose its transform RNG state.

    The shim captures these globals per logical thread worker itself, whereas
    native process workers expose them through torchdata's dataset contract.
    Both paths execute the same transform and record its actual random draws.
    """

    def state_dict(self):
        return {"python": random.getstate(), "numpy": np.random.get_state()}

    def load_state_dict(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])


def _generator_state_sha256(generator):
    if generator is None:
        return None
    return hashlib.sha256(_array(generator.get_state()).tobytes()).hexdigest()


def _flatten_state(prefix, value, output):
    if isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            _flatten_state("%s/%s" % (prefix, key), value[key], output)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten_state("%s/%04d" % (prefix, index), item, output)
    elif value is None:
        output[prefix] = np.asarray("<none>")
    elif isinstance(value, (str, bytes, bool, int, float, np.generic)):
        output[prefix] = np.asarray(value)
    elif hasattr(value, "detach"):
        output[prefix] = _array(value)
    elif callable(value):
        output[prefix] = np.asarray("<callable>")
    else:
        raise TypeError("unsupported state value at %s: %s" % (prefix, type(value).__name__))


def _record_step_state(model, optimizer, scheduler, step, arrays):
    for name, parameter in model.named_parameters():
        if bool(parameter.requires_grad):
            arrays["step/%04d/parameter/%s" % (step, name)] = _array(parameter)

    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer_state = optimizer.state_dict()
    for live_group, saved_group in zip(optimizer.param_groups, optimizer_state["param_groups"]):
        live_parameters = live_group.get("params", [])
        saved_parameter_ids = saved_group.get("params", [])
        if len(live_parameters) != len(saved_parameter_ids):
            raise AssertionError("optimizer state dict changed parameter group membership")
        for parameter, parameter_id in zip(live_parameters, saved_parameter_ids):
            name = parameter_names.get(id(parameter))
            if name is None:
                raise AssertionError("optimizer state belongs to an unnamed model parameter")
            state = optimizer_state["state"].get(parameter_id, {})
            _flatten_state("step/%04d/optimizer/%s" % (step, name), state, arrays)

    group_fields = ("lr", "weight_decay", "betas", "eps")
    for index, group in enumerate(optimizer.param_groups):
        group_state = {key: group[key] for key in group_fields if key in group}
        _flatten_state("step/%04d/optimizer_group/%04d" % (step, index), group_state, arrays)

    scheduler_state = scheduler.state_dict()
    portable_scheduler_fields = (
        "base_lrs", "last_epoch", "_step_count", "_last_lr", "lr_lambdas"
    )
    missing = set(portable_scheduler_fields) - set(scheduler_state)
    if missing:
        raise AssertionError("scheduler state is missing fields: %s" % sorted(missing))
    _flatten_state(
        "step/%04d/scheduler" % step,
        {key: scheduler_state[key] for key in portable_scheduler_fields},
        arrays,
    )


def _initialize(torch, path, seed):
    torch.manual_seed(seed)
    base = _make_base_model(torch)
    model = _attach_peft(torch, base)
    _save_initial(model, path)
    print(json.dumps({"mode": "initialize", "weights": str(path)}))


def _run(args):
    jt, torch = _runtime(args.runtime)
    from transformers import Trainer, TrainerCallback, TrainingArguments

    scope = (
        jt.runtime.scope(use_cuda=int(args.device == "cuda"), backend_fallback="error")
        if jt is not None else nullcontext()
    )
    with scope:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if args.runtime == "jittor":
            from jittor._runtime.fallback import forbid_backend_fallbacks

            fallback_guard = forbid_backend_fallbacks()
            fallback_guard.__enter__()
            fallback_before = jt.core.backend_fallback_count()
        else:
            fallback_guard = None
            fallback_before = None

        try:
            torch.manual_seed(args.seed)
            model = _attach_peft(torch, _make_base_model(torch, dropout=args.dropout))
            _load_initial(torch, model, args.weights)
            tokenizer, train_dataset, _, collator = _tokenizer_and_data(torch)
            if args.dataset_size is not None:
                train_dataset = _RepeatedDataset(train_dataset, args.dataset_size)
            if args.worker_random_transform:
                transform_class = (
                    _CheckpointableRandomTransform
                    if args.stateful_dataloader and args.runtime == "pytorch"
                    else _PythonNumpyRandomTransform
                )
                train_dataset = transform_class(
                    train_dataset,
                    vocab_size=len(VOCAB),
                    get_worker_info=torch.utils.data.get_worker_info,
                )
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            train_args = TrainingArguments(
                output_dir=str(output_dir / "checkpoints"),
                use_cpu=args.device == "cpu",
                seed=args.seed,
                data_seed=args.seed,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                max_steps=args.max_steps,
                learning_rate=3e-3,
                weight_decay=0.01,
                lr_scheduler_type="linear",
                optim="adamw_torch",
                eval_strategy="no",
                save_strategy="steps",
                save_steps=1,
                save_total_limit=args.max_steps + 1,
                logging_strategy="no",
                disable_tqdm=True,
                report_to=[],
                dataloader_num_workers=args.dataloader_num_workers,
                dataloader_prefetch_factor=(
                    args.dataloader_prefetch_factor
                    if args.dataloader_num_workers > 0 else None
                ),
                dataloader_persistent_workers=args.persistent_workers,
                remove_unused_columns=False,
                save_safetensors=True,
            )

            class StopAfterStep(TrainerCallback):
                def on_step_end(self, callback_args, state, control, **kwargs):
                    if args.stop_after is not None and state.global_step >= args.stop_after:
                        control.should_training_stop = True
                    return control

            trainer_base = Trainer
            if args.stateful_dataloader:
                from jittor_adapters.trainer_stateful import stateful_trainer_class

                trainer_base = stateful_trainer_class()

            class RecordingTrainer(trainer_base):
                def __init__(self, *positional, **keywords):
                    self.sample_order = []
                    self.gradients = []
                    self.trace_arrays = {}
                    self.sampler_generator = None
                    self.dataloader_generator = None
                    super().__init__(*positional, **keywords)

                def _get_train_sampler(self, train_dataset=None):
                    dataset = train_dataset if train_dataset is not None else self.train_dataset
                    if args.sampler == "random":
                        generator = None
                        if args.explicit_generators:
                            generator = torch.Generator()
                            generator.manual_seed(args.sampler_seed)
                        self.sampler_generator = generator
                        sampler_class = torch.utils.data.RandomSampler
                        if args.stateful_dataloader:
                            from torchdata.stateful_dataloader.sampler import RandomSampler

                            sampler_class = RandomSampler
                        return sampler_class(dataset, generator=generator)
                    return torch.utils.data.SequentialSampler(dataset)

                def _get_dataloader(self, *positional, **keywords):
                    dataloader = super()._get_dataloader(*positional, **keywords)
                    if args.explicit_generators and keywords.get("is_training", False):
                        generator = torch.Generator()
                        generator.manual_seed(args.dataloader_seed)
                        # Accelerate proxies reads, but assignment on the wrapper
                        # does not reach the loader that actually seeds workers.
                        dataloader.base_dataloader.generator = generator
                        self.dataloader_generator = generator
                    return dataloader

                def training_step(self, current_model, inputs, num_items_in_batch=None):
                    sample_ids = inputs.pop("sample_id")
                    sample_ids_array = _array(sample_ids)
                    self.sample_order.extend(
                        int(value) for value in sample_ids_array.reshape(-1)
                    )
                    if inputs["input_ids"].device.type != args.device:
                        raise AssertionError("Trainer batch landed on the wrong device")
                    upcoming_step = int(self.state.global_step) + 1
                    microbatch = len(self.gradients) % train_args.gradient_accumulation_steps
                    if args.worker_random_transform:
                        prefix = "step/%04d/batch/%02d/" % (upcoming_step, microbatch)
                        self.trace_arrays[prefix + "sample_id"] = sample_ids_array
                        for field in (
                            "worker_id",
                            "transform_python_draw",
                            "transform_numpy_draw",
                        ):
                            self.trace_arrays[prefix + field] = _array(inputs.pop(field))
                        for field in ("input_ids", "attention_mask", "labels"):
                            if field in inputs:
                                self.trace_arrays[prefix + field] = _array(inputs[field])
                    loss = super().training_step(current_model, inputs, num_items_in_batch)
                    gradients = {}
                    for name, parameter in current_model.named_parameters():
                        if not bool(parameter.requires_grad):
                            continue
                        gradient = getattr(parameter, "grad", None)
                        if gradient is None:
                            raise AssertionError("missing trainable gradient: %s" % name)
                        value = _array(gradient)
                        if not np.all(np.isfinite(value)):
                            raise AssertionError("non-finite trainable gradient: %s" % name)
                        gradients[name] = value
                        self.trace_arrays[
                            "step/%04d/gradient/%02d/%s"
                            % (upcoming_step, microbatch, name)
                        ] = value
                    self.gradients.append(gradients)
                    return loss

            def stateful_loader_factory(owner):
                from torchdata.stateful_dataloader import StatefulDataLoader
                from torchdata.stateful_dataloader.sampler import BatchSampler

                sampler = owner._get_train_sampler()
                generator = None
                if args.explicit_generators:
                    generator = torch.Generator().manual_seed(args.dataloader_seed)
                owner.dataloader_generator = generator
                return StatefulDataLoader(
                    owner.train_dataset,
                    batch_sampler=BatchSampler(sampler, 2, False),
                    collate_fn=owner.data_collator,
                    generator=generator,
                    num_workers=args.dataloader_num_workers,
                    prefetch_factor=(args.dataloader_prefetch_factor
                                     if args.dataloader_num_workers else None),
                    persistent_workers=args.persistent_workers,
                )

            stateful_options = ({"stateful_dataset_id": "tiny-bert-wordlevel-v1-size-%d-transform-%s"
                                 % (len(train_dataset), args.worker_random_transform)}
                                if args.stateful_dataloader else {})
            if args.stateful_dataloader:
                stateful_options["stateful_dataloader_factory"] = stateful_loader_factory
            trainer = RecordingTrainer(
                model=model,
                args=train_args,
                train_dataset=train_dataset,
                data_collator=collator,
                processing_class=tokenizer,
                callbacks=[StopAfterStep()],
                **stateful_options,
            )

            class CaptureTrainerState(TrainerCallback):
                def on_step_end(self, callback_args, state, control, **kwargs):
                    step = int(state.global_step)
                    live_model = kwargs.get("model")
                    optimizer = kwargs.get("optimizer")
                    scheduler = kwargs.get("lr_scheduler")
                    if live_model is None or optimizer is None or scheduler is None:
                        raise AssertionError("Trainer did not expose model/optimizer/scheduler at step end")
                    _record_step_state(live_model, optimizer, scheduler, step, trainer.trace_arrays)
                    return control

            trainer.add_callback(CaptureTrainerState())
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            train_output = trainer.train(resume_from_checkpoint=args.resume_checkpoint)
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            expected_step = args.stop_after if args.stop_after is not None else args.max_steps
            if int(trainer.state.global_step) != expected_step:
                raise AssertionError(
                    "Trainer ended at step %d, expected %d"
                    % (trainer.state.global_step, expected_step)
                )
            if len(trainer.gradients) != expected_step * train_args.gradient_accumulation_steps - (
                int(args.resume_checkpoint is not None) * args.start_step
                * train_args.gradient_accumulation_steps
            ):
                expected_batches = len(trainer.sample_order) // train_args.per_device_train_batch_size
                if len(trainer.gradients) != expected_batches:
                    raise AssertionError("gradient and microbatch counts diverged")
            final_parameters = {
                "final/parameter/%s" % name: _array(parameter)
                for name, parameter in trainer.model.named_parameters()
                if bool(parameter.requires_grad)
            }
            arrays = dict(trainer.trace_arrays)
            arrays.update(final_parameters)
            arrays["sample_order"] = np.asarray(trainer.sample_order, dtype=np.int64)
            arrays["trainable_names"] = np.asarray(
                sorted(name for name, parameter in trainer.model.named_parameters()
                       if bool(parameter.requires_grad)), dtype="U"
            )
            artifact = output_dir / "trajectory.npz"
            np.savez(artifact, **arrays)

            fallback_count = (
                jt.core.backend_fallback_count() - fallback_before if jt is not None else 0
            )
            if fallback_count:
                raise AssertionError("nonzero Jittor backend fallback count")
            report = {
                "runtime": args.runtime,
                "device": args.device,
                "mode": "resume" if args.resume_checkpoint else "train",
                "resume_checkpoint": args.resume_checkpoint,
                "start_step": args.start_step,
                "global_step": int(trainer.state.global_step),
                "max_steps": args.max_steps,
                "stop_after": args.stop_after,
                "sampler": args.sampler,
                "model_dropout": args.dropout,
                "dataset_size": len(train_dataset),
                "dataloader_num_workers": args.dataloader_num_workers,
                "dataloader_prefetch_factor": (
                    args.dataloader_prefetch_factor
                    if args.dataloader_num_workers > 0 else None
                ),
                "persistent_workers": args.persistent_workers,
                "worker_random_transform": args.worker_random_transform,
                "explicit_generators": args.explicit_generators,
                "sampler_generator_initial_seed": (
                    args.sampler_seed if args.explicit_generators else None
                ),
                "dataloader_generator_initial_seed": (
                    args.dataloader_seed if args.explicit_generators else None
                ),
                "sampler_generator_final_state_sha256": _generator_state_sha256(
                    trainer.sampler_generator
                ),
                "dataloader_generator_final_state_sha256": _generator_state_sha256(
                    trainer.dataloader_generator
                ),
                "stateful_dataloader": args.stateful_dataloader,
                "stateful_restored_step": getattr(trainer, "stateful_restored_step", None),
                "stateful_saved_checkpoints": getattr(trainer, "stateful_saved_checkpoints", []),
                "custom_generator_states_in_trainer_checkpoint": args.stateful_dataloader,
                "trainer_epoch": trainer.state.epoch,
                "actual_parameter_devices": sorted({str(p.device) for p in trainer.model.parameters()}),
                "sample_order": trainer.sample_order,
                "microbatches": len(trainer.gradients),
                "trainable_parameter_tensors": len(final_parameters),
                "fallback_count": fallback_count,
                "elapsed_seconds": elapsed,
                "train_loss": train_output.metrics.get("train_loss"),
                "artifact": str(artifact),
                "output_dir": str(output_dir),
            }
            (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, sort_keys=True))
        finally:
            if fallback_guard is not None:
                fallback_guard.__exit__(None, None, None)


def _merge(args):
    with np.load(args.first, allow_pickle=False) as first, np.load(
        args.resumed, allow_pickle=False
    ) as resumed:
        merged = {}
        for archive_index, archive in enumerate((first, resumed)):
            for key in archive.files:
                if key == "sample_order":
                    continue
                if archive_index == 0 and not key.startswith("step/"):
                    continue
                if key in merged:
                    raise AssertionError("duplicate trajectory state key: %s" % key)
                merged[key] = np.asarray(archive[key])
        merged["sample_order"] = np.concatenate(
            [first["sample_order"], resumed["sample_order"]]
        )
        np.savez(args.output, **merged)
    print(json.dumps({"merged": args.output}))


def _compare(args):
    with np.load(args.left, allow_pickle=False) as left, np.load(
        args.right, allow_pickle=False
    ) as right:
        if set(left.files) != set(right.files):
            raise AssertionError("trajectory keys differ: left-only=%s right-only=%s" % (
                sorted(set(left.files) - set(right.files)),
                sorted(set(right.files) - set(left.files)),
            ))
        failures = []
        worst = {"key": None, "max_abs": 0.0}
        for name in sorted(left.files):
            a, b = np.asarray(left[name]), np.asarray(right[name])
            if a.shape != b.shape or a.dtype != b.dtype:
                failures.append({"key": name, "kind": "shape_or_dtype"})
                continue
            if a.dtype.kind in "US":
                if not np.array_equal(a, b):
                    failures.append({"key": name, "kind": "exact"})
                continue
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                failures.append({"key": name, "kind": "non_finite"})
                continue
            difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
            max_abs = float(difference.max()) if difference.size else 0.0
            if max_abs > worst["max_abs"]:
                worst = {"key": name, "max_abs": max_abs}
            if not np.allclose(a, b, atol=args.atol, rtol=args.rtol, equal_nan=False):
                failures.append({"key": name, "max_abs": max_abs})
        report = {
            "left": args.left,
            "right": args.right,
            "items": len(left.files),
            "atol": args.atol,
            "rtol": args.rtol,
            "worst": worst,
            "failure_count": len(failures),
            "failures": failures[:20],
        }
        print(json.dumps(report, sort_keys=True))
        if failures:
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--runtime", choices=("pytorch",), required=True)
    initialize.add_argument("--weights-out", required=True)
    initialize.add_argument("--seed", type=int, default=20261014)
    run = subparsers.add_parser("run")
    run.add_argument("--runtime", choices=("pytorch", "jittor"), required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--weights", required=True)
    run.add_argument("--resume-checkpoint", default=None)
    run.add_argument("--start-step", type=int, default=0)
    run.add_argument("--stop-after", type=int, default=None)
    run.add_argument("--max-steps", type=int, default=4)
    run.add_argument("--seed", type=int, default=20261014)
    run.add_argument("--sampler", choices=("sequential", "random"), default="sequential")
    run.add_argument("--dropout", type=float, default=0.0)
    run.add_argument("--dataset-size", type=int, default=None)
    run.add_argument("--dataloader-num-workers", type=int, default=0)
    run.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    run.add_argument("--persistent-workers", action="store_true")
    run.add_argument("--worker-random-transform", action="store_true")
    run.add_argument("--explicit-generators", action="store_true")
    run.add_argument("--stateful-dataloader", action="store_true")
    run.add_argument("--sampler-seed", type=int, default=None)
    run.add_argument("--dataloader-seed", type=int, default=None)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--first", required=True)
    merge.add_argument("--resumed", required=True)
    merge.add_argument("--output", required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--atol", type=float, default=0.0)
    compare.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    if args.command == "run":
        if args.sampler_seed is None:
            args.sampler_seed = args.seed + 11
        if args.dataloader_seed is None:
            args.dataloader_seed = args.seed + 29
    if args.command == "initialize":
        if args.runtime != "pytorch":
            parser.error("initial fixtures must be produced by independent PyTorch")
        _, torch = _runtime("pytorch")
        _initialize(torch, args.weights_out, args.seed)
    elif args.command == "run":
        if args.start_step < 0 or args.max_steps <= 0:
            parser.error("step counts must be positive")
        if args.dataset_size is not None and args.dataset_size <= 0:
            parser.error("--dataset-size must be positive")
        if args.dataloader_num_workers < 0:
            parser.error("--dataloader-num-workers must be nonnegative")
        if args.dataloader_num_workers > 0 and args.dataloader_prefetch_factor <= 0:
            parser.error("--dataloader-prefetch-factor must be positive")
        if args.persistent_workers and args.dataloader_num_workers == 0:
            parser.error("--persistent-workers requires dataloader workers")
        if args.worker_random_transform and args.dataloader_num_workers == 0:
            parser.error("--worker-random-transform requires dataloader workers")
        if args.explicit_generators and args.sampler != "random":
            parser.error("--explicit-generators requires --sampler=random")
        if args.resume_checkpoint and args.start_step <= 0:
            parser.error("--start-step must name the checkpoint global step")
        if args.stop_after is not None and not args.start_step < args.stop_after <= args.max_steps:
            parser.error("--stop-after must be after start-step and at most max-steps")
        _run(args)
    elif args.command == "merge":
        _merge(args)
    else:
        _compare(args)


if __name__ == "__main__":
    main()
