"""Bounded real-data Trainer workflows in independent torch/shim processes.

initialize uses independent PyTorch to freeze shared public model/adapter files.
run never changes Trainer's ordinary sampler unless --fixed-order is requested.
compare checks final complete state and an optional continuous-run prefix against
a fresh-process resume. Performance must be measured in a separate harness.
"""

import argparse
from contextlib import ExitStack
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from _hf_trainer_e2e_runner import _runtime
from _release_hf_data import (
    TASKS, build_data, digest_file, initialize, load_model, metrics_for,
    resolve_initial,
)
from _release_hf_lifecycle import evaluation_probe, lifecycle, on_device
from _release_hf_observe import (
    _GRADIENT_NORM_ALGORITHM, observer_classes, register_projection_observer,
)
from _release_hf_state import (
    assert_same_signature, compare_archives, compare_runs, frozen_signature, save_state,
)


def _versions():
    packages = ("transformers", "accelerate", "peft", "tokenizers", "safetensors", "numpy")
    return {name: importlib.metadata.version(name) for name in packages}


def _training_arguments(args, expected_updates):
    from transformers import Seq2SeqTrainingArguments, TrainingArguments

    constructor = Seq2SeqTrainingArguments if args.task == "t5" else TrainingArguments
    arguments = dict(
        output_dir=str(Path(args.output_dir) / "checkpoints"),
        use_cpu=args.device == "cpu", seed=args.seed, data_seed=args.seed,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        weight_decay=0.01, max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="linear", warmup_ratio=args.warmup_ratio,
        optim="adamw_torch", eval_strategy="steps", save_strategy="steps",
        # Explicit public callbacks request the sparse shared save/eval schedule.
        # The framework's automatic schedule is beyond the bounded run budget.
        eval_steps=expected_updates + 1, save_steps=expected_updates + 1,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="accuracy" if args.task == "bert" else "eval_loss",
        greater_is_better=args.task == "bert", save_safetensors=True,
        logging_strategy="steps", logging_steps=1, disable_tqdm=True, report_to=[],
        dataloader_num_workers=args.num_workers,
        dataloader_drop_last=False, remove_unused_columns=False,
        bf16=args.mixed_precision == "bf16", fp16=args.mixed_precision == "fp16",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
    )
    if args.task == "t5":
        arguments.update(predict_with_generate=True,
                         generation_max_length=args.max_new_tokens + 1,
                         generation_num_beams=1)
    return constructor(**arguments)


def _run(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    jt, torch = _runtime(args.runtime)
    from transformers import AutoTokenizer, Seq2SeqTrainer, Trainer, TrainerCallback

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA runtime is unavailable")
    if args.device == "cpu" and args.mixed_precision != "no":
        raise ValueError("release AMP fixtures require actual CUDA")
    if args.mixed_precision == "fp16" and args.max_grad_norm <= 0:
        raise ValueError("FP16 health fixture requires positive gradient clipping for pre-step unscale observation")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("workflow output destination already exists: %s" % output)
    output.mkdir(parents=True)
    with ExitStack() as stack:
        if jt is not None:
            stack.enter_context(jt.runtime.scope(use_cuda=int(args.device == "cuda"),
                                                backend_fallback="error"))
            from jittor._runtime.fallback import forbid_backend_fallbacks

            stack.enter_context(forbid_backend_fallbacks())
            fallback_before = jt.core.backend_fallback_count()
        else:
            fallback_before = None
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        base_dir, adapter_dir, initialization = resolve_initial(args)
        model = load_model(torch, args.task, base_dir, adapter_dir)
        if args.task == "qwen-lora":
            model.config.use_cache = not args.gradient_checkpointing
        tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
        train_dataset, eval_dataset, collator = build_data(args, torch, model, tokenizer)
        microbatches_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
        updates_per_epoch = math.ceil(microbatches_per_epoch / args.gradient_accumulation_steps)
        args.expected_updates = args.epochs * updates_per_epoch
        if not args.save_at:
            args.save_at = [updates_per_epoch, args.expected_updates]
        if any(step < 1 or step > args.expected_updates for step in args.save_at):
            raise ValueError("--save-at steps must be inside the training budget")
        if args.stop_after is not None and not 0 < args.stop_after <= args.expected_updates:
            raise ValueError("--stop-after is outside the training budget")
        start_step = 0
        if args.resume_checkpoint:
            saved = json.loads((Path(args.resume_checkpoint) / "trainer_state.json").read_text())
            start_step = int(saved["global_step"])
            if start_step >= args.expected_updates:
                raise ValueError("resume checkpoint has already completed the training budget")
        train_args = _training_arguments(args, args.expected_updates)
        trainer_base = Seq2SeqTrainer if args.task == "t5" else Trainer
        RecordingTrainer, CaptureCallback = observer_classes(torch, trainer_base, TrainerCallback, args)
        callback = CaptureCallback()
        keywords = dict(
            model=model, args=train_args, train_dataset=train_dataset,
            eval_dataset=eval_dataset, data_collator=collator,
            processing_class=tokenizer, callbacks=[callback],
        )
        if args.task in ("bert", "t5"):
            keywords["compute_metrics"] = metrics_for(args.task, tokenizer)
        trainer = RecordingTrainer(**keywords)
        callback.owner = trainer
        actual_devices = on_device(torch, trainer.model, args.device, jt)
        hook = register_projection_observer(torch, trainer.model, trainer, args.device)
        stack.callback(hook.remove)
        frozen_before = frozen_signature(trainer.model)
        initial, initial_metadata = evaluation_probe(
            trainer.model, collator, eval_dataset, args.device,
            autocast=trainer.accelerator.autocast())
        np.savez(output / "initial-probe.npz", **initial)
        del initial
        if args.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        train_output = trainer.train(resume_from_checkpoint=args.resume_checkpoint)
        if args.device == "cuda":
            torch.cuda.synchronize()
        train_elapsed = time.perf_counter() - started
        expected_end = args.stop_after or args.expected_updates
        if int(trainer.state.global_step) != expected_end:
            raise AssertionError("Trainer completed %d updates, expected %d" %
                                 (trainer.state.global_step, expected_end))
        expected_steps = list(range(start_step + 1, expected_end + 1))
        if [item["step"] for item in trainer.updates] != expected_steps:
            raise AssertionError("observed optimizer updates do not cover the expected cursor")
        if len(trainer.unscaled_gradients) != len(expected_steps):
            raise AssertionError("not every update exposed unscaled gradients")
        if args.mixed_precision != "fp16" and any(item["optimizer_step_skipped"] for item in trainer.updates):
            raise AssertionError("ordinary finite workflow unexpectedly skipped an optimizer update")
        if trainer.successful_optimizer_updates + trainer.skipped_optimizer_attempts != expected_end:
            raise AssertionError("attempted windows do not equal successful updates plus validated skips")
        if trainer.successful_optimizer_updates < 1:
            raise AssertionError("training budget contains no successful optimizer update after scale calibration")
        if args.mixed_precision != "no":
            required = "bfloat16" if args.mixed_precision == "bf16" else "float16"
            if not any(required in dtype for dtype in trainer.projection_dtypes):
                raise AssertionError("AMP did not execute a projection in the requested low precision")
            if True not in trainer.observed_autocast:
                raise AssertionError("AMP was not enabled inside the observed projection")
        if args.mixed_precision == "fp16" and trainer.accelerator.scaler is None:
            raise AssertionError("FP16 Trainer did not create a GradScaler")
        frozen_after = frozen_signature(trainer.model)
        assert_same_signature(frozen_before, frozen_after, "frozen base parameters")
        save_state(output / "final-state.npz", trainer.model, trainer.optimizer,
                   trainer.lr_scheduler, trainer.accelerator.scaler, args.state_format)
        evaluation = trainer.evaluate()
        prediction = trainer.predict(eval_dataset)
        if not np.isfinite(float(evaluation["eval_loss"])):
            raise AssertionError("Trainer evaluation loss is nonfinite")
        if not trainer.state.best_model_checkpoint:
            raise AssertionError("best-checkpoint selection did not record a public checkpoint")
        result_lifecycle = lifecycle(args, torch, trainer, tokenizer, eval_dataset, collator,
                                     base_dir, jt)
        fallback_count = (jt.core.backend_fallback_count() - fallback_before if jt is not None else 0)
        if fallback_count != 0:
            raise AssertionError("Jittor backend fallback count is nonzero")
        expected_dtype = {str(parameter.dtype) for parameter in trainer.model.parameters()}
        optimizer_dtypes = sorted({str(value.dtype)
                                   for state in trainer.optimizer.state.values()
                                   for value in state.values() if hasattr(value, "dtype")})
        report = {
            "runtime": args.runtime, "task": args.task, "device": args.device,
            "seed": args.seed, "versions": _versions(), "initialization": initialization,
            "gradient_norm_algorithm": _GRADIENT_NORM_ALGORITHM,
            "max_grad_norm": float(args.max_grad_norm),
            "optimizer_gradient_phase": ("postclip_unscaled" if args.max_grad_norm > 0
                                         else "pre_optimizer_unscaled"),
            "initial_model_dir": str(base_dir), "initial_adapter_dir": adapter_dir,
            "train_file_sha256": digest_file(args.train_file),
            "eval_file_sha256": digest_file(args.eval_file),
            "train_samples": len(train_dataset), "eval_samples": len(eval_dataset),
            "epochs": args.epochs, "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "microbatches_per_epoch": microbatches_per_epoch, "updates_per_epoch": updates_per_epoch,
            "expected_updates": args.expected_updates, "global_step": int(trainer.state.global_step),
            "global_step_semantics": "attempted optimizer windows; successful updates counted separately",
            "expected_optimizer_attempts": args.expected_updates,
            "optimizer_updates_completed": trainer.successful_optimizer_updates,
            "optimizer_skips_completed": trainer.skipped_optimizer_attempts,
            "optimizer_attempts_in_process": len(trainer.updates),
            "optimizer_updates_in_process": sum(item["optimizer_update_performed"] for item in trainer.updates),
            "trainer_epoch": float(trainer.state.epoch), "start_step": start_step,
            "resume_checkpoint": args.resume_checkpoint, "resume_loaded_state_exact": trainer.resume_state_exact,
            "fixed_order": args.fixed_order, "sampler": "sequential fixture" if args.fixed_order else "default Trainer sampler",
            "num_workers": args.num_workers, "mixed_precision": args.mixed_precision,
            "gradient_checkpointing": args.gradient_checkpointing,
            "actual_parameter_devices": actual_devices, "fallback_count": fallback_count,
            "initial_probe": initial_metadata, "projection_dtypes": sorted(trainer.projection_dtypes),
            "observed_autocast": sorted(trainer.observed_autocast),
            "master_parameter_dtypes": sorted(expected_dtype), "optimizer_state_dtypes": optimizer_dtypes,
            "scaler_enabled": trainer.accelerator.scaler is not None,
            "initial_scaler_state": trainer.initial_scaler_state,
            "final_scaler_state": trainer.accelerator.scaler.state_dict() if trainer.accelerator.scaler else None,
            "frozen_base_unchanged": frozen_before == frozen_after,
            "state_format": args.state_format, "checkpoint_signatures": trainer.checkpoint_signatures,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "train_metrics": train_output.metrics, "eval_metrics": evaluation,
            "predict_metrics": prediction.metrics, "microbatches": trainer.microbatches,
            "updates": trainer.updates, "unscaled_gradients": trainer.unscaled_gradients,
            "tail_microbatch_observed": any(item["batch_size"] < args.batch_size for item in trainer.microbatches),
            "tail_accumulation_observed": any(item["microbatches"] < args.gradient_accumulation_steps
                                               for item in trainer.updates),
            "lifecycle": result_lifecycle, "train_elapsed_seconds_diagnostic": train_elapsed,
            "coverage": "bounded functional acceptance; no performance claim",
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items()
                          if key not in ("microbatches", "updates", "unscaled_gradients")}, sort_keys=True))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("initialize")
    init.add_argument("--task", choices=TASKS, required=True)
    init.add_argument("--runtime", choices=("torch",), default="torch")
    init.add_argument("--model-dir", required=True)
    init.add_argument("--output-dir", required=True)
    init.add_argument("--seed", type=int, default=821)
    init.add_argument("--compact", action="store_true")
    init.add_argument("--dropout", type=float, default=None)
    run = commands.add_parser("run")
    run.add_argument("--task", choices=TASKS, required=True)
    run.add_argument("--runtime", choices=("torch", "jittor"), required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run.add_argument("--initialization-dir")
    run.add_argument("--initial-model-dir")
    run.add_argument("--initial-adapter-dir")
    run.add_argument("--train-file", required=True)
    run.add_argument("--eval-file", required=True)
    run.add_argument("--train-samples", type=int)
    run.add_argument("--eval-samples", type=int)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--epochs", type=int, default=2)
    run.add_argument("--batch-size", type=int, default=2)
    run.add_argument("--eval-batch-size", type=int, default=2)
    run.add_argument("--gradient-accumulation-steps", type=int, default=2)
    run.add_argument("--max-length", type=int, default=96)
    run.add_argument("--max-target-length", type=int, default=48)
    run.add_argument("--max-new-tokens", type=int, default=8)
    run.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    run.add_argument("--learning-rate", type=float, default=3e-5)
    run.add_argument("--warmup-ratio", type=float, default=0.05)
    run.add_argument("--max-grad-norm", type=float, default=1.0)
    run.add_argument("--num-workers", type=int, default=0)
    run.add_argument("--seed", type=int, default=821)
    run.add_argument("--save-at", default="", help="comma-separated sparse checkpoint/evaluation steps")
    run.add_argument("--stop-after", type=int)
    run.add_argument("--resume-checkpoint")
    run.add_argument("--fixed-order", action="store_true")
    run.add_argument("--gradient-checkpointing", action="store_true")
    run.add_argument("--skip-merge", action="store_true")
    run.add_argument("--state-format", choices=("full", "hashes"), default="full")
    run.add_argument("--reload-atol", type=float, default=2e-5)
    run.add_argument("--reload-rtol", type=float, default=2e-4)
    compare = commands.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--first", help="stopped first run or continuous run whose prefix precedes resume")
    compare.add_argument("--atol", type=float, default=0.0)
    compare.add_argument("--rtol", type=float, default=0.0)
    compare.add_argument("--report")
    probe = commands.add_parser("compare-probes")
    probe.add_argument("--left", required=True)
    probe.add_argument("--right", required=True)
    probe.add_argument("--atol", type=float, default=1e-5)
    probe.add_argument("--rtol", type=float, default=1e-4)
    probe.add_argument("--report")
    return result


def main():
    args = parser().parse_args()
    if args.command == "initialize":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        _, torch = _runtime("torch")
        print(json.dumps(initialize(args, torch), sort_keys=True))
    elif args.command == "run":
        for name in ("epochs", "batch_size", "eval_batch_size", "gradient_accumulation_steps",
                     "max_length", "max_target_length", "max_new_tokens"):
            if getattr(args, name) < 1:
                raise SystemExit("--%s must be positive" % name.replace("_", "-"))
        args.save_at = sorted(set(int(value) for value in args.save_at.split(",") if value))
        _run(args)
    elif args.command == "compare":
        compare_runs(args)
    else:
        result = compare_archives(args.left, args.right, args.atol, args.rtol)
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.report:
            Path(args.report).write_text(json.dumps(result, indent=2) + "\n")
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
