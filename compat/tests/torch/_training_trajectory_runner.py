"""Run and compare tiny multi-step Torch/PEFT/Accelerate trajectories.

This runner exercises the ``Accelerator.prepare`` path only. It does not provide
a direct-Torch control run, and its fixed synthetic token fixture does not
establish exact continuation for stochastic data or model RNG streams.

Example (invoke once per runtime, always in a fresh process)::

    python _training_trajectory_runner.py run --runtime torch --model-case peft_lora_llama --steps 10 \
        --weights-out /tmp/initial.npz --output /tmp/torch-full.npz
    python _training_trajectory_runner.py run --runtime jittor --steps 10 \
        --weights /tmp/initial.npz --output /tmp/jittor-full.npz

The split run uses ``--stop-after 5 --checkpoint DIR`` followed by a new
process with ``--resume DIR --start-step 5``. ``compare`` merges split outputs
using step-indexed losses (input file order does not matter), then compares all
per-step states with an uninterrupted run or with the other runtime.
"""

import argparse
from contextlib import ExitStack, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ecosystem_cases
from _training_state_helpers import _named_tensors


def _import_runtime(runtime):
    if runtime == "jittor":
        os.environ["JITTOR_TORCH_SHIM"] = "1"
        import jittor as jt

        import torch

        if not hasattr(torch, "_torch_compat_install_context"):
            raise RuntimeError("torch did not resolve to Jittor compatibility shim")
        return jt, torch
    os.environ.pop("JITTOR_TORCH_SHIM", None)
    import torch

    if not hasattr(torch, "_C") or hasattr(torch, "_torch_compat_install_context"):
        raise RuntimeError("torch did not resolve to independent PyTorch")
    return None, torch


def _array(value):
    return value.detach().clone().cpu().numpy().copy()


def _write_initial_weights(model, path):
    np.savez(path, **{name: _array(value) for name, value in _named_tensors(model)})


def _load_initial_weights(torch, model, path):
    saved = np.load(path)
    targets = dict(_named_tensors(model))
    if set(saved.files) != set(targets):
        raise RuntimeError("initial fixture keys differ: missing=%s extra=%s" % (
            sorted(set(targets) - set(saved.files))[:8],
            sorted(set(saved.files) - set(targets))[:8],
        ))
    no_grad = getattr(torch, "no_grad", None)
    scope = no_grad() if callable(no_grad) else nullcontext()
    with scope:
        for name, target in targets.items():
            source = torch.from_numpy(saved[name].copy())
            target.copy_(source)


def _snapshot_gradients(model, step, arrays, prefix, allow_nonfinite=False):
    nonfinite_count = 0
    nonfinite_names = []
    for name, parameter in model.named_parameters():
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        value = _array(grad)
        finite = np.isfinite(value)
        label = "%s/%s" % (prefix, name)
        if not np.all(finite):
            if not allow_nonfinite:
                raise AssertionError("non-finite gradient outside injected overflow: %s" % name)
            nonfinite_count += int((~finite).sum())
            nonfinite_names.append(name)
            code = np.zeros(value.shape, dtype=np.int8)
            code[np.isnan(value)] = 2
            code[np.isposinf(value)] = 1
            code[np.isneginf(value)] = -1
            arrays["state/step_%02d/amp/nonfinite_grad_code/%s" % (step, name)] = code
            value = np.where(finite, value, 0)
        arrays[label] = value
    return nonfinite_count, nonfinite_names


def _snapshot(model, optimizer, step, arrays, include_gradients=True):
    named_parameters = list(model.named_parameters())
    parameters = dict(named_parameters)
    optimizer_owner = getattr(optimizer, "optimizer", optimizer)
    arrays["state/step_%02d/trainable_names" % step] = np.asarray(
        [name for name, value in named_parameters if bool(value.requires_grad)],
        dtype="U",
    )
    for name, parameter in named_parameters:
        arrays["state/step_%02d/param/%s" % (step, name)] = _array(parameter)
        grad = getattr(parameter, "grad", None)
        if include_gradients and grad is not None:
            arrays["state/step_%02d/grad/%s" % (step, name)] = _array(grad)
    for name, buffer in getattr(model, "named_buffers", lambda: [])():
        arrays["state/step_%02d/buffer/%s" % (step, name)] = _array(buffer)
    for name, parameter in parameters.items():
        state = optimizer_owner.state.get(parameter, {})
        for key, value in state.items():
            label = "state/step_%02d/optim/%s/%s" % (step, name, key)
            if key == "step":
                counter = float(_array(value).reshape(-1)[0]) if hasattr(
                    value, "detach") else float(value)
                if not np.isfinite(counter) or counter < 0 or counter != int(counter):
                    raise AssertionError("optimizer step must be a finite non-negative integer")
                # Torch exposes this count as a float32 scalar tensor, while
                # the shim's live state is a Python float. Compare its meaning
                # as a counter, not the container's representation.
                arrays[label] = np.asarray(int(counter), dtype=np.int64)
                continue
            if hasattr(value, "detach"):
                arrays[label] = _array(value)
            else:
                arrays[label] = np.asarray(value)
    arrays["state/step_%02d/lr" % step] = np.asarray(
        [float(group["lr"]) for group in optimizer_owner.param_groups], dtype=np.float64
    )


def _hash_named(model, names):
    digest = hashlib.sha256()
    by_name = dict(model.named_parameters())
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(_array(by_name[name]).tobytes())
    return digest.hexdigest()


def _assert_device(tensor, expected, name):
    actual = str(getattr(tensor, "device", "cpu"))
    is_cuda = "cuda" in actual.lower()
    if is_cuda != (expected == "cuda"):
        raise AssertionError("%s is on %s, expected %s" % (name, actual, expected))


def _assert_complete_updates(start_step, stop_after, completed_updates, pending_batches):
    expected_updates = stop_after - start_step
    if completed_updates != expected_updates or pending_batches:
        raise AssertionError(
            "expected %d complete updates, completed %d with %d pending batches"
            % (expected_updates, completed_updates, pending_batches)
        )


def _run(args):
    jt, torch = _import_runtime(args.runtime)
    from accelerate import Accelerator

    stack = ExitStack()
    if jt is not None:
        stack.enter_context(jt.runtime.scope(
            use_cuda=int(args.device == "cuda"), backend_fallback="error"
        ))
        from jittor._runtime.fallback import forbid_backend_fallbacks

        stack.enter_context(forbid_backend_fallbacks())
        fallback_before = jt.core.backend_fallback_count()
    else:
        fallback_before = None

    with stack:
        torch.manual_seed(20261013)
        builder, _ = _ecosystem_cases.CASES[args.model_case]
        model, _ = builder(torch)
        if not hasattr(model, "config") or not hasattr(model.config, "use_cache"):
            raise AssertionError("training trajectory model must expose config.use_cache")
        model.config.use_cache = False
        model.train()
        observed_transformer_projection_dtypes = set()
        projection = next((module for name, module in model.named_modules()
                           if name.endswith("self_attn.q_proj")), None)
        if projection is None:
            raise AssertionError("tiny Llama case has no self-attention q_proj")

        def _record_projection_dtype(_module, _inputs, output):
            values = output if isinstance(output, (tuple, list)) else (output,)
            observed_transformer_projection_dtypes.update(
                str(value.dtype) for value in values if hasattr(value, "dtype")
            )

        projection.register_forward_hook(_record_projection_dtype)
        first_parameter = next(model.parameters())
        # Independent PyTorch constructs the model on CPU and Accelerate moves
        # it to CUDA during ``prepare``.  Jittor may construct directly on the
        # requested runtime device.  Validate the requested CPU path before
        # prepare, while deferring the CUDA requirement to the post-prepare
        # assertion below.
        if args.device == "cpu":
            _assert_device(first_parameter, "cpu", "initial model parameter")
        if jt is not None:
            active_cuda = bool(jt.introspection.policy.runtime.use_cuda)
            if active_cuda != (args.device == "cuda"):
                raise AssertionError("Jittor runtime device flag disagrees with requested device")
        trainable = {name for name, value in model.named_parameters()
                     if bool(value.requires_grad)}
        frozen = set(dict(model.named_parameters())) - trainable

        if args.weights:
            _load_initial_weights(torch, model, args.weights)
        elif args.weights_out:
            _write_initial_weights(model, args.weights_out)
        elif not args.resume:
            raise ValueError("new run requires --weights-out; replay requires --weights")

        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-8,
                weight_decay=0.01, foreach=False, fused=False,
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=3e-3, momentum=0.9, weight_decay=0.01,
            )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.steps,
        )
        rng = np.random.RandomState(20261014)
        sample_count = (
            args.steps * args.gradient_accumulation_steps * args.batch_size
        )
        token_ids = rng.randint(0, 128, size=(sample_count, 8)).astype(np.int64)
        data = torch.from_numpy(token_ids)
        sample_ids = torch.arange(sample_count, dtype=torch.int64)
        # Keep DataLoader iterator seeding off the model/dropout RNG stream.
        # PyTorch creates a base seed even for ``num_workers=0``; using a
        # dedicated generator makes a fresh resume process consume the same
        # loader seed as the uninterrupted process before the next dropout
        # mask is sampled.
        loader_generator = torch.Generator()
        loader_generator.manual_seed(20261015)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data, sample_ids),
            batch_size=args.batch_size,
            shuffle=args.shuffle, num_workers=args.num_workers,
            generator=loader_generator,
        )
        accelerator = Accelerator(
            cpu=args.device == "cpu", mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        model, optimizer, scheduler, loader = accelerator.prepare(
            model, optimizer, scheduler, loader,
        )
        _assert_device(next(model.parameters()), args.device, "prepared model parameter")

        if args.resume:
            accelerator.load_state(args.resume)

        if args.mixed_precision == "fp16" and accelerator.scaler is None:
            raise AssertionError("FP16 Accelerate run did not create a GradScaler")
        scaler_state_at_resume = (
            accelerator.scaler.state_dict()
            if args.resume and accelerator.scaler is not None else None
        )

        # A fresh resume process starts from independently initialized weights;
        # only the state restored from the checkpoint is the training baseline.
        frozen_before = _hash_named(model, sorted(frozen))

        arrays = {}
        active_loader = accelerator.skip_first_batches(
            loader, args.start_step * args.gradient_accumulation_steps
        )
        losses = []
        update_step = args.start_step
        accumulated_loss = 0.0
        accumulated_loss_sum = 0.0
        accumulated_valid_tokens = 0
        accumulated_microbatch_losses = []
        accumulated_microbatch_tokens = []
        accumulated_sample_ids = []
        observed_logits_dtypes = set()
        observed_loss_dtypes = set()
        observed_autocast_enabled = set()
        observed_autocast_dtypes = set()
        overflow_step_observed = False
        recovery_step_observed = False
        recovery_scale = None
        started = time.perf_counter()
        for inputs, batch_sample_ids in active_loader:
            if update_step >= args.stop_after:
                break
            with accelerator.accumulate(model):
                labels = inputs[:, 1:].clone()
                attention_mask = None
                if args.mask_padding:
                    valid_lengths = (_array(batch_sample_ids).astype(np.int64) %
                                     labels.shape[1]) + 1
                    input_positions = np.arange(inputs.shape[1])[None, :]
                    label_positions = np.arange(labels.shape[1])[None, :]
                    valid_input = input_positions < (valid_lengths[:, None] + 1)
                    valid_labels = label_positions < valid_lengths[:, None]
                    input_values = _array(inputs)
                    input_values[~valid_input] = 0
                    inputs = inputs.new_tensor(input_values)
                    label_values = _array(labels)
                    label_values[~valid_labels] = -100
                    labels = labels.new_tensor(label_values)
                    attention_mask = inputs.new_tensor(valid_input.astype(np.int64))

                _assert_device(inputs, args.device, "training input_ids")
                if attention_mask is not None:
                    _assert_device(attention_mask, args.device, "training attention_mask")

                model_inputs = {"input_ids": inputs}
                if attention_mask is not None:
                    model_inputs["attention_mask"] = attention_mask
                with accelerator.autocast():
                    model_outputs = model(**model_inputs)
                    logits = model_outputs.logits
                    observed_autocast_enabled.add(
                        bool(torch.is_autocast_enabled(args.device))
                    )
                    observed_autocast_dtypes.add(
                        str(torch.get_autocast_dtype(args.device))
                    )
                    flat_logits = logits[:, :-1, :].reshape(-1, 128)
                    flat_labels = labels.reshape(-1)
                    valid_token_count = int(np.count_nonzero(_array(flat_labels) != -100))
                    if valid_token_count == 0:
                        raise AssertionError("training microbatch has no valid label tokens")
                    loss_sum = torch.nn.functional.cross_entropy(
                        flat_logits, flat_labels, reduction="sum",
                    )
                _assert_device(logits, args.device, "training logits")
                observed_logits_dtypes.add(str(logits.dtype))
                observed_loss_dtypes.add(str(loss_sum.dtype))
                mean_loss = loss_sum / valid_token_count
                if args.loss_normalization == "global_tokens":
                    window_size = args.gradient_accumulation_steps * args.batch_size
                    first_sample = update_step * window_size
                    window_valid_tokens = sum(
                        (sample_id % (labels.shape[1]) + 1)
                        for sample_id in range(first_sample, first_sample + window_size)
                    )
                    loss = loss_sum * args.gradient_accumulation_steps / window_valid_tokens
                else:
                    loss = mean_loss
                if not np.isfinite(float(_array(loss).reshape(-1)[0])):
                    raise AssertionError("training loss became non-finite")
                accelerator.backward(loss)
                microbatch_mean = float(_array(mean_loss).reshape(-1)[0])
                microbatch_sum = float(_array(loss_sum).reshape(-1)[0])
                accumulated_loss += microbatch_mean
                accumulated_loss_sum += microbatch_sum
                accumulated_valid_tokens += valid_token_count
                accumulated_microbatch_losses.append(microbatch_mean)
                accumulated_microbatch_tokens.append(valid_token_count)
                missing_grads = [
                    name for name, parameter in model.named_parameters()
                    if bool(parameter.requires_grad)
                    and getattr(parameter, "grad", None) is None
                ]
                if missing_grads:
                    raise AssertionError(
                        "trainable parameters have no gradient: %s"
                        % missing_grads[:12]
                    )
                accumulated_sample_ids.append(_array(batch_sample_ids))
                should_sync = bool(accelerator.sync_gradients)
                absolute_step = update_step + 1
                inject_overflow = (
                    should_sync and absolute_step == args.inject_overflow_step
                )
                recovery_step = (
                    should_sync and args.inject_overflow_step > 0
                    and absolute_step == args.inject_overflow_step + 1
                )
                optimizer_owner = getattr(optimizer, "optimizer", optimizer)
                lr_before = np.asarray(
                    [float(group["lr"]) for group in optimizer_owner.param_groups],
                    dtype=np.float64,
                ) if should_sync else None
                parameter_hash_before = None
                scaler = accelerator.scaler
                scaler_state_before = scaler.state_dict() if should_sync and scaler else None
                scale_before = (
                    float(scaler_state_before["scale"])
                    if scaler_state_before is not None else None
                )
                if inject_overflow:
                    if scaler is None:
                        raise AssertionError("overflow injection requires an Accelerate GradScaler")
                    trainable_parameters = [
                        (name, parameter) for name, parameter in model.named_parameters()
                        if bool(parameter.requires_grad)
                    ]
                    if not trainable_parameters:
                        raise AssertionError("overflow injection found no trainable parameters")
                    target_name, target_parameter = trainable_parameters[0]
                    target_grad = getattr(target_parameter, "grad", None)
                    if target_grad is None:
                        raise AssertionError("overflow injection target has no gradient")
                    with torch.no_grad():
                        target_grad.copy_(torch.full_like(target_grad, float("inf")))
                    nonfinite_count, nonfinite_names = _snapshot_gradients(
                        model, absolute_step, arrays,
                        "state/step_%02d/pre_optimizer_grad" % absolute_step,
                        allow_nonfinite=True,
                    )
                    if nonfinite_count == 0 or target_name not in nonfinite_names:
                        raise AssertionError("injected overflow gradient was not observable")
                    arrays["state/step_%02d/amp/overflow_injected" % absolute_step] = np.asarray(True)
                    arrays["state/step_%02d/amp/overflow_grad_count" % absolute_step] = np.asarray(
                        nonfinite_count, dtype=np.int64
                    )
                    arrays["state/step_%02d/amp/overflow_grad_names" % absolute_step] = np.asarray(
                        nonfinite_names, dtype="U"
                    )
                    parameter_hash_before = _hash_named(model, sorted(trainable))
                elif recovery_step:
                    parameter_hash_before = _hash_named(model, sorted(trainable))

                optimizer.step()
                scheduler.step()
                if should_sync:
                    step_was_skipped = bool(accelerator.optimizer_step_was_skipped)
                    lr_after = np.asarray(
                        [float(group["lr"]) for group in optimizer_owner.param_groups],
                        dtype=np.float64,
                    )
                    scaler_state_after = scaler.state_dict() if scaler else None
                    scale_after = (
                        float(scaler_state_after["scale"])
                        if scaler_state_after is not None else None
                    )
                    amp_prefix = "state/step_%02d/amp/" % absolute_step
                    arrays[amp_prefix + "mixed_precision"] = np.asarray(args.mixed_precision)
                    arrays[amp_prefix + "logits_dtype"] = np.asarray(str(logits.dtype))
                    arrays[amp_prefix + "q_proj_output_dtypes"] = np.asarray(
                        sorted(observed_transformer_projection_dtypes), dtype="U"
                    )
                    arrays[amp_prefix + "loss_dtype"] = np.asarray(str(loss_sum.dtype))
                    arrays[amp_prefix + "autocast_enabled"] = np.asarray(
                        all(observed_autocast_enabled)
                    )
                    arrays[amp_prefix + "autocast_dtypes"] = np.asarray(
                        sorted(observed_autocast_dtypes), dtype="U"
                    )
                    arrays[amp_prefix + "scaler_enabled"] = np.asarray(scaler is not None)
                    arrays[amp_prefix + "optimizer_step_skipped"] = np.asarray(step_was_skipped)
                    arrays[amp_prefix + "master_parameter_dtypes"] = np.asarray(
                        sorted({str(parameter.dtype) for _, parameter in model.named_parameters()}),
                        dtype="U",
                    )
                    if scaler_state_after is not None:
                        arrays[amp_prefix + "scale"] = np.asarray(scale_after, dtype=np.float64)
                        arrays[amp_prefix + "growth_tracker"] = np.asarray(
                            int(scaler_state_after.get("_growth_tracker", 0)), dtype=np.int64
                        )
                    if inject_overflow:
                        if not step_was_skipped:
                            raise AssertionError("Accelerate did not skip the injected overflow step")
                        if _hash_named(model, sorted(trainable)) != parameter_hash_before:
                            raise AssertionError("parameters changed during an overflow-skipped step")
                        if not np.array_equal(lr_after, lr_before):
                            raise AssertionError("scheduler advanced during an overflow-skipped step")
                        if scale_before is None or scale_after >= scale_before:
                            raise AssertionError("GradScaler did not back off after injected overflow")
                        arrays[amp_prefix + "parameters_unchanged"] = np.asarray(True)
                        arrays[amp_prefix + "scheduler_unchanged"] = np.asarray(True)
                        overflow_step_observed = True
                        recovery_scale = scale_after
                    elif args.mixed_precision == "fp16" and step_was_skipped:
                        raise AssertionError("finite FP16 trajectory step was unexpectedly skipped")
                    if recovery_step:
                        if step_was_skipped:
                            raise AssertionError("GradScaler did not recover on the next finite step")
                        if _hash_named(model, sorted(trainable)) == parameter_hash_before:
                            raise AssertionError("parameters did not update after overflow recovery")
                        if scale_before is None or scale_after != scale_before:
                            raise AssertionError("GradScaler changed scale on the recovery step")
                        arrays[amp_prefix + "recovery_updated_parameters"] = np.asarray(True)
                        recovery_step_observed = True
                    arrays["data/step_%02d/microbatch_mean_losses" % absolute_step] = np.asarray(
                        accumulated_microbatch_losses, dtype=np.float64
                    )
                    arrays["data/step_%02d/microbatch_valid_tokens" % absolute_step] = np.asarray(
                        accumulated_microbatch_tokens, dtype=np.int64
                    )
                    arrays["data/step_%02d/global_token_mean_loss" % absolute_step] = np.asarray(
                        accumulated_loss_sum / accumulated_valid_tokens,
                        dtype=np.float64,
                    )
                    arrays["data/step_%02d/valid_token_count" % absolute_step] = np.asarray(
                        accumulated_valid_tokens, dtype=np.int64
                    )
                    # Capture the accumulated gradients before zero_grad while
                    # params and optimizer state already reflect this update.
                    _snapshot(
                        model, optimizer, absolute_step, arrays,
                        include_gradients=not inject_overflow,
                    )
                optimizer.zero_grad(set_to_none=True)
            if should_sync:
                absolute_step = update_step + 1
                arrays["data/step_%02d/sample_ids" % absolute_step] = np.concatenate(
                    accumulated_sample_ids, axis=0
                )
                step_loss = (
                    (accumulated_loss / args.gradient_accumulation_steps
                     if args.loss_normalization == "microbatch_mean"
                     else accumulated_loss_sum / accumulated_valid_tokens)
                )
                losses.append(step_loss)
                arrays["data/step_%02d/loss" % absolute_step] = np.asarray(
                    step_loss, dtype=np.float64
                )
                accumulated_loss = 0.0
                accumulated_loss_sum = 0.0
                accumulated_valid_tokens = 0
                accumulated_microbatch_losses = []
                accumulated_microbatch_tokens = []
                accumulated_sample_ids = []
                update_step = absolute_step
                if args.checkpoint and absolute_step == args.save_at:
                    accelerator.save_state(
                        args.checkpoint, safe_serialization=False
                    )
        _assert_complete_updates(
            args.start_step, args.stop_after, len(losses), len(accumulated_sample_ids)
        )
        if (args.inject_overflow_step > 0
                and args.start_step < args.inject_overflow_step <= args.stop_after
                and not overflow_step_observed):
            raise AssertionError("requested injected overflow step did not execute")
        if (args.inject_overflow_step > 0
                and args.start_step < args.inject_overflow_step + 1 <= args.stop_after
                and not recovery_step_observed):
            raise AssertionError("requested post-overflow recovery step did not execute")
        elapsed = time.perf_counter() - started
        arrays["losses"] = np.asarray(losses, dtype=np.float64)
        frozen_after = _hash_named(model, sorted(frozen))
        arrays["frozen_before_sha256"] = np.asarray(frozen_before)
        arrays["frozen_after_sha256"] = np.asarray(frozen_after)
        arrays["trainable_names"] = np.asarray(sorted(trainable), dtype="U")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **arrays)
        fallback_count = (jt.core.backend_fallback_count() - fallback_before
                          if jt is not None else 0)
        report = {
            "runtime": args.runtime,
            "device": args.device,
            "model_case": args.model_case,
            "use_cache": bool(model.config.use_cache),
            "shuffle": args.shuffle,
            "num_workers": args.num_workers,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "mask_padding": args.mask_padding,
            "loss_normalization": args.loss_normalization,
            "optimizer": args.optimizer,
            "mixed_precision": args.mixed_precision,
            "scaler_enabled": accelerator.scaler is not None,
            "scaler_state_at_resume": scaler_state_at_resume,
            "observed_logits_dtypes": sorted(observed_logits_dtypes),
            "observed_q_proj_output_dtypes": sorted(observed_transformer_projection_dtypes),
            "observed_loss_dtypes": sorted(observed_loss_dtypes),
            "observed_autocast_enabled": sorted(observed_autocast_enabled),
            "observed_autocast_dtypes": sorted(observed_autocast_dtypes),
            "master_parameter_dtypes": sorted(
                {str(parameter.dtype) for _, parameter in model.named_parameters()}
            ),
            "inject_overflow_step": args.inject_overflow_step,
            "overflow_step_observed": overflow_step_observed,
            "overflow_scale_after": recovery_scale,
            "recovery_step_observed": recovery_step_observed,
            "loader_generator_seed": 20261015,
            "steps_saved": sorted({int(key.split("/")[1][5:]) for key in arrays
                                    if key.startswith("state/step_")}),
            "updates_in_this_process": len(losses),
            "data_cursor": args.stop_after,
            "trainable_count": len(trainable),
            "frozen_count": len(frozen),
            "frozen_base_unchanged": frozen_before == frozen_after,
            "fallback_count": fallback_count,
            "elapsed_seconds": elapsed,
            "losses": losses,
            "coverage_boundary": {
                "direct_torch_control": "not_covered",
                "stochastic_data_or_model_rng_resume": "not_covered",
            },
            "output": args.output,
            "checkpoint": args.checkpoint,
        }
        if fallback_count:
            raise AssertionError("Jittor backend fallback count was nonzero")
        if frozen_before != frozen_after:
            raise AssertionError("frozen base parameters changed during the trajectory")
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, sort_keys=True))


def _step_loss_index(name):
    prefix = "data/step_"
    suffix = "/loss"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    value = name[len(prefix):-len(suffix)]
    try:
        return int(value)
    except ValueError:
        return None


def _arrays_equal(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.kind in "fc":
        return np.array_equal(left, right, equal_nan=True)
    return np.array_equal(left, right)


def _merge(paths):
    merged = {}
    indexed_losses = {}
    unindexed_loss_arrays = []
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            source_losses = None
            source_step_losses = {}
            for name in source.files:
                if name == "losses":
                    source_losses = source[name].copy()
                    continue
                step = _step_loss_index(name)
                if step is not None:
                    value = source[name].copy()
                    if value.size != 1 or value.dtype.kind not in "biufc":
                        raise AssertionError("step loss must be one numeric value: %s" % name)
                    if step in source_step_losses:
                        raise AssertionError("duplicate step loss key in %s: %d" % (path, step))
                    source_step_losses[step] = value
                    continue
                if name in merged:
                    if not _arrays_equal(merged[name], source[name]):
                        raise AssertionError("duplicate state differs: %s" % name)
                else:
                    merged[name] = source[name].copy()
            if source_losses is not None:
                if source_step_losses:
                    if any(value.dtype != source_losses.dtype
                           for value in source_step_losses.values()):
                        raise AssertionError(
                            "step loss dtype disagrees with losses array in %s" % path
                        )
                    ordered = [source_step_losses[step] for step in sorted(source_step_losses)]
                    indexed = np.asarray([value.reshape(-1)[0] for value in ordered],
                                         dtype=source_losses.dtype)
                    if indexed.shape != source_losses.shape or not np.array_equal(
                            indexed, source_losses, equal_nan=True):
                        raise AssertionError(
                            "losses array disagrees with step-indexed losses in %s" % path
                        )
                else:
                    unindexed_loss_arrays.append((path, source_losses))
            elif source_step_losses:
                raise AssertionError("step-indexed losses require a losses array in %s" % path)
            for step, value in source_step_losses.items():
                if indexed_losses and value.dtype != next(iter(indexed_losses.values())).dtype:
                    raise AssertionError("step loss dtypes differ across artifacts")
                if step in indexed_losses and not np.array_equal(
                        indexed_losses[step], value, equal_nan=True):
                    raise AssertionError("duplicate step loss differs at update %d" % step)
                indexed_losses[step] = value

    if indexed_losses:
        if unindexed_loss_arrays:
            raise AssertionError("cannot merge indexed and unindexed loss artifacts")
        ordered_steps = sorted(indexed_losses)
        if ordered_steps != list(range(ordered_steps[0], ordered_steps[-1] + 1)):
            raise AssertionError("merged step losses contain a gap: %s" % ordered_steps)
        merged["losses"] = np.asarray(
            [indexed_losses[step].reshape(-1)[0] for step in ordered_steps],
            dtype=indexed_losses[ordered_steps[0]].dtype,
        )
        for step in ordered_steps:
            merged["data/step_%02d/loss" % step] = indexed_losses[step]
    elif unindexed_loss_arrays:
        if len(unindexed_loss_arrays) != 1:
            raise AssertionError(
                "multi-artifact merge requires step-indexed losses; rerun with the current runner"
            )
        merged["losses"] = unindexed_loss_arrays[0][1]
    return merged


def _compare_array(name, left, right, atol, rtol):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        return {"key": name, "kind": "shape", "left": list(a.shape),
                "right": list(b.shape)}
    if a.dtype != b.dtype:
        return {"key": name, "kind": "dtype", "left": str(a.dtype),
                "right": str(b.dtype)}
    if a.dtype.kind in "biufc":
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return {"key": name, "kind": "non-finite"}
        if a.dtype.kind in "biu":
            if np.array_equal(a, b):
                return {"key": name, "max_abs": 0.0, "max_relative": 0.0}
            difference = np.abs(a.astype(np.longdouble) - b.astype(np.longdouble))
            max_abs = float(difference.max()) if difference.size else 0.0
            denominator = np.maximum(np.abs(b.astype(np.longdouble)), 1e-12)
            max_relative = float((difference / denominator).max()) if difference.size else 0.0
            return {"key": name, "kind": "numeric", "max_abs": max_abs,
                    "max_relative": max_relative}
        difference = np.abs(a.astype(np.complex128) - b.astype(np.complex128))
        max_abs = float(difference.max()) if difference.size else 0.0
        denominator = np.maximum(np.abs(b.astype(np.complex128)), 1e-12)
        max_relative = float((difference / denominator).max()) if difference.size else 0.0
        if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False):
            return {"key": name, "kind": "numeric", "max_abs": max_abs,
                    "max_relative": max_relative}
        return {"key": name, "max_abs": max_abs, "max_relative": max_relative}
    if not np.array_equal(a, b):
        return {"key": name, "kind": "exact", "equal": False}
    return {"key": name, "exact": True}


def _compare(args):
    left = _merge(args.left)
    right = _merge(args.right)
    left_keys, right_keys = set(left), set(right)
    if left_keys != right_keys:
        raise AssertionError("artifact keys differ: left-only=%s right-only=%s" % (
            sorted(left_keys - right_keys)[:12], sorted(right_keys - left_keys)[:12],
        ))
    failures = []
    worst_absolute = {"key": None, "value": 0.0}
    worst_relative = {"key": None, "value": 0.0}
    for name in sorted(left):
        result = _compare_array(name, left[name], right[name], args.atol, args.rtol)
        if result.get("kind") and result["kind"] != "numeric":
            failures.append(result)
            continue
        if "max_abs" not in result:
            continue
        if result["max_abs"] > worst_absolute["value"]:
            worst_absolute = {"key": name, "value": result["max_abs"]}
        if result["max_relative"] > worst_relative["value"]:
            worst_relative = {"key": name, "value": result["max_relative"]}
        if result.get("kind") == "numeric":
            failures.append(result)
    result = {
        "left": args.left,
        "right": args.right,
        "atol": args.atol,
        "rtol": args.rtol,
        "tensor_count": len(left),
        "worst_absolute": worst_absolute,
        "worst_relative": worst_relative,
        "failures": failures[:30],
        "failure_count": len(failures),
        "passed": not failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--runtime", choices=("torch", "jittor"), required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument(
        "--model-case",
        choices=tuple(sorted(name for name in _ecosystem_cases.CASES
                             if name.startswith("peft_lora_llama"))),
        default="peft_lora_llama",
    )
    run.add_argument("--steps", type=int, default=10)
    run.add_argument("--shuffle", action="store_true")
    run.add_argument("--num-workers", type=int, default=0)
    run.add_argument("--batch-size", type=int, default=2)
    run.add_argument("--gradient-accumulation-steps", type=int, default=1)
    run.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    run.add_argument("--inject-overflow-step", type=int, default=0)
    run.add_argument("--mask-padding", action="store_true")
    run.add_argument(
        "--loss-normalization",
        choices=("microbatch_mean", "global_tokens"),
        default="microbatch_mean",
    )
    run.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    run.add_argument("--start-step", type=int, default=0)
    run.add_argument("--stop-after", type=int, default=10)
    run.add_argument("--weights", default=None)
    run.add_argument("--weights-out", default=None)
    run.add_argument("--checkpoint", default=None)
    run.add_argument("--save-at", type=int, default=5)
    run.add_argument("--resume", default=None)
    run.add_argument("--output", required=True)
    run.add_argument("--report", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--left", action="append", required=True)
    compare.add_argument("--right", action="append", required=True)
    compare.add_argument("--atol", type=float, default=1e-5)
    compare.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.command == "run":
        if args.num_workers < 0 or args.batch_size < 1:
            raise SystemExit("num-workers must be non-negative and batch-size positive")
        if args.gradient_accumulation_steps < 1:
            raise SystemExit("gradient-accumulation-steps must be positive")
        if args.mixed_precision == "fp16" and args.device != "cuda":
            raise SystemExit("FP16 Accelerate training requires --device cuda")
        if args.inject_overflow_step < 0 or args.inject_overflow_step >= args.steps:
            raise SystemExit("inject-overflow-step must be 0 or inside the training steps")
        if args.inject_overflow_step and args.mixed_precision != "fp16":
            raise SystemExit("overflow injection requires --mixed-precision fp16")
        if args.loss_normalization == "global_tokens" and not args.mask_padding:
            raise SystemExit("global-token normalization requires --mask-padding")
        if args.loss_normalization == "global_tokens" and args.shuffle:
            raise SystemExit("global-token normalization currently requires --no-shuffle")
        if args.device == "cuda" and args.runtime == "torch":
            import torch
            if not torch.cuda.is_available():
                raise SystemExit("CUDA was requested but is unavailable in the PyTorch oracle")
        if not (0 <= args.start_step < args.stop_after <= args.steps):
            raise SystemExit("require 0 <= start-step < stop-after <= steps")
        if args.resume and args.weights:
            raise SystemExit("resume loads weights from the checkpoint; omit --weights")
        if args.checkpoint and not (args.start_step < args.save_at <= args.stop_after):
            raise SystemExit("checkpoint save-at must be after start-step and <= stop-after")
        _run(args)
    else:
        _compare(args)


if __name__ == "__main__":
    main()
