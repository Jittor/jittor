"""Observation-only Trainer subclasses and callbacks for release workflows."""

import json
from pathlib import Path

import numpy as np

from _release_hf_data import array
from _release_hf_state import assert_same_signature, state_signature
from _release_hf_precision import validate_optimizer_attempt


# Diagnostic measurement only; clipping and optimizer updates never use this.
# Elementwise additions define every reduction edge without floating atomics.
_GRADIENT_NORM_GROUP = 256
_GRADIENT_NORM_ALGORITHM = "fp64_pairwise_256_v1"


def diagnostic_squared_l2(torch, gradient):
    values = gradient.detach().to(dtype=torch.float64).reshape(-1).square()
    count = int(values.numel())
    if count == 0:
        return 0.0
    while count > 1:
        padding = (-count) % _GRADIENT_NORM_GROUP
        if padding:
            values = torch.cat((values, torch.zeros(padding, dtype=values.dtype,
                                                    device=values.device)))
        values = values.reshape(-1, _GRADIENT_NORM_GROUP)
        width = _GRADIENT_NORM_GROUP
        while width > 1:
            half = width // 2
            values = values[:, :half] + values[:, half:]
            width = half
        values = values.reshape(-1)
        count = int(values.numel())
    return float(values.item())


def gradient_health(torch, model, require_all=True, allow_nonfinite=False):
    missing, nonfinite, dtypes, squared_norm = [], [], set(), 0.0
    for name, parameter in model.named_parameters():
        if not bool(parameter.requires_grad):
            continue
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            missing.append(name)
            continue
        dtypes.add(str(gradient.dtype))
        if not bool(torch.isfinite(gradient).all().item()):
            nonfinite.append(name)
        else:
            squared_norm += diagnostic_squared_l2(torch, gradient)
    if missing and require_all:
        raise AssertionError("trainable parameters lack gradients: %s" % missing[:12])
    if nonfinite and not allow_nonfinite:
        raise AssertionError("trainable gradients are nonfinite: %s" % nonfinite[:12])
    return {"missing_count": len(missing), "nonfinite_count": len(nonfinite),
            "nonfinite_names": nonfinite[:12], "dtypes": sorted(dtypes),
            "l2_norm": None if nonfinite else float(np.sqrt(squared_norm)),
            "l2_norm_algorithm": _GRADIENT_NORM_ALGORITHM}


def observer_classes(torch, trainer_base, callback_base, args):
    class RecordingTrainer(trainer_base):
        def __init__(self, *positional, **keywords):
            self.microbatches = []
            self.updates = []
            self.unscaled_gradients = []
            self.projection_dtypes = set()
            self.observed_autocast = set()
            self.resume_state_exact = None
            self.checkpoint_signatures = []
            self._pending_batches = []
            self._attempt_context = None
            self.successful_optimizer_updates = 0
            self.skipped_optimizer_attempts = 0
            self.initial_scaler_state = None
            super().__init__(*positional, **keywords)

        def _get_train_sampler(self, train_dataset=None):
            if args.fixed_order:
                dataset = self.train_dataset if train_dataset is None else train_dataset
                return torch.utils.data.SequentialSampler(dataset)
            return super()._get_train_sampler(train_dataset)

        def training_step(self, model, inputs, num_items_in_batch=None):
            sample_ids = array(inputs.pop("sample_id")).reshape(-1).tolist()
            if inputs["input_ids"].device.type != args.device:
                raise AssertionError("Trainer input_ids are on the wrong device")
            if inputs["labels"].device.type != args.device:
                raise AssertionError("Trainer labels are on the wrong device")
            labels = array(inputs["labels"])
            record = {
                "step": int(self.state.global_step) + 1, "sample_ids": sample_ids,
                "batch_size": len(sample_ids), "input_shape": list(inputs["input_ids"].shape),
                "label_shape": list(labels.shape), "masked_label_tokens": int((labels == -100).sum()),
                "actual_device": str(inputs["input_ids"].device),
            }
            loss = super().training_step(model, inputs, num_items_in_batch)
            loss_value = float(array(loss))
            if not np.isfinite(loss_value):
                raise AssertionError("nonfinite training loss")
            record["loss"] = loss_value
            scaler = self.accelerator.scaler
            scaled = args.mixed_precision == "fp16" and scaler is not None
            record["gradient_health"] = gradient_health(torch, model, allow_nonfinite=scaled)
            record["gradient_phase"] = "scaled_backward" if scaled else "backward"
            record["gradient_scale"] = float(scaler.get_scale()) if scaled else None
            self.microbatches.append(record)
            self._pending_batches.append(record)
            return loss

    class CaptureCallback(callback_base):
        owner = None

        def on_train_begin(self, callback_args, state, control, **kwargs):
            if args.resume_checkpoint:
                path = Path(args.resume_checkpoint) / "release-state-signature.json"
                if not path.is_file():
                    raise FileNotFoundError("resume checkpoint lacks independent state signature: %s" % path)
                saved = json.loads(path.read_text())
                expected = saved["state"]
                observed = state_signature(self.owner.model, self.owner.optimizer,
                                           self.owner.lr_scheduler, self.owner.accelerator.scaler)
                assert_same_signature(expected, observed, "loaded model/optimizer/scheduler/scaler")
                self.owner.resume_state_exact = True
                self.owner.successful_optimizer_updates = int(
                    saved.get("optimizer_updates_completed", state.global_step))
                self.owner.skipped_optimizer_attempts = int(saved.get("optimizer_skips_completed", 0))
                if (self.owner.successful_optimizer_updates + self.owner.skipped_optimizer_attempts
                        != int(state.global_step)):
                    raise AssertionError("resume checkpoint optimizer attempt/update counters disagree")
            scaler = self.owner.accelerator.scaler
            self.owner.initial_scaler_state = scaler.state_dict() if scaler is not None else None
            return control

        def on_pre_optimizer_step(self, callback_args, state, control, **kwargs):
            owner = self.owner
            fp16 = args.mixed_precision == "fp16"
            health = gradient_health(torch, kwargs["model"], allow_nonfinite=fp16)
            health["step"] = int(state.global_step) + 1
            health["gradient_phase"] = ("postclip_unscaled" if args.max_grad_norm > 0
                                        else "pre_optimizer_unscaled")
            owner.unscaled_gradients.append(health)
            scaled_nonfinite = any(item["gradient_health"]["nonfinite_count"]
                                   for item in owner._pending_batches)
            overflow = scaled_nonfinite or bool(health["nonfinite_count"])
            scaler = owner.accelerator.scaler
            owner._attempt_context = {
                "scaled_nonfinite": scaled_nonfinite, "unscaled_health": health,
                "scaler_before": scaler.state_dict() if scaler is not None else None,
                "overflow_state": (state_signature(owner.model, owner.optimizer,
                                                    owner.lr_scheduler, None) if overflow else None),
            }
            return control

        def on_step_end(self, callback_args, state, control, **kwargs):
            owner = self.owner
            pending = owner._pending_batches
            if not pending:
                raise AssertionError("optimizer step has no observed training microbatch")
            scaler = owner.accelerator.scaler
            context = owner._attempt_context
            if context is None:
                raise AssertionError("optimizer attempt lacks pre-step health observation")
            skipped = bool(owner.accelerator.optimizer_step_was_skipped)
            unchanged = None
            if context["overflow_state"] is not None:
                observed = state_signature(owner.model, owner.optimizer, owner.lr_scheduler, None)
                assert_same_signature(context["overflow_state"], observed,
                                      "overflow skip model/optimizer/scheduler")
                unchanged = True
            protocol = validate_optimizer_attempt(
                args.mixed_precision, context["scaled_nonfinite"], context["unscaled_health"],
                skipped, context["scaler_before"], scaler.state_dict() if scaler is not None else None,
                unchanged)
            owner.successful_optimizer_updates += int(not skipped)
            owner.skipped_optimizer_attempts += int(skipped)
            owner.updates.append({
                "step": int(state.global_step), "epoch": float(state.epoch),
                "microbatches": len(pending), "sample_count": sum(item["batch_size"] for item in pending),
                "mean_microbatch_loss": float(np.mean([item["loss"] for item in pending])),
                "learning_rates": [float(group["lr"]) for group in owner.optimizer.param_groups],
                "optimizer_step_skipped": skipped,
                "optimizer_update_performed": not skipped,
                "optimizer_updates_completed": owner.successful_optimizer_updates,
                "optimizer_skips_completed": owner.skipped_optimizer_attempts,
                "precision_protocol": protocol,
                "scaler": scaler.state_dict() if scaler is not None else None,
            })
            owner._pending_batches = []
            owner._attempt_context = None
            step = int(state.global_step)
            if step in args.save_at or step == args.expected_updates:
                control.should_save = True
                control.should_evaluate = True
            if args.stop_after is not None and step >= args.stop_after:
                control.should_training_stop = True
                control.should_save = True
                control.should_evaluate = True
            return control

        def on_save(self, callback_args, state, control, **kwargs):
            step = int(state.global_step)
            checkpoint = Path(callback_args.output_dir) / ("checkpoint-%d" % step)
            signature = state_signature(self.owner.model, self.owner.optimizer,
                                        self.owner.lr_scheduler, self.owner.accelerator.scaler)
            (checkpoint / "release-state-signature.json").write_text(json.dumps(
                {"step": step, "state": signature,
                 "optimizer_updates_completed": self.owner.successful_optimizer_updates,
                 "optimizer_skips_completed": self.owner.skipped_optimizer_attempts}, indent=2) + "\n")
            self.owner.checkpoint_signatures.append(str(checkpoint))
            return control

    return RecordingTrainer, CaptureCallback


def register_projection_observer(torch, model, owner, device):
    def observe(_module, _inputs, output):
        value = output[0] if isinstance(output, (list, tuple)) else output
        if value.device.type != device:
            raise AssertionError("projection output is on the wrong device")
        owner.projection_dtypes.add(str(value.dtype))
        owner.observed_autocast.add(bool(torch.is_autocast_enabled(device)))

    candidates = [(name, module) for name, module in model.named_modules()
                  if name.rsplit(".", 1)[-1] in ("query", "q_proj", "q")]
    if not candidates:
        raise AssertionError("no attention projection available for real AMP observation")
    return candidates[0][1].register_forward_hook(observe)
