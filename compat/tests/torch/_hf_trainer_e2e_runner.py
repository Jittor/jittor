"""Run a CPU-only Transformers Trainer + PEFT + Tokenizers lifecycle case.

The runner is invoked in separate real-PyTorch and Jittor-shim processes. The
first process writes the complete initialized model fixture; the second loads
that fixture so that training starts from exactly the same tensors.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from _training_state_helpers import _named_tensors


TRAIN_TEXTS = [
    "bright useful film",
    "slow dull story",
    "warm funny film",
    "cold boring story",
    "kind moving ending",
    "harsh empty ending",
    "excellent joyful acting",
    "awful tedious acting",
]
TRAIN_LABELS = [1, 0, 1, 0, 1, 0, 1, 0]
EVAL_TEXTS = ["bright joyful ending", "cold dull acting", "warm useful story", "harsh boring film"]
EVAL_LABELS = [1, 0, 1, 0]
VOCAB = {
    "[UNK]": 0,
    "[PAD]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
    "bright": 5,
    "useful": 6,
    "film": 7,
    "slow": 8,
    "dull": 9,
    "story": 10,
    "warm": 11,
    "funny": 12,
    "cold": 13,
    "boring": 14,
    "kind": 15,
    "moving": 16,
    "ending": 17,
    "harsh": 18,
    "empty": 19,
    "excellent": 20,
    "joyful": 21,
    "acting": 22,
    "awful": 23,
    "tedious": 24,
}


def _runtime(name):
    if name == "jittor":
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


def _make_base_model(torch, dropout=0.0):
    from transformers import BertConfig, BertForSequenceClassification

    base = BertForSequenceClassification(
        BertConfig(
            vocab_size=len(VOCAB),
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=32,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            num_labels=2,
            pad_token_id=VOCAB["[PAD]"],
        )
    )
    return base


def _attach_peft(torch, base):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
            task_type="SEQ_CLS",
        ),
    )
    return model


def _save_initial(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{name: _array(value) for name, value in _named_tensors(model)})


def _load_initial(torch, model, path):
    with np.load(path, allow_pickle=False) as saved:
        targets = dict(_named_tensors(model))
        if set(saved.files) != set(targets):
            raise AssertionError("initial model keys differ: missing=%s extra=%s" % (
                sorted(set(targets) - set(saved.files))[:8],
                sorted(set(saved.files) - set(targets))[:8],
            ))
        with torch.no_grad():
            for name, target in targets.items():
                target.copy_(torch.from_numpy(saved[name].copy()))


def _tokenizer_and_data(torch):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast

    raw = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=16,
    )

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, include_sample_id=False):
            self.encodings = tokenizer(texts, truncation=True, max_length=16)
            self.labels = labels
            self.include_sample_id = include_sample_id

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {key: value[index] for key, value in self.encodings.items()}
            item["labels"] = self.labels[index]
            if self.include_sample_id:
                item["sample_id"] = index
            return item

    return (
        tokenizer,
        TextDataset(TRAIN_TEXTS, TRAIN_LABELS, include_sample_id=True),
        TextDataset(EVAL_TEXTS, EVAL_LABELS),
        DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
    )


def _run(args):
    jt, torch = _runtime(args.runtime)
    from transformers import Trainer, TrainingArguments, TrainerCallback

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

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this runtime")

    with stack:
        torch.manual_seed(args.seed)
        base_model = _make_base_model(torch)
        if args.base_weights_out:
            _save_initial(base_model, args.base_weights_out)
        model = _attach_peft(torch, base_model)
        if args.weights:
            _load_initial(torch, model, args.weights)
            if not args.base_weights:
                raise ValueError("--base-weights is required when loading --weights")
        elif args.weights_out:
            _save_initial(model, args.weights_out)
            if not args.base_weights_out:
                raise ValueError("--base-weights-out is required when writing --weights-out")
        else:
            raise ValueError("one of --weights or --weights-out is required")

        tokenizer, train_dataset, eval_dataset, collator = _tokenizer_and_data(torch)
        trainable_names = sorted(
            name for name, parameter in model.named_parameters()
            if bool(parameter.requires_grad)
        )
        if not trainable_names:
            raise AssertionError("PEFT model has no trainable parameters")
        class RecordCallbacks(TrainerCallback):
            def __init__(self):
                self.events = []

            def on_step_begin(self, args, state, control, **kwargs):
                self.events.append(["step_begin", int(state.global_step)])

            def on_step_end(self, args, state, control, **kwargs):
                self.events.append(["step_end", int(state.global_step)])

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                self.events.append(["evaluate", int(state.global_step)])

            def on_save(self, args, state, control, **kwargs):
                self.events.append(["save", int(state.global_step)])

        callback = RecordCallbacks()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            use_cpu=args.device == "cpu",
            seed=args.seed,
            data_seed=args.seed,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=4,
            learning_rate=3e-3,
            weight_decay=0.01,
            lr_scheduler_type="constant",
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=2,
            save_strategy="steps",
            save_steps=2,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            logging_strategy="steps",
            logging_steps=1,
            disable_tqdm=True,
            report_to=[],
            dataloader_num_workers=0,
            remove_unused_columns=False,
            save_safetensors=True,
        )

        def compute_metrics(prediction):
            predictions = np.asarray(prediction.predictions).argmax(axis=-1)
            labels = np.asarray(prediction.label_ids)
            return {"accuracy": float(np.mean(predictions == labels))}

        class RecordingTrainer(Trainer):
            def __init__(self, *positional, **keywords):
                self.training_gradients = []
                self.sample_order = []
                super().__init__(*positional, **keywords)

            def _get_train_sampler(self, train_dataset=None):
                # Pin the local fixture's ordering while keeping Trainer's
                # normal data-loader preparation and optimization loop.
                dataset = train_dataset if train_dataset is not None else self.train_dataset
                return torch.utils.data.SequentialSampler(dataset)

            def training_step(self, model, inputs, num_items_in_batch=None):
                sample_ids = inputs.pop("sample_id")
                self.sample_order.extend(int(value) for value in _array(sample_ids).reshape(-1))
                if inputs["input_ids"].device.type != args.device:
                    raise AssertionError("Trainer input_ids were not placed on the requested device")
                loss = super().training_step(model, inputs, num_items_in_batch)
                gradients = {}
                for name, parameter in model.named_parameters():
                    if not bool(parameter.requires_grad):
                        continue
                    grad = getattr(parameter, "grad", None)
                    if grad is None:
                        raise AssertionError(
                            "trainable parameter has no gradient during training: %s" % name
                        )
                    value = _array(grad)
                    if not np.all(np.isfinite(value)):
                        raise AssertionError("non-finite trainable gradient: %s" % name)
                    gradients[name] = value
                self.training_gradients.append(gradients)
                return loss

        trainer = RecordingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[callback],
        )
        first_parameter = next(trainer.model.parameters())
        parameter_device = str(first_parameter.device)
        if first_parameter.device.type != args.device:
            raise AssertionError(
                "model parameters landed on %s, requested %s"
                % (parameter_device, args.device)
            )
        if jt is not None:
            for parameter in trainer.model.parameters():
                backend, _ = jt.core.dispatch_context([parameter])
                if backend != args.device:
                    raise AssertionError(
                        "Jittor parameter backend is %s, requested %s"
                        % (backend, args.device)
                    )
        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        prediction_output = trainer.predict(eval_dataset)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if int(trainer.state.global_step) != 4:
            raise AssertionError("Trainer completed %d steps, expected 4" % trainer.state.global_step)
        if sum(event[0] == "step_end" for event in callback.events) != 4:
            raise AssertionError("Trainer callback did not observe all four optimizer steps")
        if not trainer.state.best_model_checkpoint:
            raise AssertionError("load_best_model_at_end did not select a checkpoint")

        if len(trainer.training_gradients) != 8:
            raise AssertionError(
                "expected 8 microbatch gradient snapshots, got %d"
                % len(trainer.training_gradients)
            )
        expected_sample_order = list(range(len(train_dataset))) * 2
        if trainer.sample_order != expected_sample_order:
            raise AssertionError(
                "Trainer sample order differed from the pinned fixture: %s"
                % trainer.sample_order
            )

        final_dir = output_dir / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        adapter_file = final_dir / "adapter_model.safetensors"
        if not adapter_file.is_file():
            raise AssertionError("Trainer did not save PEFT weights as Safetensors")
        from safetensors import safe_open

        with safe_open(str(adapter_file), framework="np", device="cpu") as handle:
            safe_keys = sorted(handle.keys())
            if not safe_keys:
                raise AssertionError("saved Safetensors adapter has no tensors")
            safetensors_sha256 = hashlib.sha256(adapter_file.read_bytes()).hexdigest()

        # Reload the adapter into a fresh base and compare its predictions to
        # the Trainer's selected best checkpoint in the same runtime.
        reloaded_base = _make_base_model(torch)
        if args.base_weights:
            reload_weights = args.base_weights
        elif args.base_weights_out:
            reload_weights = args.base_weights_out
        else:
            raise ValueError("base weight fixture missing for final adapter reload")
        _load_initial(torch, reloaded_base, reload_weights)
        from peft import PeftModel

        reloaded = PeftModel.from_pretrained(reloaded_base, str(final_dir))
        reloaded.eval()
        tokenized = tokenizer(
            EVAL_TEXTS, padding=True, truncation=True, return_tensors="pt"
        ).to(args.device)
        reloaded.to(args.device)
        with torch.no_grad():
            reload_logits = reloaded(**tokenized).logits
        original_logits = np.asarray(prediction_output.predictions)
        reload_logits_array = _array(reload_logits)
        if not np.all(np.isfinite(original_logits)) or not np.all(np.isfinite(reload_logits_array)):
            raise AssertionError("prediction logits became non-finite")
        if not np.allclose(original_logits, reload_logits_array, atol=2e-5, rtol=2e-4):
            raise AssertionError("fresh Safetensors adapter reload changed eval logits")

        arrays = {
            "predictions": original_logits,
            "labels": np.asarray(prediction_output.label_ids),
            "reloaded_predictions": reload_logits_array,
            "trainable_names": np.asarray(trainable_names, dtype="U"),
            "safe_keys": np.asarray(safe_keys, dtype="U"),
            "callback_events": np.asarray(callback.events, dtype="U"),
            "sample_order": np.asarray(trainer.sample_order, dtype=np.int64),
        }
        for microbatch, gradients in enumerate(trainer.training_gradients):
            if set(gradients) != set(trainable_names):
                raise AssertionError("trainable gradient key set changed during training")
            for name, value in gradients.items():
                arrays["gradient/microbatch_%02d/%s" % (microbatch, name)] = value
        for name, parameter in trainer.model.named_parameters():
            if bool(parameter.requires_grad):
                arrays["trainable/" + name] = _array(parameter)
        artifact = output_dir / "result.npz"
        np.savez(artifact, **arrays)
        fallback_count = (jt.core.backend_fallback_count() - fallback_before
                          if jt is not None else 0)
        if fallback_count:
            raise AssertionError("Jittor backend fallback count was nonzero")

        report = {
            "runtime": args.runtime,
            "device": args.device,
            "seed": args.seed,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "steps": int(trainer.state.global_step),
            "gradient_accumulation_steps": 2,
            "sample_order": trainer.sample_order,
            "sampler": "sequential fixture sampler",
            "trainable_names": trainable_names,
            "eval_metrics": eval_metrics,
            "train_metrics": train_result.metrics,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "callback_events": callback.events,
            "safetensors_keys": safe_keys,
            "safetensors_sha256": safetensors_sha256,
            "fallback_count": fallback_count,
            "elapsed_seconds": elapsed,
            "artifact": str(artifact),
            "output_dir": str(output_dir),
        }
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, sort_keys=True))


def _compare(args):
    with np.load(args.left, allow_pickle=False) as left, np.load(args.right, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            raise AssertionError("artifact keys differ: left-only=%s right-only=%s" % (
                sorted(set(left.files) - set(right.files)),
                sorted(set(right.files) - set(left.files)),
            ))
        failures = []
        worst = {"key": None, "max_abs": 0.0}
        for name in sorted(left.files):
            a, b = np.asarray(left[name]), np.asarray(right[name])
            if a.dtype != b.dtype or a.shape != b.shape:
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
        result = {
            "left": args.left,
            "right": args.right,
            "atol": args.atol,
            "rtol": args.rtol,
            "tensor_count": len(left.files),
            "worst_absolute": worst,
            "failures": failures[:20],
            "failure_count": len(failures),
            "passed": not failures,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        if failures:
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--runtime", choices=("torch", "jittor"), required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--weights")
    run.add_argument("--weights-out")
    run.add_argument("--base-weights")
    run.add_argument("--base-weights-out")
    run.add_argument("--seed", type=int, default=20261013)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--atol", type=float, default=2e-4)
    compare.add_argument("--rtol", type=float, default=2e-3)
    args = parser.parse_args()
    if args.command == "run":
        if bool(args.weights) == bool(args.weights_out):
            raise SystemExit("pass exactly one of --weights or --weights-out")
        if args.weights and (not args.base_weights or args.base_weights_out):
            raise SystemExit("loading --weights requires only --base-weights")
        if args.weights_out and (not args.base_weights_out or args.base_weights):
            raise SystemExit("writing --weights-out requires only --base-weights-out")
        _run(args)
    else:
        _compare(args)


if __name__ == "__main__":
    main()
