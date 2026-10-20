"""Real CIFAR classification through the public Torch training API.

Run each runtime in a fresh process. ``run`` writes small per-update summaries
and one complete final state; ``compare`` joins split-process summaries and
uses the last segment's final state. Augmentation is explicitly keyed by
seed/epoch/sample, so this protocol does not claim arbitrary worker RNG replay.
No framework is imported until ``run`` starts.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _training_trajectory_runner import (
    _array, _assert_device, _import_runtime, _load_initial_weights,
    _snapshot, _write_initial_weights,
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _batches(sample_count, batch_size):
    return [list(range(start, min(start + batch_size, sample_count)))
            for start in range(0, sample_count, batch_size)]


def _windows(sample_count, batch_size, accumulation):
    batches = _batches(sample_count, batch_size)
    return [batches[start:start + accumulation]
            for start in range(0, len(batches), accumulation)]


def _cursor_after(epoch, batch_end, batch_count):
    return {"epoch": epoch + int(batch_end == batch_count),
            "batch": 0 if batch_end == batch_count else batch_end}


def _augmentation(image, seed, epoch, sample_id, training):
    image = np.asarray(image, dtype=np.uint8).reshape(3, 32, 32)
    if training:
        generator = np.random.RandomState(
            (seed * 1000003 + epoch * 10007 + sample_id * 97) % (2 ** 32))
        padded = np.pad(image, ((0, 0), (4, 4), (4, 4)), mode="reflect")
        row, column = generator.randint(0, 9, size=2)
        image = padded[:, row:row + 32, column:column + 32]
        if generator.randint(0, 2):
            image = image[:, :, ::-1]
    value = image.astype(np.float32) / np.float32(255.0)
    mean = np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32)[:, None, None]
    std = np.asarray([0.2470, 0.2435, 0.2616], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray((value - mean) / std)


def _load_cifar(directory, train_count, eval_count):
    directory = Path(directory)
    images, labels, hashes = [], [], {}
    for name in ("data_batch_1", "data_batch_2", "data_batch_3",
                 "data_batch_4", "data_batch_5", "test_batch"):
        path = directory / name
        hashes[name] = _sha256(path)
        with path.open("rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        values = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
        targets = np.asarray(batch[b"labels"], dtype=np.int64)
        if name == "test_batch":
            evaluation = (values[:eval_count].copy(), targets[:eval_count].copy())
        else:
            images.append(values)
            labels.append(targets)
    training = (np.concatenate(images)[:train_count].copy(),
                np.concatenate(labels)[:train_count].copy())
    if len(training[0]) != train_count or len(evaluation[0]) != eval_count:
        raise ValueError("requested CIFAR subset exceeds available examples")
    digest = hashlib.sha256()
    for value in training + evaluation:
        digest.update(value.tobytes())
    return training, evaluation, {"files": hashes, "subset_sha256": digest.hexdigest()}


def _model(torch, args):
    nn = torch.nn
    if args.model == "vit":
        from transformers import ViTConfig, ViTForImageClassification

        return ViTForImageClassification(ViTConfig(
            image_size=32, patch_size=8, num_channels=3, num_labels=10,
            hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
            intermediate_size=64, hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
        ))

    class Block(nn.Module):
        def __init__(self, channels_in, channels_out, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(channels_in, channels_out, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels_out)
            self.conv2 = nn.Conv2d(channels_out, channels_out, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels_out)
            self.downsample = (nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 1, stride, bias=False),
                nn.BatchNorm2d(channels_out),
            ) if stride != 1 or channels_in != channels_out else None)

        def forward(self, inputs):
            outputs = torch.nn.functional.relu(self.bn1(self.conv1(inputs)))
            outputs = self.bn2(self.conv2(outputs))
            residual = inputs if self.downsample is None else self.downsample(inputs)
            return torch.nn.functional.relu(outputs + residual)

    class CIFARResNet18(nn.Module):
        def __init__(self):
            super().__init__()
            width = args.width
            self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(width)
            self.layer1 = nn.Sequential(Block(width, width), Block(width, width))
            self.layer2 = nn.Sequential(Block(width, width * 2, 2),
                                        Block(width * 2, width * 2))
            self.layer3 = nn.Sequential(Block(width * 2, width * 4, 2),
                                        Block(width * 4, width * 4))
            self.layer4 = nn.Sequential(Block(width * 4, width * 8, 2),
                                        Block(width * 8, width * 8))
            self.dropout = nn.Dropout(args.dropout)
            self.fc = nn.Linear(width * 8, 10)

        def forward(self, inputs):
            outputs = torch.nn.functional.relu(self.bn1(self.conv1(inputs)))
            for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
                outputs = layer(outputs)
            outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, 1).flatten(1)
            return self.fc(self.dropout(outputs))

    return CIFARResNet18()


def _logits(model, inputs, model_case):
    return model(pixel_values=inputs).logits if model_case == "vit" else model(inputs)


def _trainable(model):
    return [name for name, value in model.named_parameters() if bool(value.requires_grad)]


def _hash_parameters(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode("utf8"))
        digest.update(_array(value).tobytes())
    return digest.hexdigest()


def _flatten(prefix, value, arrays):
    if isinstance(value, dict):
        for key, item in sorted(value.items(), key=lambda entry: str(entry[0])):
            _flatten(prefix + "/" + str(key), item, arrays)
    elif hasattr(value, "detach"):
        arrays[prefix] = _array(value)
    elif isinstance(value, (list, tuple)):
        if not value:
            arrays[prefix] = np.asarray([], dtype=np.float64)
        else:
            for index, item in enumerate(value):
                _flatten(prefix + "/" + str(index), item, arrays)
    elif value is None:
        arrays[prefix] = np.asarray("None")
    else:
        arrays[prefix] = np.asarray(value)


def _configuration(args, data_info):
    names = ("model", "width", "dropout", "seed", "train_samples", "eval_samples",
             "epochs", "batch_size", "gradient_accumulation_steps", "precision",
             "learning_rate", "clip_norm", "num_workers")
    return {"protocol_version": 1, "data": data_info,
            **{name: getattr(args, name) for name in names}}


def _validate_checkpoint(checkpoint, configuration, batch_count, accumulation):
    if checkpoint["configuration"] != configuration:
        raise ValueError("checkpoint configuration or CIFAR fixture differs")
    cursor = checkpoint["cursor"]
    epoch, batch = cursor["epoch"], cursor["batch"]
    if not (0 <= epoch <= configuration["epochs"] and 0 <= batch < batch_count):
        raise ValueError("checkpoint cursor is outside the training fixture")
    if batch % accumulation or (epoch == configuration["epochs"] and batch):
        raise ValueError("checkpoint cursor is inside an accumulation window")
    expected = epoch * math.ceil(batch_count / accumulation) + batch // accumulation
    if checkpoint["completed_updates"] != expected:
        raise ValueError("checkpoint update count disagrees with data cursor")


def _checkpoint(torch, model, optimizer, scheduler, scaler, configuration,
                cursor, update, initial_hash, path, device):
    state = {"configuration": configuration, "cursor": cursor,
             "completed_updates": update, "initial_parameter_sha256": initial_hash,
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
             "torch_rng": torch.get_rng_state()}
    if device == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / "training-state.tmp"
    torch.save(state, temporary)
    temporary.replace(path / "training-state.pt")


def _evaluate(torch, model, args, evaluation, device, trainable):
    model.eval()
    if _trainable(model) != trainable:
        raise AssertionError("eval changed trainable parameter ownership")
    all_logits, losses = [], []
    with torch.no_grad():
        for batch in _batches(len(evaluation[0]), args.batch_size):
            inputs = torch.from_numpy(np.stack([
                _augmentation(evaluation[0][index], args.seed, 0, index, False)
                for index in batch])).to(device)
            targets = torch.from_numpy(evaluation[1][batch].copy()).to(device)
            logits = _logits(model, inputs, args.model)
            _assert_device(logits, args.device, "evaluation logits")
            values = _array(logits.float())
            loss = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="sum")
            all_logits.append(values)
            losses.append(float(_array(loss).reshape(-1)[0]))
    model.train()
    if _trainable(model) != trainable:
        raise AssertionError("train after eval changed trainable parameters")
    logits = np.concatenate(all_logits)
    return {"logits": logits, "loss": np.asarray(sum(losses) / len(logits)),
            "accuracy": np.asarray(np.mean(logits.argmax(1) == evaluation[1]))}


def _run(args):
    jt, torch = _import_runtime(args.runtime)
    stack = ExitStack()
    if jt is not None:
        stack.enter_context(jt.runtime.scope(use_cuda=int(args.device == "cuda"),
                                            backend_fallback="error"))
        from jittor._runtime.fallback import forbid_backend_fallbacks

        stack.enter_context(forbid_backend_fallbacks())
    with stack:
        _run_active(args, jt, torch)


def _run_active(args, jt, torch):
    fallback_before = jt.core.backend_fallback_count() if jt is not None else 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    training, evaluation, data_info = _load_cifar(
        args.cifar_dir, args.train_samples, args.eval_samples)
    configuration = _configuration(args, data_info)
    model = _model(torch, args).to(device)
    model.train()
    trainable = _trainable(model)
    _assert_device(next(model.parameters()), args.device, "model parameter")
    if args.weights:
        _load_initial_weights(torch, model, args.weights)
    elif args.weights_out:
        Path(args.weights_out).parent.mkdir(parents=True, exist_ok=True)
        _write_initial_weights(model, args.weights_out)
    elif not args.resume:
        raise ValueError("new run requires --weights-out or --weights")
    initial_hash = _hash_parameters(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=0.9, weight_decay=5e-4, foreach=False)
    batch_count = math.ceil(args.train_samples / args.batch_size)
    updates_per_epoch = math.ceil(batch_count / args.gradient_accumulation_steps)
    total_updates = args.epochs * updates_per_epoch
    stop_after = total_updates if args.stop_after is None else args.stop_after
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_updates)
    scaler = torch.amp.GradScaler(args.device, enabled=args.precision == "fp16")
    cursor, update = {"epoch": 0, "batch": 0}, 0
    if args.resume:
        checkpoint = torch.load(Path(args.resume) / "training-state.pt", weights_only=False)
        _validate_checkpoint(checkpoint, configuration, batch_count,
                             args.gradient_accumulation_steps)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        cursor, update = checkpoint["cursor"], checkpoint["completed_updates"]
        initial_hash = checkpoint["initial_parameter_sha256"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if not update < stop_after <= total_updates:
        raise ValueError("stop-after must advance checkpoint and not exceed total updates")
    if args.checkpoint and not update < args.save_at <= stop_after:
        raise ValueError("save-at must be after the resumed cursor and <= stop-after")
    arrays = {"configuration_json": np.asarray(json.dumps(configuration, sort_keys=True))}
    start_update = update
    observed_dtypes, train_forwards = set(), 0

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, epoch):
            self.epoch = epoch
            self.order = np.random.RandomState(args.seed + epoch).permutation(args.train_samples)

        def __len__(self):
            return args.train_samples

        def __getitem__(self, index):
            sample_id = int(self.order[index])
            image = _augmentation(training[0][sample_id], args.seed, self.epoch,
                                  sample_id, True)
            return image, int(training[1][sample_id]), sample_id

    while cursor["epoch"] < args.epochs and update < stop_after:
        epoch = cursor["epoch"]
        loader_generator = torch.Generator().manual_seed(args.seed + epoch + 100000)
        loader = torch.utils.data.DataLoader(
            Dataset(epoch), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, generator=loader_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        sample_ids, input_digest, loss_sum = [], hashlib.sha256(), 0.0
        microbatch_sizes, logits_dtype = [], None
        for batch_index, (inputs, targets, ids) in enumerate(loader):
            if batch_index < cursor["batch"]:
                continue
            window_start = (batch_index // args.gradient_accumulation_steps
                            * args.gradient_accumulation_steps)
            window_end = min(window_start + args.gradient_accumulation_steps, batch_count)
            window_samples = min(window_end * args.batch_size, args.train_samples) - (
                window_start * args.batch_size)
            input_digest.update(_array(inputs).tobytes())
            input_digest.update(_array(targets).tobytes())
            inputs, targets = inputs.to(device), targets.to(device)
            _assert_device(inputs, args.device, "training images")
            _assert_device(targets, args.device, "training targets")
            dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
            with torch.autocast(args.device, dtype=dtype, enabled=args.precision != "no"):
                logits = _logits(model, inputs, args.model)
                loss = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="sum")
            _assert_device(logits, args.device, "training logits")
            observed_dtypes.add(str(logits.dtype))
            logits_dtype = str(logits.dtype)
            value = float(_array(loss).reshape(-1)[0])
            if not np.isfinite(value):
                raise AssertionError("non-finite training loss")
            scaler.scale(loss / window_samples).backward()
            train_forwards += 1
            loss_sum += value
            sample_ids.extend(_array(ids).astype(np.int64).tolist())
            microbatch_sizes.append(len(inputs))
            if batch_index + 1 != window_end:
                continue
            scaler.unscale_(optimizer)
            missing = [name for name, parameter in model.named_parameters()
                       if name in trainable and parameter.grad is None]
            if missing:
                raise AssertionError("missing trainable gradients: %s" % missing[:12])
            finite = all(np.isfinite(_array(parameter.grad)).all()
                         for _, parameter in model.named_parameters()
                         if parameter.requires_grad)
            if not finite:
                raise AssertionError("non-finite gradients in ordinary training")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm,
                                                       error_if_nonfinite=True)
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            if args.precision == "fp16" and float(scaler.get_scale()) < scale_before:
                raise AssertionError("finite FP16 training unexpectedly skipped an update")
            scheduler.step()
            update += 1
            prefix = "history/step_%05d/" % update
            arrays.update({prefix + "loss": np.asarray(loss_sum / window_samples),
                           prefix + "lr": np.asarray(scheduler.get_last_lr()),
                           prefix + "sample_ids": np.asarray(sample_ids, dtype=np.int64),
                           prefix + "input_sha256": np.asarray(input_digest.hexdigest()),
                           prefix + "microbatch_sizes": np.asarray(microbatch_sizes, dtype=np.int64),
                           prefix + "epoch": np.asarray(epoch, dtype=np.int64),
                           prefix + "gradient_norm": np.asarray(float(_array(grad_norm).reshape(-1)[0])),
                           prefix + "finite_gradient_count": np.asarray(len(trainable), dtype=np.int64),
                           prefix + "logits_dtype": np.asarray(logits_dtype),
                           prefix + "scaler_scale": np.asarray(float(scaler.get_scale()))})
            print(json.dumps({"event": "training_update", "step": update,
                              "epoch": epoch, "loss": loss_sum / window_samples,
                              "device": str(logits.device), "precision": args.precision}),
                  flush=True)
            optimizer.zero_grad(set_to_none=True)
            cursor = _cursor_after(epoch, batch_index + 1, batch_count)
            if cursor["batch"] == 0:
                for name, value in _evaluate(torch, model, args, evaluation, device, trainable).items():
                    arrays["eval/epoch_%02d/%s" % (epoch, name)] = value
            if args.checkpoint and update == args.save_at:
                _checkpoint(torch, model, optimizer, scheduler, scaler, configuration,
                            cursor, update, initial_hash, args.checkpoint, args.device)
            sample_ids, input_digest, loss_sum = [], hashlib.sha256(), 0.0
            microbatch_sizes = []
            if update >= stop_after or cursor["epoch"] != epoch:
                break
    if update != stop_after:
        raise AssertionError("training stopped before the requested update")
    expected_forwards = cursor["epoch"] * batch_count + cursor["batch"]
    for name, buffer in model.named_buffers():
        if name.endswith("num_batches_tracked"):
            actual_forwards = int(_array(buffer).reshape(-1)[0])
            if actual_forwards != expected_forwards:
                raise AssertionError("BatchNorm counter differs from consumed microbatches: "
                                     "%s actual=%d expected=%d" % (
                                         name, actual_forwards, expected_forwards))
    if args.precision != "no":
        expected_dtype = str(torch.bfloat16 if args.precision == "bf16" else torch.float16)
        if observed_dtypes != {expected_dtype}:
            raise AssertionError("AMP did not execute the classifier in the requested dtype")
        if any(str(parameter.dtype) != str(torch.float32) for parameter in model.parameters()):
            raise AssertionError("AMP did not retain float32 master parameters")
    final_eval = _evaluate(torch, model, args, evaluation, device, trainable)
    final_state = {}
    _snapshot(model, optimizer, 0, final_state, include_gradients=False)
    for name, value in final_state.items():
        arrays[name.replace("state/step_00/", "final/", 1)] = value
    _flatten("final/scheduler", scheduler.state_dict(), arrays)
    _flatten("final/scaler", scaler.state_dict(), arrays)
    arrays["completed_updates"] = np.asarray(update, dtype=np.int64)
    for name, value in final_eval.items():
        arrays["final/eval/" + name] = value
    final_hash = _hash_parameters(model)
    if final_hash == initial_hash:
        raise AssertionError("training did not update parameters")
    reload_path = Path(args.output).with_suffix(".model.pt")
    reload_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), reload_path)
    reloaded = _model(torch, args).to(device)
    reloaded.load_state_dict(torch.load(reload_path, weights_only=False), strict=True)
    reload_eval = _evaluate(torch, reloaded, args, evaluation, device, _trainable(reloaded))
    np.testing.assert_array_equal(final_eval["logits"], reload_eval["logits"])
    fallback_count = jt.core.backend_fallback_count() - fallback_before if jt is not None else 0
    if fallback_count:
        raise AssertionError("nonzero backend fallback count")
    np.savez(args.output, **arrays)
    report = {"runtime": args.runtime, "device": args.device,
              "actual_parameter_device": str(next(model.parameters()).device),
              "fallback_count": fallback_count, "configuration": configuration,
              "start_update": start_update, "completed_updates": update,
              "updates_per_epoch": updates_per_epoch, "microbatches_per_epoch": batch_count,
              "train_forwards_this_process": train_forwards, "cursor": cursor,
              "expected_total_train_forwards": expected_forwards,
              "trainable_count": len(trainable), "observed_logits_dtypes": sorted(observed_dtypes),
              "master_parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
              "scaler": scaler.state_dict(), "reload_logits_exact": True,
              "initial_parameter_sha256": initial_hash, "final_parameter_sha256": final_hash,
              "eval_loss": float(final_eval["loss"]), "eval_accuracy": float(final_eval["accuracy"]),
              "output": args.output, "checkpoint": args.checkpoint,
              "coverage_boundary": "explicit seed/epoch/sample augmentation; no default worker RNG claim"}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


def _merge(paths):
    merged, final_update, final_values = {}, -1, {}
    for path in paths:
        with np.load(path, allow_pickle=False) as artifact:
            update = int(artifact["completed_updates"])
            candidate = {key: artifact[key].copy() for key in artifact.files
                         if key.startswith("final/")}
            if update > final_update:
                final_update, final_values = update, candidate
            elif update == final_update:
                if set(candidate) != set(final_values) or any(
                        not np.array_equal(candidate[key], final_values[key]) for key in candidate):
                    raise AssertionError("conflicting final states at the same update")
            for key in artifact.files:
                if key.startswith("final/") or key == "completed_updates":
                    continue
                value = artifact[key]
                if key in merged and not np.array_equal(merged[key], value):
                    raise AssertionError("conflicting segment value: " + key)
                merged[key] = value.copy()
    steps = sorted({int(key.split("/")[1][5:]) for key in merged if key.startswith("history/")})
    if steps != list(range(1, final_update + 1)):
        raise AssertionError("split artifacts must cover every update without gaps")
    merged.update(final_values)
    merged["completed_updates"] = np.asarray(final_update, dtype=np.int64)
    return merged


def _difference(name, actual, reference, args):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        return {"key": name, "kind": "shape_or_dtype"}
    if actual.dtype.kind not in "fc":
        return None if np.array_equal(actual, reference) else {"key": name, "kind": "exact"}
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return {"key": name, "kind": "nonfinite"}
    max_abs = float(np.abs(actual.astype(np.float64) - reference.astype(np.float64)).max(initial=0))
    scale = float(np.abs(reference.astype(np.float64)).max(initial=0))
    if name.endswith("accuracy"):
        threshold = args.accuracy_atol
    elif "/param/" in name or "/optim/" in name:
        threshold = max(args.parameter_atol, args.parameter_rtol * scale)
    elif "/buffer/" in name or name.endswith("logits"):
        threshold = max(args.buffer_atol, args.buffer_rtol * scale)
    elif name.endswith("lr") or "/scheduler/" in name or "/scaler" in name:
        threshold = 0.0
    else:
        threshold = max(args.loss_atol, args.loss_rtol * scale)
    if max_abs > threshold:
        return {"key": name, "kind": "numeric", "max_abs": max_abs,
                "scale": scale, "threshold": threshold}
    return None


def _compare(args):
    left, right = _merge(args.left), _merge(args.right)
    failures = [{"kind": "keys", "left_only": sorted(set(left) - set(right)),
                 "right_only": sorted(set(right) - set(left))}] if set(left) != set(right) else []
    for name in sorted(set(left) & set(right)):
        difference = _difference(name, left[name], right[name], args)
        if difference:
            failures.append(difference)
    report = {"passed": not failures, "failure_count": len(failures),
              "failures": failures[:40], "tensor_count": len(left),
              "completed_updates": int(left["completed_updates"])}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--runtime", choices=("torch", "jittor"), required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--model", choices=("cnn", "vit"), default="cnn")
    run.add_argument("--width", type=int, default=64)
    run.add_argument("--dropout", type=float, default=0.0)
    run.add_argument("--seed", type=int, default=20261020)
    run.add_argument("--cifar-dir", required=True)
    run.add_argument("--train-samples", type=int, default=3137)
    run.add_argument("--eval-samples", type=int, default=65)
    run.add_argument("--epochs", type=int, default=2)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--gradient-accumulation-steps", type=int, default=2)
    run.add_argument("--precision", choices=("no", "fp16", "bf16"), default="no")
    run.add_argument("--learning-rate", type=float, default=0.001)
    run.add_argument("--clip-norm", type=float, default=1.0)
    run.add_argument("--num-workers", type=int, default=0)
    run.add_argument("--weights-out")
    run.add_argument("--weights")
    run.add_argument("--checkpoint")
    run.add_argument("--save-at", type=int, default=25)
    run.add_argument("--resume")
    run.add_argument("--stop-after", type=int)
    run.add_argument("--output", required=True)
    run.add_argument("--report", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--left", action="append", required=True)
    compare.add_argument("--right", action="append", required=True)
    compare.add_argument("--loss-rtol", type=float, default=0.004)
    compare.add_argument("--loss-atol", type=float, default=2e-5)
    compare.add_argument("--parameter-rtol", type=float, default=0.003)
    compare.add_argument("--parameter-atol", type=float, default=2e-5)
    compare.add_argument("--buffer-rtol", type=float, default=0.004)
    compare.add_argument("--buffer-atol", type=float, default=5e-5)
    compare.add_argument("--accuracy-atol", type=float, default=0.0)
    compare.add_argument("--report")
    args = parser.parse_args()
    if args.command == "compare":
        _compare(args)
        return
    for name in ("width", "train_samples", "eval_samples", "epochs", "batch_size",
                 "gradient_accumulation_steps"):
        if getattr(args, name) < 1:
            parser.error(name + " must be positive")
    if args.num_workers < 0 or not 0 <= args.dropout < 1:
        parser.error("invalid num-workers or dropout")
    if args.precision != "no" and args.device != "cuda":
        parser.error("this workflow's AMP acceptance requires CUDA")
    if args.resume and (args.weights or args.weights_out):
        parser.error("resume uses checkpoint weights; omit initial weight options")
    if args.weights and args.weights_out:
        parser.error("choose --weights or --weights-out")
    _run(args)


if __name__ == "__main__":
    main()
