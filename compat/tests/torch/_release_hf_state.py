"""Portable, bounded state artifacts for public Trainer workflow acceptance."""

import hashlib
import json
from pathlib import Path

import numpy as np

from _release_hf_data import array


def flatten(prefix, value):
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from flatten(prefix + "/" + str(key), value[key])
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from flatten(prefix + "/%04d" % index, item)
    elif value is None:
        yield prefix, np.asarray("<none>")
    elif callable(value):
        yield prefix, np.asarray("<callable>")
    elif hasattr(value, "detach"):
        yield prefix + "/@dtype", np.asarray(str(value.dtype))
        yield prefix, array(value)
    elif isinstance(value, (str, bytes, bool, int, float, np.generic)):
        yield prefix, np.asarray(value)
    else:
        raise TypeError("unsupported state value %s at %s" % (type(value).__name__, prefix))


def state_items(model, optimizer, scheduler, scaler):
    trainable = dict((name, value) for name, value in model.named_parameters()
                     if bool(value.requires_grad))
    yield "model/trainable_names", np.asarray(sorted(trainable), dtype="U")
    for name, value in sorted(trainable.items()):
        yield from flatten("model/parameter/" + name, value)
    for name, value in model.named_buffers():
        yield from flatten("model/buffer/" + name, value)
    parameter_names = {id(value): name for name, value in model.named_parameters()}
    saved = optimizer.state_dict()
    if len(optimizer.param_groups) != len(saved["param_groups"]):
        raise AssertionError("optimizer state changed parameter group count")
    for group_index, (live_group, saved_group) in enumerate(zip(optimizer.param_groups,
                                                              saved["param_groups"])):
        if len(live_group["params"]) != len(saved_group["params"]):
            raise AssertionError("optimizer state changed parameter group membership")
        for parameter, identifier in zip(live_group["params"], saved_group["params"]):
            name = parameter_names[id(parameter)]
            yield from flatten("optimizer/parameter/" + name, saved["state"].get(identifier, {}))
        metadata = {key: value for key, value in saved_group.items() if key != "params"}
        yield from flatten("optimizer/group/%04d" % group_index, metadata)
    yield from flatten("scheduler", scheduler.state_dict())
    yield from flatten("scaler", scaler.state_dict() if scaler is not None else None)


def signature(items):
    result = {}
    for name, value in items:
        value = np.asarray(value)
        if value.dtype.kind in "biufc" and not np.isfinite(value).all():
            raise AssertionError("nonfinite state tensor at %s" % name)
        digest = hashlib.sha256()
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
        result[name] = {"sha256": digest.hexdigest(), "shape": list(value.shape),
                        "dtype": value.dtype.str}
    return result


def state_signature(model, optimizer, scheduler, scaler):
    return signature(state_items(model, optimizer, scheduler, scaler))


def frozen_signature(model):
    return signature((name, array(value)) for name, value in model.named_parameters()
                     if not bool(value.requires_grad))


def assert_same_signature(expected, actual, description):
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(name for name in set(expected) & set(actual)
                         if expected[name] != actual[name])
        raise AssertionError("%s differ: missing=%s extra=%s changed=%s" %
                             (description, missing[:8], extra[:8], changed[:8]))


def save_state(path, model, optimizer, scheduler, scaler, mode="full"):
    items = state_items(model, optimizer, scheduler, scaler)
    if mode == "hashes":
        items = ((name, np.asarray(json.dumps(value, sort_keys=True)))
                 for name, value in signature(items).items())
    # Only the final state is materialized, never every training update.
    np.savez(path, **dict(items))


def compare_array(name, left, right, atol, rtol):
    if left.shape != right.shape or left.dtype != right.dtype:
        return {"key": name, "kind": "shape_or_dtype"}
    if left.dtype.kind not in "biufc":
        return None if np.array_equal(left, right) else {"key": name, "kind": "exact"}
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return {"key": name, "kind": "nonfinite"}
    if np.allclose(left, right, atol=atol, rtol=rtol):
        return None
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"key": name, "kind": "numeric", "max_abs": float(difference.max())}


def compare_archives(left, right, atol, rtol):
    failures = []
    with np.load(left, allow_pickle=False) as one, np.load(right, allow_pickle=False) as two:
        if set(one.files) != set(two.files):
            failures.append({"kind": "keys", "left_only": sorted(set(one.files) - set(two.files)),
                             "right_only": sorted(set(two.files) - set(one.files))})
        for name in sorted(set(one.files) & set(two.files)):
            failure = compare_array(name, one[name], two[name], atol, rtol)
            if failure:
                failures.append(failure)
        return {"tensor_count": len(one.files), "failure_count": len(failures),
                "failures": failures[:30], "passed": not failures,
                "atol": atol, "rtol": rtol}


_PROTOCOL_FIELDS = (
    "runtime", "task", "device", "seed", "versions", "initialization",
    "initial_model_dir", "initial_adapter_dir", "mixed_precision",
    "train_file_sha256", "eval_file_sha256", "train_samples", "eval_samples",
    "epochs", "batch_size", "gradient_accumulation_steps", "state_format",
    "fixed_order", "sampler", "num_workers", "gradient_checkpointing",
    "microbatches_per_epoch", "updates_per_epoch", "expected_updates",
    "gradient_norm_algorithm", "optimizer_gradient_phase", "max_grad_norm",
)


def _validate_run_report(report):
    # These fields are emitted by every real-data workflow report. A missing
    # observation must never turn into evidence of equal protocols or devices.
    for field in _PROTOCOL_FIELDS + (
            "fallback_count", "actual_parameter_devices", "start_step",
            "global_step", "resume_checkpoint", "resume_loaded_state_exact"):
        if field not in report:
            raise AssertionError("workflow report is missing %s" % field)
    if type(report["fallback_count"]) is not int or report["fallback_count"] != 0:
        raise AssertionError("workflow report has nonzero or invalid fallback_count")
    device = report["device"]
    devices = report["actual_parameter_devices"]
    if device not in ("cpu", "cuda") or not isinstance(devices, list) or not devices:
        raise AssertionError("workflow report has invalid actual_parameter_devices")
    for actual in devices:
        if not isinstance(actual, str) or not (
                actual == device or (actual.startswith(device + ":") and
                                     actual[len(device) + 1:].isdigit())):
            raise AssertionError("workflow actual_parameter_devices differ from requested device")
    if not isinstance(report["gradient_norm_algorithm"], str) or not report["gradient_norm_algorithm"]:
        raise AssertionError("workflow gradient_norm_algorithm is invalid")
    if report["optimizer_gradient_phase"] not in ("postclip_unscaled", "pre_optimizer_unscaled"):
        raise AssertionError("workflow optimizer_gradient_phase is invalid")
    if report["mixed_precision"] == "fp16" and report["optimizer_gradient_phase"] != "postclip_unscaled":
        raise AssertionError("workflow optimizer_gradient_phase requires FP16 clipping/unscale")
    for health in report["unscaled_gradients"]:
        if health.get("l2_norm_algorithm") != report["gradient_norm_algorithm"]:
            raise AssertionError("workflow gradient_norm_algorithm disagrees with an observation")
        if health.get("gradient_phase") != report["optimizer_gradient_phase"]:
            raise AssertionError("workflow optimizer_gradient_phase disagrees with an observation")
    start, end = report["start_step"], report["global_step"]
    if type(start) is not int or type(end) is not int or not 0 <= start <= end:
        raise AssertionError("workflow report has invalid step bounds")
    if start > 0 and (not report["resume_checkpoint"] or
                      report["resume_loaded_state_exact"] is not True):
        raise AssertionError("workflow resume_loaded_state_exact was not established")
    if start == 0 and report["resume_checkpoint"] is not None:
        raise AssertionError("workflow resume checkpoint has zero start_step")


def _assert_protocol_matches(expected, actual):
    for field in _PROTOCOL_FIELDS:
        if expected[field] != actual[field]:
            raise AssertionError("workflow contract differs at %s" % field)


def compare_runs(args):
    left_dir, right_dir = Path(args.left), Path(args.right)
    left = json.loads((left_dir / "report.json").read_text())
    right = json.loads((right_dir / "report.json").read_text())
    _validate_run_report(left)
    _validate_run_report(right)
    _assert_protocol_matches(left, right)
    if left["global_step"] != right["global_step"]:
        raise AssertionError("workflow contract differs at global_step")
    first = None
    if args.first:
        first = json.loads((Path(args.first) / "report.json").read_text())
        _validate_run_report(first)
        _assert_protocol_matches(left, first)
        if (left["start_step"] != first["start_step"] or
                not left["start_step"] <= right["start_step"] <= first["global_step"]):
            raise AssertionError("workflow first run does not cover the continuous resume prefix")
        if (any(not left["start_step"] < item["step"] <= first["global_step"]
                for name in ("microbatches", "updates", "unscaled_gradients")
                for item in first[name])):
            raise AssertionError("workflow first run contains events outside its segment")
    result = compare_archives(left_dir / "final-state.npz", right_dir / "final-state.npz",
                              args.atol, args.rtol)
    result["left"], result["right"] = str(left_dir), str(right_dir)
    result["protocol_exact"] = True
    if first is not None:
        prefix_step = right["start_step"]
        combined_batches = [item for item in first["microbatches"] if item["step"] <= prefix_step] + right["microbatches"]
        combined_steps = [item for item in first["updates"] if item["step"] <= prefix_step] + right["updates"]
        # Wall clock and logging accumulators are intentionally absent.
        expected_batches = [(item["step"], item["sample_ids"]) for item in left["microbatches"]]
        observed_batches = [(item["step"], item["sample_ids"]) for item in combined_batches]
        result["sample_cursor_exact"] = expected_batches == observed_batches
        result["update_count_exact"] = [item["step"] for item in left["updates"]] == [
            item["step"] for item in combined_steps]
        update_failures = []
        for expected, actual in zip(left["updates"], combined_steps):
            for field in ("learning_rates", "microbatches", "sample_count", "optimizer_step_skipped",
                          "optimizer_update_performed", "optimizer_updates_completed",
                          "optimizer_skips_completed", "precision_protocol", "scaler"):
                if expected.get(field) != actual.get(field):
                    update_failures.append({"step": expected["step"], "field": field})
            if not np.isclose(expected["mean_microbatch_loss"], actual["mean_microbatch_loss"],
                              atol=args.atol, rtol=args.rtol):
                update_failures.append({"step": expected["step"], "field": "mean_microbatch_loss"})
        for expected, actual in zip(left["microbatches"], combined_batches):
            if not np.isclose(expected["loss"], actual["loss"], atol=args.atol, rtol=args.rtol):
                update_failures.append({"step": expected["step"], "field": "microbatch_loss"})
        combined_gradients = [item for item in first["unscaled_gradients"] if item["step"] <= prefix_step] + right["unscaled_gradients"]
        if len(left["unscaled_gradients"]) != len(combined_gradients):
            update_failures.append({"field": "unscaled_gradient_count"})
        for expected, actual in zip(left["unscaled_gradients"], combined_gradients):
            norms = expected["l2_norm"], actual["l2_norm"]
            norm_equal = (norms[0] is norms[1] if any(value is None for value in norms) else
                          bool(np.isclose(*norms, atol=args.atol, rtol=args.rtol)))
            if (expected["step"] != actual["step"] or not norm_equal or
                    expected.get("nonfinite_count", 0) != actual.get("nonfinite_count", 0)):
                update_failures.append({"step": expected["step"], "field": "unscaled_gradient_norm"})
        result["update_contract_failures"] = update_failures[:20]
        result["passed"] = bool(result["passed"] and result["sample_cursor_exact"]
                                and result["update_count_exact"] and not update_failures)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(1)
