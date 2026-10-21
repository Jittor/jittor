"""Strict, complete HF workflow evidence comparisons without importing a runtime.

Artifact locations are validated inside each run and retained with their SHA256;
only their location strings differ across runs. Numeric budgets come exclusively
from the caller's frozen fixture protocol. Same-runtime gradient replay is exact.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from _release_hf_data import digest_file as _sha
from _release_hf_state import _PROTOCOL_FIELDS, _validate_run_report


_MOMENTS = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer",
            "square_avg", "grad_avg", "exp_avg_diff", "pre_grad"}
_PHASES = ("original", "reloaded", "merged", "merged_reload")
_GENERATION_METADATA = {"original": "generation", "reloaded": "reloaded_generation",
                        "merged": "merged_generation", "merged_reload": "merged_reload_generation"}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _json(path):
    def invalid(value):
        raise AssertionError("nonfinite JSON number: " + value)
    def finite_float(value):
        result = float(value)
        _require(np.isfinite(result), "nonfinite JSON number: " + value)
        return result
    return json.loads(Path(path).read_text(), parse_constant=invalid, parse_float=finite_float)


def _same_json(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_same_json(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    return left == right




def _within(run, path):
    candidate = Path(path)
    candidate = (candidate if candidate.is_absolute() else run / candidate).resolve()
    try:
        candidate.relative_to(run)
    except ValueError as error:
        raise AssertionError("archive must be inside its run: %s" % candidate) from error
    _require(candidate.is_file(), "missing archive: %s" % candidate)
    return candidate


def _archive(path):
    with np.load(path, allow_pickle=False) as values:
        _require(len(values.files) == len(set(values.files)), "duplicate archive keys: %s" % path)
        result = {key: values[key] for key in values.files}
    for key, value in result.items():
        _require(value.dtype.kind != "O", "object array: %s:%s" % (path, key))
        if value.dtype.kind in "biufc":
            _require(np.isfinite(value).all(), "nonfinite archive value: %s:%s" % (path, key))
    return result


def _budget(protocol_path, report):
    protocol_path = Path(protocol_path).resolve()
    protocol = _json(protocol_path)
    precision = {"no": "float32", "bf16": "bfloat16"}.get(report["mixed_precision"])
    _require(precision is not None, "no frozen fixture budget for this mixed_precision")
    source = protocol["numeric_budgets"][precision]
    for key in ("forward_atol", "forward_rtol", "gradient_atol", "gradient_rtol"):
        _require(type(source[key]) in (int, float) and np.isfinite(source[key]) and source[key] >= 0,
                 "invalid fixture numeric budget: " + key)
    return {"precision": precision, "protocol": str(protocol_path), "sha256": _sha(protocol_path),
            "forward": (source["forward_atol"], source["forward_rtol"]),
            "gradient": (source["gradient_atol"], source["gradient_rtol"])}


def _measurement(report):
    if report["runtime"] == "jittor":
        _require(report.get("fallback_measurement") == "jittor_core_attempt_counter" and
                 type(report["fallback_count"]) is int and report["fallback_count"] == 0,
                 "Jittor requires measured zero fallback attempts")
    else:
        expected = ("not_available_in_native_torch_npu" if report["device"] == "npu"
                    else "not_available_in_native_torch")
        _require(report.get("fallback_measurement") == expected and report["fallback_count"] is None,
                 "native fallback_count must be null and explicitly unavailable")


def _read_report(run, full=True):
    report = _json(run / "report.json")
    for key in _PROTOCOL_FIELDS + ("updates", "microbatches", "unscaled_gradients", "gradient_archives",
            "optimizer_updates_completed", "optimizer_skips_completed", "optimizer_attempts_in_process",
            "optimizer_updates_in_process", "expected_optimizer_attempts", "frozen_base_unchanged"):
        _require(key in report, "workflow report is missing " + key)
    _validate_run_report(report)
    _require(report["runtime"] in ("torch", "jittor"), "invalid runtime")
    _measurement(report)
    _require(report["state_format"] == "full", "full numeric state evidence is required")
    _require(report["frozen_base_unchanged"] is True, "frozen base parameters changed")
    start, end, expected = report["start_step"], report["global_step"], report["expected_updates"]
    _require(type(expected) is int and expected > 0 and end <= expected, "invalid expected update count")
    if full:
        _require(start == 0 and end == expected, "comparison needs a full 0 -> expected_updates run")
    steps = list(range(start + 1, end + 1))
    for key in ("updates", "unscaled_gradients", "gradient_archives"):
        _require(isinstance(report[key], list) and [row.get("step") for row in report[key]] == steps,
                 "incomplete or duplicate workflow steps: " + key)
    _require(report["expected_optimizer_attempts"] == expected and
             report["optimizer_attempts_in_process"] == len(steps), "optimizer attempt counter mismatch")
    _require(report["optimizer_updates_completed"] + report["optimizer_skips_completed"] == end,
             "optimizer success/skip counters do not cover attempts")
    batches = report["microbatches"]
    _require(isinstance(batches, list) and batches, "missing microbatch execution evidence")
    batch_steps = [row.get("step") for row in batches]
    _require(batch_steps == sorted(batch_steps) and set(batch_steps) == set(steps), "microbatch step coverage")
    if full:
        for field in ("epochs", "train_samples", "batch_size", "gradient_accumulation_steps",
                      "microbatches_per_epoch", "updates_per_epoch"):
            _require(type(report[field]) is int and report[field] > 0, "invalid training count: " + field)
        microbatches = (report["train_samples"] + report["batch_size"] - 1) // report["batch_size"]
        updates = (microbatches + report["gradient_accumulation_steps"] - 1) // report["gradient_accumulation_steps"]
        _require(report["microbatches_per_epoch"] == microbatches and report["updates_per_epoch"] == updates and
                 expected == updates * report["epochs"] and len(batches) == microbatches * report["epochs"] and
                 sum(row["batch_size"] for row in batches) == report["train_samples"] * report["epochs"],
                 "full training sample/microbatch/update execution contract")
    for row in batches:
        for key in ("sample_ids", "batch_size", "input_shape", "label_shape", "actual_device", "loss",
                    "gradient_health", "gradient_phase", "gradient_scale", "masked_label_tokens"):
            _require(key in row, "microbatch is missing " + key)
        _require(type(row["batch_size"]) is int and row["batch_size"] > 0 and
                 len(row["sample_ids"]) == row["batch_size"] and
                 all(type(item) is int and item >= 0 for item in row["sample_ids"]), "microbatch sample contract")
        for key in ("input_shape", "label_shape"):
            shape = row[key]
            _require(isinstance(shape, list) and shape and shape[0] == row["batch_size"] and
                     all(type(item) is int and item > 0 for item in shape), "microbatch shape contract")
        _require(np.isfinite(row["loss"]), "nonfinite microbatch loss")
        _require(row["actual_device"] in report["actual_parameter_devices"], "microbatch actual device mismatch")
    for health in [row["gradient_health"] for row in batches] + report["unscaled_gradients"]:
        _require(health.get("missing_count") == 0 and health.get("nonfinite_count") == 0 and
                 health.get("nonfinite_names") == [] and np.isfinite(health.get("l2_norm", np.nan)),
                 "missing or nonfinite gradient observation")
    successful = sum(row.get("optimizer_update_performed") is True for row in report["updates"])
    skipped = sum(row.get("optimizer_step_skipped") is True for row in report["updates"])
    _require(successful + skipped == len(steps) and report["optimizer_updates_in_process"] == successful,
             "optimizer update execution counter mismatch")
    success_base = report["optimizer_updates_completed"] - successful
    skip_base = report["optimizer_skips_completed"] - skipped
    for row in report["updates"]:
        window = [batch for batch in batches if batch["step"] == row["step"]]
        _require(row.get("microbatches") == len(window) and row.get("sample_count") ==
                 sum(batch["batch_size"] for batch in window), "optimizer microbatch window mismatch")
        _require(type(row.get("optimizer_update_performed")) is bool and
                 type(row.get("optimizer_step_skipped")) is bool and
                 row["optimizer_update_performed"] != row["optimizer_step_skipped"], "invalid optimizer execution flags")
        success_base += int(row["optimizer_update_performed"])
        skip_base += int(row["optimizer_step_skipped"])
        _require(row.get("optimizer_updates_completed") == success_base and
                 row.get("optimizer_skips_completed") == skip_base, "per-step optimizer counter mismatch")
        _require(np.isfinite(row.get("mean_microbatch_loss", np.nan)), "nonfinite optimizer loss")
        _require(isinstance(row.get("precision_protocol"), dict) and "scaler" in row and
                 isinstance(row.get("learning_rates"), list), "missing optimizer precision/scheduler evidence")
        if report["mixed_precision"] != "fp16":
            _require(not row["optimizer_step_skipped"], "finite workflow skipped an optimizer step")
    return report


def _state(run, report):
    values = _archive(_within(run, "final-state.npz"))
    _require("model/trainable_names" in values, "missing full trainable inventory")
    inventory = values["model/trainable_names"]
    _require(inventory.ndim == 1 and inventory.dtype.kind in "US" and inventory.size > 0,
             "invalid trainable inventory")
    names = inventory.tolist()
    _require(all(isinstance(name, str) and name for name in names) and len(names) == len(set(names)),
             "duplicate/invalid trainable names")
    parameters = {key[len("model/parameter/"):] for key in values
                  if key.startswith("model/parameter/") and not key.endswith("/@dtype")}
    _require(parameters == set(names), "trainable parameter inventory is incomplete")
    for name in names:
        prefix = "model/parameter/" + name
        required = [prefix, prefix + "/@dtype"]
        for field in ("step", "exp_avg", "exp_avg_sq"):
            required.extend(["optimizer/parameter/" + name + "/" + field,
                             "optimizer/parameter/" + name + "/" + field + "/@dtype"])
        _require(all(key in values for key in required), "missing parameter/moment/counter evidence: " + name)
        step = values["optimizer/parameter/" + name + "/step"]
        _require(step.size == 1 and float(step.item()) == report["optimizer_updates_completed"],
                 "optimizer parameter step counter mismatch: " + name)
    optimizer_names = {key[len("optimizer/parameter/"):].rsplit("/", 1)[0]
                       for key in values if key.startswith("optimizer/parameter/") and not key.endswith("/@dtype")}
    _require(optimizer_names == set(names), "optimizer parameter inventory is incomplete or has unknown names")
    groups = {key.split("/")[2] for key in values if key.startswith("optimizer/group/")}
    _require(groups, "missing optimizer group metadata")
    for group in groups:
        for field in ("lr", "betas/0000", "betas/0001", "eps", "weight_decay", "amsgrad", "maximize",
                      "foreach", "capturable", "differentiable", "fused", "decoupled_weight_decay"):
            _require("optimizer/group/" + group + "/" + field in values, "missing optimizer group metadata: " + field)
    _require("scheduler/last_epoch" in values and "scheduler/_step_count" in values and
             any(key.startswith("scaler") for key in values), "missing scheduler/scaler state")
    _require(values["scheduler/last_epoch"].size == 1 and
             int(values["scheduler/last_epoch"].item()) == report["optimizer_updates_completed"],
             "scheduler counter mismatch")
    return values, names


def validate_gradient_archives(run, report, state, names):
    """Return full per-step evidence; callers may concatenate complete segments."""
    run = Path(run).resolve()
    expected = {prefix + name for name in names for prefix in ("gradient/", "dtype/")}
    result = []
    for row in report["gradient_archives"]:
        path = _within(run, row["path"])
        _require(row.get("phase") == report["optimizer_gradient_phase"] and
                 row.get("trainable_gradient_count") == len(names) and row.get("finite") is True and
                 row.get("nonfinite_names") == [], "invalid full gradient manifest")
        window = [batch for batch in report["microbatches"] if batch["step"] == row["step"]]
        _require(row.get("microbatches") == window, "gradient archive window does not match executed microbatches")
        values = _archive(path)
        _require(set(values) == expected, "gradient archive keys must exactly cover every trainable and dtype")
        for name in names:
            gradient, parameter = values["gradient/" + name], state["model/parameter/" + name]
            _require(gradient.dtype == parameter.dtype and gradient.shape == parameter.shape and
                     gradient.dtype.kind in "fc", "gradient dtype/shape mismatch: " + name)
            _require(np.array_equal(values["dtype/" + name], state["model/parameter/" + name + "/@dtype"]),
                     "gradient declared dtype mismatch: " + name)
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            _require(_json(sidecar) == row, "gradient sidecar disagrees with run report")
        result.append({"step": row["step"], "path": str(path), "sha256": _sha(path), "arrays": values})
    return result

def _generation(values, metadata, prefix, report):
    keys = {name: prefix + "/" + name for name in ("input_ids", "attention_mask", "greedy", "sampled")}
    _require(all(key in values for key in keys.values()), "missing complete generation evidence: " + prefix)
    for field in ("input_lengths", "greedy_cache_exact", "sampling", "compute_policy", "actual_device",
                  "max_new_tokens", "vocab_size", "is_encoder_decoder", "decoder_start_token_id"):
        _require(field in metadata, "missing generation metadata: " + field)
    inputs, mask = values[keys["input_ids"]], values[keys["attention_mask"]]
    _require(inputs.ndim == 2 and inputs.shape == mask.shape and inputs.size > 0 and
             inputs.dtype.kind in "iu" and mask.dtype.kind in "biu", "generation input/mask dtype or shape")
    _require(np.isin(mask, [0, 1]).all() and metadata["input_lengths"] == mask.sum(-1).tolist() and
             all(0 < size <= inputs.shape[1] for size in metadata["input_lengths"]) and
             len(set(metadata["input_lengths"])) > 1, "generation variable-length input contract")
    vocab, maximum = metadata["vocab_size"], metadata["max_new_tokens"]
    _require(type(vocab) is int and vocab > 0 and type(maximum) is int and maximum > 0 and
             ((inputs >= 0) & (inputs < vocab)).all(), "generation input token range")
    _require(metadata["greedy_cache_exact"] is True and metadata["compute_policy"] ==
             "FP32 model without autocast" and metadata["actual_device"] in report["actual_parameter_devices"],
             "generation cache/compute/device contract")
    _require(metadata["sampling"] == {"top_k": 10, "top_p": 0.9, "temperature": 0.8} and
             type(metadata["is_encoder_decoder"]) is bool, "generation sampling/model metadata")
    if metadata["is_encoder_decoder"]:
        start = metadata["decoder_start_token_id"]
        _require(type(start) is int and 0 <= start < vocab, "generation decoder start token")
        expected_prefix = np.full((inputs.shape[0], 1), start, dtype=inputs.dtype)
    else:
        expected_prefix = inputs
    width = expected_prefix.shape[1]
    for name in ("greedy", "sampled"):
        value = values[keys[name]]
        _require(value.ndim == 2 and value.dtype.kind in "iu" and value.shape[0] == inputs.shape[0] and
                 width < value.shape[1] <= width + maximum, "generation output shape/dtype: " + name)
        _require(((value >= 0) & (value < vocab)).all(), "generation output token range: " + name)
        _require(np.array_equal(value[:, :width], expected_prefix), "generation prefix changed: " + name)


def _outputs(run, report):
    initial_path, lifecycle_path = (_within(run, name) for name in ("initial-probe.npz", "lifecycle.npz"))
    initial, lifecycle = _archive(initial_path), _archive(lifecycle_path)
    meta = report.get("initial_probe", {})
    _require(set(meta) == {"logits_dtype", "loss_dtype", "device", "loss"} and
             meta["device"] in report["actual_parameter_devices"] and
             isinstance(meta["logits_dtype"], str) and isinstance(meta["loss_dtype"], str),
             "missing initial probe dtype/device evidence")
    _require("loss" in initial and initial["loss"].size == 1 and
             meta["loss"] == float(initial["loss"].item()), "initial probe loss metadata mismatch")
    phases = _PHASES if report["task"] == "qwen-lora" else _PHASES[:2]
    _require(set(initial) == {"logits", "loss"} | {key for key in initial if key.startswith("input/")},
             "unexpected initial probe keys")
    expected_keys = {"logits_sample", "loss"} | {phase + "/" + key for phase in phases for key in initial}
    if report["task"] != "bert":
        generation_keys = ("input_ids", "attention_mask", "greedy", "sampled")
        expected_keys |= {phase + "/generation/" + key for phase in phases for key in generation_keys}
        expected_keys |= {"generation/" + key for key in generation_keys}
    # Check completeness explicitly below before rejecting any extra unvalidated keys.
    for phase in (None,) + phases:
        source = initial if phase is None else lifecycle
        prefix = "" if phase is None else phase + "/"
        for key in ("logits", "loss", "input/input_ids", "input/attention_mask", "input/labels"):
            _require(prefix + key in source, "missing full output: " + prefix + key)
        logits, loss = source[prefix + "logits"], source[prefix + "loss"]
        inputs, labels = source[prefix + "input/input_ids"], source[prefix + "input/labels"]
        mask = source[prefix + "input/attention_mask"]
        _require(inputs.ndim == 2 and inputs.dtype.kind in "iu" and inputs.shape == mask.shape and
                 mask.dtype.kind in "biu" and np.isin(mask, [0, 1]).all(), "evaluation input/mask contract")
        _require(labels.dtype.kind in "iu" and labels.shape[0] == inputs.shape[0] and
                 logits.dtype.kind == "f" and logits.shape[0] == inputs.shape[0] and
                 logits.ndim == (2 if report["task"] == "bert" else 3) and
                 (logits.ndim == 2 or logits.shape[1] == labels.shape[1]) and
                 loss.size == 1 and loss.dtype.kind == "f", "evaluation full logits/loss shape/dtype")
        if phase is not None:
            for key in initial:
                if key.startswith("input/"):
                    _require(prefix + key in source and np.array_equal(initial[key], source[prefix + key]),
                             "evaluation input changed within lifecycle: " + prefix + key)
    expected_sample = lifecycle["original/logits"]
    expected_sample = expected_sample[:, :4, :64] if expected_sample.ndim == 3 else expected_sample
    _require("logits_sample" in lifecycle and "loss" in lifecycle and
             np.array_equal(lifecycle["logits_sample"], expected_sample) and
             np.array_equal(lifecycle["loss"], lifecycle["original/loss"]), "legacy lifecycle aliases changed")
    info = report.get("lifecycle", {})
    _require(info.get("compute_policy") ==
             "FP32 master parameters; Accelerate AMP forward wrapper removed", "lifecycle compute policy")
    _require(info.get("reloaded_devices") == report["actual_parameter_devices"], "reload device evidence")
    if report["task"] == "qwen-lora":
        for key in ("adapter_saved", "reload_generation_exact", "merge_generation_exact", "merged_generation_exact"):
            _require(info.get(key) is True, "incomplete reload/merge evidence: " + key)
        for key in ("merged_devices", "merged_reload_devices"):
            _require(info.get(key) == report["actual_parameter_devices"], "merge device evidence: " + key)
    if report["task"] != "bert":
        _require(info.get("reload_generation_exact") is True, "missing exact reload generation evidence")
        for phase in phases:
            meta_key = _GENERATION_METADATA[phase]
            _require(meta_key in info, "missing phase generation metadata: " + meta_key)
            _generation(lifecycle, info[meta_key], phase + "/generation", report)
            for name in ("input_ids", "attention_mask", "greedy"):
                _require(np.array_equal(lifecycle[phase + "/generation/" + name],
                                        lifecycle["original/generation/" + name]),
                         "generation input/mask/greedy changed within lifecycle: " + phase + "/" + name)
        for name in ("input_ids", "attention_mask", "greedy", "sampled"):
            _require("generation/" + name in lifecycle and np.array_equal(
                lifecycle["generation/" + name], lifecycle["original/generation/" + name]),
                "legacy generation aliases changed")
    _require(set(lifecycle) == expected_keys, "unexpected or incomplete lifecycle archive keys")
    return [("initial-probe", initial_path, initial), ("lifecycle", lifecycle_path, lifecycle)]


def _array_result(left, right, label, budget=(0, 0), policy="exact"):
    _require(left.shape == right.shape, "shape mismatch: " + label)
    _require(left.dtype == right.dtype, "dtype mismatch: " + label)
    numeric = left.dtype.kind in "fciub"
    result = {"key": label, "shape": list(left.shape), "dtype": str(left.dtype), "policy": policy,
              "atol": budget[0], "rtol": budget[1], "elements": int(left.size)}
    if policy == "sampling_rng_ids":
        # Range, shape and the mandatory causal/decoder prefix were checked in each run.
        result.update({"passed": True, "different_ids": int(np.count_nonzero(left != right)),
                       "outside_budget": None, "max_abs": None, "max_rel": None})
        return result
    if not numeric:
        outside = int(np.count_nonzero(left != right))
        result.update({"passed": outside == 0, "outside_budget": outside,
                       "max_abs": None, "max_rel": None})
        return result
    difference = np.abs(left.astype(np.complex128 if left.dtype.kind == "c" else np.float64) - right)
    reference = np.abs(right.astype(np.complex128 if right.dtype.kind == "c" else np.float64))
    # Integer metadata and counters stay exact, including values too large for FP64.
    outside_mask = (left != right) if policy == "exact" else difference > budget[0] + budget[1] * reference
    nonzero = reference != 0
    zero_mismatches = int(np.count_nonzero((~nonzero) & (difference != 0)))
    relative = np.divide(difference, reference, out=np.zeros_like(difference), where=nonzero)
    outside = int(np.count_nonzero(outside_mask))
    result.update({"passed": outside == 0, "outside_budget": outside,
                   "max_abs": float(difference.max()) if difference.size else 0.0,
                   "max_rel": None if zero_mismatches else float(relative.max()) if relative.size else 0.0,
                   "max_rel_unbounded": bool(zero_mismatches), "zero_reference_mismatches": zero_mismatches})
    return result


def _compare_arrays(left, right, label, budgets, state=False, sampling=False, exact=False):
    _require(set(left) == set(right), "archive keys differ: " + label)
    result = []
    for key in sorted(left):
        policy = "exact"
        if not exact and sampling and key.endswith("/sampled"):
            policy = "sampling_rng_ids"
        elif not exact and left[key].dtype.kind in "fc":
            if state:
                if key.startswith(("model/parameter/", "model/buffer/")):
                    policy = "forward"
                elif key.startswith("optimizer/parameter/") and key.rsplit("/", 1)[-1] in _MOMENTS:
                    policy = "gradient"
            elif key.startswith("gradient/"):
                policy = "gradient"
            elif key.endswith("logits") or key.endswith("loss") or key == "logits_sample":
                policy = "forward"
        budget = budgets.get(policy, (0, 0))
        result.append(_array_result(left[key], right[key], label + ":" + key, budget, policy))
    return result


def _records(left, right, label, budgets, exact=False):
    if isinstance(left, dict):
        _require(isinstance(right, dict) and set(left) == set(right), "metadata keys differ: " + label)
        result = []
        for key in sorted(left):
            result.extend(_records(left[key], right[key], label + "/" + key, budgets, exact))
        return result
    if isinstance(left, list):
        _require(isinstance(right, list) and len(left) == len(right), "metadata list shape: " + label)
        result = []
        for index, (a, b) in enumerate(zip(left, right)):
            result.extend(_records(a, b, label + "/" + str(index), budgets, exact))
        return result
    _require(type(left) is type(right), "metadata dtype mismatch: " + label)
    policy = "exact"
    if not exact and label.rsplit("/", 1)[-1] in ("loss", "mean_microbatch_loss"):
        policy = "forward"
    elif not exact and label.rsplit("/", 1)[-1] == "l2_norm":
        policy = "gradient"
    return [_array_result(np.asarray(left), np.asarray(right), label, budgets.get(policy, (0, 0)), policy)]


def _finish(result, comparisons):
    result["comparisons"] = comparisons
    result["passed"] = all(item["passed"] for item in comparisons)
    result["failed_comparisons"] = sum(not item["passed"] for item in comparisons)
    result["outside_budget_elements"] = sum(item["outside_budget"] or 0 for item in comparisons)
    return result


def compare_cross_runtime(left, right, protocol):
    """Compare full Jittor/native runs; malformed/incomplete evidence fails closed."""
    left, right = Path(left).resolve(), Path(right).resolve()
    reports = [_read_report(run) for run in (left, right)]
    a, b = reports
    _require(a["runtime"] == "jittor" and b["runtime"] == "torch", "requires left Jittor / right native Torch")
    for field in _PROTOCOL_FIELDS:
        if field != "runtime":
            _require(_same_json(a[field], b[field]), "workflow protocol differs: " + field)
    _require(a["actual_parameter_devices"] == b["actual_parameter_devices"], "actual devices differ")
    budgets = _budget(protocol, a)
    states = [_state(run, report) for run, report in zip((left, right), reports)]
    _require(states[0][1] == states[1][1], "trainable_names differ")
    gradients = [validate_gradient_archives(run, report, state, names)
                 for run, report, (state, names) in zip((left, right), reports, states)]
    outputs = [_outputs(run, report) for run, report in zip((left, right), reports)]
    comparisons = _compare_arrays(states[0][0], states[1][0], "final-state", budgets, state=True)
    artifacts = {"left": {"report": _sha(left / "report.json"), "final-state": _sha(left / "final-state.npz")},
                 "right": {"report": _sha(right / "report.json"), "final-state": _sha(right / "final-state.npz")}}
    for left_entry, right_entry in zip(outputs[0], outputs[1]):
        label, left_path, left_values = left_entry
        _, right_path, right_values = right_entry
        comparisons.extend(_compare_arrays(left_values, right_values, label, budgets, sampling=True))
        artifacts["left"][label], artifacts["right"][label] = _sha(left_path), _sha(right_path)
    for x, y in zip(*gradients):
        _require(x["step"] == y["step"], "gradient step mismatch")
        comparisons.extend(_compare_arrays(x["arrays"], y["arrays"], "gradient-step-%d" % x["step"], budgets))
    for field in ("updates", "microbatches", "unscaled_gradients"):
        comparisons.extend(_records(a[field], b[field], field, budgets))
    for field in ("global_step", "start_step", "global_step_semantics", "expected_optimizer_attempts",
                  "optimizer_updates_completed", "optimizer_skips_completed", "optimizer_attempts_in_process",
                  "optimizer_updates_in_process", "projection_dtypes", "observed_autocast",
                  "master_parameter_dtypes", "optimizer_state_dtypes", "scaler_enabled",
                  "initial_scaler_state", "final_scaler_state", "tail_microbatch_observed",
                  "tail_accumulation_observed"):
        _require(field in a and field in b, "missing precision/execution metadata: " + field)
        comparisons.extend(_records(a[field], b[field], field, budgets, exact=True))
    comparisons.extend(_records(a["initial_probe"], b["initial_probe"], "initial_probe", budgets))
    for phase in (_PHASES if a["task"] == "qwen-lora" else _PHASES[:2]):
        if a["task"] != "bert":
            key = _GENERATION_METADATA[phase]
            comparisons.extend(_records(a["lifecycle"][key], b["lifecycle"][key], "lifecycle/" + key, budgets, exact=True))
    gradient_manifest = [{"step": x["step"], "left_path": x["path"], "left_sha256": x["sha256"],
                          "right_path": y["path"], "right_sha256": y["sha256"]} for x, y in zip(*gradients)]
    return _finish({"kind": "strict_full_cross_runtime", "left": str(left), "right": str(right),
                    "budgets": budgets, "artifacts_sha256": artifacts,
                    "trainable_names": states[0][1], "gradient_archives": gradient_manifest,
                    "sampling_policy": "IDs may differ; dtype/shape/range/prefix/metadata validated"}, comparisons)


def compare_gradient_segments(reference, segments, protocol):
    """Exact full-gradient replay supplement; the existing resume comparer is unchanged."""
    reference = Path(reference).resolve()
    runs = [Path(run).resolve() for run in segments]
    expected = _read_report(reference)
    state, names = _state(reference, expected)
    originals = validate_gradient_archives(reference, expected, state, names)
    actual, cursor, batch_rows, update_rows, health_rows = [], 0, [], [], []
    for run in runs:
        report = _read_report(run, full=False)
        for field in _PROTOCOL_FIELDS:
            _require(_same_json(report[field], expected[field]), "gradient replay protocol differs: " + field)
        _require(report["actual_parameter_devices"] == expected["actual_parameter_devices"], "gradient replay devices differ")
        _require(report["start_step"] == cursor and report["global_step"] > cursor,
                 "gradient replay segments must be contiguous, ordered and nonempty")
        final, inventory = _state(run, report)
        _require(inventory == names, "gradient replay trainable inventory differs")
        actual.extend(validate_gradient_archives(run, report, final, inventory))
        cursor = report["global_step"]
        batch_rows.extend(report["microbatches"])
        update_rows.extend(report["updates"])
        health_rows.extend(report["unscaled_gradients"])
    _require(cursor == expected["global_step"] and len(actual) == len(originals), "incomplete gradient replay")
    budgets = _budget(protocol, expected)
    comparisons = []
    for a, b in zip(originals, actual):
        _require(a["step"] == b["step"], "gradient replay step mismatch")
        comparisons.extend(_compare_arrays(a["arrays"], b["arrays"], "gradient-step-%d" % a["step"], budgets, exact=True))
    for field, rows in (("microbatches", batch_rows), ("updates", update_rows), ("unscaled_gradients", health_rows)):
        comparisons.extend(_records(expected[field], rows, field, budgets, exact=True))
    manifests = [{"step": a["step"], "reference_path": a["path"], "reference_sha256": a["sha256"],
                  "segment_path": b["path"], "segment_sha256": b["sha256"]} for a, b in zip(originals, actual)]
    return _finish({"kind": "exact_same_runtime_full_gradients", "reference": str(reference),
                    "segments": [str(run) for run in runs], "budgets": {"atol": 0, "rtol": 0},
                    "gradient_archives": manifests, "trainable_names": names}, comparisons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    cross = modes.add_parser("cross")
    cross.add_argument("--left", required=True)
    cross.add_argument("--right", required=True)
    replay = modes.add_parser("gradients")
    replay.add_argument("--reference", required=True)
    replay.add_argument("--segments", nargs="+", required=True)
    for mode in (cross, replay):
        mode.add_argument("--protocol", required=True)
        mode.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        result = (compare_cross_runtime(args.left, args.right, args.protocol) if args.mode == "cross" else
                  compare_gradient_segments(args.reference, args.segments, args.protocol))
    except (AssertionError, KeyError, ValueError, OSError) as error:
        result = {"passed": False, "failed_comparisons": 1, "outside_budget_elements": None,
                  "integrity_error": str(error), "error_type": type(error).__name__}
    Path(args.report).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "failed_comparisons", "outside_budget_elements")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
