"""Host-only integrity and numeric contracts for complete HF evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from _helpers.child_process import PYTHON, child_env

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _release_hf_cross_runtime as cross
import _release_hf_lifecycle as life


_BLOCKED_IMPORTS = """
import sys
_RUNTIME_PREFIXES = ('torch', 'torch_npu', 'jittor', 'transformers', 'peft', 'accelerate')
assert not any(name.split('.', 1)[0] in _RUNTIME_PREFIXES for name in sys.modules)
class _RejectRuntimeImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in _RUNTIME_PREFIXES:
            raise AssertionError('blocked runtime import: ' + fullname)
        return None
sys.meta_path.insert(0, _RejectRuntimeImports())
"""
_CLI_WITH_BLOCKED_IMPORTS = _BLOCKED_IMPORTS + """
from pathlib import Path
import runpy
sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""

_BUDGET = {"numeric_budgets": {
    "float32": {"forward_atol": 1e-5, "forward_rtol": 1e-4,
                "gradient_atol": 1e-5, "gradient_rtol": 1e-3},
    "bfloat16": {"forward_atol": .005, "forward_rtol": .03,
                 "gradient_atol": .01, "gradient_rtol": .05}}}
_NAMES = ["layer.a", "layer.b"]


def _save(path, values):
    np.savez(path, **values)


def _mutate_archive(path, transform):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    transform(values)
    _save(path, values)


def _health(step=None):
    value = {"missing_count": 0, "nonfinite_count": 0, "nonfinite_names": [],
             "dtypes": ["torch.float32"], "l2_norm": .5,
             "l2_norm_algorithm": "host_fp64_pairwise_256_v1", "l2_norm_device": "cpu"}
    if step is not None:
        value.update(step=step, gradient_phase="postclip_unscaled")
    return value


def _make_run(path, runtime="jittor", start=0, end=4):
    path.mkdir(parents=True)
    (path / "gradients").mkdir()
    report = {
        "runtime": runtime, "task": "qwen-lora", "device": "cpu", "mixed_precision": "no",
        "seed": 821, "versions": {"transformers": "fixed", "numpy": "fixed"},
        "initialization": {"task": "qwen-lora", "shared": True},
        "initial_model_dir": "shared/base", "initial_adapter_dir": "shared/adapter",
        "train_file_sha256": "train", "eval_file_sha256": "eval", "train_samples": 16,
        "eval_samples": 4, "epochs": 1, "batch_size": 2, "gradient_accumulation_steps": 2,
        "state_format": "full", "fixed_order": True, "sampler": "sequential fixture",
        "num_workers": 0, "gradient_checkpointing": False, "microbatches_per_epoch": 8,
        "updates_per_epoch": 4, "expected_updates": 4, "gradient_norm_algorithm": "host_fp64_pairwise_256_v1",
        "optimizer_gradient_phase": "postclip_unscaled", "max_grad_norm": 1.0,
        "start_step": start, "global_step": end, "resume_checkpoint": "checkpoint-%d" % start if start else None,
        "resume_loaded_state_exact": True if start else None, "actual_parameter_devices": ["cpu"],
        "fallback_count": 0 if runtime == "jittor" else None,
        "fallback_measurement": "jittor_core_attempt_counter" if runtime == "jittor" else "not_available_in_native_torch",
        "frozen_base_unchanged": True, "updates": [], "microbatches": [], "unscaled_gradients": [],
        "gradient_archives": [], "optimizer_updates_completed": end, "optimizer_skips_completed": 0,
        "optimizer_attempts_in_process": end - start, "optimizer_updates_in_process": end - start,
        "expected_optimizer_attempts": 4, "global_step_semantics": "attempted optimizer windows",
        "projection_dtypes": ["torch.float32"], "observed_autocast": [False],
        "master_parameter_dtypes": ["torch.float32"], "optimizer_state_dtypes": ["torch.float32"],
        "scaler_enabled": False, "initial_scaler_state": None, "final_scaler_state": None,
        "tail_microbatch_observed": False, "tail_accumulation_observed": False}
    for step in range(start + 1, end + 1):
        batches = []
        for index in range(2):
            sample = (step - 1) * 4 + index * 2
            batches.append({"step": step, "sample_ids": [sample, sample + 1], "batch_size": 2,
                "input_shape": [2, 5], "label_shape": [2, 5], "masked_label_tokens": 2,
                "actual_device": "cpu", "loss": 1., "gradient_health": _health(),
                "gradient_phase": "backward", "gradient_scale": None})
        report["microbatches"].extend(batches)
        report["updates"].append({"step": step, "epoch": step / 4., "microbatches": 2, "sample_count": 4,
            "mean_microbatch_loss": 1., "learning_rates": [.001], "optimizer_step_skipped": False,
            "optimizer_update_performed": True, "optimizer_updates_completed": step,
            "optimizer_skips_completed": 0, "precision_protocol": {"overflow": False, "successful_update": True},
            "scaler": None})
        report["unscaled_gradients"].append(_health(step))
        gradient = {"gradient/" + name: np.full((2,), step / 10, np.float32) for name in _NAMES}
        gradient.update({"dtype/" + name: np.asarray("torch.float32") for name in _NAMES})
        archive = path / "gradients" / ("step-%06d.npz" % step)
        _save(archive, gradient)
        row = {"step": step, "path": str(archive), "phase": "postclip_unscaled",
               "trainable_gradient_count": 2, "finite": True, "nonfinite_names": [], "microbatches": batches}
        report["gradient_archives"].append(row)
        archive.with_suffix(".json").write_text(json.dumps(row))
    state = {"model/trainable_names": np.asarray(_NAMES), "scheduler/last_epoch": np.asarray(end),
             "scheduler/_step_count": np.asarray(end + 1), "scaler": np.asarray("<none>")}
    for name in _NAMES:
        state["model/parameter/" + name] = np.ones(2, np.float32)
        state["model/parameter/" + name + "/@dtype"] = np.asarray("torch.float32")
        for key in ("exp_avg", "exp_avg_sq", "step"):
            prefix = "optimizer/parameter/" + name + "/" + key
            state[prefix] = np.asarray(end, np.float32) if key == "step" else np.ones(2, np.float32)
            state[prefix + "/@dtype"] = np.asarray("torch.float32")
    group = {"lr": .001, "betas/0000": .9, "betas/0001": .999, "eps": 1e-8, "weight_decay": .01,
             "amsgrad": False, "maximize": False, "foreach": "<none>", "capturable": False,
             "differentiable": False, "fused": "<none>", "decoupled_weight_decay": True}
    state.update({"optimizer/group/0000/" + key: np.asarray(value) for key, value in group.items()})
    _save(path / "final-state.npz", state)
    inputs = np.asarray([[1, 2, 3, 4, 5], [0, 0, 1, 2, 3]], np.int64)
    probe = {"input/input_ids": inputs, "input/attention_mask": (inputs != 0).astype(np.int64),
             "input/labels": inputs.copy(), "logits": np.ones((2, 5, 16), np.float32),
             "loss": np.asarray(1., np.float32)}
    _save(path / "initial-probe.npz", probe)
    report["initial_probe"] = {"logits_dtype": "torch.float32", "loss_dtype": "torch.float32", "device": "cpu", "loss": 1.}
    gen_inputs = inputs[:, :3]
    generated = np.concatenate((gen_inputs, np.full((2, 2), 6, np.int64)), axis=1)
    generation = {"input_ids": gen_inputs, "attention_mask": (gen_inputs != 0).astype(np.int64),
                  "greedy": generated, "sampled": generated.copy()}
    meta = {"input_lengths": [3, 1], "greedy_cache_exact": True,
            "sampling": {"top_k": 10, "top_p": .9, "temperature": .8},
            "compute_policy": "FP32 model without autocast", "actual_device": "cpu",
            "max_new_tokens": 2, "vocab_size": 16, "is_encoder_decoder": False,
            "decoder_start_token_id": None}
    result = {"compute_policy": "FP32 master parameters; Accelerate AMP forward wrapper removed",
              "adapter_saved": True, "reloaded_devices": ["cpu"], "merged_devices": ["cpu"],
              "merged_reload_devices": ["cpu"], "reload_generation_exact": True,
              "merge_generation_exact": True, "merged_generation_exact": True}
    arrays = {"logits_sample": probe["logits"][:, :4, :64], "loss": probe["loss"]}
    for phase in cross._PHASES:
        arrays.update({phase + "/" + key: value.copy() for key, value in probe.items()})
        arrays.update({phase + "/generation/" + key: value.copy() for key, value in generation.items()})
        result[cross._GENERATION_METADATA[phase]] = copy.deepcopy(meta)
    arrays.update({"generation/" + key: value.copy() for key, value in generation.items()})
    _save(path / "lifecycle.npz", arrays)
    report["lifecycle"] = result
    (path / "report.json").write_text(json.dumps(report))
    return report


@pytest.fixture
def runs(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(_BUDGET))
    left, right = tmp_path / "shim", tmp_path / "native"
    _make_run(left)
    _make_run(right, "torch")
    return left, right, protocol


def test_complete_runs_keep_every_value_and_sampling_can_use_another_rng(runs):
    left, right, protocol = runs
    def change(values):
        for key in values:
            if key.endswith("/sampled"):
                values[key][:, -1] = 7
    _mutate_archive(left / "lifecycle.npz", change)
    result = cross.compare_cross_runtime(left, right, protocol)
    assert result["passed"] and len(result["gradient_archives"]) == 4
    assert all(len(row[side + "_sha256"]) == 64 for row in result["gradient_archives"] for side in ("left", "right"))
    assert len(result["comparisons"]) > 200
    assert any(row["policy"] == "sampling_rng_ids" and row["different_ids"] for row in result["comparisons"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kind", ["missing_logits", "missing_gradient", "extra_gradient", "gradient_dtype",
    "missing_step", "counter", "dtype", "input", "nonfinite", "fallback", "group", "path_escape",
    "protocol", "sample_range", "sample_prefix", "greedy", "metadata_counter"])
def test_corrupted_complete_evidence_fails(runs, kind):
    left, right, protocol = runs
    if kind == "missing_logits":
        _mutate_archive(left / "lifecycle.npz", lambda v: v.pop("merged_reload/logits"))
    elif kind in ("missing_gradient", "extra_gradient", "gradient_dtype"):
        def corrupt(values):
            if kind == "missing_gradient":
                values.pop("gradient/layer.a")
            elif kind == "extra_gradient":
                values["gradient/unknown"] = np.ones(2, np.float32)
            else:
                values["gradient/layer.a"] = values["gradient/layer.a"].astype(np.float64)
        _mutate_archive(left / "gradients/step-000001.npz", corrupt)
    elif kind in ("counter", "dtype", "group"):
        key = {"counter": "optimizer/parameter/layer.a/step", "dtype": "model/parameter/layer.a",
               "group": "optimizer/group/0000/decoupled_weight_decay"}[kind]
        def corrupt(values):
            if kind == "group":
                values.pop(key)
            elif kind == "counter":
                values[key] = np.asarray(5., np.float32)
            else:
                values[key] = values[key].astype(np.float64)
        _mutate_archive(left / "final-state.npz", corrupt)
    elif kind in ("input", "nonfinite", "sample_range", "sample_prefix", "greedy"):
        def corrupt(values):
            if kind == "input":
                values["reloaded/input/input_ids"][0, 0] = 3
            elif kind == "nonfinite":
                values["merged_reload/logits"][0, 4, 15] = np.inf
            elif kind == "sample_range":
                values["merged_reload/generation/sampled"][0, -1] = 16
            elif kind == "sample_prefix":
                values["merged_reload/generation/sampled"][0, 0] = 3
            else:
                values["merged_reload/generation/greedy"][0, -1] = 7
        _mutate_archive(left / "lifecycle.npz", corrupt)
    else:
        target = right if kind == "fallback" else left
        report = json.loads((target / "report.json").read_text())
        if kind == "missing_step":
            report["gradient_archives"].pop()
        elif kind == "fallback":
            report["fallback_count"] = 0
        elif kind == "path_escape":
            report["gradient_archives"][0]["path"] = str(right / "gradients/step-000001.npz")
        elif kind == "metadata_counter":
            report["updates"][0]["optimizer_updates_completed"] = 2
        else:
            report["seed"] += 1
        (target / "report.json").write_text(json.dumps(report))
    try:
        result = cross.compare_cross_runtime(left, right, protocol)
    except AssertionError:
        return
    assert not result["passed"], kind


def test_original_fp32_budgets_and_optimizer_metadata_are_strict(runs):
    left, right, protocol = runs
    def corrupt(values):
        # Outside FP32 budget, well inside BF16. Keep legacy alias coherent.
        values["original/logits"][0, 4, 15] += .001
        values["merged_reload/logits"][0, 4, 15] += .001
    _mutate_archive(left / "lifecycle.npz", corrupt)
    result = cross.compare_cross_runtime(left, right, protocol)
    assert not result["passed"] and result["budgets"]["precision"] == "float32"
    assert result["outside_budget_elements"] == 2
    _mutate_archive(left / "final-state.npz", lambda v: v.__setitem__(
        "optimizer/group/0000/lr", np.asarray(.001000000001)))
    result = cross.compare_cross_runtime(left, right, protocol)
    assert any(row["key"].endswith("optimizer/group/0000/lr") and not row["passed"] for row in result["comparisons"])


def test_full_gradient_and_moments_use_gradient_budget_not_forward(runs):
    left, right, protocol = runs
    _mutate_archive(left / "gradients/step-000001.npz", lambda v: v["gradient/layer.a"].__setitem__(0, .10005))
    _mutate_archive(left / "final-state.npz", lambda v: v["optimizer/parameter/layer.a/exp_avg"].__setitem__(0, 1.0005))
    result = cross.compare_cross_runtime(left, right, protocol)
    assert result["passed"]
    rows = [row for row in result["comparisons"] if row["key"].endswith("gradient/layer.a") or
            row["key"].endswith("optimizer/parameter/layer.a/exp_avg")]
    assert rows and all(row["policy"] == "gradient" for row in rows)


def test_exact_gradient_segments_cover_all_steps_without_weakening_resume(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(_BUDGET))
    full, first, resume = [tmp_path / name for name in ("full", "first", "resume")]
    _make_run(full)
    _make_run(first, end=2)
    _make_run(resume, start=2)
    assert cross.compare_gradient_segments(full, [first, resume], protocol)["passed"]
    with pytest.raises(AssertionError, match="contiguous"):
        cross.compare_gradient_segments(full, [resume, first], protocol)
    _mutate_archive(resume / "gradients/step-000003.npz", lambda v: v["gradient/layer.a"].__setitem__(0, .30000004))
    assert not cross.compare_gradient_segments(full, [first, resume], protocol)["passed"]


def test_fp16_without_frozen_fixture_budget_is_rejected(runs):
    with pytest.raises(AssertionError, match="frozen"):
        cross._budget(runs[2], {"mixed_precision": "fp16"})


def test_comparator_import_does_not_load_either_runtime():
    code = _BLOCKED_IMPORTS + """
import _release_hf_cross_runtime as cross
from _release_hf_data import digest_file
assert cross._sha is digest_file
assert not any(name.split('.', 1)[0] in _RUNTIME_PREFIXES for name in sys.modules)
"""
    subprocess.run([PYTHON, "-c", code], check=True,
                   env=child_env({"PYTHONPATH": str(Path(__file__).resolve().parent)},
                                 without_torch_mode=True),
                   cwd=str(Path(__file__).resolve().parent))


def test_runtime_import_blocker_rejects_an_actual_forbidden_import():
    result = subprocess.run([PYTHON, "-c", _BLOCKED_IMPORTS + "\nimport transformers"],
                            env=child_env(without_torch_mode=True), capture_output=True)
    assert result.returncode != 0
    assert b"blocked runtime import: transformers" in result.stderr


def test_lifecycle_archives_full_unsampled_values_and_all_four_stages(tmp_path):
    model = SimpleNamespace(to=lambda _: model,
        save_pretrained=lambda directory, **kw: Path(directory).mkdir(parents=True),
        merge_and_unload=lambda **kw: model)
    class Tokenizer:
        def save_pretrained(self, directory):
            Path(directory).mkdir(exist_ok=True)
        def __call__(self, texts):
            return {"input_ids": [[1] for _ in texts]}
    tokenizer = Tokenizer()
    def save_model(directory):
        Path(directory).mkdir()
        (Path(directory) / "adapter_model.safetensors").write_bytes(b"adapter")
    trainer = SimpleNamespace(save_model=save_model)
    inputs = np.ones((2, 6), np.int64)
    probe = {"input/input_ids": inputs, "input/attention_mask": inputs,
             "input/labels": inputs, "logits": np.arange(2 * 6 * 80, dtype=np.float32).reshape(2, 6, 80),
             "loss": np.asarray(1., np.float32)}
    generated = {"greedy": inputs, "sampled": inputs, "input_ids": inputs, "attention_mask": inputs}
    args = SimpleNamespace(output_dir=str(tmp_path), task="qwen-lora", device="cpu", seed=821,
                           max_new_tokens=2, skip_merge=False, reload_atol=1e-5, reload_rtol=1e-4)
    tf = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer))
    peft = SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=lambda *a, **kw: model))
    factory = SimpleNamespace(from_pretrained=lambda *a, **kw: model)
    with mock.patch.dict(sys.modules, {"transformers": tf, "peft": peft}), \
            mock.patch.object(life, "unwrap_fp32_model", return_value=model), \
            mock.patch.object(life, "evaluation_probe", return_value=(probe, {})), \
            mock.patch.object(life, "generation", return_value=(generated, {"complete": True})), \
            mock.patch.object(life, "load_base", return_value=model), \
            mock.patch.object(life, "model_class", return_value=factory), \
            mock.patch.object(life, "on_device", return_value=["cpu"]):
        result = life.lifecycle(args, SimpleNamespace(float32="float32"), trainer, tokenizer, [], None, None, None)
    with np.load(tmp_path / "lifecycle.npz") as archive:
        assert archive["logits_sample"].shape == (2, 4, 64)
        for phase in cross._PHASES:
            np.testing.assert_array_equal(archive[phase + "/logits"], probe["logits"])
            assert archive[phase + "/logits"][1, 5, 79] == 959
            for key in generated:
                np.testing.assert_array_equal(archive[phase + "/generation/" + key], generated[key])
            assert cross._GENERATION_METADATA[phase] in result


def test_cli_preserves_json_failure_for_incomplete_full_outputs(runs, tmp_path):
    left, right, protocol = runs
    report = tmp_path / "comparison.json"
    command = [PYTHON, "-c", _CLI_WITH_BLOCKED_IMPORTS, str(Path(cross.__file__).resolve()), "cross", "--left", str(left),
               "--right", str(right), "--protocol", str(protocol), "--report", str(report)]
    assert subprocess.run(command, env=child_env(without_torch_mode=True), capture_output=True).returncode == 0
    assert json.loads(report.read_text())["passed"]
    _mutate_archive(left / "lifecycle.npz", lambda v: v.pop("merged_reload/logits"))
    assert subprocess.run(command, env=child_env(without_torch_mode=True), capture_output=True).returncode == 1
    assert "missing full output" in json.loads(report.read_text())["integrity_error"]


def test_nonfinite_json_numbers_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    for value in ("NaN", "Infinity", "1e999"):
        path.write_text('{"value": ' + value + '}')
        with pytest.raises(AssertionError, match="nonfinite"):
            cross._json(path)


@pytest.mark.parametrize("task", ["bert", "t5"])
def test_non_lora_lifecycles_and_encoder_decoder_prefix(runs, task):
    left, right, protocol = runs
    for run in (left, right):
        report = json.loads((run / "report.json").read_text())
        report["task"] = task
        def transform(values):
            for key in list(values):
                if key.startswith(("merged/", "merged_reload/")) or (task == "bert" and "/generation/" in key):
                    values.pop(key)
                elif task == "bert" and key.startswith("generation/"):
                    values.pop(key)
                elif task == "bert" and (key.endswith("logits") or key == "logits_sample"):
                    values[key] = values[key][:, 0, :]
                elif task == "t5" and (key.endswith("/greedy") or key.endswith("/sampled")):
                    values[key] = np.asarray([[0, 6, 6], [0, 6, 6]], np.int64)
        _mutate_archive(run / "lifecycle.npz", transform)
        if task == "bert":
            _mutate_archive(run / "initial-probe.npz", lambda v: v.__setitem__("logits", v["logits"][:, 0, :]))
        else:
            for key in ("generation", "reloaded_generation"):
                report["lifecycle"][key].update(is_encoder_decoder=True, decoder_start_token_id=0)
        (run / "report.json").write_text(json.dumps(report))
    assert cross.compare_cross_runtime(left, right, protocol)["passed"]
    if task == "t5":
        _mutate_archive(left / "lifecycle.npz", lambda v: v["reloaded/generation/sampled"].__setitem__((0, 0), 1))
        with pytest.raises(AssertionError, match="prefix"):
            cross.compare_cross_runtime(left, right, protocol)



def test_generation_returns_full_inputs_masks_and_model_metadata():
    from contextlib import nullcontext
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.int64)
            self.device = "cpu"
    class Batch(dict):
        def to(self, device):
            assert device == "cpu"
            return self
    batch = Batch(input_ids=Tensor([[1, 2, 3], [0, 0, 4]]),
                  attention_mask=Tensor([[1, 1, 1], [0, 0, 1]]))
    class Tokenizer:
        padding_side = "right"
        pad_token_id = 0
        def __call__(self, *args, **kwargs):
            assert self.padding_side == "left"
            return batch
    calls, seeds = [], []
    class Model:
        config = SimpleNamespace(vocab_size=16, is_encoder_decoder=False, decoder_start_token_id=None)
        def eval(self):
            pass
        def generate(self, **kwargs):
            calls.append(kwargs)
            return Tensor(np.concatenate((batch["input_ids"].values, np.full((2, 2), 6)), axis=1))
    tokenizer = Tokenizer()
    fake_torch = SimpleNamespace(no_grad=nullcontext, manual_seed=seeds.append)
    guard = SimpleNamespace(_assert_values_device=lambda *args: None)
    with mock.patch.dict(sys.modules, {"torch": fake_torch, "_hf_trainer_e2e_runner": guard}), \
            mock.patch.object(life, "array", side_effect=lambda value: value.values.copy()):
        values, metadata = life.generation(Model(), tokenizer, "qwen-lora", "cpu", 821, 2)
    assert set(values) == {"input_ids", "attention_mask", "greedy", "sampled"}
    np.testing.assert_array_equal(values["input_ids"], batch["input_ids"].values)
    np.testing.assert_array_equal(values["attention_mask"], batch["attention_mask"].values)
    assert metadata["vocab_size"] == 16 and metadata["decoder_start_token_id"] is None
    assert metadata["input_lengths"] == [3, 1] and metadata["is_encoder_decoder"] is False
    assert tokenizer.padding_side == "right" and seeds == [821 + 913]
    assert [row["use_cache"] for row in calls] == [True, False, True]



def test_protocol_boolean_is_not_an_integer_alias(runs):
    left, right, protocol = runs
    report = json.loads((left / "report.json").read_text())
    report["fixed_order"] = 1
    (left / "report.json").write_text(json.dumps(report))
    with pytest.raises(AssertionError, match="fixed_order"):
        cross.compare_cross_runtime(left, right, protocol)
