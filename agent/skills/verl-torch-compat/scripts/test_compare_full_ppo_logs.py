import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).with_name("compare_full_ppo_logs.py")
SPEC = importlib.util.spec_from_file_location("compare_full_ppo_logs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _log(path, *, steps=(1, 2, 3, 4, 5), time=10.0, throughput=7.6,
         actor_mem=20.0, critic_mem=15.0, changes=None):
    rows = ["Training Progress: 100%|##########| 5/5"]
    for step in steps:
        metrics = {
            "actor/loss": "0.25", "actor/grad_norm": "np.float64(2.0)",
            "critic/loss": "np.float64(0.5)", "critic/grad_norm": "3.0",
            "actor/perf/max_memory_allocated_gb": str(actor_mem),
            "critic/perf/max_memory_allocated_gb": str(critic_mem),
            "timing_s/step": str(time), "perf/total_num_tokens": "76",
            "perf/throughput": str(throughput),
        }
        metrics.update((changes or {}).get(step, {}))
        rows.append(f"step:{step} - " + " - ".join(f"{key}:{value}" for key, value in metrics.items()))
    path.write_text("\n".join(rows) + "\n")


def _manifest(path, role):
    return {
        "schema": 1, "role": role,
        "framework": "jittor" if role == "candidate" else "torch",
        "backend": "npu", "world_size": 2, "model": "Qwen3.5-0.8B",
        "weights_sha256": "a" * 64, "dtype": "bf16",
        "downstream_versions": {"verl": "0.9.1", "transformers": "5.9.0",
                                "tensordict": "0.10.0", "ray": "2.58.0"},
        "verl_source_sha256": "b" * 64, "input_sha256": "c" * 64,
        "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "exit_code": 0,
        "protocol": {"warmup_steps": [1, 2], "timed_steps": [3, 4, 5],
                     "config_sha256": "d" * 64, "throughput_scope": "global",
                     "memory_metric": "allocated"},
        "completed_steps": [1, 2, 3, 4, 5],
    }


@pytest.fixture
def paired(tmp_path):
    candidate, oracle = tmp_path / "candidate.log", tmp_path / "oracle.log"
    _log(candidate, time=20.0, throughput=3.8, actor_mem=22.0, critic_mem=18.0)
    _log(oracle)
    return (MODULE.parse_log(candidate), MODULE.parse_log(oracle),
            _manifest(candidate, "candidate"), _manifest(oracle, "oracle"))


def test_parse_and_compare(paired):
    candidate, oracle, left, right = paired
    assert candidate["status"] == "log_smoke_only"
    assert [row["step"] for row in candidate["steps"]] == [1, 2, 3, 4, 5]
    result = MODULE.compare(candidate, oracle, left, right)
    assert result["step_time_ratio_candidate_over_oracle"] == 2.0
    assert result["throughput_ratio_candidate_over_oracle"] == 0.5
    assert result["actor_peak_memory_ratio_candidate_over_oracle"] == 1.1
    assert result["critic_peak_memory_ratio_candidate_over_oracle"] == 1.2


def test_requires_paired_manifests(paired):
    candidate, oracle, left, _ = paired
    with pytest.raises(ValueError, match="both.*manifests"):
        MODULE.compare(candidate, oracle)
    with pytest.raises(ValueError, match="both.*manifests"):
        MODULE.compare(candidate, oracle, left)


@pytest.mark.parametrize("bad", ["nan", "inf", "np.float64(nan)", "-inf", "1e999"])
@pytest.mark.parametrize("step", [1, 3, 5])
def test_rejects_nonfinite_metrics_in_any_step(tmp_path, bad, step):
    path = tmp_path / "bad.log"
    _log(path, changes={step: {"actor/loss": bad}})
    with pytest.raises(ValueError, match="non-finite"):
        MODULE.parse_log(path)


@pytest.mark.parametrize("steps", [(1, 3, 4, 5), (1, 2, 2, 3, 4), (2, 3, 4, 5),
                                   (1, 3, 2, 4, 5), (1, 2, 3, 4, 9)])
def test_rejects_nonconsecutive_steps(tmp_path, steps):
    path = tmp_path / "bad.log"
    _log(path, steps=steps)
    with pytest.raises(ValueError, match="consecutive"):
        MODULE.parse_log(path)


@pytest.mark.parametrize("metric", ["timing_s/step", "perf/throughput", "perf/total_num_tokens",
    "actor/perf/max_memory_allocated_gb", "critic/perf/max_memory_allocated_gb"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_rejects_nonpositive_performance_metrics(tmp_path, metric, value):
    path = tmp_path / "bad.log"
    _log(path, changes={2: {metric: value}})
    with pytest.raises(ValueError, match="positive"):
        MODULE.parse_log(path)


@pytest.mark.parametrize("kind", ["missing_metric", "missing_step", "duplicate_metric",
    "traceback", "incomplete", "fractional_tokens", "nonfinite_optional"])
def test_rejects_malformed_logs(tmp_path, kind):
    path = tmp_path / "bad.log"
    _log(path)
    text = path.read_text()
    replacements = {
        "missing_metric": ("actor/loss:0.25", "actor/loss:unknown"),
        "missing_step": ("step:1", "step:unknown"),
        "duplicate_metric": ("actor/loss:0.25", "actor/loss:0.25 - actor/loss:0.3"),
        "incomplete": ("Training Progress: 100%", "Training Progress: 20%"),
        "fractional_tokens": ("perf/total_num_tokens:76", "perf/total_num_tokens:76.5"),
        "nonfinite_optional": ("actor/loss:0.25", "actor/loss:0.25 - extra/metric:nan"),
    }
    if kind == "traceback":
        text += "Traceback (most recent call last):\n"
    else:
        text = text.replace(*replacements[kind], 1)
    path.write_text(text)
    with pytest.raises(ValueError):
        MODULE.parse_log(path)


@pytest.mark.parametrize("field,value", [
    ("schema", 2), ("role", "candidate"), ("framework", "jittor"), ("backend", "cpu"),
    ("world_size", 1), ("world_size", True), ("exit_code", 1), ("exit_code", None),
    ("exit_code", False), ("log_sha256", "e" * 64), ("model", "different"),
    ("weights_sha256", "f" * 64), ("dtype", "fp32"), ("input_sha256", "e" * 64),
    ("verl_source_sha256", "f" * 64), ("completed_steps", [1, 2, 3, 4]),
    ("completed_steps", [True, 2, 3, 4, 5]), ("weights_sha256", "not-a-digest"),
    ("weights_sha256", None), ("protocol", []),
    ("downstream_versions", {"verl": "0.9.1"}),
])
def test_rejects_invalid_or_mismatched_manifests(paired, field, value):
    candidate, oracle, left, right = paired
    right[field] = value
    with pytest.raises(ValueError):
        MODULE.compare(candidate, oracle, left, right)


@pytest.mark.parametrize("field", ["schema", "framework", "model", "dtype", "exit_code",
    "world_size", "weights_sha256", "input_sha256", "verl_source_sha256", "log_sha256",
    "downstream_versions", "protocol", "completed_steps"])
def test_rejects_missing_manifest_fields(paired, field):
    candidate, oracle, left, right = paired
    del right[field]
    with pytest.raises(ValueError):
        MODULE.compare(candidate, oracle, left, right)


@pytest.mark.parametrize("package", ["verl", "transformers", "tensordict", "ray"])
def test_rejects_different_downstream_versions(paired, package):
    candidate, oracle, left, right = paired
    right["downstream_versions"][package] = "different"
    with pytest.raises(ValueError, match="downstream_versions"):
        MODULE.compare(candidate, oracle, left, right)


@pytest.mark.parametrize("field,value", [
    ("warmup_steps", [1]), ("timed_steps", [3, 4]), ("timed_steps", [2, 3, 4]),
    ("timed_steps", [3, 5, 4]), ("config_sha256", "e" * 64),
    ("throughput_scope", "per_rank"), ("memory_metric", "reserved"),
])
def test_rejects_invalid_or_mismatched_protocols(paired, field, value):
    candidate, oracle, left, right = paired
    right["protocol"][field] = value
    with pytest.raises(ValueError):
        MODULE.compare(candidate, oracle, left, right)


def test_short_run_is_only_smoke(tmp_path):
    path = tmp_path / "short.log"
    _log(path, steps=(1,))
    parsed = MODULE.parse_log(path)
    assert parsed["status"] == "log_smoke_only"
    with pytest.raises(ValueError, match="observed"):
        MODULE.validate_manifest(_manifest(path, "candidate"), parsed, "candidate")


def test_warmup_excluded_and_all_token_counts_checked(tmp_path):
    candidate, oracle = tmp_path / "candidate.log", tmp_path / "oracle.log"
    _log(candidate, changes={1: {"timing_s/step": "10000"}, 2: {"timing_s/step": "20000"},
        3: {"timing_s/step": "1000"}, 4: {"timing_s/step": "20"}, 5: {"timing_s/step": "30"}})
    _log(oracle)
    result = MODULE.compare(MODULE.parse_log(candidate), MODULE.parse_log(oracle),
                            _manifest(candidate, "candidate"), _manifest(oracle, "oracle"))
    assert result["step_time_ratio_candidate_over_oracle"] == 3.0
    _log(oracle, changes={1: {"perf/total_num_tokens": "77"}})
    with pytest.raises(ValueError, match="token counts"):
        MODULE.compare(MODULE.parse_log(candidate), MODULE.parse_log(oracle),
                       _manifest(candidate, "candidate"), _manifest(oracle, "oracle"))


@pytest.mark.parametrize("with_manifests", [False, True])
def test_cli_labels_limits_without_claiming_gates(tmp_path, monkeypatch, capsys, with_manifests):
    candidate, oracle = tmp_path / "candidate.log", tmp_path / "oracle.log"
    _log(candidate)
    _log(oracle)
    args = [str(SCRIPT), "--candidate", str(candidate), "--oracle", str(oracle)]
    if with_manifests:
        for path, role in ((candidate, "candidate"), (oracle, "oracle")):
            manifest_path = path.with_suffix(".json")
            manifest_path.write_text(json.dumps(_manifest(path, role)))
            args.extend(["--" + role + "-manifest", str(manifest_path)])
    monkeypatch.setattr(sys, "argv", args)
    MODULE.main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == ("reported_metrics_comparison_only" if with_manifests else "log_smoke_only")
    assert ("comparison" in report) == with_manifests
    assert any("self-reported" in item for item in report["limitations"])
    assert any("L4" in item and "L5" in item for item in report["limitations"])


def test_cli_rejects_one_manifest(tmp_path, monkeypatch):
    candidate, oracle = tmp_path / "candidate.log", tmp_path / "oracle.log"
    _log(candidate)
    _log(oracle)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--candidate", str(candidate),
        "--oracle", str(oracle), "--candidate-manifest", str(tmp_path / "unused.json")])
    with pytest.raises(SystemExit) as error:
        MODULE.main()
    assert error.value.code == 2


def test_rejects_overflow_in_derived_ratio(tmp_path):
    candidate, oracle = tmp_path / "candidate.log", tmp_path / "oracle.log"
    _log(candidate, time=1e308)
    _log(oracle, time=1e-308)
    with pytest.raises(ValueError, match="ratios must be finite"):
        MODULE.compare(MODULE.parse_log(candidate), MODULE.parse_log(oracle),
                       _manifest(candidate, "candidate"), _manifest(oracle, "oracle"))
