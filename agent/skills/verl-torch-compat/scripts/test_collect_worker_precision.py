"""Host-only startup-evidence collector contracts; no framework imports."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).with_name("collect_worker_precision.py")
MARKER = "VERL_CUDA_STRICT_FP32_READY "


def record(pid=11, framework="torch"):
    return {"framework": framework, "pid": pid, "scope": "public_precision_readback_only",
            "runtime_conditions": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                                   "cudnn_benchmark": False, "float32_matmul_precision": "highest"}}


def run_collector(tmp_path, contents, *, framework="torch", required=(11,)):
    arguments = [sys.executable, "-B", str(SCRIPT), "--framework", framework]
    paths = []
    for index, content in enumerate(contents):
        path = tmp_path / ("source-%d.log" % index)
        path.write_bytes(content.encode())
        paths.append(path)
        arguments += ["--log", str(path)]
    for pid in required:
        arguments += ["--require-pid", str(pid)]
    output = tmp_path / "result.json"
    arguments += ["--output", str(output)]
    result = subprocess.run(arguments, text=True, capture_output=True, timeout=10)
    return result, output, paths


@pytest.mark.parametrize("framework", ("torch", "jittor"))
def test_collects_explicit_logs_pid_coverage_hashes_and_line_provenance(tmp_path, framework):
    line = MARKER + json.dumps(record(framework=framework)) + "\n"
    result, output, paths = run_collector(tmp_path, ["ordinary line\n" + line,
        "\x1b[36m(Worker pid=11)\x1b[0m " + line + MARKER + json.dumps(record(22, framework)) + "\n"],
        framework=framework, required=(11, 22))
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["schema"] == "verl-worker-startup-precision-v1"
    assert report["status"] == "startup_precision_observed"
    assert report["required_pids"] == [11, 22] and report["observed_pids"] == [11, 22]
    assert report["unproven"] == ["computation_thread_precision", "device_execution", "backend_fallback", "performance", "L5"]
    assert report["logs"] == [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                              for path in paths]
    assert report["workers"][0]["observations"] == [{"log_index": 0, "line": 2}, {"log_index": 1, "line": 1}]
    assert report["workers"][1]["observations"] == [{"log_index": 1, "line": 2}]


@pytest.mark.parametrize("field,value", (("cuda_matmul_allow_tf32", True), ("cuda_matmul_allow_tf32", 0),
    ("cudnn_allow_tf32", "false"), ("cudnn_benchmark", None), ("float32_matmul_precision", "high")))
def test_rejects_invalid_or_nonboolean_readback(tmp_path, field, value):
    item = record()
    item["runtime_conditions"][field] = value
    result, output, _ = run_collector(tmp_path, [MARKER + json.dumps(item)])
    assert result.returncode != 0
    assert "precision" in result.stderr.lower()
    assert not output.exists()


@pytest.mark.parametrize("field,value", (("pid", 0), ("pid", -1), ("pid", True), ("pid", "11"),
    ("framework", "jittor"), ("scope", "full_worker_L5"), ("runtime_conditions", [])))
def test_rejects_invalid_marker_schema(tmp_path, field, value):
    item = record()
    item[field] = value
    result, output, _ = run_collector(tmp_path, [MARKER + json.dumps(item)])
    assert result.returncode != 0
    assert "marker" in result.stderr.lower()
    assert not output.exists()


@pytest.mark.parametrize("mutation", ("missing", "extra", "missing_condition", "extra_condition"))
def test_rejects_marker_schema_drift(tmp_path, mutation):
    item = record()
    if mutation == "missing":
        del item["scope"]
    elif mutation == "extra":
        item["meets_target"] = True
    elif mutation == "missing_condition":
        del item["runtime_conditions"]["cudnn_benchmark"]
    else:
        item["runtime_conditions"]["backend_fallback_count"] = 0
    result, output, _ = run_collector(tmp_path, [MARKER + json.dumps(item)])
    assert result.returncode != 0
    assert "schema" in result.stderr.lower()
    assert not output.exists()


def test_missing_required_pid_cannot_be_replaced_by_duplicate_driver_marker(tmp_path):
    line = MARKER + json.dumps(record()) + "\n"
    result, output, _ = run_collector(tmp_path, [line * 2], required=(11, 22))
    assert result.returncode != 0
    assert "missing required PIDs: 22" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("bad_line", (MARKER + "not-json", MARKER + "[]", MARKER + "{}",
    MARKER + '{"pid":11,"pid":22}', MARKER + json.dumps(record()) + " [repeated 2x]"))
def test_malformed_exact_marker_fails_closed(tmp_path, bad_line):
    result, output, _ = run_collector(tmp_path, [bad_line])
    assert result.returncode != 0
    assert "marker" in result.stderr.lower()
    assert not output.exists()


def test_similar_marker_is_not_evidence(tmp_path):
    result, output, _ = run_collector(tmp_path, ["NOT_" + MARKER + json.dumps(record())])
    assert result.returncode != 0
    assert "no precision markers" in result.stderr
    assert not output.exists()


def test_conflicting_duplicate_readback_fails_closed(tmp_path):
    changed = record()
    changed["runtime_conditions"]["cudnn_benchmark"] = True
    result, output, _ = run_collector(tmp_path, [MARKER + json.dumps(record()) + "\n" + MARKER + json.dumps(changed)])
    assert result.returncode != 0
    assert not output.exists()


def test_ray_prefix_pid_mismatch_fails_closed(tmp_path):
    result, output, _ = run_collector(tmp_path, ["(Worker pid=22) " + MARKER + json.dumps(record())])
    assert result.returncode != 0
    assert "PID" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("pid", (0, -1, "bad"))
def test_cli_rejects_invalid_required_pid(tmp_path, pid):
    result, output, _ = run_collector(tmp_path, [MARKER + json.dumps(record())], required=(pid,))
    assert result.returncode != 0
    assert not output.exists()


def test_optional_pid_coverage_does_not_claim_all_workers(tmp_path):
    result, output, _ = run_collector(tmp_path, [MARKER + json.dumps(record())], required=())
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["required_pid_coverage"] == "not_requested"
    assert report["required_pids"] == []
    assert "L5" in report["unproven"]


@pytest.mark.parametrize("invalid", ("missing_log", "invalid_utf8", "existing_output", "output_is_log"))
def test_io_failures_do_not_overwrite_existing_evidence_or_logs(tmp_path, invalid):
    source = tmp_path / "input.log"
    original = (MARKER + json.dumps(record())).encode()
    if invalid != "missing_log":
        source.write_bytes(b"\xff" if invalid == "invalid_utf8" else original)
    output = source if invalid == "output_is_log" else tmp_path / "result.json"
    if invalid == "existing_output":
        output.write_text("older-evidence\n")
    before = output.read_bytes() if output.exists() else None
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--log", str(source),
                             "--framework", "torch", "--require-pid", "11", "--output", str(output)],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    if before is not None:
        assert output.read_bytes() == before
    else:
        assert not output.exists()


def test_duplicate_required_pids_are_reported_once(tmp_path):
    result, output, _ = run_collector(tmp_path, [MARKER + json.dumps(record())], required=(11, 11))
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["required_pids"] == [11]
