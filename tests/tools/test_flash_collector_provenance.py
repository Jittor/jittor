"""Historical measurement roots cannot select historical admission guards."""

import hashlib
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path("agent/skills/flash-attention-torch-compat/scripts")
RUNBOOK = ROOT / "agent/skills/torch-compat-repo-runbook/scripts"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def collector():
    return load_module("flash_provenance_collector_test", ROOT / SCRIPT_DIR / "collect_scorecard.py")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def measured_root(tmp_path):
    root = tmp_path / "measured"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Provenance fixture")
    git(root, "config", "user.email", "provenance@example.invalid")
    probe = root / SCRIPT_DIR / "flash_training_probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text("MEASURED_PROBE = 1\n", encoding="utf-8")
    # Importing either historical admission tool is a real test failure.
    for name in ("compare_training.py", "collect_scorecard.py"):
        (probe.parent / name).write_text(
            "raise RuntimeError('historical admission guard executed')\n", encoding="utf-8"
        )
    git(root, "add", ".")
    git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "Freeze measured fixture")
    return root, git(root, "rev-parse", "HEAD"), hashlib.sha256(probe.read_bytes()).hexdigest()


def test_probe_hash_is_bound_to_full_frozen_commit(collector, measured_root):
    root, baseline, expected = measured_root
    assert collector.measured_probe_hash(root, baseline) == expected
    # Untracked outputs do not mutate the tracked measurement source.
    (root / "untracked-note.txt").write_text("not a measured source file\n", encoding="utf-8")
    assert collector.measured_probe_hash(root, baseline) == expected


@pytest.mark.parametrize("change", ("short-sha", "symbolic-ref", "unknown-sha", "new-head", "unstaged", "staged"))
def test_nonfrozen_measurement_root_is_rejected(collector, measured_root, change):
    root, baseline, _ = measured_root
    probe = root / SCRIPT_DIR / "flash_training_probe.py"
    if change == "short-sha":
        baseline = baseline[:12]
    elif change == "symbolic-ref":
        baseline = "HEAD"
    elif change == "unknown-sha":
        baseline = "0" * 40
    elif change == "new-head":
        git(root, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "Different commit")
    else:
        probe.write_text("MEASURED_PROBE = 2\n", encoding="utf-8")
        if change == "staged":
            git(root, "add", str(probe.relative_to(root)))
    with pytest.raises(ValueError):
        collector.measured_probe_hash(root, baseline)


@pytest.mark.parametrize("module_name", ("_flash_attention_scorecard", "_accelerate_scorecard", "_deepspeed_scorecard"))
def test_only_flash_protocol_receives_measurement_root(monkeypatch, tmp_path, module_name):
    dispatcher = load_module("protocol_provenance_dispatch_test", RUNBOOK / "_protocol_scorecards.py")
    measured = tmp_path / "explicit-measured-root"
    record, artifacts, source = {}, [], {}
    calls = []
    if module_name == "_flash_attention_scorecard":
        def validate(actual_record, actual_artifacts, actual_source, *, root):
            calls.append((actual_record, actual_artifacts, actual_source, root))
    else:
        # The old three-argument contract must remain sufficient for other protocols.
        def validate(actual_record, actual_artifacts, actual_source):
            calls.append((actual_record, actual_artifacts, actual_source))
    module = SimpleNamespace(__name__=module_name, validate_measurements=validate)
    monkeypatch.setattr(dispatcher, "validator", lambda *_: module)
    dispatcher.validate_measurements(record, artifacts, source, root=measured)
    expected = (record, artifacts, source, measured) if module_name == "_flash_attention_scorecard" else (record, artifacts, source)
    assert calls == [expected]


def test_historical_root_supplies_source_not_executable_guards(monkeypatch, measured_root, tmp_path):
    monkeypatch.syspath_prepend(str(RUNBOOK))
    proof = load_module("flash_provenance_proof_test", RUNBOOK / "_flash_attention_scorecard.py")
    monkeypatch.setattr(proof, "validate_spec", lambda *_: None)
    root, baseline, expected_hash = measured_root
    original_spec = importlib.util.spec_from_file_location
    loaded_paths = []

    class ReachedCurrentCollector(Exception):
        pass

    def current_case(directory, source, actual_baseline, probe_hash, comparison, report, command, output, *, root):
        assert root == measured_root[0]
        assert actual_baseline == baseline and probe_hash == expected_hash
        assert Path(comparison.__file__).resolve() == ROOT / SCRIPT_DIR / "compare_training.py"
        raise ReachedCurrentCollector

    def recording_spec(name, path, *args, **kwargs):
        path = Path(path).resolve()
        loaded_paths.append(path)
        spec = original_spec(name, path, *args, **kwargs)
        if path == ROOT / SCRIPT_DIR / "collect_scorecard.py":
            execute = spec.loader.exec_module

            def execute_current(module):
                execute(module)
                module.collect_case = current_case

            spec.loader.exec_module = execute_current
        return spec

    monkeypatch.setattr(proof.importlib.util, "spec_from_file_location", recording_spec)
    record = {"jittor_commit": baseline, "evidence": {"report": "fixture", "command": "fixture"}}
    artifacts = [{"role": role, "path": str(tmp_path / (role + ".npz"))} for role in ("candidate", "reference")]
    with pytest.raises(ReachedCurrentCollector):
        proof.validate_measurements(record, artifacts, {}, root=root)
    assert loaded_paths == [ROOT / SCRIPT_DIR / "collect_scorecard.py", ROOT / SCRIPT_DIR / "compare_training.py"]
