"""Host contracts for resource admission between paired NPU calls."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "benchmarks/profiling/paired_benchmark_control.py"


def load_controller():
    spec = importlib.util.spec_from_file_location("paired_resource_contract", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("problem", ["headroom", "foreign", "identity"])
def test_live_guard_rejects_invalid_resources(monkeypatch, tmp_path, problem):
    m = load_controller()
    ids = [101, 202, 999] if problem == "foreign" else [101, 202]
    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="snapshot", stderr=""),
    )
    monkeypatch.setattr(
        m,
        "_device_snapshot",
        lambda raw, physical: {
            "free_mib": 20000,
            "process_memory_mib": {pid: 15000 for pid in ids},
        },
    )
    monkeypatch.setattr(
        m, "_start_ticks", lambda pid: "reused" if problem == "identity" else "original"
    )
    spec = {
        "physical_device": 5,
        "workers": {
            name: {"env": {"ASCEND_RT_VISIBLE_DEVICES": "5"}} for name in ("torch", "jittor")
        },
    }
    workers = {
        name: SimpleNamespace(process=SimpleNamespace(pid=pid), start_ticks="original")
        for name, pid in [("torch", 101), ("jittor", 202)]
    }
    message = {
        "headroom": "insufficient HBM",
        "foreign": "processes differ",
        "identity": "identity changed",
    }[problem]
    with pytest.raises(RuntimeError, match=message):
        m._guard_device(spec, workers, tmp_path, "check", startup_headroom=problem == "headroom")
    if problem == "headroom":
        assert (
            m._guard_device(spec, workers, tmp_path, "resident", startup_headroom=False)["free_mib"]
            == 20000
        )


@pytest.mark.parametrize("reject", [False, True])
def test_every_pair_checks_resource_before_admission_and_after_completion(
    monkeypatch, tmp_path, reject
):
    m = load_controller()
    stages, admissions, stopped = [], [], []
    event = {"call": 1, "phase": "prefill", "operation": "prefill", "slot": 0, "use_cache": True}

    class Worker:
        def __init__(self, name, spec, output):
            self.name, self.calls, self.alive = name, 0, True
            self.process = SimpleNamespace(
                pid=101 if name == "torch" else 202, poll=lambda: None if self.alive else 0
            )

        def next(self, deadline):
            self.calls += 1
            if self.calls == 1:
                return "ready", event.copy()
            self.alive = False
            return "exit", 0

        def admit(self, call):
            admissions.append((self.name, call))

        def stop(self):
            stopped.append(self.name)

        def provenance(self):
            return {"pid": self.process.pid, "returncode": self.process.poll()}

    def guard(spec, workers, output, stage, *, startup_headroom=True):
        stages.append((stage, sorted(workers), startup_headroom))
        if reject and stage == "before-pair-1":
            raise RuntimeError("foreign task appeared")
        return {"stage": stage}

    monkeypatch.setattr(m, "Worker", Worker)
    monkeypatch.setattr(m, "_guard_device", guard)
    output = tmp_path / "pair"
    if reject:
        with pytest.raises(RuntimeError, match="foreign task"):
            m.run_pair({"workers": {"torch": {}, "jittor": {}}}, output, 5)
        result = json.loads((output / "execution.json").read_text())
        assert not result["passed"] and not result["performance_accepted"]
        assert admissions == []
    else:
        result = m.run_pair({"workers": {"torch": {}, "jittor": {}}}, output, 5)
        assert result["passed"]
        assert stages == [
            ("before-torch", [], True),
            ("before-jittor", ["torch"], True),
            ("before-pair-1", ["jittor", "torch"], False),
            ("after-pair-1", [], False),
        ]
        assert admissions == [("torch", 1), ("jittor", 1)]
    assert stopped == ["torch", "jittor"]


@pytest.mark.parametrize("identity_matches", [False, True])
def test_cleanup_only_signals_original_owned_process(monkeypatch, identity_matches):
    m = load_controller()
    signals = []
    instance = m.Worker.__new__(m.Worker)
    instance.process = SimpleNamespace(pid=123, poll=lambda: None, wait=lambda **kwargs: 0)
    instance.start_ticks = "original"
    instance.reader = SimpleNamespace(join=lambda **kwargs: None)
    monkeypatch.setattr(m, "_start_ticks", lambda pid: "original" if identity_matches else "reused")
    monkeypatch.setattr(m.os, "killpg", lambda pid, signal: signals.append(pid))
    instance.stop()
    assert signals == ([123] if identity_matches else [])
