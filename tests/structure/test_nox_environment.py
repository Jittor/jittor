import os
import runpy
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeOptions:
    pass


class _FakeSession:
    python = "3.11"

    def __init__(self, root, python_config):
        self.root = root
        self.python_config = python_config
        self.calls = []
        self.logs = []

    def create_tmp(self):
        return str(self.root / "session-tmp")

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.python_config + "\n"

    def error(self, message):
        raise AssertionError(message)

    def log(self, message):
        self.logs.append(message)


def _load_noxfile(monkeypatch, tmp_path):
    fake_nox = types.ModuleType("nox")
    fake_nox.options = _FakeOptions()
    fake_nox.session = lambda *args, **kwargs: lambda function: function
    monkeypatch.setitem(sys.modules, "nox", fake_nox)
    monkeypatch.setenv("JITTOR_LAB_ROOT", str(tmp_path / "lab"))
    return runpy.run_path(str(REPO_ROOT / "noxfile.py"), run_name="jittor_noxfile")


def test_session_env_uses_the_session_interpreters_python_config(monkeypatch, tmp_path):
    expected = "/opt/python-3.11/bin/python3.11-config"
    monkeypatch.setenv("python_config_path", "/opt/python-3.12/bin/python3.12-config")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, expected)

    _root, env = module["_session_env"](session, "structure")

    assert env["python_config_path"] == expected
    assert session.calls[0][0][:2] == ("python", "-c")


def test_sessions_share_one_jittor_cache(monkeypatch, tmp_path):
    """A session must not start from an empty Jittor cache.

    Every session used to get its own JITTOR_HOME under the scratch directory
    it wipes on entry, so each one rebuilt the C++ core and every operator and
    re-downloaded MKL, cub, cutt and NCCL first. Nothing about that was under
    test: the build is a function of the build configuration, which the cache
    path already partitions by.
    """
    monkeypatch.delenv("JITTOR_NOX_SHARED_CACHE", raising=False)
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/usr/bin/python3-config")

    root, first = module["_session_env"](session, "structure")
    _root, second = module["_session_env"](session, "py312")

    assert first["JITTOR_HOME"] == second["JITTOR_HOME"]
    assert Path(first["JITTOR_HOME"]) == module["NOX_JITTOR_CACHE"]
    assert root not in Path(first["JITTOR_HOME"]).parents
    # The scratch directories stay under the session root; only the cache
    # moves out of it.
    for name in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "JITTOR_TEST_STATE_ROOT"):
        assert Path(first[name]).is_relative_to(root), name


def test_the_shared_cache_can_be_turned_off(monkeypatch, tmp_path):
    monkeypatch.setenv("JITTOR_NOX_SHARED_CACHE", "0")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/usr/bin/python3-config")

    root, env = module["_session_env"](session, "structure")

    assert Path(env["JITTOR_HOME"]).is_relative_to(root)


def test_a_populated_mirror_is_offered_to_the_session(monkeypatch, tmp_path):
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/usr/bin/python3-config")

    _root, env = module["_session_env"](session, "structure")
    assert "JITTOR_OFFLINE_PATH" not in env

    module["NOX_JITTOR_ASSETS"].mkdir(parents=True, exist_ok=True)
    _root, env = module["_session_env"](session, "structure")
    assert Path(env["JITTOR_OFFLINE_PATH"]) == module["NOX_JITTOR_ASSETS"]


def test_session_env_clears_an_inherited_config_when_the_helper_is_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("python_config_path", "/opt/python-3.12/bin/python3.12-config")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "")

    _root, env = module["_session_env"](session, "py37")

    assert env["python_config_path"] is None


def test_hardware_session_uses_the_external_interpreters_python_config(monkeypatch, tmp_path):
    expected = "/opt/hardware-python/bin/python3.10-config"
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, expected)
    env = {"python_config_path": "/opt/host-python/bin/python3.12-config"}

    module["_set_hardware_python_config"](session, "/opt/hardware-python/bin/python", env)

    assert env["python_config_path"] == expected
    args, kwargs = session.calls[0]
    assert args[:2] == ("/opt/hardware-python/bin/python", "-c")
    assert kwargs["external"] is True


def test_session_env_blocks_host_test_controls(monkeypatch, tmp_path):
    monkeypatch.setenv("OMP_NUM_THREADS", "999")
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    monkeypatch.setenv("MKL_DYNAMIC", "true")
    monkeypatch.setenv("HOST_ONLY_TEST_CONTROL", "must-not-leak")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/usr/bin/python3-config")

    _root, env = module["_session_env"](session, "cpu")

    assert env["HOST_ONLY_TEST_CONTROL"] is None
    assert env["OMP_NUM_THREADS"] != "999"
    assert int(env["OMP_NUM_THREADS"]) > 0
    assert env["OMP_PROC_BIND"] == "false"
    assert env["OMP_DYNAMIC"] == "false"
    assert env["MKL_DYNAMIC"] == "false"
    assert env["PATH"] == os.environ["PATH"]


def test_session_env_records_the_exact_cpu_affinity(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {7, 3, 5})
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/usr/bin/python3-config")

    _root, env = module["_session_env"](session, "cpu")

    assert env["JITTOR_GATE_CPU_AFFINITY"] == "3,5,7"
    probe_env = session.calls[0][1]["env"]
    assert probe_env is env


def test_worker_split_updates_every_thread_pool(monkeypatch, tmp_path):
    module = _load_noxfile(monkeypatch, tmp_path)
    env = {name: "99" for name in module["_THREAD_ENV_NAMES"]}

    split = module["_split_threads"](env, workers=4)

    expected = str(module["worker_thread_budget"](4))
    assert {split[name] for name in module["_THREAD_ENV_NAMES"]} == {expected}


def test_gate_workers_respect_a_smaller_cgroup_quota(monkeypatch, tmp_path):
    """xdist must not create idle workers beyond the CPU quota."""
    module = _load_noxfile(monkeypatch, tmp_path)
    module["_runtime_gate_workers"].__globals__["GATE_WORKERS"] = 4
    module["_runtime_gate_workers"].__globals__["effective_cpu_count"] = lambda: 1

    assert module["_runtime_gate_workers"]() == 1


def test_gate_workers_keep_the_configured_count_when_quota_is_sufficient(
        monkeypatch, tmp_path):
    module = _load_noxfile(monkeypatch, tmp_path)
    module["_runtime_gate_workers"].__globals__["GATE_WORKERS"] = 4
    module["_runtime_gate_workers"].__globals__["effective_cpu_count"] = lambda: 8

    assert module["_runtime_gate_workers"]() == 4


def test_smoke_budget_log_reports_actual_and_configured_workers(
        monkeypatch, tmp_path):
    module = _load_noxfile(monkeypatch, tmp_path)
    module["_enforce_smoke_budget"].__globals__["GATE_WORKERS"] = 4
    module["_enforce_smoke_budget"].__globals__["budget_report"] = (
        lambda workers, configured_workers: {
            "predicted_seconds": 120.0,
            "budget_seconds": 480.0,
            "headroom_seconds": 360.0,
            "workers": workers,
            "configured_workers": configured_workers,
            "effective_cpus": 1,
            "threads_per_worker": 1,
        })
    session = _FakeSession(tmp_path, "/usr/bin/python3-config")

    module["_enforce_smoke_budget"](session, workers=1)

    assert session.logs == [
        "smoke budget: predicted 120s / 480s (headroom 360s; 1 actual/4 "
        "configured workers; 1 CPU quota; 1 threads/worker)"
    ]
