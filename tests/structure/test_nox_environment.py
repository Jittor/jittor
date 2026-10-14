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
    posargs = ()

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


def test_nccl_warms_serially_then_launches_native_and_torch_modes(monkeypatch, tmp_path):
    monkeypatch.setenv("nvcc_path", "/opt/cuda/bin/nvcc")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    monkeypatch.setenv("JITTOR_NCCL_WORLD_SIZE", "2")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/usr/bin/python3-config")
    module["nccl"](session)
    warm = [(i, args, kw) for i, (args, kw) in enumerate(session.calls)
            if "-c" in args and "probe_library('nccl', load=True)" in str(args)]
    launch = [(i, args, kw) for i, (args, kw) in enumerate(session.calls)
              if any(str(arg).endswith("distributed/launch.py") for arg in args)]
    assert len(warm) == len(launch) == 2
    assert max(i for i, _, _ in warm) < min(i for i, _, _ in launch)
    assert [kw["env"]["CUDA_VISIBLE_DEVICES"] for _, _, kw in warm] == ["3", "5"]
    assert len({kw["env"]["cache_name"] for _, _, kw in warm}) == 2
    assert all(kw["env"]["JITTOR_TORCH_SHIM"] == "0" for _, _, kw in warm)
    assert [kw["env"]["JITTOR_TORCH_SHIM"] for _, _, kw in launch] == ["0", "1"]
    assert module["NCCL_TESTS"][0] in launch[0][1]
    assert module["NCCL_TESTS"][1] in launch[1][1]
    assert module["NCCL_TESTS"][1] not in launch[0][1]
    logs = [args[args.index("--logdir") + 1] for _, args, _ in launch]
    assert len(set(logs)) == 2


def test_nccl_target_overrides_preserve_pytest_flags_and_choose_their_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("nvcc_path", "/opt/cuda/bin/nvcc")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    monkeypatch.setenv("JITTOR_NCCL_WORLD_SIZE", "2")
    module = _load_noxfile(monkeypatch, tmp_path)
    for mode, target in enumerate(module["NCCL_TESTS"]):
        session = _FakeSession(tmp_path, "/usr/bin/python3-config")
        session.posargs = (target + "::selected_case", "-k", "selection", "-q")
        module["nccl"](session)
        launch = [(args, kw) for args, kw in session.calls
                  if any(str(arg).endswith("distributed/launch.py") for arg in args)]
        assert len(launch) == 1
        args, kw = launch[0]
        assert args[-4:] == session.posargs
        assert kw["env"]["JITTOR_TORCH_SHIM"] == str(mode)


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
    """Every thread-pool variable gets the shared per-worker budget.

    The expected value comes from ``_helpers.tiers``, which is where the
    budget is defined and what ``noxfile._split_threads`` delegates to. It
    used to be read out of the noxfile's own namespace, which stopped
    containing the name once the noxfile narrowed its import to
    ``apply_worker_thread_budget`` -- so the check failed with a ``KeyError``
    on its own expectation and never reached the split it exists to measure.
    """
    from _helpers.tiers import worker_thread_budget

    module = _load_noxfile(monkeypatch, tmp_path)
    env = {name: "99" for name in module["_THREAD_ENV_NAMES"]}

    split = module["_split_threads"](env, workers=4)

    expected = str(worker_thread_budget(4))
    assert {split[name] for name in module["_THREAD_ENV_NAMES"]} == {expected}
    assert expected != "99"


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


def test_verl_gate_keeps_cpu_explicit_and_uses_real_registered_test(monkeypatch, tmp_path):
    oracle = tmp_path / "oracle-python"
    oracle.touch()
    monkeypatch.setenv("REAL_TORCH_PYTHON", str(oracle))
    monkeypatch.setenv("JITTOR_CI_PYTHON", "/preprovisioned/shim-python")
    monkeypatch.setenv("JITTOR_TEST_DEVICES", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "allocated-but-not-for-cpu")
    monkeypatch.setenv("nvcc_path", "/cuda/bin/nvcc")
    monkeypatch.setenv("JITTOR_ECOSYSTEM_SPEED_RATIO", "1.01")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/usr/bin/python3-config")
    session.python = False
    module["ecosystem_verl"](session)
    launches = [(args, kw) for args, kw in session.calls if "pytest" in args]
    assert len(launches) == 1
    args, kw = launches[0]
    assert module["VERL_COMPONENT_TESTS"][0] in args
    assert args[0] == "/preprovisioned/shim-python"
    env = kw["env"]
    assert env["JITTOR_TEST_DEVICES"] == "cpu"
    assert env["CUDA_VISIBLE_DEVICES"] == env["nvcc_path"] == env["JT_BUILD_NVCC_PATH"] == ""
    assert env["JT_USE_CUDA"] == env["use_cuda"] == "0"
    assert env["JITTOR_REQUIRE_REAL_TORCH"] == env["JITTOR_TEST_REQUIRE_EXECUTION"] == "1"
    assert env["backend_fallback"] == "error"
    assert env["JITTOR_ECOSYSTEM_SPEED_RATIO"] == ""
    assert env["JITTOR_REQUIRE_VERL"] == "1"
    assert env["cache_name"] == "nox_ecosystem_verl_cpu"
    assert "verl-jittor-cache" in env["JITTOR_HOME"]
    oracle_calls = [(args, kw) for args, kw in session.calls if args[0] == str(oracle)]
    assert len(oracle_calls) == 1
    assert oracle_calls[0][1]["env"]["JITTOR_TORCH_SHIM"] == "0"
    assert oracle_calls[0][1]["env"]["PYTHONPATH"] == ""
    code = oracle_calls[0][0][-1]
    assert "Version('2.0')" in code and "Version('5.5.3')" in code
    assert "_torch_compat_install_context" in code


def test_verl_gate_rejects_missing_oracle_and_implicit_cuda(monkeypatch, tmp_path):
    import pytest

    monkeypatch.delenv("REAL_TORCH_PYTHON", raising=False)
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/usr/bin/python3-config")
    with pytest.raises(AssertionError, match="independent PyTorch"):
        module["ecosystem_verl"](session)
    oracle = tmp_path / "oracle-python"
    oracle.touch()
    monkeypatch.setenv("REAL_TORCH_PYTHON", str(oracle))
    monkeypatch.setenv("JITTOR_TEST_DEVICES", "cuda")
    monkeypatch.setenv("nvcc_path", "/cuda/bin/nvcc")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(AssertionError, match="explicitly allocated"):
        module["ecosystem_verl"](session)
    assert not any("pytest" in args for args, _kw in session.calls)


def test_verl_gate_cuda_cannot_silently_select_cpu(monkeypatch, tmp_path):
    oracle = tmp_path / "oracle-python"
    oracle.touch()
    monkeypatch.setenv("REAL_TORCH_PYTHON", str(oracle))
    monkeypatch.setenv("JITTOR_TEST_DEVICES", "cuda")
    monkeypatch.setenv("nvcc_path", "/cuda/bin/nvcc")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "allocated")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/usr/bin/python3-config")
    module["ecosystem_verl"](session)
    args, kw = next((args, kw) for args, kw in session.calls if "pytest" in args)
    assert kw["env"]["JITTOR_TEST_DEVICES"] == "cuda"
    assert kw["env"]["JT_USE_CUDA"] == kw["env"]["use_cuda"] == "1"
    assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "allocated"
    assert kw["env"]["JT_BUILD_NVCC_PATH"] == "/cuda/bin/nvcc"
    assert kw["env"]["cache_name"] == "nox_ecosystem_verl_cuda"
    oracle_call = next(args for args, _kw in session.calls if args[0] == str(oracle))
    assert "assert torch.cuda.is_available()" in oracle_call[-1]
