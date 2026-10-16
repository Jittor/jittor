"""Optional CPU Trainer resume gate, orchestrated without importing either runtime.

Set REAL_TORCH_PYTHON and JITTOR_STATEFUL_PYTHON to separately prepared Python
environments with matching Transformers/Accelerate/PEFT/Tokenizers/torchdata.
JITTOR_REQUIRE_TRAINER_STATEFUL=1 makes missing environments/dependencies an error.
Compiler and shared-library environment settings are inherited from the caller.
No packages, checkpoints or datasets are downloaded. Every training phase runs
in a fresh subprocess; artifacts and isolated caches live below JITTOR_LAB_ROOT.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "compat/tests/torch/_hf_trainer_resume_runner.py"
REQUIRE_ENV = "JITTOR_REQUIRE_TRAINER_STATEFUL"
PYTHON_ENV = {"pytorch": "REAL_TORCH_PYTHON", "jittor": "JITTOR_STATEFUL_PYTHON"}
PACKAGES = ("transformers", "accelerate", "peft", "tokenizers", "numpy", "torchdata")
PINNED = {"transformers": "4.56.2", "accelerate": "1.10.1"}
PROBE = """
import importlib.metadata as metadata
import json
versions = {}
for name in %r:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps(versions, sort_keys=True))
""" % (PACKAGES + ("torch",),)


def _unavailable(reason, environ):
    message = "Trainer stateful integration: " + reason
    if environ.get(REQUIRE_ENV) == "1":
        pytest.fail("%s=1 requires this gate: %s" % (REQUIRE_ENV, message), pytrace=False)
    pytest.skip(message)


def _configured_pythons(environ):
    missing = [name for name in PYTHON_ENV.values() if not environ.get(name, "").strip()]
    if missing:
        _unavailable("missing independent torch oracle/runtime configuration: " + ", ".join(missing), environ)
    result = {}
    for runtime, name in PYTHON_ENV.items():
        value = os.path.expanduser(environ[name].strip())
        executable = shutil.which(value)
        if executable is None:
            _unavailable("missing torch runtime executable %s=%s" % (name, value), environ)
        # Preserve a venv's Python symlink; resolving it would discard its prefix.
        result[runtime] = os.path.abspath(executable)
    return result


def _new_run_root(environ):
    lab = Path(environ.get("JITTOR_LAB_ROOT", str(ROOT.parent / "jittor-lab"))).expanduser().resolve()
    if lab == ROOT or ROOT in lab.parents:
        raise ValueError("JITTOR_LAB_ROOT must keep integration artifacts outside the repository")
    # multiprocessing's resource-sharer uses Unix sockets under TMPDIR. Keep
    # the topic name short enough for Linux's 108-byte AF_UNIX address limit.
    parent = lab / "_state" / "trainer-resume"
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="cpu-", dir=str(parent)))


def _runtime_environment(runtime, run_root, inherited):
    env = dict(inherited)
    state = run_root / runtime
    locations = {
        "HOME": "home", "JITTOR_HOME": "jittor-home", "TMPDIR": "tmp",
        "XDG_CACHE_HOME": "xdg-cache", "TORCH_HOME": "torch-home",
        "CUDA_CACHE_PATH": "cuda-cache", "HF_HOME": "hf-home",
        "JITTOR_TORCH_RUNTIME_ROOT": "torch-shim",
    }
    for variable, child in locations.items():
        path = state / child
        path.mkdir(parents=True, exist_ok=True)
        env[variable] = str(path)
    env.update({
        "CUDA_VISIBLE_DEVICES": "", "JITTOR_TORCH_SHIM": "1" if runtime == "jittor" else "0",
        "JITTOR_TORCH_STRICT_BOOTSTRAP": "1" if runtime == "jittor" else "0",
        "JITTOR_TORCH_PROJECT_ROOT": str(ROOT), "JITTOR_REPO_ROOT": str(ROOT),
        "JITTOR_LAB_ROOT": str(run_root.parents[2]),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "JT_BUILD_USE_MPI": "0", "JT_BUILD_USE_NCCL": "0",
        "JT_BUILD_NVCC_PATH": "", "nvcc_path": "", "JITTOR_CUDA": "0", "JT_BUILD_USE_CUTT": "0",
        "cache_name": "trainer_stateful_%s_%s" % (run_root.name, runtime),
    })
    env.pop("use_cuda", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "adapters"), str(ROOT / "python")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def _invoke(python, arguments, env, log_path, timeout=1200):
    command = [python, "-u", str(RUNNER)] + [str(value) for value in arguments]
    with log_path.open("w") as handle:
        process = subprocess.run(command, cwd=str(ROOT), env=env, stdout=handle,
                                 stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if process.returncode:
        raise AssertionError("Trainer stateful subprocess failed (rc=%d); inspect %s" % (
            process.returncode, log_path))
    return log_path.read_text()


def _dependency_versions(python, env, log_path, required):
    # -I excludes source-path startup hooks: this probe must not import Jittor.
    probe_env = dict(env, JITTOR_TORCH_SHIM="0", JITTOR_TORCH_STRICT_BOOTSTRAP="0")
    result = subprocess.run([python, "-I", "-c", PROBE], cwd=str(ROOT), env=probe_env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=60, check=False)
    log_path.write_text(result.stdout)
    if result.returncode:
        raise AssertionError("Trainer stateful dependency probe failed; inspect %s" % log_path)
    versions = json.loads(result.stdout)
    missing = [name for name in required if not versions.get(name)]
    if missing:
        _unavailable("missing optional torch ecosystem dependencies: " + ", ".join(missing), env)
    for name, expected in PINNED.items():
        if versions[name].split("+", 1)[0] != expected:
            raise AssertionError("Trainer stateful gate requires %s==%s, found %s" % (name, expected, versions[name]))
    return versions


def _assert_report(path, runtime, final_step, restored_step, microbatches):
    report = json.loads(path.read_text())
    expected = {
        "runtime": runtime, "device": "cpu", "global_step": final_step,
        "stateful_dataloader": True, "stateful_restored_step": restored_step,
        "custom_generator_states_in_trainer_checkpoint": True,
        "fallback_count": 0, "dataloader_num_workers": 2,
        "sampler": "random", "worker_random_transform": True, "microbatches": microbatches,
    }
    for name, value in expected.items():
        assert report[name] == value, (path, name, report[name], value)
    assert report["actual_parameter_devices"] == ["cpu"], report["actual_parameter_devices"]
    assert report["stateful_saved_checkpoints"], "training did not save loader state"
    for checkpoint in report["stateful_saved_checkpoints"]:
        assert Path(checkpoint).is_file(), checkpoint
    return report


def _exercise_runtime(runtime, python, env, run_root, weights):
    destination = run_root / runtime
    continuous = destination / "continuous"
    first = destination / "first"
    resumed = destination / "resumed"
    common = ["run", "--runtime", runtime, "--device", "cpu", "--weights", weights,
              "--max-steps", "4", "--sampler", "random", "--dropout", "0.1",
              "--dataset-size", "32", "--dataloader-num-workers", "2",
              "--dataloader-prefetch-factor", "2", "--worker-random-transform",
              "--explicit-generators", "--stateful-dataloader"]
    _invoke(python, common + ["--output-dir", continuous], env, destination / "continuous.log")
    _invoke(python, common + ["--output-dir", first, "--stop-after", "2"], env, destination / "first.log")
    checkpoint = first / "checkpoints" / "checkpoint-2"
    assert (checkpoint / "stateful_dataloader.pt").is_file(), checkpoint
    _invoke(python, common + ["--output-dir", resumed, "--resume-checkpoint", checkpoint,
                             "--start-step", "2"], env, destination / "resumed.log")
    reports = {
        "continuous": _assert_report(continuous / "report.json", runtime, 4, None, 8),
        "first": _assert_report(first / "report.json", runtime, 2, None, 4),
        "resumed": _assert_report(resumed / "report.json", runtime, 4, 2, 4),
    }
    merged = destination / "resumed-merged.npz"
    _invoke(python, ["merge", "--first", first / "trajectory.npz", "--resumed", resumed / "trajectory.npz",
                     "--output", merged], env, destination / "merge.log")
    output = _invoke(python, ["compare", "--left", continuous / "trajectory.npz", "--right", merged,
                             "--atol", "0", "--rtol", "0"], env, destination / "compare.log")
    comparison = json.loads(output.strip().splitlines()[-1])
    assert comparison["failure_count"] == 0, comparison
    assert comparison["items"] > 0, "empty trajectories cannot establish resume correctness"
    generator_hashes = {
        name: {phase: report[name] for phase, report in reports.items()}
        for name in ("sampler_generator_final_state_sha256", "dataloader_generator_final_state_sha256")
    }
    for name, hashes in generator_hashes.items():
        assert hashes["continuous"] is not None, (runtime, name)
    # Native torchdata reconstructs its worker iterator and consumes an extra
    # loader-generator draw even after restoring base_seed. The supported first
    # epoch contract is the trajectory, not subsequent-epoch generator equality.
    sampler_hashes = generator_hashes["sampler_generator_final_state_sha256"]
    assert sampler_hashes["continuous"] == sampler_hashes["resumed"], runtime
    return {"trajectory_equal": True, "comparison": comparison, "reports": reports,
            "generator_hashes": generator_hashes}


@pytest.mark.cpu
def test_stateful_trainer_fresh_process_cpu_resume():
    inherited = dict(os.environ)
    pythons = _configured_pythons(inherited)
    run_root = _new_run_root(inherited)
    environments = {runtime: _runtime_environment(runtime, run_root, inherited) for runtime in pythons}
    versions = {}
    for runtime, python in pythons.items():
        required = PACKAGES + (("torch",) if runtime == "pytorch" else ())
        versions[runtime] = _dependency_versions(python, environments[runtime],
                                                 run_root / runtime / "dependencies.log", required)
    for name in PACKAGES:
        assert versions["pytorch"][name] == versions["jittor"][name], (name, versions)
    weights = run_root / "initial-weights.npz"
    _invoke(pythons["pytorch"], ["initialize", "--runtime", "pytorch", "--weights-out", weights],
            environments["pytorch"], run_root / "initialize.log")
    results = {runtime: _exercise_runtime(runtime, python, environments[runtime], run_root, weights)
               for runtime, python in pythons.items()}
    (run_root / "summary.json").write_text(json.dumps({
        "device": "cpu", "pythons": pythons, "versions": versions, "results": results,
    }, indent=2) + "\n")
    print("Trainer stateful integration artifacts: %s" % run_root)


def test_missing_oracle_is_a_visible_skip_or_a_required_error():
    with pytest.raises(pytest.skip.Exception, match="missing independent torch oracle"):
        _configured_pythons({})
    with pytest.raises(pytest.fail.Exception, match="JITTOR_REQUIRE_TRAINER_STATEFUL=1"):
        _configured_pythons({REQUIRE_ENV: "1"})


def test_missing_optional_dependency_probe_honors_the_required_gate(tmp_path):
    missing = {name: "1.0" for name in PACKAGES}
    missing["torchdata"] = None
    response = subprocess.CompletedProcess([], 0, stdout=json.dumps(missing))
    with mock.patch.object(subprocess, "run", return_value=response) as invoke:
        with pytest.raises(pytest.skip.Exception, match="torchdata"):
            _dependency_versions("fake-python", {}, tmp_path / "optional.log", PACKAGES)
        with pytest.raises(pytest.fail.Exception, match="JITTOR_REQUIRE_TRAINER_STATEFUL=1"):
            _dependency_versions("fake-python", {REQUIRE_ENV: "1"}, tmp_path / "required.log", PACKAGES)
    assert invoke.call_args[0][0][1:3] == ["-I", "-c"]


def test_runtime_environments_are_cpu_only_isolated_and_keep_the_abi_setup(tmp_path):
    inherited = {"HOME": "/inherited-home", "use_cuda": "1", "LD_PRELOAD": "/system/libstdc++.so.6"}
    environments = {name: _runtime_environment(name, tmp_path / "run", inherited) for name in PYTHON_ENV}
    for env in environments.values():
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert "use_cuda" not in env
        assert env["LD_PRELOAD"] == inherited["LD_PRELOAD"]
        assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
        assert env["JT_BUILD_NVCC_PATH"] == env["nvcc_path"] == ""
        assert env["JITTOR_CUDA"] == env["JT_BUILD_USE_CUTT"] == "0"
    assert environments["jittor"]["JITTOR_HOME"] != environments["pytorch"]["JITTOR_HOME"]
    assert environments["jittor"]["HOME"] != inherited["HOME"]
    assert environments["jittor"]["JITTOR_TORCH_SHIM"] == "1"
    assert environments["pytorch"]["JITTOR_TORCH_SHIM"] == "0"


def test_artifact_destinations_are_guarded_under_lab_state(tmp_path):
    env = {"JITTOR_LAB_ROOT": str(tmp_path / "lab")}
    first, second = _new_run_root(env), _new_run_root(env)
    assert first != second
    assert first.parent == tmp_path / "lab" / "_state" / "trainer-resume"
    with pytest.raises(ValueError, match="outside the repository"):
        _new_run_root({"JITTOR_LAB_ROOT": str(ROOT / "forbidden")})


def test_fresh_process_commands_preserve_the_four_step_resume_contract(tmp_path):
    state = tmp_path / "jittor"
    state.mkdir()
    checkpoint = state / "first" / "checkpoints" / "checkpoint-2" / "stateful_dataloader.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    reports = {
        "sampler_generator_final_state_sha256": "sampler-hash",
        "dataloader_generator_final_state_sha256": "loader-hash",
    }
    with mock.patch(__name__ + "._invoke", return_value=json.dumps({"items": 272, "failure_count": 0})) as invoke:
        with mock.patch(__name__ + "._assert_report", return_value=reports):
            result = _exercise_runtime("jittor", "fake-python", {}, tmp_path, tmp_path / "weights.npz")
    commands = [list(call[0][1]) for call in invoke.call_args_list]
    assert [command[0] for command in commands] == ["run", "run", "run", "merge", "compare"]
    for command in commands[:3]:
        assert "--stateful-dataloader" in command
        assert command[command.index("--max-steps") + 1] == "4"
        assert command[command.index("--device") + 1] == "cpu"
    assert commands[1][commands[1].index("--stop-after") + 1] == "2"
    assert commands[2][commands[2].index("--start-step") + 1] == "2"
    assert commands[2][commands[2].index("--resume-checkpoint") + 1] == checkpoint.parent
    assert result["trajectory_equal"] is True
