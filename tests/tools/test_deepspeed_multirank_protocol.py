"""Artifact isolation and failure cleanup need no framework or GPUs."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from urllib.parse import urlparse

import numpy as np
import pytest

from _helpers.child_process import PYTHON, child_env


def _launcher():
    path = Path(__file__).resolve().parents[2] / "compat/tests/torch/_ecosystem_deepspeed_multirank.py"
    spec = importlib.util.spec_from_file_location("deepspeed_multirank_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_case_identity_and_previous_evidence_cannot_be_overwritten(tmp_path):
    launch = _launcher()
    first = launch.reserve_case(tmp_path, 1, "AdamW", "bfloat16", 2, 0.)
    evidence = first / "rank0.npz"
    evidence.write_bytes(b"previous numerical evidence")
    with pytest.raises(FileExistsError):
        launch.reserve_case(tmp_path, 1, "AdamW", "bfloat16", 2, 0.)
    assert evidence.read_bytes() == b"previous numerical evidence"
    second = launch.reserve_case(tmp_path, 2, "AdamW", "bfloat16", 2, 0.)
    perturbed = launch.reserve_case(tmp_path, 2, "AdamW", "bfloat16", 2, .125)
    assert len({first, second, perturbed}) == 3
    assert json.loads((second / "protocol.json").read_text())["zero_stage"] == 2
    assert all(part in second.parts for part in ("zero2", "AdamW", "bfloat16", "world2"))
    with pytest.raises(ValueError, match="Unverified"):
        launch.reserve_case(tmp_path, 2, "AdamW", "float32", 1, 0.)


def test_artifacts_inside_checkout_are_rejected():
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="outside"):
        _launcher().reserve_case(root / "forbidden-output", 2, "AdamW", "bfloat16", 2, 0.)
    with pytest.raises(ValueError, match="outside"):
        _launcher()._rank_environment("jittor", 0, "0", root, "")


def test_rank_caches_and_oracle_framework_environment_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JITTOR_HOME", "/foreign/cache")
    monkeypatch.setenv("JITTOR_TORCH_SHIM", "1")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/foreign/cuda/lib")
    launch = _launcher()
    first = launch._rank_environment("jittor", 0, "0", tmp_path, "")
    second = launch._rank_environment("jittor", 1, "1", tmp_path, "")
    oracle = launch._rank_environment("torch", 0, "0", tmp_path, "")
    for name in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR"):
        assert first[name] != second[name] and first[name] != oracle[name]
        assert Path(first[name]).is_dir()
    assert first["JITTOR_HOME"] != second["JITTOR_HOME"]
    assert "JITTOR_HOME" not in oracle and "JITTOR_TORCH_SHIM" not in oracle
    assert "LD_LIBRARY_PATH" not in oracle


def test_same_case_has_shared_control_plane_and_private_rank_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "foreign-launcher")
    monkeypatch.setenv("MASTER_PORT", "29500")
    first, second = _launcher().pair_environments(
        "jittor", ("0", "1"), tmp_path / "caches", "", tmp_path / "case")
    assert first["JITTOR_DIST_RENDEZVOUS_DIR"] == second["JITTOR_DIST_RENDEZVOUS_DIR"]
    assert Path(first["JITTOR_DIST_RENDEZVOUS_DIR"]).is_dir()
    for name in ("HOME", "JITTOR_HOME", "TMPDIR"):
        assert first[name] != second[name]
    assert all(name not in environment for environment in (first, second)
               for name in ("MASTER_ADDR", "MASTER_PORT"))


def test_cpu_protocol_has_independent_artifacts_and_real_mpi_launch_controls(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "foreign-gpu")
    monkeypatch.setenv("JT_BUILD_NVCC_PATH", "/foreign/nvcc")
    launch = _launcher()
    cpu = launch.reserve_case(tmp_path / "evidence", 1, "AdamW", "bfloat16", 2, 0., device="cpu")
    cuda = launch.reserve_case(tmp_path / "evidence", 1, "AdamW", "bfloat16", 2, 0., device="cuda")
    assert cpu != cuda and json.loads((cpu / "protocol.json").read_text())["device"] == "cpu"
    native = cpu / "jittor"
    native.mkdir()
    first, second = launch.pair_environments("jittor", (), tmp_path / "cache", "", native, device="cpu")
    assert first["JITTOR_DIST_RENDEZVOUS_DIR"] == second["JITTOR_DIST_RENDEZVOUS_DIR"]
    for name in ("HOME", "TMPDIR", "JITTOR_HOME"):
        assert first[name] != second[name]
    for rank, environment in enumerate((first, second)):
        assert environment["JT_BUILD_USE_MPI"] == "1" and environment["JT_BUILD_USE_NCCL"] == "0"
        assert environment["CUDA_VISIBLE_DEVICES"] == environment["JT_BUILD_NVCC_PATH"] == ""
        assert environment["LOCAL_RANK"] == str(rank)
        assert environment["JITTOR_TORCH_DISTRIBUTED_AUTO_INIT"] == "0"
    oracle = launch._rank_environment("torch", 1, "", tmp_path / "cache", "", device="cpu")
    assert oracle["DS_ACCELERATOR"] == "cpu" and oracle["CUDA_VISIBLE_DEVICES"] == ""
    assert all(not key.startswith(("JITTOR_", "JT_")) for key in oracle)


@pytest.mark.parametrize("world,rank,local", ((None, None, None), ("1", "0", "0"),
                                            ("2", "2", "2"), ("2", "1", "0")))
def test_mpi_entry_rejects_missing_or_inconsistent_actual_launcher_identity(
        tmp_path, monkeypatch, world, rank, local):
    for name, value in (("OMPI_COMM_WORLD_SIZE", world), ("OMPI_COMM_WORLD_RANK", rank),
                        ("OMPI_COMM_WORLD_LOCAL_RANK", local)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="OpenMPI"):
        _launcher()._mpi_rank_main(["--mpi-case", str(tmp_path / "missing"), "--cache-root", str(tmp_path),
                                   "--rendezvous", "file:///unused"])


def test_mpi_entry_routes_rank_output_and_exclusively_claims_it(tmp_path, monkeypatch):
    launch = _launcher()
    case = launch.reserve_case(tmp_path / "evidence", 2, "AdamW", "bfloat16", 2, .125, device="cpu")
    directory = case / "jittor"
    directory.mkdir()
    (directory / "rendezvous").mkdir()
    for name, value in (("OMPI_COMM_WORLD_SIZE", "2"), ("OMPI_COMM_WORLD_RANK", "1"),
                        ("OMPI_COMM_WORLD_LOCAL_RANK", "1"), ("EXAMPLE_SECRET", "not-for-artifacts")):
        monkeypatch.setenv(name, value)
    observed = {}
    class Executed(Exception):
        pass
    def execute(python, command, environment):
        observed.update(command=command, environment=environment)
        raise Executed
    monkeypatch.setattr(launch.os, "execve", execute)
    monkeypatch.setattr(launch.os, "dup2", lambda *args: None)
    arguments = ["--mpi-case", str(directory), "--cache-root", str(tmp_path / "cache"),
                 "--rendezvous", launch.file_rendezvous(directory)]
    with pytest.raises(Executed):
        launch._mpi_rank_main(arguments)
    command = observed["command"]
    assert command[command.index("--rank") + 1] == "1"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--output") + 1] == str(directory / "rank1.npz")
    assert observed["environment"]["LOCAL_RANK"] == "1"
    claim = (directory / "rank1-launch.json").read_text()
    assert "EXAMPLE_SECRET" not in claim and "not-for-artifacts" not in claim
    assert json.loads(claim)["environment"]["JITTOR_DIST_RENDEZVOUS_DIR"] == str(directory / "rendezvous")
    with pytest.raises(FileExistsError):
        launch._mpi_rank_main(arguments)


def test_cpu_warm_does_not_select_nccl_mode_and_launch_keeps_parent_affinity(tmp_path, monkeypatch):
    launch = _launcher()
    monkeypatch.setenv("JT_NCCL_WORLD_SIZE", "99")
    case = launch.reserve_case(tmp_path / "case", 1, "AdamW", "bfloat16", 2, 0., device="cpu")
    warm, started = [], []
    monkeypatch.setattr(launch.shutil, "which", lambda name: "/private/mpi/bin/mpirun")
    def run(command, **kwargs):
        warm.append(kwargs["env"])
    def start(command, **kwargs):
        started.append(command)
        for rank in (0, 1):
            (case / "jittor" / ("rank%d.npz" % rank)).write_bytes(b"control-only-placeholder")
        return SimpleNamespace(poll=lambda: 0)
    monkeypatch.setattr(launch.subprocess, "run", run)
    monkeypatch.setattr(launch.subprocess, "Popen", start)
    launch.run_pair(PYTHON, "jittor", case, tmp_path / "cache", (), "")
    assert len(warm) == 2
    for environment in warm:
        assert environment["JT_BUILD_USE_MPI"] == "1"
        assert environment["WORLD_SIZE"] == "1"
        assert not any(name.startswith("JT_NCCL_") for name in environment)
    assert len(started) == 1
    command = started[0]
    assert command[command.index("--bind-to") + 1] == "none"
    assert command[command.index("-n") + 1] == "2"


@pytest.mark.parametrize("bad", ("rank", "world", "local", "devices", "control"))
def test_jtrun_wrapper_preserves_public_launcher_identity_and_rejects_inconsistency(
        tmp_path, monkeypatch, bad):
    launch = _launcher()
    case = launch.reserve_case(tmp_path / "case", 2, "AdamW", "bfloat16", 2, 0.)
    directory = case / "jittor"
    directory.mkdir()
    control = directory / "rendezvous"
    control.mkdir()
    environment = {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "0",
                   "JT_NCCL_RANK": "1", "JT_NCCL_WORLD_SIZE": "2", "JT_NCCL_LOCAL_RANK": "0",
                   "JT_NCCL_ROOTINFO_FILE": str(control / "nccl-rootinfo.bin"),
                   "CUDA_VISIBLE_DEVICES": "GPU-example", "JITTOR_DIST_RENDEZVOUS_DIR": str(control)}
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    observed = {}
    class Executed(Exception):
        pass
    def execute(python, command, env):
        observed.update(command=command, environment=env)
        raise Executed
    monkeypatch.setattr(launch.os, "execve", execute)
    monkeypatch.setattr(launch.os, "dup2", lambda *args: None)
    arguments = ["--jtrun-case", str(directory), "--cache-root", str(tmp_path / "cache"),
                 "--rendezvous", launch.file_rendezvous(directory)]
    with pytest.raises(Executed):
        launch._jtrun_rank_main(arguments)
    result = observed["environment"]
    assert all(result[key] == value for key, value in environment.items())
    assert result["JITTOR_TORCH_DISTRIBUTED_AUTO_INIT"] == "0"
    assert result["JITTOR_DEEPSPEED_LAUNCHER"] == "jtrun-module"
    assert str(tmp_path / "cache" / "jittor" / "rank1") in result["JITTOR_HOME"]
    (directory / "rank1-launch.json").unlink()
    (directory / "rank1.log").unlink()
    key, value = {"rank": ("JT_NCCL_RANK", "0"), "world": ("WORLD_SIZE", "1"),
                  "local": ("LOCAL_RANK", "1"), "devices": ("CUDA_VISIBLE_DEVICES", "0,1"),
                  "control": ("JITTOR_DIST_RENDEZVOUS_DIR", "/other-case")}[bad]
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="jtrun"):
        launch._jtrun_rank_main(arguments)


def test_cuda_candidate_uses_public_jtrun_with_cache_wrapper_before_site_hooks(tmp_path, monkeypatch):
    launch = _launcher()
    case = launch.reserve_case(tmp_path / "case", 1, "AdamW", "bfloat16", 2, 0.)
    observed = []
    monkeypatch.setattr(launch.subprocess, "run", lambda *args, **kwargs: None)
    def start(command, **kwargs):
        observed.append(command)
        for rank in (0, 1):
            (case / "jittor" / ("rank%d.npz" % rank)).write_bytes(b"control-only-placeholder")
        return SimpleNamespace(poll=lambda: 0)
    monkeypatch.setattr(launch.subprocess, "Popen", start)
    launch.run_pair(PYTHON, "jittor", case, tmp_path / "cache", ("allocated-a", "allocated-b"), "")
    assert len(observed) == 1
    command = observed[0]
    assert command[:3] == [PYTHON, "-m", "jittor.distributed.launch"]
    assert command[command.index("--device-ids") + 1] == "allocated-a,allocated-b"
    assert command[command.index("--log-dir") + 1] == str(case / "jittor" / "rendezvous")
    assert command[command.index("--") + 1:command.index("--") + 3] == [PYTHON, "-S"]


def test_file_rendezvous_is_per_runtime_and_rejects_stale_stores(tmp_path):
    launch = _launcher()
    native = tmp_path / "jittor"
    oracle = tmp_path / "torch"
    native.mkdir()
    oracle.mkdir()
    native_url = launch.file_rendezvous(native)
    oracle_url = launch.file_rendezvous(oracle)
    assert native_url != oracle_url
    assert launch.file_rendezvous(native) == native_url
    parsed = urlparse(native_url)
    assert parsed.scheme == "file" and not parsed.netloc
    store = Path(parsed.path)
    assert store.parent == native and not store.exists()
    store.write_bytes(b"old process-group state")
    with pytest.raises(FileExistsError, match="already exists"):
        launch.file_rendezvous(native)
    assert store.read_bytes() == b"old process-group state"


@pytest.mark.parametrize("failure", ("rank", "timeout"))
def test_failed_or_timed_out_rank_terminates_its_live_peer(tmp_path, failure):
    paths = [tmp_path / ("rank%d.log" % i) for i in range(2)]
    handles = [path.open("w") for path in paths]
    programs = ["import time; time.sleep(60)",
                "raise SystemExit(7)" if failure == "rank" else "import time; time.sleep(60)"]
    processes = []
    try:
        for program, handle in zip(programs, handles):
            processes.append(subprocess.Popen(
                [PYTHON, "-c", program], env=child_env(without_torch_mode=True),
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        expected = RuntimeError if failure == "rank" else TimeoutError
        with pytest.raises(expected):
            _launcher().wait_for_ranks(processes, paths, 10 if failure == "rank" else .05)
        assert all(process.poll() is not None for process in processes)
        if failure == "rank":
            assert processes[1].returncode == 7
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
        for handle in handles:
            handle.close()


def _runner_definition(name):
    path = Path(__file__).resolve().parents[2] / "compat/tests/torch/_ecosystem_deepspeed_runner.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


def test_full_gradient_dtype_preserves_the_upstream_stage1_shortcut_and_stage2_partition():
    dtype = _runner_definition("_full_gradient_dtype")
    torch = SimpleNamespace(float32="float32")
    stage1 = SimpleNamespace(grad=SimpleNamespace(dtype="bfloat16"))
    stage2 = SimpleNamespace(grad=None)
    assert dtype(torch, stage1) == "bfloat16"
    assert dtype(torch, stage2) == "float32"


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_backend_label_does_not_promote_a_singleton_to_a_transport_claim(device):
    # Controlled NumPy observations test the comparison's metadata boundary;
    # these are not framework execution evidence or a DeepSpeed scorecard.
    directory = Path(__file__).resolve().parents[2] / "compat/tests/torch"
    namespace = {"np": np, "json": json, "EcosystemComparison": SimpleNamespace(
        forward_tolerance=.002, backward_tolerance=.01)}
    for filename, names in (("_ecosystem_harness.py", ("_versions", "_divergence", "_comparison_floor")),
                            ("_ecosystem_deepspeed_comparison.py", ("_compare_engine_outputs",))):
        nodes = [node for node in ast.parse((directory / filename).read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    class Archive(dict):
        @property
        def files(self):
            return list(self)
    compare = namespace["_compare_engine_outputs"]
    for world in (1, 2):
        common = dict(device=device, world_size=world, rank=0, dtype="float32", optimizer="AdamW",
            zero_stage=1, perturbation=0., engine_class="controlled", parameter_gradients_per_microstep=0,
            input_gradients_per_microstep=1, microsteps=1, global_steps=1, runtime_conditions={},
            optimizer_groups=[], optimizer_partition_sizes=[], model_parameter_elements=0,
            trainable_parameters=[], frozen_parameters=[], strict_resume_fields=1,
            gradient_dtypes={"step0/input_grad/x": "float32"}, jittor_commit="controlled-host-fixture",
            seed=0, hf_hub_offline=True, protocol={"controlled": True}, measurement_source_sha256={},
            optimizer_class="controlled.Optimizer", updated_master_elements=1)
        ours = dict(common, runtime="jittor", process_group_backend="mpi" if world == 1 or device == "cpu" else "nccl",
                    fallback_policy="error", fallback_count=0, backend={"use_cuda": device == "cuda"})
        theirs = dict(common, runtime="torch", torch_binary="controlled-oracle.so",
                      process_group_backend="nccl" if device == "cuda" else "gloo")
        values = {"output": np.array([1.], dtype="float32"),
                  "step0/input_grad/x": np.array([.25], dtype="float32")}
        actual, reference = Archive(values, metadata=np.asarray(json.dumps(ours))), Archive(
            values, metadata=np.asarray(json.dumps(theirs)))
        compare(actual, reference, device=device, world_size=world)
        if world == 2:
            ours["process_group_backend"] = "gloo" if device == "cpu" else "mpi"
            actual["metadata"] = np.asarray(json.dumps(ours))
            with pytest.raises(AssertionError):
                compare(actual, reference, device=device, world_size=world)


@pytest.mark.parametrize("tamper", ("gradient", "moment", "missing", "dtype"))
def test_strict_resume_rejects_hidden_gradient_or_optimizer_divergence(tamper):
    arrays = {}
    for before, after, field in (("saved_optimizer", "restored_optimizer", "exp_avg"),
                                 ("saved_master_partition", "restored_master_partition", "0/0"),
                                 ("final_optimizer", "resumed_optimizer", "exp_avg"),
                                 ("final_master_partition", "resumed_master_partition", "0/0"),
                                 ("step4", "resumed4", "input_grad/inputs"),
                                 ("step5", "resumed5", "incoming_parameter_grad/weight")):
        value = np.array([2.] if before.startswith("final") else [1.], dtype="float32")
        arrays[before + "/" + field] = value
        arrays[after + "/" + field] = value.copy()
    strict = _runner_definition("_assert_strict_resume")
    dtypes = {key: "bfloat16" for key in arrays if "_grad/" in key}
    assert strict(arrays, dtypes)[1] == 1
    if tamper == "gradient":
        arrays["resumed5/incoming_parameter_grad/weight"][0] += .01
    elif tamper == "moment":
        arrays["resumed_optimizer/exp_avg"][0] += .01
    elif tamper == "missing":
        arrays.pop("resumed4/input_grad/inputs")
    else:
        dtypes["resumed5/incoming_parameter_grad/weight"] = "float32"
    with pytest.raises(AssertionError):
        strict(arrays, dtypes)
