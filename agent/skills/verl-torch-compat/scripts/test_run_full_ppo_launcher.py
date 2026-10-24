"""Pure shell control-plane tests; no framework or accelerator is loaded."""
import os
from pathlib import Path
import subprocess

import pytest


LAUNCHER = Path(__file__).with_name("run_full_ppo_hf_async.sh")


def _run(tmp_path, cards, *, cuda_pids="", cuda_metrics="0, GPU-host-stub, 12, 61440, 81920",
         cuda_query_exit="0", npu_process_info="occupied NPU host stub", framework="jittor", **extra):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text(
        "#!/bin/sh\necho OCCUPANCY_CHECK_WAS_CALLED >&2\n"
        "case \"$*\" in\n"
        "*--query-compute-apps=pid*) printf '%s\\n' \"$HOST_TEST_CUDA_PIDS\";;\n"
        "*--query-gpu=*) printf '%s\\n' \"$HOST_TEST_CUDA_METRICS\"; exit \"$HOST_TEST_GPU_QUERY_EXIT\";;\n"
        "esac\nexit 0\n"
    )
    smi.chmod(0o755)
    npu_smi = bin_dir / "npu-smi"
    npu_smi.write_text("#!/bin/sh\nprintf '%s\\n' \"$HOST_TEST_NPU_PROCESS_INFO\"\n")
    npu_smi.chmod(0o755)
    cann_env = tmp_path / "cann-host-stub.sh"
    cann_env.write_text(": # Host stub only; never a real CANN setup.\n")
    source = tmp_path / "source"
    source.mkdir()
    verl = tmp_path / "verl"
    verl.mkdir()
    full_root = tmp_path / "full"
    python_dir = full_root / "candidate/bin"
    python_dir.mkdir(parents=True)
    python = python_dir / "python"
    python.write_text(
        "#!/bin/sh\nenv | awk '/^(JT_HCCL_|JT_NCCL_|RANK=|WORLD_SIZE=|LOCAL_RANK=|MASTER_ADDR=|MASTER_PORT=|VERL_L4_PERFORMANCE_ELIGIBLE=|VERL_CUDA_STRICT_FP32=|VERL_PLATFORM=|RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=|RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=|JITTOR_VERL_RAY_CUDA_CONTEXT=|JITTOR_VERL_RAY_NPU_CONTEXT=)/'\n"
        "case \"$1\" in */build_minimal_ppo_dataset.py) exit 0;; esac\n"
        "printf '%s\\n' \"$@\"\nexit 37\n"
    )
    python.chmod(0o755)
    # Official-framework branch is also a shell stub, not an oracle runtime.
    wrapper = source / "agent/skills/verl-torch-compat/runtime/oracle_worker_python.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(python.read_text())
    wrapper.chmod(0o755)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("JITTOR_", "JT_HCCL_", "JT_NCCL_", "VERL_"))}
    environment.update(
        PATH=str(bin_dir) + os.pathsep + environment["PATH"],
        VERL_DEVICE="cuda", VERL_STATE_ROOT=str(tmp_path / "state"),
        JITTOR_SOURCE_ROOT=str(source), VERL_SOURCE_ROOT=str(verl),
        VERL_MODEL_PATH=str(tmp_path / "model"), VERL_FULL_PPO_ROOT=str(full_root),
        VERL_NVCC_PATH="/stub/nvcc",
        VERL_RAY_TMPDIR=str(tmp_path / "ray"),
        VERL_PROCESS_TMPDIR=str(tmp_path / "process-tmp"),
        HOST_TEST_CUDA_PIDS=cuda_pids,
        HOST_TEST_CUDA_METRICS=cuda_metrics,
        HOST_TEST_GPU_QUERY_EXIT=cuda_query_exit,
        HOST_TEST_NPU_PROCESS_INFO=npu_process_info,
        CANN_SET_ENV=str(cann_env),
    )
    environment.update(extra)
    return subprocess.run(["bash", str(LAUNCHER), framework, "host-contract", cards],
                          env=environment, text=True, capture_output=True, timeout=15)


def test_duplicate_cuda_cards_are_rejected_before_device_checks(tmp_path):
    result = _run(tmp_path, "0,0")
    assert result.returncode == 2
    assert "duplicate" in result.stderr
    assert "OCCUPANCY_CHECK_WAS_CALLED" not in result.stderr


@pytest.mark.parametrize("framework", ("jittor", "torch"))
@pytest.mark.parametrize("device,dtype", (("cuda", "fp32"), ("cuda", "bf16"), ("npu", "fp32"), ("npu", "bf16")))
def test_launcher_enables_strict_fp32_only_for_cuda_fp32(tmp_path, framework, device, dtype):
    result = _run(tmp_path, "0", framework=framework, VERL_DEVICE=device,
                  VERL_TRAIN_DTYPE=dtype, VERL_CUDA_STRICT_FP32="0", npu_process_info="No process in device")
    assert result.returncode == 37
    policies = [line for line in result.stdout.splitlines() if line.startswith("VERL_CUDA_STRICT_FP32=")]
    assert policies == (["VERL_CUDA_STRICT_FP32=1"] * 2 if (device, dtype) == ("cuda", "fp32") else [])


@pytest.mark.parametrize("value,device,dtype", (("1", "npu", "fp32"), ("1", "cuda", "bf16"), ("", "cuda", "fp32"), ("2", "cuda", "fp32")))
def test_launcher_rejects_misconfigured_strict_fp32_before_device_checks(tmp_path, value, device, dtype):
    result = _run(tmp_path, "0", VERL_CUDA_STRICT_FP32=value, VERL_DEVICE=device, VERL_TRAIN_DTYPE=dtype)
    assert result.returncode == 2
    assert "VERL_CUDA_STRICT_FP32" in result.stderr
    assert "OCCUPANCY_CHECK_WAS_CALLED" not in result.stderr


@pytest.mark.parametrize("critic_override", (None, "/host/derived-critic"))
def test_critic_override_does_not_change_other_model_paths(tmp_path, critic_override):
    extra = {} if critic_override is None else {"VERL_CRITIC_MODEL_PATH": critic_override}
    result = _run(tmp_path, "0", **extra)
    assert result.returncode == 37
    model = str(tmp_path / "model")
    assert "critic.model.path=" + (critic_override or model) in result.stdout.splitlines()
    assert "actor_rollout_ref.model.path=" + model in result.stdout.splitlines()


def test_cuda_driver_does_not_inherit_stale_distributed_rank(tmp_path):
    result = _run(
        tmp_path, "0", JT_HCCL_WORLD_SIZE="2", JT_HCCL_RANK="1",
        JT_HCCL_LOCAL_RANK="1", JT_HCCL_ROOTINFO_FILE="/stale/hccl-root",
        JT_NCCL_WORLD_SIZE="2", JT_NCCL_RANK="1", JT_NCCL_LOCAL_RANK="1",
        JT_NCCL_ROOTINFO_FILE="/stale/nccl-root", RANK="1", WORLD_SIZE="2",
        LOCAL_RANK="1", MASTER_ADDR="stale-host", MASTER_PORT="59999",
    )
    assert result.returncode == 37  # Stub Python, never a trainer success.
    for name in ("JT_HCCL_WORLD_SIZE", "JT_HCCL_RANK", "JT_HCCL_LOCAL_RANK",
                 "JT_HCCL_ROOTINFO_FILE", "JT_NCCL_WORLD_SIZE", "JT_NCCL_RANK",
                 "JT_NCCL_LOCAL_RANK", "JT_NCCL_ROOTINFO_FILE", "RANK",
                 "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        assert not any(line.startswith(name + "=") for line in result.stdout.splitlines()), name


@pytest.mark.parametrize("device", ("cuda", "npu"))
@pytest.mark.parametrize("framework", ("jittor", "torch"))
@pytest.mark.parametrize("cards", ("0", "0,1"))
def test_driver_drops_inherited_ray_context_and_binds_platform(tmp_path, device, framework, cards):
    result = _run(
        tmp_path, cards, framework=framework, VERL_DEVICE=device,
        npu_process_info="No process in device", VERL_PLATFORM="stale-platform",
        RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1",
        RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="1",
        JITTOR_VERL_RAY_CUDA_CONTEXT="1", JITTOR_VERL_RAY_NPU_CONTEXT="1",
    )
    assert result.returncode == 37
    platform = {"cuda": "nvidia", "npu": "huawei"}[device]
    assert "VERL_PLATFORM=" + platform in result.stdout.splitlines()
    for name in ("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO",
                 "JITTOR_VERL_RAY_CUDA_CONTEXT", "JITTOR_VERL_RAY_NPU_CONTEXT"):
        assert not any(line.startswith(name + "=") for line in result.stdout.splitlines()), name


def test_full_ppo_pins_rollout_and_dataset_determinism(tmp_path):
    result = _run(tmp_path, "0")
    assert result.returncode == 37
    for setting in ("actor_rollout_ref.rollout.full_determinism=true",
                    "actor_rollout_ref.rollout.seed=42", "data.shuffle=false", "data.seed=42"):
        assert setting in result.stdout.splitlines()


@pytest.mark.parametrize("training,rollout", (("fp32", "float32"), ("bf16", "bfloat16")))
def test_rollout_precision_matches_declared_training_precision(tmp_path, training, rollout):
    result = _run(tmp_path, "0", VERL_TRAIN_DTYPE=training)
    assert result.returncode == 37
    assert "actor_rollout_ref.rollout.dtype=" + rollout in result.stdout.splitlines()


@pytest.mark.parametrize("authorization", (None, "0"))
def test_occupied_cuda_is_rejected_by_default(tmp_path, authorization):
    extra = {} if authorization is None else {"VERL_ALLOW_SHARED_CUDA": authorization}
    result = _run(tmp_path, "0", cuda_pids="1234\n5678", **extra)
    assert result.returncode == 20
    assert "refusing occupied CUDA device" in result.stderr
    assert "launch_main_ppo.py" not in result.stdout


def test_explicit_shared_cuda_records_load_and_disallows_l4(tmp_path):
    result = _run(tmp_path, "0", cuda_pids="1234\n5678", VERL_ALLOW_SHARED_CUDA="1")
    assert result.returncode == 37  # Stub only, not real PPO success.
    assert "VERL_L4_PERFORMANCE_ELIGIBLE=0" in result.stdout.splitlines()
    assert "L4" in result.stderr and "shared" in result.stderr.lower()
    evidence = tmp_path / "full/runs/host-contract/shared-cuda-context.txt"
    assert evidence.is_file()
    text = evidence.read_text()
    assert "shared_cuda=true" in text and "performance_eligible=false" in text
    assert "l4_performance_eligible=0" in text
    assert "utilization.gpu" in text and "memory.free" in text and "MiB" in text
    assert "0, GPU-host-stub, 12, 61440, 81920" in text
    assert "1234" in text and "5678" in text


@pytest.mark.parametrize("authorization", ("", "yes", "true", "2", "-1"))
def test_invalid_shared_authorization_is_rejected_before_device_checks(tmp_path, authorization):
    result = _run(tmp_path, "0", VERL_ALLOW_SHARED_CUDA=authorization)
    assert result.returncode == 2
    assert "VERL_ALLOW_SHARED_CUDA" in result.stderr
    assert "OCCUPANCY_CHECK_WAS_CALLED" not in result.stderr


def test_shared_cuda_permission_cannot_bypass_npu_occupancy(tmp_path):
    result = _run(tmp_path, "0", VERL_DEVICE="npu", VERL_ALLOW_SHARED_CUDA="1")
    assert result.returncode == 20
    assert "refusing occupied NPU" in result.stderr
    assert not (tmp_path / "full/runs/host-contract/shared-cuda-context.txt").exists()


def test_shared_cuda_requires_observable_load_metadata(tmp_path):
    result = _run(tmp_path, "0", cuda_pids="1234", cuda_metrics="", VERL_ALLOW_SHARED_CUDA="1")
    assert result.returncode != 37
    assert "empty CUDA GPU status" in result.stderr


def test_shared_cuda_observation_failure_is_not_ignored(tmp_path):
    result = _run(tmp_path, "0", cuda_pids="1234", cuda_query_exit="42",
                  VERL_ALLOW_SHARED_CUDA="1")
    assert result.returncode == 42
    assert "launch_main_ppo.py" not in result.stdout
    assert not (tmp_path / "full/runs/host-contract/shared-cuda-context.txt").exists()


def test_shared_cuda_even_if_currently_idle_is_not_l4_evidence(tmp_path):
    result = _run(tmp_path, "0", VERL_ALLOW_SHARED_CUDA="1")
    assert result.returncode == 37
    evidence = tmp_path / "full/runs/host-contract/shared-cuda-context.txt"
    assert "compute_pids=none" in evidence.read_text()
    assert "VERL_L4_PERFORMANCE_ELIGIBLE=0" in result.stdout.splitlines()


def test_shared_cuda_records_each_selected_card(tmp_path):
    result = _run(tmp_path, "0,1", cuda_pids="1234", VERL_ALLOW_SHARED_CUDA="1")
    assert result.returncode == 37
    text = (tmp_path / "full/runs/host-contract/shared-cuda-context.txt").read_text()
    assert "requested_gpu=0" in text and "requested_gpu=1" in text
    assert text.count("compute_pids=1234") == 2


def test_free_npu_path_is_unchanged_by_cuda_sharing_permission(tmp_path):
    result = _run(tmp_path, "0", npu_process_info="No process in device",
                  VERL_DEVICE="npu", VERL_ALLOW_SHARED_CUDA="1")
    assert result.returncode == 37
    assert "VERL_L4_PERFORMANCE_ELIGIBLE=0" not in result.stdout.splitlines()
    assert not (tmp_path / "full/runs/host-contract/shared-cuda-context.txt").exists()
