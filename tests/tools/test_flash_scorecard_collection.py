"""Observed platform identity and immutable case completeness stay fail-closed."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def collector():
    root = Path(__file__).resolve().parents[2]
    path = root / "agent/skills/flash-attention-torch-compat/scripts/collect_scorecard.py"
    spec = importlib.util.spec_from_file_location("flash_scorecard_collection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def metadata():
    platform = {"cuda_visible_devices": "2", "actual_device_count": 1, "actual_device": "Test CUDA",
                "gpu_driver_inventory": ["0, Other CUDA, GPU-other, 580", "2, Test CUDA, GPU-test, 580"],
                "os": "Linux", "cpu": "test CPU", "nvidia_wheels": {}, "torch_cuda": "12.8",
                "compiler_environment": {"nvcc": "Cuda compilation tools, release 12.2, V12.2.140", "cxx": "GCC 11.5"}}
    precision = {"matmul": False, "cudnn": False, "cudnn_benchmark": False, "matmul_precision": "highest"}
    candidate = {"platform": platform, "actual_dispatch": ["cuda", 0], "precision": precision,
                 "native_flags": {"auto_graph_replay": 1, "use_acl": 1},
                 "native_flag_aliases": {"use_acl": "legacy CUDA alias"}}
    reference = {"platform": dict(platform), "torch": "2.11", "binary_extension": "torch/_C.so"}
    return candidate, reference


def test_actual_cuda_dispatch_controls_backend_flags(collector, metadata):
    platform = collector.platform_from_metadata(*metadata)
    assert platform["card_indices"] == ["2"]
    assert platform["jittor_flags"]["use_cuda"] == 1
    assert platform["jittor_flags"]["use_acl"] == 0
    assert platform["cuda"] == "12.2" and platform["oracle"]["cuda"] == "12.8"


@pytest.mark.parametrize("defect", ("allocation", "model", "device_count", "cpu_dispatch", "oracle_binary"))
def test_unobserved_or_mismatched_platform_is_rejected(collector, metadata, defect):
    candidate, reference = metadata
    if defect == "allocation":
        reference["platform"]["cuda_visible_devices"] = "0"
    elif defect == "model":
        reference["platform"]["actual_device"] = "Different CUDA"
    elif defect == "device_count":
        candidate["platform"]["actual_device_count"] = 2
    elif defect == "cpu_dispatch":
        candidate["actual_dispatch"] = ["cpu", 0]
    else:
        reference["binary_extension"] = "torch/_C.py"
    with pytest.raises(ValueError):
        collector.platform_from_metadata(candidate, reference)


def test_missing_raw_case_cannot_publish_a_record(collector, tmp_path):
    with pytest.raises(ValueError, match="Incomplete"):
        collector.collect_case(tmp_path, {}, "a" * 40, "b" * 64, None, "", "", tmp_path)
