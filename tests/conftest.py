"""Shared pytest policy for the repository-only test suite."""

import os
from pathlib import Path
import sys

import pytest


TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))


_LEGACY_SELECTION = {
    "test_skip_l": "select an explicit path or nodeid",
    "test_skip_r": "select an explicit path or nodeid",
    "test_only": "select paths/nodeids or use -k",
    "test_skip": "use -k 'not ...' or a registered marker",
    "seperate_test": "use pytest isolation/timeout options",
}

_DEVICE_MARKERS = frozenset(("cpu", "cuda", "rocm", "npu"))


def _selected_test_devices():
    selected = os.environ.get("JITTOR_TEST_DEVICES", "")
    return tuple(
        device.strip() for device in selected.split(",") if device.strip() in _DEVICE_MARKERS
    )


def _backend_markers(item, relative):
    parts = set(relative)
    if "structure" in parts:
        return ("structure",)
    if "distributed" in parts:
        return ("mpi",)
    if relative == ("ops", "test_ops.py"):
        selected = _selected_test_devices()
        if len(selected) == 1:
            return selected
        device = getattr(getattr(item, "cls", None), "device_type", None)
        if device in _DEVICE_MARKERS:
            return (device,)
        return selected or ("cpu",)
    if "parity" in parts:
        selected = tuple(device for device in _selected_test_devices() if device in ("cuda", "npu"))
        return selected or ("cuda", "npu")
    if "triton" in parts:
        return ("cuda",)
    for device in ("cuda", "rocm", "npu", "cpu"):
        if device in parts:
            return (device,)
    return ("cpu",)


def pytest_sessionstart(session):
    found = [name for name in _LEGACY_SELECTION if name in os.environ]
    if found:
        guidance = "; ".join("{}: {}".format(name, _LEGACY_SELECTION[name]) for name in found)
        raise pytest.UsageError(
            "legacy jittor.test selection variables are unsupported; " + guidance
        )


def pytest_collection_modifyitems(items):
    for item in items:
        try:
            relative = Path(str(item.fspath)).resolve().relative_to(TEST_ROOT).parts
        except ValueError:
            continue
        parts = set(relative)
        for marker in _backend_markers(item, relative):
            item.add_marker(getattr(pytest.mark, marker))
        if "manual" in parts or "system" in parts:
            item.add_marker(pytest.mark.manual)
        if relative[-1] == "test_notebooks.py":
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.manual)


@pytest.fixture(autouse=True)
def rocm_backend(request):
    if request.node.get_closest_marker("rocm") is None:
        yield
        return

    import jittor as jt

    if not jt.compiler.has_rocm:
        pytest.skip("ROCm backend is unavailable")
    previous = jt.flags.use_rocm
    jt.flags.use_rocm = 1
    try:
        yield
    finally:
        jt.flags.use_rocm = previous
