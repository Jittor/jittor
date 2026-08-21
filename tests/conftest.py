"""Shared pytest policy for the repository-only test suite."""

import importlib
import os
from pathlib import Path
import sys

import pytest

from _helpers.process_modes import TORCH_MODE_PATHS


#: Whether this session selected whole directories rather than named files.
SELECTION_IS_BROAD = False


def _select_torch_mode_for_test_process():
    """Keep native and Torch compatibility semantics in separate processes."""

    global SELECTION_IS_BROAD

    # Oracle tests preload an independent binary PyTorch and intentionally
    # keep Jittor native.  Installing the Jittor Torch shim in that same
    # process would try to replace an already-owned ``torch`` module graph.
    if os.environ.get("REAL_TORCH_SITE", "").strip():
        return

    repo_root = Path(__file__).resolve().parents[1]
    selected = []
    for arg in sys.argv[1:]:
        if not arg or arg.startswith("-"):
            continue
        raw_path = arg.split("::", 1)[0]
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            normalized = path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            normalized = path.resolve().as_posix()
        selected.append(normalized.rstrip("/"))
    broad_roots = (".", "tests")
    SELECTION_IS_BROAD = not selected or any(path in broad_roots for path in selected)
    if SELECTION_IS_BROAD:
        # A broad selection runs the native suite. Torch mode is process-global
        # and changes lazy execution, reduction defaults and gradient
        # semantics, so switching the whole tree into it made ordinary native
        # tests fail. The Torch-mode paths are skipped here and covered by
        # their own session; ``tools/run_test_suite.py`` runs both and reports
        # a combined result.
        return
    if any(path.startswith(TORCH_MODE_PATHS) for path in selected):
        os.environ.setdefault("JITTOR_TORCH_SHIM", "1")


_select_torch_mode_for_test_process()


def _torch_mode_is_active():
    module = sys.modules.get("torch")
    if module is not None and hasattr(module, "_torch_compat_install_context"):
        return True
    value = os.environ.get("JITTOR_TORCH_SHIM", "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _preload_real_torch():
    """Claim the Torch namespace before Jittor composition in oracle sessions."""
    raw_site = os.environ.get("REAL_TORCH_SITE", "").strip()
    if not raw_site:
        return
    site = Path(raw_site).expanduser().resolve()
    if not site.is_dir():
        raise pytest.UsageError("REAL_TORCH_SITE is not a directory: {}".format(site))
    if "jittor" in sys.modules:
        raise pytest.UsageError("REAL_TORCH_SITE must be configured before Jittor is imported")
    site_text = str(site)
    sys.path[:] = [path for path in sys.path if path != site_text]
    sys.path.insert(0, site_text)
    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        raise pytest.UsageError("failed to preload real Torch from {}: {}".format(site, error))
    finally:
        sys.path[:] = [path for path in sys.path if path != site_text]
        sys.path.append(site_text)
    origin = Path(getattr(torch, "__file__", "")).resolve()
    binary_origin = Path(getattr(getattr(torch, "_C", None), "__file__", "")).resolve()
    if (
        getattr(torch, "__name__", None) != "torch"
        or site not in origin.parents
        or site not in binary_origin.parents
        or hasattr(torch, "_torch_compat_install_context")
    ):
        raise pytest.UsageError(
            "REAL_TORCH_SITE did not provide independent binary PyTorch: {}, {}".format(
                origin, binary_origin
            )
        )


_preload_real_torch()


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


def _network_is_enabled():
    """Whether tests that fetch external assets may run.

    They are opt-in because the failure mode without a reachable host is not a
    quick error: the download blocks until the suite-level timeout fires, so a
    single unreachable asset can cost fifteen minutes and report as a framework
    failure. Set ``JITTOR_TEST_NETWORK=1`` (or pass ``--network``) to enable.
    """
    value = os.environ.get("JITTOR_TEST_NETWORK", "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def pytest_addoption(parser):
    parser.addoption(
        "--network",
        action="store_true",
        default=False,
        help="run tests that download external assets",
    )


def pytest_ignore_collect(collection_path, config):
    """Keep Torch-mode paths out of a native session entirely.

    Marking their tests as skipped is not enough: several of these modules
    import ``jittor.torch_compat`` at module scope, so a native session fails
    during collection before any marker is applied. ``tools/run_test_suite.py``
    runs them in their own session.

    pluggy matches these parameter names against the hookspec and rejects any
    it does not recognise, so the signature has to name only arguments every
    supported pytest offers: ``collection_path`` arrived in 7.0 and the legacy
    ``path`` was removed in 9.0.
    """
    if _torch_mode_is_active():
        return None
    target = collection_path
    if target is None:
        return None
    try:
        relative = Path(str(target)).resolve().relative_to(TEST_ROOT.parent).as_posix()
    except ValueError:
        return None
    if relative.startswith(TORCH_MODE_PATHS):
        return True
    return None


def pytest_sessionstart(session):
    found = [name for name in _LEGACY_SELECTION if name in os.environ]
    if found:
        guidance = "; ".join("{}: {}".format(name, _LEGACY_SELECTION[name]) for name in found)
        raise pytest.UsageError(
            "legacy jittor.test selection variables are unsupported; " + guidance
        )


def pytest_collection_modifyitems(config, items):
    network_enabled = _network_is_enabled() or config.getoption("--network")
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
        if SELECTION_IS_BROAD and item.get_closest_marker("manual") is not None:
            # The marker is documented as "must be selected explicitly". Some of
            # these probes drive Jupyter kernels or spawn their own processes and
            # do not survive being run after a thousand other tests have already
            # used the runtime, so a whole-tree run deselects them and a named
            # path still runs them.
            item.add_marker(
                pytest.mark.skip(
                    reason="manual probe; select its path explicitly to run it"
                )
            )
        if item.get_closest_marker("network") is not None and not network_enabled:
            item.add_marker(
                pytest.mark.skip(
                    reason="downloads external assets; enable with --network or "
                    "JITTOR_TEST_NETWORK=1"
                )
            )
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
