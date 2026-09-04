"""Shared pytest policy for the repository-only test suite."""

import importlib
import os
from collections import Counter
from pathlib import Path
import sys


TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

# `_helpers.child_process` reaches nothing that could pull jittor in, and owning
# `source_python_dir` there is what keeps this process' `sys.path` and the
# `PYTHONPATH` of every child process it starts from ever disagreeing.
from _helpers.child_process import source_python_dir  # noqa: E402

# Before any import that might reach jittor: this replaces what the pytest
# `pythonpath` ini option used to do, and has to happen just as early.
#
# `pyproject.toml` used to say `pythonpath = ["python"]`, which pytest resolves
# against *rootdir*. Right for the ordinary case and wrong for the two people
# actually hit: checking a second copy of the tree, which needed
# `-o pythonpath=...` on every single invocation, and deliberately exercising
# the installed package. `JITTOR_SOURCE_ROOT` names the checkout instead.
_python_dir = source_python_dir()
if _python_dir is not None:
    while _python_dir in sys.path:
        sys.path.remove(_python_dir)
    sys.path.insert(0, _python_dir)

import pytest

from _helpers.process_modes import TORCH_MODE_PATHS


def _torch_mode_is_active():
    """Which of the two process modes this session runs in.

    ``JITTOR_TORCH_SHIM`` decides, and nothing else does. It used to be decided
    by reading ``sys.argv``: a selection that mentioned a Torch-mode path
    switched the *whole process* into Torch mode, so adding one directory to a
    command changed the semantics of every other directory in it -- lazy
    execution, reduction defaults and gradient meaning all differ between the
    two. The same tests then passed or failed depending on how they were
    invoked, and ``-k``, xdist workers and IDE runners each produced a
    different answer from the same source.
    """
    module = sys.modules.get("torch")
    if module is not None and hasattr(module, "_torch_compat_install_context"):
        return True
    # Oracle sessions preload an independent binary PyTorch and stay native:
    # installing the Jittor shim there would try to replace a module graph
    # something else already owns.
    if os.environ.get("REAL_TORCH_SITE", "").strip():
        return False
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


def _torch_mode_paths_named_on_the_command_line(config):
    """Selected paths that belong to the other process mode.

    ``pytest_ignore_collect`` only filters what collection *walks into*; a path
    named on the command line is collected whatever it says. So a native
    session pointed straight at a Torch-mode file used to import it natively
    and fail somewhere confusing. Reading the selection here decides nothing --
    the mode is already fixed by ``JITTOR_TORCH_SHIM`` -- it only lets the
    session say what is wrong instead of failing on a module-scope import.
    """
    named = []
    for argument in config.args:
        raw = str(argument).split("::", 1)[0]
        path = Path(raw)
        if not path.exists():
            continue
        try:
            relative = path.resolve().relative_to(TEST_ROOT.parent).as_posix()
        except ValueError:
            continue
        if relative.rstrip("/").startswith(TORCH_MODE_PATHS):
            named.append(relative)
    return named


def pytest_sessionstart(session):
    found = [name for name in _LEGACY_SELECTION if name in os.environ]
    if found:
        guidance = "; ".join("{}: {}".format(name, _LEGACY_SELECTION[name]) for name in found)
        raise pytest.UsageError(
            "legacy jittor.test selection variables are unsupported; " + guidance
        )
    if not _torch_mode_is_active():
        named = _torch_mode_paths_named_on_the_command_line(session.config)
        if named:
            raise pytest.UsageError(
                "these paths run under Torch compatibility mode: %s. The mode is "
                "process-global -- it changes lazy execution, reduction defaults "
                "and what a gradient means -- so it is stated, not inferred from "
                "the command line: re-run with JITTOR_TORCH_SHIM=1. Mixing them "
                "into a native selection is what made the same test pass or fail "
                "depending on which other directory was named alongside it."
                % ", ".join(sorted(set(named))[:4])
            )


def _load_sensitive_tests_are_enabled(config):
    """Wall-clock upper bounds only mean something on an idle machine.

    A test that asserts ``elapsed < <constant>`` fails under load for reasons
    that have nothing to do with the code, and it fails *the same way* a real
    regression does -- red. The difference is where to look: a regression is in
    the diff, this is in ``uptime``. Three of these cost three different people
    an A/B bisect on one evening, so they are opt-in and the reason is in the
    skip message.
    """
    value = os.environ.get("JITTOR_TEST_LOAD_SENSITIVE", "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    option = getattr(config, "option", None)
    return "load_sensitive" in (getattr(option, "markexpr", "") or "")


def _manual_probes_are_enabled(config):
    """Manual probes are opt-in, by a variable rather than by how you invoked.

    They used to be enabled by "the selection was not a whole directory", which
    made the same file behave differently under ``pytest tests`` and
    ``pytest tests/integration`` -- and, worse, applied the decision *before*
    some of the markers were attached, so ``test_notebooks.py`` was never
    actually deselected and cost a whole-tree run 537 seconds.
    """
    value = os.environ.get("JITTOR_TEST_MANUAL", "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    # getattr twice: the structure tests call this hook with a stand-in config
    # that carries only the fields under test, and a marker rule should not
    # depend on the rest of pytest's Config existing.
    option = getattr(config, "option", None)
    return "manual" in (getattr(option, "markexpr", "") or "")


def pytest_collection_modifyitems(config, items):
    _snapshot_selected_files(config)
    network_enabled = _network_is_enabled() or config.getoption("--network")
    manual_enabled = _manual_probes_are_enabled(config)
    load_sensitive_enabled = _load_sensitive_tests_are_enabled(config)
    from _helpers import tiers
    for item in items:
        repo_relative = _relative_to_repo(item.fspath)
        _FILES_WITH_ITEMS.add(repo_relative)
        # The fast tier selects with `-m "not slow"`; the marker is attached from
        # one recorded list rather than from decorators scattered through the
        # tree, so "what a pull request waits for" is reviewable in one diff.
        # Attached, never skipped: outside the fast tier this changes nothing.
        if tiers.is_slow(repo_relative):
            item.add_marker(pytest.mark.slow)
        try:
            relative = Path(str(item.fspath)).resolve().relative_to(TEST_ROOT).parts
        except ValueError:
            continue
        parts = set(relative)
        for marker in _backend_markers(item, relative):
            item.add_marker(getattr(pytest.mark, marker))
        # Every marker this file attaches is attached before anything reads
        # them. The previous order attached `manual` to the notebook smokes
        # *after* deciding whether to skip manual probes, so the decision was
        # made against markers that did not exist yet.
        if "manual" in parts or "system" in parts:
            item.add_marker(pytest.mark.manual)
        if relative[-1] == "test_notebooks.py":
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.manual)
        if not manual_enabled and item.get_closest_marker("manual") is not None:
            # These probes drive Jupyter kernels or spawn their own processes
            # and do not survive being run after a thousand other tests have
            # already used the runtime.
            item.add_marker(
                pytest.mark.skip(
                    reason="manual probe; set JITTOR_TEST_MANUAL=1 or pass "
                    "-m manual to run it"
                )
            )
        if (not load_sensitive_enabled
                and item.get_closest_marker("load_sensitive") is not None):
            item.add_marker(
                pytest.mark.skip(
                    reason="asserts an upper bound on wall-clock time; this "
                    "machine's load decides the result. Run it on an idle box "
                    "with JITTOR_TEST_LOAD_SENSITIVE=1 or -m load_sensitive"
                )
            )
        if item.get_closest_marker("network") is not None and not network_enabled:
            item.add_marker(
                pytest.mark.skip(
                    reason="downloads external assets; enable with --network or "
                    "JITTOR_TEST_NETWORK=1"
                )
            )



def _input_generator():
    """``_helpers.common``, but only if the session already imported it.

    Importing it here would pull Jittor into every static structure test, so this
    stays a lookup: a test that never generated an input has nothing to seed.
    """
    return sys.modules.get("_helpers.common")


@pytest.fixture(autouse=True)
def deterministic_generated_inputs(request):
    """Seed generated inputs from the test's own nodeid, not from run order.

    ``make_tensor`` used to draw from a process-level counter, so the data a case
    received depended on how many draws happened before it. A case that failed in
    a full run got different data under ``-k`` and could not be reproduced.
    """
    common = _input_generator()
    if common is not None:
        common.begin_test_inputs(request.node.nodeid)
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Print the seeds a failing test drew, so it can be reproduced exactly."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return
    common = _input_generator()
    if common is None:
        return
    drawn = common.drawn_inputs()
    if drawn:
        report.sections.append(
            ("generated inputs (deterministic; rerun this nodeid to reproduce)",
             "\n".join(drawn)))


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

# --------------------------------------------------------------------------
#  Cross-file state leakage survey (report only)
# --------------------------------------------------------------------------
_STATE_LEAKS = []


def _state_leak_survey_enabled():
    """On by default: it only reports, and the report is the deliverable.

    Set ``JITTOR_TEST_STATE_LEAKS=0`` to switch it off -- worth doing when
    timing something, since the snapshot forces a ``gc.collect()`` per file.
    """
    value = os.environ.get("JITTOR_TEST_STATE_LEAKS", "").strip().lower()
    return value not in ("0", "false", "no", "off")


@pytest.fixture(autouse=True, scope="module")
def report_runtime_state_left_behind(request):
    """Name the test *file* that changed runtime state, not its next victim.

    Three known failures in this tree are one file leaving state for another
    (see ``_helpers/state_leaks``). The expensive part of each was that the
    symptom surfaced somewhere innocent, so this records the culprit instead.
    """
    if not _state_leak_survey_enabled():
        yield
        return
    from _helpers import state_leaks

    before = state_leaks.snapshot()
    yield
    after = state_leaks.snapshot()
    changes = state_leaks.differences(before, after)
    if changes:
        path = getattr(request.node, "nodeid", str(request.node))
        _STATE_LEAKS.append((path, changes))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _SELECTED_FILES:
        # The xdist controller collects nothing, so it never reached the
        # collection-time snapshot. Taking it here is the exception the
        # docstring on `_snapshot_selected_files` warns about, and it is the
        # right trade: a slightly later reading of the tree beats reporting
        # "no file collected nothing" because nothing was ever recorded.
        _snapshot_selected_files(config)
    _report_files_that_executed_nothing(terminalreporter, config)
    _report_skip_reason_buckets(terminalreporter)
    _report_reference_caches(terminalreporter)
    if _MISSING_REAL_TORCH:
        terminalreporter.write_sep(
            "=", "skipped for want of the PyTorch this session declared it has")
        terminalreporter.write_line(
            "JITTOR_REQUIRE_REAL_TORCH=1 says an independent PyTorch is "
            "configured. These cases disagreed, so the comparison this session "
            "exists for did not happen -- check REAL_TORCH_PYTHON and "
            "REAL_TORCH_SITE rather than the tests."
        )
        for nodeid, reason in _MISSING_REAL_TORCH:
            terminalreporter.write_line("%s  (%s)" % (nodeid, reason))
    if not _STATE_LEAKS:
        workers = getattr(config.option, "numprocesses", None)
        destination = os.environ.get("JITTOR_TEST_STATE_LEAK_REPORT", "").strip()
        if workers and destination:
            terminalreporter.write_sep(
                "=", "runtime state survey ran per worker")
            terminalreporter.write_line(
                "This is the xdist controller, which executed no test file. "
                "Each worker wrote its own report next to %s (one file per "
                "worker id); which files shared a process is a per-worker fact."
                % destination)
        return
    terminalreporter.write_sep("=", "runtime state left behind by a test file")
    terminalreporter.write_line(
        "Reported, not failed: each line is a file that changed process-wide "
        "state and did not put it back. A later file that reads that state "
        "fails instead, in a place that has nothing to do with the cause."
    )
    for path, changes in _STATE_LEAKS:
        terminalreporter.write_line(path)
        for change in changes:
            terminalreporter.write_line("    " + change)
    _write_state_leak_report()


def _write_state_leak_report(suffix=""):
    destination = os.environ.get("JITTOR_TEST_STATE_LEAK_REPORT", "").strip()
    if not destination:
        return
    with open(destination + suffix, "w") as handle:
        for path, changes in _STATE_LEAKS:
            for change in changes:
                handle.write("%s\t%s\n" % (path, change))


def _xdist_worker_id(config):
    return getattr(config, "workerinput", {}).get("workerid")


def pytest_sessionfinish(session, exitstatus):
    """Flush this worker's survey when the summary hook will not run here.

    ``pytest_terminal_summary`` runs on the xdist controller only, and the
    survey above is per process: the moment the gate went parallel the report
    would have gone silently empty, which is the exact shape of failure -- a
    check that stops checking without anyone noticing -- the survey exists to
    catch. Each worker writes its own file instead; which files shared a process
    is a per-worker fact anyway, so a merged report would be misleading.
    """
    worker = _xdist_worker_id(session.config)
    if worker is not None:
        _write_state_leak_report(suffix="." + worker)


# --------------------------------------------------------------------------
#  Per-file execution accounting (0.18)
# --------------------------------------------------------------------------
#: {relative path: {"executed": n, "skipped": n}} for this session.
_FILE_OUTCOMES = {}
_FILES_WITH_ITEMS = set()
_SKIP_REASON_BUCKETS = Counter()
_SKIP_BUCKET_ORDER = (
    "accelerator", "backend", "mpi", "torch", "network", "manual", "other")
_SKIP_BUCKET_PATTERNS = {
    "accelerator": ("cuda", "cudnn", "cublas", "cutt", "cusparse", "cufft",
                    "curand", "gpu", "rocm", "hip", "acl", "npu",
                    "ascend", "cann", "triton", "accelerator"),
    "backend": ("backend", "driver", "device library"),
    "mpi": ("mpi", "nccl", "hccl", "world size", "multi-rank"),
    "torch": ("torch",),
    "network": ("download", "dataset", "network", "internet"),
    "manual": ("manual",),
}


def _relative_to_repo(path):
    try:
        return Path(str(path)).resolve().relative_to(TEST_ROOT.parent).as_posix()
    except ValueError:
        return str(path)


def pytest_runtest_logreport(report):
    """Count what each test *file* actually executed.

    "The gate is green" and "the gate ran nothing" produce identical output, and
    this repository has already paid for that: the OpInfo backward battery --
    227 operators' derivative formulas -- instantiated zero cases in all three
    gates while all three reported success. Counting per file makes the
    difference visible.
    """
    relative = _relative_to_repo(report.fspath)
    # Also the xdist controller's only view of what was collected: the
    # collection hook runs in the workers, so without this the per-file
    # accounting below reports nothing at all the moment the gate goes
    # parallel -- a check that stops checking, silently.
    _FILES_WITH_ITEMS.add(relative)
    record = _FILE_OUTCOMES.setdefault(
        relative, {"executed": 0, "skipped": 0, "reasons": set()})
    # An expected failure ran, and it proved something: that a registered defect
    # is still there. pytest reports it as *skipped* with a `wasxfail` note, so
    # counting it as a skip would let a file whose only case is an xfail look
    # like a file that executed nothing -- and its "skip reason" is then the
    # assertion text, which can match an environment pattern by accident and
    # explain the emptiness away. Measured: an xfail whose message happened to
    # contain the word "dataset" was accepted as "this machine has no dataset".
    if report.when == "call" and (not report.skipped
                                  or hasattr(report, "wasxfail")):
        record["executed"] += 1
    elif report.skipped and report.when in ("setup", "call"):
        record["skipped"] += 1
        reason = _skip_reason(report)
        record["reasons"].add(reason)
        _SKIP_REASON_BUCKETS[classify_skip_reason_bucket(reason)] += 1
        if _real_torch_is_required() and _blames_missing_torch(reason):
            _MISSING_REAL_TORCH.append((report.nodeid, reason))


def _blames_missing_torch(reason):
    try:
        from _helpers.gate_scope import REAL_TORCH_PATTERNS
    except Exception:
        return False
    return any(pattern in reason for pattern in REAL_TORCH_PATTERNS)


def _skip_reason(report):
    """What the test said when it skipped, lowercased.

    pytest puts it in ``longrepr`` as ``(path, lineno, "Skipped: <reason>")``.
    The reason is the only thing that can distinguish "this box has no GPU"
    from "this entry quietly stopped testing anything".
    """
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2]).lower()
    return str(longrepr or "").lower()


def _real_torch_is_required():
    """Whether this session promised to have an independent PyTorch."""
    value = os.environ.get("JITTOR_REQUIRE_REAL_TORCH", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _accepted_skip_patterns():
    try:
        from _helpers.gate_scope import (
            ENVIRONMENT_SKIP_PATTERNS, REAL_TORCH_PATTERNS)
    except Exception:
        return ()
    if not _real_torch_is_required():
        return ENVIRONMENT_SKIP_PATTERNS
    # The inversion: a session that declares it has real PyTorch cannot also
    # accept "no torch" as an explanation, or it reports success for the one
    # thing it exists to check.
    return tuple(pattern for pattern in ENVIRONMENT_SKIP_PATTERNS
                 if pattern not in REAL_TORCH_PATTERNS)


def _environment_explains(reasons):
    """Whether every skip in a file blamed something this machine lacks."""
    patterns = _accepted_skip_patterns()
    if not reasons or not patterns:
        return False
    return all(
        any(pattern in reason for pattern in patterns)
        for reason in reasons
    )


#: ``(nodeid, reason)`` for skips that a real-PyTorch session must not have.
_MISSING_REAL_TORCH = []


def _files_that_executed_nothing():
    return sorted(
        path for path in _FILES_WITH_ITEMS
        if _FILE_OUTCOMES.get(path, {}).get("executed", 0) == 0
    )


#: Files the selection covered, snapshotted when collection ended.
_SELECTED_FILES = set()


def _snapshot_selected_files(config):
    """What the selection covers, recorded at collection time.

    Deliberately not recomputed at the end: on a shared machine the tree moves
    under a long run (other people land commits), and a file that appeared an
    hour after collection would otherwise be reported as "collected 0 tests" --
    a finding about nothing.
    """
    try:
        from _helpers.gate_scope import selected_files

        arguments = [str(argument) for argument in config.args]
        if not arguments:
            return
        _SELECTED_FILES.update(selected_files(TEST_ROOT.parent, arguments + [
            "--ignore=" + item for item in getattr(config.option, "ignore", []) or []
        ]))
    except Exception:
        pass


def _files_that_collected_nothing():
    """Selected test files that produced no test at all.

    Distinct from "everything skipped": a file that generates zero cases never
    reaches a skip either, so it is invisible in every count pytest prints.
    """
    return sorted(_SELECTED_FILES - _FILES_WITH_ITEMS)


def _execution_exemptions():
    """{path: reason} for files a gate may legitimately never execute."""
    try:
        from _helpers.gate_scope import EXECUTES_NOTHING

        return dict(EXECUTES_NOTHING)
    except Exception:
        return {}


def _requires_execution():
    value = os.environ.get("JITTOR_TEST_REQUIRE_EXECUTION", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _report_files_that_executed_nothing(terminalreporter, config):
    if getattr(config.option, "collectonly", False):
        # --collect-only executes nothing on purpose; every file would be listed.
        return
    silent = _files_that_executed_nothing()
    empty = _files_that_collected_nothing()
    if not silent and not empty:
        return
    exemptions = _execution_exemptions()
    terminalreporter.write_sep("=", "files this session proved nothing about")
    for path in empty:
        terminalreporter.write_line(
            "%s  collected 0 tests%s"
            % (path, _exemption_note(path, exemptions)))
    for path in silent:
        skipped = _FILE_OUTCOMES.get(path, {}).get("skipped", 0)
        terminalreporter.write_line(
            "%s  %d skipped, 0 executed%s"
            % (path, skipped, _exemption_note(path, exemptions)))
    if not _requires_execution():
        terminalreporter.write_line(
            "Reported only. Set JITTOR_TEST_REQUIRE_EXECUTION=1 (the gates do) "
            "to make an unexplained entry fail the run.")


def _exemption_note(path, exemptions):
    reason = exemptions.get(path)
    if reason:
        return "  -- expected here: " + reason
    record = _FILE_OUTCOMES.get(path, {})
    if _environment_explains(record.get("reasons", set())):
        return "  -- explained: " + "; ".join(sorted(record["reasons"]))[:120]
    return ""


def _skip_reason_buckets():
    """Return stable bucket counts in the documented priority order."""
    return tuple((bucket, _SKIP_REASON_BUCKETS.get(bucket, 0))
                 for bucket in _SKIP_BUCKET_ORDER)


def classify_skip_reason_bucket(reason):
    """Classify a skip reason using fixed, overlap-safe priority."""
    text = str(reason or "").lower()
    for bucket in _SKIP_BUCKET_ORDER[:-1]:
        if any(pattern in text for pattern in _SKIP_BUCKET_PATTERNS[bucket]):
            return bucket
    return "other"


def _other_skip_count():
    """Count skips classified as ``other`` for the execution threshold."""
    return _SKIP_REASON_BUCKETS.get("other", 0)


def _report_skip_reason_buckets(terminalreporter):
    """Print deterministic skip buckets for CI and local diagnostics."""
    buckets = _skip_reason_buckets()
    if not buckets:
        return
    terminalreporter.write_sep("=", "skip reasons")
    for bucket, count in buckets:
        terminalreporter.write_line("%d skipped: %s" % (count, bucket))
    terminalreporter.write_line("other skipped: %d" % _other_skip_count())


def _report_reference_caches(terminalreporter):
    """Say how many reference values this session reused instead of computing.

    A cache in front of an oracle is invisible in a pass/fail line: the run
    looks identical whether it derived the expected values or read them from
    disk. It is a legitimate optimisation (the values are device-independent and
    keyed by a content hash of the implementation, see
    ``_helpers/reference_cache.py``) and it still has to be *stated*, for the
    same reason 0.18 counts what each file executed -- a number nobody prints is
    a number nobody checks.
    """
    from _helpers import reference_cache

    used = [cache for cache in reference_cache.registry()
            if cache.hits or cache.misses or cache.writes]
    if not used:
        return
    terminalreporter.write_sep("=", "reference values reused from cache")
    for cache in used:
        terminalreporter.write_line(cache.summary())


def pytest_sessionfinish(session, exitstatus):
    """Fail a gate that proved nothing, in either of the two ways it can.

    A gate entry that only ever skips is indistinguishable from one that passes,
    which is how 227 operators' backward formulas went unverified in three green
    gates. Under ``JITTOR_TEST_REQUIRE_EXECUTION=1`` an entry has to either run
    something or be listed in ``gate_scope.EXECUTES_NOTHING`` with a reason.
    """
    if _MISSING_REAL_TORCH:
        # Not gated on JITTOR_TEST_REQUIRE_EXECUTION: a session that declared it
        # has real PyTorch and then skipped for want of it is misconfigured
        # whatever else it was asked to check.
        session.exitstatus = 1
    if not _requires_execution():
        return
    if getattr(session.config.option, "collectonly", False):
        return
    exemptions = _execution_exemptions()
    unexplained = [
        path for path in _files_that_executed_nothing()
        if path not in exemptions
        and not _environment_explains(
            _FILE_OUTCOMES.get(path, {}).get("reasons", set()))
    ]
    # A file that collected nothing never reached a skip, so nothing explains
    # it -- that is exactly the shape 0.01 had.
    unexplained += [path for path in _files_that_collected_nothing()
                    if path not in exemptions]
    if unexplained or _other_skip_count() > 0:
        session.exitstatus = 1
