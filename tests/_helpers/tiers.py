"""Two tiers, one tree: what a pull request runs and what the nightly adds.

``gate_scope`` answers *which files a gate may run* (all of them). This answers
*which of them a pull request waits for*. The two are deliberately separate: a
test dropped from the fast tier is still gated, just later, and a test dropped
from ``gate_scope`` is not gated at all. Only the second is a hole.

The inversion is the same one 0.04 used, and for the same reason. The fast tier
is "the tree minus stated exceptions", so a new test file is in it the moment it
is written, and a slow one has to say -- here, in one place -- how long it takes
and why that is worth an entry. A hand-maintained include list drifts; this
cannot, because the default is inclusion.

Why seconds are written down
----------------------------
The tier's promise is a number ("a pull request gate under five minutes") and a
promise about a number needs a check. The obvious check -- assert the wall clock
of the run -- is the one thing this repository has learned not to do: an upper
bound on elapsed time fails on a loaded machine for reasons that have nothing to
do with the change, in exactly the way a real regression fails (see the
``load_sensitive`` marker). So the budget is checked *arithmetically* instead,
against measured costs recorded here, by ``tests/structure/test_gate_tiers.py``.
That check is a fact about the selection, not about the machine, and it is what
tells you the tier has drifted before the tier tells you by being slow.

The seconds are measurements, not estimates: ``--durations=0`` over a whole-tree
run on an idle machine, recorded with the run that produced them. They will
drift; the structure test asserts the arithmetic, not the accuracy, and a
re-measurement is a normal commit.
"""


#: ``(path, seconds, reason)`` -- a test file the fast tier does not run.
#:
#: ``path`` is a file, not a nodeid: node ids churn with every parametrisation
#: and a stale one silently selects nothing, which is the failure mode this list
#: exists to prevent. ``seconds`` is the file's measured total. ``reason`` says
#: what makes it slow, because "it is slow" is not reviewable -- a file that is
#: slow because it compiles two hundred kernels is a different decision from one
#: that is slow because it sleeps.
SLOW_FILES = ()


#: Wall-clock budget for the fast tier, in seconds (0.15: "smoke < 5 minutes").
#:
#: Covers *both* process modes, because that is what a pull request waits for:
#: Torch compatibility mode is process-global, so the fast tier is two pytest
#: invocations one after the other and the budget has to buy both.
SMOKE_BUDGET_SECONDS = 300.0

#: Workers the fast tier is sized for. ``noxfile.GATE_WORKERS`` is the same
#: number and ``tests/structure/test_gate_tiers.py`` checks the budget against
#: it, so the three have to agree. It describes the CI runner the promise is
#: made about, not the biggest machine anyone has.
SMOKE_WORKERS = 4

#: What one whole-tree run cost, per process mode. Measured, with the run named
#: below, so the budget arithmetic has real numbers under it.
#:
#: * ``total`` -- every test in that mode, summed from ``--durations=0``.
#: * ``longest_fast_file`` -- the longest single file the fast tier keeps.
#:   ``--dist loadfile`` cannot split a file, so this is a floor on the tier's
#:   wall clock however many workers it gets. Dividing a total by the worker
#:   count without it predicts three minutes for a tier holding a nine-minute
#:   file.
#: * ``startup`` -- interpreter start, jittor import and collection, paid once
#:   per invocation and not divisible by workers.
MEASURED = {
    "native": {"total": 0.0, "longest_fast_file": 0.0, "startup": 30.0},
    "torch": {"total": 0.0, "longest_fast_file": 0.0, "startup": 30.0},
}

#: Where MEASURED and the seconds in SLOW_FILES come from. Named so a
#: re-measurement can say what changed rather than only that it changed.
MEASURED_FROM = "not yet measured"


def slow_paths():
    return tuple(path for path, _seconds, _reason in SLOW_FILES)


def is_slow(relative_path):
    """Whether ``relative_path`` (posix, repo-relative) is out of the fast tier."""
    return relative_path in slow_paths()


def slow_seconds():
    return sum(seconds for _path, seconds, _reason in SLOW_FILES)


def session_of(path):
    """Which of the two process modes runs this file. Not a choice: Torch
    compatibility mode is process-global, so the path decides (``gate_scope``)."""
    from _helpers.process_modes import TORCH_MODE_PATHS

    return "torch" if path.startswith(TORCH_MODE_PATHS) else "native"


def _slow_seconds_in(session):
    return sum(seconds for path, seconds, _reason in SLOW_FILES
               if session_of(path) == session)


def predicted_session_seconds(session, workers=None):
    """What the fast tier should cost in one process mode.

    ``max(work / workers, longest single file)`` is the standard makespan bound
    for a list scheduler that cannot split a job, and ``--dist loadfile`` is
    exactly that. Plus the startup nobody parallelises away.
    """
    workers = workers or SMOKE_WORKERS
    measured = MEASURED[session]
    work = measured["total"] - _slow_seconds_in(session)
    return max(work / float(workers), measured["longest_fast_file"]) \
        + measured["startup"]


def predicted_smoke_seconds(workers=None):
    """Both modes, one after the other -- that is what a pull request waits."""
    return sum(predicted_session_seconds(session, workers) for session in MEASURED)
