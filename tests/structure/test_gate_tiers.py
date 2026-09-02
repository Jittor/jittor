"""The fast tier's promise is a number, so the number is checked here.

0.15 splits the CPU gate in two: ``nox -s smoke`` for a pull request and
``nox -s cpu`` for everything. A split like that fails in three ways, and each
one is silent:

* the fast tier grows until it is not fast, one entry at a time;
* it gets fast by deferring most of the tree, and nobody notices what it stopped
  covering;
* an entry in the deferral list stops matching anything -- a renamed file, a
  typo -- and the list quietly means less than it says.

None of the three is visible in a green run, which is why they are asserted
statically here rather than by timing anything. **The budget check is
deliberately arithmetic**: an assertion on elapsed wall clock fails on a loaded
machine in exactly the way a real regression fails, and this repository has
already paid for that lesson three times (see the ``load_sensitive`` marker).
So the costs are measured once, written down in ``tests/_helpers/tiers.py`` with
the run that produced them, and the prediction is checked against the budget.
The measurement drifts; the arithmetic does not, and it is the arithmetic that
says the tier has been diluted.
"""

import unittest
from pathlib import Path

from _helpers import tiers
from _helpers.gate_scope import (
    native_arguments, selected_files, torch_arguments)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _gate_files():
    return (selected_files(REPO_ROOT, native_arguments())
            | selected_files(REPO_ROOT, torch_arguments()))


class TestSlowList(unittest.TestCase):

    def test_every_entry_names_a_file_the_gate_actually_runs(self):
        """A stale path defers nothing and says it defers something."""
        gated = _gate_files()
        for path, _seconds, _reason in tiers.SLOW_FILES:
            with self.subTest(path=path):
                self.assertTrue((REPO_ROOT / path).is_file(),
                                "%s does not exist" % path)
                self.assertIn(path, gated,
                              "%s is not reached by any gate, so deferring it "
                              "from the fast tier defers nothing" % path)

    def test_no_path_is_listed_twice(self):
        paths = tiers.slow_paths()
        self.assertEqual(len(paths), len(set(paths)))

    def test_every_entry_states_a_measured_cost_and_a_reason(self):
        for path, seconds, reason in tiers.SLOW_FILES:
            with self.subTest(path=path):
                self.assertGreater(seconds, 0, path)
                self.assertTrue(reason.strip(), path)
                # "slow" is not a reason: a file that is slow because it
                # compiles two hundred kernels is a different decision from one
                # that is slow because it sleeps, and only the second is a bug.
                self.assertNotEqual(reason.strip().lower(), "slow", path)


class TestFastTierIsStillWorthRunning(unittest.TestCase):

    #: The fast tier may defer files, not most of the tree. A tier that runs a
    #: fifth of the repository is a tier nobody should trust, however green.
    MINIMUM_FILE_SHARE = 0.85

    def test_the_fast_tier_still_covers_most_of_the_tree(self):
        gated = _gate_files()
        deferred = set(tiers.slow_paths()) & gated
        share = (len(gated) - len(deferred)) / float(len(gated))
        self.assertGreaterEqual(
            share, self.MINIMUM_FILE_SHARE,
            "the fast tier runs %d of %d gated files (%.0f%%); it is being made "
            "fast by deferring the tree rather than by deferring what is slow"
            % (len(gated) - len(deferred), len(gated), 100 * share))


class TestBudget(unittest.TestCase):

    def test_the_predicted_fast_tier_fits_the_budget(self):
        predicted = tiers.predicted_smoke_seconds()
        self.assertLessEqual(
            predicted, tiers.SMOKE_BUDGET_SECONDS,
            "the fast tier is predicted to take %.0fs at %d workers, over the "
            "%.0fs budget. Numbers from tests/_helpers/tiers.py -- either defer "
            "something (with its measured cost and a reason) or raise the "
            "budget deliberately."
            % (predicted, tiers.SMOKE_WORKERS, tiers.SMOKE_BUDGET_SECONDS))

    def test_the_prediction_accounts_for_the_slowest_single_file(self):
        """``--dist loadfile`` cannot split a file, so one file is a floor.

        Dividing a total by the worker count is the mistake this guards: it
        predicts three minutes for a tier containing one nine-minute file.
        """
        for session, measured in sorted(tiers.MEASURED.items()):
            with self.subTest(session=session):
                self.assertGreaterEqual(
                    tiers.predicted_session_seconds(session),
                    measured["longest_fast_file"],
                    session)

    def test_every_measured_session_is_a_session_the_gate_runs(self):
        self.assertEqual(set(tiers.MEASURED), {"native", "torch"})

    def test_the_budget_is_checked_against_the_worker_count_the_gate_uses(self):
        """Two files have to agree on one number, so read both.

        The budget above divides by ``tiers.SMOKE_WORKERS``. If ``noxfile``
        runs the tier with a different ``-n``, the arithmetic is checking a
        gate nobody runs -- and it would keep passing while doing so.
        """
        import ast

        source = (REPO_ROOT / "noxfile.py").read_text(encoding="utf-8")
        assignments = [
            node for node in ast.parse(source).body
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "GATE_WORKERS" for t in node.targets)
        ]
        self.assertEqual(len(assignments), 1, "GATE_WORKERS is assigned once")
        # `int(os.environ.get("JITTOR_GATE_WORKERS", "4"))` -- the default is
        # what CI uses, and the default is what the budget describes.
        default = ast.literal_eval(assignments[0].value.args[0].args[1])
        self.assertEqual(int(default), tiers.SMOKE_WORKERS)


if __name__ == "__main__":
    unittest.main()
