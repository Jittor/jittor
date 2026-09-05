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

    def test_smoke_passes_not_slow_to_both_process_modes(self):
        """The PR smoke tier must defer only the measured slow files."""
        import ast

        source = (REPO_ROOT / "noxfile.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        smoke = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "smoke")
        fast_assignments = [
            node for node in smoke.body
            if isinstance(node, ast.Assign)
            and any(getattr(target, "id", None) == "fast"
                    for target in node.targets)
        ]
        self.assertEqual(len(fast_assignments), 1,
                         "smoke must define one fast marker argument")
        fast_constants = {
            node.value for node in ast.walk(fast_assignments[0].value)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("-m", fast_constants)
        self.assertIn("not slow", fast_constants)

        smoke_calls = [
            node for node in ast.walk(smoke)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_run_pytest_once"
        ]
        self.assertEqual(len(smoke_calls), 2,
                         "smoke must run native and torch process modes")
        for call in smoke_calls:
            self.assertGreaterEqual(len(call.args), 2)
            self.assertTrue(
                any(isinstance(node, ast.Name) and node.id == "fast"
                    for node in ast.walk(call.args[1])),
                "both smoke process modes must receive the fast marker args")

    def test_smoke_requires_execution_for_both_process_modes(self):
        """The fast tier must fail on an unexplained non-executing file."""
        import ast

        source = (REPO_ROOT / "noxfile.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        smoke = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "smoke")
        required_calls = [
            node for node in ast.walk(smoke)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_require_execution"
        ]
        self.assertEqual(len(required_calls), 1,
                         "smoke must require execution through its shared env")
        env_assignment = next(
            node for node in smoke.body
            if isinstance(node, ast.Assign)
            and any(getattr(target, "id", None) == "env"
                    for target in node.targets))
        self.assertIsInstance(env_assignment.value, ast.Call)
        self.assertEqual(getattr(env_assignment.value.func, "id", None),
                         "_require_execution")
        smoke_calls = [
            node for node in ast.walk(smoke)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_run_pytest_once"
        ]
        self.assertEqual(len(smoke_calls), 2)
        for call in smoke_calls:
            self.assertTrue(
                any(isinstance(node, ast.Name) and node.id in {"env", "torch_env"}
                    for node in ast.walk(call)),
                "native and torch smoke runs must receive required env")

    def test_smoke_uses_grouped_distribution_for_shared_module_files(self):
        """Smoke may split independent tests but must keep marked groups intact."""
        import ast

        source = (REPO_ROOT / "noxfile.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        smoke = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "smoke")
        calls = [
            node for node in ast.walk(smoke)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_xdist"
        ]
        self.assertEqual(len(calls), 1)
        distribution = next(
            keyword for keyword in calls[0].keywords
            if keyword.arg == "distribution")
        self.assertIsInstance(distribution.value, ast.Constant)
        self.assertEqual(distribution.value.value, "loadgroup")

        alias = (REPO_ROOT / "tests/compat/torch/test_torch_shim_aliases.py")
        alias_source = alias.read_text(encoding="utf-8")
        self.assertGreaterEqual(alias_source.count("xdist_group"), 4)


class TestBudget(unittest.TestCase):

    def test_budget_report_exposes_each_non_divisible_cost(self):
        report = tiers.budget_report()
        self.assertEqual(set(report["sessions"]), {"native", "torch"})
        self.assertEqual(report["workers"], tiers.SMOKE_WORKERS)
        self.assertEqual(report["configured_workers"], tiers.SMOKE_WORKERS)
        self.assertGreaterEqual(report["effective_cpus"], 1)
        self.assertGreaterEqual(report["threads_per_worker"], 1)
        self.assertAlmostEqual(
            report["predicted_seconds"], tiers.predicted_smoke_seconds())
        for item in report["sessions"].values():
            self.assertIn(item["bottleneck"], {"worker_work", "longest_file"})
            self.assertGreater(item["startup_seconds"], 0)

    def test_budget_report_caps_workers_to_runtime_cpu_quota(self):
        report = tiers.budget_report(workers=tiers.SMOKE_WORKERS + 100)
        self.assertEqual(report["configured_workers"], tiers.SMOKE_WORKERS + 100)
        self.assertEqual(report["workers"], tiers.SMOKE_WORKERS + 100)
        self.assertEqual(report["threads_per_worker"],
                         max(1, report["effective_cpus"] // report["workers"]))

    def test_budget_report_keeps_configured_and_runtime_workers_distinct(self):
        """A cgroup-capped run must report both values, not relabel it."""
        report = tiers.budget_report(workers=1, configured_workers=4)
        self.assertEqual(report["workers"], 1)
        self.assertEqual(report["configured_workers"], 4)

    def test_budget_report_rejects_non_positive_worker_counts(self):
        for kwargs in ({"workers": 0}, {"workers": -1},
                       {"workers": 1, "configured_workers": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    tiers.budget_report(**kwargs)

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

    def test_the_fast_tier_cost_was_measured_inside_the_fast_tier(self):
        """``fast_work`` has to come from a run at ``SMOKE_WORKERS`` workers.

        The tempting way to fill it in is ``total - deferred seconds``, and
        that number is always too small: the serial run gave every test all
        eight cores, while a worker in the tier gets two, so the same files
        cost more inside it. Measured, 407 s serial against 560 s in the tier
        on the native half and 287 s against 903 s on the torch half. The
        arithmetic ran on the serial figures once and predicted 254 s for a
        tier that measures 390 s -- passing the budget check while doing it.

        So the one property that distinguishes a real measurement from the
        subtraction is asserted: a parallel run cannot be cheaper.
        """
        for session, measured in sorted(tiers.MEASURED.items()):
            with self.subTest(session=session):
                serial = measured["total"] - tiers._slow_seconds_in(session)
                # Strictly greater, not >=: the value someone would paste in
                # is `total - deferred`, which is *equal* to `serial`. Equality
                # is the bug, so equality has to fail.
                self.assertGreater(
                    measured["fast_work"], serial,
                    "%s fast_work=%.0fs is not above the serial figure %.0fs, so "
                    "it was not measured in the tier's own configuration -- a "
                    "worker holding a quarter of the cores cannot be as fast as "
                    "the serial run. Re-measure with -n %d."
                    % (session, measured["fast_work"], serial, tiers.SMOKE_WORKERS))

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
