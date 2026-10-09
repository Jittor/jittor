"""The lost-conclusion detector has to work under xdist, or it detects nothing.

``tools/gate_conclusion_diff.py`` exists for one job: prove that a gate change
did not stop answering for some nodeid. It does that by writing down two sets --
what the session *collected* and what it *concluded* -- and reporting the
difference.

Under xdist the controller never collects. The workers do, and
``pytest_collection_modifyitems`` runs only there, so the recorder's
``collected`` list stayed empty for every parallel run. An empty ``collected``
does not make ``compare`` noisy, it makes it silent: both branches that report a
missing answer are gated on that set, so they skip every nodeid and the run is
declared ``IDENTICAL`` however many conclusions disappeared. Measured before the
fix, on a four-test fixture with one test deselected: ``compare`` printed
``passed 4 -> 3`` and ``IDENTICAL`` in the same output, and exited zero.

That is the failure mode the tool was written to catch (0.16 lost three of
twenty-six), in the configuration most likely to cause it (a dead worker), in
the configuration the smoke tier always uses (``-n 4``). So it is checked here
rather than assumed.

The fixture suite is four trivial tests in a temporary directory with its own
``pytest.ini``: this asserts a property of the recorder, and running it against
the real tree would pay a Jittor import in each of two children to learn nothing
extra. The children never import jittor, which is why this file costs seconds.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "tests"
TOOLS_DIR = REPO_ROOT / "tools"

if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from _helpers.child_process import run_python_child  # noqa: E402


def _requires_xdist():
    """The parallel half of this file needs the distribution it is about.

    `run_python_child` runs the fixture suite under `sys.executable`, so the
    import that matters is this interpreter's. pytest-xdist is a declared dev
    tool (requirements/dev-tools.txt) and the gates install it; a checkout
    without it cannot answer the question these two cases ask, and saying so is
    not the same as answering "yes".
    """
    pytest.importorskip(
        "xdist", reason="pytest-xdist (requirements/dev-tools.txt) is not installed here"
    )


_FIXTURE_TESTS = {
    "test_one.py": "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n",
    "test_two.py": "def test_c():\n    pass\n\n\ndef test_d():\n    pass\n",
}
_EXPECTED = 4
_PARALLEL = ("-n", "2", "--dist", "loadfile")


def _write_fixture_suite(directory):
    (directory / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    for name, source in _FIXTURE_TESTS.items():
        (directory / name).write_text(source, encoding="utf-8")


def _record(directory, extra_arguments):
    """Run the fixture suite under the recorder and return its JSON record."""
    out = directory / "record.json"
    environment = {
        "GATE_CONCLUSION_OUT": str(out),
        # The plugin lives in tools/, which is not importable by default; this
        # is the same prepend gate_conclusion_diff.py does for its own children.
        "PYTHONPATH": os.pathsep.join(
            [str(TOOLS_DIR)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        ),
    }
    arguments = [
        "-m",
        "pytest",
        "-p",
        "gate_conclusion_plugin",
        "-p",
        "no:cacheprovider",
        # Its own config: the repository's addopts and pythonpath would drag a
        # Jittor import into a test about bookkeeping.
        "-c",
        str(directory / "pytest.ini"),
        "-q",
        str(directory),
    ] + list(extra_arguments)
    completed = run_python_child(
        arguments, cwd=directory, env=environment, merge_stderr=True, timeout=300
    )
    assert out.exists(), "the recorder wrote nothing; pytest said:\n" + completed.stdout[-3000:]
    return json.loads(out.read_text(encoding="utf-8")), completed.stdout


class TestGateConclusionRecord(unittest.TestCase):
    def test_a_parallel_session_records_what_it_collected(self):
        """The regression: ``collected`` was empty for every ``-n`` run."""
        _requires_xdist()
        with tempfile.TemporaryDirectory(prefix="gate-conclusion-") as raw:
            directory = Path(raw)
            _write_fixture_suite(directory)
            record, output = _record(directory, _PARALLEL)
        self.assertEqual(
            len(record["conclusions"]),
            _EXPECTED,
            "the fixture suite did not run as expected:\n" + output[-2000:],
        )
        self.assertEqual(
            len(record["collected"]),
            _EXPECTED,
            "a parallel session recorded %d collected nodeids for %d "
            "conclusions; an empty collected set silently disables every "
            "lost-conclusion check in gate_conclusion_diff.compare"
            % (len(record["collected"]), len(record["conclusions"])),
        )
        self.assertEqual(
            set(record["collected"]),
            set(record["conclusions"]),
            "collected and concluded disagree with nothing lost",
        )

    def test_a_serial_session_still_records_what_it_collected(self):
        """The path that already worked, kept honest while the other is fixed."""
        with tempfile.TemporaryDirectory(prefix="gate-conclusion-") as raw:
            directory = Path(raw)
            _write_fixture_suite(directory)
            record, output = _record(directory, ())
        self.assertEqual(
            len(record["collected"]), _EXPECTED, "serial collection regressed:\n" + output[-2000:]
        )
        self.assertEqual(set(record["collected"]), set(record["conclusions"]))

    def test_a_conclusion_dropped_under_xdist_is_reported_as_a_difference(self):
        """The property the tool is for, exercised end to end.

        Deselecting one test stands in for the way a real run loses an answer (a
        worker dies, a distribution mode drops an item). Before the fix this
        comparison printed ``IDENTICAL`` and exited zero.
        """
        _requires_xdist()
        with tempfile.TemporaryDirectory(prefix="gate-conclusion-") as raw:
            directory = Path(raw)
            _write_fixture_suite(directory)
            baseline, _ = _record(directory, _PARALLEL)
            (directory / "record.json").rename(directory / "baseline.json")
            candidate, _ = _record(
                directory, tuple(_PARALLEL) + ("--deselect", "test_two.py::test_d")
            )
            (directory / "record.json").rename(directory / "candidate.json")

            self.assertEqual(len(baseline["conclusions"]), _EXPECTED)
            self.assertEqual(len(candidate["conclusions"]), _EXPECTED - 1)

            completed = run_python_child(
                [
                    str(TOOLS_DIR / "gate_conclusion_diff.py"),
                    "compare",
                    str(directory / "baseline.json"),
                    str(directory / "candidate.json"),
                ],
                merge_stderr=True,
                timeout=300,
            )

        self.assertNotEqual(
            completed.returncode,
            0,
            "compare called a run that answered for one fewer nodeid "
            "equivalent:\n" + completed.stdout,
        )
        self.assertNotIn("IDENTICAL", completed.stdout, completed.stdout)
        self.assertIn("test_d", completed.stdout, completed.stdout)


class TestDeselectionIsNotCollection(unittest.TestCase):
    """A deselected test is not a lost conclusion, and must not look like one.

    The recorder used ``pytest_collection_modifyitems`` to answer "what will
    this session run". It is registered from ``pytest_configure`` and therefore
    runs *before* pytest's own implementation of that hook -- which is where
    ``--deselect``, ``-k`` and ``-m`` drop items. So deselected nodeids landed in
    ``collected``, never concluded, and ``compare`` reported every one of them
    as ``COLLECTED BUT NO CONCLUSION``.

    That is this tool's central signal going off for a benign reason. It matters
    more than a cosmetic miscount: the signal is there to catch a genuinely
    lost answer, and one that fires on every ``--deselect`` is one its reader
    learns to skip -- the same way an always-``IDENTICAL`` comparison was worse
    than no comparison at all.

    It also made the two record kinds incomparable: the xdist hook has always
    reported post-deselection ids, so a serial record and a parallel record
    disagreed about what ``collected`` meant.
    """

    def test_a_deselected_test_is_not_recorded_as_collected(self):
        with tempfile.TemporaryDirectory(prefix="gate-conclusion-") as raw:
            directory = Path(raw)
            _write_fixture_suite(directory)
            record, output = _record(directory, ("--deselect", "test_two.py::test_d"))
        self.assertEqual(
            len(record["conclusions"]),
            _EXPECTED - 1,
            "the fixture suite did not run as expected:\n" + output[-2000:],
        )
        self.assertNotIn(
            "test_two.py::test_d",
            record["collected"],
            "a --deselect'ed nodeid was recorded as collected; compare would "
            "report it as a lost conclusion: %r" % (record["collected"],),
        )
        self.assertEqual(
            set(record["collected"]),
            set(record["conclusions"]),
            "collected and concluded must agree when nothing was lost",
        )

    def test_a_keyword_filtered_test_is_not_recorded_as_collected(self):
        """``-k`` deselects through the same hook, so it has the same bug."""
        with tempfile.TemporaryDirectory(prefix="gate-conclusion-") as raw:
            directory = Path(raw)
            _write_fixture_suite(directory)
            record, output = _record(directory, ("-k", "not test_d"))
        self.assertEqual(len(record["conclusions"]), _EXPECTED - 1, output[-2000:])
        self.assertEqual(set(record["collected"]), set(record["conclusions"]))

    def test_deselection_and_a_real_loss_are_still_distinguishable(self):
        """The fix must not buy quiet by weakening the check.

        A record that deselects one test *and* is missing an answer for another
        must report exactly one unconcluded nodeid: the missing one.

        The loss is injected into the record rather than caused by killing a
        worker. A first attempt did kill one (``os._exit`` inside the call
        phase) and cost **300 s** before timing out -- xdist waits on the dead
        node -- which is far too expensive for a gate and tested pytest's
        crash handling more than this plugin's bookkeeping. Editing the record
        reproduces the state ``compare`` has to interpret, deterministically.
        """
        with tempfile.TemporaryDirectory(prefix="gate-conclusion-") as raw:
            directory = Path(raw)
            _write_fixture_suite(directory)
            record, output = _record(directory, ("--deselect", "test_two.py::test_c"))
            self.assertEqual(len(record["conclusions"]), _EXPECTED - 1, output[-2000:])

            baseline = directory / "baseline.json"
            candidate = directory / "candidate.json"
            baseline.write_text(json.dumps(record), encoding="utf-8")
            damaged = json.loads(json.dumps(record))
            del damaged["conclusions"]["test_one.py::test_b"]
            candidate.write_text(json.dumps(damaged), encoding="utf-8")

            completed = run_python_child(
                [
                    str(TOOLS_DIR / "gate_conclusion_diff.py"),
                    "compare",
                    str(baseline),
                    str(candidate),
                ],
                merge_stderr=True,
                timeout=300,
            )

        # The deselected nodeid is in neither record's collected set, so it is
        # not reported at all...
        self.assertNotIn("test_two.py::test_c", record["collected"])
        self.assertNotIn("test_c", completed.stdout, completed.stdout)
        # ...while the missing answer is reported, and fails the comparison.
        self.assertNotEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("test_one.py::test_b", completed.stdout, completed.stdout)


class TestExpectNew(unittest.TestCase):
    """``--expect-new``: adding tests must not require reading the diff by eye.

    10.18 is the case that needed it -- it adds test files, so a plain
    ``compare`` reports every new nodeid as a difference and exits non-zero,
    which means "did my new tests displace an old conclusion?" gets answered by
    scanning a list. That is the habit this whole tool exists to replace, so the
    allowance is explicit, narrow, and itself checked.
    """

    def _records(self, directory):
        """Baseline without ``test_two.py``, candidate with it: tests added."""
        _write_fixture_suite(directory)
        baseline, _ = _record(directory, ("--ignore", str(directory / "test_two.py")))
        (directory / "record.json").rename(directory / "baseline.json")
        candidate, _ = _record(directory, ())
        (directory / "record.json").rename(directory / "candidate.json")
        return baseline, candidate

    def _compare(self, directory, extra=()):
        return run_python_child(
            [
                str(TOOLS_DIR / "gate_conclusion_diff.py"),
                "compare",
                str(directory / "baseline.json"),
                str(directory / "candidate.json"),
            ]
            + list(extra),
            merge_stderr=True,
            timeout=300,
        )

    def test_added_tests_are_a_difference_unless_they_are_named(self):
        with tempfile.TemporaryDirectory(prefix="gate-conclusion-") as raw:
            directory = Path(raw)
            baseline, candidate = self._records(directory)
            self.assertEqual(len(baseline["conclusions"]), 2)
            self.assertEqual(len(candidate["conclusions"]), _EXPECTED)
            silent = self._compare(directory)
            named = self._compare(
                directory,
                ("--expect-new", "test_two.py::test_c", "--expect-new", "test_two.py::test_d"),
            )

        # Unnamed, the addition is reported -- the default stays strict.
        self.assertNotEqual(silent.returncode, 0, silent.stdout)
        self.assertIn("NEWLY COLLECTED", silent.stdout, silent.stdout)
        # Named, the two additions are accounted for and nothing else moved.
        self.assertEqual(named.returncode, 0, named.stdout)
        self.assertIn("IDENTICAL", named.stdout, named.stdout)
        self.assertIn("accounted for by --expect-new", named.stdout, named.stdout)

    def test_naming_one_addition_does_not_excuse_another(self):
        """The allowance is per nodeid, not a blanket "new tests are fine"."""
        with tempfile.TemporaryDirectory(prefix="gate-conclusion-") as raw:
            directory = Path(raw)
            self._records(directory)
            partial = self._compare(directory, ("--expect-new", "test_two.py::test_c"))
        self.assertNotEqual(partial.returncode, 0, partial.stdout)
        self.assertIn("test_d", partial.stdout, partial.stdout)

    def test_a_stale_expect_new_is_itself_an_error(self):
        """A name that matches nothing would silently widen the comparison."""
        with tempfile.TemporaryDirectory(prefix="gate-conclusion-") as raw:
            directory = Path(raw)
            self._records(directory)
            stale = self._compare(
                directory,
                (
                    "--expect-new",
                    "test_two.py::test_c",
                    "--expect-new",
                    "test_two.py::test_d",
                    "--expect-new",
                    "test_two.py::test_renamed_away",
                ),
            )
        self.assertNotEqual(stale.returncode, 0, stale.stdout)
        self.assertIn("EXPECTED NEW but not collected", stale.stdout, stale.stdout)

    def test_expect_new_does_not_hide_a_lost_conclusion(self):
        """The failure the tool is for must survive the new flag.

        The reverse case that matters: a run that both adds a test and loses an
        answer for an old one must still fail, even with the addition named.
        """
        with tempfile.TemporaryDirectory(prefix="gate-conclusion-") as raw:
            directory = Path(raw)
            _write_fixture_suite(directory)
            _record(directory, ("--ignore", str(directory / "test_two.py")))
            (directory / "record.json").rename(directory / "baseline.json")
            # Adds test_two.py's two tests and drops one of test_one.py's.
            _record(directory, ("--deselect", "test_one.py::test_b"))
            (directory / "record.json").rename(directory / "candidate.json")
            completed = self._compare(
                directory,
                ("--expect-new", "test_two.py::test_c", "--expect-new", "test_two.py::test_d"),
            )
        self.assertNotEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("test_b", completed.stdout, completed.stdout)


class TestAcceleratorExecutionCount(unittest.TestCase):
    """Generated device classes must satisfy the existing execution gate."""

    _CORE_METHODS = (
        "test_control_chain_on_a_shared_data_producer",
        "test_normal_data_sharing_keeps_both_readers",
        "test_unrelated_broadcast_does_not_enter_reduction",
    )

    def setUp(self):
        from collections import Counter
        from unittest import mock
        from _helpers import pytest_policy

        self.policy = pytest_policy
        for name, value in (
            ("_ACCELERATOR_EXECUTED", 0),
            ("_FILES_WITH_ITEMS", set()),
            ("_FILE_OUTCOMES", {}),
            ("_SKIP_REASON_BUCKETS", Counter()),
            ("_OTHER_SKIP_REASONS", Counter()),
            ("_MISSING_REAL_TORCH", []),
        ):
            patcher = mock.patch.object(self.policy, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        environment = mock.patch.dict(
            os.environ,
            {
                "JITTOR_TEST_ACCELERATOR_MIN_EXECUTED": "3",
                "JITTOR_TEST_REQUIRE_EXECUTION": "0",
                "JITTOR_REQUIRE_REAL_TORCH": "0",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        flush = mock.patch.object(self.policy, "_flush_worker_state_leaks")
        flush.start()
        self.addCleanup(flush.stop)

    def _report(self, nodeid, when="call", outcome="passed"):
        from types import SimpleNamespace

        return SimpleNamespace(
            nodeid=nodeid,
            fspath=TEST_ROOT / "core/test_control_dependency_fusion.py",
            when=when,
            skipped=outcome == "skipped",
            passed=outcome == "passed",
            failed=outcome == "failed",
            longrepr=("fixture.py", 1, "Skipped: no NPU"),
        )

    def _finish(self):
        from types import SimpleNamespace

        lines = []
        reporter = SimpleNamespace(write_line=lines.append)
        manager = SimpleNamespace(getplugin=lambda name: reporter)
        session = SimpleNamespace(
            config=SimpleNamespace(
                pluginmanager=manager, option=SimpleNamespace(collectonly=False)
            ),
            exitstatus=0,
        )
        self.policy.pytest_sessionfinish(session, 0)
        return session, lines

    def _core_nodes(self):
        return [
            "tests/core/test_control_dependency_fusion.py::TestControlDependencyFusionNPU::"
            + method
            for method in self._CORE_METHODS
        ]

    def test_the_helper_generated_class_names_are_counted(self):
        import ast

        # Evaluate the helper's actual class-name expression without importing
        # Jittor: this test must follow the factory, not duplicate its names.
        helper = ast.parse((TEST_ROOT / "_helpers/device_types.py").read_text(encoding="utf-8"))
        factory = next(
            node
            for node in helper.body
            if isinstance(node, ast.FunctionDef) and node.name == "instantiate_device_type_tests"
        )
        assignment = next(
            node
            for node in ast.walk(factory)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "cls_name" for target in node.targets
            )
        )
        expression = ast.Expression(body=assignment.value)
        ast.fix_missing_locations(expression)
        for device in ("npu", "cuda", "rocm", "cpu"):
            name = eval(
                compile(expression, "device_types.py", "eval"),
                {"__builtins__": {}},
                {"stem": "ControlDependencyFusion", "device_type": device},
            )
            with self.subTest(device=device):
                report = self._report("tests/core/test_plain.py::" + name + "::test_graph")
                self.assertEqual(self.policy._is_accelerator_case(report), device != "cpu")

    def test_only_complete_generated_class_components_are_new_matches(self):
        negatives = (
            "tests/core/test_plain.py::testNPU",
            "tests/core/test_plain.py::TestNPUCPU::test_graph",
            "tests/core/test_plain.py::TestNPUExtra::test_graph",
            "tests/core/test_plain.py::ExampleNPU::test_graph",
            "tests/core/test_plain.py::TestFooNPUExtra::test_graph",
            "tests/core/test_plain.py::TestFooCPU::test_graph[TestFooNPU]",
            "tests/core/test_plain.py::TestFooCPU::test_graph[TestFooNPU::test_graph]",
        )
        for nodeid in negatives:
            with self.subTest(nodeid=nodeid):
                self.assertFalse(self.policy._is_accelerator_case(self._report(nodeid)))
        for nodeid in (
            "tests/cuda/test_plain.py::test_graph",
            "tests/core/test_plain.py::test_acl_graph",
            "tests/core/test_plain.py::test_hccl_graph",
            "tests/core/test_plain.py::test_rocm_graph",
        ):
            with self.subTest(legacy_nodeid=nodeid):
                self.assertTrue(self.policy._is_accelerator_case(self._report(nodeid)))

    def test_collecting_or_skipping_three_cases_still_fails(self):
        # Collection alone supplies no call report, regardless of its nodeids.
        self.policy._FILES_WITH_ITEMS.add("tests/core/test_control_dependency_fusion.py")
        session, lines = self._finish()
        self.assertEqual(session.exitstatus, 1)
        self.assertIn("executed 0 accelerator cases; requires at least 3", lines[0])
        for nodeid in self._core_nodes():
            self.policy.pytest_runtest_logreport(self._report(nodeid, "setup"))
            self.policy.pytest_runtest_logreport(self._report(nodeid, "call", "skipped"))
            self.policy.pytest_runtest_logreport(self._report(nodeid, "teardown"))
        self.assertEqual(self.policy._ACCELERATOR_EXECUTED, 0)
        self.assertEqual(self._finish()[0].exitstatus, 1)

    def test_three_synthetic_call_reports_meet_the_unchanged_threshold(self):
        for nodeid in self._core_nodes():
            self.policy.pytest_runtest_logreport(self._report(nodeid, "setup"))
            self.policy.pytest_runtest_logreport(self._report(nodeid))
            self.policy.pytest_runtest_logreport(self._report(nodeid, "teardown"))
        self.assertEqual(self.policy._ACCELERATOR_EXECUTED, 3)
        self.assertEqual(self._finish()[0].exitstatus, 0)
        records = list(self.policy._FILE_OUTCOMES.values())
        self.assertEqual(records[0]["executed"], 3)

    def test_two_accelerator_calls_and_a_cpu_call_do_not_meet_three(self):
        nodes = self._core_nodes()
        for nodeid in nodes[:2]:
            self.policy.pytest_runtest_logreport(self._report(nodeid))
        self.policy.pytest_runtest_logreport(
            self._report(nodes[2].replace("FusionNPU", "FusionCPU"))
        )
        self.assertEqual(self.policy._ACCELERATOR_EXECUTED, 2)
        self.assertEqual(self._finish()[0].exitstatus, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
