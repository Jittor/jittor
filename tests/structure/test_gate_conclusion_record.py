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


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "tests"
TOOLS_DIR = REPO_ROOT / "tools"

if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from _helpers.child_process import run_python_child  # noqa: E402

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
            [str(TOOLS_DIR)]
            + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]),
    }
    arguments = [
        "-m", "pytest",
        "-p", "gate_conclusion_plugin", "-p", "no:cacheprovider",
        # Its own config: the repository's addopts and pythonpath would drag a
        # Jittor import into a test about bookkeeping.
        "-c", str(directory / "pytest.ini"),
        "-q", str(directory),
    ] + list(extra_arguments)
    completed = run_python_child(
        arguments, cwd=directory, env=environment, merge_stderr=True,
        timeout=300)
    assert out.exists(), (
        "the recorder wrote nothing; pytest said:\n" + completed.stdout[-3000:])
    return json.loads(out.read_text(encoding="utf-8")), completed.stdout


class TestGateConclusionRecord(unittest.TestCase):
    def test_a_parallel_session_records_what_it_collected(self):
        """The regression: ``collected`` was empty for every ``-n`` run."""
        with tempfile.TemporaryDirectory(prefix="gate-conclusion-") as raw:
            directory = Path(raw)
            _write_fixture_suite(directory)
            record, output = _record(directory, _PARALLEL)
        self.assertEqual(
            len(record["conclusions"]), _EXPECTED,
            "the fixture suite did not run as expected:\n" + output[-2000:])
        self.assertEqual(
            len(record["collected"]), _EXPECTED,
            "a parallel session recorded %d collected nodeids for %d "
            "conclusions; an empty collected set silently disables every "
            "lost-conclusion check in gate_conclusion_diff.compare"
            % (len(record["collected"]), len(record["conclusions"])))
        self.assertEqual(
            set(record["collected"]), set(record["conclusions"]),
            "collected and concluded disagree with nothing lost")

    def test_a_serial_session_still_records_what_it_collected(self):
        """The path that already worked, kept honest while the other is fixed."""
        with tempfile.TemporaryDirectory(prefix="gate-conclusion-") as raw:
            directory = Path(raw)
            _write_fixture_suite(directory)
            record, output = _record(directory, ())
        self.assertEqual(
            len(record["collected"]), _EXPECTED,
            "serial collection regressed:\n" + output[-2000:])
        self.assertEqual(set(record["collected"]), set(record["conclusions"]))

    def test_a_conclusion_dropped_under_xdist_is_reported_as_a_difference(self):
        """The property the tool is for, exercised end to end.

        Deselecting one test stands in for the way a real run loses an answer (a
        worker dies, a distribution mode drops an item). Before the fix this
        comparison printed ``IDENTICAL`` and exited zero.
        """
        with tempfile.TemporaryDirectory(prefix="gate-conclusion-") as raw:
            directory = Path(raw)
            _write_fixture_suite(directory)
            baseline, _ = _record(directory, _PARALLEL)
            (directory / "record.json").rename(directory / "baseline.json")
            candidate, _ = _record(
                directory,
                tuple(_PARALLEL) + ("--deselect", "test_two.py::test_d"))
            (directory / "record.json").rename(directory / "candidate.json")

            self.assertEqual(len(baseline["conclusions"]), _EXPECTED)
            self.assertEqual(len(candidate["conclusions"]), _EXPECTED - 1)

            completed = run_python_child(
                [str(TOOLS_DIR / "gate_conclusion_diff.py"), "compare",
                 str(directory / "baseline.json"),
                 str(directory / "candidate.json")],
                merge_stderr=True, timeout=300)

        self.assertNotEqual(
            completed.returncode, 0,
            "compare called a run that answered for one fewer nodeid "
            "equivalent:\n" + completed.stdout)
        self.assertNotIn("IDENTICAL", completed.stdout, completed.stdout)
        self.assertIn("test_d", completed.stdout, completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
