"""The runner's result marker has to survive a noisy line.

`_parse_runner_result` scanned with `line.startswith("ECOSYSTEM_RESULT ")`.
Anything the child prints without a trailing newline pushes the marker into the
middle of a line, the scan misses it, and a run that produced a perfectly good
measurement is reported as "runner failed". On a 384-core machine numexpr
prints exactly such a warning -- it caps its thread pool at 64 and says so --
so the whole ecosystem comparison failed closed on a green result.

A false red on a measurement gate is worse than a late true red: it teaches
people to distrust the gate that also checks the numbers.
"""
import json
import unittest

from _ecosystem_harness import _result_from_stdout


class TestResultMarkerScan(unittest.TestCase):
    payload = {"case": "demo", "seconds": 0.125, "tensors": 3}

    def _line(self):
        return "ECOSYSTEM_RESULT " + json.dumps(self.payload)

    def test_marker_on_its_own_line(self):
        parsed = _result_from_stdout(self._line(), "demo", "py")
        self.assertEqual(parsed, self.payload)

    def test_marker_after_a_warning_with_no_newline(self):
        # The numexpr shape: a warning, no newline, then the marker.
        noisy = "NumExpr defaulting to 64 threads." + self._line()
        parsed = _result_from_stdout(noisy, "demo", "py")
        self.assertEqual(parsed, self.payload)

    def test_marker_among_other_lines(self):
        stdout = "\n".join(["loading", "warming up", self._line(), "done"])
        parsed = _result_from_stdout(stdout, "demo", "py")
        self.assertEqual(parsed, self.payload)

    def test_a_run_with_no_marker_still_fails(self):
        # The guard must not turn every failure into a pass.
        with self.assertRaises(AssertionError):
            _result_from_stdout("boom\ntraceback", "demo", "py")


if __name__ == "__main__":
    unittest.main()
