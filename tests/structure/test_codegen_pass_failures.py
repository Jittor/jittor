"""A code-generation pass may not swallow its own failures.

``opt/pass/**`` turns the generated C++ into the kernel that runs.  A pass that
catches everything and returns leaves the kernel in whatever state it was in --
and for a pass that exists to make something *correct*, that state is wrong
code.  FloatAtomicFixPass was the case that mattered: it converts a buffer into
the ordered-int form ``cuda_atomic_max(float*)`` requires, and its
``catch (...) { return; }`` meant an unrecognised name left an integer
atomicMax running over raw float bits, never converted back.

If a pass needs to ask a question, it must ask one (``try_get_op_var_by_name``),
not catch the answer.
"""
import re
import unittest
from pathlib import Path

import jittor


_OPT = Path(jittor.__file__).resolve().parent / "src" / "opt"

_CATCH_ALL = re.compile(r"catch\s*\(\s*\.\.\.\s*\)")

# Comments are blanked, not deleted: a comment explaining why a pass no longer
# catches everything spells the construct it is talking about, and reporting
# that as a violation is how this test first failed on the very change it was
# written for. Newlines survive so the reported line numbers still point at
# the file.
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def _without_comments(text):
    return _COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


class TestCodegenPassFailures(unittest.TestCase):
    def test_no_pass_catches_everything(self):
        violations = []
        for path in sorted(_OPT.rglob("*.cc")) + sorted(_OPT.rglob("*.h")):
            text = _without_comments(path.read_text(encoding="utf8"))
            for match in _CATCH_ALL.finditer(text):
                line = text[:match.start()].count("\n") + 1
                violations.append("%s:%d" % (path.relative_to(_OPT), line))
        assert not violations, (
            "catch (...) in a code-generation pass hides the failure and emits "
            "wrong code instead:\n  " + "\n  ".join(violations))


if __name__ == "__main__":
    unittest.main()
