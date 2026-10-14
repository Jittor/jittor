"""Every ``JT_*`` variable the source reads is one the startup check knows.

Importing jittor warns about ``JT_*`` variables it does not read, because a
misspelt one used to do nothing without a word (``JT_CACHE_NAME`` for
``JT_BUILD_CACHE_NAME``). That warning is only as good as its list: a variable
read somewhere the list does not cover would be reported as unused to the very
user who set it. So the list is checked against the source here -- a new
``"JT_..."`` literal has to be a native flag, an ``env_config`` setting, or
added to ``env_report.DIRECT_VARIABLES``.
"""

import re
import unittest
from pathlib import Path

import jittor as jt
from jittor._runtime import env_report

ROOT = Path(__file__).resolve().parents[3]
SCANNED = ("src", "backends", "python/jittor", "compat", "adapters")
SUFFIXES = {".py", ".cc", ".h", ".cu", ".cuh"}
LITERAL = re.compile(r"""["'](JT_[A-Z0-9_]*[A-Z0-9])["']""")


def _source_literals():
    found = {}
    for top in SCANNED:
        for path in (ROOT / top).rglob("*"):
            if (path.suffix not in SUFFIXES or "tests" in path.parts
                    or not path.is_file()):
                continue
            for name in LITERAL.findall(path.read_text(encoding="utf8", errors="replace")):
                found.setdefault(name, path.relative_to(ROOT))
    return found


class TestEnvVariableRegistry(unittest.TestCase):

    def test_every_literal_is_known(self):
        flag_names = [n for n in dir(jt.flags) if not n.startswith("_")]
        known = env_report.known_variables(flag_names)
        missing = sorted("%s (%s)" % (name, where)
                         for name, where in _source_literals().items()
                         if name not in known)
        self.assertEqual(missing, [],
                         "add these to env_report.DIRECT_VARIABLES, or the "
                         "startup check will call them unused")

    def test_a_namespace_slip_is_named(self):
        known = {"JT_BUILD_CACHE_NAME", "JT_LOG_V"}
        self.assertEqual(
            env_report.unread_variables(
                known, {"JT_CACHE_NAME": "x", "JT_BUILD_LOG_V": "1",
                        "JT_LOG_V": "1", "JT_TEST_RANK": "0", "PATH": "/bin"}),
            [("JT_BUILD_LOG_V", "JT_LOG_V"), ("JT_CACHE_NAME", "JT_BUILD_CACHE_NAME")])

    def test_an_unrelated_name_gets_no_guess(self):
        self.assertEqual(
            env_report.unread_variables({"JT_LOG_V"}, {"JT_SOMETHING_ELSE": "1"}),
            [("JT_SOMETHING_ELSE", None)])


if __name__ == "__main__":
    unittest.main()
