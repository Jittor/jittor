"""``gen_jit_flags`` scans C++ sources for DEFINE_FLAG, and must skip comments.

It regexes the sources without removing comments, so a *commented-out*
definition was indistinguishable from a live one. src/utils/flags.cc was 27
lines of commented-out flags, and it gave twelve flags a second definition;
the scanner keeps the first one it meets and drops the rest, so which default
and doc string reached ``jt.flags`` depended on the order glob returned the
files in.
"""

import glob
import re
from pathlib import Path
import unittest

from jittor.compiler import strip_cxx_comments


REPO = Path(__file__).resolve().parents[2]
SOURCES = str(REPO / "python" / "jittor" / "src")
DEFINE_FLAG = re.compile(r"DEFINE_FLAG(_WITH_SETTER)?\((.*?)\);", re.DOTALL)


class TestStripCxxComments(unittest.TestCase):

    def test_line_comment(self):
        self.assertNotIn("hidden", strip_cxx_comments("int a; // hidden\nint b;"))

    def test_block_comment(self):
        stripped = strip_cxx_comments('/* DEFINE_FLAG(int, x, 0, "d"); */\n'
                                      'DEFINE_FLAG(int, y, 0, "d");')
        self.assertNotIn("x", stripped)
        self.assertIn("y", stripped)

    def test_a_comment_inside_a_string_is_not_a_comment(self):
        for text in ('const char* s = "// not a comment";',
                     'const char* s = "a /* b */ c";'):
            self.assertEqual(strip_cxx_comments(text), text)

    def test_an_unterminated_block_comment_does_not_hang(self):
        self.assertNotIn("x", strip_cxx_comments("/* x"))


class TestNoDuplicateFlagDefinitions(unittest.TestCase):

    def test_every_flag_is_defined_exactly_once(self):
        """Two definitions mean the effective default depends on glob order."""
        seen = {}
        for path in sorted(glob.glob(SOURCES + "/**/*.cc", recursive=True)):
            with open(path, "rb") as f:
                source = strip_cxx_comments(f.read().decode("utf8"))
            for _, args in DEFINE_FLAG.findall(source):
                name = args.split(",")[1].strip()
                seen.setdefault(name, []).append(path)
        duplicates = {name: files for name, files in seen.items()
                      if len(files) > 1}
        self.assertEqual(duplicates, {})


if __name__ == "__main__":
    unittest.main()
