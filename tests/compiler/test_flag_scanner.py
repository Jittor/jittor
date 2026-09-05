"""``gen_jit_flags`` scans C++ sources for DEFINE_FLAG, and must skip comments.

It regexes the sources without removing comments, so a *commented-out*
definition was indistinguishable from a live one. src/utils/flags.cc was 27
lines of commented-out flags, and it gave twelve flags a second definition;
the scanner keeps the first one it meets and drops the rest, so which default
and doc string reached ``jt.flags`` depended on the order glob returned the
files in.
"""

import ast
import glob
import os
import re
import runpy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[2]
SOURCES = str(REPO / "python" / "jittor" / "src")
DEFINE_FLAG = re.compile(r"DEFINE_(RUNTIME_)?FLAG(_WITH_SETTER)?\((.*?)\);", re.DOTALL)


def _flag_functions(**environment):
    # These generators need source paths, not an imported/compiled Jittor core.
    source = REPO / "python" / "jittor" / "compiler.py"
    tree = ast.parse(source.read_text(encoding="utf8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"strip_cxx_comments", "gen_jit_flags"}]
    namespace = dict(glob=glob, os=os, re=re,
                     LOG=SimpleNamespace(vv=lambda *args: None,
                                         vvvv=lambda *args: None))
    namespace.update(environment)
    namespace["flag_category"] = runpy.run_path(
        str(REPO / "python/jittor/_runtime/flag_policy.py"))["flag_category"]
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace


strip_cxx_comments = _flag_functions()["strip_cxx_comments"]


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
            for _, _, args in DEFINE_FLAG.findall(source):
                name = args.split(",")[1].strip()
                seen.setdefault(name, []).append(path)
        duplicates = {name: files for name, files in seen.items()
                      if len(files) > 1}
        self.assertEqual(duplicates, {})


class TestRuntimeFlagGeneration(unittest.TestCase):

    def test_runtime_storage_getters_keep_aliases_and_setters(self):
        for has_cuda in (False, True):
            with self.subTest(has_cuda=has_cuda), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "src" / "runtime").mkdir(parents=True)
                (root / "gen").mkdir()
                (root / "src" / "runtime" / "device_state.cc").write_text(
                    'DEFINE_RUNTIME_FLAG_WITH_SETTER(int, use_cuda, 0, "backend");\n'
                    'DEFINE_RUNTIME_FLAG(int, sync_run, 1, "sync");\n'
                    'DEFINE_FLAG(int, lazy_execution, 1, "schedule");\n'
                    '// DEFINE_RUNTIME_FLAG(int, ignored, 0, "comment");\n',
                    encoding="utf8")
                namespace = _flag_functions(jittor_path=str(root), cache_path=str(root),
                                            has_cuda=has_cuda)
                namespace["gen_jit_flags"]()
                generated = (root / "gen" / "jit_flags.h").read_text(encoding="utf8")
                self.assertIn("DECLARE_RUNTIME_FLAG(int, use_cuda);", generated)
                self.assertIn("DECLARE_RUNTIME_FLAG(int, sync_run);", generated)
                self.assertIn("_get_use_cuda() { return runtime_flag_use_cuda(); }", generated)
                self.assertIn("_get_sync_run() { return runtime_flag_sync_run(); }", generated)
                self.assertIn("_set_use_cuda(int v) { set_use_cuda(v); }", generated)
                self.assertIn("_set_use_cuda(bool v) { set_use_cuda(v); }", generated)
                for alias in ("use_cuda", "use_device", "use_acl", "use_rocm", "use_corex"):
                    self.assertIn("__get__" + alias, generated)
                    self.assertIn("__set__" + alias, generated)
                self.assertIn("DECLARE_FLAG(int, lazy_execution);", generated)
                self.assertIn("_get_lazy_execution() { return lazy_execution; }", generated)
                self.assertNotIn("DECLARE_FLAG(int, use_cuda);", generated)
                self.assertNotIn("ignored", generated)


if __name__ == "__main__":
    unittest.main()
