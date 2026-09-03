# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``python/jittor/utils`` is being emptied; what leaves must not come back.

Task 5.25. ``utils/`` held eleven files with no shared responsibility --
compiler resources, repository tooling, a 718-line PyTorch-source translator and
a Flask app whose launcher lived in ``tools/``. It is not even a package: there
is no ``__init__.py``, so it is an implicit namespace directory shipped by one
explicit ``MANIFEST.in`` line, and the repository-layout gate waves the whole
directory through as a single entry. Junk accumulated on the inside of the gate.

This file pins the part that has moved, so it cannot drift back, and records the
part that has not and why.

Static only.
"""

import ast
import unittest
from pathlib import Path

import jittor


PACKAGE = Path(jittor.__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]


class TestTheConverterLivesInCompat(unittest.TestCase):
    """The PyTorch-source translator is compatibility code, not a utility."""

    def test_it_moved(self):
        self.assertTrue((PACKAGE / "compat" / "pytorch_converter.py").is_file())
        self.assertFalse((PACKAGE / "utils" / "pytorch_converter.py").exists())

    def test_its_http_front_end_moved_with_it(self):
        self.assertTrue((PACKAGE / "compat" / "converter_server.py").is_file())
        self.assertFalse((PACKAGE / "utils" / "converter_server.py").exists())

    def test_nothing_imports_the_old_path(self):
        # This file names the retired paths in order to forbid them, so it is
        # the one file the scan has to skip -- otherwise the rule reports
        # itself and can never go green.
        self_path = Path(__file__).resolve()
        offenders = []
        for base in (PACKAGE, REPO_ROOT / "tests", REPO_ROOT / "tools"):
            for path in sorted(base.rglob("*")):
                if path.suffix not in (".py", ".sh") or "__pycache__" in path.parts:
                    continue
                if path.resolve() == self_path:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                for stale in ("jittor.utils.pytorch_converter",
                              "jittor.utils.converter_server"):
                    if stale in text:
                        offenders.append("%s -> %s" % (path, stale))
        self.assertEqual(offenders, [])

    def test_the_launcher_names_the_module_it_runs(self):
        # The container installs the published wheel and names this module in
        # FLASK_APP, so the script and the package have to agree.
        script = (REPO_ROOT / "tools" / "services" / "legacy"
                  / "converter_server.sh").read_text(encoding="utf-8")
        self.assertIn("FLASK_APP=jittor.compat.converter_server", script)


class TestWhatStaysInUtilsAndWhy(unittest.TestCase):
    """The compiler resources are pinned by the layout doc; 3.18 removes them."""

    #: ``docs/architecture/repository-layout.md`` lists these as paths the
    #: compiler contract depends on. ``jit_compiler.cc`` builds the asm_tuner
    #: command line from ``jittor_path + "/utils/asm_tuner.py"``, so moving them
    #: is a C++ change -- task 3.18 deletes the chain instead.
    _COMPILER_CONTRACT = ("asm_tuner.py", "dlink_compiler.py", "dumpdef.py")

    def test_the_compiler_resources_are_still_where_the_contract_says(self):
        for name in self._COMPILER_CONTRACT:
            self.assertTrue((PACKAGE / "utils" / name).is_file(), name)

    def test_the_layout_document_still_pins_them(self):
        doc = (REPO_ROOT / "docs" / "architecture"
               / "repository-layout.md").read_text(encoding="utf-8")
        self.assertIn(
            "python/jittor/utils/{asm_tuner.py,dlink_compiler.py,dumpdef.py}",
            doc,
            "if the contract moved, this test and 5.25's remaining half both "
            "need rewriting")

    def test_the_drawer_only_shrinks(self):
        # Whatever is still in there, nothing NEW may be added: a file with no
        # home belongs in a package that states one.
        remaining = sorted(p.name for p in (PACKAGE / "utils").glob("*.py"))
        self.assertEqual(
            remaining,
            ["asm_tuner.py", "bench_klo.py", "dlink_compiler.py", "dumpdef.py",
             "gen_pyi.py", "jtune.py", "local_doc_builder.py", "nvtx.py",
             "tracer.py"],
            "python/jittor/utils is being dismantled (task 5.25). Put new code "
            "in a package whose name says what it is for.")


if __name__ == "__main__":
    unittest.main()
