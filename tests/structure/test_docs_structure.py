"""Contracts for the canonical Sphinx/MyST documentation toolchain."""

from __future__ import print_function

import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest


EXPECTED_EXTENSIONS = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
]


def _assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
    raise AssertionError("missing assignment: {}".format(name))


class TestDocsStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.docs_root = cls.repo_root / "docs"
        if not (cls.repo_root / "pyproject.toml").is_file():
            raise unittest.SkipTest("documentation contracts require a source checkout")

    def test_one_canonical_documentation_tree_remains(self):
        self.assertTrue((self.docs_root / "conf.py").is_file())
        self.assertTrue((self.docs_root / "index.md").is_file())
        retired = (
            "doc",
            "README.src.md",
            "python/jittor_utils/translator.py",
            "tools/docs/legacy/make_doc.py",
        )
        for relative in retired:
            with self.subTest(path=relative):
                self.assertFalse((self.repo_root / relative).exists())
        tracked = subprocess.run(
            ("git", "ls-files", "--", "*.src.md"),
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        self.assertEqual(tracked.stdout, "")

    def test_sphinx_configuration_is_myst_only_and_import_safe(self):
        path = self.docs_root / "conf.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path), feature_version=(3, 7))
        self.assertEqual(ast.literal_eval(_assignment(tree, "extensions")), EXPECTED_EXTENSIONS)
        self.assertEqual(ast.literal_eval(_assignment(tree, "source_suffix")), {".md": "markdown"})
        self.assertEqual(ast.literal_eval(_assignment(tree, "html_theme")), "furo")
        self.assertIs(ast.literal_eval(_assignment(tree, "autosummary_generate")), False)
        self.assertIs(ast.literal_eval(_assignment(tree, "nitpicky")), True)
        intersphinx = ast.literal_eval(_assignment(tree, "intersphinx_mapping"))
        self.assertEqual(set(intersphinx), {"python", "numpy", "pytorch"})

        release = _assignment(tree, "release")
        self.assertIsInstance(release, ast.Call)
        self.assertIsInstance(release.func, ast.Name)
        self.assertEqual(release.func.id, "version")
        self.assertEqual(ast.literal_eval(release.args[0]), "jittor")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                self.assertNotIn("jittor", names)
            if isinstance(node, ast.Attribute) and node.attr == "path":
                self.assertFalse(isinstance(node.value, ast.Name) and node.value.id == "sys")
        for forbidden in (
            "recommonmark",
            "AutoStructify",
            "sphinx_rtd_theme",
            "../../python",
            "Var.__module__",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("from importlib_metadata import version", source)
        adapter = (self.docs_root / "_myst_autodoc.py").read_text(encoding="utf-8")
        ast.parse(adapter, filename="docs/_myst_autodoc.py", feature_version=(3, 7))
        self.assertIn("MockRSTParser", adapter)
        self.assertNotIn("eval-rst", adapter)

    def test_content_manifest_accounts_for_every_legacy_page(self):
        manifest = json.loads(
            (self.docs_root / "content-manifest.json").read_text(encoding="utf-8")
        )
        entries = manifest["entries"]
        legacy = [entry["legacy"] for entry in entries]
        self.assertEqual(len(legacy), len(set(legacy)))
        self.assertTrue(legacy)
        for entry in entries:
            with self.subTest(legacy=entry["legacy"]):
                self.assertEqual(entry["status"], "verified")
                self.assertFalse((self.repo_root / entry["legacy"]).exists())
                docname = entry["docname"]
                if docname is not None:
                    target = self.docs_root / docname
                    if target.suffix == "":
                        target = target.with_suffix(".md")
                    self.assertTrue(target.is_file(), str(target))
                for asset in entry["assets"]:
                    self.assertTrue((self.docs_root / asset).is_file(), asset)

    def test_api_inventory_is_explicit_myst_and_unique(self):
        inventory = json.loads(
            (self.docs_root / "api" / "inventory.json").read_text(encoding="utf-8")
        )
        objects = []
        for page in inventory["pages"]:
            path = self.docs_root / (page["docname"] + ".md")
            source = path.read_text(encoding="utf-8")
            self.assertIn(":::{autosummary}", source)
            self.assertNotIn(":toctree:", source)
            self.assertNotIn("eval_rst", source)
            self.assertNotIn("undoc-members", source)
            for name in page["objects"]:
                self.assertIn(name.rsplit(".", 1)[-1], source)
                objects.append(name)
        self.assertEqual(len(objects), len(set(objects)))

    def test_docs_dependencies_are_locked_without_legacy_plugins(self):
        direct = {
            line.strip()
            for line in (self.repo_root / "requirements" / "docs.in")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.startswith("#")
        }
        self.assertEqual(
            direct,
            {
                "sphinx",
                "myst-parser",
                "furo",
                "sphinx-intl",
                "jupytext",
                "nbconvert",
                "ipykernel",
                'importlib-metadata; python_version < "3.8"',
            },
        )
        lock = (self.repo_root / "requirements" / "docs.txt").read_text(encoding="utf-8")
        lines = [line for line in lock.splitlines() if line and not line.startswith("#")]
        self.assertTrue(lines)
        self.assertTrue(all("==" in line for line in lines))
        lowered = lock.lower()
        self.assertIn('importlib-metadata==6.7.0; python_version < "3.8"', lowered)
        for forbidden in ("recommonmark", "sphinx-rtd-theme", "sphinx-autobuild"):
            self.assertNotIn(forbidden, lowered)

    def test_nox_and_ci_expose_all_documentation_gates(self):
        nox_source = (self.repo_root / "noxfile.py").read_text(encoding="utf-8")
        tree = ast.parse(nox_source, filename="noxfile.py", feature_version=(3, 7))
        functions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertEqual({"docs", "docs_zh", "docs_links", "tutorials"} - functions, set())
        for option in ('"-W"', '"--keep-going"', '"-n"'):
            self.assertIn(option, nox_source)
        self.assertIn("autodoc imported the source tree", nox_source)
        workflow = (self.repo_root / ".github" / "workflows" / "docs.yml").read_text(
            encoding="utf-8"
        )
        for session in ("docs", "docs_zh", "docs_links", "tutorials"):
            self.assertIn(session, workflow)

    def test_sources_and_catalogs_contain_no_generated_output(self):
        self.assertFalse(list(self.docs_root.rglob("*.rst")))
        self.assertFalse(list(self.docs_root.rglob("*.pot")))
        self.assertFalse(list(self.docs_root.rglob("*.mo")))
        self.assertFalse(list(self.docs_root.rglob("*.html")))
        self.assertFalse((self.docs_root / "_build").exists())
        for path in self.docs_root.rglob("*.po"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("#, fuzzy", source)
            self.assertNotIn("#~ msgid", source)

    def test_active_docs_references_do_not_point_at_retired_pipeline(self):
        candidates = [
            self.repo_root / "README.md",
            self.repo_root / "noxfile.py",
            self.repo_root / "pyproject.toml",
        ]
        candidates.extend(self.docs_root.rglob("*.md"))
        candidates.extend((self.repo_root / ".github").rglob("*.yml"))
        forbidden = (
            "doc/source",
            "build_doc.sh",
            "README.src.md",
            "jittor_utils.translator",
            "tools/docs/legacy",
            "recommonmark",
            "AutoStructify",
        )
        violations = []
        for path in candidates:
            source = path.read_text(encoding="utf-8", errors="replace")
            for marker in forbidden:
                if marker in source:
                    violations.append("{}: {}".format(path.relative_to(self.repo_root), marker))
        self.assertEqual(violations, [])

    def test_internal_markdown_links_pass(self):
        result = subprocess.run(
            (sys.executable, "tools/docs/check_links.py"),
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
