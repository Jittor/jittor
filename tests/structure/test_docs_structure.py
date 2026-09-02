"""Contracts for the canonical Sphinx/MyST documentation toolchain."""

from __future__ import print_function

import ast
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from _helpers.child_process import run_python_child


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
        self.assertFalse(list(self.repo_root.rglob("*.src.md")))
        self.assertFalse(list(self.repo_root.rglob("*.ipynb")))

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
        self.assertIn("nodes.literal_block", adapter)
        self.assertIn('"inventory.json"', adapter)
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

    def test_api_pages_auto_discover_members_and_inventory_is_unique(self):
        inventory = json.loads(
            (self.docs_root / "api" / "inventory.json").read_text(encoding="utf-8")
        )
        objects = []
        docnames = []
        for page in inventory["pages"]:
            docnames.append(page["docname"])
            path = self.docs_root / (page["docname"] + ".md")
            source = path.read_text(encoding="utf-8")
            self.assertIn(":::{autopublicmodule}", source)
            self.assertNotIn(":members:", source)
            self.assertNotIn(":imported-members:", source)
            self.assertNotIn("eval_rst", source)
            self.assertNotIn("undoc-members", source)
            modules = {
                line.split(None, 1)[1]
                for line in source.splitlines()
                if line.startswith(":::{autopublicmodule} ")
            }
            for name in page["objects"]:
                objects.append(name)
                self.assertIn(name.rsplit(".", 1)[0], modules)
        self.assertEqual(len(docnames), len(set(docnames)))
        self.assertEqual(len(objects), len(set(objects)))
        index = (self.docs_root / "api" / "index.md").read_text(encoding="utf-8")
        self.assertIn(":::{autosummary}", index)

    def test_api_build_checker_rejects_duplicate_html_ids(self):
        checker_path = self.repo_root / "tools" / "docs" / "check_build.py"
        spec = spec_from_file_location("_jittor_docs_check_build", str(checker_path))
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        checker = module_from_spec(spec)
        spec.loader.exec_module(checker)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            html = root / "api" / "sample.html"
            html.parent.mkdir(parents=True)
            html.write_text(
                '<div id="jittor.sample"></div><div id="jittor.sample"></div>',
                encoding="utf-8",
            )
            inventory = root / "inventory.json"
            inventory.write_text(
                json.dumps({"pages": [{"docname": "api/sample", "objects": ["jittor.sample"]}]}),
                encoding="utf-8",
            )
            issues, checked = checker._check_api(root, inventory)
        self.assertEqual(checked, 1)
        self.assertTrue(any("duplicate HTML anchors" in issue for issue in issues))

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
                "jupytext==1.17.3",
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
        for option in ('"-E"', '"-a"', '"-W"', '"--keep-going"', '"-n"'):
            self.assertIn(option, nox_source)
        self.assertIn("autodoc imported the source tree", nox_source)
        workflow = (self.repo_root / ".github" / "workflows" / "docs.yml").read_text(
            encoding="utf-8"
        )
        for session in ("docs", "docs_zh", "docs_links", "tutorials"):
            self.assertIn(session, workflow)

    def _tracked(self, pattern):
        """Repository-tracked documentation paths matching ``pattern``."""
        completed = subprocess.run(
            ("git", "ls-files", "--", "docs/" + pattern),
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        return [line for line in completed.stdout.splitlines() if line]

    def test_sources_and_catalogs_contain_no_generated_output(self):
        """No generated documentation output is committed.

        The check is against tracked content, not the working tree: building the
        translated documentation writes ``.mo`` catalogs next to their ``.po``
        sources, and ``.gitignore`` already declares them as output. Failing
        merely because someone built the docs would report a clean repository as
        broken.
        """
        for pattern in ("*.rst", "*.pot", "*.mo", "*.html"):
            self.assertEqual(self._tracked(pattern), [], pattern)
        self.assertEqual(self._tracked("_build/**"), [])
        self.assertFalse(any(path.is_dir() for path in self.docs_root.glob("_build/*")))
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
        result = run_python_child(
            ["tools/docs/check_links.py"], cwd=self.repo_root,
            merge_stderr=True)
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
