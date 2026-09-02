"""Source-checkout contracts for the installed Jittor self-test."""

import ast
from types import SimpleNamespace
import unittest
from pathlib import Path


class TestSelftestStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.selftest_path = cls.repo_root / "python" / "jittor" / "selftest.py"
        if not (cls.repo_root / "pyproject.toml").is_file():
            raise unittest.SkipTest("self-test structure requires a source checkout")

    def test_selftest_is_standalone_and_python37_compatible(self):
        source = self.selftest_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(self.selftest_path), feature_version=(3, 7))
        functions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        self.assertEqual({"main", "run"} - functions, set())
        self.assertNotIn("jittor.test", source)
        self.assertIn("jt.array", source)
        self.assertIn("jt.grad", source)

    def test_selftest_reports_the_active_backend(self):
        source = self.selftest_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(self.selftest_path))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_backend_name"
        )
        module = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(module)

        cases = (
            ({}, {}, "cpu"),
            ({"use_cuda": 1}, {}, "cuda"),
            ({"use_cuda": 1}, {"has_acl": 1}, "npu"),
            ({"use_cuda": 1}, {"has_rocm": 1}, "rocm"),
            ({"use_cuda": 0}, {"has_acl": 1, "has_rocm": 1}, "cpu"),
        )
        for flags, compiler, expected in cases:
            namespace = {
                "jt": SimpleNamespace(
                    flags=SimpleNamespace(**flags),
                    compiler=SimpleNamespace(**compiler),
                )
            }
            exec(compile(module, str(self.selftest_path), "exec"), namespace)
            self.assertEqual(namespace["_backend_name"](), expected)

    def test_the_selftest_trains_rather_than_squaring_three_numbers(self):
        """A forward and backward over `[1,2,3]**2` proves the core built.

        It proves nothing else: one elementwise operator and the autodiff
        bookkeeping around it. A wheel whose convolution, normalisation or
        optimiser update is broken passed it, and the release pipeline has
        nothing else that runs Jittor at all.
        """
        source = self.selftest_path.read_text(encoding="utf-8")
        for needle in ("Conv2d", "BatchNorm2d", "optim.SGD", "optimizer.step"):
            self.assertIn(needle, source)
        namespace = {}
        tree = ast.parse(source)
        assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None) == "KEY_MODULES"
        )
        modules = ast.literal_eval(assignment.value)
        self.assertGreaterEqual(len(modules), 10)
        for name in modules:
            self.assertTrue(name.startswith("jittor."), name)
        # Importing the torch shim changes process-wide state, so it must not
        # be on a list every release runs.
        self.assertNotIn("jittor.compat.torch", modules)

    def test_the_release_workflow_runs_the_selftest(self):
        """Otherwise a wheel whose core does not compile can be published.

        Every other step in the release pipeline reads the wheel as an
        archive: version, member list, three resource files. None of them
        imports it.
        """
        workflow = (self.repo_root / ".github" / "workflows"
                    / "release.yml").read_text(encoding="utf-8")
        self.assertIn("jittor.selftest", workflow)
        validation = workflow.split("platform-validation:", 1)
        self.assertEqual(len(validation), 2, "no platform-validation job")
        after = validation[1].split("\n  publish-", 1)[0]
        self.assertIn("jittor.selftest", after)

    def test_nox_structure_runs_the_installed_wheel_selftest(self):
        source = (self.repo_root / "noxfile.py").read_text(encoding="utf-8")
        self.assertIn('"--target",\n        str(wheel_install)', source)
        self.assertIn('selftest_env["PYTHONPATH"] = str(wheel_install)', source)
        self.assertIn('"python", "-m", "jittor.selftest"', source)

    def test_active_callers_do_not_use_the_packaged_test_suite(self):
        legacy_module = "jittor.test." + "test_example"
        candidates = [
            self.repo_root / "Dockerfile",
            self.repo_root / "README.md",
            self.repo_root / "CONTRIBUTING.md",
        ]
        python_root = self.repo_root / "python"
        candidates.extend(
            path
            for path in python_root.rglob("*")
            if path.is_file() and (path.suffix in {".py", ".sh"} or "Dockerfile" in path.name)
        )

        violations = []
        for path in candidates:
            if legacy_module in path.read_text(encoding="utf-8", errors="replace"):
                violations.append(path.relative_to(self.repo_root).as_posix())
        self.assertEqual(violations, [])

    def test_installation_and_environment_callers_use_selftest(self):
        expected_callers = [
            self.repo_root / "Dockerfile",
            self.repo_root / "README.md",
            self.repo_root / "CONTRIBUTING.md",
            self.repo_root / "tools" / "install" / "legacy" / "install.sh",
            self.repo_root / "tests" / "compiler" / "test_lock.py",
            self.repo_root / "tools" / "release" / "legacy" / "polish_centos.py",
            self.repo_root / "python" / "jittor_utils" / "class" / "setup_env.py",
        ]
        expected_callers.extend(
            sorted((self.repo_root / "tests" / "system" / "legacy").glob("test_*ubuntu*.sh"))
        )

        missing = []
        for path in expected_callers:
            if "-m jittor.selftest" not in path.read_text(encoding="utf-8"):
                missing.append(path.relative_to(self.repo_root).as_posix())
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
