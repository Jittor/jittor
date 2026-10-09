"""Source-checkout contracts for package discovery and runtime resources."""

import unittest
import ast
import importlib.util
from pathlib import Path


def _docker_pattern_matches(path, pattern):
    """Match the default-deny context's path globs, including zero-level **."""
    import fnmatch

    parts = path.split("/")
    patterns = pattern.rstrip("/").split("/")

    def match(remaining, rules):
        if not rules:
            return not remaining
        if rules[0] == "**":
            return any(match(remaining[index:], rules[1:]) for index in range(len(remaining) + 1))
        return (
            bool(remaining)
            and fnmatch.fnmatchcase(remaining[0], rules[0])
            and match(remaining[1:], rules[1:])
        )

    return match(parts, patterns)


def _docker_copy_file_map(root, dockerfile, dockerignore):
    """Materialize simple COPY selection independently of package-data owners.

    This checks payload bytes and context boundaries, not Docker RUN execution.
    Unsupported COPY syntax fails loudly rather than pretending to model it.
    """
    import shlex

    rules = [
        line.strip()
        for line in dockerignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    copied = {}
    for line in dockerfile.splitlines():
        if not line.startswith("COPY "):
            continue
        tokens = shlex.split(line)[1:]
        if len(tokens) < 2 or any(token.startswith("--") for token in tokens):
            raise ValueError("unsupported COPY syntax: " + line)
        for source in tokens[:-1]:
            if source in (".", "./") or any(char in source for char in "*?[{"):
                raise ValueError("COPY must name explicit source owners: " + line)
            selected = root / source
            candidates = (
                selected.rglob("*")
                if selected.is_dir() and not selected.is_symlink()
                else (selected,)
            )
            for path in candidates:
                if not path.is_file() and not path.is_symlink():
                    continue
                relative = path.relative_to(root).as_posix()
                components = relative.split("/")
                ancestors = [
                    "/".join(components[:index]) for index in range(1, len(components) + 1)
                ]
                allowed = True
                for rule in rules:
                    pattern = rule[1:] if rule.startswith("!") else rule
                    if any(_docker_pattern_matches(parent, pattern) for parent in ancestors):
                        allowed = rule.startswith("!")
                if allowed:
                    copied[relative] = path
    return copied


def _assert_docker_runtime_resources(root, copied):
    generator_path = root / "tools/build/generate_manifest.py"
    spec = importlib.util.spec_from_file_location("docker_resource_owners", generator_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    required = {
        path for path in generator.runtime_resources(root) if path.startswith(("src/", "backends/"))
    }
    if (
        not required
        or not any(path.startswith("src/") for path in required)
        or not any(path.startswith("backends/") for path in required)
    ):
        raise AssertionError("the package must declare both native resource owners")
    missing = sorted(required - set(copied))
    if missing:
        raise AssertionError(
            "Docker COPY context drops runtime resources: " + ", ".join(missing[:4])
        )
    for name in required:
        if copied[name].read_bytes() != (root / name).read_bytes():
            raise AssertionError("Docker COPY changed runtime resource bytes: " + name)


class TestPackagingStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.python_root = cls.repo_root / "python"
        cls.pyproject_path = cls.repo_root / "pyproject.toml"
        if not cls.pyproject_path.is_file():
            raise unittest.SkipTest("packaging metadata requires a source checkout")

    def test_find_packages_matches_every_regular_package(self):
        from setuptools import find_packages

        expected = {
            path.parent.relative_to(self.python_root).as_posix().replace("/", ".")
            for path in self.python_root.rglob("__init__.py")
        }
        # `python/jittor/compat` is a development symlink to the top-level
        # compat/ tree -- the separate jittor-torch distribution. rglob does not
        # follow it, and setup.py excludes it so the core wheel never ships
        # another distribution's packages, which `test_pyproject_uses_regular
        # _package_discovery` pins; discovery is asked the same question here.
        discovered = set(
            find_packages(where=str(self.python_root), exclude=("jittor.compat", "jittor.compat.*"))
        )
        self.assertEqual(discovered, expected)
        backend_root = self.repo_root / "backends"
        backend_expected = {
            "jittor.backends." + path.parent.relative_to(backend_root).as_posix().replace("/", ".")
            for path in backend_root.rglob("__init__.py")
        }
        backend_discovered = {
            "jittor.backends." + name for name in find_packages(where=str(backend_root))
        }
        self.assertEqual(backend_discovered, backend_expected)
        self.assertIn("jittor.backends.cuda.kernels.cublas", backend_discovered)
        self.assertTrue((backend_root / "cuda/kernels/cublas/lt_linear_cuda.py").is_file())

    def test_pyproject_uses_regular_package_discovery(self):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                from setuptools._vendor import tomli as tomllib

        with self.pyproject_path.open("rb") as stream:
            config = tomllib.load(stream)
        package_dirs = config["tool"]["setuptools"]["package-dir"]
        self.assertEqual(package_dirs[""], "python")
        for backend in ("cuda", "acl", "comm"):
            self.assertEqual(package_dirs["jittor.backends." + backend], "backends/" + backend)
        setup_tree = ast.parse((self.repo_root / "setup.py").read_text())
        discovery_roots = {
            ast.literal_eval(node.args[0])
            for node in ast.walk(setup_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "find_packages"
        }
        self.assertEqual(discovery_roots, {"python", "backends"})
        compat_exclusions = {
            tuple(ast.literal_eval(keyword.value))
            for node in ast.walk(setup_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "find_packages"
            and ast.literal_eval(node.args[0]) == "python"
            for keyword in node.keywords
            if keyword.arg == "exclude"
        }
        self.assertEqual(compat_exclusions, {("jittor.compat", "jittor.compat.*")})
        self.assertTrue(config["tool"]["setuptools"]["include-package-data"])
        with (self.repo_root / "compat/pyproject.toml").open("rb") as stream:
            compat_config = tomllib.load(stream)
        self.assertEqual(compat_config["project"]["name"], "jittor-torch")
        self.assertNotIn("jittor-torch-shim", config["project"].get("scripts", {}))
        self.assertEqual(
            compat_config["project"]["scripts"]["jittor-torch-shim"],
            "jittor.compat.shim.deploy:main",
        )

    def test_manifest_covers_runtime_trees_without_cache_payloads(self):
        from importlib.util import module_from_spec, spec_from_file_location

        path = self.repo_root / "tools/build/generate_manifest.py"
        spec = spec_from_file_location("manifest_contract", path)
        generator = module_from_spec(spec)
        spec.loader.exec_module(generator)
        for project in (self.repo_root, self.repo_root / "compat"):
            self.assertEqual(
                (project / "MANIFEST.in").read_text(), generator.manifest_text(project)
            )
        resources = generator.runtime_resources(self.repo_root)
        self.assertEqual(resources["src/core/common.h"], "jittor/src/core/common.h")
        self.assertEqual(
            resources["backends/cuda/include/helper_cuda.h"],
            "jittor/backends/cuda/include/helper_cuda.h",
        )
        self.assertIn("python/jittor/contrib/math_util/src/igamma.h", resources)
        self.assertFalse(any(path.startswith("compat/") for path in resources))
        compat_resources = generator.runtime_resources(self.repo_root / "compat")
        self.assertEqual(
            compat_resources["shim/cpp_extension/include/ATen/cuda/detail/UnpackRaw.cuh"],
            "jittor/compat/shim/cpp_extension/include/ATen/cuda/detail/UnpackRaw.cuh",
        )
        self.assertTrue(set(resources.values()).isdisjoint(compat_resources.values()))
        for source in (self.repo_root / "backends").rglob("*"):
            if source.is_file() and source.suffix in {".h", ".cc", ".cpp", ".cu", ".cuh"}:
                self.assertIn(source.relative_to(self.repo_root).as_posix(), resources)

    def test_backend_python_modules_have_a_distribution_owner(self):
        from setuptools import find_packages

        spec = importlib.util.spec_from_file_location(
            "backend_resource_contract", self.repo_root / "tools/build/generate_manifest.py"
        )
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        resources = generator.runtime_resources(self.repo_root)
        backend_root = self.repo_root / "backends"
        packages = set(find_packages(where=str(backend_root)))
        for source in backend_root.rglob("*.py"):
            relative = source.relative_to(self.repo_root)
            if any(part in generator.IGNORED_DIRECTORIES for part in relative.parts):
                continue
            package = source.parent.relative_to(backend_root).as_posix().replace("/", ".")
            with self.subTest(source=relative.as_posix()):
                self.assertTrue(
                    package in packages or relative.as_posix() in resources,
                    "backend module is neither a regular package member nor a runtime resource",
                )

    def test_generated_manifest_handles_spaces_and_dirty_source_caches(self):
        from importlib.util import module_from_spec, spec_from_file_location
        from tempfile import TemporaryDirectory
        from setuptools._distutils.filelist import FileList

        spec = spec_from_file_location(
            "manifest_fixture", self.repo_root / "tools/build/generate_manifest.py"
        )
        generator = module_from_spec(spec)
        spec.loader.exec_module(generator)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text(
                '[tool.setuptools]\npackage-dir={demo="pkg"}\n'
                '[tool.setuptools.package-data]\ndemo=["assets/**/*"]\n'
                '[tool.jittor.sdist]\ninclude=["examples/**"]\nexclude=[]\n'
            )
            files = [
                "setup.py",
                "MANIFEST.in",
                "pkg/assets/value.dat",
                "pkg/assets/__pycache__/leak.dat",
                "examples/tutorial 1.md",
                "examples/.pytest_cache/v/cache/data",
            ]
            for relative in files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            resources = generator.runtime_resources(root)
            self.assertEqual(resources, {"pkg/assets/value.dat": "demo/assets/value.dat"})
            manifest = generator.manifest_text(root)
            selected = FileList()
            selected.allfiles = files + ["pyproject.toml"]
            for line in manifest.splitlines():
                if line.startswith("include "):
                    selected.process_template_line(line)
            self.assertIn("examples/tutorial 1.md", selected.files)
            self.assertFalse(any("cache" in name for name in selected.files))
            (root / "pkg/assets/new.dat").write_text("new resource")
            self.assertIn("pkg/assets/new.dat", generator.runtime_resources(root))
            self.assertNotEqual(manifest, generator.manifest_text(root))

    def test_root_development_trees_do_not_become_runtime_packages(self):
        for relative in ("examples", "tools"):
            root = self.repo_root / relative
            self.assertTrue(root.is_dir(), relative)
            self.assertFalse((root / "__init__.py").exists(), relative)

    def test_built_sdist_has_an_executable_contents_gate(self):
        checker = self.repo_root / "tools" / "release" / "check_sdist_contents.py"
        self.assertTrue(checker.is_file())

    def test_required_deep_runtime_resources_exist(self):
        required = (
            "compat/shim/cpp_extension/include/ATen/cuda/detail/UnpackRaw.cuh",
            "compat/shim/resources/stubs/flash_attn/flash_attn_interface.py",
            "compat/shim/resources/torch/__init__.py",
            "backends/cuda/kernels/nn/softmax_cuda.py",
            "backends/cuda/kernels/nn/group_norm_cuda.py",
            "backends/cuda/include/helper_cuda.h",
            "backends/cuda/libraries/cutt/include/cutt_wrapper.h",
            "python/jittor/tools/tracer.py",
        )
        for relative in required:
            with self.subTest(path=relative):
                self.assertTrue((self.repo_root / relative).is_file())

    def test_wheel_audit_distinguishes_runtime_helpers_from_build_artifacts(self):
        path = self.repo_root / "tools/release/check_wheel_contents.py"
        spec = importlib.util.spec_from_file_location("wheel_layout_contract", path)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        for name in ("__init__.py", "dlink_compiler.py", "dumpdef.py"):
            self.assertIsNone(checker._pollution_reason("jittor/build/" + name))
        self.assertIsNotNone(checker._pollution_reason("jittor/build/temp.o"))
        self.assertIsNotNone(checker._pollution_reason("build/jittor/core.py"))

    def test_docker_copy_payload_preserves_declared_runtime_resources(self):
        dockerfile = (self.repo_root / "Dockerfile").read_text()
        dockerignore = (self.repo_root / ".dockerignore").read_text()
        copied = _docker_copy_file_map(self.repo_root, dockerfile, dockerignore)
        _assert_docker_runtime_resources(self.repo_root, copied)

    def test_docker_copy_or_context_missing_either_owner_is_rejected(self):
        dockerfile = (self.repo_root / "Dockerfile").read_text()
        dockerignore = (self.repo_root / ".dockerignore").read_text()
        for owner in ("src", "backends"):
            with self.subTest(owner=owner, failure="COPY"):
                broken = "\n".join(
                    line
                    for line in dockerfile.splitlines()
                    if not line.startswith("COPY " + owner + " ")
                )
                copied = _docker_copy_file_map(self.repo_root, broken, dockerignore)
                with self.assertRaisesRegex(AssertionError, "drops runtime resources"):
                    _assert_docker_runtime_resources(self.repo_root, copied)
            with self.subTest(owner=owner, failure="context"):
                broken = "\n".join(
                    line
                    for line in dockerignore.splitlines()
                    if not line.startswith("!" + owner + "/")
                )
                copied = _docker_copy_file_map(self.repo_root, dockerfile, broken)
                with self.assertRaisesRegex(AssertionError, "drops runtime resources"):
                    _assert_docker_runtime_resources(self.repo_root, copied)

    def test_docker_context_keeps_caches_logs_and_old_trees_out(self):
        from tempfile import TemporaryDirectory

        dockerfile = (self.repo_root / "Dockerfile").read_text()
        dockerignore = (self.repo_root / ".dockerignore").read_text()
        good = ("src/core/common.h", "backends/cuda/include/helper_cuda.h")
        bad = (
            "src/__pycache__/leak.pyc",
            "src/deep/.pytest_cache/leak",
            "backends/acl/.ruff_cache/leak",
            "backends/cuda/compile.log",
            "old-tree/src/experiment.cc",
            "examples/output.ipynb",
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in good + bad:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode("utf-8"))
            copied = _docker_copy_file_map(root, dockerfile, dockerignore)
            self.assertEqual(set(copied), set(good))
            self.assertEqual(
                {name: path.read_bytes() for name, path in copied.items()},
                {name: name.encode("utf-8") for name in good},
            )

    def test_docker_cold_selftest_has_required_onednn_build_tool(self):
        import ast
        import shlex
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch

        dockerfile = (self.repo_root / "Dockerfile").read_text()

        def assert_cold_build_tools(text):
            commands = text.replace("\\\n", " ").splitlines()
            packages = set()
            for command in commands:
                if not command.startswith("RUN "):
                    continue
                tokens = shlex.split(command[4:])
                for index in range(len(tokens) - 1):
                    if tokens[index : index + 2] != ["apt-get", "install"]:
                        continue
                    for token in tokens[index + 2 :]:
                        if token in ("&&", ";"):
                            break
                        if not token.startswith("-"):
                            packages.add(token)
            if "cmake" not in packages:
                raise AssertionError("cold oneDNN selftest requires an installed cmake")

        assert_cold_build_tools(dockerfile)
        with self.assertRaisesRegex(AssertionError, "cold oneDNN"):
            assert_cold_build_tools(
                "\n".join(
                    line
                    for line in dockerfile.splitlines()
                    if not line.strip().startswith("cmake ")
                )
            )

        compile_extern = ast.parse(
            (self.repo_root / "python/jittor/build/compile_extern.py").read_text()
        )
        default = next(
            node.value
            for node in compile_extern.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "use_mkl" for target in node.targets
            )
        )
        self.assertEqual(default.func.id, "build_flag")
        self.assertEqual(default.args[0].value, "use_mkl")
        self.assertIs(default.args[1].value, True)
        train = ast.parse((self.repo_root / "python/jittor/selftest.py").read_text())
        train_function = next(
            node
            for node in train.body
            if isinstance(node, ast.FunctionDef) and node.name == "_train_three_steps"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Conv2d"
                for node in ast.walk(train_function)
            )
        )

        provider_path = self.repo_root / "python/jittor/build/onednn.py"
        spec = importlib.util.spec_from_file_location("cold_docker_onednn_provider", provider_path)
        provider = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(provider)
        asset = SimpleNamespace(sha256="0" * 64, url="never-download", filename="never-download")
        with TemporaryDirectory() as directory:
            compiler = str(Path(__file__).resolve())

            def unavailable_cmake(name):
                return None if name == "cmake" else compiler

            def must_not_download(*args):
                raise AssertionError("cmake availability must fail before download or native build")

            with patch.object(provider.shutil, "which", side_effect=unavailable_cmake):
                with self.assertRaisesRegex(RuntimeError, "requires cmake"):
                    provider.install_source(
                        directory, asset, "test", compiler, must_not_download, must_not_download
                    )

    def test_container_pull_request_trigger_covers_native_resource_owners(self):
        import fnmatch

        workflow = (self.repo_root / ".github/workflows/containers.yml").read_text()

        def patterns(text):
            paths = []
            in_pull_request = False
            in_paths = False
            for line in text.splitlines():
                if line == "  pull_request:":
                    in_pull_request = True
                elif (
                    in_pull_request
                    and line.startswith("  ")
                    and not line.startswith("    ")
                    and line.strip()
                ):
                    break
                elif in_pull_request and line == "    paths:":
                    in_paths = True
                elif in_paths and line.startswith("      - "):
                    paths.append(line[8:].strip().strip('"').strip("'"))
            if not paths:
                raise AssertionError("container workflow must declare pull_request path owners")
            return paths

        def assert_owner_trigger(text, owner):
            selected = patterns(text)
            for relative in (owner + "/resource.cc", owner + "/deep/include/resource.h"):
                if not any(fnmatch.fnmatchcase(relative, rule) for rule in selected):
                    raise AssertionError("container PR trigger misses resource owner: " + owner)

        for owner in ("src", "backends"):
            assert_owner_trigger(workflow, owner)
            broken = "\n".join(
                line for line in workflow.splitlines() if line.strip() != '- "' + owner + '/**"'
            )
            with self.assertRaisesRegex(AssertionError, "misses resource owner"):
                assert_owner_trigger(broken, owner)


if __name__ == "__main__":
    unittest.main()
