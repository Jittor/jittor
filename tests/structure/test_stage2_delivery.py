"""Contracts for the Stage 2 CI, packaging, container, and ASV boundaries."""

import ast
import json
from pathlib import Path
import shlex
import unittest


class TestStage2Delivery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.workflows = cls.repo_root / ".github" / "workflows"
        cls.baseline = cls._read_baseline(cls.repo_root / ".github" / "ci-baseline.env")

    @staticmethod
    def _read_baseline(path):
        values = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, raw_value = line.partition("=")
            if not separator or not key:
                raise AssertionError("malformed CI baseline line: %r" % raw_line)
            parsed = shlex.split(raw_value, posix=True)
            if len(parsed) != 1:
                raise AssertionError("CI baseline value must be one shell word: %r" % raw_line)
            values[key] = parsed[0]
        return values

    def test_baseline_is_complete_and_machine_readable(self):
        required = {
            "CI_HOST_RUNNER",
            "CI_PYTHON_VERSION",
            "CI_CPU_CI_IMAGE",
            "CI_CUDA_VERSION",
            "CI_CUDA_RUNNER_LABELS",
            "CI_NPU_RUNNER_LABELS",
            "CI_CONTAINER_MATRIX",
            "CI_RELEASE_PLATFORM_MATRIX",
        }
        self.assertEqual(set(self.baseline), required)
        self.assertIn("cuda12-2", json.loads(self.baseline["CI_CUDA_RUNNER_LABELS"]))
        self.assertIn("cann9", json.loads(self.baseline["CI_NPU_RUNNER_LABELS"]))
        self.assertEqual(
            len(json.loads(self.baseline["CI_RELEASE_PLATFORM_MATRIX"])["include"]),
            3,
        )

    def test_workflows_consume_the_reusable_baseline(self):
        for name in ("containers", "cpu", "cuda", "docs", "npu", "release", "structure"):
            with self.subTest(workflow=name):
                source = (self.workflows / (name + ".yml")).read_text(encoding="utf-8")
                self.assertIn("uses: ./.github/workflows/_ci-baseline.yml", source)
                self.assertIn("needs.baseline.outputs", source)

        reusable = (self.workflows / "_ci-baseline.yml").read_text(encoding="utf-8")
        self.assertIn("source .github/ci-baseline.env", reusable)
        self.assertIn("runs-on: " + self.baseline["CI_HOST_RUNNER"], reusable)

    def test_retired_gitlab_ci_does_not_return(self):
        self.assertFalse((self.repo_root / ".gitlab-ci.yml").exists())
        layout_gate = (self.repo_root / "agent" / "scripts" / "check_repo_layout.sh").read_text(
            encoding="utf-8",
        )
        self.assertNotIn(".gitlab-ci.yml", layout_gate)

    def test_container_and_release_architectures_match_the_baseline(self):
        matrix = json.loads(self.baseline["CI_CONTAINER_MATRIX"])["include"]
        cpu_base = next(item["base"] for item in matrix if item["name"] == "CPU")
        self.assertEqual(cpu_base, self.baseline["CI_CPU_CI_IMAGE"])
        dockerfile = (self.repo_root / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG FROM_IMAGE=" + cpu_base, dockerfile)

        containers = (self.workflows / "containers.yml").read_text(encoding="utf-8")
        self.assertIn("fromJSON(needs.baseline.outputs.container_matrix)", containers)
        for item in matrix:
            self.assertNotIn("base: " + item["base"], containers)

        release = (self.workflows / "release.yml").read_text(encoding="utf-8")
        self.assertIn("fromJSON(needs.baseline.outputs.release_platform_matrix)", release)
        self.assertIn("canonical wheel must be universal", release)
        self.assertIn("-py3-none-any.whl", release)

    @staticmethod
    def _cpu_gate_files():
        """Test files the CPU gate would collect, across both process modes."""
        from _helpers.gate_scope import (
            native_arguments, selected_files, torch_arguments)

        repo_root = Path(__file__).resolve().parents[2]
        return (selected_files(repo_root, native_arguments())
                | selected_files(repo_root, torch_arguments()))

    def test_nox_keeps_fast_structure_and_packaging_separate(self):
        path = self.repo_root / "noxfile.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        functions = {
            node.name: ast.get_source_segment(source, node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        structure = functions["structure"]
        packaging = functions["packaging"]
        benchmark = functions["benchmark"] + functions["_record_asv"]
        # Both CPU tiers plus the environment helper they share (0.15): the
        # claims below are about what the CPU gate does, and after the split
        # they are spread over three functions rather than one. Asserting only
        # `cpu` would have gone quietly green while `smoke` -- the tier a pull
        # request actually waits for -- inherited none of them.
        cpu = (functions["cpu"] + functions["smoke"]
               + functions["_cpu_gate_env"])
        upper_python = functions["_upper_python_compatibility"]
        py312 = functions["py312"]
        py313 = functions["py313"]

        self.assertIn("STRUCTURE_TESTS", structure)
        self.assertNotIn('"build"', structure)
        self.assertNotIn("check_wheel_contents.py", structure)
        for token in (
            '"build"',
            "check_sdist_contents.py",
            "check_wheel_contents.py",
            "jittor.selftest",
        ):
            self.assertIn(token, packaging)
        for token in (
            '"run"',
            "--set-commit-hash",
            "_asv_has_measurement",
            '"compare"',
            '"publish"',
        ):
            self.assertIn(token, benchmark)
        self.assertIn('env["ASV_PYTHONPATH"]', benchmark)
        self.assertIn("CPU_TORCH_ORACLE_TESTS", cpu)
        self.assertNotIn('"tests/optim"', source)
        # The oracle list is still a list: each entry names a file that has to
        # run against an independent binary PyTorch, which is a property of the
        # test, not of the tree.
        for target in (
            "tests/optim/test_adamw.py",
            "tests/nn/test_affine_grid.py",
            "tests/nn/test_attention_oracle.py",
            "tests/nn/test_batchnorm.py",
            "tests/nn/test_loss.py",
            "tests/nn/test_relu.py",
        ):
            self.assertIn(target, source)
        # These two used to be named in noxfile's CPU_TESTS. 0.04 deleted that
        # list -- the gate is now the tree minus stated exceptions -- so the
        # question is whether the gate reaches them, not whether the file
        # mentions them. Asserting the old text would assert an implementation
        # that no longer exists.
        for target in ("tests/optim/test_optimizer.py", "tests/core/test_array.py"):
            self.assertIn(target, self._cpu_gate_files(), target)
        self.assertIn('env["REAL_TORCH_SITE"] = ""', cpu)
        self.assertIn("CPU_GATE_REQUIREMENTS", cpu)
        self.assertIn("SCIPY,", source)
        self.assertIn('SCIPY = "scipy==', source)
        # The fast tier is a deferral, not a second selection: it starts from the
        # same two gate_scope selections and drops what tiers.py records.
        smoke = functions["smoke"]
        self.assertIn("gate_native_arguments()", smoke)
        self.assertIn("gate_torch_arguments()", smoke)
        self.assertIn('"-m", "not slow"', smoke)
        development_requirements = (self.repo_root / "requirements" / "dev-tools.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("scipy==1.13.1", development_requirements.splitlines())

        structure_workflow = (self.workflows / "structure.yml").read_text(encoding="utf-8")
        self.assertIn("nox / packaging", structure_workflow)
        self.assertIn("python -m nox -s packaging", structure_workflow)

        cpu_workflow = (self.workflows / "cpu.yml").read_text(encoding="utf-8")
        structure_workflow = (self.workflows / "structure.yml").read_text(encoding="utf-8")
        self.assertIn("image: ${{ needs.baseline.outputs.cpu_ci_image }}", cpu_workflow)
        self.assertIn("image: ${{ needs.baseline.outputs.cpu_ci_image }}", structure_workflow)
        self.assertIn("actions/cache/restore@v4", cpu_workflow)
        self.assertIn("actions/upload-artifact@v6", cpu_workflow)
        self.assertIn("ASV_RESULTS_DIR", cpu_workflow)
        self.assertIn("torch==2.7.1", cpu_workflow)
        self.assertIn('JITTOR_REQUIRE_REAL_TORCH: "1"', cpu_workflow)
        self.assertIn('echo "REAL_TORCH_SITE=$(python -c', cpu_workflow)

        self.assertIn("JITTOR_REQUIRE_REAL_TORCH", cpu)
        self.assertIn("requires REAL_TORCH_SITE", cpu)
        for token in (
            "compile OK without SyntaxWarning",
            '"ls-files"',
            '"--exclude-standard"',
            '"build"',
            '"jittor.selftest"',
            "trace_py_var=2",
        ):
            self.assertIn(token, upper_python)

        self.assertIn('@nox.session(python="3.12", venv_backend="venv")', source)
        self.assertIn('@nox.session(python="3.13", venv_backend="venv")', source)
        self.assertIn('"numpy==1.26.4"', py312)
        self.assertIn('"numpy>=2.1,<3.0"', py313)
        self.assertIn('"py312"', source)
        self.assertIn('"py313"', source)
        project_metadata = (self.repo_root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"Programming Language :: Python :: 3.12"', project_metadata)
        self.assertIn(
            '"Programming Language :: Python :: 3.13"',
            project_metadata,
        )
        self.assertIn('"numpy<3.0"', project_metadata)
        self.assertIn("nox / py312", structure_workflow)
        self.assertIn("nox / py313", structure_workflow)
        self.assertIn('python-version: "3.12"', structure_workflow)
        self.assertIn('python-version: "3.13"', structure_workflow)
        self.assertIn("python -m nox -s py312", structure_workflow)
        self.assertIn("python -m nox -s py313", structure_workflow)

    def test_asv_teardown_tolerates_skipped_parameter_setup(self):
        from benchmarks.operators import OperatorBenchmarks
        from benchmarks.optimizer_step import OptimizerStepBenchmarks
        from benchmarks.tiny_llama import TinyLlamaBenchmarks

        OperatorBenchmarks().teardown("torch", "cpu", "matmul")
        OptimizerStepBenchmarks().teardown("sgd", 32, "cuda")
        TinyLlamaBenchmarks().teardown("torch", "forward")

    def test_cudnn_math_policy_is_guarded_for_rocm_and_old_cudnn(self):
        """The two guards must survive; 8.03 moved where they live.

        Each of the six convolution ops used to spell the ``#ifndef IS_ROCM``
        / ``#if CUDNN_VERSION >= 8000`` / ``CUDNN_FMA_MATH`` chain itself, and
        the six copies had drifted apart -- forward asked for tensor-op math
        on float16 while backward left it at ``CUDNN_DEFAULT_MATH``. The chain
        is now in one shared helper, so the ops are checked for *calling* it
        and the helper is checked for the guards.
        """
        cudnn = self.repo_root / "python" / "jittor" / "extern" / "cuda" / "cudnn"
        names = (
            "cudnn_conv_op.cc",
            "cudnn_conv_backward_x_op.cc",
            "cudnn_conv_backward_w_op.cc",
            "cudnn_conv3d_op.cc",
            "cudnn_conv3d_backward_x_op.cc",
            "cudnn_conv3d_backward_w_op.cc",
        )
        for name in names:
            source = (cudnn / "ops" / name).read_text(encoding="utf-8")
            with self.subTest(operation=name):
                self.assertIn("int conv_math_key = 0;", source)
                self.assertIn("#ifndef IS_ROCM", source)
                # One policy, not six: the accumulate type and the math type
                # both come from the shared helper.
                self.assertIn("cudnn_conv_math_type(", source)
                self.assertIn("cudnn_conv_compute_type(", source)
                self.assertIn('jk << "math=" << conv_math_key', source)

        wrapper = (cudnn / "inc" / "cudnn_wrapper.h").read_text(encoding="utf-8")
        self.assertIn("cudnnMathType_t cudnn_conv_math_type(", wrapper)
        self.assertIn("#ifndef IS_ROCM", wrapper)
        self.assertIn("#if CUDNN_VERSION >= 8000", wrapper)
        self.assertIn("CUDNN_FMA_MATH", wrapper)


if __name__ == "__main__":
    unittest.main()
