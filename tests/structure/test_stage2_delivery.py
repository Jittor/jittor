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
        cpu = functions["cpu"]

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
        for target in (
            "tests/optim/test_optimizer.py",
            "tests/optim/test_adamw.py",
            "tests/nn/test_affine_grid.py",
            "tests/nn/test_attention_oracle.py",
            "tests/nn/test_batchnorm.py",
            "tests/nn/test_loss.py",
            "tests/nn/test_relu.py",
            "tests/core/test_array.py::TestArray::test_array_dtype",
        ):
            self.assertIn(target, source)
        self.assertIn('env.pop("REAL_TORCH_SITE", None)', cpu)
        self.assertIn("SCIPY", cpu)
        self.assertIn('SCIPY = "scipy==', source)

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

    def test_asv_teardown_tolerates_skipped_parameter_setup(self):
        from benchmarks.operators import OperatorBenchmarks
        from benchmarks.optimizer_step import OptimizerStepBenchmarks
        from benchmarks.tiny_llama import TinyLlamaBenchmarks

        OperatorBenchmarks().teardown("torch", "cpu", "matmul")
        OptimizerStepBenchmarks().teardown("sgd", 32, "cuda")
        TinyLlamaBenchmarks().teardown("torch", "forward")


if __name__ == "__main__":
    unittest.main()
