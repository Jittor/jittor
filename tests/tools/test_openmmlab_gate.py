"""Host contracts for mandatory OpenMMLab owners, tensors and opt-in wiring."""

import ast
import copy
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

from _helpers.child_process import run_python_child

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "agent/skills/torch-compat-repo-runbook/scripts"
sys.path.insert(0, str(ROOT / "compat/tests/torch"))


class OpenMMLabGateTests(unittest.TestCase):
    def test_host_listing_does_not_import_tensor_or_downstream_packages(self):
        code = (
            "import sys;sys.path.insert(0,sys.argv[1]);import _ecosystem_openmmlab_cases as c;"
            "assert len(c.REQUIRED_GRADIENTS['mmengine_train_step'])==7;"
            "assert len(c.REQUIRED_GRADIENTS['mmcv_convmodule_train_step'])==9;"
            "assert not {'torch','jittor','mmcv','mmengine'}.intersection(sys.modules)"
        )
        result = run_python_child(
            ["-S", "-c", code, str(ROOT / "compat/tests/torch")], without_torch_mode=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for repo, case in (
            ("mmengine", "mmengine_train_step"),
            ("mmcv", "mmcv_convmodule_train_step"),
        ):
            result = run_python_child(
                ["-S", str(SCRIPTS / "verify_repo.py"), "--repo", repo, "--list-only"],
                without_torch_mode=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(
                any(
                    line.split()[:3] == ["CASE", case, "requires"]
                    for line in result.stdout.splitlines()
                ),
                result.stdout,
            )
        tree = ast.parse((ROOT / "compat/tests/torch/test_ecosystem_openmmlab.py").read_text())
        methods = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertTrue({"test_mmengine_train_step", "test_mmcv_convmodule_train_step"} <= methods)

    @staticmethod
    def reports():
        from _ecosystem_openmmlab_cases import PARAMETERS_BY_CASE

        return {
            "case": "mmengine_train_step",
            "device": "cpu",
            "steps": 3,
            "learning_rate": 0.01,
            "seed": 0,
            "mode": "train",
            "runtime": "torch",
            "trainable_parameters": sorted(PARAMETERS_BY_CASE["mmengine_train_step"]),
            "frozen_parameters": [],
            "input_gradient_names": ["sample"],
            "target_requires_grad": False,
            "dependencies": {},
            "tf32": {},
            "runtime_conditions": {},
            "weights_sha256": "same",
            "buffer_names": [],
            "jittor_commit": "same",
            "probe_sha256": "same",
            "fallback_count": 0,
            "fallback_policy": "error",
            "training_protocol": {
                "owner": "mmengine.model.BaseModel.train_step",
                "wrapper": "mmengine.optim.OptimWrapper",
                "wrapper_built_by": "mmengine.registry.OPTIM_WRAPPERS.build",
                "optimizer": "torch.optim.SGD",
                "momentum": 0.9,
                "clip_grad": {"max_norm": 1.0},
                "gradient_stage": "after_clip_before_sgd_step",
                "optimizer_updates": 3,
                "loss_forward_calls": 3,
                "log_keys": ["loss", "loss_regression", "loss_regularization"],
            },
        }

    def test_training_owner_updates_and_real_protocol_are_mandatory(self):
        from _ecosystem_openmmlab_harness import assert_training_reports

        self.device = "cpu"
        good = self.reports()
        candidate = copy.deepcopy(good)
        candidate["runtime"] = "jittor"
        assert_training_reports(self, good["case"], good, candidate)
        for side in ("oracle", "candidate"):
            for field, value in (
                ("owner", "manual SGD loop"),
                ("optimizer_updates", 2),
                ("loss_forward_calls", 0),
                ("gradient_stage", "before_clip"),
                ("momentum", 0.0),
                ("clip_grad", {}),
                ("log_keys", ["loss"]),
            ):
                bad = copy.deepcopy(good if side == "oracle" else candidate)
                bad["training_protocol"][field] = value
                with self.subTest(side=side, field=field), self.assertRaises(AssertionError):
                    assert_training_reports(
                        self,
                        good["case"],
                        bad if side == "oracle" else good,
                        bad if side == "candidate" else candidate,
                    )

    @staticmethod
    def trajectory(case="mmengine_train_step"):
        import numpy as np
        from _ecosystem_openmmlab_cases import PARAMETERS_BY_CASE, BUFFER_NAMES_BY_CASE

        arrays = {}
        for step in range(3):
            prefix = "step/%d/" % step
            for key in (
                "output",
                "loss",
                "loss_regression",
                "loss_regularization",
                "input::sample",
                "target",
                "ingrad::sample",
            ):
                arrays[prefix + key] = np.ones((2, 2))
            for name in PARAMETERS_BY_CASE[case]:
                for kind in ("grad::", "parameter::", "momentum::"):
                    arrays[prefix + kind + name] = np.ones((2, 2))
            for name in BUFFER_NAMES_BY_CASE[case]:
                arrays[prefix + "buffer::" + name] = (
                    np.array(1, dtype="int64")
                    if name.endswith("num_batches_tracked")
                    else np.ones(4)
                )
        return arrays

    def compare_arrays(self, reference, candidate, case="mmengine_train_step"):
        import numpy as np
        from _ecosystem_openmmlab_harness import assert_training_arrays

        self.forward_tolerance, self.backward_tolerance = 2e-3, 1e-2
        self.device = "cpu"
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.npz", Path(directory) / "b.npz"
            np.savez(a, **reference)
            np.savez(b, **candidate)
            with np.load(a) as ref, np.load(b) as actual:
                assert_training_arrays(self, case, ref, actual)

    def test_missing_gradient_momentum_or_loss_is_rejected_even_when_both_omit_it(self):
        for key in (
            "step/0/grad::fc1.weight",
            "step/1/momentum::norm.bias",
            "step/2/loss_regularization",
            "step/0/ingrad::sample",
        ):
            good = self.trajectory()
            bad = copy.deepcopy(good)
            del bad[key]
            for side in ("oracle", "candidate", "both"):
                with self.subTest(key=key, side=side), self.assertRaisesRegex(
                    AssertionError, "missing training tensors"
                ):
                    self.compare_arrays(
                        bad if side in ("oracle", "both") else good,
                        bad if side in ("candidate", "both") else good,
                    )

    def test_bn_counter_buffer_cannot_be_omitted_on_both_sides(self):
        case = "mmcv_convmodule_train_step"
        arrays = self.trajectory(case)
        self.compare_arrays(arrays, arrays, case)
        del arrays["step/2/buffer::block1.bn.num_batches_tracked"]
        with self.assertRaisesRegex(AssertionError, "missing training tensors"):
            self.compare_arrays(arrays, arrays, case)

    def test_shape_nonfinite_extra_keys_and_changed_inputs_are_rejected(self):
        import numpy as np

        for kind in ("shape", "dtype", "nan", "extra", "input"):
            good = self.trajectory()
            bad = copy.deepcopy(good)
            if kind == "shape":
                bad["step/0/output"] = np.ones((1, 2))
            elif kind == "dtype":
                bad["step/0/output"] = np.ones((2, 2), dtype="float32")
            elif kind == "nan":
                bad["step/0/momentum::fc1.bias"][0, 0] = np.nan
            elif kind == "extra":
                bad["unexpected"] = np.ones(1)
            else:
                bad["step/0/input::sample"][0, 0] = 1.00001
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                self.compare_arrays(good, bad)
        good = self.trajectory()
        self.compare_arrays(good, good)

    def test_training_runner_reuses_shared_process_contract(self):
        import _ecosystem_openmmlab_harness as helper

        with mock.patch.object(helper, "_run_command", return_value=({}, "")) as run:
            helper._training_run("oracle", "torch", "mmengine_train_step", Path("a.npz"), "cpu")
        python, case, command = run.call_args.args
        self.assertEqual((python, case), ("oracle", "mmengine_train_step"))
        self.assertIn(str(helper.PROBE), command)
        self.assertIn("--steps", command)
        self.assertIn("0.01", command)

    def test_selected_gate_fails_missing_oracle_dependencies_and_device(self):
        fake = types.ModuleType("_ecosystem_harness")
        fake.REAL_TORCH_PYTHON = ""
        fake.SPEED_RATIO = ""
        fake.EcosystemComparison = unittest.TestCase
        fake._enabled = lambda name: True
        fake._torch_shim_is_active = lambda: True
        fake._distributions_available = lambda names: True
        training_fake = types.ModuleType("_ecosystem_openmmlab_harness")
        training_fake.compare_training = lambda *args: None
        path = ROOT / "compat/tests/torch/test_ecosystem_openmmlab.py"
        spec = importlib.util.spec_from_file_location("openmmlab_admission_probe", path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {
            "_ecosystem_harness": fake,
            "_ecosystem_openmmlab_harness": training_fake,
        }):
            spec.loader.exec_module(module)
        gate = module.EcosystemOpenMMLab()
        with self.assertRaisesRegex(AssertionError, "independent real Torch"):
            gate.setUp()
        module.REAL_TORCH_PYTHON = "real-oracle"
        with mock.patch.dict(os.environ, {"JITTOR_TEST_DEVICES": "cpu,cuda"}):
            with self.assertRaisesRegex(AssertionError, "exactly one"):
                gate.setUp()
        module._distributions_available = lambda names: False
        with mock.patch.dict(os.environ, {"JITTOR_TEST_DEVICES": "cpu"}):
            gate.setUp()
            for method in (gate.test_mmengine_train_step, gate.test_mmcv_convmodule_train_step):
                with self.assertRaisesRegex(AssertionError, "requires available"):
                    method()

    def nox_session(self, environment):
        tree = ast.parse((ROOT / "noxfile.py").read_text())
        selected = [
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.Assign))
            and (
                (isinstance(n, ast.FunctionDef) and n.name == "ecosystem_openmmlab")
                or (
                    isinstance(n, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == "OPENMMLAB_TRAINING_TESTS"
                        for t in n.targets
                    )
                )
            )
        ]
        for node in selected:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
        calls = []

        class Session:
            def error(self, text):
                raise ValueError(text)

            def run(self, *args, **kwargs):
                calls.append((args, kwargs))

        ns = {
            "Path": Path,
            "os": os,
            "shutil": shutil,
            "_session_env": lambda *a: (Path("/isolated"), {}),
            "NOX_STATE_ROOT": Path("/isolated-state"),
            "_hardware_python": lambda: "candidate",
            "_set_hardware_python_config": lambda *a: None,
            "_install_compat_source": lambda *a: None,
            "_run_pytest_once": lambda session, targets, env, *a, **kw: calls.append(
                (targets, dict(env))
            ),
        }
        exec(
            compile(ast.Module(body=selected, type_ignores=[]), "nox-openmmlab-contract", "exec"),
            ns,
        )
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
            Path, "is_file", return_value=True
        ):
            ns["ecosystem_openmmlab"](Session())
        return calls

    def test_nox_cpu_really_disables_cuda_and_requires_execution(self):
        calls = self.nox_session(
            {
                "REAL_TORCH_PYTHON": "oracle",
                "JITTOR_TEST_DEVICES": "cpu",
                "CUDA_VISIBLE_DEVICES": "occupied",
            }
        )
        targets, env = calls[-1]
        self.assertEqual(
            targets,
            (
                "compat/tests/torch/test_ecosystem_openmmlab.py",
                "compat/tests/torch/test_torch_compat_batchnorm_state.py",
            ),
        )
        for flag in (
            "JITTOR_REQUIRE_MMENGINE",
            "JITTOR_REQUIRE_MMCV",
            "JITTOR_REQUIRE_REAL_TORCH",
            "JITTOR_TEST_REQUIRE_EXECUTION",
            "JITTOR_ECOSYSTEM_OPENMMLAB",
        ):
            self.assertEqual(env[flag], "1")
        for flag in (
            "CUDA_VISIBLE_DEVICES",
            "nvcc_path",
            "JT_BUILD_NVCC_PATH",
            "JITTOR_ECOSYSTEM_SPEED_RATIO",
        ):
            self.assertEqual(env[flag], "")
        self.assertEqual(env["use_cuda"], "0")
        self.assertEqual(env["JT_USE_CUDA"], "0")
        self.assertIn("openmmlab", env["JITTOR_HOME"])
        self.assertIn("hasattr(torch, '_C')", calls[0][0][-1])
        self.assertEqual(calls[0][1]["env"]["JITTOR_TORCH_SHIM"], "0")

    def test_nox_rejects_missing_oracle_ambiguous_device_and_unallocated_cuda(self):
        for environment, message in (
            ({}, "REAL_TORCH_PYTHON"),
            ({"REAL_TORCH_PYTHON": "oracle", "JITTOR_TEST_DEVICES": "cpu,cuda"}, "exactly"),
            (
                {"REAL_TORCH_PYTHON": "oracle", "JITTOR_TEST_DEVICES": "cuda"},
                "explicitly allocated",
            ),
        ):
            with self.subTest(environment=environment), self.assertRaisesRegex(ValueError, message):
                self.nox_session(environment)
        calls = self.nox_session(
            {
                "REAL_TORCH_PYTHON": "oracle",
                "JITTOR_TEST_DEVICES": "cuda",
                "CUDA_VISIBLE_DEVICES": "allocated",
                "nvcc_path": "/compiler/nvcc",
            }
        )
        _, env = calls[-1]
        self.assertEqual(env["use_cuda"], "1")
        self.assertEqual(env["JT_BUILD_NVCC_PATH"], "/compiler/nvcc")
        self.assertIn("torch.cuda.is_available()", calls[0][0][-1])
