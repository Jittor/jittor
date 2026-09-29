"""The parity harness must put Jittor on the device it claims to measure.

Jittor has no per-tensor device: one global flag moves the whole graph, and on
a machine with a GPU that flag starts out *on*. A CPU comparison that merely
refrains from requesting CUDA therefore measures Jittor on the accelerator
against PyTorch on the CPU. That does not fail -- the numbers still agree,
because both runtimes are correct -- it silently reports the CPU half of the
2.0 goals as covered when it never ran, and it turned a 1.8x slowdown into a
reported 20x speedup.

These tests exercise the selection helper directly so the contract is checked
without an oracle interpreter or a downstream library.
"""

from _helpers import capability as _test_capability

import os
import sys
from pathlib import Path
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import jittor as jt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import child_process  # noqa: E402
from _helpers.runtime_policy import fixture_stack

import _ecosystem_runner  # noqa: E402
import _ecosystem_harness  # noqa: E402


class _StubTorch(object):
    """Stands in for the ``torch`` argument on the Jittor paths."""


class _StubFlags(object):
    use_cuda = 0
    use_acl = 0


def _stub_observation_runtime(flags, acl):
    from jittor._runtime.state import RuntimeContext, RuntimeState
    from jittor._runtime.introspection import EffectivePolicy
    @contextmanager
    def scope(**changes):
        before = {key: getattr(flags, key) for key in changes}
        try:
            for key, value in changes.items():
                setattr(flags, key, value)
            yield
        finally:
            for key, value in before.items():
                setattr(flags, key, value)
    runtime = RuntimeState(RuntimeContext(flags), scope)
    introspection = SimpleNamespace(
        policy=EffectivePolicy(SimpleNamespace(), runtime.context),
        capabilities=SimpleNamespace(backend=lambda name: SimpleNamespace(
            enabled=bool(acl and name == "acl"), failed=False)))
    return runtime, introspection


class _SharedNumpyTensor(object):
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class TestEcosystemDeviceSelection(unittest.TestCase):
    def setUp(self):
        self._policy_stack = fixture_stack(self)

    def test_cpu_request_turns_cuda_off(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1 if _test_capability.check_accelerator('cuda', backend=jt).enabled else 0))
            _ecosystem_runner._select_device(_StubTorch(), "jittor", "cpu", policy_stack=_test_policy_stack)
            self.assertEqual(jt.introspection.policy.runtime.use_cuda, 0)

    def test_cpu_request_is_reported_as_cpu(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1 if _test_capability.check_accelerator('cuda', backend=jt).enabled else 0))
            _ecosystem_runner._select_device(_StubTorch(), "jittor", "cpu", policy_stack=_test_policy_stack)
            self.assertEqual(
                _ecosystem_runner._device_in_use(_StubTorch(), "jittor", "cpu"), "cpu"
            )

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "CUDA is unavailable")
    def test_cuda_request_turns_cuda_on_and_is_reported(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=0))
            _ecosystem_runner._select_device(_StubTorch(), "jittor", "cuda", policy_stack=_test_policy_stack)
            self.assertEqual(jt.introspection.policy.runtime.use_cuda, 1)
            self.assertEqual(
                _ecosystem_runner._device_in_use(_StubTorch(), "jittor", "cuda"), "cuda"
            )

    def test_npu_request_requires_acl_and_is_reported_separately(self):
        flags = _StubFlags()
        runtime, observation = _stub_observation_runtime(flags, True)
        with mock.patch.object(jt, "runtime", runtime):
            with mock.patch.object(jt, "introspection", observation):
                _ecosystem_runner._select_device(_StubTorch(), "jittor", "npu", policy_stack=self._policy_stack)
                self.assertEqual(flags.use_cuda, 1)
                self.assertEqual(flags.use_acl, 1)
                self.assertEqual(
                    _ecosystem_runner._device_in_use(
                        _StubTorch(), "jittor", "npu"
                    ),
                    "npu",
                )

    @unittest.skipUnless(_test_capability.check_accelerator("acl", backend=jt).enabled,
                         "ACL is unavailable")
    def test_npu_request_moves_host_indices_before_embedding(self):
        import torch
        from jittor._runtime.fallback import forbid_backend_fallbacks

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            move = _ecosystem_runner._select_device(
                torch, "jittor", "npu", policy_stack=self._policy_stack)
            indices = move(torch.from_numpy(np.array([0, 2], dtype=np.int64)))
            weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="npu")
            output = torch.nn.functional.embedding(indices, weights)
            jt.sync([indices, weights, output], device_sync=True)
            for value in (indices, weights, output):
                self.assertEqual(value.location(), "device")
                self.assertEqual(value.device_id, 0)
                self.assertIn(value.placement_backend, (-1, 2))
            np.testing.assert_array_equal(output.detach().cpu().numpy(), [[1.0, 2.0], [5.0, 6.0]])
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    def test_npu_request_fails_without_acl(self):
        runtime, observation = _stub_observation_runtime(_StubFlags(), False)
        with mock.patch.object(jt, "runtime", runtime), mock.patch.object(jt, "introspection", observation):
            with self.assertRaisesRegex(SystemExit, "ACL is unavailable"):
                _ecosystem_runner._select_device(_StubTorch(), "jittor", "npu", policy_stack=self._policy_stack)

    def test_jittor_tensors_are_never_moved_by_hand(self):
        """The returned callable is identity: Jittor moves the graph, not tensors."""
        move = _ecosystem_runner._select_device(_StubTorch(), "jittor", "cpu", policy_stack=self._policy_stack)
        sentinel = object()
        self.assertIs(move(sentinel), sentinel)

    def test_shared_package_site_is_inserted_without_duplicates(self):
        original = list(sys.path)

        def restore_path():
            sys.path[:] = original

        self.addCleanup(restore_path)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {"JITTOR_ECOSYSTEM_PACKAGE_SITE": directory},
            ):
                sys.path.extend([directory, directory])
                actual = _ecosystem_runner._activate_package_site()
                self.assertEqual(actual, str(Path(directory).resolve()))
                self.assertEqual(sys.path[0], actual)
                self.assertEqual(sys.path.count(actual), 1)

    def test_harness_selects_independent_reference_package_site(self):
        with mock.patch.object(
            _ecosystem_harness, "PACKAGE_SITE", "/packages/python39"
        ):
            with mock.patch.object(
                _ecosystem_harness,
                "REFERENCE_PACKAGE_SITE",
                "/packages/python310",
            ):
                with mock.patch.object(
                    _ecosystem_harness, "REFERENCE_SHARES_PACKAGE_SITE", False
                ):
                    self.assertEqual(
                        _ecosystem_harness._runner_package_site(child_process.PYTHON),
                        "/packages/python39",
                    )
                    self.assertEqual(
                        _ecosystem_harness._runner_package_site(
                            "/oracle/bin/python"
                        ),
                        "/packages/python310",
                    )

    def test_harness_shares_package_site_for_compatible_abis(self):
        with mock.patch.object(
            _ecosystem_harness, "PACKAGE_SITE", "/packages/shared"
        ):
            with mock.patch.object(
                _ecosystem_harness, "REFERENCE_PACKAGE_SITE", ""
            ):
                with mock.patch.object(
                    _ecosystem_harness, "REFERENCE_SHARES_PACKAGE_SITE", True
                ):
                    self.assertEqual(
                        _ecosystem_harness._runner_package_site(
                            "/oracle/bin/python"
                        ),
                        "/packages/shared",
                    )

    def test_correctness_snapshot_does_not_alias_runtime_storage(self):
        storage = np.array([1.0, 2.0], dtype="float32")
        snapshot = _ecosystem_runner._numpy_snapshot(_SharedNumpyTensor(storage))
        storage[:] = -1.0
        np.testing.assert_array_equal(snapshot, np.array([1.0, 2.0], dtype="float32"))


    def test_comparison_rejects_nonfinite_values_on_either_side(self):
        for value in (np.nan, np.inf, -np.inf):
            for actual, expected in (([value], [1.0]), ([1.0], [value])):
                with self.subTest(actual=actual, expected=expected):
                    with self.assertRaisesRegex(AssertionError, "non-finite"):
                        _ecosystem_harness._divergence(actual, expected, 1e-3)

    def test_comparison_rejects_broadcastable_shape_mismatch(self):
        with self.assertRaisesRegex(AssertionError, "shape mismatch"):
            _ecosystem_harness._divergence(np.ones((1, 2)), np.ones((2, 2)), 1e-3)

    def test_comparison_preserves_finite_error_scale(self):
        self.assertAlmostEqual(
            _ecosystem_harness._divergence([1.0, 2.1], [1.0, 2.0], 1e-3),
            0.05,
        )




class TestEcosystemEvidenceContracts(unittest.TestCase):
    """Metadata mocks exercise fail-closed evidence without model computation."""

    @staticmethod
    def _parameter(enabled, grad=None):
        value = SimpleNamespace(requires_grad=enabled, grad=grad)
        value.requires_grad_ = lambda enabled: setattr(value, "requires_grad", enabled)
        return value

    @staticmethod
    def _model(parameters):
        return SimpleNamespace(named_parameters=lambda: parameters.items())

    @staticmethod
    def _loaded(values):
        class Loaded(dict):
            @property
            def files(self):
                return list(self)
        return Loaded(values)

    def test_transfer_rejects_missing_extra_shape_and_dtype(self):
        value = SimpleNamespace(shape=(2, 2), dtype="torch.float32")
        available = {"weight": value}
        accepted = self._loaded({"weight": SimpleNamespace(shape=(2, 2), dtype="float32")})
        _ecosystem_runner._validate_transfer_state(available, accepted)
        bad_states = [
            {},
            {"weight": value, "extra": value},
            {"weight": SimpleNamespace(shape=(1, 2), dtype="float32")},
            {"weight": SimpleNamespace(shape=(2, 2), dtype="float64")},
        ]
        for values in bad_states:
            with self.subTest(values=values), self.assertRaises(AssertionError):
                _ecosystem_runner._validate_transfer_state(available, self._loaded(values))

    def test_assert_policy_rejects_freeze_and_unfreeze_without_repair(self):
        for changed_name, changed_value in (("base", True), ("lora", False)):
            with self.subTest(parameter=changed_name):
                parameters = {"base": self._parameter(False), "lora": self._parameter(True)}
                model = self._model(parameters)
                policy = _ecosystem_runner._parameter_grad_policy(model)
                parameters[changed_name].requires_grad = changed_value
                before = _ecosystem_runner._parameter_grad_policy(model)
                for value in parameters.values():
                    value.requires_grad_ = mock.Mock(side_effect=AssertionError("unexpected write"))
                with self.assertRaisesRegex(AssertionError, "requires_grad"):
                    _ecosystem_runner._assert_parameter_grad_policy(model, policy)
                self.assertEqual(_ecosystem_runner._parameter_grad_policy(model), before)
                for value in parameters.values():
                    value.requires_grad_.assert_not_called()

    def test_assert_policy_rejects_parameter_set_drift_without_repair(self):
        for change in ("missing", "extra"):
            with self.subTest(change=change):
                parameters = {"base": self._parameter(False), "lora": self._parameter(True)}
                model = self._model(parameters)
                policy = _ecosystem_runner._parameter_grad_policy(model)
                if change == "missing":
                    del parameters["base"]
                else:
                    parameters["extra"] = self._parameter(True)
                before = _ecosystem_runner._parameter_grad_policy(model)
                with self.assertRaisesRegex(AssertionError, "parameter set"):
                    _ecosystem_runner._assert_parameter_grad_policy(model, policy)
                self.assertEqual(_ecosystem_runner._parameter_grad_policy(model), before)

    def test_assert_policy_accepts_unchanged_state_without_writes(self):
        parameters = {"base": self._parameter(False), "lora": self._parameter(True)}
        model = self._model(parameters)
        policy = _ecosystem_runner._parameter_grad_policy(model)
        for value in parameters.values():
            value.requires_grad_ = mock.Mock(side_effect=AssertionError("unexpected write"))
        _ecosystem_runner._assert_parameter_grad_policy(model, policy)
        for value in parameters.values():
            value.requires_grad_.assert_not_called()

    def test_both_runtimes_missing_same_trainable_gradient_fail(self):
        parameters = {"lora_a": self._parameter(True, object()),
                      "lora_b": self._parameter(True)}
        model = self._model(parameters)
        policy = _ecosystem_runner._parameter_grad_policy(model)
        for runtime in ("oracle", "candidate"):
            with self.subTest(runtime=runtime), self.assertRaisesRegex(AssertionError, "lora_b"):
                _ecosystem_runner._required_gradients(model, {}, policy)

    def test_required_inputs_and_frozen_parameter_contract(self):
        parameters = {"base": self._parameter(False), "lora": self._parameter(True, object())}
        model = self._model(parameters)
        policy = _ecosystem_runner._parameter_grad_policy(model)
        inputs = {"input_ids": self._parameter(False), "embeds": self._parameter(True, object())}
        gradients, na = _ecosystem_runner._required_gradients(model, inputs, policy)
        self.assertEqual(set(gradients), {"grad::lora", "ingrad::embeds"})
        self.assertEqual(na, ["input_ids"])
        inputs["embeds"].grad = None
        with self.assertRaisesRegex(AssertionError, "input gradient"):
            _ecosystem_runner._required_gradients(model, inputs, policy)
        inputs["embeds"].grad = object()
        parameters["base"].grad = object()
        with self.assertRaisesRegex(AssertionError, "frozen parameter"):
            _ecosystem_runner._required_gradients(model, inputs, policy)

    def test_primary_output_structure_keeps_return_contract(self):
        tensor = SimpleNamespace(shape=(2, 8, 128))
        self.assertEqual(_ecosystem_runner._output_structure({"logits": tensor}, tensor),
                         {"type": "dict", "keys": ["logits"], "primary_shape": [2, 8, 128]})



class TestEcosystemNpuEvidence(unittest.TestCase):
    @staticmethod
    def _candidate(shape=(2,), location="device", placement=2, device_id=0):
        return SimpleNamespace(shape=shape, dtype="float32", placement_backend=placement,
                               device_id=device_id, location=lambda: location)

    def test_candidate_requires_placement_and_physical_residency(self):
        observe = _ecosystem_runner._npu_tensor_evidence
        report = observe(self._candidate(), "jittor", "weight")
        self.assertEqual(report["location"], "device")
        self.assertEqual(observe(self._candidate(placement=-1), "jittor", "weight")["location"], "device")
        for kwargs in ({"placement": 0}, {"device_id": 1},
                       {"location": "cpu"}, {"location": "none"}, {"location": "disk"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(AssertionError):
                observe(self._candidate(**kwargs), "jittor", "weight")

    def test_empty_exception_never_accepts_cpu_placement_or_storage(self):
        observe = _ecosystem_runner._npu_tensor_evidence
        result = observe(self._candidate(shape=(0,), location="none"), "jittor", "empty")
        self.assertIn("residency_exception", result)
        for kwargs in ({"placement": 0, "location": "none"}, {"location": "cpu"}):
            with self.assertRaises(AssertionError):
                observe(self._candidate(shape=(0,), **kwargs), "jittor", "empty")

    def test_native_requires_npu_and_records_dtype(self):
        tensor = SimpleNamespace(shape=(2, 8), dtype="torch.int64",
                                 device=SimpleNamespace(type="npu", index=0))
        report = _ecosystem_runner._npu_tensor_evidence(tensor, "torch", "input_ids")
        self.assertEqual(report["dtype"], "int64")
        tensor.device.type = "cpu"
        with self.assertRaises(AssertionError):
            _ecosystem_runner._npu_tensor_evidence(tensor, "torch", "input_ids")

    def test_npu_observation_precedes_correctness_snapshot(self):
        import inspect
        source = inspect.getsource(_ecosystem_runner._run)
        observation = source.index("npu_evidence = _npu_evidence(")
        synchronization = source.rfind("_synchronize(", 0, observation)
        snapshot = source.index('arrays = {"__output__": _numpy_snapshot(output)}')
        self.assertGreater(synchronization, 0)
        self.assertLess(observation, snapshot)

    def test_full_inventory_includes_buffers_inputs_outputs_and_gradients(self):
        tensor = SimpleNamespace(shape=(1,), dtype="torch.float32",
                                 device=SimpleNamespace(type="npu", index=0))
        model = SimpleNamespace(named_parameters=lambda: [("p", tensor)],
                                named_buffers=lambda: [("b", tensor)])
        torch = SimpleNamespace(_C=object(), __file__="native/torch.py",
                                npu=SimpleNamespace(device_count=lambda: 1, is_available=lambda: True))
        with mock.patch.dict(sys.modules, {"torch_npu": SimpleNamespace(__file__="native/torch_npu.py")}):
            report = _ecosystem_runner._npu_evidence(
                torch, "torch", model, {"x": tensor}, tensor, {"grad::p": tensor})
        self.assertEqual(set(report["tensors"]),
                         {"parameter::p", "buffer::b", "input::x", "primary_output", "grad::p"})
        self.assertEqual(report["input_dtypes"], {"x": "float32"})
        self.assertEqual(report["primary_dtype"], "float32")




class TestNpuInventoryValidation(unittest.TestCase):
    @staticmethod
    def _reports():
        def tensors(native):
            entries = {}
            for name in ("parameter::base", "parameter::lora", "buffer::rope", "grad::lora",
                         "input::input_ids", "primary_output"):
                record = {"shape": [2], "dtype": "int64" if name.startswith("input::") else "float32",
                          "device_id": 0}
                record.update({"device_type": "npu"} if native else
                              {"placement_backend": -1, "location": "device"})
                entries[name] = record
            return entries
        oracle = {"backend_identity": {"runtime": "torch_npu", "device_count": 1,
                                       "module": "native/torch_npu", "torch_module": "native/torch"},
                  "tensors": tensors(True), "input_dtypes": {"input_ids": "int64"},
                  "primary_dtype": "float32"}
        candidate = {"backend_identity": {"runtime": "jittor", "device_count": 1,
                                          "module": "source/jittor", "build_backend": "acl",
                                          "registered_backends": ["cpu", "acl"]},
                     "tensors": tensors(False), "input_dtypes": {"input_ids": "int64"},
                     "primary_dtype": "float32"}
        return oracle, candidate

    def _check(self, oracle, candidate):
        return _ecosystem_harness._validate_npu_evidence(
            oracle, candidate, ["lora"], ["base"], ["input_ids"])

    def test_complete_inventory_accepts_follow_runtime_on_real_acl(self):
        self._check(*self._reports())

    def test_rejects_missing_inventory_or_gradient(self):
        oracle, candidate = self._reports()
        with self.assertRaises(AssertionError):
            self._check(oracle, None)
        del candidate["tensors"]["grad::lora"]
        with self.assertRaises(AssertionError):
            self._check(oracle, candidate)

    def test_rejects_dtype_shape_and_buffer_key_mismatch(self):
        for field, bad in (("dtype", "float16"), ("shape", [1, 2])):
            oracle, candidate = self._reports()
            candidate["tensors"]["parameter::lora"][field] = bad
            with self.assertRaises(AssertionError):
                self._check(oracle, candidate)
        oracle, candidate = self._reports()
        del candidate["tensors"]["buffer::rope"]
        with self.assertRaises(AssertionError):
            self._check(oracle, candidate)

    def test_rejects_cpu_residency_wrong_provider_or_device(self):
        for target, field, value in (("tensor", "location", "cpu"),
                                     ("tensor", "placement_backend", 0),
                                     ("tensor", "device_id", 1),
                                     ("identity", "build_backend", "cuda"),
                                     ("identity", "device_count", 2)):
            oracle, candidate = self._reports()
            record = candidate["tensors"]["primary_output"] if target == "tensor" else candidate["backend_identity"]
            record[field] = value
            with self.assertRaises(AssertionError):
                self._check(oracle, candidate)
        oracle, candidate = self._reports()
        oracle["backend_identity"]["runtime"] = "jittor"
        with self.assertRaises(AssertionError):
            self._check(oracle, candidate)

    def test_empty_exception_is_explicit_and_never_cpu(self):
        oracle, candidate = self._reports()
        for report in (oracle, candidate):
            report["tensors"]["buffer::rope"]["shape"] = [0]
        value = candidate["tensors"]["buffer::rope"]
        value["location"] = "none"
        with self.assertRaises(AssertionError):
            self._check(oracle, candidate)
        value["residency_exception"] = "zero-sized tensor has no allocation"
        self._check(oracle, candidate)
        value["location"] = "cpu"
        with self.assertRaises(AssertionError):
            self._check(oracle, candidate)



class TestEcosystemAdamW3Protocol(unittest.TestCase):
    """Metadata checks; real updates require the separately selected NPU case."""

    def test_trajectory_has_every_step_gradient_and_updated_parameter(self):
        names = ["lora_{}".format(index) for index in range(8)]
        keys = _ecosystem_runner._training_trajectory_keys(names)
        self.assertEqual(len(keys), 83)
        for step in range(3):
            self.assertIn("step::{}::loss".format(step), keys)
            for name in names:
                self.assertIn("step::{}::grad::{}".format(step, name), keys)
                self.assertIn("step::{}::param::{}".format(step, name), keys)
                self.assertIn("step::{}::delta::{}".format(step, name), keys)

    def test_training_registration_reuses_ms_swift_builder(self):
        cases = _ecosystem_runner._ecosystem_cases.CASES
        self.assertEqual(cases["ms_swift_lora_llama_adamw3"], cases["ms_swift_lora_llama"])

    def test_training_source_has_real_updates_and_synchronization(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(_ecosystem_runner._run_adamw3))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
        self.assertIn("AdamW", attributes)
        self.assertIn("backward", attributes)
        self.assertIn("step", attributes)
        self.assertIn("sync", attributes)
        self.assertTrue(any(isinstance(node.func, ast.Name) and node.func.id == "_synchronize"
                            for node in calls))

    def test_training_policy_is_asserted_without_repair(self):
        import inspect
        source = inspect.getsource(_ecosystem_runner._run_adamw3)
        self.assertIn("_assert_parameter_grad_policy(model, policy)", source)
        self.assertNotIn("_restore_parameter_grad_policy", source)
        self.assertNotIn("requires_grad_(", source)

    def test_training_inventory_precedes_each_phase_snapshot(self):
        import inspect
        source = inspect.getsource(_ecosystem_runner._run_adamw3)
        self.assertLess(source.index("backward_evidence = _npu_evidence"),
                        source.index('arrays[prefix + "loss"] = _numpy_snapshot(loss)'))
        self.assertLess(source.index("update_evidence = _npu_evidence"),
                        source.index('arrays.update({prefix + "param::"'))
        self.assertIn('"trajectory_dtypes"', source)
        self.assertIn('"loss_dtype"', source)

    def test_each_training_phase_delegates_to_shared_validator(self):
        from unittest.mock import patch
        names = ['lora_{}'.format(index) for index in range(8)]
        phases = ('backward_npu_evidence', 'update_npu_evidence')
        def report():
            return {'protocol': 'adamw3', 'steps': 3,
                    'trainable_parameters': names, 'frozen_parameters': ['backbone'],
                    'input_grad_not_applicable': ['input_ids'], 'optimizer': {'lr': 0.001},
                    'step_observations': [
                        dict(step=step, trainable_parameters=names, device='npu',
                             fallback_count=0, loss_dtype='float32', **{
                                 phase: {'primary_dtype': 'float32',
                                         'input_dtypes': {'input_ids': 'int64'},
                                         'step': step, 'phase': phase}
                                 for phase in phases}) for step in range(3)]}
        native, shim = report(), report()
        for observation in native['step_observations']:
            observation['fallback_count'] = None
        calls = []
        def validator(a, b, trainable, frozen, input_na):
            calls.append((a, b))
            self.assertEqual(trainable, names)
            self.assertEqual(frozen, ['backbone'])
            self.assertEqual(input_na, ['input_ids'])
            if len(calls) == 6:
                raise AssertionError('shared residency rejection')
        with patch.object(_ecosystem_harness, '_validate_npu_evidence', validator):
            with self.assertRaisesRegex(AssertionError, 'shared residency rejection'):
                _ecosystem_harness.EcosystemComparison()._compare_adamw3(
                    None, None, native, shim)
        self.assertEqual(len(calls), 6)
        # Missing/unknown/nonzero candidate counters are rejected, never zero.
        from copy import deepcopy
        for bad_count in (None, 1, False, 'missing'):
            rejected = deepcopy(shim)
            first = rejected['step_observations'][0]
            if bad_count == 'missing':
                del first['fallback_count']
            else:
                first['fallback_count'] = bad_count
            with self.subTest(candidate_count=bad_count), self.assertRaises(AssertionError):
                _ecosystem_harness.EcosystemComparison()._compare_adamw3(
                    None, None, native, rejected)
        rejected_native = deepcopy(native)
        rejected_native['step_observations'][0]['fallback_count'] = 0
        with self.assertRaises(AssertionError):
            _ecosystem_harness.EcosystemComparison()._compare_adamw3(
                None, None, rejected_native, shim)

        for index, (a, b) in enumerate(calls):
            self.assertIs(a, native['step_observations'][index // 2][phases[index % 2]])
            self.assertIs(b, shim['step_observations'][index // 2][phases[index % 2]])

    def test_current_npu_transfer_and_shared_validator_are_retained(self):
        import inspect
        self.assertIn('.to("npu")', inspect.getsource(_ecosystem_runner._select_device))
        source = inspect.getsource(_ecosystem_harness.EcosystemComparison._compare_adamw3)
        self.assertIn('_validate_npu_evidence(', source)
        self.assertNotIn('backend_identity', source)
    def test_native_counter_is_unknown_in_both_protocols(self):
        import ast
        import inspect
        training = ast.parse(inspect.getsource(_ecosystem_runner._run_adamw3))
        initial = [n for n in ast.walk(training) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'count' for t in n.targets)]
        self.assertTrue(any(isinstance(n.value, ast.Constant) and n.value.value is None
                            for n in initial))
        one_step = ast.parse(inspect.getsource(_ecosystem_runner._run))
        count = next(n.value for n in ast.walk(one_step) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'fallback_count' for t in n.targets))
        self.assertIsInstance(count, ast.IfExp)
        self.assertIsNone(count.orelse.value)



class TestLoraPerformanceMetadata(unittest.TestCase):
    def test_statistics_retains_all_samples_and_defines_throughput(self):
        values = list(range(1, 11))
        result = _ecosystem_runner._performance_statistics(values)
        self.assertEqual(result["durations_seconds"], values)
        self.assertEqual(result["median_seconds"], 5.5)
        self.assertEqual(result["min_seconds"], 1)
        self.assertAlmostEqual(result["p10_seconds"], 1.9)
        self.assertAlmostEqual(result["p90_seconds"], 9.1)
        self.assertAlmostEqual(result["input_tokens_per_second"], 512 / 5.5)
        self.assertAlmostEqual(result["loss_tokens_per_second"], 511 / 5.5)

    def test_statistics_rejects_short_or_invalid_measurement(self):
        for values in ([1.0] * 9, [0.0] * 10, [-1.0] * 10,
                       [float("nan")] * 10, [float("inf")] * 10):
            with self.assertRaises(AssertionError):
                _ecosystem_runner._performance_statistics(values)

    def test_timed_step_has_update_and_sync_without_host_snapshot(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(_ecosystem_runner._run_lora_performance))
        step = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "step")
        calls = [node for node in ast.walk(step) if isinstance(node, ast.Call)]
        attributes = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
        self.assertTrue({"zero_grad", "backward", "step", "sync"} <= attributes)
        detach = next(node for node in calls if isinstance(node.func, ast.Attribute)
                      and node.func.attr == "detach")
        sync = next(node for node in calls if isinstance(node.func, ast.Attribute)
                    and node.func.attr == "sync")
        returned = next(node for node in ast.walk(step) if isinstance(node, ast.Return))
        retained = returned.value.elts[0]
        self.assertIsInstance(retained, ast.Name)
        assignment = next(node for node in ast.walk(step) if isinstance(node, ast.Assign)
                          and node.value is detach)
        self.assertEqual(assignment.targets[0].id, retained.id)
        self.assertLess(detach.lineno, sync.lineno)
        self.assertIn(retained.id, {node.id for node in ast.walk(sync.args[0])
                                   if isinstance(node, ast.Name)})
        self.assertFalse(any(isinstance(node, ast.Call) for node in ast.walk(returned)))
        self.assertFalse({"cpu", "numpy", "item", "savez"} & attributes)
        self.assertFalse(any(isinstance(node.func, ast.Name) and node.func.id == "_numpy_snapshot"
                             for node in calls))

    def test_shared_inventory_validator_covers_warmup_and_final(self):
        from unittest.mock import patch
        fields = dict(timed_steps=10, trainable_parameters=['lora'], frozen_parameters=['base'],
                      input_grad_not_applicable=['input_ids'], warmup_npu_evidence={'phase': 'warm'},
                      npu_evidence={'phase': 'final'})
        native = dict(fields, fallback_count=None, step_fallback_counts=[None] * 10)
        shim = dict(fields, fallback_count=0, fallback_policy='error', step_fallback_counts=[0] * 10)
        calls = []
        def validator(*args):
            calls.append(args)
            if len(calls) == 2:
                raise AssertionError('physical residency failure')
        with patch.object(_ecosystem_harness, '_validate_npu_evidence', validator):
            with self.assertRaisesRegex(AssertionError, 'physical residency failure'):
                _ecosystem_harness.EcosystemComparison()._validate_lora_performance_evidence(native, shim)
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][0], native['warmup_npu_evidence'])
        self.assertIs(calls[1][0], native['npu_evidence'])
        for value in (None, False, 1):
            invalid = dict(shim, fallback_count=value)
            with self.subTest(value=value), self.assertRaises(AssertionError):
                _ecosystem_harness.EcosystemComparison()._validate_lora_performance_evidence(native, invalid)
        with self.assertRaises(AssertionError):
            _ecosystem_harness.EcosystemComparison()._validate_lora_performance_evidence(
                dict(native, fallback_count=0), shim)
    def test_artifacts_reject_missing_key_wrong_dtype_and_shape(self):
        from types import SimpleNamespace
        names = ['lora_{}'.format(i) for i in range(88)]
        keys = {'losses'} | {'final::' + n for n in names} | {'grad::' + n for n in names}
        report = dict(trainable_parameters=names, timed_steps=10,
                      artifact_dtypes={key: 'float32' for key in keys},
                      npu_evidence={'tensors': {'parameter::' + n: {'shape': [2, 4]} for n in names}})
        class Snapshot(dict):
            @property
            def files(self): return list(self)
        original = Snapshot({key: SimpleNamespace(dtype='float32', shape=(13,) if key == 'losses' else (2, 4)) for key in keys})
        check = _ecosystem_harness.EcosystemComparison()._validate_lora_performance_artifacts
        check(original, report)
        for defect in ('missing', 'dtype', 'shape'):
            sample = Snapshot(original)
            if defect == 'missing': del sample['losses']
            elif defect == 'dtype': sample['losses'] = SimpleNamespace(dtype='float64', shape=(13,))
            else: sample['losses'] = SimpleNamespace(dtype='float32', shape=(1, 13))
            with self.subTest(defect=defect), self.assertRaises(AssertionError): check(sample, report)

    def test_loss_residency_rejection_is_not_skipped(self):
        from unittest.mock import patch
        fields = dict(timed_steps=10, trainable_parameters=['lora'], frozen_parameters=['base'],
                      input_grad_not_applicable=['input_ids'], warmup_npu_evidence={'tensors': {}},
                      npu_evidence={'tensors': {}},
                      loss_npu_evidence=[{'shape': [], 'dtype': 'float32', 'location': 'cpu'}] * 13)
        native = dict(fields, fallback_count=None, step_fallback_counts=[None] * 10)
        shim = dict(fields, fallback_count=0, fallback_policy='error', step_fallback_counts=[0] * 10)
        calls = []
        def validator(a, b, *args):
            calls.append(b)
            if b['tensors'].get('primary_output', {}).get('location') == 'cpu':
                raise AssertionError('loss on CPU')
        with patch.object(_ecosystem_harness, '_validate_npu_evidence', validator):
            with self.assertRaisesRegex(AssertionError, 'loss on CPU'):
                _ecosystem_harness.EcosystemComparison()._validate_lora_performance_evidence(native, shim)
        self.assertEqual(len(calls), 3)

    def test_real_output_contract_is_observed_outside_timed_step(self):
        import ast
        import inspect
        source = inspect.getsource(_ecosystem_runner._run_lora_performance)
        tree = ast.parse(source)
        step = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'step')
        step_source = ast.unparse(step)
        self.assertIn('return (detached_loss, result, gradients)', step_source)
        self.assertNotIn('_output_structure', step_source)
        self.assertNotIn('_npu_tensor_evidence', step_source)
        self.assertIn('_output_structure(result, output)', source)
        self.assertNotIn('_output_structure(output, output)', source)

if __name__ == "__main__":
    unittest.main()
