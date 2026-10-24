"""Pure-host negative controls for the exact RNG-stream comparator, not NPU evidence."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


class TestWorkerRngArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[2] / "agent/skills/verl-torch-compat/scripts"
        spec = importlib.util.spec_from_file_location("_worker_rng_probe", scripts / "probe_worker_rng_state.py")
        module = importlib.util.module_from_spec(spec)
        original = list(sys.path)
        try:
            sys.path.insert(0, str(scripts))
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = original
        cls.compare = staticmethod(module.compare_draws)
        cls.provenance = staticmethod(module.validate_resume_provenance)
        cls.advanced = staticmethod(module.require_advanced)
        cls.continuation = staticmethod(module.validate_continuation)
        cls.provider = staticmethod(module.provider_identity)
        cls.runtime = staticmethod(module.runtime_identity)

    def fixture(self):
        return {name: np.arange(9 if name.endswith("normal") or name == "numpy" else 7,
                               dtype=np.float64 if name in ("python", "numpy") else np.float32) for name in
                ("cpu_uniform", "cpu_normal", "npu_uniform", "npu_normal", "python", "numpy")}

    def test_complete_exact_streams(self):
        self.assertEqual(self.compare(self.fixture(), self.fixture()), sorted(self.fixture()))

    def test_each_missing_stream_is_rejected(self):
        for name in self.fixture():
            actual = self.fixture()
            del actual[name]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "incomplete"):
                self.compare(self.fixture(), actual)

    def test_each_changed_stream_is_rejected_even_for_one_ulp(self):
        for name in self.fixture():
            actual = self.fixture()
            actual[name][1] = np.nextafter(actual[name][1], actual[name].dtype.type(2))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "continuation mismatch"):
                self.compare(self.fixture(), actual)

    def test_bad_shapes_dtypes_and_nonfinite_samples_are_rejected(self):
        for bad in (np.zeros((1, 9), dtype=np.float32), np.arange(9, dtype=np.float64),
                    np.full(9, np.nan, dtype=np.float32), np.full(9, np.inf, dtype=np.float32)):
            actual = self.fixture()
            actual["npu_normal"] = bad
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                self.compare(actual, actual)

    def test_matching_constant_streams_are_not_rng_evidence(self):
        expected = self.fixture()
        expected["npu_normal"] = np.zeros(9, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "invalid RNG sample"):
            self.compare(expected, expected)

    def test_fresh_process_checks_pid_not_just_new_invocation_uuid(self):
        receipt = dict(framework="jittor", baseline_pid=101, baseline_token="a" * 32)
        self.provenance(receipt, dict(framework="jittor"), 102, "b" * 32)
        with self.assertRaisesRegex(ValueError, "different process"):
            self.provenance(receipt, dict(framework="jittor"), 101, "b" * 32)
        with self.assertRaisesRegex(ValueError, "token"):
            self.provenance(receipt, dict(framework="jittor"), 102, "a" * 32)
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.provenance(receipt, dict(framework="torch"), 102, "b" * 32)

    def test_every_stream_must_advance_not_repeat_a_nonconstant_pattern(self):
        before = self.fixture()
        after = {name: value + 1 for name, value in before.items()}
        self.assertEqual(self.advanced(before, after), sorted(before))
        for name in before:
            fixed = dict(after, **{name: before[name]})
            with self.subTest(stream=name), self.assertRaisesRegex(ValueError, "did not advance: " + name):
                self.advanced(before, fixed)

    def test_no_op_restore_fails_for_each_stream(self):
        expected = self.fixture()
        unrestored = {name: value + 1 for name, value in expected.items()}
        self.assertEqual(self.continuation(expected, unrestored, expected), sorted(expected))
        for name in expected:
            no_op = dict(expected, **{name: unrestored[name]})
            with self.subTest(stream=name), self.assertRaisesRegex(ValueError, "continuation mismatch: " + name):
                self.continuation(expected, unrestored, no_op)
            unchanged = dict(unrestored, **{name: expected[name]})
            with self.subTest(already_matching=name), self.assertRaisesRegex(ValueError, "did not advance: " + name):
                self.continuation(expected, unchanged, expected)

    def test_runtime_identity_observes_loaded_bindings_without_invoking_them(self):
        def unused_provider(*args, **kwargs):
            raise AssertionError("provenance must not invoke an RNG provider")

        core = SimpleNamespace(__file__=__file__, get_cpu_rng_state=unused_provider,
                               set_cpu_rng_state=unused_provider)
        jt = SimpleNamespace(__version__="candidate-fixture", core=core)
        torch = SimpleNamespace(__version__="torch-fixture", get_rng_state=unused_provider,
                                set_rng_state=unused_provider, rand=unused_provider, randn=unused_provider,
                                npu=SimpleNamespace(get_rng_state=unused_provider, set_rng_state=unused_provider))
        identity = self.runtime(torch, jt)
        self.assertEqual(identity["versions"]["jittor"], "candidate-fixture")
        self.assertEqual(identity["versions"]["torch"], "torch-fixture")
        self.assertEqual(identity["versions"]["numpy"], np.__version__)
        self.assertEqual(identity["versions"]["python"], sys.version)
        self.assertEqual(len(identity["native_core_sha256"]), 64)
        self.assertEqual(set(identity["providers"]), {
            "cpu_get", "cpu_set", "npu_get", "npu_set", "uniform", "normal",
            "python_get", "python_set", "numpy_get", "numpy_set", "native_cpu_get", "native_cpu_set"})
        self.assertEqual(identity["providers"]["npu_get"], self.provider(unused_provider))
        for provider in identity["providers"].values():
            self.assertEqual(len(provider["artifact_sha256"]), 64)
            self.assertTrue(provider["module"])
            self.assertTrue(provider["callable"])
        with self.assertRaisesRegex(RuntimeError, "native core"):
            self.runtime(torch, SimpleNamespace(__version__="candidate-fixture", core=SimpleNamespace(
                get_cpu_rng_state=unused_provider, set_cpu_rng_state=unused_provider)))
        with self.assertRaisesRegex(TypeError, "callable"):
            self.provider(None)
        with self.assertRaisesRegex(RuntimeError, "inspectable"):
            self.provider(len)  # builtins has no separately inspectable artifact.

    def test_runtime_versions_and_provider_identity_must_match_receipt(self):
        runtime = {"versions": {"python": "3.test", "torch": "2.test", "numpy": "test"},
                   "providers": {"npu_get": {"artifact_sha256": "a" * 64}},
                   "native_core_sha256": "b" * 64}
        protocol = dict(framework="jittor", runtime=runtime)
        receipt = dict(protocol, baseline_pid=101, baseline_token="a" * 32)
        self.provenance(receipt, protocol, 102, "b" * 32)
        for key, bad in (("versions", dict(runtime["versions"], torch="changed")),
                         ("providers", {"npu_get": {"artifact_sha256": "c" * 64}}),
                         ("native_core_sha256", "d" * 64)):
            different = dict(protocol, runtime=dict(runtime, **{key: bad}))
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, "protocol"):
                self.provenance(receipt, different, 102, "b" * 32)
        missing = dict(receipt)
        del missing["runtime"]
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.provenance(missing, protocol, 102, "b" * 32)


if __name__ == "__main__":
    unittest.main()
