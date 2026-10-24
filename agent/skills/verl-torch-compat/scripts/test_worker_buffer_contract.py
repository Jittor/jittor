"""Host-only positive/negative controls for initial nonpersistent buffer evidence."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from worker_buffer_contract import (buffer_shapes, capture_initial_buffers, expected_inv_freq,
                                    rope_config, validate_buffer_arrays, validate_buffer_manifest)


def loading_fixture(framework):
    active = framework == "jittor"
    return {"scope": "initialization strategy only; not an exhaustive adapter inventory",
            "requested_materialized_cpu": active, "active_materialized_cpu": active,
            "mode": "materialized_cpu_constructor" if active else "upstream_default",
            "adapter_source_sha256": "a" * 64 if active else None,
            "supported_versions": ["5.9.0"] if active else [], "meta_support_claimed": False}


def add_buffer_fixture(manifest, arrays, prefix=""):
    """Synthetic evidence only; never used to fill a real model's buffers."""
    manifest["rope_config"] = {"head_dim": 8, "rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 1.0}
    manifest["initial_buffers"] = {}
    manifest["buffer_count"] = 2
    for name in buffer_shapes():
        raw = prefix + name
        key = "initial_buffer/" + raw
        manifest["initial_buffers"][raw] = {"shape": [4], "dtype": "torch.float32", "device": "npu:0",
                                            "persistent": False, "array_key": key}
        arrays[key] = expected_inv_freq(manifest["rope_config"]).astype(np.float32)


class TestWorkerBufferContract(unittest.TestCase):
    def test_independent_float64_formula_and_config_gate(self):
        config = {"head_dim": 8, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}}
        actual = expected_inv_freq(config)
        self.assertEqual(actual.dtype, np.dtype("float64"))
        np.testing.assert_allclose(actual, [1., .1, .01, .001], rtol=1e-15)
        for change in ({"head_dim": 16}, {"rope_scaling": {"factor": 2}},
                       {"rope_parameters": {"rope_theta": 1000000}},
                       {"rope_parameters": {"rope_theta": 10000, "rope_type": "dynamic"}}):
            with self.assertRaises(ValueError):
                rope_config(dict(config, **change))

    def test_capture_aliases_requires_residency_before_snapshot_and_never_writes(self):
        tensor = SimpleNamespace(shape=(4,), dtype="torch.float32", device="npu:0")
        calls, arrays = [], {}
        class Module:
            config = {"head_dim": 8, "rope_theta": 10000}
            def named_buffers(self, remove_duplicate=True):
                calls.append(("named_buffers", remove_duplicate))
                return [(name, tensor) for name in buffer_shapes()]
            def state_dict(self):
                return {}
        def device_assert(value):
            self.assertIs(value, tensor)
            calls.append("device_assert")
        def snapshot(name, value):
            self.assertIs(value, tensor)
            self.assertEqual(calls[-1], "device_assert")
            calls.append("snapshot")
            arrays[name] = np.asarray([1, .1, .01, .001], np.float32)
        record = SimpleNamespace(report={}, tensor=snapshot)
        capture_initial_buffers(Module(), record, device_assert)
        self.assertEqual(calls[0], ("named_buffers", False))
        validate_buffer_manifest(record.report)
        validate_buffer_arrays(record.report, arrays)

    def test_both_worker_comparators_fail_closed(self):
        from test_compare_training_worker import fixture as critic_fixture
        from test_compare_actor_worker import fixture as actor_fixture
        from compare_training_worker import compare as critic_compare
        from compare_actor_worker import compare as actor_compare
        for fixture, compare, old_schema in ((critic_fixture, critic_compare, 2), (actor_fixture, actor_compare, 1)):
            for kind in ("old_schema", "missing_both", "missing_array_both", "wrong_formula_both", "nonfinite",
                         "wrong_shape", "cpu", "persistent", "unknown", "collision", "rope_config", "loading_provenance"):
                with self.subTest(comparator=compare.__module__, kind=kind):
                    oracle, oa = fixture("torch")
                    candidate, ca = fixture("jittor")
                    name = next(iter(buffer_shapes()))
                    key = "initial_buffer/" + name
                    for manifest, arrays in ((oracle, oa), (candidate, ca)):
                        if kind == "old_schema":
                            manifest["schema"] = old_schema
                        elif kind == "missing_both":
                            del manifest["initial_buffers"][name]
                            del arrays[key]
                        elif kind == "missing_array_both":
                            del arrays[key]
                        elif kind == "wrong_formula_both":
                            arrays[key] = np.zeros(4, np.float32)
                    if kind == "nonfinite":
                        ca[key][0] = np.nan
                    elif kind == "wrong_shape":
                        ca[key] = np.zeros((1, 4), np.float32)
                    elif kind == "cpu":
                        candidate["initial_buffers"][name]["device"] = "cpu"
                    elif kind == "persistent":
                        candidate["initial_buffers"][name]["persistent"] = True
                    elif kind == "unknown":
                        candidate["initial_buffers"]["unrecognized"] = candidate["initial_buffers"].pop(name)
                    elif kind == "collision":
                        candidate["initial_buffers"]["_fsdp_wrapped_module." + name] = deepcopy(candidate["initial_buffers"][name])
                    elif kind == "rope_config":
                        candidate["rope_config"]["rope_theta"] = 1000000
                    elif kind == "loading_provenance":
                        del candidate["loading_provenance"]
                    with self.assertRaises(ValueError):
                        compare(oracle, oa, candidate, ca, atol=100, rtol=100)

    def test_initial_buffer_cross_framework_comparison_remains_exact(self):
        from test_compare_training_worker import fixture as critic_fixture
        from test_compare_actor_worker import fixture as actor_fixture
        from compare_training_worker import compare as critic_compare
        from compare_actor_worker import compare as actor_compare
        for fixture, compare in ((critic_fixture, critic_compare), (actor_fixture, actor_compare)):
            oracle, oa = fixture("torch")
            candidate, ca = fixture("jittor")
            key = "initial_buffer/model.rotary_emb.inv_freq"
            ca[key][1] = np.nextafter(ca[key][1], np.float32(0))
            result = compare(oracle, oa, candidate, ca, atol=100, rtol=100)
            self.assertEqual(result["status"], "fail")
            self.assertIn(key, [failure["tensor"] for failure in result["failures"]])


if __name__ == "__main__":
    unittest.main()
