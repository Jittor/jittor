import importlib
import unittest

import numpy as np

import jittor as jt
import torch


FACTORY_NAMES = (
    "arange", "bernoulli", "empty", "empty_like", "full", "full_like",
    "linspace", "multinomial", "normal", "ones", "ones_like", "rand",
    "rand_like", "randint", "randn", "randn_like", "randperm", "tril",
    "triu", "zeros", "zeros_like",
)


class TestTorchFactoryFidelity(unittest.TestCase):

    def test_factory_objects_are_module_level_and_keep_public_identity(self):
        factories = importlib.import_module(
            "jittor.compat.torch.installers.factories")
        for name in FACTORY_NAMES:
            with self.subTest(name=name):
                implementation = getattr(factories, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertEqual(implementation.__name__, name)

    def test_fidelity_report_is_complete_deterministic_and_conservative(self):
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        report = fidelity.fidelity_report(prefix="torch.")
        factory_records = tuple(
            record for record in report
            if record.api in {"torch." + name for name in FACTORY_NAMES}
        )
        self.assertEqual(
            tuple(record.api for record in factory_records),
            tuple("torch." + name for name in FACTORY_NAMES),
        )
        for record in factory_records:
            self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
            self.assertIs(
                fidelity.fidelity_of(record.api).implementation,
                record.implementation,
            )
            self.assertTrue(record.detail)

    def test_independently_imported_factory_executes_on_cpu(self):
        factories = importlib.import_module(
            "jittor.compat.torch.installers.factories")
        with jt.flag_scope(use_cuda=0):
            value = factories.zeros((2, 3), dtype=torch.float32)
        np.testing.assert_array_equal(value.numpy(), np.zeros((2, 3)))

    def test_explicit_meta_empty_matches_torch_metadata_contract(self):
        with jt.flag_scope(use_cuda=0):
            value = torch.empty(
                size=(2, 3), dtype=torch.float32, device="meta")
        self.assertEqual(tuple(value.shape), (2, 3))
        self.assertIs(value.dtype, torch.float32)
        self.assertFalse(value.dtype == torch.float8_e4m3fn)
        self.assertFalse(torch.float8_e4m3fn == value.dtype)
        self.assertEqual(str(value.device), "meta")
        self.assertTrue(value.is_meta)
        self.assertFalse(value.is_cpu)
        self.assertFalse(value.is_cuda)
        self.assertEqual(value.get_device(), -1)
        self.assertEqual(value.numel(), 6)

    def test_empty_like_implementation_is_family_owned_and_runs_on_cpu(self):
        factories = importlib.import_module(
            "jittor.compat.torch.installers.factories")
        self.assertEqual(factories.empty_like.__module__, factories.__name__)
        self.assertIs(getattr(torch, "empty_like"), factories.empty_like)
        record = importlib.import_module(
            "jittor.compat.torch.fidelity").fidelity_of("torch.empty_like")
        self.assertIn("device", record.detail)
        with jt.flag_scope(use_cuda=0):
            source = torch.ones((2, 3), dtype=torch.float64)
            value = factories.empty_like(source)
        self.assertEqual(tuple(value.shape), (2, 3))
        self.assertEqual(value.dtype, source.dtype)


if __name__ == "__main__":
    unittest.main()


def _python_int_records(t, device):
    records = []
    for data in ([], [2 ** 40, -(2 ** 40)]):
        value = t.tensor(data, dtype=int, device=device)
        assert value.dtype == t.int64
        assert value.device.type == device.split(":")[0]
        records.append({"shape": list(value.shape), "dtype": str(value.dtype),
                        "device": value.device.type, "values": value.cpu().tolist()})
    return records


def test_python_int_dtype_matches_independent_torch(tmp_path):
    import inspect
    import json
    import os
    import subprocess
    import pytest
    from _helpers.child_process import child_env, default_timeout

    oracle = os.environ.get("REAL_TORCH_PYTHON", "")
    if not oracle:
        if os.environ.get("JITTOR_REQUIRE_REAL_TORCH") == "1":
            pytest.fail("REAL_TORCH_PYTHON is required")
        pytest.skip("REAL_TORCH_PYTHON is not configured")
    device = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
    assert device in ("cpu", "npu", "npu:0")
    device = "npu:0" if device.startswith("npu") else "cpu"
    source = "import torch, json\nassert not hasattr(torch, '_torch_compat_install_context')\n"
    if device.startswith("npu"):
        source += "import torch_npu\nassert torch.npu.is_available()\n"
    source += inspect.getsource(_python_int_records)
    source += "\nprint(json.dumps(_python_int_records(torch, %r)))\n" % device
    env = child_env(without_torch_mode=True, repo_paths=False)
    if device == "cpu":
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    result = subprocess.run([oracle, "-c", source], env=env, cwd=str(tmp_path),
                            capture_output=True, text=True, timeout=default_timeout())
    assert result.returncode == 0, result.stdout + result.stderr
    expected = json.loads(result.stdout.strip().splitlines()[-1])
    before = jt.core.backend_fallback_count()
    assert _python_int_records(torch, device) == expected
    assert jt.core.backend_fallback_count() == before
