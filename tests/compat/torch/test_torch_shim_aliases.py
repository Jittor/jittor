"""Identity and module-key contracts for the converged Torch shim."""

from __future__ import print_function

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import jittor as jt


DEPLOYED_ONLY_BASELINE_KEYS = {
    "torch.autograd.function",
    "torch.autograd.graph",
    "torch.autograd.variable",
    "torch.distributions.geometric",
    "torch.jit.annotations",
    "torch.linalg",
    "torch.nn.modules.conv",
    "torch.nn.modules.instancenorm",
    "torch.nn.modules.pooling",
    "torch.nn.utils.stateless",
    "torch.onnx",
    "torch.optim.adam",
    "torch.optim.adamw",
    "torch.optim.rmsprop",
    "torch.optim.sgd",
    "torch.sparse",
    "torch.special",
    "torch.testing",
    "torch.utils._python_dispatch",
    "torch.utils.model_zoo",
}


class TestTorchShimAliases(unittest.TestCase):
    def _subprocess_env(self, extra_pythonpath=None):
        env = os.environ.copy()
        python_root = Path(__file__).resolve().parents[3] / "python"
        entries = [os.fspath(path) for path in (extra_pythonpath or ())]
        entries.append(os.fspath(python_root))
        if env.get("PYTHONPATH"):
            entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(entries)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["nvcc_path"] = ""
        env["JITTOR_TORCH_KEEP_CUDA"] = "1"
        return env

    def _run_order(self, source, extra_pythonpath=None):
        result = subprocess.run(
            [sys.executable, "-c", source],
            env=self._subprocess_env(extra_pythonpath),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        line = next(
            line for line in result.stdout.splitlines() if line.startswith("RESULT=")
        )
        return json.loads(line[len("RESULT="):])

    def test_legacy_package_and_audit_submodules_are_same_objects(self):
        pairs = (
            ("jittor.compat.shim", "jittor.torch_shim"),
            ("jittor.compat.shim.bootstrap", "jittor.torch_shim.bootstrap"),
            ("jittor.compat.shim.deploy", "jittor.torch_shim.deploy"),
            ("jittor.compat.shim.cpp_extension", "jittor.torch_shim.cpp_extension"),
            (
                "jittor.compat.shim.cpp_extension.torch_utils",
                "jittor.torch_shim.cpp_extension.torch_utils",
            ),
            (
                "jittor.compat.shim.backends.flash_attention",
                "jittor.torch_shim.flashattn_jittor",
            ),
            (
                "jittor.compat.shim.extensions.readonly",
                "jittor.torch_shim.readonly_extensions",
            ),
        )
        for canonical, legacy in pairs:
            with self.subTest(legacy=legacy):
                self.assertIs(importlib.import_module(legacy), importlib.import_module(canonical))

    def test_legacy_imports_do_not_reexecute_modules(self):
        canonical = importlib.import_module("jittor.compat.shim.backends.flash_attention")
        marker = object()
        canonical._stage7_identity_marker = marker
        try:
            legacy = importlib.import_module("jittor.torch_shim.flashattn_jittor")
            self.assertIs(legacy._stage7_identity_marker, marker)
        finally:
            del canonical._stage7_identity_marker

    def test_preloaded_canonical_child_binds_legacy_parent_attribute(self):
        source = """
import importlib, json
canonical = importlib.import_module(
    'jittor.compat.shim.cpp_extension.torch_utils'
)
legacy = importlib.import_module('jittor.torch_shim.torch_utils')
parent = importlib.import_module('jittor.torch_shim')
assert legacy is canonical
assert parent.torch_utils is canonical
nested = importlib.import_module(
    'jittor.torch_shim.cpp_extension.torch_utils'
)
cpp_parent = importlib.import_module('jittor.torch_shim.cpp_extension')
assert nested is canonical
assert cpp_parent.torch_utils is canonical
print('RESULT=' + json.dumps({'bound': True}))
"""
        self.assertEqual(self._run_order(source), {"bound": True})

    def test_plain_jittor_registers_deployed_only_baseline_keys(self):
        self.assertTrue(DEPLOYED_ONLY_BASELINE_KEYS.issubset(sys.modules))

    def test_plain_jittor_registers_canonical_triton_once(self):
        canonical = sys.modules["jittor.compat.triton"]
        self.assertIs(jt.triton_shim, canonical)
        bare = sys.modules.get("triton")
        self.assertIsNotNone(bare)
        if bare is not canonical:
            self.assertFalse(getattr(bare, "__triton_shim__", False))
            self.assertFalse(getattr(bare, "__jittor_triton_shim__", False))

        self.assertIs(importlib.import_module("jittor.compat.triton"), canonical)
        legacy = importlib.import_module("jittor.triton_shim")
        self.assertIs(legacy, canonical)
        if bare is canonical:
            first = canonical.install(force=True)
            module_ids = {
                name: id(sys.modules[name])
                for name in (
                    "triton",
                    "triton.language",
                    "triton.runtime",
                )
            }
            second = canonical.install(force=True)
            self.assertIs(first, canonical)
            self.assertIs(second, canonical)
            self.assertEqual(
                {
                    name: id(sys.modules[name])
                    for name in module_ids
                },
                module_ids,
            )

    def test_preloaded_real_torch_graph_is_not_claimed_by_plain_jittor(self):
        result = self._run_order("""
import json, sys, types
real = types.ModuleType('torch')
child = types.ModuleType('torch.nn')
real.nn = child
sys.modules['torch'] = real
sys.modules['torch.nn'] = child
import jittor
assert sys.modules['torch'] is real
assert sys.modules['torch.nn'] is child
assert real.nn is child
assert not hasattr(jittor, '_torch_compat_install_context')
print('RESULT=' + json.dumps({'preserved': True}))
""")
        self.assertEqual(result, {"preserved": True})

    def test_compat_installer_remains_idempotent(self):
        from jittor.compat import torch as compat

        before_keys = {name for name in sys.modules if name.startswith("torch.")}
        before_objects = {
            "grad": jt.grad,
            "no_grad": jt.no_grad,
            "nn": jt.nn,
            "functional": jt.nn.functional,
            "interpolate": jt.nn.functional.interpolate,
        }
        self.assertIs(compat.install(jt), jt)
        self.assertIs(compat.install(jt), jt)
        self.assertEqual(
            {name for name in sys.modules if name.startswith("torch.")}, before_keys
        )
        after_objects = {
            "grad": jt.grad,
            "no_grad": jt.no_grad,
            "nn": jt.nn,
            "functional": jt.nn.functional,
            "interpolate": jt.nn.functional.interpolate,
        }
        for name, value in before_objects.items():
            self.assertIs(after_objects[name], value)

    def test_canonical_and_legacy_first_orders_keep_module_graph(self):
        expected = sorted(
            name for name in sys.modules
            if name == "torch" or name.startswith("torch.")
        )
        scripts = (
            """
import importlib, json, sys
import jittor
canonical = importlib.import_module('jittor.compat.shim.backends.flash_attention')
legacy = importlib.import_module('jittor.torch_shim.flashattn_jittor')
assert canonical is legacy
from jittor.compat import torch as compat
compat.install(jittor); compat.install(jittor)
assert sum(r.step == 'core' for r in jittor._torch_compat_install_context.reports) == 1
print('RESULT=' + json.dumps(sorted(k for k in sys.modules if k == 'torch' or k.startswith('torch.'))))
""",
            """
import importlib, json, sys
legacy = importlib.import_module('jittor.torch_shim.flashattn_jittor')
import jittor
canonical = importlib.import_module('jittor.compat.shim.backends.flash_attention')
assert canonical is legacy
from jittor.compat import torch as compat
compat.install(jittor); compat.install(jittor)
assert sum(r.step == 'core' for r in jittor._torch_compat_install_context.reports) == 1
print('RESULT=' + json.dumps(sorted(k for k in sys.modules if k == 'torch' or k.startswith('torch.'))))
""",
        )
        for source in scripts:
            with self.subTest(order=source.splitlines()[2].strip()):
                self.assertEqual(self._run_order(source), expected)

    def test_deployed_first_is_same_module_and_same_module_graph(self):
        from jittor.compat.shim.deploy import deploy

        expected = sorted(
            name for name in sys.modules
            if name == "torch" or name.startswith("torch.")
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "site-packages"
            deploy(target)
            result = self._run_order(
                """
import json, sys
import torch
import jittor
from jittor.compat import torch as compat
compat.install(jittor); compat.install(jittor)
assert torch is jittor is sys.modules['torch']
assert jittor.__version__ == '1.3.11.0'
assert jittor.__torch_version__ == '2.11.0'
assert jittor.version.__version__ == '2.11.0'
assert sum(r.step == 'core' for r in jittor._torch_compat_install_context.reports) == 1
print('RESULT=' + json.dumps(sorted(k for k in sys.modules if k == 'torch' or k.startswith('torch.'))))
""",
                [target],
            )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
