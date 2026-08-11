"""Identity and module-key contracts for the converged Torch shim."""

from __future__ import print_function

import importlib
import sys
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

    def test_plain_jittor_registers_deployed_only_baseline_keys(self):
        self.assertTrue(DEPLOYED_ONLY_BASELINE_KEYS.issubset(sys.modules))

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


if __name__ == "__main__":
    unittest.main()
