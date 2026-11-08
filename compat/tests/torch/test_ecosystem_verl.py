"""Real VERL components in its separate NumPy 2 / Transformers 5 environment.

Use ``nox -s ecosystem_verl`` after provisioning both interpreters. This is
component parity, not PPO rollout, Ray/FSDP2 execution, or a speed gate.
"""

import os
import unittest

from _ecosystem_harness import (
    REAL_TORCH_PYTHON,
    EcosystemComparison,
    _torch_shim_is_active,
    _distributions_available,
)

from _ecosystem_cases import CASES


_DEVICE = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
if _DEVICE not in ("cpu", "cuda"):
    raise ValueError("VERL component gate needs exactly JITTOR_TEST_DEVICES=cpu or cuda")


@unittest.skipUnless(REAL_TORCH_PYTHON, "REAL_TORCH_PYTHON is not configured")
@unittest.skipUnless(_torch_shim_is_active(), "this interpreter does not run torch as Jittor")
class EcosystemVERL(EcosystemComparison):
    device = _DEVICE

    def setUp(self):
        super().setUp()
        for case in ("verl_ppo_gae_vanilla", "verl_distillation_k3"):
            requirements = CASES[case][1]
            self.assertTrue(
                _distributions_available(requirements),
                "VERL component gate requires available dependencies: "
                + ", ".join(requirements)
                + "; check JITTOR_ECOSYSTEM_PACKAGE_SITE for split source/site installs",
            )

    def test_verl_ppo_gae_vanilla(self):
        self._compare(
            "verl_ppo_gae_vanilla",
            required_gradients=(
                "grad::policy.weight",
                "grad::policy.bias",
                "ingrad::features",
            ),
        )

    def test_verl_distillation_k3(self):
        self._compare(
            "verl_distillation_k3",
            required_gradients=(
                "grad::student.weight",
                "grad::student.bias",
                "ingrad::features",
                "ingrad::teacher",
            ),
        )


if __name__ == "__main__":
    unittest.main()
