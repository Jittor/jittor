"""Wall-clock comparison against PyTorch at a size where kernels matter.

The parity gate in ``test_ecosystem_parity`` also reports timings, but its models
are deliberately tiny so the numbers there measure dispatch overhead more than
kernels. This module runs the same two-interpreter harness over the realistic
configurations in ``_ecosystem_speed`` and reports one table.

It is a measurement, not an assertion, unless ``JITTOR_ECOSYSTEM_SPEED_RATIO``
is set -- a shared machine makes an unconditional timing gate flaky, and a flaky
gate is worse than an honest number. Run it as::

    REAL_TORCH_PYTHON=/path/to/python python -m pytest \\
        tests/compat/torch/test_ecosystem_speed.py -s

Add ``JITTOR_ECOSYSTEM_SPEED_RATIO=1.2`` on a quiet machine to turn the ratios
into a gate.

The runner reports the fastest of at least ten timed repeats, which is the
right statistic -- interference only ever makes a sample slower. The same
PyTorch CPU case has come back 25% apart between shorter whole-suite runs,
wider than the ratios being judged; ``JITTOR_ECOSYSTEM_REPEATS`` may raise,
but not lower, this floor.
"""

import os
import unittest

from _ecosystem_harness import (
    REAL_TORCH_PYTHON,
    EcosystemComparison,
    _cuda_is_available,
    _torch_shim_is_active,
)


def _speed_repeats():
    value = int(os.environ.get("JITTOR_ECOSYSTEM_REPEATS", "10"))
    if value < 10:
        raise ValueError("JITTOR_ECOSYSTEM_REPEATS must be at least 10")
    return value


SPEED_CASES = (
    "large_convnet",
    "large_transformers_bert",
    "large_transformers_gpt2",
    "large_transformers_llama",
    "large_transformers_qwen3",
    "large_transformers_vit",
    "large_diffusers_unet2d",
)


class _Speed:
    """Test methods shared by the CPU and CUDA measurement classes.

    A plain mixin, not a ``TestCase``: pytest collects every ``TestCase``
    subclass regardless of a leading underscore, and an abstract base with
    real test methods would run all six large models ungated in the middle
    of the ordinary suite.
    """

    # A 12-layer model accumulates over many more operations than the two-layer
    # parity cases, so the acceptable band is wider. Correctness is still owned
    # by the small cases, which compare every gradient at a tight tolerance.
    forward_tolerance = 2e-2
    backward_tolerance = 5e-2
    repeats = _speed_repeats()

    def test_large_convnet(self):
        self._compare("large_convnet")

    def test_large_transformers_bert(self):
        self._compare("large_transformers_bert")

    def test_large_transformers_gpt2(self):
        self._compare("large_transformers_gpt2")

    def test_large_transformers_llama(self):
        self._compare("large_transformers_llama")

    def test_large_transformers_qwen3(self):
        self._compare("large_transformers_qwen3")

    def test_large_transformers_vit(self):
        self._compare("large_transformers_vit")

    def test_large_diffusers_unet2d(self):
        self._compare("large_diffusers_unet2d")


@unittest.skipUnless(REAL_TORCH_PYTHON, "REAL_TORCH_PYTHON is not configured")
@unittest.skipUnless(_torch_shim_is_active(), "this interpreter does not run torch as Jittor")
@unittest.skipUnless(
    os.environ.get("JITTOR_ECOSYSTEM_LARGE", "").strip() not in ("", "0"),
    "set JITTOR_ECOSYSTEM_LARGE=1 to run the realistic-size measurement",
)
class EcosystemSpeedCPU(_Speed, EcosystemComparison):
    device = "cpu"


@unittest.skipUnless(REAL_TORCH_PYTHON, "REAL_TORCH_PYTHON is not configured")
@unittest.skipUnless(_torch_shim_is_active(), "this interpreter does not run torch as Jittor")
@unittest.skipUnless(_cuda_is_available(), "CUDA is unavailable")
@unittest.skipUnless(
    os.environ.get("JITTOR_ECOSYSTEM_LARGE", "").strip() not in ("", "0"),
    "set JITTOR_ECOSYSTEM_LARGE=1 to run the realistic-size measurement",
)
class EcosystemSpeedCUDA(_Speed, EcosystemComparison):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
