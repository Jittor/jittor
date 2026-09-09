"""Every torch TF32 spelling reaches the cuBLAS/cuDNN call it selects.

``torch.backends``' six TF32 spellings are views of the two frontend-owned
precision tiers in ``installers/cuda/api.py`` (``_PRECISION_FIELDS``). They are
deliberately *not* views of the deprecated native overrides
``cuda_allow_tf32`` / ``cuda_allow_cudnn_tf32``: an independent Torch frontend
carries its own (matmul, cuDNN) pair into the ops it builds, and native
Runtime-following calls keep following the native policy. See
``docs/notes/float32-precision-policy.md`` and
``refactor-wip/results/2026-09-08-frontend-precision-isolation.md``.

That makes "did this write take effect?" a question about the library call, not
about a flag. These tests answer it the only way a stored value cannot fake:
enable a spelling, run a real float32 CUDA matmul or convolution, and read the
compute type the op logged immediately before calling cuBLAS/cuDNN.

This file used to assert that a torch write moved
``jt.introspection.policy.runtime.cuda_allow_tf32``. That was the pre-7.19
ownership and it stopped being the contract, but "the write reaches the
library" -- what those assertions were reaching for -- still is, and reading a
written value back does not show it: the defect worth catching here is one
where the write is accepted, reads back correct, and the library keeps using
the previous tier. ``test_torch_backends_tf32.py`` pins that the six spellings
agree with each other; this file pins that what they agree on is executed.
"""
from contextlib import ExitStack
import re
import unittest

from _helpers import capability as _test_capability

import jittor as jt
import torch

from jittor.compat.torch.installers.cuda.api import _cuda_runtime

#: The line each op logs immediately before calling the library, at ``vvv``.
_CUBLAS_SELECT = re.compile(
    r"algo select: precision=(\S+) computeType=(\S+) algo=(\S+)")
_CUDNN_SELECT = re.compile(
    r"precision select: precision=(\S+) computeType=(\S+) mathType=(\S+)")

_MATMUL_TF32 = ("high", "CUBLAS_COMPUTE_32F_FAST_TF32",
                "CUBLAS_GEMM_DEFAULT_TENSOR_OP")
_MATMUL_IEEE = ("highest", "CUBLAS_COMPUTE_32F", "CUBLAS_GEMM_DEFAULT")
_CUDNN_TF32 = ("high", "CUDNN_DATA_FLOAT",
               "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION")
_CUDNN_IEEE = ("highest", "CUDNN_DATA_FLOAT", "CUDNN_FMA_MATH")

#: One writer per spelling of each domain's switch. Same six spellings as
#: ``_PRECISION_FIELDS``' table, written out here so a mapping change has to be
#: made twice on purpose.
_MATMUL_SPELLINGS = (
    ("torch.backends.cuda.matmul.allow_tf32",
     lambda on: setattr(torch.backends.cuda.matmul, "allow_tf32", on)),
    ("torch.backends.cuda.matmul.fp32_precision",
     lambda on: setattr(torch.backends.cuda.matmul, "fp32_precision",
                        "tf32" if on else "ieee")),
    ("torch.set_float32_matmul_precision",
     lambda on: torch.set_float32_matmul_precision("high" if on else "highest")),
)

_CUDNN_SPELLINGS = (
    ("torch.backends.cudnn.allow_tf32",
     lambda on: setattr(torch.backends.cudnn, "allow_tf32", on)),
    ("torch.backends.cudnn.conv.fp32_precision",
     lambda on: setattr(torch.backends.cudnn.conv, "fp32_precision",
                        "tf32" if on else "ieee")),
    ("torch.backends.cudnn.rnn.fp32_precision",
     lambda on: setattr(torch.backends.cudnn.rnn, "fp32_precision",
                        "tf32" if on else "ieee")),
)

_NATIVE_OVERRIDES = ("float32_matmul_precision", "use_tensorcore",
                     "cuda_allow_tf32", "cuda_allow_cudnn_tf32")


def _cuda_devices():
    inventory = jt.introspection.capabilities.devices("cuda")
    return inventory.count if inventory.capability.enabled else 0


@unittest.skipUnless(
    _test_capability.check_accelerator('cuda', backend=jt).enabled
    and _cuda_devices(),
    "the executed cuBLAS/cuDNN compute type needs a real CUDA device")
class TestTorchTf32ReachesTheLibraryCall(unittest.TestCase):
    def setUp(self):
        state = _cuda_runtime()
        saved = (torch.get_float32_matmul_precision(),
                 torch.backends.cudnn.allow_tf32, state.matmul_refinement)
        stack = ExitStack()
        self.addCleanup(stack.close)
        # Every native override is off, so nothing but this frontend's own
        # policy can raise the tier of the calls below.
        stack.enter_context(jt.runtime.scope(
            use_cuda=1, float32_matmul_precision="highest", use_tensorcore=0,
            cuda_allow_tf32=0, cuda_allow_cudnn_tf32=0))

        def _restore():
            torch.set_float32_matmul_precision(saved[0])
            torch.backends.cudnn.allow_tf32 = saved[1]
            state.matmul_refinement = saved[2]

        self.addCleanup(_restore)
        # Normalize the high/medium refinement so a boolean spelling means
        # "high"; `test_the_matmul_tier_is_carried_not_just_a_boolean` is where
        # medium is exercised.
        torch.set_float32_matmul_precision("high")
        torch.set_float32_matmul_precision("highest")

    def _selections(self, pattern, prefix, build):
        with jt.log_capture_scope(log_silent=1, log_v=0,
                                  log_vprefix="%s=100" % prefix) as logs:
            jt.fetch_sync([build()])
        found = [match for line in logs
                 for match in pattern.findall(line.get("msg", ""))]
        self.assertTrue(found, "no %s call was executed, so this test would "
                               "pass without exercising anything" % prefix)
        return set(found)

    def _matmul_selection(self):
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        return self._selections(_CUBLAS_SELECT, "cublas_matmul",
                                lambda: torch.matmul(a, b))

    def _conv_selection(self):
        x = torch.randn(1, 2, 8, 8, device="cuda")
        w = torch.randn(3, 2, 3, 3, device="cuda")
        return self._selections(_CUDNN_SELECT, "cudnn_conv",
                                lambda: torch.nn.functional.conv2d(x, w))

    def _native_matmul_selection(self):
        return self._selections(
            _CUBLAS_SELECT, "cublas_matmul",
            lambda: jt.matmul(jt.randn(64, 64), jt.randn(64, 64)))

    def _read_back(self):
        """Every spelling, as the boolean it reports."""
        return {
            "cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cuda.matmul.fp32_precision":
                torch.backends.cuda.matmul.fp32_precision == "tf32",
            "get_float32_matmul_precision":
                torch.get_float32_matmul_precision() != "highest",
            "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn.conv.fp32_precision":
                torch.backends.cudnn.conv.fp32_precision == "tf32",
            "cudnn.rnn.fp32_precision":
                torch.backends.cudnn.rnn.fp32_precision == "tf32",
        }

    def test_every_matmul_spelling_switches_the_executed_cublas_call(self):
        for name, write in _MATMUL_SPELLINGS:
            for enabled, expected in ((True, _MATMUL_TF32),
                                      (False, _MATMUL_IEEE)):
                with self.subTest(spelling=name, enabled=enabled):
                    write(enabled)
                    read_back = self._read_back()
                    self.assertEqual(
                        {key: value for key, value in read_back.items()
                         if not key.startswith("cudnn")},
                        {"cuda.matmul.allow_tf32": enabled,
                         "cuda.matmul.fp32_precision": enabled,
                         "get_float32_matmul_precision": enabled})
                    self.assertEqual(self._matmul_selection(), {expected})

    def test_every_cudnn_spelling_switches_the_executed_cudnn_call(self):
        for name, write in _CUDNN_SPELLINGS:
            for enabled, expected in ((True, _CUDNN_TF32),
                                      (False, _CUDNN_IEEE)):
                with self.subTest(spelling=name, enabled=enabled):
                    write(enabled)
                    read_back = self._read_back()
                    self.assertEqual(
                        {key: value for key, value in read_back.items()
                         if key.startswith("cudnn")},
                        {"cudnn.allow_tf32": enabled,
                         "cudnn.conv.fp32_precision": enabled,
                         "cudnn.rnn.fp32_precision": enabled})
                    self.assertEqual(self._conv_selection(), {expected})

    def test_the_matmul_tier_is_carried_not_just_a_boolean(self):
        # `medium` is the one value the two boolean spellings cannot express.
        # A layer that collapsed the tier to "tf32 on/off" passes every check
        # above and silently gives `medium` callers tf32.
        torch.set_float32_matmul_precision("medium")
        self.assertEqual(torch.get_float32_matmul_precision(), "medium")
        self.assertEqual(
            self._matmul_selection(),
            {("medium", "CUBLAS_COMPUTE_32F_FAST_16BF",
              "CUBLAS_GEMM_DEFAULT_TENSOR_OP")})

    def test_a_torch_write_leaves_native_overrides_and_native_calls_alone(self):
        # The frontend owns its pair; it does not reach into the native policy.
        # This is the direction this file used to assert backwards.
        before = {name: getattr(jt.introspection.policy.runtime, name)
                  for name in _NATIVE_OVERRIDES}
        for _name, write in _MATMUL_SPELLINGS + _CUDNN_SPELLINGS:
            write(True)
        self.assertEqual(
            {name: getattr(jt.introspection.policy.runtime, name)
             for name in _NATIVE_OVERRIDES}, before)
        self.assertEqual(self._matmul_selection(), {_MATMUL_TF32})
        self.assertEqual(self._native_matmul_selection(), {_MATMUL_IEEE})

    def test_a_native_override_does_not_raise_an_explicit_torch_call(self):
        # The other direction: the deprecated overrides can only raise the tier
        # of a native Runtime-following call, never of an op a Torch frontend
        # built with an explicit policy.
        for _name, write in _MATMUL_SPELLINGS + _CUDNN_SPELLINGS:
            write(False)
        with jt.runtime.scope(cuda_allow_tf32=1, cuda_allow_cudnn_tf32=1):
            self.assertEqual(self._matmul_selection(), {_MATMUL_IEEE})
            self.assertEqual(self._conv_selection(), {_CUDNN_IEEE})
            self.assertEqual(self._native_matmul_selection(), {_MATMUL_TF32})


if __name__ == "__main__":
    unittest.main()
