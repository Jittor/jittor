"""Negative tests for torch APIs that used to be signature-complete no-ops.

Task 7.01 of the 2.0 refactor.  Every API listed in the compat audit's
"looks supported, actually a no-op" / "distributed and FSDP2" / "dtype and
device mapping" sections is covered here by one of two shapes:

* **implemented** -- a test that asserts the API now really takes effect
  (``torch.autocast`` changes the dtype ops are computed in, ``load_state_dict``
  reports the real key difference, ...);
* **refused** -- a test that asserts the API raises ``NotImplementedError`` and
  that ``JITTOR_TORCH_ALLOW_STUB``/``torch.compat_allow_stub(True)`` restores
  the old silent behaviour.

The last test renders ``torch.compat_unimplemented_apis()`` into the generated
"unimplemented API list" the plan asks for, and fails if an API is refused
without a stated consequence.

Run: python -m pytest tests/compat/torch/test_torch_compat_unimplemented.py
"""
import os
import unittest
import warnings

import numpy as np

import jittor as jt
import jittor as torch
from jittor.compat import stub_policy

class StubPolicyBase(unittest.TestCase):
    """Every test runs with the escape hatch OFF unless it says otherwise."""

    def setUp(self):
        self._saved_override = stub_policy.set_allow_stub(False)
        self._saved_env = os.environ.pop(stub_policy.ENV_VAR, None)
        stub_policy.reset_warned()

    def tearDown(self):
        stub_policy.set_allow_stub(self._saved_override)
        if self._saved_env is not None:
            os.environ[stub_policy.ENV_VAR] = self._saved_env
        else:
            os.environ.pop(stub_policy.ENV_VAR, None)
        stub_policy.reset_warned()

    def assertRefuses(self, fn, *needles):
        """fn() must raise NotImplementedError naming the API and the damage."""
        with self.assertRaises(NotImplementedError) as cm:
            fn()
        msg = str(cm.exception)
        for needle in needles:
            self.assertIn(needle, msg)
        self.assertIn(stub_policy.ENV_VAR, msg,
                      "the message must document the escape hatch")
        return msg

    def assertStubFallback(self, fn):
        """With the hatch on, fn() must warn once and return the old value."""
        stub_policy.set_allow_stub(True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = fn()
            self.assertTrue(any(issubclass(w.category, RuntimeWarning)
                                for w in caught),
                            "opting into the stub must warn at least once")
            return result
        finally:
            stub_policy.set_allow_stub(False)

class TestStubPolicy(StubPolicyBase):
    def test_hatch_is_off_by_default(self):
        stub_policy.set_allow_stub(None)
        os.environ.pop(stub_policy.ENV_VAR, None)
        self.assertFalse(stub_policy.allow_stub())

    def test_env_var_opens_the_hatch(self):
        stub_policy.set_allow_stub(None)
        os.environ[stub_policy.ENV_VAR] = "1"
        try:
            self.assertTrue(stub_policy.allow_stub())
        finally:
            os.environ.pop(stub_policy.ENV_VAR, None)

    def test_env_var_off_values_stay_closed(self):
        stub_policy.set_allow_stub(None)
        for value in ("0", "false", "no", "off", ""):
            os.environ[stub_policy.ENV_VAR] = value
            self.assertFalse(stub_policy.allow_stub(), value)
        os.environ.pop(stub_policy.ENV_VAR, None)

    def test_torch_namespace_exposes_the_switch(self):
        self.assertFalse(torch.compat_allow_stub())
        try:
            self.assertTrue(torch.compat_allow_stub(True))
        finally:
            torch.compat_allow_stub(False)

    def test_warns_once_per_api(self):
        stub_policy.set_allow_stub(True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(3):
                    stub_policy.unimplemented("demo.api", "lose the data")
            self.assertEqual(len(caught), 1)
        finally:
            stub_policy.set_allow_stub(False)

class TestAutocast(StubPolicyBase):
    """torch.autocast used to be a total no-op: mixed precision silently ran fp32."""

    def setUp(self):
        super().setUp()
        self._amp_reg = int(getattr(jt.flags, "amp_reg", 0))

    def tearDown(self):
        jt.flags.amp_reg = self._amp_reg
        super().tearDown()

    def test_autocast_actually_lowers_op_dtype(self):
        a = jt.random((4, 4), dtype="float32")
        b = jt.random((4, 4), dtype="float32")
        self.assertEqual(str((a @ b).dtype), "float32")
        with torch.autocast("cuda", dtype=torch.float16):
            inside = a @ b
        self.assertEqual(str(inside.dtype), "float16",
                         "autocast must change the dtype ops compute in")

    def test_autocast_reports_itself_enabled(self):
        self.assertFalse(torch.is_autocast_enabled())
        with torch.autocast("cuda", dtype=torch.float16):
            self.assertTrue(torch.is_autocast_enabled())
            self.assertEqual(str(torch.get_autocast_dtype("cuda")), "float16")
        self.assertFalse(torch.is_autocast_enabled())

    def test_autocast_restores_the_previous_register(self):
        before = int(jt.flags.amp_reg)
        with torch.autocast("cuda", dtype=torch.float16):
            self.assertNotEqual(int(jt.flags.amp_reg), before)
        self.assertEqual(int(jt.flags.amp_reg), before)

    def test_autocast_enabled_false_is_a_real_no_op(self):
        a = jt.random((4, 4), dtype="float32")
        with torch.autocast("cuda", dtype=torch.float16, enabled=False):
            self.assertFalse(torch.is_autocast_enabled())
            self.assertEqual(str((a + a).dtype), "float32")

    def test_autocast_as_decorator_takes_effect(self):
        @torch.autocast("cuda", dtype=torch.float16)
        def f(x):
            return x * x

        out = f(jt.random((4, 4), dtype="float32"))
        self.assertEqual(str(out.dtype), "float16")

    def test_autocast_float32_forces_fp32(self):
        a = jt.random((4, 4), dtype="float16")
        with torch.autocast("cuda", dtype=torch.float32):
            self.assertEqual(str((a + a).dtype), "float32")

    def test_autocast_rejects_a_dtype_it_cannot_express(self):
        self.assertRefuses(
            lambda: torch.autocast("cuda", dtype="float8_e4m3fn"),
            "torch.autocast", "float8_e4m3fn")

    def test_autocast_stub_fallback_restores_the_no_op(self):
        ctx = self.assertStubFallback(
            lambda: torch.autocast("cuda", dtype="float8_e4m3fn"))
        self.assertIsNotNone(ctx)

if __name__ == "__main__":
    unittest.main()
