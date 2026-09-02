"""One semantic, one state: torch.backends' TF32 switches all agree.

torch spells "may fp32 math use reduced-precision tensor cores" three ways per
domain, and Jittor keeps one flag per domain. Each spelling used to hold its
own state:

* ``fp32_precision`` was the literal ``"ieee"`` on all four backend objects --
  it never reported tf32 being on, and assigning to it did nothing at all;
* ``get_float32_matmul_precision()`` read a string that
  ``matmul.allow_tf32 = True`` never updated, so the two disagreed after any
  write that did not go through the one spelling holding the state.

Whether tf32 helps is a numerics question. Whether the six spellings of it
report the same thing is not, and that is what these pin. The table below is
the same one ``installers/cuda.py`` is built from, written out independently so
a change to the mapping has to be made twice, on purpose.
"""
import unittest

import jittor as torch
import jittor as jt

from jittor.compat.torch.installers import cuda as _cuda_installer
from jittor.compat.torch.installers.cuda import (
    _FP32_PRECISIONS,
    _TF32_FLAGS,
    _tf32_get,
    _tf32_set,
)


#: (domain, jittor flag, [every torch spelling that is a view of it])
#: A spelling is (getter, setter) over the live torch namespace.
_SPELLINGS = (
    ("matmul", "cuda_allow_tf32", (
        ("torch.backends.cuda.matmul.allow_tf32",
         lambda: torch.backends.cuda.matmul.allow_tf32,
         lambda v: setattr(torch.backends.cuda.matmul, "allow_tf32", v)),
        ("torch.backends.cuda.matmul.fp32_precision",
         lambda: torch.backends.cuda.matmul.fp32_precision == "tf32",
         lambda v: setattr(torch.backends.cuda.matmul, "fp32_precision",
                           "tf32" if v else "ieee")),
        ("torch.set_float32_matmul_precision",
         lambda: torch.get_float32_matmul_precision() != "highest",
         lambda v: torch.set_float32_matmul_precision("high" if v else "highest")),
    )),
    ("cudnn", "cuda_allow_cudnn_tf32", (
        ("torch.backends.cudnn.allow_tf32",
         lambda: torch.backends.cudnn.allow_tf32,
         lambda v: setattr(torch.backends.cudnn, "allow_tf32", v)),
        ("torch.backends.cudnn.conv.fp32_precision",
         lambda: torch.backends.cudnn.conv.fp32_precision == "tf32",
         lambda v: setattr(torch.backends.cudnn.conv, "fp32_precision",
                           "tf32" if v else "ieee")),
        ("torch.backends.cudnn.rnn.fp32_precision",
         lambda: torch.backends.cudnn.rnn.fp32_precision == "tf32",
         lambda v: setattr(torch.backends.cudnn.rnn, "fp32_precision",
                           "tf32" if v else "ieee")),
    )),
)


class Base(unittest.TestCase):
    def setUp(self):
        self._saved = {flag: getattr(jt.flags, flag, None)
                       for flag in _TF32_FLAGS.values()}
        self._acl = getattr(jt, "acl_allow_hf32", None)
        self._fallback = dict(_cuda_installer._TF32_FALLBACK)

    def tearDown(self):
        for flag, value in self._saved.items():
            if value is not None and hasattr(jt.flags, flag):
                setattr(jt.flags, flag, value)
        if self._acl is not None:
            jt.acl_allow_hf32 = self._acl
        _cuda_installer._TF32_FALLBACK.clear()
        _cuda_installer._TF32_FALLBACK.update(self._fallback)

    @staticmethod
    def _force(domain, enabled):
        """Set the domain's state through whichever store this build has.

        A CPU-only build has neither `jt.flags.cuda_allow_tf32` nor
        `cuda_allow_cudnn_tf32`; the settings then live in the installer's
        fallback dict. The tests must exercise the behaviour on the build they
        are running on, not skip it -- these six spellings have to agree
        everywhere, and the CPU build is where they are actually run.
        """
        _tf32_set(domain, enabled)


class TestEverySpellingIsAViewOfTheSameFlag(Base):
    def test_the_table_covers_both_domains(self):
        self.assertEqual({domain for domain, _flag, _s in _SPELLINGS},
                         set(_TF32_FLAGS))
        for domain, flag, _spellings in _SPELLINGS:
            self.assertEqual(_TF32_FLAGS[domain], flag)

    def test_writing_the_underlying_state_shows_up_in_every_spelling(self):
        for domain, _flag, spellings in _SPELLINGS:
            for enabled in (True, False):
                self._force(domain, enabled)
                for name, get, _set in spellings:
                    with self.subTest(spelling=name, enabled=enabled):
                        self.assertEqual(bool(get()), enabled)

    def test_writing_any_spelling_shows_up_in_all_the_others(self):
        # The regression that motivated this: each spelling used to own its own
        # state, so this cross-check failed for every pair that did not share
        # one.
        for domain, flag, spellings in _SPELLINGS:
            for writer, _get, write in spellings:
                for enabled in (True, False):
                    write(enabled)
                    with self.subTest(writer=writer, enabled=enabled,
                                      reader="the domain's own state"):
                        self.assertEqual(_tf32_get(domain), enabled)
                    if hasattr(jt.flags, flag):
                        with self.subTest(writer=writer, enabled=enabled,
                                          reader="jt.flags." + flag):
                            self.assertEqual(
                                bool(getattr(jt.flags, flag)), enabled)
                    for reader, read, _w in spellings:
                        with self.subTest(writer=writer, reader=reader,
                                          enabled=enabled):
                            self.assertEqual(bool(read()), enabled)

    def test_the_two_domains_stay_independent(self):
        # matmul and cuDNN are separate switches in torch as well; a fix that
        # merged them would pass every test above and be wrong.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = False
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "ieee")
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.backends.cudnn.allow_tf32)


class TestFloat32MatmulPrecision(Base):
    def test_highest_means_off_and_high_medium_mean_on(self):
        for precision, enabled in (("highest", False), ("high", True),
                                   ("medium", True), ("highest", False)):
            with self.subTest(precision=precision):
                torch.set_float32_matmul_precision(precision)
                self.assertEqual(torch.get_float32_matmul_precision(), precision)
                self.assertEqual(torch.backends.cuda.matmul.allow_tf32, enabled)

    def test_it_does_not_drift_from_allow_tf32(self):
        # The exact bug: the string was independent state.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = True
        self.assertNotEqual(torch.get_float32_matmul_precision(), "highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")

    def test_the_high_medium_refinement_survives_a_round_trip(self):
        # Jittor has one flag, so "medium" cannot be distinguished from "high"
        # by the flag alone; remember the refinement so a caller that set
        # "medium" does not silently read back "high".
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = True
        self.assertEqual(torch.get_float32_matmul_precision(), "medium")

    def test_it_still_rejects_what_torch_rejects(self):
        with self.assertRaises(TypeError):
            torch.set_float32_matmul_precision(1)
        with self.assertRaises(ValueError):
            torch.set_float32_matmul_precision("turbo")


class TestFp32PrecisionRefusesWhatItCannotDeliver(Base):
    def test_the_supported_values_are_the_two_it_can_mean(self):
        self.assertEqual(set(_FP32_PRECISIONS), {"ieee", "tf32"})

    def test_bf16_and_none_are_refused_rather_than_silently_ignored(self):
        # torch accepts these; Jittor has no bf16-accumulate mode and no
        # per-op override, so accepting them would mean something else.
        for backend in (torch.backends.cuda.matmul,
                        torch.backends.cudnn.conv,
                        torch.backends.cudnn.rnn):
            for value in ("bf16", "none", "TURBO"):
                with self.subTest(backend=repr(backend), value=value):
                    with self.assertRaises(ValueError) as caught:
                        backend.fp32_precision = value
                    self.assertIn("ieee", str(caught.exception))
            with self.subTest(backend=repr(backend), value=7):
                with self.assertRaises(TypeError):
                    backend.fp32_precision = 7

    def test_it_is_case_insensitive_like_the_precision_setter(self):
        torch.backends.cuda.matmul.fp32_precision = "TF32"
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cuda.matmul.fp32_precision = "IEEE"
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

    def test_conv_and_rnn_are_distinct_objects_sharing_one_flag(self):
        # They used to share a *class attribute*, which is a different thing:
        # writing one shadowed it on that instance only and mapped to nothing.
        self.assertIsNot(torch.backends.cudnn.conv, torch.backends.cudnn.rnn)
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        self.assertEqual(torch.backends.cudnn.rnn.fp32_precision, "tf32")


class TestInstallDoesNotDisturbWhatItFinds(Base):
    def test_reinstalling_leaves_the_flags_where_they_were(self):
        # `cudnn.allow_tf32` is a view, so the install-time default assignment
        # writes back the value it just read. `benchmark` is the one setting
        # whose write hits the runtime, which is why it keeps its init gate.
        for enabled in (True, False):
            self._force("cudnn", enabled)
            before = torch.backends.cudnn.allow_tf32
            torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.allow_tf32
            with self.subTest(enabled=enabled):
                self.assertEqual(torch.backends.cudnn.allow_tf32, before)
                self.assertEqual(_tf32_get("cudnn"), enabled)


if __name__ == "__main__":
    unittest.main()
