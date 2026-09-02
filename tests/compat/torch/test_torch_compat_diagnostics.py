"""What the compatibility layer continued past has to be answerable afterwards.

The layer's handlers used to be ``except Exception: pass``. A marker that failed
to propagate, a dtype that failed to restore, an optimizer grad map that failed
to rebuild -- each of those left nothing behind, so the only symptom was that
the numbers came out different. These tests pin the replacement: the record
exists, it says what was being attempted, it is reachable from the torch
surface, and turning the debug switch on prints it with a traceback.
"""
import io
import unittest
from contextlib import redirect_stderr

import jittor as torch

from jittor.compat import diagnostics


class Base(unittest.TestCase):
    def setUp(self):
        diagnostics.clear()
        self._debug = diagnostics.set_debug(False)

    def tearDown(self):
        diagnostics.set_debug(self._debug)
        diagnostics.clear()

    @staticmethod
    def _record(what, exception=None, hint=None):
        try:
            raise (exception or AttributeError("Var refuses this attribute"))
        except Exception as exc:              # noqa: BLE001 - the test *is* the raiser
            return diagnostics.swallowed(what, exc, hint)


class TestTheRecord(Base):
    def test_it_says_what_was_attempted_and_what_failed(self):
        record = self._record("types.py _make_cpu_resident: set _jittor_torch_force_cpu",
                              hint="residency will follow the global flag instead")
        self.assertEqual(record.exception, "AttributeError")
        self.assertIn("Var refuses this attribute", record.message)
        self.assertIn("_make_cpu_resident", repr(record))
        self.assertIn("residency will follow", repr(record))

    def test_records_are_ordered_and_filterable_by_label(self):
        self._record("types.py restore dtype")
        self._record("nn.py register leaf")
        self._record("types.py restore residency")
        self.assertEqual(len(diagnostics.records()), 3)
        self.assertEqual([r.what for r in diagnostics.records("types.py")],
                         ["types.py restore dtype", "types.py restore residency"])

    def test_counts_survive_the_bounded_record_buffer(self):
        # A training loop that swallows something every step must not grow
        # without limit, and must not lose the fact that it happened either.
        for _ in range(diagnostics.LIMIT + 50):
            self._record("hot loop: propagate marker")
        self.assertEqual(len(diagnostics.records()), diagnostics.LIMIT)
        self.assertEqual(
            diagnostics.counts()[("hot loop: propagate marker", "AttributeError")],
            diagnostics.LIMIT + 50)

    def test_no_traceback_is_formatted_unless_debug_is_on(self):
        # Several of these handlers sit in per-op code; formatting a traceback
        # for every one would be a real cost.
        self.assertIsNone(self._record("quiet").stack)

    def test_debug_prints_the_record_and_its_traceback(self):
        diagnostics.set_debug(True)
        stream = io.StringIO()
        with redirect_stderr(stream):
            record = self._record("loud: rebuild grad map")
        printed = stream.getvalue()
        self.assertIn("loud: rebuild grad map", printed)
        self.assertIn("AttributeError", printed)
        self.assertIn("Traceback", printed)
        self.assertIsNotNone(record.stack)


class TestItIsReachableFromTheTorchSurface(Base):
    def test_torch_exposes_the_query_entry_points(self):
        for name in ("compat_swallowed", "compat_swallowed_counts", "compat_debug"):
            with self.subTest(api=name):
                self.assertTrue(callable(getattr(torch, name, None)))

    def test_what_the_layer_swallows_shows_up_there(self):
        self._record("probe: something the layer continued past")
        self.assertIn("probe: something the layer continued past",
                      [r.what for r in torch.compat_swallowed()])
        self.assertIn("probe: something the layer continued past",
                      [r.what for r in torch.compat_swallowed("probe:")])


class TestARealHandlerRecords(Base):
    def test_a_shape_that_is_not_a_number_is_recorded_not_ignored(self):
        # jittor/compat/fsdp2/common.py::_prod multiplies a shape together and
        # used to `except Exception: pass` on anything that would not convert,
        # silently producing a product computed from a subset of the dimensions.
        from jittor.compat.fsdp2 import common

        self.assertEqual(common._prod([2, 3]), 6)
        self.assertEqual(common._prod([2, object(), 3]), 6)   # same value as before
        records = diagnostics.records("fsdp2/common.py")
        self.assertTrue(records, "the skipped dimension must leave a record")
        self.assertEqual(records[-1].exception, "TypeError")


if __name__ == "__main__":
    unittest.main()
