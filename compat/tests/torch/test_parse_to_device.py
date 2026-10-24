"""Device parsing is host-only, including accelerator ordinals.

These tests prove argument routing, not an accelerator transfer. TensorDict's
real CPU-to-NPU movement is covered separately. Positional non_blocking and
invalid overload combinations remain separate, known signature gaps.
"""
import unittest

import numpy as np
import torch


class TestParseToDevice(unittest.TestCase):
    @staticmethod
    def parse(*args, **kwargs):
        return torch._C._nn._parse_to(*args, **kwargs)

    def assert_device(self, actual, expected):
        self.assertIsInstance(actual, torch.device)
        self.assertEqual((actual.type, actual.index), (expected.type, expected.index))

    def test_positional_integer_uses_current_accelerator_kind(self):
        for ordinal in (0, 2, np.int32(0), np.int64(2)):
            with self.subTest(ordinal=ordinal, kind=type(ordinal)):
                target, dtype, non_blocking, memory_format = self.parse(ordinal)
                self.assert_device(target, torch.device(ordinal))
                self.assertEqual(target.index, int(ordinal))
                self.assertNotEqual(target.type, "cpu")
                self.assertEqual((dtype, non_blocking, memory_format), (None, False, None))

    def test_keyword_integer_uses_same_device_resolution(self):
        for ordinal in (0, 3, np.int32(0), np.int64(3)):
            with self.subTest(ordinal=ordinal, kind=type(ordinal)):
                parsed = self.parse(device=ordinal)
                self.assert_device(parsed[0], torch.device(ordinal))
                self.assertEqual(parsed[1:], (None, False, None))

    def test_explicit_strings_and_device_objects_do_not_allocate(self):
        for spelling in ("cpu", "cuda", "cuda:2", "npu", "npu:2"):
            expected = torch.device(spelling)
            for argument in (spelling, expected):
                with self.subTest(argument=argument):
                    self.assert_device(self.parse(argument)[0], expected)
                    self.assert_device(self.parse(device=argument)[0], expected)

    def test_negative_ordinals_are_rejected(self):
        for ordinal in (-1, np.int32(-1), np.int64(-2)):
            with self.subTest(ordinal=ordinal, kind=type(ordinal)):
                with self.assertRaisesRegex(RuntimeError, "negative"):
                    self.parse(ordinal)
                with self.assertRaisesRegex(RuntimeError, "negative"):
                    self.parse(device=ordinal)

    def test_dtype_only_does_not_invent_a_device(self):
        for dtype in (torch.float32, torch.float16, torch.int64, torch.bool):
            with self.subTest(dtype=dtype):
                self.assertEqual(self.parse(dtype), (None, dtype, False, None))
                self.assertEqual(self.parse(dtype=dtype), (None, dtype, False, None))

    def test_device_dtype_keyword_flags_and_memory_format_are_retained(self):
        memory_format = torch.contiguous_format
        parsed = self.parse(0, torch.float32, non_blocking=True,
                            memory_format=memory_format)
        self.assert_device(parsed[0], torch.device(0))
        self.assertIs(parsed[1], torch.float32)
        self.assertIs(parsed[2], True)
        self.assertIs(parsed[3], memory_format)
        parsed = self.parse("cpu", dtype=torch.float16, non_blocking=False)
        self.assert_device(parsed[0], torch.device("cpu"))
        self.assertEqual(parsed[1:], (torch.float16, False, None))

    def test_boolean_keyword_device_is_not_an_ordinal(self):
        for value in (True, False):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.parse(device=value)
        for value in (True, False):
            self.assertEqual(self.parse(dtype=torch.float32, non_blocking=value),
                             (None, torch.float32, value, None))


if __name__ == "__main__":
    unittest.main()
