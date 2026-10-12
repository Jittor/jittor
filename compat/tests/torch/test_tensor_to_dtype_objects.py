"""`.to()` has to accept a torch dtype object, not only a string.

torch spells a dtype as an object. Code written against torch hands it straight
to `.to()`, and `transformers/image_processing_backends.py:327` does exactly
that for every image it preprocesses::

    images = self.normalize(images.to(dtype=torch.float32), image_mean, image_std)

`_dtype_spec` accepted only `NanoString`, callables and strings, so both the
positional and the keyword form raised, and the failure surfaced from inside
transformers as `TypeError: to() expected dtype to be a dtype spelling`. Found
by running MiniMax-H3 reference-to-video, whose Qwen2-VL image processor takes
that path for each reference image.
"""
import unittest

import jittor as jt
import torch


class TestToAcceptsTorchDtypes(unittest.TestCase):
    def setUp(self):
        self.x = jt.ones((2, 2))

    def test_positional_dtype_object(self):
        self.assertEqual(str(self.x.to(torch.float32).dtype), "float32")

    def test_keyword_dtype_object(self):
        self.assertEqual(str(self.x.to(dtype=torch.float32).dtype), "float32")

    def test_half_precision_dtype_objects(self):
        for spelling, expected in ((torch.float16, "float16"),
                                   (torch.bfloat16, "bfloat16")):
            with self.subTest(str(spelling)):
                self.assertEqual(str(self.x.to(dtype=spelling).dtype), expected)

    def test_integer_dtype_objects(self):
        self.assertEqual(str(self.x.to(dtype=torch.int64).dtype), "int64")

    def test_string_spelling_still_works(self):
        self.assertEqual(str(self.x.to(dtype="float32").dtype), "float32")

    def test_a_device_object_is_still_a_device(self):
        # `torch.device` has no `.name`, which is what keeps the dtype branch
        # from swallowing it; assert that rather than trusting it.
        moved = self.x.to(torch.device("cpu"))
        self.assertEqual(str(moved.dtype), "float32")

    def test_a_mistyped_device_is_still_rejected(self):
        with self.assertRaises(TypeError):
            self.x.to("gpu")

    def test_dtype_and_device_together(self):
        out = self.x.to(device="cpu", dtype=torch.float16)
        self.assertEqual(str(out.dtype), "float16")


if __name__ == "__main__":
    unittest.main()
