"""Numerical contracts for the deployed torchvision transform facade."""

import unittest

import numpy as np
import torch
import jittor as jt
from torchvision.transforms.v2 import functional as F

from _helpers import capability as _test_capability


_DEVICES = [("cpu", 0)] + (
    [("cuda", 1)] if _test_capability.check_accelerator("cuda", backend=jt).enabled else []
)


class TestTorchvisionFacade(unittest.TestCase):

    def test_color_crop_and_multidimensional_pad(self):
        for name, use_cuda in _DEVICES:
            with self.subTest(device=name), jt.flag_scope(use_cuda=use_cuda):
                gray = jt.array(np.arange(20, dtype=np.uint8).reshape(1, 4, 5))
                rgb = F.grayscale_to_rgb(gray)
                self.assertEqual(tuple(rgb.shape), (3, 4, 5))

                cropped = F.center_crop(rgb, (2, 3))
                self.assertEqual(tuple(cropped.shape), (3, 2, 3))

                video = rgb.unsqueeze(0)
                batched_video = F.resize(video.unsqueeze(0), (2, 3))
                self.assertEqual(tuple(batched_video.shape), (1, 1, 3, 2, 3))
                padded = F.pad(video, [0, 2, 0, 1, 0, 0, 0, 1])
                self.assertEqual(tuple(padded.shape), (2, 3, 5, 7))

                if name == "cuda":
                    jt.sync([rgb, batched_video, padded])
                    self.assertEqual(rgb.location(), "device")
                    self.assertEqual(batched_video.location(), "device")
                    self.assertEqual(padded.location(), "device")

                rgb_array = rgb.numpy()
                np.testing.assert_array_equal(
                    rgb_array, np.repeat(gray.numpy(), 3, axis=0))
                np.testing.assert_array_equal(cropped.numpy(), rgb_array[:, 1:3, 1:4])
                np.testing.assert_array_equal(padded[0, :, :4, :5].numpy(), rgb_array)
                self.assertEqual(float(padded[1].sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
