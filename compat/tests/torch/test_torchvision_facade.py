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


    def test_color_jitter_pil_and_tensor(self):
        from PIL import Image
        from torchvision.transforms import ColorJitter, Compose, ToTensor

        image = Image.new("RGB", (8, 6), (120, 60, 30))
        identity = ColorJitter(0, 0, 0, 0)
        self.assertIs(identity(image), image)

        darker = ColorJitter(brightness=(0.5, 0.5))(image)
        self.assertEqual(darker.size, image.size)
        self.assertEqual(darker.mode, image.mode)
        self.assertNotEqual(list(darker.getdata()), list(image.getdata()))

        warmer = ColorJitter(hue=(0.25, 0.25))(image)
        self.assertEqual(warmer.size, image.size)

        tensor = Compose([ColorJitter(0, 0, 0, 0), ToTensor()])(image)
        self.assertEqual(tuple(tensor.shape), (3, 6, 8))

        ones = torch.ones(3, 4, 5)
        halved = ColorJitter(brightness=(0.5, 0.5))(ones)
        np.testing.assert_allclose(halved.numpy(), np.full((3, 4, 5), 0.5), rtol=1e-6)

        with self.assertRaises(ValueError):
            ColorJitter(hue=0.7)


    def test_draw_bounding_boxes_and_box_convert(self):
        from PIL import Image
        from torchvision.ops import box_convert
        from torchvision.transforms.functional import pil_to_tensor
        from torchvision.utils import draw_bounding_boxes

        image = Image.new("RGB", (32, 24), (10, 20, 30))
        tensor = pil_to_tensor(image)
        boxes = torch.tensor([[4.0, 5.0, 10.0, 8.0]])
        converted = box_convert(boxes, "xywh", "xyxy")
        np.testing.assert_allclose(
            converted.numpy(), [[4.0, 5.0, 14.0, 13.0]], rtol=1e-6)
        drawn = draw_bounding_boxes(tensor, converted, colors="red", labels=["a"])
        self.assertEqual(tuple(drawn.shape), (3, 24, 32))
        self.assertIn("uint8", str(drawn.dtype))
        np.testing.assert_array_equal(
            drawn.numpy()[:, 5, 4], np.array([255, 0, 0], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
