"""torchvision facade's RandomResizedCrop is callable and honours its size."""

from __future__ import print_function

import unittest

from PIL import Image

from torchvision.transforms import Compose, RandomResizedCrop, ToTensor


class TestRandomResizedCrop(unittest.TestCase):
    def test_pil_image_is_cropped_and_resized(self):
        image = Image.new('RGB', (64, 48), (10, 20, 30))
        transform = Compose([RandomResizedCrop((24, 32)), ToTensor()])
        tensor = transform(image)
        self.assertEqual(tuple(tensor.shape), (3, 24, 32))

    def test_int_size_produces_a_square(self):
        image = Image.new('RGB', (40, 60), (0, 0, 0))
        tensor = RandomResizedCrop(16)(image)
        self.assertEqual(tensor.size, (16, 16))


if __name__ == '__main__':
    unittest.main()
