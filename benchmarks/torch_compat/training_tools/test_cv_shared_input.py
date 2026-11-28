import random

import numpy as np
from PIL import Image
import pytest

from .cv_shared_input import crop_box, image_array


@pytest.mark.parametrize("training", [True, False])
def test_shared_image_bytes_preserve_the_original_crop_and_fp32_pixels(tmp_path, training):
    rng = np.random.RandomState(13)
    rgb = rng.randint(0, 256, (269, 291, 3)).astype(np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(rgb).save(path)
    sample, epoch, seed = 5, 2, 42

    reference = Image.fromarray(rgb)
    if training:
        top, left, height, width = crop_box(*rgb.shape[:2], random.Random(f"{seed}:{epoch}:{sample}"))
        reference = reference.crop((left, top, left + width, top + height))
    reference = reference.resize((224, 224), Image.Resampling.BILINEAR)
    expected = np.asarray(reference, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)

    pixels = image_array(path, sample, epoch, seed, training)
    assert pixels.dtype == np.uint8
    assert pixels.flags.c_contiguous
    assert pixels.nbytes == expected.nbytes // 4
    np.testing.assert_array_equal(pixels.astype(np.float32) / np.float32(255), expected)
