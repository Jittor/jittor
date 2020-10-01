# ***************************************************************
# Copyright (c) 2020 Jittor.
# Authors:
#     Xin Yao <yaox12@outlook.com>
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

from typing import Sequence
from PIL import Image, ImageOps, ImageEnhance
import numpy as np


def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _get_image_size(img):
    if _is_pil_image(img):
        return img.size
    raise TypeError(f"Unexpected type {img}")

def hflip(img):
    """
    Function for horizontally flipping the given image.

    Args::

        [in] img(PIL Image.Image): Input image.

    Example::
        
        img = Image.open(...)
        img_ = transform.hflip(img)
    """
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')

    return img.transpose(Image.FLIP_LEFT_RIGHT)


def vflip(img):
    """
    Function for vertically flipping the given image.

    Args::

        [in] img(PIL Image.Image): Input image.

    Example::
        
        img = Image.open(...)
        img_ = transform.vflip(img)
    """
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')

    return img.transpose(Image.FLIP_TOP_BOTTOM)


def adjust_brightness(img, brightness_factor):
    """
    Function for adjusting brightness of an RGB image.

    Args::

        [in] img (PIL Image.Image): Image to be adjusted.
        [in] brightness_factor (float):  How much to adjust the brightness.
             Can be any non negative number. 0 gives a black image, 1 gives the
             original image while 2 increases the brightness by a factor of 2.

    Returns::
        [out] PIL Image.Image: Brightness adjusted image.

    Example::
        
        img = Image.open(...)
        img_ = transform.adjust_brightness(img, 0.5)
    """
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """
    Function for adjusting contrast of an image.

    Args::

        [in] img (PIL Image.Image): Image to be adjusted.
        [in] contrast_factor (float): How much to adjust the contrast.
             Can be any non negative number. 0 gives a solid gray image,
             1 gives the original image while 2 increases the contrast by a factor of 2.

    Returns::
        [out] PIL Image.Image: Contrast adjusted image.

    Example::
        
        img = Image.open(...)
        img_ = transform.adjust_contrast(img, 0.5)
    """
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """
    Function for adjusting saturation of an image.

    Args::

        [in] img (PIL Image.Image): Image to be adjusted.
        [in] saturation_factor (float):  How much to adjust the saturation.
             0 will give a black and white image, 1 will give the original image
             while 2 will enhance the saturation by a factor of 2.

    Returns::
        [out] PIL Image.Image: Saturation adjusted image.

    Example::
        
        img = Image.open(...)
        img_ = transform.adjust_saturation(img, 0.5)
    """
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """
    Function for adjusting hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args::

        [in] img (PIL Image.Image): Image to be adjusted.
        [in] hue_factor (float):  How much to shift the hue channel.
             Should be in [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of
             hue channel in HSV space in positive and negative direction respectively.
             0 means no shift. Therefore, both -0.5 and 0.5 will give an image
             with complementary colors while 0 gives the original image.

    Returns::
        [out] PIL Image.Image: Saturation adjusted image.

    Example::
        
        img = Image.open(...)
        img_ = transform.adjust_hue(img, 0.1)
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError(f'hue_factor ({hue_factor}) is not in [-0.5, 0.5].')

    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def adjust_gamma(img, gamma, gain=1):
    """
    Function for performing gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}

    See `Gamma Correction`_ for more details.

    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction

    Args::

        [in] img (PIL Image.Image): Image to be adjusted.
        [in] gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
             gamma larger than 1 make the shadows darker,
             while gamma smaller than 1 make dark regions lighter.
        [in] gain (float): The constant multiplier.

    Returns::
        [out] PIL Image.Image: Gamma adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')
    gamma_map = [(255 + 1 - 1e-3) * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    img = img.convert(input_mode)
    return img


def crop(img, top, left, height, width):
    """
    Function for cropping image.

    Args::

        [in] img(PIL Image.Image): Input image.
        [in] top(int): the top boundary of the cropping box.
        [in] left(int): the left boundary of the cropping box.
        [in] height(int): height of the cropping box.
        [in] width(int): width of the cropping box.

    Returns::
        [out] PIL Image.Image: Cropped image.

    Example::
        
        img = Image.open(...)
        img_ = transform.crop(img, 10, 10, 100, 100)
    """
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')

    return img.crop((left, top, left + width, top + height))


def resize(img, size, interpolation=Image.BILINEAR):
    """
    Function for resizing the input image to the given size.

    Args::

        [in] img(PIL Image.Image): Input image.
        [in] size(sequence or int): Desired output size. If size is a sequence like
             (h, w), the output size will be matched to this. If size is an int,
             the smaller edge of the image will be matched to this number maintaining
             the aspect ratio. If a tuple or list of length 1 is provided, it is
             interpreted as a single int.
        [in] interpolation(int, optional): type of interpolation. default: PIL.Image.BILINEAR

    Returns::
        [out] PIL Image.Image: Resized image.

    Example::
        
        img = Image.open(...)
        img_ = transform.resize(img, (100, 100))
    """
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')
    if not (isinstance(size, int) or (isinstance(size, Sequence) and len(size) in (1, 2))):
        raise TypeError(f'Got inappropriate size arg: {size}')

    if isinstance(size, int) or len(size) == 1:
        if isinstance(size, Sequence):
            size = size[0]
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def to_grayscale(img, num_output_channels):
    """
    Function for converting PIL image of any mode (RGB, HSV, LAB, etc) to grayscale version of image.

    Args::

        [in] img(PIL Image.Image): Input image.
        [in] num_output_channels (int): number of channels of the output image. Value can be 1 or 3. Default, 1.

    Returns::
        [out] PIL Image: Grayscale version of the image.
              if num_output_channels = 1 : returned image is single channel
              if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img
