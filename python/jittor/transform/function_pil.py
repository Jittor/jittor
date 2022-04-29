# ***************************************************************
# Copyright (c) 2022 Jittor.
# All Rights Reserved. 
# Maintainers:
#     Dun Liang <randonlang@gmail.com>. 
#
# Contributors:
#     Xin Yao <yaox12@outlook.com>
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
from typing import Sequence
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
import numpy as np
import numbers
import math
from math import cos, sin, tan


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _get_image_size(img):
    if _is_pil_image(img):
        return img.size
    raise TypeError(f"Unexpected type {type(img)}")


def _get_image_num_channels(img):
    if _is_pil_image(img):
        return 1 if img.mode == 'L' else 3
    raise TypeError(f"Unexpected type {type(img)}")


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


def gray(img, num_output_channels):
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

def _get_perspective_coeffs(startpoints, endpoints):
    """Helper function to get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.

    In Perspective Transform each pixel (x, y) in the orignal image gets transformed as,
     (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )

    Args:
        List containing [top-left, top-right, bottom-right, bottom-left] of the orignal image,
        List containing [top-left, top-right, bottom-right, bottom-left] of the transformed
                   image
    Returns:
        octuple (a, b, c, d, e, f, g, h) for transforming each pixel.
    """
    matrix = []

    for p1, p2 in zip(endpoints, startpoints):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.array(matrix, dtype="float")
    B = np.array(startpoints, dtype="float").reshape(8)
    res = np.linalg.lstsq(A, B, rcond=-1)[0]
    return res.tolist()


def perspective(img, startpoints, endpoints, interpolation=Image.BICUBIC):
    """Perform perspective transform of the given PIL Image.

    Args:
        img (PIL Image): Image to be transformed.
        startpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the orignal image
        endpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image
        interpolation: Default- Image.BICUBIC
    Returns:
        PIL Image:  Perspectively transformed Image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    coeffs = _get_perspective_coeffs(startpoints, endpoints)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, interpolation)


def resized_crop(img, top, left, height, width, size, interpolation=Image.BILINEAR):
    """Crop the given PIL Image and resize it to desired size.

    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
    Returns:
        PIL Image: Cropped image.
    """
    assert _is_pil_image(img), 'img should be PIL Image'
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img

def center_crop(img, output_size):
    """Crop the given PIL Image and resize it to desired size.

        Args:
            img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
        Returns:
            PIL Image: Cropped image.
        """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return crop(img, crop_top, crop_left, crop_height, crop_width)

def five_crop(img, size):
    """Crop the given PIL Image into four corners and the central crop.

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.

    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    image_width, image_height = img.size
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = img.crop((0, 0, crop_width, crop_height))
    tr = img.crop((image_width - crop_width, 0, image_width, crop_height))
    bl = img.crop((0, image_height - crop_height, crop_width, image_height))
    br = img.crop((image_width - crop_width, image_height - crop_height,
                   image_width, image_height))
    center = center_crop(img, (crop_height, crop_width))
    return (tl, tr, bl, br, center)

def ten_crop(img, size, vertical_flip=False):
    r"""Crop the given PIL Image into four corners and the central crop plus the
        flipped version of these (horizontal flipping is used by default).

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
       vertical_flip (bool): Use vertical flipping instead of horizontal

    Returns:
       tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
                Corresponding top left, top right, bottom left, bottom right and center crop
                and same for the flipped image.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)
    return first_five + second_five


def rotate(img, angle, resample=False, expand=False, center=None, fill=None):
    """Rotate the image by angle.


    Args:
        img (PIL Image): PIL Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands. This option is only available for ``pillow>=5.2.0``.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """
    def parse_fill(fill, num_bands):
        if PILLOW_VERSION < "5.2.0":
            if fill is None:
                return {}
            else:
                msg = ("The option to fill background area of the rotated image, "
                       "requires pillow>=5.2.0")
                raise RuntimeError(msg)

        if fill is None:
            fill = 0
        if isinstance(fill, (int, float)) and num_bands > 1:
            fill = tuple([fill] * num_bands)
        if not isinstance(fill, (int, float)) and len(fill) != num_bands:
            msg = ("The number of elements in 'fill' does not match the number of "
                   "bands of the image ({} != {})")
            raise ValueError(msg.format(len(fill), num_bands))

        return {"fillcolor": fill}

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    opts = parse_fill(fill, len(img.getbands()))

    return img.rotate(angle, resample, expand, center, **opts)


def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    #
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    if isinstance(shear, numbers.Number):
        shear = [shear, 0]

    if not isinstance(shear, (tuple, list)) and len(shear) == 2:
        raise ValueError(
            "Shear should be a single value or a tuple/list containing " +
            "two values. Got {}".format(shear))

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = cos(rot - sy) / cos(sy)
    b = -cos(rot - sy) * tan(sx) / cos(sy) - sin(rot)
    c = sin(rot - sy) / cos(sy)
    d = -sin(rot - sy) * tan(sx) / cos(sy) + cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    M = [d, -b, 0,
         -c, a, 0]
    M = [x / scale for x in M]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
    M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    M[2] += cx
    M[5] += cy
    return M


def affine(img, angle, translate, scale, shear, resample=0, fillcolor=None):
    """Apply affine transformation on the image keeping image center invariant

    Args:
        img (PIL Image): PIL Image to be rotated.
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float or tuple or list): shear angle value in degrees between -180 to 180, clockwise direction.
        If a tuple of list is specified, the first value corresponds to a shear parallel to the x axis, while
        the second value corresponds to a shear parallel to the y axis.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter.
            See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] >= '5' else {}
    return img.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)

