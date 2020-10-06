# ***************************************************************
# Copyright (c) 2020 Jittor.
# Authors:
#     Xin Yao <yaox12@outlook.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import numbers
import random
import math
from PIL import Image
import numpy as np
import jittor as jt
import warnings
from collections.abc import Sequence, Mapping

from . import function_pil as F_pil

__all__ = ["hflip", "vflip", "adjust_brightness", "adjust_contrast", "adjust_saturation", "adjust_hue", "adjust_gamma",
           "crop", "resize", "to_grayscale", "center_crop", "crop_and_resize", "to_tensor", "image_normalize",
           "Crop", "RandomCropAndResize", "RandomHorizontalFlip", "CenterCrop", "ImageNormalize", "Compose",
           "Resize", "Gray", "RandomGray", "RandomCrop", "ToTensor", "Lambda", "RandomApply", "RandomOrder",
           "RandomChoice", "RandomVerticalFlip", "ColorJitter"]


def _get_image_size(img):
    """
    Return image size as (w, h)
    """
    return F_pil._get_image_size(img)

def _get_image_num_channels(img):
    return F_pil._get_image_num_channels(img)

def hflip(img):
    """
    Function for horizontally flipping the given image.

    Args::

        [in] img(PIL Image.Image): Input image.

    Example::
        
        img = Image.open(...)
        img_ = transform.hflip(img)
    """
    return F_pil.hflip(img)


def vflip(img):
    """
    Function for vertically flipping the given image.

    Args::

        [in] img(PIL Image.Image): Input image.

    Example::
        
        img = Image.open(...)
        img_ = transform.vflip(img)
    """
    return F_pil.vflip(img)


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
    return F_pil.adjust_brightness(img, brightness_factor)


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
    return F_pil.adjust_contrast(img, contrast_factor)


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
    return F_pil.adjust_saturation(img, saturation_factor)


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
    return F_pil.adjust_hue(img, hue_factor)


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
    return F_pil.adjust_gamma(img, gamma, gain)


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
    return F_pil.crop(img, top, left, height, width)


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
    return F_pil.resize(img, size, interpolation)


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
    return F_pil.to_grayscale(img, num_output_channels)


def center_crop(img, output_size):
    """
    Function for cropping the given image at the center.

    Args::

        [in] img(PIL Image.Image): Input image.
        [in] output_size (sequence or int): (height, width) of the crop box.
             If int or sequence with single int, it is used for both directions.

    Returns::

        PIL Image.Image: Cropped image.
    """

    output_size = _setup_size(output_size, error_msg="If size is a sequence, it should have 2 values")

    image_width, image_height = _get_image_size(img)
    crop_height, crop_width = output_size

    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return crop(img, crop_top, crop_left, crop_height, crop_width)


def crop_and_resize(img, top, left, height, width, size, interpolation=Image.BILINEAR):
    """
    Function for cropping and resizing image.

    Args::

        [in] img(PIL Image.Image): Input image.
        [in] top(int): the top boundary of the cropping box.
        [in] left(int): the left boundary of the cropping box.
        [in] height(int): height of the cropping box.
        [in] width(int): width of the cropping box.
        [in] size: resize size.
        [in] interpolation(int): type of resize. default: PIL.Image.BILINEAR

    Example::
        
        img = Image.open(...)
        img_ = transform.resize(img, 10，10，200，200，100)
    """
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img


def to_tensor(img):
    """
    Function for turning Image.Image to jt.array.

    Args::

        [in] img(PIL Image.Image): Input image.
    
    Example::
        
        img = Image.open(...)
        img_ = transform.to_tensor(img)
    """
    # todo: handle image with various modes
    if isinstance(img, Image.Image):
        return np.array(img).transpose((2, 0, 1)) / np.float32(255)
    return img


def to_pil_image(pic, mode=None):
    """Convert a jt.array or an np.ndarray to PIL Image.

    Args::

        [in] pic (jt.array or numpy.ndarray): Image to be converted to PIL Image.
        [in] mode (`PIL.Image mode`): color space and pixel depth of input data (optional).

    Returns::

        [out] PIL Image: Image converted to PIL Image.
    """
    if not(isinstance(pic, jt.array) or isinstance(pic, np.ndarray)):
        raise TypeError(f'pic should be Tensor or ndarray. Got {type(pic)}.')

    elif isinstance(pic, jt.array):
        if pic.ndim not in {2, 3}:
            raise ValueError(f'pic should be 2/3 dimensional. Got {pic.ndim} dimensions.')

        elif pic.ndim == 2:
            # if 2D image, convert to np.ndarray and add channel dimension (CHW)
            pic = np.expand_dims(pic.data, 2)

    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError(f'pic should be 2/3 dimensional. Got {pic.ndim} dimensions.')

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)

    npimg = pic
    if not isinstance(npimg, np.ndarray):
        raise TypeError(f'Input pic must be a torch.Tensor or NumPy ndarray, not {type(npimg)}.')

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError(f"Incorrect mode ({mode}) supplied for input type {np.dtype}. Should be {expected_mode}.")
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError(f"Only modes {permitted_2_channel_modes} are supported for 2D inputs")

        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(f"Only modes {permitted_4_channel_modes} are supported for 4D inputs")

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(f"Only modes {permitted_3_channel_modes} are supported for 3D inputs")
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def image_normalize(img, mean, std):
    """
    Function for normalizing image.

        Class for normalizing the input image.

    Args::

    [in] image(PIL Image.Image or np.ndarray): input image.
         If type of input image is np.ndarray, it should be in shape (C, H, W). 
    [in] mean(list): the mean value of Normalization.
    [in] std(list): the std value of Normalization.

    Example::
        img = Image.open(...)
        img_ = transform.image_normalize(img, mean=[0.5], std=[0.5])
    """
    if isinstance(img, Image.Image):
        img = (np.array(img).transpose((2, 0, 1)) \
               - mean * np.float32(255.)) \
               / (std * np.float32(255.))
    else:
        img = (img - mean) / std
    return img


class Crop:
    """Crop and the PIL Image to given size.

    Args:

        * top(int): top pixel indexes
        * left(int): left pixel indexes
        * height(int): image height
        * width(int): image width
    """
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        return crop(img, self.top, self.left, self.height, self.width)


class RandomCropAndResize:
    """Random crop and resize the given PIL Image to given size.

    Args::

        [in] size(int or tuple): width and height of the output image.
        [in] scale(tuple): range of scale ratio of the area.
        [in] ratio(tuple): range of aspect ratio.
        [in] interpolation: type of resize. default: PIL.Image.BILINEAR.

    Example::

        transform = transform.RandomCropAndResize(224)
        img_ = transform(img)
    """
    def __init__(self, size, scale:tuple=(0.08, 1.0), ratio:tuple=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = _setup_size(size, error_msg="If size is a sequence, it should have 2 values")
        assert scale[0] <= scale[1] and ratio[0] <= ratio[1]

        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, img:Image.Image):
        width, height = img.size
        scale = self.scale
        ratio = self.ratio
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                break
        else:
            # Fallback to central crop
            in_ratio = float(width) / float(height)
            if in_ratio < min(ratio):
                w = width
                h = int(round(w / min(ratio)))
            elif in_ratio > max(ratio):
                h = height
                w = int(round(h * max(ratio)))
            else:
                w = width
                h = height
            i = (height - h) // 2
            j = (width - w) // 2
        return crop_and_resize(img, i, j, h, w, self.size, self.interpolation)


class RandomHorizontalFlip:
    """
    Random flip the image horizontally.

    Args::

        [in] p(float): The probability of image flip, default: 0.5.

    Example::

        transform = transform.RandomHorizontalFlip(0.6)
        img_ = transform(img)
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() < self.p:
            return hflip(img)
        return img


class CenterCrop:
    '''
    Class for cropping image centrally.

    Args::

    [in] size(int or tuple): Size want to crop.

    Example::

        transform = transform.CenterCrop(224)
        img_ = transform(img)
    '''
    def __init__(self, size):
        self.size = _setup_size(size, error_msg="If size is a sequence, it should have 2 values")

    def __call__(self, img: Image.Image):
        return center_crop(img, self.size)


class ImageNormalize:
    '''
    Class for normalizing the input image.

    Args::

    [in] mean(list): the mean value of Normalization.
    [in] std(list): the std value of Normalization.

    Example::

        transform = transform.ImageNormalize(mean=[0.5], std=[0.5])
        img_ = transform(img)
    '''

    def __init__(self, mean, std):
        self.mean = np.float32(mean).reshape(-1, 1, 1)
        self.std = np.float32(std).reshape(-1, 1, 1)
        
    def __call__(self, img):
        return image_normalize(img, self.mean, self.std)


class Compose:
    '''
    Base class for combining various transformations.

    Args::

    [in] transforms(list): a list of transform.

    Example::

        transform = transform.Compose([
            transform.Resize(opt.img_size),
            transform.Gray(),
            transform.ImageNormalize(mean=[0.5], std=[0.5]),
        ])
        img_ = transform(img)
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class Resize:
    '''
    Class for resizing image.

    Args::

    [in] size(int or tuple): Size want to resize.
    [in] interpolation(int): type of resize.

    Example::

        transform = transform.Resize(224)
        img_ = transform(img)
    '''
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = _setup_size(size, error_msg="If size is a sequence, it should have 2 values")
        self.interpolation = interpolation

    def __call__(self, img: Image.Image):
        return resize(img, self.size, self.interpolation)


class Gray:
    '''
    Convert image to grayscale.

    Example::

        transform = transform.Gray()
        img_ = transform(img)
    '''
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img: Image.Image):
        return to_grayscale(img, self.num_output_channels)


class RandomGray:
    '''
    Randomly convert image to grayscale.

    Args::
        [in] p (float): probability that image should be converted to grayscale, default: 0.1

    Returns::

        [out] PIL Image: Grayscale version of the image with probability p and unchanged
              with probability (1-p).
              - If input image is 1 channel: grayscale version is 1 channel
              - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    Example::

        transform = transform.Gray()
        img_ = transform(img)
    '''
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img: Image.Image):
        num_output_channels = _get_image_num_channels(img)
        if random.random() < self.p:
            return to_grayscale(img, num_output_channels=num_output_channels)
        return img


class RandomCrop:
    '''
    Class for randomly cropping the input image.

    Args::

    [in] size(tuple or int): the size want to crop.

    Example::

        transform = transform.RandomCrop(128)
        img_ = transform(img)
    '''
    def __init__(self, size):
        self.size = _setup_size(size, error_msg="If size is a sequence, it should have 2 values")

    def __call__(self, img: Image.Image):
        width, height = _get_image_size(img)
        assert self.size[0] <= height and self.size[1] <= width, f"crop size exceeds the input image in RandomCrop"
        top = np.random.randint(0, height - self.size[0] + 1)
        left = np.random.randint(0, width - self.size[1] + 1)
        return crop(img, top, left, self.size[0], self.size[1])


class ToTensor:
    """
    Convert PIL Image to jt.array.
    """
    def __call__(self, img: Image.Image):
        return to_tensor(img)


class ToPILImage:
    """
    Converts a jt.array of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.

    Args::

        [in] mode (`PIL.Image mode`): color space and pixel depth of input data (optional).
             If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
              - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
              - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
              - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
              - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``,
                ``float``, ``short``).
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        """
        Args::
            [in] pic (jt.array or numpy.ndarray): Image to be converted to PIL Image.

        Returns:

            [out] PIL Image: Image converted to PIL Image.
        """
        return to_pil_image(pic, self.mode)


class Lambda:
    """
    Apply a user-defined lambda as a transform.

    Args::

        [in] lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError(f"Argument lambd should be callable, got {repr(type(lambd).__name__)}")
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class RandomApply:
    """
    Apply randomly a list of transformations with a given probability

    Args::
        [in] transforms (list or tuple): list of transformations
        [in] p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img


class RandomOrder:
    """
    Apply a list of transformations in a random order.

    Args::
        [in] transforms (list or tuple): list of transformations
        [in] p (float): probability
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](img)
        return img


class RandomChoice:
    """
    Apply single transformation randomly picked from a list.

    Args::
        [in] transforms (list or tuple): list of transformations
        [in] p (float): probability
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, img):
        t = random.choice(self.transforms)
        return t(img)


class RandomVerticalFlip:
    """
    Random flip the image vertically.

    Args::

        [in] p(float): The probability of image flip, default: 0.5.

    Example::

        transform = transform.RandomVerticalFlip(0.6)
        img_ = transform(img)
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() < self.p:
            return vflip(img)
        return img


class ColorJitter:
    """
    Randomly change the brightness, contrast, saturation and hue of an image.

    Args::

        [in] brightness (float or tuple of float (min, max)): How much to jitter brightness.
             brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
             or the given [min, max]. Should be non negative numbers.
        [in] contrast (float or tuple of float (min, max)): How much to jitter contrast.
             contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
             or the given [min, max]. Should be non negative numbers.
        [in] saturation (float or tuple of float (min, max)): How much to jitter saturation.
             saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
             or the given [min, max]. Should be non negative numbers.
        [in] hue (float or tuple of float (min, max)): How much to jitter hue.
             hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
             Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @staticmethod
    def _check_input(value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def _get_transform(brightness, contrast, saturation, hue):
        """
        Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns::
            Transform which randomly adjusts brightness, contrast, saturation
            and hue in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args::

            [in] img (PIL Image): Input image.

        Returns::

            [out] PIL Image: Color jittered image.
        """
        transform = self._get_transform(self.brightness, self.contrast, self.saturation, self.hue)

        return transform(img)


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size
