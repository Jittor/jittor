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
from PIL import Image
import random
import math
import numpy as np
import warnings
from collections.abc import Sequence, Mapping
import numbers
import jittor as jt

from . import function_pil as F_pil

def _get_image_size(img):
    """
    Return image size as (w, h)
    """
    return F_pil._get_image_size(img)

def _get_image_num_channels(img):
    return F_pil._get_image_num_channels(img)

def _is_numpy(img):
    return isinstance(img, np.ndarray)

def _is_numpy_image(img):
    return img.ndim in {2, 3}

def crop(img, top, left, height, width):
    '''
    Function for cropping image.

    Args::

        [in] img(Image.Image): Input image.
        [in] top(int): the top boundary of the cropping box.
        [in] left(int): the left boundary of the cropping box.
        [in] height(int): height of the cropping box.
        [in] width(int): width of the cropping box.

    Example::
        
        img = Image.open(...)
        img_ = transform.crop(img, 10, 10, 100, 100)
    '''
    return img.crop((left, top, left + width, top + height))

def resize(img, size, interpolation=Image.BILINEAR):
    '''
    Function for resizing image.

    Args::

        [in] img(Image.Image): Input image.
        [in] size: resize size. [h, w]
        [in] interpolation(int): type of resize. default: PIL.Image.BILINEAR

    Example::
        
        img = Image.open(...)
        img_ = transform.resize(img, (100, 100))
    '''
    if isinstance(size, Sequence):
        return img.resize(size[::-1], interpolation)
    else:
        w, h = img.size
        if (h > w):
            return img.resize((size, int(round(size * h / w))), interpolation)
        else:
            return img.resize((int(round(size * w / h)), size), interpolation)


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
    return F_pil.gray(img, num_output_channels)


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
    '''
    Function for cropping and resizing image.

    Args::

        [in] img(Image.Image): Input image.
        [in] top(int): the top boundary of the cropping box.
        [in] left(int): the left boundary of the cropping box.
        [in] height(int): height of the cropping box.
        [in] width(int): width of the cropping box.
        [in] size: resize size. [h, w]
        [in] interpolation(int): type of resize. default: PIL.Image.BILINEAR

    Example::
        
        img = Image.open(...)
        img_ = transform.resize(img, 10，10，200，200，100)
    '''
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
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
    def __call__(self, img:Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        return crop(img, self.top, self.left, self.height, self.width)


class RandomCropAndResize:
    """Random crop and resize the given PIL Image to given size.

    Args::

        [in] size(int or tuple): [height, width] of the output image.
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

        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, img:Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
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
        
    def __call__(self, img:Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
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
    
    def __call__(self, img:Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        width, height = img.size
        return crop(img, (height - self.size[0]) / 2, (width - self.size[1]) / 2, self.size[0], self.size[1])

def to_tensor(pic):
    """
    Function for turning Image.Image to np.array with CHW format.

    Args::

        [in] img(Image.Image): Input image.
    
    Example::
        
        img = Image.open(...)
        img_ = transform.to_tensor(img)
    """
    if isinstance(pic, jt.Var):
        return pic
    
    if isinstance(pic, tuple):
        # try convert ten crop tuple
        pic = ( to_tensor(pic) for p in pic )
        pic = np.array(pic)
        return pic
    if not(F_pil._is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError(f'img should be PIL Image or ndarray. Got {type(pic)}.')

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError(f'img should be 2/3 dimensional. Got {pic.ndim} dimensions.')

    if _is_numpy(pic):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[None, :, :]

        # backward compatibility
        if pic.dtype == 'uint8':
            return np.float32(pic) * np.float32(1/255.0)
        else:
            return pic

    # handle PIL Image
    if pic.mode == 'I':
        img = np.array(pic, np.int32, copy=False)
    elif pic.mode == 'I;16':
        img = np.array(pic, np.int16, copy=False)
    elif pic.mode == 'F':
        img = np.array(pic, np.float32, copy=False)
    elif pic.mode == '1':
        img = np.array(pic, np.uint8, copy=False) * 255
    else:
        img = np.array(pic, np.uint8, copy=False)

    # put it from HWC to CHW format
    img = img.reshape(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.transpose(2, 0, 1)
    if img.dtype == 'uint8':
        return np.float32(img) * np.float32(1/255.0)
    else:
        return img



def _to_jittor_array(pic):
    """
    Function for turning Image.Image or np.ndarray (HWC) to jt.Var (CHW).
    Args::
        [in] img(PIL Image.Image or np.ndarray): Input image.
             If input type is np.ndarray, the shape should be in HWC.
    Return:
        [out] jt.Var in shape CHW.
    Example::
        
        img = Image.open(...)
        img_ = transform.to_tensor(img)
    """
    if not(F_pil._is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError(f'img should be PIL Image or ndarray. Got {type(pic)}.')

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError(f'img should be 2/3 dimensional. Got {pic.ndim} dimensions.')

    if _is_numpy(pic):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = jt.array(pic.transpose((2, 0, 1)))
        # backward compatibility
        if img.dtype == 'uint8':
            return img.float().divide(255)
        else:
            return img

    # handle PIL Image
    if pic.mode == 'I':
        img = jt.array(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = jt.array(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = jt.array(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = jt.array(np.array(pic, np.uint8, copy=False) * 255, dtype='uint8')
    else:
        img = jt.array(np.array(pic, np.uint8, copy=False))

    # put it from HWC to CHW format
    img = img.reshape(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1))
    if img.dtype == 'uint8':
        return img.float().divide(255)
    else:
        return img

def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.
    Args:
        pic (Tensor or numpy.ndarray): Image(HWC format) to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if isinstance(pic, jt.Var):
        pic = pic.data
    if not isinstance(pic, np.ndarray):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    else:
        if pic.ndim not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)

    npimg = pic
    if 'float' in str(pic.dtype) and mode != 'F' and npimg.shape[2] != 1:
        npimg = np.uint8(pic * 255)
    # npimg = np.transpose(pic, (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a jt.Var or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

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
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError("Only modes {} are supported for 2D inputs".format(permitted_2_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)



def image_normalize(img, mean, std):
    """
    Function for normalizing image.
    Args::
    [in] image(PIL Image.Image or np.ndarray): input image.
         If type of input image is np.ndarray, it should be in shape (C, H, W). 
    [in] mean(list): the mean value of Normalization.
    [in] std(list): the std value of Normalization.
    Example::
        img = Image.open(...)
        img_ = transform.image_normalize(img, mean=[0.5], std=[0.5])
    """
    if not isinstance(img, (Image.Image, jt.Var, np.ndarray)):
        raise TypeError(f'Input type should be in (PIL Image, jt.Var, np.ndarray). Got {type(img)}.')
    elif isinstance(img, Image.Image):
        assert img.mode == 'RGB', f"input image mode should be 'RGB'. Got {img.mode}."
        img = (np.array(img).transpose((2, 0, 1)) \
               - mean * np.float32(255.)) \
               / (std * np.float32(255.))
    else:
        if img.ndim < 3:
            raise ValueError(f'Expected input to be a array image of size (..., C, H, W). Got {img.shape}.')
        if isinstance(img, jt.Var):
            mean = jt.array(mean)
            std = jt.array(std)
            if (std.data == 0).any():
                raise ValueError('std cannot be zero.')
        else:
            mean = np.asarray(mean)
            std = np.asarray(std)
            if (std == 0).any():
                raise ValueError('std cannot be zero.')
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1, 1)
        if std.ndim == 1:
            std = std.reshape(-1, 1, 1)
        img = (img - mean) / std
    return img



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
        self.mean = np.float32(mean).reshape(-1,1,1)
        self.std = np.float32(std).reshape(-1,1,1)
        
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = (np.array(img).transpose((2,0,1)) \
                - self.mean*np.float32(255.)) \
                * (np.float32(1./255.)/self.std)
        else:
            img = (img - self.mean) / self.std
        return img
    
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

    [in] size(int or tuple): Size want to resize. [h, w]
    [in] mode(int): type of resize.

    Example::

        transform = transform.Resize(224)
        img_ = transform(img)
    '''
    def __init__(self, size, mode=Image.BILINEAR):
        self.size = _setup_size(size, error_msg="If size is a sequence, it should have 2 values")
        self.mode = mode
    def __call__(self, img:Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        return resize(img, self.size, self.mode)

class Gray:
    '''
    Convert image to grayscale.

    Example::

        transform = transform.Gray()
        img_ = transform(img)
    '''
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img:Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        img = np.float32(img.convert('L')) / np.float32(255.0)
        if self.num_output_channels == 1:
            return img[np.newaxis, :]
        else:
            return np.dstack([img, img, img])

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

    def __call__(self, img:Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        num_output_channels = _get_image_num_channels(img)
        if random.random() < self.p:
            return gray(img, num_output_channels=num_output_channels)
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
    def __call__(self, img:Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        width, height = img.size
        assert self.size[0] <= height and self.size[1] <= width, f"crop size exceeds the input image in RandomCrop, {(self.size, height, width)}"
        top = np.random.randint(0,height-self.size[0]+1)
        left = np.random.randint(0,width-self.size[1]+1)
        return crop(img, top, left, self.size[0], self.size[1])
        
class Lambda:
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


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

    def __call__(self, img:Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
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

    def __call__(self, img:Image.Image):
        """
        Args::
            [in] img (PIL Image): Input image.
        Returns::
            [out] PIL Image: Color jittered image.
        """
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
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

class ToTensor:
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToPILImage(object):
    """Convert a tensor or an ndarray to PIL Image.
    Args:
        pic (Tensor or numpy.ndarray): Image(HWC format) to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    Returns:
        PIL Image: Image converted to PIL Image.
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        Returns:
            PIL Image: Image converted to PIL Image.
        """
        return to_pil_image(pic, self.mode)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string



class RandomPerspective(object):
    """Performs Perspective transformation of the given PIL Image randomly with a given probability.

    Args:
        interpolation : Default- Image.BICUBIC

        p (float): probability of the image being perspectively transformed. Default value is 0.5

        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.

    """

    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=Image.BICUBIC):
        self.p = p
        self.interpolation = interpolation
        self.distortion_scale = distortion_scale

    def __call__(self, img:Image.Image):
        """
        Args:
            img (PIL Image): Image to be Perspectively transformed.

        Returns:
            PIL Image: Random perspectivley transformed image.
        """
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)

        if random.random() < self.p:
            width, height = img.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            return F_pil.perspective(img, startpoints, endpoints, self.interpolation)
        return img

    @staticmethod
    def get_params(width, height, distortion_scale):
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width : width of the image.
            height : height of the image.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img:Image.Image):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F_pil.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = str(self.interpolation)
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


RandomSizedCrop = RandomResizedCrop


class FiveCrop(object):
    """Crop the given PIL Image into four corners and the central crop

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.

    Example:
         >>> transform = Compose([
         >>>    FiveCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img:Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        return F_pil.five_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class TenCrop(object):
    """Crop the given PIL Image into four corners and the central crop plus the flipped version of
    these (horizontal flipping is used by default)

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        vertical_flip (bool): Use vertical flipping instead of horizontal

    Example:
         >>> transform = Compose([
         >>>    TenCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img:Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        return F_pil.ten_crop(img, self.size, self.vertical_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, vertical_flip={1})'.format(self.size, self.vertical_flip)


class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
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

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img:Image.Image):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        angle = self.get_params(self.degrees)

        return F_pil.rotate(img, angle, self.resample, self.expand, self.center, self.fill)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.(Pillow>=5.0.0)

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img:Image.Image):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        return F_pil.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = str(d['resample'])
        return s.format(name=self.__class__.__name__, **d)
