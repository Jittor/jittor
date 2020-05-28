# ***************************************************************
# Copyright (c) 2020
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
from PIL import Image
import random
import math
import numpy as np
import warnings
from collections.abc import Sequence, Mapping

def crop(img, top, left, height, width):
    return img.crop((left, top, left + width, top + height))

def resize(img, size, interpolation=Image.BILINEAR):
    return img.resize(size[::-1], interpolation)

def crop_and_resize(img, top, left, height, width, size, interpolation=Image.BILINEAR):
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img

class RandomCropAndResize:
    """Random crop and resize the given PIL Image to given size.

    Args:

        * size(int or tuple): width and height of the output image
        * scale(tuple): range of scale ratio of the area
        * ratio(tuple): range of aspect ratio
        * interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, scale:tuple=(0.08, 1.0), ratio:tuple=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, int):
            size = (size, size)
        assert isinstance(size, tuple)
        assert scale[0] <= scale[1] and ratio[0] <= ratio[1]

        self.size = size
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
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img:Image.Image):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
    
def to_tensor(img):
    if isinstance(img, Image.Image):
        return np.array(img).transpose((2,0,1)) / np.float32(255)
    return img

class ImageNormalize:
    def __init__(self, mean, std):
        self.mean = np.float32(mean).reshape(-1,1,1)
        self.std = np.float32(std).reshape(-1,1,1)
        
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = (np.array(img).transpose((2,0,1)) \
                - self.mean*np.float32(255.)) \
                / (self.std*np.float32(255.))
        else:
            img = (img - self.mean) / self.std
        return img
    
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

class Resize:
    def __init__(self, size, mode=Image.BILINEAR):
        if isinstance(size, int):
            size = (size, size)
        assert isinstance(size, tuple)
        self.size = size
        self.mode = mode
    def __call__(self, img:Image.Image):
        return img.resize(self.size, self.mode)

class Gray:
    def __call__(self, img:Image.Image):
        img = np.array(img.convert('L'))
        img = img[np.newaxis, :]
        return np.array((img / 255.0), dtype = np.float32)

class RandomCrop:
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        assert isinstance(size, tuple)
        self.size = size
    def __call__(self, img:Image.Image):
        width, height = img.size
        assert self.size[0] <= height and self.size[1] <= width, f"crop size exceeds the input image in RandomCrop"
        top = np.random.randint(0,height-self.size[0]+1)
        left = np.random.randint(0,width-self.size[1]+1)
        return crop(img, top, left, self.size[0], self.size[1])
        