# ***************************************************************
# Copyright(c) 2019
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import numpy as np
import os
from PIL import Image
from .dataset import Dataset, dataset_root

class VOC(Dataset):
    '''
    Jittor's own class for loading VOC dataset.

    Args::

        [in] data_root(str): your data root.
        [in] split(str): which split you want to use, train or val.
    
    Attribute::

        NUM_CLASSES: Number of total categories, default is 21.

    Example::

        from jittor.dataset.voc import VOC
        train_loader = VOC(data_root='...').set_attrs(batch_size=16, shuffle=True)
        for i, (imgs, target) in enumerate(train_loader):
            ...
    '''
    NUM_CLASSES = 21
    def __init__(self, data_root=dataset_root+'/voc/', split='train'):
        super().__init__()
        ''' total_len , batch_size, shuffle must be set '''
        self.data_root = data_root
        self.split = split

        self.image_root = os.path.join(data_root, 'JPEGImages')
        self.label_root = os.path.join(data_root, 'SegmentationClass')

        self.data_list_path = os.path.join(self.data_root, 'ImageSets', 'Segmentation', self.split + '.txt')
        self.image_path = []
        self.label_path = []

        with open(self.data_list_path, "r") as f:
            lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            _img_path = os.path.join(self.image_root, line + '.jpg')
            _label_path = os.path.join(self.label_root, line + '.png')
            assert os.path.isfile(_img_path)
            assert os.path.isfile(_label_path)
            self.image_path.append(_img_path)
            self.label_path.append(_label_path)
        self.set_attrs(total_len = len(self.image_path))

    def __getitem__(self, index):
        _img = Image.open(self.image_path[index])
        _label = Image.open(self.label_path[index])
        _img = _img.resize((513, 513))
        _label = _label.resize((513, 513))
        _img = np.array(_img)
        _label = np.array(_label)
        _img = _img.transpose(2,0,1)
        return _img, _label

