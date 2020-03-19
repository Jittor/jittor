# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import numpy as np
from urllib import request
import gzip
import pickle
import os
from jittor.dataset.utils import get_random_list, get_order_list, collate_batch
import pathlib
from PIL import Image

dataset_root = os.path.join(pathlib.Path.home(), ".cache", "jittor", "dataset")

class Dataset(object):
    '''
    base class for reading data
    
    Example:
        class YourDataset(Dataset):
            def __init__(self):
                super().__init__()
                self.set_attrs(total_len=1024)

            def __getitem__(self, k):
                return k, k*k

        dataset = YourDataset().set_attrs(batch_size=256, shuffle=True)
        for x, y in dataset:
            ......
    '''
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.total_len = None
        self.shuffle = False
        self.drop_last = False

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        assert self.total_len >= 0
        assert self.batch_size > 0
        return (self.total_len-1) // self.batch_size + 1

    def set_attrs(self, **kw):
        '''set attributes of dataset, equivalent to setattr
        
        Attrs:
            batch_size(int): batch size, default 16.
            totol_len(int): totol lenght.
            shuffle(bool): shuffle at each epoch, default False.
            drop_last(bool): if true, the last batch of dataset
                might smaller than batch_size, default True.
        '''
        for k,v in kw.items():
            assert hasattr(self, k), k
            setattr(self, k, v)
        return self

    def collate_batch(self, batch):
        return collate_batch(batch)

    def __iter__(self):
        if self.shuffle == False:
            index_list = get_order_list(self.total_len)
        else:
            index_list = get_random_list(self.total_len)
        batch_data = []
        for idx in index_list:
            batch_data.append(self[int(idx)])
            if len(batch_data) == self.batch_size:
                batch_data = self.collate_batch(batch_data)
                yield batch_data
                batch_data = []

        # depend on drop_last
        if not self.drop_last and len(batch_data) > 0:
            batch_data = self.collate_batch(batch_data)
            yield batch_data

class ImageFolder(Dataset):
    """A image classify dataset, load image and label from directory:
    
        root/label1/img1.png
        root/label1/img2.png
        ...
        root/label2/img1.png
        root/label2/img2.png
        ...
    Args:
        root(string): Root directory path.

     Attributes:
        classes(list): List of the class names.
        class_to_idx(dict): map from class_name to class_index.
        imgs(list): List of (image_path, class_index) tuples
    """
    def __init__(self, root, transform=None):
        # import ipdb; ipdb.set_trace()
        super().__init__()
        self.root = root
        self.transform = transform
        self.classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])
        self.class_to_idx = {v:k for k,v in enumerate(self.classes)}
        self.imgs = []
        image_exts = set(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
        
        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root, class_name)
            for dname, _, fnames in sorted(os.walk(class_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if os.path.splitext(fname)[-1].lower() in image_exts:
                        path = os.path.join(class_dir, fname)
                        self.imgs.append((path, i))
        print(f"Found {len(self.classes)} classes and {len(self.imgs)} images.")
        self.set_attrs(total_len=len(self.imgs))
        
    def __getitem__(self, k):
        with open(self.imgs[k][0], 'rb') as f:
            img = Image.open(f).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.imgs[k][1]
