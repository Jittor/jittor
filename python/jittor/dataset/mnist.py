# ***************************************************************
# Copyright(c) 2019
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import os
import string
import numpy as np
import gzip
from PIL import Image
# our lib jittor import
from jittor.dataset.dataset import Dataset, dataset_root
from jittor_utils.misc import ensure_dir, download_url_to_local
import jittor as jt 
import jittor.transform as trans

class MNIST(Dataset):
    '''
    Jittor's own class for loading MNIST dataset.

    Args::

        [in] data_root(str): your data root.
        [in] train(bool): choose model train or val.
        [in] download(bool): Download data automatically if download is Ture.
        [in] batch_size(int): Data batch size.
        [in] shuffle(bool): Shuffle data if true.
        [in] transform(jittor.transform): transform data.

    Example::

        from jittor.dataset.mnist import MNIST
        train_loader = MNIST(train=True).set_attrs(batch_size=16, shuffle=True)
        for i, (imgs, target) in enumerate(train_loader):
            ...
    '''
    def __init__(self, data_root=dataset_root+"/mnist_data/", 
                 train=True, 
                 download=True, 
                 batch_size = 16,
                 shuffle = False,
                 transform=None):
        # if you want to test resnet etc you should set input_channel = 3, because the net set 3 as the input dimensions
        super().__init__()
        self.data_root = data_root
        self.is_train = train
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        if download == True:
            self.download_url()

        filesname = [
                "train-images-idx3-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz"
        ]
        self.mnist = {}
        if self.is_train:
            with gzip.open(data_root + filesname[0], 'rb') as f:
                self.mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28)
            with gzip.open(data_root + filesname[2], 'rb') as f:
                self.mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
        else:
            with gzip.open(data_root + filesname[1], 'rb') as f:
                self.mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28)
            with gzip.open(data_root + filesname[3], 'rb') as f:
                self.mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
        assert(self.mnist["images"].shape[0] == self.mnist["labels"].shape[0])
        self.total_len = self.mnist["images"].shape[0]
        # this function must be called
        self.set_attrs(total_len = self.total_len)

    def __getitem__(self, index):
        img = Image.fromarray(self.mnist['images'][index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return trans.to_tensor(img), self.mnist['labels'][index]

    def download_url(self):
        '''
        Download mnist data set function, this function will be called when download is True.
        '''
        resources = [
            ("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
            ("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
            ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
            ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
        ]

        for url, md5 in resources:
            filename = url.rpartition('/')[2]
            download_url_to_local(url, filename, self.data_root, md5)

class EMNIST(Dataset):
    '''
    Jittor's own class for loading EMNIST dataset.

    Args::

        [in] data_root(str): your data root.
        [in] split(str): one of 'byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist'.
        [in] train(bool): choose model train or val.
        [in] download(bool): Download data automatically if download is Ture.
        [in] batch_size(int): Data batch size.
        [in] shuffle(bool): Shuffle data if true.
        [in] transform(jittor.transform): transform data.

    Example::

        from jittor.dataset.mnist import EMNIST
        train_loader = EMNIST(train=True).set_attrs(batch_size=16, shuffle=True)
        for i, (imgs, target) in enumerate(train_loader):
            ...
    '''

    _merged_classes = {'c', 'i', 'j', 'k', 'l', 'm', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'y', 'z'}
    _all_classes = set(string.digits + string.ascii_letters)
    classes_split_dict = {
        'byclass': sorted(list(_all_classes)),
        'bymerge': sorted(list(_all_classes - _merged_classes)),
        'balanced': sorted(list(_all_classes - _merged_classes)),
        'letters': ['N/A'] + list(string.ascii_lowercase),
        'digits': list(string.digits),
        'mnist': list(string.digits),
    }

    def __init__(self, data_root=dataset_root+"/emnist_data/", 
                 split='byclass',
                 train=True, 
                 download=True, 
                 batch_size = 16,
                 shuffle = False,
                 transform=None):
        # if you want to test resnet etc you should set input_channel = 3, because the net set 3 as the input dimensions
        super().__init__()
        self.data_root = data_root
        self.is_train = train
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        if download == True:
            self.download_url()
        data_root = os.path.join(data_root, "gzip")

        filesname = [
                f"emnist-{split}-train-images-idx3-ubyte.gz",
                f"emnist-{split}-t10k-images-idx3-ubyte.gz",
                f"emnist-{split}-train-labels-idx1-ubyte.gz",
                f"emnist-{split}-t10k-labels-idx1-ubyte.gz"
        ]
        for i in range(4):
            filesname[i] = os.path.join(data_root, filesname[i])
        self.mnist = {}
        if self.is_train:
            with gzip.open(filesname[0], 'rb') as f:
                self.mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28).transpose(0,2,1)
            with gzip.open(filesname[2], 'rb') as f:
                self.mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
        else:
            with gzip.open(filesname[1], 'rb') as f:
                self.mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28).transpose(0,2,1)
            with gzip.open(filesname[3], 'rb') as f:
                self.mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
        assert(self.mnist["images"].shape[0] == self.mnist["labels"].shape[0])
        self.total_len = self.mnist["images"].shape[0]
        # this function must be called
        self.set_attrs(total_len = self.total_len)

    def __getitem__(self, index):
        img = Image.fromarray(self.mnist['images'][index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return trans.to_tensor(img), self.mnist['labels'][index]

    def download_url(self):
        '''
        Download mnist data set function, this function will be called when download is True.
        '''
        resources = [
            ("https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip", "58c8d27c78d21e728a6bc7b3cc06412e"),
        ]

        for url, md5 in resources:
            filename = "emnist.zip"
            download_url_to_local(url, filename, self.data_root, md5)
            import zipfile
            zf = zipfile.ZipFile(os.path.join(self.data_root, filename))
            try:
                zf.extractall(path=self.data_root)
            except RuntimeError as e:
                print(e)
                raise
            zf.close()

