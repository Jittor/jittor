# ***************************************************************
# Copyright(c) 2019
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
#     Hou Ming <houming818@qq.com>.
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import errno
import gzip
import hashlib
import os
import pickle
import sys
import tarfile
import zipfile

import numpy as np
from PIL import Image
from jittor.dataset.dataset import Dataset, dataset_root
from jittor.utils.misc import ensure_dir, download_url_to_local

class CIFAR10(Dataset):
    def __init__(self, dataset_root=dataset_root,
                 train=True,
                 download=True,
                 batch_size = 16,
                 shuffle = False,
                 transform=None,
                 target_transform=None):
        # if you want to test resnet etc you should set input_channel = 3, because the net set 3 as the input dimensions
        super().__init__()
        self.dataset_root = dataset_root
        self.base_folder = 'cifar-10-batches-py'
        self.is_train = train
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.meta = {
            'filename': 'batches.meta',
            'key': 'label_names',
            'md5': '5ff9c542aee3614f3951f8cda6e48888',
        }

        self.filesname = [
                "cifar-10-python.tar.gz",
        ]

        self.train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ]

        self.test_list = [
            ['test_batch', '40351d587109b95175f43aff81a1287e'],
        ]

        if download == True:
            self.download_and_extract_archive()

        if self.is_train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.dataset_root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        self.total_len = len(self.data)

    def __len__(self):
        return len(self.data)

    def _check_integrity(self, fpath, md5=None):
        if not os.path.isfile(fpath):
            return False
        if md5 is None:
            return True
        md5fpath = hashlib.md5()
        with open(fpath, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                md5fpath.update(chunk)
        return md5 == md5fpath.hexdigest()

    def _load_meta(self):
        path = os.path.join(self.dataset_root, self.base_folder, self.meta['filename'])
        if not self._check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def download_url(self, url, root, filename=None, md5=None):
        """Download a file from a url and place it in root.

        Args:
            url (str): URL to download file from
            root (str): Directory to place downloaded file in
            filename (str, optional): Name to save the file under. If None, use the basename of the URL
            md5 (str, optional): MD5 checksum of the download. If None, do not check
        """
        from six.moves import urllib

        root = os.path.expanduser(root)
        if not filename:
            filename = os.path.basename(url)
        fpath = os.path.join(root, filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # check if file is already present locally
        if self._check_integrity(fpath, md5):
            print('Using downloaded and verified file: ' + fpath)
        else:   # download the file
            try:
                print('Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )
            except (urllib.error.URLError, IOError) as e:
                if url[:5] == 'https':
                    url = url.replace('https:', 'http:')
                    print('Failed download. Trying https -> http instead.'
                          ' Downloading ' + url + ' to ' + fpath)
                    urllib.request.urlretrieve(
                        url, fpath,
                        reporthook=gen_bar_updater()
                    )
                else:
                    raise e
            # check integrity of downloaded file
            if not check_integrity(fpath, md5):
                raise RuntimeError("File not found or corrupted.")

    def extract_archive(self, from_path, to_path=None, remove_finished=False):
        if to_path is None:
            to_path = os.path.dirname(from_path)

        if from_path.endswith(".tar"):
            with tarfile.open(from_path, 'r') as tar:
                tar.extractall(path=to_path)
        elif from_path.endswith(".tar.gz") or from_path.endswith(".tgz"):
            with tarfile.open(from_path, 'r:gz') as tar:
                tar.extractall(path=to_path)
        elif from_path.endswith(".tar.xz") and PY3:
            # .tar.xz archive only supported in Python 3.x
            with tarfile.open(from_path, 'r:xz') as tar:
                tar.extractall(path=to_path)
        elif from_path.endswith(".gz") and not from_path.endswith(".tar.gz"):
            to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
            with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
                out_f.write(zip_f.read())
        elif from_path.endswith(".zip"):
            with zipfile.ZipFile(from_path, 'r') as z:
                z.extractall(to_path)
        else:
            raise ValueError("Extraction of {} not supported".format(from_path))

        if remove_finished:
            os.remove(from_path)

    def download_and_extract_archive(
        self, url=None, download_root=None, extract_root=None, filename=None, md5=None, remove_finished=False):
        resources = [
            ("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "c58f30108f718f92721af3b95e74349a"),
        ]

        url, md5 = resources[0]
        filename = url.rpartition('/')[2]
        download_url_to_local(url, filename, self.dataset_root, md5)
        download_root = os.path.expanduser(self.dataset_root)

        extract_root = download_root
        if not filename:
            filename = os.path.basename(url)

        self.download_url(url, download_root, filename, md5)

        archive = os.path.join(download_root, filename)
        print("Extracting {} to {}".format(archive, extract_root))
        self.extract_archive(archive, extract_root, remove_finished)

if __name__ == "__main__":
    # Download and construct CIFAR-10 dataset.
    train_dataset = CIFAR10(dataset_root=dataset_root,
                            train=True,
                            #transform=transforms.ToTensor(),
                            download=True)

    # Fetch one data pair (read data from disk).
    image, label = train_dataset[0]
    print (image.size)
    print (label)

    # When iteration starts, queue and thread start to load data from files.
    data_iter = iter(train_dataset)

    # Mini-batch images and labels.
    images, labels = data_iter.__next__()

    # Actual usage of the data loader is as below.
    for images, labels in data_iter:
        # Training code should be written here.
        pass


