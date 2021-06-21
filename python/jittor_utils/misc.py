# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers:
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os
import hashlib
import urllib.request
from tqdm import tqdm
from jittor_utils import lock
import gzip
import tarfile
import zipfile

def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def _progress():
    pbar = tqdm(total=None)

    def bar_update(block_num, block_size, total_size):
        """ reporthook
        @block_num: the num of downloaded data block
        @block_size: the size of data block
        @total_size: the total size of remote file
        """
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = block_num * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

@lock.lock_scope()
def download_url_to_local(url, filename, root_folder, md5):
    ensure_dir(root_folder)
    file_path = os.path.join(root_folder, filename)
    if check_file_exist(file_path, md5):
        return
    else:
        print('Downloading ' + url + ' to ' + file_path)
        urllib.request.urlretrieve(
            url, file_path,
            reporthook=_progress()
        )
    if not check_file_exist(file_path, md5):
        raise RuntimeError("File downloads failed.")



def check_file_exist(file_path, md5):
    if not os.path.isfile(file_path):
        return False
    if md5 is None:
        return True
    return check_md5(file_path, md5)


def calculate_md5(file_path, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(file_path, md5, **kwargs):
    return md5 == calculate_md5(file_path, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tgz(filename):
    return filename.endswith(".tgz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        # .tar.xz archive only supported in Python 3.x
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(url, download_root, extract_root=None, filename=None,
                                 md5=None, remove_finished=False):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url_to_local(url, filename, download_root, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)
