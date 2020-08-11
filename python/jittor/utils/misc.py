# ***************************************************************
# Copyright (c) 2020 Jittor. Authors:
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import os
from six.moves import urllib
import hashlib
from tqdm import tqdm
from .. import lock

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
        print("Data file has been downloaded and verified")
    else:
        try:
            print('Downloading ' + url + ' to ' + file_path)
            urllib.request.urlretrieve(
                url, file_path,
                reporthook=_progress()
            )
        except(urllib.error.URLError, IOError) as e:
            raise e
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
