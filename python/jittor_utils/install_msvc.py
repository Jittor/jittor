import os
import sys
from jittor_utils.misc import download_url_to_local
from jittor_utils import LOG


def install(path):
    LOG.i("Installing MSVC...")
    filename = "msvc.zip"
    url = "https://cg.cs.tsinghua.edu.cn/jittor/assets/" + filename
    md5sum = "55f0c175fdf1419b124e0fc498b659d2"
    download_url_to_local(url, filename, path, md5sum)
    fullname = os.path.join(path, filename)
    import zipfile
    with zipfile.ZipFile(fullname, "r") as f:
        f.extractall(path)
