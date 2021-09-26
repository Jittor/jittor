import os
import sys
from jittor_utils.misc import download_url_to_local
from jittor_utils import LOG


def install(path):
    LOG.i("Installing MSVC...")
    filename = "msvc.zip"
    url = "https://cg.cs.tsinghua.edu.cn/jittor/assets/" + filename
    md5sum = "929c4b86ea68160121f73c3edf6b5463"
    download_url_to_local(url, filename, path, md5sum)
    fullname = os.path.join(path, filename)
    import zipfile
    with zipfile.ZipFile(fullname, "r") as f:
        f.extractall(path)
