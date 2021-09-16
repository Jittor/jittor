import os
import sys
from jittor_utils.misc import download_url_to_local
from jittor_utils import LOG


def install(path):
    LOG.i("Installing MSVC...")
    filename = "msvc.zip"
    url = "https://cg.cs.tsinghua.edu.cn/jittor/assets/" + filename
    md5sum = "0fd71436c034808649b24baf28998ccc"
    download_url_to_local(url, filename, path, md5sum)
    fullname = os.path.join(path, filename)
    import zipfile
    with zipfile.ZipFile(fullname, "r") as f:
        f.extractall(path)
