import os
import sys
from jittor_utils.misc import download_url_to_local
from jittor_utils import LOG
from jittor_utils import manifest


def install(path):
    LOG.i("Installing MSVC...")
    asset = manifest.MSVC
    digest = manifest.digest_of(asset)[1]
    download_url_to_local(asset.url, asset.filename, path, digest)
    fullname = os.path.join(path, asset.filename)
    import zipfile
    # zipfile.extractall already refuses to write outside `path`: it strips
    # drive letters, leading separators and ".." from every member name. That
    # is not true of tarfile, which is why tar extraction goes through
    # misc.safe_tar_extractall instead.
    with zipfile.ZipFile(fullname, "r") as f:
        f.extractall(path)
