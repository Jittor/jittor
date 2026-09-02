# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
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
from jittor_utils import lock, LOG
import gzip
import tarfile
import zipfile
# A directory of already-fetched third-party archives to copy from instead of
# downloading. JITTOR_OFFLINE_PATH names one directly; the `jittor_offline`
# package is the older way to ship one. Naming a directory is what lets a CI
# job, or several test sessions on one machine, share a single mirror that was
# populated once -- see `nox -s prefetch`.
jittor_offline_path = os.environ.get("JITTOR_OFFLINE_PATH") or None
if jittor_offline_path is None:
    try:
        import jittor_offline
        jittor_offline_path = os.path.dirname(jittor_offline.__file__)
    except Exception:
        pass


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def _progress():
    pbar = tqdm(total=None,
        unit="B",
        unit_scale=True,
        unit_divisor=1024)

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

def digest_algorithm(digest):
    """Which hash a hex string is, by length. None for "do not check"."""
    if not digest:
        return None
    if len(digest) == 64:
        return "sha256"
    if len(digest) == 32:
        return "md5"
    raise ValueError(f"unrecognised digest {digest!r}: expected 64 hex "
                     f"characters (sha256) or 32 (md5)")


def calculate_digest(file_path, algorithm="sha256", chunk_size=1024 * 1024):
    h = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    value = h.hexdigest()
    LOG.v(f"file {file_path} {algorithm}: {value}")
    return value


def calculate_md5(file_path, chunk_size=1024 * 1024):
    return calculate_digest(file_path, "md5", chunk_size)


def check_md5(file_path, md5, **kwargs):
    return md5 == calculate_md5(file_path, **kwargs)


def check_file_exist(file_path, digest):
    """Is this file present and does it match ``digest``?

    ``digest`` may be a SHA-256 or an MD5 hex string; which one is decided by
    its length, so callers that still pass an MD5 keep working while the
    manifest moves to SHA-256.
    """
    if not os.path.isfile(file_path):
        return False
    algorithm = digest_algorithm(digest)
    if algorithm is None:
        return True
    return digest == calculate_digest(file_path, algorithm)


def check_integrity(fpath, md5=None):
    return check_file_exist(fpath, md5)


@lock.lock_scope()
def download_url_to_local(url, filename, root_folder, digest):
    """Fetch ``url`` to ``root_folder/filename`` and verify it.

    Two things this did not use to do. It downloaded straight onto the final
    path, so an interrupted transfer left a truncated file that looked
    complete to the next run; the transfer now lands on ``<name>.part`` and is
    renamed into place only after it verifies. And a checksum mismatch raised
    but *kept* the bad file, so every later run recomputed the same hash of the
    same corrupt bytes and failed the same way -- the usual cause being a
    captive portal or proxy that returned an HTML error page.
    """
    ensure_dir(root_folder)
    file_path = os.path.join(root_folder, filename)
    if check_file_exist(file_path, digest):
        return
    if os.path.isfile(file_path):
        LOG.w(f"{file_path} does not match its recorded checksum; "
              f"removing it and fetching again")
        os.remove(file_path)
    if jittor_offline_path:
        offpath = os.path.join(jittor_offline_path, filename)
        if check_file_exist(offpath, digest):
            import shutil
            print('Using offline jittor', file_path)
            shutil.copy(offpath, file_path)
            return
    print('Downloading ' + url + ' to ' + file_path)
    part_path = file_path + ".part"
    try:
        urllib.request.urlretrieve(url, part_path, reporthook=_progress())
    except Exception as e:
        msg = f"{e}\nDownload File failed, url: {url}, path: {file_path}"
        print(msg)
        if os.path.isfile(part_path):
            os.remove(part_path)
        raise RuntimeError(msg)
    if not check_file_exist(part_path, digest):
        algorithm = digest_algorithm(digest)
        got = calculate_digest(part_path, algorithm)
        os.remove(part_path)
        raise RuntimeError(
            f"{filename} downloaded from {url} has {algorithm} {got}, "
            f"expected {digest}. The partial file has been removed. A proxy "
            f"or captive portal returning an error page is the usual cause; "
            f"check that {url} is reachable and try again.")
    os.replace(part_path, file_path)


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


def safe_tar_extractall(tar, path, members=None):
    """``extractall`` that refuses members writing outside ``path``.

    A tar entry may name ``../../etc/whatever`` or be a symlink pointing out of
    the tree; ``extractall`` used to follow it. ``filter="data"`` is Python's
    own answer and became the default in 3.14, so this is also what stops the
    behaviour from changing under us on that upgrade. Older interpreters do
    not accept the argument and get the old behaviour, which is what they had
    anyway.
    """
    try:
        tar.extractall(path=path, members=members, filter="data")
    except TypeError:
        tar.extractall(path=path, members=members)


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            safe_tar_extractall(tar, to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            safe_tar_extractall(tar, to_path)
    elif _is_tarxz(from_path):
        # .tar.xz archive only supported in Python 3.x
        with tarfile.open(from_path, 'r:xz') as tar:
            safe_tar_extractall(tar, to_path)
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
    # ``md5`` is kept as the parameter name for callers outside the repository;
    # any hex digest length that digest_algorithm() recognises works.
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url_to_local(url, filename, download_root, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)
