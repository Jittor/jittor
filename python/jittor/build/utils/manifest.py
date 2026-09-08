# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
'''One table of every third-party binary Jittor downloads.

The url, the file name and the checksum of each archive used to be written out
again in every place that wanted one -- ``compile_extern.py``,
``jittor_utils/install_cuda.py``, ``jittor_utils/install_msvc.py`` and
``tools/release/pack_offline.py``. Four copies of a list drift, and these had:
``pack_offline.py``, whose whole job is to pre-fetch everything so that an
install can run without a network, was missing ``msvc.zip`` and every jtcuda
archive -- so its "offline" package still went to the network on any Windows
or driver-only CUDA machine.

Checksums are SHA-256 where one has been verified against the published
archive. MD5 is kept alongside because that is what the mirror's history is
recorded in, but it is only used when there is no SHA-256; ``digest_of``
returns the strongest available.
'''

import collections
import platform
import os

ASSET_BASE = "https://cg.cs.tsinghua.edu.cn/jittor/assets/"

Asset = collections.namedtuple(
    "Asset", "key filename url sha256 md5 platform")


def _asset(key, filename, sha256, md5, url=None, plat="any"):
    return Asset(key, filename, url or (ASSET_BASE + filename),
                 sha256, md5, plat)


#: one portable upstream v3 source archive; built for the requested host.
ONEDNN_VERSION = "3.9.1"
MKL = {
    host: _asset(
        "mkl", "oneDNN-v" + ONEDNN_VERSION + ".tar.gz",
        "fa44702f5979ed5ab927f7ccc1d2947adb4e6d0e58c433149465c5fc71e3bd45", "",
        url="https://codeload.github.com/uxlfoundation/oneDNN/tar.gz/refs/tags/v" + ONEDNN_VERSION,
        plat=host)
    for host in ("linux-x86_64", "linux-aarch64", "windows", "darwin-arm64", "darwin-x86_64")
}

CUB = _asset(
    "cub", "cub-1.11.0.tgz",
    "58948cfd0d46ed5ff683b5337a5de43d3c46a922f55404c32d09a6375b4f3108",
    "97196a885598e40592100e1caaf3d5ea",
    url="https://codeload.github.com/NVIDIA/cub/tar.gz/1.11.0")

CUTT = _asset(
    "cutt", "cutt-1.2.zip",
    "755aa0d75bf89cbe5db1b70adc0271c957afa4aa0531eac96e6cba4e4241e1f3",
    "14d0fd1132c8cd657dc3cf29ce4db931",
    url="https://codeload.github.com/Jittor/cutt/zip/v1.2")

NCCL = _asset(
    "nccl", "nccl.tgz",
    "a5c1b4da6e1608ee63baa87f6df424bba7a8b1cedad597a25d5b4cf8d56d0865",
    "900666558c5bc43e0a5e84045b88a06f",
    url="https://codeload.github.com/NVIDIA/nccl/tar.gz/v2.8.4-1")

MSVC = _asset(
    "msvc", "msvc.zip",
    "07900f69b877915d2db81d8134c5dde62f00f8d4081ab73d3bbde6c77ba56712",
    "55f0c175fdf1419b124e0fc498b659d2", plat="windows")

#: The bundled CUDA toolkits, chosen by driver version. No SHA-256 here: each
#: archive is around two gigabytes and none has been re-downloaded and hashed
#: since these entries were written. digest_of() falls back to the MD5 and
#: says so.
JTCUDA_LINUX = (
    ((12, 2), _asset("jtcuda", "cuda12.2_cudnn8_linux.tgz", "",
                     "7afda9332a268f29354488f13b489f53", plat="linux")),
    ((11, 2), _asset("jtcuda", "cuda11.2_cudnn8_linux.tgz", "",
                     "b93a1a5d19098e93450ee080509e9836", plat="linux")),
    ((11,), _asset("jtcuda", "cuda11.0_cudnn8_linux.tgz", "",
                   "5dbdb43e35b4db8249027997720bf1ca", plat="linux")),
    ((10, 2), _asset("jtcuda", "cuda10.2_cudnn7_linux.tgz", "",
                     "40f0563e8eb176f53e55943f6d212ad7", plat="linux")),
    ((10,), _asset("jtcuda", "cuda10.0_cudnn7_linux.tgz", "",
                   "f16d3ff63f081031d21faec3ec8b7dac", plat="linux")),
)

JTCUDA_WINDOWS = (
    ((11, 2), _asset("jtcuda", "cuda11.2_cudnn8_win.zip", "",
                     "b5543822c21bc460c1a414af47754556", plat="windows")),
    ((11,), _asset("jtcuda", "cuda11.0_cudnn8_win.zip", "",
                   "7a248df76ee5e79623236b0560f8d1fd", plat="windows")),
    ((10,), _asset("jtcuda", "cuda10.2_cudnn7_win.zip", "",
                   "7dd9963833a91371299a2ba58779dd71", plat="windows")),
)

#: MNIST, fetched by the dataset rather than the build, but part of what an
#: offline package has to carry.
MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST = tuple(
    Asset("mnist", name, MNIST_BASE + name, "", "", "any")
    for name in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
)


def mkl_asset(system=None, machine=None):
    """The MKL archive for a platform, or raise saying which are supported."""
    system = system or platform.system()
    machine = machine or platform.machine()
    if system == "Linux":
        if machine == "x86_64":
            return MKL["linux-x86_64"]
        if machine == "aarch64":
            return MKL["linux-aarch64"]
    elif system == "Windows" or (system is None and os.name == "nt"):
        return MKL["windows"]
    elif system == "Darwin":
        return MKL["darwin-arm64"] if machine == "arm64" \
            else MKL["darwin-x86_64"]
    raise RuntimeError(
        f"no MKL build is published for {system}/{machine}. Supported: "
        + ", ".join(sorted(MKL)) +
        ". Set use_mkl=0 to run without it, or report this at "
        "https://github.com/jittor/jittor")


def jtcuda_asset(driver_version, windows=False):
    """The bundled CUDA toolkit for a driver version, or None if too old."""
    table = JTCUDA_WINDOWS if windows else JTCUDA_LINUX
    for minimum, asset in table:
        if list(driver_version) >= list(minimum):
            return asset
    return None


def digest_of(asset):
    """(algorithm, hex digest) -- the strongest checksum recorded for it."""
    if asset.sha256:
        return "sha256", asset.sha256
    if asset.md5:
        return "md5", asset.md5
    return None, None


def offline_assets(include_cuda=True):
    """Everything an offline install may need to have on disk beforehand."""
    assets = list(MKL.values()) + [CUB, CUTT, NCCL, MSVC] + list(MNIST)
    if include_cuda:
        assets += [asset for _, asset in JTCUDA_LINUX]
        assets += [asset for _, asset in JTCUDA_WINDOWS]
    return assets
