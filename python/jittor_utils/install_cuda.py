# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os
import sys
import subprocess as sp
import jittor_utils as jit_utils
from jittor_utils import LOG
from jittor_utils.misc import download_url_to_local
import pathlib

def get_cuda_driver_win():
    try:
        import ctypes
        cuda_driver = ctypes.CDLL(r"nvcuda")
        driver_version = ctypes.c_int()
        r = cuda_driver.cuDriverGetVersion(ctypes.byref(driver_version))
        if r != 0: return None
        v = driver_version.value
        return [v//1000, v%1000//10, v%10]
    except:
        return None

def get_cuda_driver():
    if os.name == 'nt':
        return get_cuda_driver_win()
    ret, out = sp.getstatusoutput("nvidia-smi -q -u")
    if ret != 0: return None
    try:
        out = out.lower()
        out = out.split('cuda version')[1] \
            .split(':')[1] \
            .splitlines()[0] \
            .strip()
        out = [ int(s) for s in out.split('.')]
        return out
    except:
        return None

def has_installation():
    jtcuda_path = os.path.join(jit_utils.home(), ".cache", "jittor", "jtcuda")
    return os.path.isdir(jtcuda_path)

def install_cuda():
    if "nvcc_path" in os.environ and os.environ["nvcc_path"] == "":
        return None
    cuda_driver_version = get_cuda_driver()
    if not cuda_driver_version:
        return None
    LOG.i("cuda_driver_version: ", cuda_driver_version)
    if "JTCUDA_VERSION" in os.environ:
        cuda_driver_version = list(map(int,os.environ["JTCUDA_VERSION"].split(".")))
        LOG.i("JTCUDA_VERSION: ", cuda_driver_version)

    if os.name == 'nt':
        # TODO: cuda11.4 has bug fit with
        # current msvc, FIXME
        # if cuda_driver_version >= [11,4]:
        #     cuda_tgz = "cuda11.4_cudnn8_win.zip"
        #     md5 = "06eed370d0d44bb2cc57809343911187"
        if cuda_driver_version >= [11,2]:
            cuda_tgz = "cuda11.2_cudnn8_win.zip"
            md5 = "b5543822c21bc460c1a414af47754556"
        elif cuda_driver_version >= [11,]:
            cuda_tgz = "cuda11.0_cudnn8_win.zip"
            md5 = "7a248df76ee5e79623236b0560f8d1fd"
        elif cuda_driver_version >= [10,]:
            cuda_tgz = "cuda10.2_cudnn7_win.zip"
            md5 = "7dd9963833a91371299a2ba58779dd71"
        else:
            raise RuntimeError(f"Unsupport cuda driver version: {cuda_driver_version}, at least 10.2")
    else:
        if cuda_driver_version >= [11,2]:
            cuda_tgz = "cuda11.2_cudnn8_linux.tgz"
            md5 = "b93a1a5d19098e93450ee080509e9836"
        elif cuda_driver_version >= [11,]:
            cuda_tgz = "cuda11.0_cudnn8_linux.tgz"
            md5 = "5dbdb43e35b4db8249027997720bf1ca"
        elif cuda_driver_version >= [10,2]:
            cuda_tgz = "cuda10.2_cudnn7_linux.tgz"
            md5 = "40f0563e8eb176f53e55943f6d212ad7"
        elif cuda_driver_version >= [10,]:
            cuda_tgz = "cuda10.0_cudnn7_linux.tgz"
            md5 = "f16d3ff63f081031d21faec3ec8b7dac"
        else:
            raise RuntimeError(f"Unsupport cuda driver version: {cuda_driver_version}, at least 10.0")
    jtcuda_path = os.path.join(jit_utils.home(), ".cache", "jittor", "jtcuda")
    nvcc_path = os.path.join(jtcuda_path, cuda_tgz[:-4], "bin", "nvcc")
    if os.name=='nt': nvcc_path += '.exe'
    nvcc_lib_path = os.path.join(jtcuda_path, cuda_tgz[:-4], "lib64")
    sys.path.append(nvcc_lib_path)
    new_ld_path = os.environ.get("LD_LIBRARY_PATH", "") + ":" + nvcc_lib_path
    os.environ["LD_LIBRARY_PATH"] = new_ld_path

    if os.path.isfile(nvcc_path):
        return nvcc_path

    os.makedirs(jtcuda_path, exist_ok=True)
    cuda_tgz_path = os.path.join(jtcuda_path, cuda_tgz)
    download_url_to_local("https://cg.cs.tsinghua.edu.cn/jittor/assets/"+cuda_tgz, cuda_tgz, jtcuda_path, md5)


    if cuda_tgz.endswith(".zip"):
        import zipfile
        zf = zipfile.ZipFile(cuda_tgz_path)
        zf.extractall(path=cuda_tgz_path[:-4])
    else:
        import tarfile
        with tarfile.open(cuda_tgz_path, "r") as tar:
            tar.extractall(cuda_tgz_path[:-4])

    assert os.path.isfile(nvcc_path), nvcc_path
    return nvcc_path


if __name__ == "__main__":
    nvcc_path = install_cuda()
    LOG.i("nvcc is installed at ", nvcc_path)
