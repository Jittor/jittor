# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
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

def get_cuda_driver():
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
    jtcuda_path = os.path.join(pathlib.Path.home(), ".cache", "jittor", "jtcuda")
    return os.path.isdir(jtcuda_path)

def install_cuda():
    cuda_driver_version = get_cuda_driver()
    if not cuda_driver_version:
        return None
    LOG.i("cuda_driver_version: ", cuda_driver_version)

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
        raise RuntimeError(f"Unsupport cuda driver version: {cuda_driver_version}")
    jtcuda_path = os.path.join(pathlib.Path.home(), ".cache", "jittor", "jtcuda")
    nvcc_path = os.path.join(jtcuda_path, cuda_tgz[:-4], "bin", "nvcc")
    nvcc_lib_path = os.path.join(jtcuda_path, cuda_tgz[:-4], "lib64")
    sys.path.append(nvcc_lib_path)
    new_ld_path = os.environ.get("LD_LIBRARY_PATH", "") + ":" + nvcc_lib_path
    os.environ["LD_LIBRARY_PATH"] = new_ld_path

    if os.path.isfile(nvcc_path):
        return nvcc_path

    os.makedirs(jtcuda_path, exist_ok=True)
    cuda_tgz_path = os.path.join(jtcuda_path, cuda_tgz)
    download_url_to_local("https://cg.cs.tsinghua.edu.cn/jittor/assets/"+cuda_tgz, cuda_tgz, jtcuda_path, md5)


    import tarfile
    with tarfile.open(cuda_tgz_path, "r") as tar:
        tar.extractall(cuda_tgz_path[:-4])

    assert os.path.isfile(nvcc_path)
    return nvcc_path


if __name__ == "__main__":
    nvcc_path = install_cuda()
    LOG.i("nvcc is installed at ", nvcc_path)
