# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
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
from jittor_utils import cuda_wheel
from jittor_utils import manifest
import pathlib


_cuda_wheel_stacks = {}


def _truthy(value):
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _prepend_env_path(name, path):
    paths = [p for p in os.environ.get(name, "").split(os.pathsep) if p]
    paths = [path] + [p for p in paths if os.path.abspath(p) != os.path.abspath(path)]
    os.environ[name] = os.pathsep.join(paths)


def get_cuda_wheel_stack(nvcc_version=None, refresh=False):
    """Resolve and activate Jittor's supported NVIDIA component-wheel stack."""

    key = str(nvcc_version or "")
    if not refresh and key in _cuda_wheel_stacks:
        return _cuda_wheel_stacks[key]
    strict = _truthy(os.environ.get("JITTOR_CUDA_WHEEL_STRICT"))
    stack = cuda_wheel.discover_cuda_wheel_stack(nvcc_version, strict=strict)
    _cuda_wheel_stacks[key] = stack
    if stack:
        # This affects compiler subprocesses and future child processes.  The
        # current process loads every selected library by absolute path.
        for lib_dir in reversed(stack.lib_dirs()):
            _prepend_env_path("LD_LIBRARY_PATH", lib_dir)
        LOG.i("Using NVIDIA CUDA pip wheels: ", stack.fingerprint)
    return stack

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
    """The driver's CUDA version, cached against the driver itself.

    ``nvidia-smi`` takes a noticeable fraction of a second and is started on
    every import on every machine that has a GPU.
    """
    if os.name == 'nt':
        return get_cuda_driver_win()
    driver = ""
    try:
        with open("/proc/driver/nvidia/version") as f:
            driver = f.readline().strip()
    except OSError:
        pass
    return jit_utils.probe.cached("cuda_driver", [], _read_cuda_driver,
                                  extra=driver)


def _read_cuda_driver():
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

def check_cuda_env():
    if not has_installation():
        return
    if os.name == "nt":
        return
    def fix_env(key):
        env = os.environ.get(key, "")
        env = env.replace(";",":").split(":")
        new_env = []
        changed = False
        for cp in env:
            x = cp.lower()
            if cuda_wheel.is_nvidia_wheel_path(cp):
                new_env.append(cp)
                continue
            if "cuda" in x and "jtcuda" not in x:
                changed = True
                continue
            if "jtcuda" in x:
                new_env.insert(0, cp)
            else:
                new_env.append(cp)
        os.environ[key] = ":".join(new_env)
        return changed
    changed = fix_env("PATH") \
        + fix_env("LD_LIBRARY_PATH") \
        + fix_env("CUDA_HOME")
    if changed:
        try:
            # LD_LIBRARY_PATH change must triggle restart
            # because dyloader already setup            
            # with open("/proc/self/maps", "r") as f:
            #     cudart_loaded = "cudart" in f.read().lower()
            # if cudart_loaded:
                with open("/proc/self/cmdline", "r") as f:
                    argv = f.read().split("\x00")
                    if len(argv[-1]) == 0: del argv[-1]
                if 'ipykernel_launcher' in argv:
                    LOG.i(f"needed restart but not {sys.executable} {argv[1:]}, you can ignore this warning.")
                else:
                    LOG.i(f"restart {sys.executable} {argv[1:]}")
                    os.execl(sys.executable, sys.executable, *argv[1:])
        except:
            pass
    

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

    asset = manifest.jtcuda_asset(cuda_driver_version, windows=os.name == 'nt')
    if asset is None:
        LOG.w(f"Unsupport cuda driver version: {cuda_driver_version}")
        return None
    cuda_tgz = asset.filename
    md5 = manifest.digest_of(asset)[1]
    jtcuda_path = os.path.join(jit_utils.home(), ".cache", "jittor", "jtcuda")
    nvcc_path = os.path.join(jtcuda_path, cuda_tgz[:-4], "bin", "nvcc")
    if os.name=='nt': nvcc_path += '.exe'
    nvcc_lib_path = os.path.join(jtcuda_path, cuda_tgz[:-4], "lib64")
    sys.path.append(nvcc_lib_path)
    new_ld_path = os.environ.get("LD_LIBRARY_PATH", "") + ":" + nvcc_lib_path
    os.environ["LD_LIBRARY_PATH"] = new_ld_path
    check_cuda_env()

    if os.path.isfile(nvcc_path):
        return nvcc_path

    os.makedirs(jtcuda_path, exist_ok=True)
    cuda_tgz_path = os.path.join(jtcuda_path, cuda_tgz)
    download_url_to_local(asset.url, cuda_tgz, jtcuda_path, md5)


    if cuda_tgz.endswith(".zip"):
        import zipfile
        zf = zipfile.ZipFile(cuda_tgz_path)
        zf.extractall(path=cuda_tgz_path[:-4])
    else:
        import tarfile
        from jittor_utils.misc import safe_tar_extractall
        with tarfile.open(cuda_tgz_path, "r") as tar:
            safe_tar_extractall(tar, cuda_tgz_path[:-4])

    assert os.path.isfile(nvcc_path), nvcc_path
    return nvcc_path


if __name__ == "__main__":
    nvcc_path = install_cuda()
    LOG.i("nvcc is installed at ", nvcc_path)
