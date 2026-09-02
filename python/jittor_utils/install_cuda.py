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
    report = cuda_wheel.inspect_cuda_wheel_stack(nvcc_version)
    stack = report.stack
    if stack is None and report.reason:
        # A failure here is not fatal -- the build falls back to the system
        # CUDA, which is a supported way to run. What was fatal was doing it
        # in silence: the user who installed jittor[cuda12] and got the
        # system CUDA anyway learned about it several minutes later through a
        # cuDNN 9 error with no visible connection to the cause.
        #
        # JITTOR_CUDA_WHEEL_STRICT=1 turns any such failure into an error;
        # without it, a stack that resolved completely at its pinned versions
        # and then failed validation still raises, because nothing but a
        # broken jittor[cuda12] can produce that.
        if _truthy(os.environ.get("JITTOR_CUDA_WHEEL_STRICT")) or report.broken:
            raise cuda_wheel.CudaWheelError(report.reason)
        message = (
            "not using the NVIDIA CUDA pip wheels: %s. Falling back to the "
            "CUDA found on this system. If that CUDA ships cuDNN 9, the "
            "failure you get later comes from here." % report.reason)
        if report.present:
            # Some of the stack is installed, so this is plausibly a
            # jittor[cuda12] that drifted rather than a machine that never
            # had one.
            LOG.w(message)
        else:
            LOG.v(message)
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
        # NOTE: this used to read /proc/self/cmdline and os.execl() the user's
        # own process, so that the loader would pick up the new
        # LD_LIBRARY_PATH. A library must not restart the program it was
        # imported into:
        #
        #  * a script started via its shebang has argv[0] == the script, so
        #    `argv[1:]` drops the script and the "restart" runs
        #    `python <first argument>`;
        #  * everything the process did before `import jittor` -- opened files,
        #    parsed arguments, allocated memory -- is gone;
        #  * inside an MPI rank or a multiprocessing worker, one rank
        #    re-exec'ing itself takes the job with it;
        #  * the whole thing was wrapped in `except: pass`, so when it failed
        #    the process continued in the state the restart was meant to fix.
        #
        # Nothing here needs a restart. LD_LIBRARY_PATH matters to the loader
        # when it resolves a *name*; jittor loads the CUDA libraries by
        # absolute path (see preload_cuda_library / search_file), which the
        # loader honours in an already-running process. The corrected
        # environment is still exported for the child processes that read it.
        LOG.v("corrected CUDA paths in the environment for child processes")
    

#: Every published jtcuda archive is between one and a half and two and a half
#: gigabytes; the exact figure is not worth a HEAD request on the import path.
JTCUDA_APPROX_SIZE_GB = 2


def _download_is_allowed(asset):
    """Whether to fetch a ~2 GB CUDA toolkit without being asked to.

    ``import jittor`` used to start this download by itself whenever the
    machine had a driver but no nvcc on PATH. The user saw one line of
    "Downloading" and an import that did not return for a quarter of an hour,
    with no size, no progress, and no way to say no -- and on a restricted
    network, no output at all.

    It now happens only when asked: run

        python -m jittor_utils.install_cuda

    or set JTCUDA_AUTO_INSTALL=1 for unattended machines that want the old
    behaviour.
    """
    if _install_cuda_requested:
        return True
    if os.environ.get("JTCUDA_AUTO_INSTALL", "0") == "1":
        return True
    LOG.i(
        f"No nvcc found, and jittor can install a bundled CUDA toolkit "
        f"({asset.filename}, about {JTCUDA_APPROX_SIZE_GB} GB) for you. It "
        f"will not do that on its own: run "
        f"`{os.path.basename(sys.executable)} -m jittor_utils.install_cuda` "
        f"once, or set JTCUDA_AUTO_INSTALL=1, or install CUDA yourself and "
        f"put nvcc on PATH. Continuing on CPU.")
    return False


#: Set when this module is run as a command, i.e. the user asked for it.
_install_cuda_requested = False


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

    if not _download_is_allowed(asset):
        return None

    os.makedirs(jtcuda_path, exist_ok=True)
    cuda_tgz_path = os.path.join(jtcuda_path, cuda_tgz)
    LOG.i(f"Downloading {cuda_tgz} (about {JTCUDA_APPROX_SIZE_GB} GB) "
          f"from {asset.url} into {jtcuda_path}")
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
    # Running the module *is* the request; the import path never is.
    _install_cuda_requested = True
    nvcc_path = install_cuda()
    LOG.i("nvcc is installed at ", nvcc_path)
