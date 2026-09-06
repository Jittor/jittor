# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Zheng-Ning Liu <lzhengning@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os
import ctypes
import glob
import tarfile

import jittor_utils
from jittor_utils import env_or_try_find, run_cmd, LOG
from jittor_utils.misc import safe_tar_extractall
from jittor_utils.backend_resources import backend_root


def check_gcc_use_cxx11_abi():
    gcc_info = run_cmd("gcc -v")
    if "--with-default-libstdcxx-abi=new" in gcc_info:
        return True
    elif "--with-default-libstdcxx-abi=gcc4-compatible" in gcc_info:
        return False
    else:
        LOG.d("unknown cxx abi, defaults to gcc4-compatible")
        return False


def configure(context):
    hipcc_path = env_or_try_find('hipcc_path', 'hipcc')
    if not hipcc_path:
        raise RuntimeError("ROCm compiler hipcc is unavailable")
    rocm_home = run_cmd("hipconfig -R")
    rocm_version = run_cmd("hipconfig -v")
    
    rocm_compiler_home = os.path.dirname(__file__)
    rocm_cache_gz_path = os.path.join(rocm_compiler_home, "rocm_cache.tar.gz")
    if not os.path.isfile(rocm_cache_gz_path):
        raise RuntimeError("ROCm converter archive is missing: " + rocm_cache_gz_path)
    object_dir = os.path.join(context.config.cache_path, "rocm")
    context.make_cache_dir(object_dir)
    member_name = "rocm_cache_cxx11.o" if check_gcc_use_cxx11_abi() else "rocm_cache.o"
    with tarfile.open(rocm_cache_gz_path, "r:gz") as tar:
        safe_tar_extractall(tar, object_dir, members=[tar.getmember(member_name)])
    o_files = [os.path.join(object_dir, member_name)]
    
    cc_files = sorted(glob.glob(rocm_compiler_home + "/**/*.cc", recursive=True))
    cc_flags = f" -DHAS_CUDA -DIS_ROCM -I{rocm_compiler_home} "
    cc_flags += " " + run_cmd("hipconfig -C") + " "
    cc_flags += '  -L"' + os.path.join(rocm_home, "lib") + '" -lamdhip64 '
    LOG.i(f"ROCm ({rocm_version}) detected in {rocm_home}")

    mod = context.compile_module('''
#include "common.h"
namespace jittor {
// @pyjt(process)
string process_rocm(const string& src, const string& name, const map<string,string>& kargs);
}''', context.config.cc_flags + " " + " ".join(cc_files + o_files) + cc_flags)
    config = context.transform_sources(context.config, "rocm", mod.process)

    # preload hip driver to ensure the correct initialization of hip context
    hip_driver = ctypes.CDLL(os.path.join(rocm_home, 'lib', 'libamdhip64.so'), os.RTLD_GLOBAL | os.RTLD_NOW)
    status = hip_driver.hipDeviceSynchronize()
    if status:
        raise RuntimeError("ROCm driver initialization failed: hipDeviceSynchronize=%s" % status)
    cc_flags = config.cc_flags + cc_flags
    return config.evolve(
        backend="rocm", has_rocm=True, has_cuda=True, is_cuda=False, hipcc_path=hipcc_path,
        cc_flags=cc_flags, nvcc_path=hipcc_path,
        nvcc_flags=cc_flags.replace("-std=c++14", "-std=c++17"),
        convert_nvcc_flags=convert_nvcc_flags,
        resources=dict(config.resources, rocm_home=rocm_home,
                       rocm_version=rocm_version, rocm_driver=hip_driver,
                       rocm_converter=mod),
    )


def install_hip(context):
    config = context.config
    LOG.vv("setup rocm extern...")
    cache_path_cuda = os.path.join(config.cache_path, "cuda")
    cuda_root = backend_root(config.jittor_path, "cuda")
    cuda_include = os.path.join(cuda_root, "include")
    context.make_cache_dir(cache_path_cuda)
    cuda_extern_src = os.path.join(cuda_root, "src")
    cuda_extern_files = [os.path.join(cuda_extern_src, name) for name in os.listdir(cuda_extern_src)]
    so_name = os.path.join(cache_path_cuda, "libcuda_extern" + context.so)
    context.compile(config.cc_path, config.cc_flags+f" -I\"{cuda_include}\" ", cuda_extern_files, so_name)
    return context.load_library(so_name, os.RTLD_NOW | os.RTLD_GLOBAL)


def install_rocm_library(context, lib_name, cuda_name, link=True):
    config = context.config
    rocm_home = config.resources["rocm_home"]
    LOG.vv(f"setup {lib_name}...")
    rocmlib_include_path = os.path.join(rocm_home, lib_name.lower(), "include")
    
    cuda_root = backend_root(config.jittor_path, "cuda")
    jt_cuda_include = os.path.join(cuda_root, "include")
    jt_culib_include = os.path.join(cuda_root, "libraries", cuda_name, "include")
    culib_src_dirs = (os.path.join(cuda_root, "kernels", cuda_name),
                      os.path.join(cuda_root, "libraries", cuda_name))
    if cuda_name == "nccl":
        culib_src_dirs = (os.path.join(config.jittor_path, "extern", "cuda", "nccl"),)
        jt_culib_include = os.path.join(culib_src_dirs[0], "inc")
    culib_src_files = []
    for directory in culib_src_dirs:
        for r, _, f in os.walk(directory):
            for fname in f:
                if fname.endswith((".h", ".cc", ".cu", ".cuh")):
                    culib_src_files.append(os.path.join(r, fname))

    extra_flags = f" -I\"{jt_cuda_include}\" -I\"{jt_culib_include}\" -I\"{rocmlib_include_path}\" "
    extra_flags += f" -L\"{os.path.join(config.cache_path, 'cuda')}\" -llibcuda_extern "
    if lib_name == "rccl":
        extra_flags += context.mpi_compile_flags

    if link:
        rocmlib_lib_path = os.path.join(rocm_home, lib_name.lower(), "lib")
        if os.path.exists(os.path.join(rocmlib_lib_path, f"lib{lib_name}.so")):
            jittor_utils.LOG.i(f"Found {os.path.join(rocmlib_lib_path, 'lib' + lib_name + '.so')}")
        extra_flags += f" -L{rocmlib_lib_path} -l{lib_name} "

    rocmlib = context.compile_custom_ops(culib_src_files, return_module=True,
                                         extra_flags=extra_flags, backend="accelerator")
    context.publish_library(cuda_name, rocmlib)


def install_extern(context):
    if context.config.has_rocm:
        hip_library = install_hip(context)
        context.publish_library("cuda", hip_library)
        install_rocm_library(context, "MIOpen", "cudnn")
        install_rocm_library(context, "rocblas", "cublas")
        install_rocm_library(context, "rocprim", "cub", link=False)
        install_rocm_library(context, "rccl", "nccl")
        return True
    else:
        return False

def convert_nvcc_flags(nvcc_flags):
    return nvcc_flags

def post_process(context):
    return context.config
