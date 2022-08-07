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
from jittor_utils import env_or_try_find, run_cmd, cache_path, LOG


has_rocm = 0
cc_flags = ""
hipcc_path = env_or_try_find('hipcc_path', 'hipcc')
rocm_home = ""
dlopen_flags = os.RTLD_NOW | os.RTLD_GLOBAL


def check_gcc_use_cxx11_abi():
    gcc_info = run_cmd("gcc -v")
    if "--with-default-libstdcxx-abi=new" in gcc_info:
        return True
    elif "--with-default-libstdcxx-abi=gcc4-compatible" in gcc_info:
        return False
    else:
        LOG.d("unknown cxx abi, defaults to gcc4-compatible")
        return False


def install_rocm_jittor_core():
    import jittor.compiler as compiler
    global has_rocm, cc_flags, rocm_home
    rocm_home = run_cmd("hipconfig -R")
    rocm_version = run_cmd("hipconfig -v")
    
    rocm_compiler_home = os.path.dirname(__file__)
    rocm_cache_gz_path = os.path.join(rocm_compiler_home, "rocm_cache.tar.gz")
    if os.path.exists(rocm_cache_gz_path):
        for o_file in glob.glob(rocm_compiler_home + "/**/*.o", recursive=True):
            os.remove(o_file)
        with tarfile.open(rocm_cache_gz_path, "r:gz") as tar:
            if (check_gcc_use_cxx11_abi()):
                tar.extractall(rocm_compiler_home, members=[tar.getmember("rocm_cache_cxx11.o")])
                o_files = [ os.path.join(rocm_compiler_home, "rocm_cache_cxx11.o") ]
            else:
                tar.extractall(rocm_compiler_home, members=[tar.getmember("rocm_cache.o")])
                o_files = [ os.path.join(rocm_compiler_home, "rocm_cache.o") ]
    
    cc_files = sorted(glob.glob(rocm_compiler_home + "/**/*.cc", recursive=True))
    cc_flags += f" -DHAS_CUDA -DIS_ROCM -I{rocm_compiler_home} "
    cc_flags += " " + run_cmd("hipconfig -C") + " "
    cc_flags += '  -L"' + os.path.join(rocm_home, "lib") + '" -lamdhip64 '
    LOG.i(f"ROCm ({rocm_version}) detected in {rocm_home}")

    mod = jittor_utils.compile_module('''
#include "common.h"
namespace jittor {
// @pyjt(process)
string process_rocm(const string& src, const string& name, const map<string,string>& kargs);
}''', compiler.cc_flags + " " + " ".join(cc_files + o_files) + cc_flags) 
    jittor_utils.process_jittor_source("rocm", mod.process)

    # preload hip driver to ensure the correct initialization of hip context
    hip_driver = ctypes.CDLL(os.path.join(rocm_home, 'lib', 'libamdhip64.so'), os.RTLD_GLOBAL | os.RTLD_NOW)
    r = hip_driver.hipDeviceSynchronize()

    has_rocm = 1


def install_hip():
    import jittor.compiler as compiler

    LOG.vv("setup rocm extern...")
    cache_path_cuda = os.path.join(cache_path, "cuda")
    cuda_include = os.path.join(compiler.jittor_path, "extern", "cuda", "inc")
    compiler.make_cache_dir(cache_path_cuda)
    cuda_extern_src = os.path.join(compiler.jittor_path, "extern", "cuda", "src")
    cuda_extern_files = [os.path.join(cuda_extern_src, name) for name in os.listdir(cuda_extern_src)]
    so_name = os.path.join(cache_path_cuda, "libcuda_extern" + compiler.so)
    compiler.compile(compiler.cc_path, compiler.cc_flags+f" -I\"{cuda_include}\" ", cuda_extern_files, so_name)
    ctypes.CDLL(so_name, dlopen_flags)


def install_rocm_library(lib_name, cuda_name, link=True):
    import jittor.compiler as compiler
    import jittor.compile_extern as compile_extern

    LOG.vv(f"setup {lib_name}...")
    rocmlib_include_path = os.path.join(rocm_home, lib_name.lower(), "include")
    
    jt_cuda_include = os.path.join(compiler.jittor_path, "extern", "cuda", "inc")
    jt_culib_include = os.path.join(compiler.jittor_path, "extern", "cuda", cuda_name, "inc")

    culib_src_dir = os.path.join(compiler.jittor_path, "extern", "cuda", cuda_name)
    culib_src_files = []
    for r, _, f in os.walk(culib_src_dir):
        for fname in f:
            culib_src_files.append(os.path.join(r, fname))

    extra_flags = f" -I\"{jt_cuda_include}\" -I\"{jt_culib_include}\" -I\"{rocmlib_include_path}\" "
    extra_flags += f" -L\"{os.path.join(cache_path, 'cuda')}\" -llibcuda_extern "
    if lib_name == "rccl":
        extra_flags += compile_extern.mpi_compile_flags

    if link:
        rocmlib_lib_path = os.path.join(rocm_home, lib_name.lower(), "lib")
        if os.path.exists(os.path.join(rocmlib_lib_path, f"lib{lib_name}.so")):
            jittor_utils.LOG.i(f"Found {os.path.join(rocmlib_lib_path, 'lib' + lib_name + '.so')}")
        extra_flags += f" -L{rocmlib_lib_path} -l{lib_name} "

    rocmlib = compiler.compile_custom_ops(culib_src_files, return_module=True, extra_flags=extra_flags)
    setattr(compile_extern, cuda_name, rocmlib)
    setattr(compile_extern, cuda_name + "_ops", rocmlib.ops)


def install_extern():
    if has_rocm:
        install_hip()
        install_rocm_library("MIOpen", "cudnn")
        install_rocm_library("rocblas", "cublas")
        install_rocm_library("rocprim", "cub", link=False)
        install_rocm_library("rccl", "nccl")
        return True
    else:
        return False

def convert_nvcc_flags(nvcc_flags):
    return nvcc_flags

def check():
    import jittor.compiler as compiler
    global has_rocm, cc_flags
    if hipcc_path:
        try:
            install_rocm_jittor_core()
        except Exception as e:
            jittor_utils.LOG.w(f"load ROCm failed, exception: {e}")
            has_rocm = 0
    compiler.has_rocm = has_rocm
    compiler.hipcc_path = hipcc_path
    if not has_rocm: 
        return False
    
    compiler.cc_flags += cc_flags
    compiler.nvcc_path = hipcc_path
    compiler.nvcc_flags = compiler.cc_flags.replace("-std=c++14", "-std=c++17")
    compiler.convert_nvcc_flags = convert_nvcc_flags
    return True
