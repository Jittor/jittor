# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os, sys
from .compiler import *
from jittor.dataset.utils import download_url_to_local

def search_file(dirs, name):
    for d in dirs:
        fname = os.path.join(d, name)
        if os.path.isfile(fname):
            LOG.i(f"found {fname}")
            return fname
    LOG.f(f"file {name} not found in {dirs}")

def install_mkl(root_folder):
    url = "https://github.com/intel/mkl-dnn/releases/download/v1.0.2/mkldnn_lnx_1.0.2_cpu_gomp.tgz"
    filename = "mkldnn_lnx_1.0.2_cpu_gomp.tgz"
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, filename.replace(".tgz",""))

    if not os.path.isfile(os.path.join(dirname, "examples", "test")):
        LOG.i("Downloading mkl...")
        download_url_to_local(url, filename, root_folder, "47187284ede27ad3bd64b5f0e7d5e730")
        import tarfile

        with tarfile.open(fullname, "r") as tar:
            tar.extractall(root_folder)

        assert 0 == os.system(f"cd {dirname}/examples && "
            f"{cc_path} -std=c++14 cpu_cnn_inference_f32.cpp -Ofast -lmkldnn -I ../include -L ../lib -o test && LD_LIBRARY_PATH=../lib/ ./test")

def setup_mkl():
    global mkl_ops, use_mkl
    use_mkl = os.environ.get("use_mkl", "1")=="1"
    mkl_ops = None
    if not use_mkl: return
    mkl_include_path = os.environ.get("mkl_include_path")
    mkl_lib_path = os.environ.get("mkl_lib_path")
    
    if mkl_lib_path is None or mkl_include_path is None:
        mkl_install_sh = os.path.join(jittor_path, "script", "install_mkl.sh")
        LOG.v("setup mkl...")
        # mkl_path = os.path.join(cache_path, "mkl")
        # mkl_path decouple with cc_path
        from pathlib import Path
        mkl_path = os.path.join(str(Path.home()), ".cache", "jittor", "mkl")
        
        make_cache_dir(mkl_path)
        install_mkl(mkl_path)
        mkl_home = ""
        for name in os.listdir(mkl_path):
            if name.startswith("mkldnn_lnx") and os.path.isdir(os.path.join(mkl_path, name)):
                mkl_home = os.path.join(mkl_path, name)
                break
        assert mkl_home!=""
        mkl_include_path = os.path.join(mkl_home, "include")
        mkl_lib_path = os.path.join(mkl_home, "lib")

    mkl_lib_name = os.path.join(mkl_lib_path, "libmkldnn.so")
    assert os.path.isdir(mkl_include_path)
    assert os.path.isdir(mkl_lib_path)
    assert os.path.isfile(mkl_lib_name)
    LOG.v(f"mkl_include_path: {mkl_include_path}")
    LOG.v(f"mkl_lib_path: {mkl_lib_path}")
    LOG.v(f"mkl_lib_name: {mkl_lib_name}")
    # We do not link manualy, link in custom ops
    # ctypes.CDLL(mkl_lib_name, dlopen_flags)

    mkl_op_dir = os.path.join(jittor_path, "extern", "mkl", "ops")
    mkl_op_files = [os.path.join(mkl_op_dir, name) for name in os.listdir(mkl_op_dir)]
    mkl_ops = compile_custom_ops(mkl_op_files, 
        extra_flags=f" -I'{mkl_include_path}' -L'{mkl_lib_path}' -lmkldnn -Wl,-rpath='{mkl_lib_path}' ")
    LOG.vv("Get mkl_ops: "+str(dir(mkl_ops)))


def install_cub(root_folder):
    url = "https://github.com/NVlabs/cub/archive/v1.8.0.tar.gz"
    filename = "cub-1.8.0.tgz"
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, filename.replace(".tgz",""))
    
    if not os.path.isfile(os.path.join(dirname, "examples", "test")):
        LOG.i("Downloading cub...")
        download_url_to_local(url, filename, root_folder, "9203ea2499b56782601fddf8a12e9b08")
        import tarfile
    
        with tarfile.open(fullname, "r") as tar:
            tar.extractall(root_folder)
    
        assert 0 == os.system(f"cd {dirname}/examples && "
            f"{nvcc_path} device/example_device_radix_sort.cu -O2 -I.. -o test && ./test")
    return dirname

def setup_cub():
    from pathlib import Path
    cub_path = os.path.join(str(Path.home()), ".cache", "jittor", "cub")
    cub_home = install_cub(cub_path)
    setup_cuda_lib("cub", link=False, extra_flags=f"-I{cub_home}")

def setup_cuda_extern():
    if not has_cuda: return
    LOG.vv("setup cuda extern...")
    cache_path_cuda = os.path.join(cache_path, "cuda")
    cuda_include = os.path.join(jittor_path, "extern", "cuda", "inc")
    make_cache_dir(cache_path_cuda)
    cuda_extern_src = os.path.join(jittor_path, "extern", "cuda", "src")
    cuda_extern_files = [os.path.join(cuda_extern_src, name)
        for name in os.listdir(cuda_extern_src)]
    so_name = os.path.join(cache_path_cuda, "cuda_extern.so")
    compile(cc_path, cc_flags+f" -I'{cuda_include}' ", cuda_extern_files, so_name)
    ctypes.CDLL(so_name, dlopen_flags)

    try:
        setup_cub()
    except Exception as e:
        import traceback
        line = traceback.format_exc()
        LOG.w(f"CUDA found but cub is not loaded:\n{line}")

    libs = ["cublas", "cudnn", "curand"]
    for lib_name in libs:
        try:
            setup_cuda_lib(lib_name)
        except Exception as e:
            import traceback
            line = traceback.format_exc()
            LOG.w(f"CUDA found but {lib_name} is not loaded:\n{line}")

def setup_cuda_lib(lib_name, link=True, extra_flags=""):
    globals()[lib_name+"_ops"] = None
    if not has_cuda: return
    LOG.v(f"setup {lib_name}...")

    culib_path = os.path.join(cuda_lib, f"lib{lib_name}.so")
    jt_cuda_include = os.path.join(jittor_path, "extern", "cuda", "inc")
    jt_culib_include = os.path.join(jittor_path, "extern", "cuda", lib_name, "inc")

    link_flags = ""
    if link:
        cuda_include_name = search_file([cuda_include, "/usr/include"], lib_name+".h")
        culib_path = search_file([cuda_lib, "/usr/lib/x86_64-linux-gnu"], f"lib{lib_name}.so")
        # dynamic link cuda library
        ctypes.CDLL(culib_path, dlopen_flags)
        link_flags = f"-l{lib_name} -L'{cuda_lib}'"

    # find all source files
    culib_src_dir = os.path.join(jittor_path, "extern", "cuda", lib_name)
    culib_src_files = []
    for r, _, f in os.walk(culib_src_dir):
        for fname in f:
            culib_src_files.append(os.path.join(r, fname))
    if len(culib_src_files) == 0:
        return

    # compile and get operators
    culib_ops = compile_custom_ops(culib_src_files, 
        extra_flags=f" -I'{jt_cuda_include}' -I'{jt_culib_include}' {link_flags} {extra_flags} ")
    globals()[lib_name+"_ops"] = culib_ops
    LOG.vv(f"Get {lib_name}_ops: "+str(dir(culib_ops)))

def install_cutt(root_folder):
    url = "https://cloud.tsinghua.edu.cn/f/4be7e1dd51c6459aa119/?dl=1"
    filename = "cutt.tgz"
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, filename.replace(".tgz",""))
    true_md5 = "28a67bb3a713e29ce434303df6577507"

    if os.path.exists(fullname):
        md5 = os.popen('md5sum ' + fullname).read().split()[0]
    else:
        md5 = '233'
    if md5 != true_md5:
        os.system('rm ' + fullname)
        os.system('rm -rf ' + dirname)
    if not os.path.isfile(os.path.join(dirname, "bin", "cutt_test")):
        LOG.i("Downloading cub...")
        download_url_to_local(url, filename, root_folder, true_md5)

        import tarfile
    
        with tarfile.open(fullname, "r") as tar:
            tar.extractall(root_folder)

        from jittor_utils import run_cmd
        LOG.i("installing cutt...")
        run_cmd(f"cd {dirname} && make")
    return dirname

def setup_cutt():
    global cutt_ops, use_cutt
    if not has_cuda:
        use_cutt = False
        return
    use_cutt = os.environ.get("use_cutt", "1")=="1"
    cutt_ops = None
    if not use_cutt: return
    cutt_include_path = os.environ.get("cutt_include_path")
    cutt_lib_path = os.environ.get("cutt_lib_path")
    
    if cutt_lib_path is None or cutt_include_path is None:
        LOG.v("setup cutt...")
        # cutt_path decouple with cc_path
        from pathlib import Path
        cutt_path = os.path.join(str(Path.home()), ".cache", "jittor", "cutt")
        
        make_cache_dir(cutt_path)
        install_cutt(cutt_path)
        cutt_home = os.path.join(cutt_path, "cutt")
        cutt_include_path = os.path.join(cutt_home, "src")
        cutt_lib_path = os.path.join(cutt_home, "lib")

    cutt_lib_name = os.path.join(cutt_lib_path, "libcutt.so")
    assert os.path.isdir(cutt_include_path)
    assert os.path.isdir(cutt_lib_path)
    assert os.path.isfile(cutt_lib_name), cutt_lib_name
    LOG.v(f"cutt_include_path: {cutt_include_path}")
    LOG.v(f"cutt_lib_path: {cutt_lib_path}")
    LOG.v(f"cutt_lib_name: {cutt_lib_name}")
    # We do not link manualy, link in custom ops
    ctypes.CDLL(cutt_lib_name, dlopen_flags)

    cutt_op_dir = os.path.join(jittor_path, "extern", "cuda", "cutt", "ops")
    cutt_op_files = [os.path.join(cutt_op_dir, name) for name in os.listdir(cutt_op_dir)]
    cutt_ops = compile_custom_ops(cutt_op_files, 
        extra_flags=f" -I'{cutt_include_path}'")
    LOG.vv("Get cutt_ops: "+str(dir(cutt_ops)))


setup_cutt()
setup_mkl()

setup_cuda_extern()
