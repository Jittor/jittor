# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os, sys, shutil
import platform
from .compiler import *
from jittor_utils import run_cmd, get_version, get_int_version
from jittor_utils.misc import download_url_to_local
import jittor_utils as jit_utils

def search_file(dirs, name, prefer_version=()):
    if os.name == 'nt':
        if name.startswith("lib"):
            name = name[3:].replace(".so", "64*.dll")
    for d in dirs:
        fname = os.path.join(d, name)
        if os.name == 'nt':
            lname = os.path.join(d, name)
            names = glob.glob(lname)
            if len(names):
                return names[0]
            continue
        prefer_version = tuple( str(p) for p in prefer_version )
        for i in range(len(prefer_version),-1,-1):
            vname = ".".join((fname,)+prefer_version[:i])
            if os.path.isfile(vname):
                LOG.v(f"found {vname}")
                return vname
    LOG.f(f"file {name} not found in {dirs}")

def install_mkl(root_folder):
    # origin url is
    # url = "https://github.com/intel/mkl-dnn/releases/download/v1.0.2/mkldnn_lnx_1.0.2_cpu_gomp.tgz"
    import platform
    url = None
    if platform.system()=="Linux":
        if platform.machine()=='x86_64':
            filename = "dnnl_lnx_2.2.0_cpu_gomp.tgz"
            md5 = "35bbbdf550a9d8ad54db798e372000f6"
        elif platform.machine()=='aarch64':
            filename = "dnnl_lnx_2.2.0_cpu_gomp_aarch64.tgz"
            md5 = "72cf9b0b8fd6c3c786d35a9daaee22b8"
        else:
            raise RuntimeError(f"platform.machine()=={platform.machine()} not support yet,"
            " Please contact us on https://github.com/jittor/jittor ")
    elif os.name == "nt":
        # url = "https://github.com/oneapi-src/oneDNN/releases/download/v2.2/dnnl_win_2.2.0_cpu_iomp.zip"
        # url = "https://github.com/oneapi-src/oneDNN/releases/download/v2.2/dnnl_win_2.2.0_cpu_vcomp.zip"
        filename = "dnnl_win_2.2.0_cpu_vcomp.zip"
        md5 = "fa12c693b2ec07700d174e1e99d60a7e"
    elif platform.system() == "Darwin":
        if platform.machine() == "arm64":
            filename = "dnnl_mac_2.2.0_cpu_omp_arm64.tgz"
            md5 = "d8fdf56d3cf618685d22d18f08119f88"
        else:
            filename = "dnnl_mac_2.2.0_cpu_omp_x86_64.tgz"
            md5 = "6e2f065d6a589c82081536b684768fe6"
    else:
        raise RuntimeError(f"platform.machine()=={platform.machine()} not support yet,"
        " Please contact us on https://github.com/jittor/jittor ")

    if not url:
        url = "https://cg.cs.tsinghua.edu.cn/jittor/assets/" + filename
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, filename.rsplit(".",1)[0])

    if not (os.path.isfile(os.path.join(dirname, "lib", "libmkldnn.so")) or
        os.path.isfile(os.path.join(dirname, "bin", "dnnl.dll")) or 
        os.path.isfile(os.path.join(dirname, "lib", "libmkldnn.dylib"))):
        LOG.i("Downloading mkl...")
        download_url_to_local(url, filename, root_folder, md5)
        if fullname.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(fullname, "r") as f:
                f.extractall(root_folder)
        else:
            import tarfile
            with tarfile.open(fullname, "r") as tar:
                tar.extractall(root_folder)
        if os.name == 'nt':
            # this env is used for execute example/text
            bin_path = os.path.join(dirname, "bin")
            sys.path.append(bin_path)
            os.environ["PATH"] = os.environ.get("PATH", "") + ";" + bin_path
            cmd = f"cd /d {dirname}/examples && {cc_path} {dirname}/examples/cnn_inference_f32.cpp -I{dirname}/include -Fe: {dirname}/examples/test.exe {fix_cl_flags(cc_flags).replace('-LD', '')} {dirname}/lib/mkldnn.lib"
            
            assert 0 == os.system(cmd)
            assert 0 == os.system(f"{dirname}/examples/test")
        elif platform.system() == "Darwin":
            assert 0 == os.system(f"cd {dirname}/examples && "
            f"{cc_path} -std=c++14 cnn_inference_f32.cpp -Ofast -lmkldnn -I ../include -L ../lib -o test && DYLD_LIBRARY_PATH=../lib/ ./test")
        else:
            assert 0 == os.system(f"cd {dirname}/examples && "
            f"{cc_path} -std=c++14 cnn_inference_f32.cpp -Ofast -lmkldnn -I ../include -L ../lib -o test && LD_LIBRARY_PATH=../lib/ ./test")

def setup_mkl():
    global mkl_ops, use_mkl
    use_mkl = os.environ.get("use_mkl", "1")=="1"
    mkl_ops = None
    if not use_mkl: return

    # pytorch mkl is conflict with jittor mkl
    # yield error "free: invalide size" or
    # "mmap error"
    # import pytorch(>1.8) first can fix this problem
    # try:
    #     # jt.dirty_fix_pytorch_runtime_error()
    #     import torch
    #     from torch import nn
    # except:
    #     torch = None

    mkl_include_path = os.environ.get("mkl_include_path")
    mkl_lib_path = os.environ.get("mkl_lib_path")
    
    if mkl_lib_path is None or mkl_include_path is None:
        LOG.v("setup mkl...")
        # mkl_path = os.path.join(cache_path, "mkl")
        # mkl_path decouple with cc_path
        mkl_path = os.path.join(jit_utils.home(), ".cache", "jittor", "mkl")
        
        make_cache_dir(mkl_path)
        install_mkl(mkl_path)
        mkl_home = ""
        for name in os.listdir(mkl_path):
            if name.startswith("dnnl") and os.path.isdir(os.path.join(mkl_path, name)):
                mkl_home = os.path.join(mkl_path, name)
                break
        assert mkl_home!=""
    mkl_include_path = os.path.join(mkl_home, "include")
    mkl_lib_path = os.path.join(mkl_home, "lib")

    mkl_lib_name = os.path.join(mkl_lib_path, "libmkldnn.so")
    extra_flags = f" -I\"{mkl_include_path}\" -L\"{mkl_lib_path}\" -lmkldnn "
    if os.name == 'nt':
        mkl_lib_name = os.path.join(mkl_home, 'bin', 'dnnl.dll')
        mkl_bin_path = os.path.join(mkl_home, 'bin')
        extra_flags = f" -I\"{mkl_include_path}\"  -L\"{mkl_lib_path}\" -L\"{mkl_bin_path}\" -ldnnl "
    elif platform.system() == "Darwin":
        mkl_lib_name = os.path.join(mkl_lib_path, "libmkldnn.dylib")

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
    mkl_ops = compile_custom_ops(mkl_op_files, extra_flags=extra_flags)
    LOG.vv("Get mkl_ops: "+str(dir(mkl_ops)))


def install_cub(root_folder):
    url = "https://github.com/NVIDIA/cub/archive/1.11.0.tar.gz"
    url = "https://codeload.github.com/NVIDIA/cub/tar.gz/1.11.0"
    filename = "cub-1.11.0.tgz"
    md5 = "97196a885598e40592100e1caaf3d5ea"
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, filename.replace(".tgz",""))
    
    if not os.path.isfile(os.path.join(dirname, "examples", "device/example_device_radix_sort.cu")):
        LOG.i("Downloading cub...")
        download_url_to_local(url, filename, root_folder, md5)
        import tarfile
    
        with tarfile.open(fullname, "r") as tar:
            tar.extractall(root_folder)
        # assert 0 == os.system(f"cd {dirname}/examples && "
        #             f"{nvcc_path} --cudart=shared -ccbin=\"{cc_path}\"  device/example_device_radix_sort.cu -O2 -I.. -std=c++14 -o test")
        # if core.get_device_count():
        #     assert 0 == os.system(f"cd {dirname}/examples && ./test")
    return dirname

def setup_cub():
    global cub_home
    cub_home = ""
    cub_path = os.path.join(jit_utils.home(), ".cache", "jittor", "cub")
    cuda_version = int(get_version(nvcc_path)[1:-1].split('.')[0])
    extra_flags = ""
    if cuda_version < 11:
        cub_home = install_cub(cub_path)
        extra_flags = f"-I{cub_home}"
        cub_home += "/"
    setup_cuda_lib("cub", link=False, extra_flags=extra_flags)

def setup_cuda_extern():
    if not has_cuda: return
    def split(a): return a.replace(";",":").split(":")
    check_ld_path = split(os.environ.get("LD_LIBRARY_PATH", "")) + \
        split(os.environ.get("PATH", ""))
    for cp in check_ld_path:
        cp = cp.lower()
        if "cuda" in cp and \
            "lib" in cp and \
            "jtcuda" not in cp:
            LOG.w(f"CUDA related path found in LD_LIBRARY_PATH or PATH, "
            "This path may cause jittor found the wrong libs, "
            "please unset LD_LIBRARY_PATH and remove cuda lib path in Path. \n"
            "Or you can let jittor install cuda for you: `python3.x -m jittor_utils.install_cuda`")
            break
    LOG.vv("setup cuda extern...")
    cache_path_cuda = os.path.join(cache_path, "cuda")
    cuda_include = os.path.join(jittor_path, "extern", "cuda", "inc")
    make_cache_dir(cache_path_cuda)
    cuda_extern_src = os.path.join(jittor_path, "extern", "cuda", "src")
    cuda_extern_files = [os.path.join(cuda_extern_src, name)
        for name in os.listdir(cuda_extern_src)]
    so_name = os.path.join(cache_path_cuda, "libcuda_extern"+so)
    compile(cc_path, cc_flags+f" -I\"{cuda_include}\" ", cuda_extern_files, so_name)
    link_cuda_extern = f" -L\"{cache_path_cuda}\" -llibcuda_extern "
    ctypes.CDLL(so_name, dlopen_flags)

    try:
        setup_cub()
    except Exception as e:
        import traceback
        line = traceback.format_exc()
        LOG.w(f"CUDA found but cub is not loaded:\n{line}")

    libs = ["cublas", "cudnn", "curand", "cufft"]
    # in cuda 11.4, module memory comsumptions:
    # default context: 259 MB
    # cublas: 340 MB
    # cudnn: 340 MB
    if int(os.environ.get("conv_opt", "0")):
        libs = ["cublas", "curand"]
    for lib_name in libs:
        try:
            setup_cuda_lib(lib_name, extra_flags=link_cuda_extern)
        except Exception as e:
            msg = f"CUDA found but {lib_name} is not loaded:\n"
            if lib_name == "cudnn":
                msg += """Develop version of CUDNN not found, 
please refer to CUDA offical tar file installation: 
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar"""
            if platform.machine() in ["x86_64", "AMD64"]:
                msg += f"""
or you can let jittor install cuda and cudnn for you:
>>> python3.{sys.version_info.minor} -m jittor_utils.install_cuda
"""
            LOG.f(msg)

def setup_cuda_lib(lib_name, link=True, extra_flags=""):
    arch_key = "x86_64"
    if platform.machine() not in ["x86_64", "AMD64"]:
        arch_key = "aarch64"
    globals()[lib_name+"_ops"] = None
    globals()[lib_name] = None
    if not has_cuda: return
    LOG.v(f"setup {lib_name}...")

    culib_path = os.path.join(cuda_lib, f"lib{lib_name}.so")
    jt_cuda_include = os.path.join(jittor_path, "extern", "cuda", "inc")
    jt_culib_include = os.path.join(jittor_path, "extern", "cuda", lib_name, "inc")

    link_flags = ""
    if link:
        extra_include_path = os.path.abspath(os.path.join(cuda_include, "..", f"targets/{arch_key}-linux/include"))
        extra_lib_path = os.path.abspath(os.path.join(cuda_lib, "..", f"targets/{arch_key}-linux/lib"))
        cuda_include_name = search_file([cuda_include, extra_include_path, "/usr/include"], lib_name+".h")
        # cuda11 prefer cudnn 8
        nvcc_version = get_int_version(nvcc_path)
        if has_corex:
            nvcc_version = (10,2,89)
        prefer_version = ()
        if nvcc_version[0] == 11:
            prefer_version = ("8",)
        culib_path = search_file([cuda_bin, cuda_lib, extra_lib_path, f"/usr/lib/{arch_key}-linux-gnu", "/usr/lib"], f"lib{lib_name}.so", prefer_version)

        if lib_name == "cublas" and nvcc_version[0] >= 10:
            # manual link libcublasLt.so
            try:
                cublas_lt_lib_path = search_file([cuda_bin, cuda_lib, extra_lib_path, f"/usr/lib/{arch_key}-linux-gnu", "/usr/lib"], f"libcublasLt.so", nvcc_version)
                ctypes.CDLL(cublas_lt_lib_path, dlopen_flags)
            except:
                # some aarch64 os, such as uos with FT2000 cpu,
                # it's cuda 10 doesn't have libcublasLt.so
                pass



        if lib_name == "cudnn":
            # cudnn cannot found libcudnn_cnn_train.so.8, we manual link for it.
            if nvcc_version >= (11,0,0):
                libs = ["libcudnn_ops_infer.so", "libcudnn_ops_train.so", "libcudnn_cnn_infer.so", "libcudnn_cnn_train.so"]
                for l in libs:
                    ex_cudnn_path = search_file([cuda_bin, cuda_lib, extra_lib_path, f"/usr/lib/{arch_key}-linux-gnu", "/usr/lib"], l, prefer_version)
                    ctypes.CDLL(ex_cudnn_path, dlopen_flags)

        # dynamic link cuda library
        # ctypes.CDLL(culib_path, dlopen_flags)
        # link_flags = f"-l{lib_name} -L\"{cuda_lib}\""
        link_flags = f"-l{lib_name} -L\"{os.path.dirname(culib_path)}\""
        # print("link_flags", link_flags, culib_path)

    # find all source files
    culib_src_dir = os.path.join(jittor_path, "extern", "cuda", lib_name)
    culib_src_files = []
    for r, _, f in os.walk(culib_src_dir):
        for fname in f:
            culib_src_files.append(os.path.join(r, fname))
    if len(culib_src_files) == 0:
        return

    # compile and get operators
    culib = compile_custom_ops(culib_src_files, return_module=True,
        extra_flags=f" -I\"{jt_cuda_include}\" -I\"{jt_culib_include}\" {link_flags} {extra_flags} ")
    culib_ops = culib.ops
    globals()[lib_name+"_ops"] = culib_ops
    globals()[lib_name] = culib
    LOG.vv(f"Get {lib_name}_ops: "+str(dir(culib_ops)))


def _setup_fake_cuda_lib(lib_name=None, link=True, extra_flags=""):
    if lib_name is None:
        lib_names = ["cudnn", "cublas", "curand", "cufft", "cub", "cutt", "cutlass"]
        for lib_name in lib_names:
            _setup_fake_cuda_lib(lib_name, link, extra_flags)
        return
    arch_key = "x86_64"
    if platform.machine() not in ["x86_64", "AMD64"]:
        arch_key = "aarch64"
    globals()[lib_name+"_ops"] = None
    globals()[lib_name] = None
    LOG.v(f"setup {lib_name}...")

    jt_cuda_include = os.path.join(jittor_path, "extern", "cuda", "inc")
    jt_culib_include = os.path.join(jittor_path, "extern", "cuda", lib_name, "inc")

    # find all source files
    culib_src_dir = os.path.join(jittor_path, "extern", "cuda", lib_name, "ops")
    culib_src_files = []
    for r, _, f in os.walk(culib_src_dir):
        for fname in f:
            if fname.endswith("op.cc") or fname.endswith("op.h"):
                culib_src_files.append(os.path.join(r, fname))
    if len(culib_src_files) == 0:
        return

    # compile and get operators
    culib = compile_custom_ops(culib_src_files, return_module=True,
        extra_flags=f" -I\"{jt_cuda_include}\" -I\"{jt_culib_include}\" {extra_flags} ")
    culib_ops = culib.ops
    globals()[lib_name+"_ops"] = culib_ops
    globals()[lib_name] = culib
    LOG.vv(f"Get {lib_name}_ops: "+str(dir(culib_ops)))

if setup_fake_cuda_lib:
    _setup_fake_cuda_lib()

def install_cutt(root_folder):
    # Modified from: https://github.com/ap-hynninen/cutt
    url = "https://codeload.github.com/Jittor/cutt/zip/v1.2"

    filename = "cutt-1.2.zip"
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, filename.replace(".zip",""))
    true_md5 = "14d0fd1132c8cd657dc3cf29ce4db931"

    if os.path.exists(fullname):
        from jittor_utils.misc import calculate_md5
        md5 = calculate_md5(fullname)
        if md5 != true_md5:
            os.remove(fullname)
            shutil.rmtree(dirname)
    CUTT_PATH = os.environ.get("CUTT_PATH", "")
    if not os.path.isfile(os.path.join(cache_path, "libcutt"+so)) or CUTT_PATH:
        if CUTT_PATH:
            dirname = CUTT_PATH
        else:
            LOG.i("Downloading cutt...")
            download_url_to_local(url, filename, root_folder, true_md5)

            import zipfile

            zf = zipfile.ZipFile(fullname)
            try:
                zf.extractall(path=root_folder)
            except RuntimeError as e:
                print(e)
                raise
            zf.close()

        LOG.i("installing cutt...")
        # -Xptxas -dlcm=ca actually not work
        arch_flag = " -Xptxas -dlcm=ca "
        if len(flags.cuda_archs):
            arch_flag = f" -arch=compute_{min(flags.cuda_archs)} "
            arch_flag += ''.join(map(lambda x:f' -code=sm_{x} ', flags.cuda_archs))
        cutt_include = f" -I\"{dirname}/include\" -I\"{dirname}/src\" "
        files = glob.glob(dirname+"/src/*.c*", recursive=True)
        files2 = []
        for f in files:
            if f.endswith("cutt_bench.cpp") or \
                f.endswith("cutt_test.cpp"):
                continue
            files2.append(f)
        cutt_flags = cc_flags+opt_flags+cutt_include
        compile(cc_path, cutt_flags, files2, cache_path+"/libcutt"+so, cuda_flags=arch_flag)
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
        cutt_path = os.path.join(jit_utils.home(), ".cache", "jittor", "cutt")
        
        make_cache_dir(cutt_path)
        install_cutt(cutt_path)
        cutt_home = os.path.join(cutt_path, "cutt-1.2")
        cutt_include_path = os.path.join(cutt_home, "src")
        cutt_lib_path = cache_path

    cutt_lib_name = os.path.join(cutt_lib_path, "libcutt"+so)
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
        extra_flags=f" -I\"{cutt_include_path}\" -L\"{cutt_lib_path}\" -llibcutt ")
    LOG.vv("Get cutt_ops: "+str(dir(cutt_ops)))

def install_cutlass(root_folder):
    # Modified from: https://github.com/ap-hynninen/cutlass
    url = "https://cloud.tsinghua.edu.cn/f/171e49e5825549548bc4/?dl=1"

    filename = "cutlass.zip"
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, filename.replace(".zip",""))
    true_md5 = "999ecb7e217e40c497bc3d0ded6643f0"

    if os.path.exists(fullname):
        from jittor_utils.misc import calculate_md5
        md5 = calculate_md5(fullname)
        if md5 != true_md5:
            os.remove(fullname)
            shutil.rmtree(dirname)
    CUTLASS_PATH = os.environ.get("CUTLASS_PATH", "")
    if not os.path.isfile(os.path.join(jit_utils.home(), ".cache/jittor/cutlass/cutlass/include/cutlass/cutlass.h")) or CUTLASS_PATH:
        if CUTLASS_PATH:
            dirname = CUTLASS_PATH
        else:
            LOG.i("Downloading cutlass...")
            download_url_to_local(url, filename, root_folder, true_md5)

            import zipfile

            zf = zipfile.ZipFile(fullname)
            try:
                zf.extractall(path=root_folder)
            except RuntimeError as e:
                print(e)
                raise
            zf.close()

        # LOG.i("installing cutlass...")
        # # -Xptxas -dlcm=ca actually not work
        # arch_flag = " -Xptxas -dlcm=ca "
        # if len(flags.cuda_archs):
        #     arch_flag = f" -arch=compute_{min(flags.cuda_archs)} "
        #     arch_flag += ''.join(map(lambda x:f' -code=sm_{x} ', flags.cuda_archs))
        # cutlass_include = f" -I\"{dirname}/include\" -I\"{dirname}/src\" "
        # files = glob.glob(dirname+"/src/*.c*", recursive=True)
        # files2 = []
        # for f in files:
        #     if f.endswith("cutlass_bench.cpp") or \
        #         f.endswith("cutlass_test.cpp"):
        #         continue
        #     files2.append(f)
        # cutlass_flags = cc_flags+opt_flags+cutlass_include
        # compile(cc_path, cutlass_flags, files2, cache_path+"/libcutlass"+so, cuda_flags=arch_flag)
    return dirname

def setup_cutlass():
    global cutlass_ops, use_cutlass
    if not has_cuda:
        use_cutlass = False
        return
    use_cutlass = os.environ.get("use_cutlass", "1")=="1"
    cutlass_ops = None
    if not use_cutlass: return
    cutlass_include_path = os.environ.get("cutlass_include_path")
    
    if cutlass_include_path is None:
        LOG.v("setup cutlass...")
        # cutlass_path decouple with cc_path
        cutlass_path = os.path.join(jit_utils.home(), ".cache", "jittor", "cutlass")
        
        make_cache_dir(cutlass_path)
        install_cutlass(cutlass_path)


def install_nccl(root_folder):
    url = "https://github.com/NVIDIA/nccl/archive/v2.8.4-1.tar.gz"
    url = "https://codeload.github.com/NVIDIA/nccl/tar.gz/v2.8.4-1"

    filename = "nccl.tgz"
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, "nccl-2.8.4-1")
    true_md5 = "900666558c5bc43e0a5e84045b88a06f"

    if os.path.exists(fullname):
        md5 = run_cmd('md5sum '+fullname).split()[0]
        if md5 != true_md5:
            os.remove(fullname)
            if os.path.isdir(dirname):
                shutil.rmtree(dirname)
    if not os.path.isfile(os.path.join(dirname, "build", "lib", "libnccl.so")):
        if not os.path.isfile(os.path.join(root_folder, filename)):
            LOG.i("Downloading nccl...")
        download_url_to_local(url, filename, root_folder, true_md5)

        if core.get_device_count() == 0:
            return
        if not inside_mpi():
            return

        import tarfile
        with tarfile.open(fullname, "r") as tar:
            tar.extractall(root_folder)

        LOG.i("installing nccl...")
        arch_flag = ""
        if len(flags.cuda_archs):
            arch_flag = f" -arch=compute_{min(flags.cuda_archs)} "
            arch_flag += ''.join(map(lambda x:f' -code=sm_{x} ', flags.cuda_archs))
        run_cmd(f"CC=\"{cc_path}\" CXX=\"{cc_path}\" make -j8 src.build CUDA_HOME='{cuda_home}' NVCC_GENCODE='{arch_flag} --cudart=shared ' ", cwd=dirname)
    return dirname

def setup_nccl():
    global nccl, nccl_ops, use_nccl
    use_nccl = os.environ.get("use_nccl", "1")=="1"
    nccl = None
    nccl_ops = None
    if not has_cuda or not has_mpi:
        use_nccl = False
        return
    if not use_nccl: return
    nccl_include_path = os.environ.get("nccl_include_path")
    nccl_lib_path = os.environ.get("nccl_lib_path")
    
    if nccl_lib_path is None or nccl_include_path is None:
        LOG.v("setup nccl...")
        # nccl_path decouple with cc_path
        nccl_path = os.path.join(jit_utils.home(), ".cache", "jittor", "nccl")
        
        make_cache_dir(nccl_path)
        nccl_home = install_nccl(nccl_path)
        if nccl_home is None: return
        nccl_include_path = os.path.join(nccl_home, "build", "include")
        nccl_lib_path = os.path.join(nccl_home, "build", "lib")
        
    if not inside_mpi():
        return

    nccl_lib_name = os.path.join(nccl_lib_path, "libnccl.so")
    assert os.path.isdir(nccl_include_path)
    assert os.path.isdir(nccl_lib_path)
    assert os.path.isfile(nccl_lib_name), nccl_lib_name
    LOG.v(f"nccl_include_path: {nccl_include_path}")
    LOG.v(f"nccl_lib_path: {nccl_lib_path}")
    LOG.v(f"nccl_lib_name: {nccl_lib_name}")
    # We do not link manualy, link in custom ops
    ctypes.CDLL(nccl_lib_name, dlopen_flags)

    nccl_src_dir = os.path.join(jittor_path, "extern", "cuda", "nccl")
    nccl_src_files = []
    for r, _, f in os.walk(nccl_src_dir):
        for fname in f:
            nccl_src_files.append(os.path.join(r, fname))

    nccl = compile_custom_ops(nccl_src_files, 
        extra_flags=f" -I\"{nccl_include_path}\" {mpi_compile_flags} ",
        return_module=True, dlopen_flags=os.RTLD_GLOBAL | os.RTLD_NOW,
        gen_name_="jittor_nccl_core")
    nccl_ops = nccl.ops
    LOG.vv("Get nccl_ops: "+str(dir(nccl_ops)))

def manual_link(flags):
    lib_dirs = []
    libs = []
    for f in flags.split():
        if f.startswith("-l"):
            libs.append(f[2:])
        elif f.startswith("-L"):
            lib_dirs.append(f[2:])
    LOG.v("manual_link:", flags)
    LOG.v("lib_dirs:", lib_dirs)
    LOG.v("libs:", libs)
    for lib in libs:
        for d in lib_dirs:
            libname = os.path.join(d, f"lib{lib}.so")
            if os.path.isfile(libname):
                LOG.v("link:", libname)
                ctypes.CDLL(libname, dlopen_flags)
                break

def inside_mpi():
    return "OMPI_COMM_WORLD_SIZE" in os.environ

def setup_mpi():
    global mpi_ops, mpi, use_mpi
    global mpicc_path, has_mpi
    use_mpi = os.environ.get("use_mpi", "1")=="1"
    mpi_ops = None
    mpi = None
    has_mpi = False
    if not use_mpi: return
    mpicc_path = env_or_try_find('mpicc_path', 'mpicc')
    if mpicc_path == "":
        # LOG.i("mpicc not found, distribution disabled.")
        use_mpi = False
    else:
        use_mpi = True
        has_mpi = True
    if not use_mpi:
        return

    global mpi_compile_flags, mpi_link_flags, mpi_flags
    mpi_compile_flags = run_cmd(mpicc_path+" --showme:compile")
    mpi_link_flags = run_cmd(mpicc_path+" --showme:link")
    mpi_flags = mpi_compile_flags + " " + mpi_link_flags
    LOG.v("mpi_flags: "+mpi_flags)

    # find all source files
    mpi_src_dir = os.path.join(jittor_path, "extern", "mpi")
    mpi_src_files = []
    for r, _, f in os.walk(mpi_src_dir):
        for fname in f:
            mpi_src_files.append(os.path.join(r, fname))

    # mpi compile flags add for nccl
    mpi_compile_flags += f" -I\"{os.path.join(mpi_src_dir, 'inc')}\" "
    mpi_compile_flags = mpi_compile_flags.replace("-pthread", "")

    mpi_version = get_version(mpicc_path)
    if mpi_version.startswith("(1.") or mpi_version.startswith("(2."):
        # mpi version 1.x need to link like this
        manual_link(mpi_flags)
    # mpi(4.x) cannot use deepbind, it need to
    # share the 'environ' symbol.
    mpi = compile_custom_ops(mpi_src_files, 
        extra_flags=f" {mpi_flags} ", return_module=True,
        dlopen_flags=os.RTLD_GLOBAL | os.RTLD_NOW, gen_name_="jittor_mpi_core")
    mpi_ops = mpi.ops
    LOG.vv("Get mpi: "+str(mpi.__dict__.keys()))
    LOG.vv("Get mpi_ops: "+str(mpi_ops.__dict__.keys()))
    def wrapper(func):
        def inner(self, *args, **kw):
            return func(self, *args, **kw)
        inner.__doc__ = func.__doc__
        return inner
    for k in mpi_ops.__dict__:
        if not k.startswith("mpi_"): continue
        if k == "mpi_test": continue
        setattr(core.Var, k, wrapper(mpi_ops.__dict__[k]))

in_mpi = inside_mpi()
FIX_TORCH_ERROR = 0
if os.name != 'nt' and not in_mpi:
    FIX_TORCH_ERROR = 1
if "FIX_TORCH_ERROR" in os.environ:
    FIX_TORCH_ERROR = os.environ["FIX_TORCH_ERROR"] != "0"
if FIX_TORCH_ERROR:
    try:
        import torch
        from jittor_utils import dirty_fix_pytorch_runtime_error
        dirty_fix_pytorch_runtime_error()
    except:
        pass

cudnn = cublas = curand = cufft = None
setup_mpi()
rank = mpi.world_rank() if in_mpi else 0
world_size = mpi.world_size() if in_mpi else 1
setup_nccl()

setup_cutt()
setup_cutlass()

# try:
setup_mkl()
# except Exception as e:
#     LOG.w("MKL install failed, msg:", e)

setup_cuda_extern()

# install backend extern library
for mod in jit_utils.backends:
    if mod.install_extern():
        break
