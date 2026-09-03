# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os, sys, shutil, re
import platform
from .compiler import *
from jittor_utils import run_cmd, get_version, get_int_version
from jittor_utils.misc import download_url_to_local, safe_tar_extractall
from jittor_utils import manifest
import jittor_utils as jit_utils

def search_file(dirs, name, prefer_version=()):
    if os.name == 'nt':
        if name.startswith("lib"):
            name = name[3:].replace(".so", "64*.dll")
    prefer_version = tuple(str(p) for p in prefer_version)
    for d in dirs:
        fname = os.path.join(d, name)
        if os.name == 'nt':
            lname = os.path.join(d, name)
            names = glob.glob(lname)
            if len(names):
                return names[0]
            continue
        for i in range(len(prefer_version),-1,-1):
            vname = ".".join((fname,)+prefer_version[:i])
            if os.path.isfile(vname):
                LOG.v(f"found {vname}")
                return vname
        versioned = glob.glob(fname + ".*") if ".so" in name else []
        versioned = [path for path in versioned if os.path.isfile(path)]
        if versioned:
            versioned.sort(key=lambda path: (
                os.path.basename(path).count("."),
                len(os.path.basename(path)),
                os.path.basename(path),
            ))
            LOG.v(f"found {versioned[0]}")
            return versioned[0]
    raise RuntimeError(
        f"{name} not found. Searched: " + ", ".join(dirs) +
        ". Install the package that provides it, or point Jittor at it with "
        "cuda_home/CUDA_HOME.")


#: The versioned split libraries a cuDNN installation is made of, by major
#: version. cuDNN ships the public ``libcudnn.so`` as a thin dispatcher and
#: puts the kernels in these; the wheels contain no unversioned symlinks and
#: are not on the loader path, so jittor loads them itself, in dependency
#: order, before the dispatcher.
#:
#: cuDNN 9 renamed every one of them: the six ``<domain>_<infer|train>``
#: libraries became a graph/ops/cnn/adv split plus the engine and heuristic
#: libraries. Asking for the 8 names on a 9 install fails at
#: "libcudnn_ops_infer.so not found", which names something that no longer
#: exists rather than the version mismatch that caused it.
CUDNN_SPLIT_LIBRARIES = {
    8: (
        "libcudnn_ops_infer.so", "libcudnn_ops_train.so",
        "libcudnn_cnn_infer.so", "libcudnn_cnn_train.so",
        "libcudnn_adv_infer.so", "libcudnn_adv_train.so",
    ),
    9: (
        "libcudnn_graph.so",
        "libcudnn_ops.so",
        "libcudnn_cnn.so",
        "libcudnn_adv.so",
        "libcudnn_heuristic.so",
        "libcudnn_engines_precompiled.so",
        "libcudnn_engines_runtime_compiled.so",
    ),
}


def cudnn_major_version(culib_path, cudnn_header_path=None):
    """The major version of the cuDNN that `culib_path` is, or None.

    The SONAME carries it (``libcudnn.so.9``); a path that does not, such as a
    plain ``libcudnn.so``, falls back to ``cudnn_version.h`` next to the header
    jittor is about to compile against. Returning None means "could not tell",
    which is a different thing from "old": callers treat it as the current
    layout rather than guessing a legacy one.
    """
    match = re.search(r"\.so\.(\d+)", os.path.basename(os.path.realpath(culib_path)))
    if match:
        return int(match.group(1))
    if not cudnn_header_path:
        return None
    version_header = os.path.join(
        os.path.dirname(cudnn_header_path), "cudnn_version.h")
    try:
        with open(version_header, "r", encoding="utf-8") as f:
            version_text = f.read()
    except OSError:
        return None
    match = re.search(
        r"^\s*#\s*define\s+CUDNN_MAJOR\s+(\d+)", version_text, re.MULTILINE)
    return int(match.group(1)) if match else None


def cudnn_split_libraries(cudnn_major):
    """The split libraries to preload for this cuDNN, newest layout by default."""
    if cudnn_major is None:
        cudnn_major = max(CUDNN_SPLIT_LIBRARIES)
    return CUDNN_SPLIT_LIBRARIES.get(
        cudnn_major, CUDNN_SPLIT_LIBRARIES[max(CUDNN_SPLIT_LIBRARIES)])

def install_mkl(root_folder):
    # origin url is
    # https://github.com/oneapi-src/oneDNN/releases/download/v2.2/
    asset = manifest.mkl_asset()
    filename = asset.filename
    url = asset.url
    md5 = manifest.digest_of(asset)[1]
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
                safe_tar_extractall(tar, root_folder)
        if os.name == 'nt':
            # dnnl.dll and its dependencies live here.
            bin_path = os.path.join(dirname, "bin")
            sys.path.append(bin_path)
            os.environ["PATH"] = os.environ.get("PATH", "") + ";" + bin_path
        check_mkl_usable(dirname)


def check_mkl_usable(dirname):
    """Load the library we just unpacked and look for the symbol we use.

    This used to compile one of the upstream examples with the user's compiler
    and then *run* the resulting binary, on the import path, with `assert 0 ==
    os.system(...)` as the only diagnostic -- so a broken download reached the
    user as a bare AssertionError, and every import of a fresh cache built and
    executed a third-party program. Loading the library and resolving
    ``dnnl_sgemm`` (the entry point jittor's own MKL operators call) answers
    the same question: is this archive usable from this process.
    """
    candidates = [
        os.path.join(dirname, "lib", "libmkldnn.so"),
        os.path.join(dirname, "lib", "libmkldnn.dylib"),
        os.path.join(dirname, "bin", "dnnl.dll"),
    ]
    for lib_path in candidates:
        if os.path.isfile(lib_path):
            break
    else:
        raise RuntimeError(
            f"the MKL/oneDNN archive unpacked into {dirname} but none of "
            f"{candidates} exists; delete that directory and its archive to "
            f"download it again.")
    try:
        lib = ctypes.CDLL(lib_path, dlopen_flags)
    except OSError as error:
        raise RuntimeError(
            f"could not load {lib_path}: {error}. Delete {dirname} and its "
            f"archive to download it again.") from error
    if not hasattr(lib, "dnnl_sgemm"):
        raise RuntimeError(
            f"{lib_path} loaded but has no dnnl_sgemm, which jittor's MKL "
            f"operators call. This is not the expected oneDNN build; delete "
            f"{dirname} and its archive to download it again.")
    LOG.v(f"mkl usable: {lib_path}")

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
    asset = manifest.CUB
    url, filename = asset.url, asset.filename
    md5 = manifest.digest_of(asset)[1]
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, filename.replace(".tgz",""))
    
    if not os.path.isfile(os.path.join(dirname, "examples", "device/example_device_radix_sort.cu")):
        LOG.i("Downloading cub...")
        download_url_to_local(url, filename, root_folder, md5)
        import tarfile
    
        with tarfile.open(fullname, "r") as tar:
            safe_tar_extractall(tar, root_folder)
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
        if cuda_wheel_stack and cuda_wheel_stack.owns_path(cp):
            continue
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

    libs = ["cublas", "cudnn", "curand", "cufft", "cusparse"]
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
            if lib_name == "cusparse":
                msg += """CUSPARSE library is not loaded, 
please ensure it is installed along with the CUDA toolkit."""
            if platform.machine() in ["x86_64", "AMD64"]:
                msg += f"""
or you can let jittor install cuda and cudnn for you:
>>> python3.{sys.version_info.minor} -m jittor_utils.install_cuda
"""
            raise RuntimeError(msg) from e

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
        component_include_dirs = []
        component_lib_dirs = []
        if cuda_wheel_stack:
            component_include_dirs = cuda_wheel_stack.include_dirs(lib_name)
            component_lib_dirs = cuda_wheel_stack.lib_dirs(lib_name)
        include_search_dirs = component_include_dirs + [cuda_include, extra_include_path, "/usr/include"]
        library_search_dirs = component_lib_dirs + [
            cuda_bin, cuda_lib, extra_lib_path,
            f"/usr/lib/{arch_key}-linux-gnu", "/usr/lib",
        ]
        cuda_include_name = search_file(include_search_dirs, lib_name+".h")
        extra_flags = f' -I"{os.path.dirname(cuda_include_name)}" ' + extra_flags
        # cuda11 prefer cudnn 8
        nvcc_version = get_int_version(nvcc_path)
        if globals().get("has_corex", False):
            nvcc_version = (10,2,89)
        prefer_version = ()
        if nvcc_version[0] == 11:
            prefer_version = ("8",)
        culib_path = search_file(library_search_dirs, f"lib{lib_name}.so", prefer_version)
        if cuda_wheel_stack:
            preload_cuda_library(lib_name, required=True)

        if lib_name == "cublas" and nvcc_version[0] >= 10:
            # manual link libcublasLt.so
            try:
                cublas_lt_lib_path = search_file(library_search_dirs, f"libcublasLt.so", nvcc_version)
                ctypes.CDLL(cublas_lt_lib_path, dlopen_flags)
            except:
                # some aarch64 os, such as uos with FT2000 cpu,
                # it's cuda 10 doesn't have libcublasLt.so
                pass



        if lib_name == "cudnn":
            # cuDNN 9 was refused here, because the RNN ops were written on the
            # v6 RNN entry points that cuDNN 9 removed. They are on the v8 API
            # now (8.04), so the only thing the major version still decides is
            # what the split libraries are called.
            cudnn_major = cudnn_major_version(culib_path, cuda_include_name)
            LOG.v(f"found cudnn {cudnn_major} at {culib_path}")
            # cuDNN wheels contain only versioned split libraries. Load them
            # before the public library so RNN and convolution symbols are
            # available without relying on LD_LIBRARY_PATH.
            if nvcc_version >= (11,0,0) and not cuda_wheel_stack:
                prefer = (str(cudnn_major),) if cudnn_major else ()
                for l in cudnn_split_libraries(cudnn_major):
                    ex_cudnn_path = search_file(library_search_dirs, l, prefer)
                    ctypes.CDLL(ex_cudnn_path, dlopen_flags)

        if not cuda_wheel_stack:
            ctypes.CDLL(culib_path, dlopen_flags)
        link_flags = cuda_library_link_flags(lib_name, culib_path)
        # print("link_flags", link_flags, culib_path)

        if lib_name == "cusparse" :
            try:
                cusparse_spmv_path = search_file(library_search_dirs, "libcusparse.so")
                ctypes.CDLL(cusparse_spmv_path, dlopen_flags)
            except:
                LOG.w("Failed to load cusparse-specific shared libraries.")

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
        lib_names = ["cudnn", "cublas", "curand", "cufft", "cub", "cutt"]
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
    asset = manifest.CUTT
    url, filename = asset.url, asset.filename
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, filename.replace(".zip",""))
    true_md5 = manifest.digest_of(asset)[1]

    if os.path.exists(fullname):
        from jittor_utils.misc import check_file_exist
        if not check_file_exist(fullname, true_md5):
            os.remove(fullname)
            if os.path.isdir(dirname):
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
            arch_flag = cuda_arch_flags(flags.cuda_archs)
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
    global cutt_ops, cutt, use_cutt
    if not has_cuda:
        use_cutt = False
        return
    use_cutt = os.environ.get("use_cutt", "1")=="1"
    cutt_ops = None
    cutt = None
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
    # Keep the module, not just its .ops: the plan-cache accessors are free
    # functions on the module, and every other backend is exposed this way.
    cutt = compile_custom_ops(cutt_op_files, return_module=True,
        extra_flags=f" -I\"{cutt_include_path}\" -L\"{cutt_lib_path}\" -llibcutt ")
    cutt_ops = cutt.ops
    LOG.vv("Get cutt_ops: "+str(dir(cutt_ops)))

def install_nccl(root_folder):
    asset = manifest.NCCL
    url, filename = asset.url, asset.filename
    fullname = os.path.join(root_folder, filename)
    dirname = os.path.join(root_folder, "nccl-2.8.4-1")
    true_md5 = manifest.digest_of(asset)[1]

    if os.path.exists(fullname):
        # Was `md5sum` via the shell, which is neither on Windows nor on macOS.
        from jittor_utils.misc import check_file_exist
        if not check_file_exist(fullname, true_md5):
            os.remove(fullname)
            if os.path.isdir(dirname):
                shutil.rmtree(dirname)
    if not os.path.isfile(os.path.join(dirname, "build", "lib", "libnccl.so")):
        # Decide before fetching anything. NCCL is only ever built for a
        # multi-process run on a machine that has GPUs, and these two checks
        # used to come *after* the download, so every CPU-only and every
        # single-process import paid for a trip to the mirror to fetch an
        # archive it then threw away.
        if core.get_device_count() == 0:
            return
        if not inside_mpi():
            return
        if not os.path.isfile(os.path.join(root_folder, filename)):
            LOG.i("Downloading nccl...")
        download_url_to_local(url, filename, root_folder, true_md5)

        import tarfile
        with tarfile.open(fullname, "r") as tar:
            safe_tar_extractall(tar, root_folder)

        LOG.i("installing nccl...")
        arch_flag = ""
        if len(flags.cuda_archs):
            arch_flag = cuda_arch_flags(flags.cuda_archs)
        run_cmd(f"CC=\"{cc_path}\" CXX=\"{cc_path}\" make -j8 src.build CUDA_HOME='{cuda_home}' NVCC_GENCODE='{arch_flag} --cudart=shared ' ", cwd=dirname)
    return dirname

def _skip_nccl_p2p_without_peer_access():
    """Turn off NCCL's p2p transport on boards that have no peer access at all.

    Consumer GeForce over PCIe cannot do direct GPU-to-GPU transfers --
    `nvidia-smi topo -p2p r` reports CNS for every pair -- and NCCL treats that as
    fatal at ncclCommInitRank ("unhandled cuda error") instead of falling back to
    shared memory. The decision has to be made here, before libnccl is loaded:
    setting it from the ops module's static initialiser is already too late.

    Only when NO pair on the machine can reach each other, so a box with working
    P2P keeps it, and never over an explicit setting from the operator.
    """
    if os.environ.get("NCCL_P2P_DISABLE") is not None:
        return
    try:
        runtime = None
        for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
            try:
                runtime = ctypes.CDLL(name)
                break
            except OSError:
                runtime = None
        if runtime is None:
            return
        count = ctypes.c_int(0)
        if runtime.cudaGetDeviceCount(ctypes.byref(count)) != 0 or count.value < 2:
            return
        for a in range(count.value):
            for b in range(count.value):
                if a == b:
                    continue
                reachable = ctypes.c_int(0)
                if (runtime.cudaDeviceCanAccessPeer(
                        ctypes.byref(reachable), a, b) == 0
                        and reachable.value):
                    return
        os.environ["NCCL_P2P_DISABLE"] = "1"
        LOG.v("no GPU pair supports peer access; setting NCCL_P2P_DISABLE=1")
    except Exception:
        # A probe that cannot run must not stop NCCL from being set up: the
        # init-time diagnostic in nccl_wrapper.cc still names the cure.
        pass


def _nccl_store_timeout():
    raw = os.environ.get("JT_RENDEZVOUS_TIMEOUT_S", "120")
    try:
        timeout = float(raw)
    except ValueError as error:
        raise ValueError(
            "JT_RENDEZVOUS_TIMEOUT_S must be a positive number"
        ) from error
    if timeout <= 0:
        raise ValueError("JT_RENDEZVOUS_TIMEOUT_S must be a positive number")
    return timeout


def _init_nccl_from_store(nccl_module, store=None):
    """Exchange NCCL's opaque bootstrap id through a real Store."""
    from jittor.distributed.store import FileStore, Store, TCPStore

    world_size = int(os.environ.get("JT_NCCL_WORLD_SIZE", "1"))
    world_rank = int(os.environ.get("JT_NCCL_RANK", "0"))
    if world_size < 1 or world_rank < 0 or world_rank >= world_size:
        raise RuntimeError(
            "NCCL(store): JT_NCCL_RANK={} is not a rank of a "
            "JT_NCCL_WORLD_SIZE={} job".format(world_rank, world_size)
        )
    timeout = _nccl_store_timeout()
    owned = store is None
    description = "the provided Store"
    try:
        if store is None:
            address = os.environ.get("MASTER_ADDR", "").strip()
            port = os.environ.get("MASTER_PORT", "").strip()
            rootinfo = os.environ.get("JT_NCCL_ROOTINFO_FILE", "").strip()
            if address or port:
                if not address or not port:
                    raise RuntimeError(
                        "NCCL store rendezvous requires both MASTER_ADDR and "
                        "MASTER_PORT"
                    )
                description = "TCPStore at {}:{}".format(address, port)
                store = TCPStore(
                    address, int(port), world_size, world_rank == 0,
                    timeout=timeout,
                )
            elif rootinfo:
                description = "FileStore at {}".format(rootinfo)
                store = FileStore(rootinfo, world_size, timeout=timeout)
            elif world_size == 1:
                store = Store(timeout=timeout)
                description = "the local singleton Store"
            else:
                raise RuntimeError(
                    "NCCL store rendezvous needs MASTER_ADDR/MASTER_PORT or "
                    "JT_NCCL_ROOTINFO_FILE"
                )

        unique_id_key = "jittor/nccl/world/unique_id"
        if world_rank == 0:
            store.set(unique_id_key, bytes(nccl_module.nccl_get_unique_id()))
        unique_id = store.get(unique_id_key)
        nccl_module.nccl_init_with_unique_id(list(unique_id))

        arrived = "jittor/nccl/world/initialized/{}".format(world_rank)
        store.set(arrived, b"1")
        store.wait([
            "jittor/nccl/world/initialized/{}".format(rank)
            for rank in range(world_size)
        ])
    except TimeoutError as error:
        raise RuntimeError(
            "NCCL store rendezvous timeout: rank {} waited {:.6g} s and "
            "timed out using {}: {}".format(
                world_rank, timeout, description, error)
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "NCCL store rendezvous failed for rank {} using {}: {}".format(
                world_rank, description, error)
        ) from error
    finally:
        if owned and store is not None:
            close = getattr(store, "close", None)
            if callable(close):
                close()


def setup_nccl(store=None):
    global nccl, nccl_ops, use_nccl
    use_nccl = os.environ.get("use_nccl", "1")=="1"
    nccl = None
    nccl_ops = None
    # NCCL is normally only built under MPI; also build it for the MPI-free
    # env/file rendezvous (JT_NCCL_WORLD_SIZE set by the torchrun-style launcher),
    # so NVIDIA multi-card DDP works without mpirun (mirrors the Ascend HCCL path).
    _jt_nccl_envfile = os.environ.get("JT_NCCL_WORLD_SIZE") is not None
    if not has_cuda or (not has_mpi and not _jt_nccl_envfile):
        use_nccl = False
        return
    if not use_nccl: return
    nccl_include_path = os.environ.get("nccl_include_path")
    nccl_lib_path = os.environ.get("nccl_lib_path")
    nccl_lib_name = None
    
    if nccl_lib_path is None or nccl_include_path is None:
        if cuda_wheel_stack:
            nccl_include_path = cuda_wheel_stack.include_dirs("nccl")[0]
            nccl_lib_path = cuda_wheel_stack.lib_dirs("nccl")[0]
            nccl_lib_name = cuda_wheel_stack.find_library("nccl")
        else:
            LOG.v("setup nccl...")
            # nccl_path decouple with cc_path
            nccl_path = os.path.join(jit_utils.home(), ".cache", "jittor", "nccl")

            make_cache_dir(nccl_path)
            nccl_home = install_nccl(nccl_path)
            if nccl_home is None: return
            nccl_include_path = os.path.join(nccl_home, "build", "include")
            nccl_lib_path = os.path.join(nccl_home, "build", "lib")
        
    # MPI-free env/file rendezvous: build NCCL ops even without MPI (compile the
    # MPI bootstrap branch out via -DJT_NCCL_NO_MPI). This is the no-mpirun path.
    _nccl_envfile = os.environ.get("JT_NCCL_WORLD_SIZE") is not None
    if not inside_mpi() and not _nccl_envfile:
        return

    _skip_nccl_p2p_without_peer_access()

    if nccl_lib_name is None:
        nccl_lib_name = search_file([nccl_lib_path], "libnccl.so")
    assert os.path.isdir(nccl_include_path)
    assert os.path.isdir(nccl_lib_path)
    assert os.path.isfile(nccl_lib_name), nccl_lib_name
    LOG.v(f"nccl_include_path: {nccl_include_path}")
    LOG.v(f"nccl_lib_path: {nccl_lib_path}")
    LOG.v(f"nccl_lib_name: {nccl_lib_name}")
    # We do not link manualy, link in custom ops
    if cuda_wheel_stack:
        preload_cuda_library("nccl", required=True)
    else:
        ctypes.CDLL(nccl_lib_name, dlopen_flags)

    nccl_src_dir = os.path.join(jittor_path, "extern", "cuda", "nccl")
    nccl_src_files = []
    for r, _, f in os.walk(nccl_src_dir):
        for fname in f:
            nccl_src_files.append(os.path.join(r, fname))

    # no MPI -> compile out the MPI_Bcast bootstrap. jittor's include scanner is
    # not #ifdef-aware, so it still must LOCATE mpi_wrapper.h (in the compiled-out
    # #else) and the <mpi.h> it includes; add their dirs + a stub mpi.h WITHOUT
    # any libmpi link, so no MPI install is required (#15).
    if _nccl_envfile and not inside_mpi():
        _mpi_inc = os.path.join(jittor_path, "extern", "mpi", "inc")
        _stub_inc = os.path.join(jittor_path, "extern", "cuda", "nccl", "nompi_inc")
        _mpi_flags = f' -DJT_NCCL_NO_MPI -I"{_mpi_inc}" -I"{_stub_inc}" '
    else:
        _mpi_flags = mpi_compile_flags
    nccl = compile_custom_ops(nccl_src_files,
        extra_flags=(
            f" -I\"{nccl_include_path}\" {_mpi_flags} "
            + cuda_library_link_flags("nccl", nccl_lib_name)
        ),
        return_module=True, dlopen_flags=os.RTLD_GLOBAL | os.RTLD_NOW,
        gen_name_="jittor_nccl_core")
    nccl_ops = nccl.ops
    # Build the communicator explicitly, here, instead of from a static
    # constructor that ran during the dlopen two lines up. Same reason HCCL
    # does (see hccl_init): a rendezvous that blocks or fails inside a static
    # constructor cannot report itself -- the exception unwinds through the
    # dynamic loader's C frames and aborts the process, so the operator sees
    # `terminate called after throwing` with no Python traceback. Called here
    # rather than at the end of import so nothing can observe nccl_ops before
    # its communicator exists. 8.09.
    #
    # THE LOCK MUST BE DROPPED FIRST. This blocks until every other rank
    # arrives -- MPI_Bcast of the unique id, or the shared-file rendezvous --
    # and the other ranks cannot arrive while this one holds jittor.lock,
    # because they need it to compile. compile_custom_ops releases the lock
    # around its own dlopen for exactly this reason ("unlock scope when
    # initialize"); taking the communicator build out of that dlopen took it
    # out of that release too, and a cold two-rank MPI run then deadlocked:
    # rank 0 spinning in MPI_Bcast holding the lock, rank 1 waiting for the
    # lock to build its core.
    with lock.unlock_scope():
        if _nccl_envfile:
            _init_nccl_from_store(nccl, store=store)
        else:
            nccl.nccl_init()
    LOG.vv("Get nccl_ops: "+str(dir(nccl_ops)))

def setup_hccl(no_mpi=False):
    ''' Build + load the HCCL collective ops module.

    no_mpi=False (default): the original MPI-based path. The communicator is
        bootstrapped with MPI_Bcast of the HCCL root info. Requires MPI to be
        set up (mpi_compile_flags) and a working MPI launch. This is the normal
        N-card path and is unchanged.
    no_mpi=True: an MPI-free path that bootstraps the communicator via an
        env/file rendezvous (JT_HCCL_* env vars), so libmpi is never loaded.
        Used on Ascend where the available OpenMPI build crashes coexisting
        with CANN. Built as a separate module so it never collides with the
        MPI build's cache/symbols.
    '''
    global hccl_ops, hccl_mod

    hccl_src_dir = os.path.join(jittor_path, "extern", "acl", "hccl")
    hccl_src_files = []
    for r, _, f in os.walk(hccl_src_dir):
        for fname in f:
            hccl_src_files.append(os.path.join(r, fname))

    ascend_home = os.environ.get("ASCEND_TOOLKIT_HOME")
    # CANN layouts differ across versions: some expose headers/libs under an
    # <arch>-linux/ subdir, others directly under the toolkit root. Probe both.
    hccl_include_path = None
    hccl_lib_name = None
    for prefix in ("aarch64-linux/", "x86_64-linux/", ""):
        inc = os.path.join(ascend_home, prefix, "include", "hccl")
        lib = os.path.join(ascend_home, prefix, "lib64", "libhccl.so")
        if hccl_include_path is None and os.path.isfile(os.path.join(inc, "hccl.h")):
            hccl_include_path = inc
        if hccl_lib_name is None and os.path.isfile(lib):
            hccl_lib_name = lib
    assert hccl_include_path is not None, f"hccl.h not found under {ascend_home}"
    assert hccl_lib_name is not None, f"libhccl.so not found under {ascend_home}"
    ctypes.CDLL(hccl_lib_name, dlopen_flags)

    # acl backend cc_flags already carry the acl/aclnn includes the hccl ops
    # need (acl_jittor.h, acl/acl.h); reuse them so the build sees those headers.
    from jittor import compiler
    extra = f" -I\"{hccl_include_path}\" "
    if no_mpi:
        # MPI-free build: env/file rendezvous, no libmpi linked. BUT jittor's
        # include scanner does not honor #ifdef, so it still needs to *locate*
        # mpi_wrapper.h (referenced in the guarded-out #else branch) and the
        # <mpi.h> it pulls in. Add the include dirs only -- NOT the link flags --
        # so nothing from libmpi is actually compiled in or linked.
        extra += " -DJT_HCCL_NO_MPI "
        mpi_inc = os.path.join(jittor_path, "extern", "mpi", "inc")
        extra += f" -I\"{mpi_inc}\" "
        if 'mpi_compile_flags' in globals() and mpi_compile_flags:
            # reuse just the -I parts of the mpi compile flags (for <mpi.h>)
            for tok in mpi_compile_flags.split():
                if tok.startswith("-I"):
                    extra += f" {tok} "
        gen_name = "jittor_hccl_core_nompi"
        LOG.i("setup_hccl: compiling hccl ops (MPI-free)...")
    else:
        # Normal MPI path: needs the MPI headers/flags for MPI_Bcast rendezvous.
        extra += f" {mpi_compile_flags} "
        gen_name = "jittor_hccl_core"
        LOG.i("setup_hccl: compiling hccl ops (MPI)...")
    extra += getattr(compiler, "cc_flags", "")
    hccl = compile_custom_ops(hccl_src_files,
        extra_flags=extra,
        return_module=True, dlopen_flags=os.RTLD_GLOBAL | os.RTLD_NOW,
        gen_name_=gen_name)
    LOG.i("setup_hccl: hccl ops compiled+loaded")
    hccl_ops = hccl.ops
    hccl_mod = hccl
    LOG.vv("Get hccl_ops: "+str(dir(hccl_ops)))

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

# Which environment variables mean "this process was started by an MPI
# launcher". Mirrored by detect_inside_mpi() in extern/mpi/src/mpi_wrapper.cc,
# which must answer the same question in C++ before MPI_Init; both lists are
# pinned together by tests/distributed/test_mpi_launcher_env.py.
#
# Only OMPI_COMM_WORLD_SIZE used to be recognized, so MPICH, Intel MPI, MVAPICH
# and srun all fell through to a silent single-card run. 6.B15.
_MPI_LAUNCHER_VARS = (
    "OMPI_COMM_WORLD_SIZE",   # Open MPI (mpirun / orterun / prterun)
    "PMI_SIZE",               # MPICH, Intel MPI (mpiexec.hydra)
    "MV2_COMM_WORLD_SIZE",    # MVAPICH2
    "PMIX_RANK",              # PMIx-based launchers
)

# Slurm is different: srun is routinely used to start ordinary single-task jobs
# that have nothing to do with MPI, so require an actual multi-task allocation.
_MPI_LAUNCHER_SIZE_VARS = ("SLURM_NTASKS", "SLURM_NPROCS")


def inside_mpi():
    """Whether this process is one rank of an MPI job.

    ``JT_MPI`` overrides in either direction: ``0`` to stay single-process under
    a launcher, ``1`` to declare an MPI job started by a launcher we do not
    recognize.
    """
    forced = os.environ.get("JT_MPI")
    if forced:
        return forced != "0"
    if any(v in os.environ for v in _MPI_LAUNCHER_VARS):
        return True
    for var in _MPI_LAUNCHER_SIZE_VARS:
        value = os.environ.get(var, "")
        if value.strip().isdigit() and int(value) > 1:
            return True
    return False

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
    mpi_compile_flags = jit_utils.probe.cached(
        "mpi_compile_flags:" + mpicc_path, [mpicc_path],
        lambda: run_cmd(mpicc_path+" --showme:compile"))
    mpi_link_flags = jit_utils.probe.cached(
        "mpi_link_flags:" + mpicc_path, [mpicc_path],
        lambda: run_cmd(mpicc_path+" --showme:link"))
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
    # On Ascend, the CANN libraries are dlopened RTLD_GLOBAL before this module.
    # Loading the MPI ops RTLD_GLOBAL too lets the linker interpose libmpi's
    # internal symbols (opal_*/orte_*/pmix_*) with same-named CANN symbols,
    # producing a wild jump (SIGBUS) inside MPI_Init. RTLD_DEEPBIND makes the
    # module prefer its own (libmpi) symbols first, avoiding the collision.
    # On Ascend, MPI must be brought up (mpi4py) BEFORE the CANN libs load to
    # avoid an ABI/symbol clash; see jittor/__init__.py. With MPI already
    # initialized, the normal RTLD_GLOBAL load of our ops module is safe.
    mpi_dlopen_flags = os.RTLD_GLOBAL | os.RTLD_NOW
    mpi = compile_custom_ops(mpi_src_files,
        extra_flags=f" {mpi_flags} ", return_module=True,
        dlopen_flags=mpi_dlopen_flags, gen_name_="jittor_mpi_core")
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
# Importing Jittor must not probe/import Torch as a side effect. Keep the old
# loader workaround as an explicit opt-in for applications that need it.
FIX_TORCH_ERROR = os.environ.get("FIX_TORCH_ERROR", "0") == "1"
if FIX_TORCH_ERROR:
    from jittor_utils import dirty_fix_pytorch_runtime_error
    dirty_fix_pytorch_runtime_error()

cudnn = cublas = curand = cufft = cusparse = cutt = None

# Env/file-based distributed mode (MPI-free) for Ascend. The launcher sets
# JT_HCCL_* and spawns one plain process per rank. We use this because the
# available OpenMPI build crashes coexisting with CANN in one process. In this
# mode we must NOT load jittor's MPI op module / libmpi at all -- so disable
# the MPI setup before it runs. The normal (MPI) N-card path is untouched when
# JT_HCCL_WORLD_SIZE is not set.
_jt_hccl_ws = os.environ.get("JT_HCCL_WORLD_SIZE")
_jt_hccl_no_mpi = _jt_hccl_ws is not None
# Same MPI-free env/file rendezvous for NVIDIA/NCCL (JT_NCCL_*), so DDP works
# without mpirun on BOTH backends (torchrun-style, one plain process per rank).
_jt_nccl_ws = os.environ.get("JT_NCCL_WORLD_SIZE")
_jt_nccl_no_mpi = _jt_nccl_ws is not None
if _jt_hccl_no_mpi or _jt_nccl_no_mpi:
    os.environ["use_mpi"] = "0"   # make setup_mpi() a no-op (no libmpi load)

setup_mpi()


def _resolve_distributed_state():
    """The one place (rank, world_size, in_mpi) is decided. 6.B15.

    There is exactly one authority per bootstrap path, and Python reads it
    rather than deriving its own answer:

    * MPI: the C++ globals, filled by MPI_Comm_rank/MPI_Comm_size in
      mpi_wrapper.cc and read back through mpi.world_rank()/world_size().
    * MPI-free env/file rendezvous (JT_HCCL_* / JT_NCCL_*): the launcher's
      environment, which nccl_wrapper.cc / hccl_wrapper.cc read into the very
      same C++ globals, so the two sides cannot disagree.

    Keeping the three branches in one function is the point: the previous shape
    had them as three separate assignments to module globals, and any branch
    that forgot one left C++ believing rank 0 while Python believed rank 2.
    """
    if _jt_hccl_no_mpi:
        # in_mpi True so the optimizer takes the distributed path.
        return True, int(os.environ.get("JT_HCCL_RANK", "0")), int(_jt_hccl_ws)
    if _jt_nccl_no_mpi:
        return True, int(os.environ.get("JT_NCCL_RANK", "0")), int(_jt_nccl_ws)
    if in_mpi:
        return True, mpi.world_rank(), mpi.world_size()
    return False, 0, 1


in_mpi, rank, world_size = _resolve_distributed_state()


_DISTRIBUTED_STATE_NAMES = ("in_mpi", "rank", "world_size")


def distributed_state_getattr(name):
    """Module ``__getattr__`` serving the distributed identity from its owner.

    Bound as ``__getattr__`` in both ``jittor/__init__.py`` and
    ``jittor/_runtime/core_api.py``, so ``jt.rank`` and ``core_api.in_mpi`` are
    read channels rather than copies taken at import time. They used to be
    copies -- ``jt.rank`` from the import list, ``core_api.in_mpi`` via
    ``from jittor import *`` -- and anything that later corrected
    ``compile_extern.rank`` (the torch NCCL installer does exactly that) left
    them stale with no error. ``Module.mpi_param_broadcast()`` read the stale
    one and silently did nothing, so every rank kept its own random init. 6.B15.

    Assigning any of these names on either module would put an entry in that
    module's ``__dict__``, which shadows ``__getattr__`` permanently and brings
    the snapshot straight back. Write to ``compile_extern`` instead.
    """
    if name in _DISTRIBUTED_STATE_NAMES:
        return globals()[name]
    raise AttributeError(name)


def distributed_requested():
    """Why this process believes it is one rank of a multi-rank job, or None.

    Only the launcher knows this, and it tells us through the environment
    before any collective backend is brought up. That ordering is the whole
    point: once distributed has been *requested*, a backend that then fails to
    initialize must be a hard error. Falling back to single card turns one
    N-card job into N independent single-card jobs which look like they are
    training correctly and are completely wrong. 6.B04.

    Returns None when nothing asked for distributed, so a plain single-process
    run stays silent exactly as before.
    """
    for var in ("JT_HCCL_WORLD_SIZE", "JT_NCCL_WORLD_SIZE"):
        value = os.environ.get(var)
        if value and value.strip().isdigit() and int(value) > 1:
            return "{}={}".format(var, value)
    if in_mpi and world_size > 1:
        return "MPI world_size={}".format(world_size)
    return None


def check_rank_agrees_with_cxx():
    """C++ and Python must report the same rank/world_size. 6.B15.

    The C++ side is what the collective operators actually use (mpi_world_rank
    and friends); the Python side is what user code and the optimizer read. A
    process where C++ thinks it is rank 0 and Python thinks it is rank 2 does
    not crash -- it exchanges tensors with the wrong peers and trains something
    meaningless. Cheap to check once at import, so check it.

    Only checkable where both sides exist, i.e. the MPI path. On the env/file
    rendezvous the C++ globals are filled from the same launcher variables this
    module reads, inside nccl_wrapper.cc / hccl_wrapper.cc.
    """
    if mpi is None or _jt_hccl_no_mpi or _jt_nccl_no_mpi:
        return
    # First: did both sides recognize the launcher? This is where the two
    # detection lists (inside_mpi() here, detect_inside_mpi() in
    # mpi_wrapper.cc) would show up as having drifted apart.
    cxx_enabled = bool(mpi.get_state())
    if cxx_enabled != bool(in_mpi):
        raise RuntimeError(
            "distributed state is inconsistent: the C++ MPI layer is {} while "
            "Python believes in_mpi={}. The launcher-detection lists in "
            "compile_extern.inside_mpi() and detect_inside_mpi() in "
            "extern/mpi/src/mpi_wrapper.cc have drifted apart.".format(
                "enabled" if cxx_enabled else "disabled", in_mpi))
    if not in_mpi:
        return
    cxx = (mpi.world_rank(), mpi.world_size())
    py = (rank, world_size)
    if cxx != py:
        raise RuntimeError(
            "distributed state is inconsistent: C++ reports rank/world_size {} "
            "while Python reports {}. The collectives would use the C++ values "
            "and user code the Python ones.".format(cxx, py))


def check_distributed_backend_ready():
    """Fail loudly if distributed was requested but no collective backend came up.

    Every silent-degradation path funnels here: setup_nccl()'s several early
    returns (no CUDA, no nccl install, no rendezvous), a failed setup_hccl(),
    an MPI module that did not load. Any one of them leaves the process
    reporting world_size>1 with nothing to communicate through.
    """
    check_rank_agrees_with_cxx()
    reason = distributed_requested()
    if reason is None:
        return
    backends = [name for name, ops in (("hccl", hccl_ops),
                                       ("nccl", nccl_ops),
                                       ("mpi", mpi_ops)) if ops is not None]
    if backends:
        LOG.v("distributed requested ({}), collective backends: {}".format(
            reason, ", ".join(backends)))
        return
    raise RuntimeError(
        "distributed was requested ({}) but no collective backend could be "
        "initialized (hccl/nccl/mpi ops are all unavailable). Every rank would "
        "run as an independent single-card job and silently produce wrong "
        "results, so this is a hard error. Check that the backend's runtime is "
        "installed and visible to this process.".format(reason))

# Enable the device collective backend used for multi-card data parallel.
# HCCL on Ascend, NCCL on CUDA.
#
# TODO(multi-card refactor): there are now two HCCL bootstrap paths --
#   (1) MPI: setup_hccl(no_mpi=False) + MPI_Bcast rendezvous (the original
#       N-card path; requires a working, CANN-compatible MPI).
#   (2) MPI-free: setup_hccl(no_mpi=True) + JT_HCCL_* env/file rendezvous
#       (added for Ascend where the available OpenMPI crashes with CANN).
# Long term these should be unified behind a single launcher + a pluggable
# rendezvous (env/file, TCPStore, or HCCL rank-table via HcclCommInitClusterInfo
# -- the device IPs are available via `hccn_tool -i N -ip -g`). For now they are
# kept as distinct, separately-compiled modules so neither can regress the
# other. The MPI path's behavior is unchanged when JT_HCCL_WORLD_SIZE is unset.
hccl_ops = None
hccl_mod = None
import jittor.compiler as compiler
_want_hccl = (getattr(compiler, "has_acl", 0) and
              (_jt_hccl_no_mpi or (in_mpi and has_mpi)))
if _want_hccl:
    try:
        setup_hccl(no_mpi=_jt_hccl_no_mpi)
        # Initialize the communicator now, after import is otherwise complete.
        # (Doing this in a static ctor at dlopen time hung.)
        LOG.i("setup_hccl: initializing HCCL communicator...")
        # Outside the build lock, for the same reason as nccl_init above: this
        # waits for the other ranks, and they need the lock to get here.
        with lock.unlock_scope():
            hccl_mod.hccl_init()
        LOG.i("setup_hccl: HCCL communicator ready")
    except Exception as e:
        if distributed_requested():
            # Distributed was explicitly requested by a launcher. Continuing
            # here would leave every rank running as an independent single-card
            # job -- they train, they print sensible losses, and nothing is ever
            # exchanged between them. 6.B04.
            raise RuntimeError(
                "HCCL setup failed, but distributed was requested ({}). "
                "Refusing to silently fall back to single card.".format(
                    distributed_requested())) from e
        LOG.w("HCCL setup failed, multi-card on Ascend disabled, msg:", e)

setup_nccl()
setup_cutt()

# try:
setup_mkl()
# except Exception as e:
#     LOG.w("MKL install failed, msg:", e)

setup_cuda_extern()

# install backend extern library
for mod in jit_utils.backends:
    if mod.install_extern():
        break

# Last gate: distributed was requested -> a collective backend must exist.
check_distributed_backend_ready()
