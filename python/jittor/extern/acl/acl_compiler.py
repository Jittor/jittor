# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os
import glob
import shutil
# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/aoe/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/stub:/usr/local/Ascend/ascend-toolkit/latest/tools/tikicpulib/lib/Ascend910A:/usr/local/Ascend/ascend-toolkit/latest/toolkit/tools/simulator/Ascend910A/lib:/opt/AXESMI/lib64:/usr/local/Ascend/driver/lib64/driver/
# export PYTHONPATH=/home/cjld/new_jittor/jittor/python
# export tikcc_path=g++

# conda activate cann
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# export PYTHONPATH=/home/cjld/new_jittor/jittor/python:/home/cjld/new_jittor/jittor/my/jtorch/python:$PYTHONPATH
# export TASK_QUEUE_ENABLE=0
# python3 -m pytest tests/backends/npu/test_acl.py -k array
# jittor: conda activate cann && source /usr/local/Ascend/ascend-toolkit/set_env.sh && PYTHONPATH=/home/cjld/new_jittor/jittor/python:/home/cjld/new_jittor/jittor/my/jtorch/python:$PYTHONPATH && cd /home/cjld/new_jittor/jittor/my/mm_benchmark
# python3 -m pytest tests/backends/npu/test_acl.py -k test_sum
# export ASCEND_SLOG_PRINT_TO_STDOUT=0
# ASCEND_GLOBAL_LOG_LEVEL
# export DUMP_GE_GRAPH=1
# export DUMP_GRAPH_LEVEL=1

# build pytorch-npu
# bash ./ci/build.sh
# python3 -m pip install ./dist/torch_npu-1.11.0.post1-cp37-cp37m-linux_x86_64.whl  --force-reinstall
# pytorch: conda activate cann && source /usr/local/Ascend/ascend-toolkit/set_env.sh && export TASK_QUEUE_ENABLE=0  && cd /home/cjld/new_jittor/jittor/my/mm_benchmark
# python3 ./mm_bench_pt_npu.py


def configure(context):
    """Return ACL build inputs without mutating the compiler or environment."""
    config = context.config
    requested_compiler = os.environ.get("tikcc_path", "ccec")
    tikcc_path = shutil.which(requested_compiler) if requested_compiler else None
    if not tikcc_path:
        raise RuntimeError(
            "ACL was selected but its compiler was not found; set tikcc_path "
            "or put ccec on PATH"
        )
    ascend_toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME", "")
    if not ascend_toolkit_home or not os.path.isdir(ascend_toolkit_home):
        raise RuntimeError(
            "ACL requires ASCEND_TOOLKIT_HOME to name an existing CANN toolkit directory"
        )
    if context.load_library is None:
        raise RuntimeError("ACL configuration requires a load_library service")
    acl_compiler_home = os.path.dirname(__file__)
    cc_files = sorted(glob.glob(acl_compiler_home + "/**/*.cc",
                                recursive=True))
    cc_files2 = []
    extra_core_files = list(config.extra_core_files)
    for name in cc_files:
        # Skip files in hccl directory
        if "hccl" in name:
            continue
        # if "acl_op_exec" in name or "_op_acl.cc" in name:
        if "acl_op_exec" in name or "_op_acl.cc" in name or "utils.cc" in name:
            extra_core_files.append(name)
        else:
            cc_files2.append(name)
    cc_files = cc_files2
    cc_flags = f" -MD -DHAS_CUDA -DIS_ACL  \
    -I{ascend_toolkit_home}/include/ \
    -I{ascend_toolkit_home}/include/acl/ \
    -I{ascend_toolkit_home}/include/aclnn/ \
    -I{ascend_toolkit_home}/include/aclnnop/ \
    -I{acl_compiler_home} -lascendcl -lacl_op_compiler \
    -I{acl_compiler_home}/aclnn \
    -I{acl_compiler_home}/aclops \
    -L{ascend_toolkit_home}/lib64/"

    cc_flags += " -llibascendcl "
    cc_flags += " -llibnnopbase "
    cc_flags += " -llibopapi "

    library = context.load_library("libascendcl.so", os.RTLD_NOW | os.RTLD_GLOBAL)
    # Compile the registration entry point as a normal backend module.  ACL
    # kernels are native provider sources; no whole-tree source rewriting is
    # performed here.  In particular, do not expose a converter resource:
    # generated CUDA source is either handled by a registered ACL operation or
    # rejected explicitly by the provider.
    mod = context.compile_module(
        '''
#include "common.h"
namespace jittor {
// @pyjt(init_acl_ops)
void init_acl_ops();
}''', config.cc_flags + " " + " ".join(cc_files) + cc_flags)
    final_flags = config.cc_flags + cc_flags
    return config.evolve(
        backend="acl", has_acl=True, has_cuda=True, is_cuda=False,
        has_rocm=False, has_corex=False,
        tikcc_path=tikcc_path, nvcc_path=tikcc_path,
        cc_flags=final_flags, nvcc_flags=final_flags.replace("-std=c++14", ""),
        setup_fake_cuda_lib=True, extra_core_files=tuple(extra_core_files),
        environment={**config.environment, "use_mkl": "0"},
        resources={**config.resources, "acl_initializer": mod, "acl_library": library},
    )


def install(context):
    return configure(context)


def install_extern(context):
    return False


def post_process(context):
    if context.config.has_acl:
        context.config.resources["acl_initializer"].init_acl_ops()
