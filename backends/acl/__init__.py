# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import os
import shutil
from jittor_utils.env_config import build_env
from jittor_utils.build_config import BuildConfig, BuildContext, BuildSource
# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/aoe/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/stub:/usr/local/Ascend/ascend-toolkit/latest/tools/tikicpulib/lib/Ascend910A:/usr/local/Ascend/ascend-toolkit/latest/toolkit/tools/simulator/Ascend910A/lib:/opt/AXESMI/lib64:/usr/local/Ascend/driver/lib64/driver/
# export PYTHONPATH=/home/cjld/new_jittor/jittor/python
# export JT_BUILD_TIKCC_PATH=g++

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


# Preserve the previous provider's sorted source order and compilation split.
# The provider runtime in src/backend.cc and src/workspace.cc is compiled by
# BuildConfig separately; it must never enter this registration module.
REGISTRATION_SOURCES = (
    "src/acl_error_code.cc",
    "src/acl_jittor.cc",
    "src/aclnn.cc",
)
PROVIDER_RUNTIME_SOURCES = (
    "src/backend.cc",
    "src/foreach_coefficients.cc",
    "src/workspace.cc",
)
CORE_SOURCES = (
    "src/acl_op_exec.cc",
    "kernels/native/adamw_op_acl.cc",
    "kernels/native/arg_reduce_op_acl.cc",
    "kernels/native/base_op_acl.cc",
    "kernels/native/binary_op_acl.cc",
    "kernels/native/bmm_op_acl.cc",
    "kernels/native/clamp_op_acl.cc",
    "kernels/native/concat_op_acl.cc",
    "kernels/native/conv_op_acl.cc",
    "kernels/native/cross_entropy_loss_op_acl.cc",
    "kernels/native/cumsum_op_acl.cc",
    "kernels/native/dropout_op_acl.cc",
    "kernels/native/embedding_op_acl.cc",
    "kernels/native/expand_op_acl.cc",
    "kernels/native/flashattention_op_acl.cc",
    "kernels/native/flip_op_acl.cc",
    "kernels/native/floor_op_acl.cc",
    "kernels/native/gather_scatter_op_acl.cc",
    "kernels/native/foreach_op_acl.cc",
    "kernels/native/fused_sgd_op_acl.cc",
    "kernels/native/gelu_op_acl.cc",
    "kernels/native/getitem_op_acl.cc",
    "kernels/native/index_op_acl.cc",
    "kernels/native/matmul_op_acl.cc",
    "kernels/native/nantonum_op_acl.cc",
    "kernels/native/native_indexing_op_acl.cc",
    "kernels/native/norms_op_acl.cc",
    "kernels/native/pool_op_acl.cc",
    "kernels/native/random_op_acl.cc",
    "kernels/native/reduce_op_acl.cc",
    "kernels/native/relu_op_acl.cc",
    "kernels/native/roll_op_acl.cc",
    "kernels/native/rope_op_acl.cc",
    "kernels/native/setitem_op_acl.cc",
    "kernels/native/sigmoid_op_acl.cc",
    "kernels/native/silu_op_acl.cc",
    "kernels/native/softmax_op_acl.cc",
    "kernels/native/stack_op_acl.cc",
    "kernels/native/ternary_op_acl.cc",
    "kernels/native/transpose_op_acl.cc",
    "kernels/native/triu_op_acl.cc",
    "kernels/native/truth_reduce_op_acl.cc",
    "kernels/native/unary_op_acl.cc",
    "kernels/native/upsample_op_acl.cc",
    "kernels/native/utils.cc",
    "kernels/native/where_op_acl.cc",
)


def configure(context: BuildContext) -> BuildConfig:
    """Return ACL build inputs without mutating the compiler or environment."""
    config = context.config
    requested_compiler = build_env("tikcc_path", "ccec")
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
    cc_files = [os.path.join(acl_compiler_home, name)
                for name in REGISTRATION_SOURCES]
    extra_core_files = list(config.extra_core_files) + [
        os.path.join(acl_compiler_home, name) for name in CORE_SOURCES
    ]
    cc_flags = f" -MD -DHAS_ACCELERATOR -DHAS_CUDA -DIS_ACL  \
    -I{ascend_toolkit_home}/include/ \
    -I{ascend_toolkit_home}/include/acl/ \
    -I{ascend_toolkit_home}/include/aclnn/ \
    -I{ascend_toolkit_home}/include/aclnnop/ \
    -I{acl_compiler_home}/include -lascendcl -lacl_op_compiler \
    -I{acl_compiler_home}/include/aclnn \
    -I{acl_compiler_home}/include/aclops \
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
#include "core/common.h"
namespace jittor {
// @pyjt(init_acl_ops)
void init_acl_ops();
}''', config.cc_flags + " " + " ".join(cc_files) + cc_flags)
    final_flags = config.cc_flags + cc_flags
    provider_sources = tuple(
        BuildSource(os.path.join(acl_compiler_home, name), flags=cc_flags)
        for name in PROVIDER_RUNTIME_SOURCES
    )
    return config.evolve(
        backend="acl", has_acl=True, has_cuda=True, is_cuda=False,
        has_accelerator=True, has_rocm=False, has_corex=False,
        backend_sources=config.backend_sources + provider_sources,
        tikcc_path=tikcc_path, nvcc_path=tikcc_path,
        cc_flags=final_flags, nvcc_flags=final_flags.replace("-std=c++14", ""),
        # A generated ACL operator is host C++ that calls aclnn, not device
        # source for ccec, so the JIT accelerator compiler is the host compiler
        # carrying the full ACL flags. Without this the accelerator branch of
        # jit_compiler::compile inherits the CPU defaults and emits a command
        # with no include paths at all.
        kernel_compiler=config.cc_path, kernel_language="cxx",
        kernel_compile_flags=final_flags, kernel_source_suffix=".cc",
        kernel_device_link=False,
        # No fake CUDA libraries: that path compiles backends/cuda/kernels/<lib>
        # sources, which are real CUDA/cuDNN translation units. They only ever
        # built under ACL because the 1.x provider rewrote every jittor source
        # through process_acl(); this provider exposes no converter, and ACL
        # publishes its own conv/matmul kernels from kernels/install.py.
        setup_fake_cuda_lib=False, extra_core_files=tuple(extra_core_files),
        environment={**config.environment, "use_mkl": "0"},
        resources={**config.resources, "acl_initializer": mod, "acl_library": library},
    )


def install(context: BuildContext) -> BuildConfig:
    return configure(context)


def install_extern(context: BuildContext) -> bool:
    return False


def post_process(context: BuildContext) -> None:
    if context.config.has_acl:
        context.config.resources["acl_initializer"].init_acl_ops()
