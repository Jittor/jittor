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
    "src/acl_fused_ascendc.cc",
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
    "kernels/native/multinomial_op_acl.cc",
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
    "kernels/native/sort_op_acl.cc",
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
        kernel_source_roots=(), kernel_flag_filter=(),
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
        initializer = context.config.resources.get("acl_initializer")
        if initializer is None or not callable(getattr(initializer, "init_acl_ops", None)):
            raise RuntimeError("ACL initializer resource is unavailable")
        if context.native_core is None:
            raise RuntimeError("ACL device metadata requires the loaded native core")
        initializer.init_acl_ops()
        info = getattr(initializer, "_device_info", None)
        if info is None:
            # configure() runs before the native provider library is loaded.
            # Binding its symbols there would rely on RTLD_LAZY. Compile this
            # small binding only after core load, using the injected service.
            info = context.compile_module('''
#include "acl_runtime.h"
namespace jittor {
// @pyjt(acl_device_name)
string acl_device_name(int device);
}''', context.config.cc_flags)
            if not callable(getattr(info, "acl_device_name", None)):
                raise RuntimeError("ACL device metadata binding is incomplete")
            initializer._device_info = info
        elif not callable(getattr(info, "acl_device_name", None)):
            raise RuntimeError("ACL device metadata resource is invalid")
        rng = getattr(initializer, "_rng_info", None)
        rng_names = ("acl_get_rng_state", "acl_set_rng_state", "acl_get_rng_state_all",
                     "acl_set_rng_state_all", "acl_manual_seed", "acl_manual_seed_all",
                     "acl_initial_seed")
        if rng is None:
            # Same post-core native binding boundary as device metadata. The
            # provider owns all mutable RNG state, never this Python module.
            rng = context.compile_module('''
#include "acl_runtime.h"
namespace jittor {
// @pyjt(acl_get_rng_state)
string acl_get_rng_state(int device);
// @pyjt(acl_set_rng_state)
void acl_set_rng_state(int device, const string& blob);
// @pyjt(acl_get_rng_state_all)
vector<string> acl_get_rng_state_all();
// @pyjt(acl_set_rng_state_all)
void acl_set_rng_state_all(const vector<string>& blobs);
// @pyjt(acl_manual_seed)
void acl_manual_seed(int device, int64 seed);
// @pyjt(acl_manual_seed_all)
void acl_manual_seed_all(int64 seed);
// @pyjt(acl_initial_seed)
int64 acl_initial_seed(int device);
}''', context.config.cc_flags)
            if not all(callable(getattr(rng, name, None)) for name in rng_names):
                raise RuntimeError("ACL RNG binding is incomplete")
            initializer._rng_info = rng
        elif not all(callable(getattr(rng, name, None)) for name in rng_names):
            raise RuntimeError("ACL RNG resource is invalid")


def get_device_name(index):
    """Query a visible device's real SoC name without changing device policy."""
    import operator
    from jittor.build import compiler

    if isinstance(index, bool):
        raise TypeError("ACL device index must be an integer, not bool")
    index = operator.index(index)
    if not compiler.build_config.has_acl:
        raise RuntimeError("ACL device names require the ACL backend")
    initializer = compiler.build_config.resources.get("acl_initializer")
    info = getattr(initializer, "_device_info", None)
    query = getattr(info, "acl_device_name", None)
    if not callable(query):
        raise RuntimeError("ACL device metadata was not initialized after core load")
    return query(index)


def _rng_info():
    from jittor.build import compiler
    if not compiler.build_config.has_acl:
        raise RuntimeError("ACL RNG requires the ACL backend")
    initializer = compiler.build_config.resources.get("acl_initializer")
    rng = getattr(initializer, "_rng_info", None)
    if rng is None:
        raise RuntimeError("ACL RNG was not initialized after core load")
    return rng


def get_rng_state(device):
    """Opaque same-runtime ACL stream state, not cuRAND or torch_npu bytes."""
    return _rng_info().acl_get_rng_state(device)


def set_rng_state(device, state):
    return _rng_info().acl_set_rng_state(device, state)


def get_rng_state_all():
    return _rng_info().acl_get_rng_state_all()


def set_rng_state_all(states):
    return _rng_info().acl_set_rng_state_all(states)


def manual_seed(device, seed):
    return _rng_info().acl_manual_seed(device, seed)


def manual_seed_all(seed):
    return _rng_info().acl_manual_seed_all(seed)


def initial_seed(device):
    return _rng_info().acl_initial_seed(device)
