# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers:
#   Dun Liang <randonlang@gmail.com>.
#   Meng-Hao Guo <guomenghao1997@gmail.com>
#
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

__version__ = '1.3.11.0'

import os as _os
import sys as _sys

from .compat.shim.preflight import (
    configure_torch_math_flags as _configure_compat_math_flags,
    is_truthy as _compat_is_truthy,
    prepare_import_environment as _prepare_compat_import,
)

_compat_preflight_result = _prepare_compat_import(
    argv=_sys.argv,
    environ=_os.environ,
)

from jittor_utils import lock as _lock
from jittor_utils import limit_openmp_to_physical_cores as _limit_openmp


_NATIVE_CORE_EXPORTS = (
    "DumpGraphs", "Flags", "MemInfo", "NanoString", "NanoVector",
    "RingBuffer", "Var", "ZipFile", "binary_dtype_infer", "clean_graph",
    "cleanup", "clear_trace_data", "current_device", "display_max_memory_info",
    "display_memory_info", "dump_all_graphs", "dump_trace_data", "fetch_sync",
    "gc", "get_device_count", "get_max_memory_info", "get_mem_info",
    "get_seed", "grad", "grad_optional", "graph_check", "hash",
    "jt_init_subprocess", "lock_acquire", "lock_is_held", "lock_release",
    "migrate_all_to_cpu", "number_of_hold_vars", "number_of_lived_ops",
    "number_of_lived_vars", "op_compiler", "ops", "print_trace", "profiler",
    "reuse_np_array", "seed", "set_device", "set_lock_fd", "set_seed",
    "sync", "sync_all", "tape_together", "ternary_out_hint", "tests",
    "wrap_var_addr",
)

_NATIVE_OP_EXPORTS = (
    "abs", "acos", "acosh", "add", "all_", "any_", "arccos", "arccosh",
    "arcsin", "arcsinh", "arctan", "arctanh", "arg_reduce", "argsort",
    "array", "array_", "asin", "asinh", "atan", "atanh", "bfloat16",
    "binary", "bitwise_and", "bitwise_not", "bitwise_or", "bitwise_xor",
    "bool", "broadcast", "broadcast_var", "candidate", "cast", "ceil",
    "ceil_int", "clone", "code", "conj", "copy", "cos", "cosh",
    "device_copy", "div", "divide", "empty", "equal", "erf", "erfinv",
    "exp", "fetch", "float16", "float32", "float64", "floor",
    "floor_divide", "floor_int", "fuse_transpose", "fused_adamw", "getitem",
    "greater", "greater_equal", "index", "index_var", "int16", "int32",
    "int64", "int8", "left_shift", "less", "less_equal", "log",
    "logical_and", "logical_not", "logical_or", "logical_xor", "max",
    "maximum", "mean", "min", "minimum", "mod", "mul", "multiply",
    "negative", "not_equal", "numpy_code", "pow", "prod", "product",
    "random", "reduce", "reduce_add", "reduce_bitwise_and",
    "reduce_bitwise_or", "reduce_bitwise_xor", "reduce_logical_and",
    "reduce_logical_or", "reduce_logical_xor", "reduce_maximum",
    "reduce_minimum", "reduce_multiply", "reindex", "reindex_reduce",
    "reindex_var", "reinterpret_view", "reshape", "right_shift", "round",
    "round_int", "safe_clip", "setitem", "sigmoid", "sin", "sinh", "sqrt",
    "sub", "subtract", "sum", "tan", "tanh", "tape", "ternary",
    "transpose", "uint16", "uint32", "uint64", "uint8", "unary", "where",
)


def _publish(module, names):
    namespace = globals()
    for name in names:
        namespace[name] = getattr(module, name)

# Must run before anything links OpenMP: the runtime reads OMP_NUM_THREADS when
# it starts, and the default is one thread per *logical* CPU.
_limit_openmp(_os.environ)

# On Ascend (NPU), bringing MPI up via mpi4py BEFORE the CANN libraries are
# loaded avoids a fatal ABI/symbol clash (CANN's globally-loaded libs interpose
# OpenMPI's internal symbols, causing a wild-jump SIGBUS inside MPI_Init when
# our own mpi op module is loaded later). Doing it here, first thing, is the
# only point early enough. Guarded to multi-process launches (mpirun sets
# OMPI_COMM_WORLD_SIZE); harmless/no-op otherwise. jittor's mpi module detects
# the already-initialized MPI and skips its own MPI_Init.
if _os.environ.get("OMPI_COMM_WORLD_SIZE") and _os.environ.get("use_mpi", "1") != "0":
    try:
        import mpi4py
        mpi4py.rc.initialize = True
        mpi4py.rc.finalize = False
        from mpi4py import MPI as _MPI  # triggers MPI_Init before CANN loads
    except Exception as _e:
        print("jittor: mpi4py pre-init skipped:", _e)

with _lock.lock_scope():
    ori_int = int
    ori_float = float
    ori_bool = bool
    from . import compiler
    from .compiler import LOG, has_cuda
    from .compiler import compile_custom_ops, compile_custom_op
    import jittor_core
    import jittor_core as core
    _publish(jittor_core, _NATIVE_CORE_EXPORTS)
    _publish(jittor_core.ops, _NATIVE_OP_EXPORTS)
    _core_profiler = core.profiler
    from . import compile_extern
    from .compile_extern import mkl_ops, mpi, mpi_ops
    # in_mpi / rank / world_size are deliberately NOT imported here. Importing
    # them bound a snapshot of compile_extern.rank taken at import time, and
    # anything that later corrected compile_extern.rank -- the torch NCCL
    # installer does exactly that -- left jt.rank stale with no error.
    # distributed_state_getattr serves all three from their single owner
    # instead, and it is installed before the submodules below are imported
    # because several of them read jt.rank at import time.
    #
    # Assigning jt.rank / jt.world_size / jt.in_mpi anywhere would put an entry
    # in this module's __dict__, which shadows __getattr__ permanently and
    # brings the stale copy straight back. Write to compile_extern. 6.B15.
    from .compile_extern import distributed_state_getattr as __getattr__
    if core.get_device_count() == 0:
        has_cuda = compile_extern.has_cuda = compiler.has_cuda = False
    if has_cuda:
        from .compile_extern import cudnn, curand, cublas, cufft, cusparse
        from .init_cupy import numpy2cupy
    else:
        # No CUDA device visible (e.g. CUDA_VISIBLE_DEVICES="" in a CPU-only Ray
        # orchestrator). Skip CUDA-library / cupy init (they call into the CUDA
        # runtime and would raise cudaErrorNoDevice); run CPU-only.
        cudnn = curand = cublas = cufft = cusparse = None
        numpy2cupy = None
        # CPU arrays default to the CUDA pinned-host allocator (cudaMallocHost),
        # which also fails with no device -- switch to the plain host allocator.
        try:
            core.flags.use_cuda_host_allocator = 0
        except Exception:
            pass


if _compat_preflight_result.active:
    _configure_compat_math_flags(_sys.modules[__name__])


from ._runtime import core_api as _core_api
from ._runtime.core_api import _core_flags
_publish(_core_api, _core_api.__all__)

# The runtime installs its monkeypatches from here on, in a fixed order that
# used to exist only as the physical arrangement of the statements below.
# jittor/_install_order.py declares that order and checks it.
from . import _install_order as _install_order
from ._install_order import record as _record_install

from . import nn
from . import fft
from .optim import legacy_schedulers as lr_scheduler
from . import linalg
from .linalg import einsum
from .nn import attention as attention
from .nn import matmul, \
    bmm, bmm_transpose, \
    baddbmm
from .nn.functional.softmax import logsumexp
from .nn.functional.tensor import kron, tensordot
from . import numpy2cupy
from .misc.concatenation import concat, cat
from .misc.indexing import install_var_indexing as _install_var_indexing
from .nn.backends.full_reduce_cuda import (
    install_full_reduce_fast_path as _install_full_reduce,
)

_install_var_indexing()
_record_install("misc.var_indexing")
del _install_var_indexing
_install_full_reduce()
_record_install("nn.full_reduce_fast_path")
del _install_full_reduce

from .compat import contrib as contrib
from . import misc as misc
_MISC_EXPORTS = tuple(misc.tensor_ops.__all__) + (
    "amax", "amin", "cat", "concat", "count_nonzero",
)
_publish(misc, _MISC_EXPORTS)
from . import sparse
from . import optim
from . import dataset
from . import init
from . import autograd

# The legacy ``jittor.gradfunctional`` spelling is published as a same-object
# alias after compatibility composition. Keep the root attribute available
# during composition as well.
gradfunctional = autograd

dtype = NanoString

import jittor_utils

_ran_post_process = False
for backend in jittor_utils.backends:
    if hasattr(backend, "post_process"):
        backend.post_process()
        _ran_post_process = True
if _ran_post_process:
    _record_install("backends.post_process")
del _ran_post_process

# ``all_`` and ``any_`` are reduction primitive names, not mutating methods.
delattr(Var, "all_")
delattr(Var, "any_")

_INPLACE_ALIASES = {
    "deg2rad_": deg2rad,
    "hardsigmoid_": nn.hardsigmoid,
    "hardswish_": nn.hardswish,
    "log2_": log2,
    "masked_fill_": masked_fill,
    "mul_": mul,
    "pow_": pow,
    "rad2deg_": rad2deg,
    "rrelu_": nn.rrelu,
    "rsqrt_": rsqrt,
    "scatter_reduce_": scatter_reduce,
    "sqr_": sqr,
    "sub_": sub,
    "squeeze_": squeeze,
    "t_": t,
    "transpose_": transpose,
    "unsqueeze_": unsqueeze,
}


def _make_inplace_alias(name, operation):
    def inplace(self, *args, **kwargs):
        return self.assign(operation(self, *args, **kwargs))
    inplace.__name__ = name
    return inplace


for _alias_name, _operation in _INPLACE_ALIASES.items():
    if not hasattr(Var, _alias_name):
        setattr(Var, _alias_name, _make_inplace_alias(_alias_name, _operation))
del _alias_name
del _operation
_record_install("root.inplace_aliases")

from . import math_util
_publish(math_util, math_util.__all__)
from . import distributions

if compiler.has_acl:
    from jittor.extern.acl.acl_compiler import change_function
    change_function()
    _record_install("acl.change_function")

# MPI-free Ascend multi-card: the optimizer/users call Var.mpi_all_reduce /
# Var.mpi_broadcast, normally provided by the MPI op module. In the env/file
# HCCL mode that module isn't loaded, so route those names to the HCCL ops
# directly (collectives never needed MPI -- only the bootstrap did).
if compile_extern.hccl_ops is not None and not compile_extern.has_mpi:
    _hops = compile_extern.hccl_ops
    def _hccl_all_reduce(self, op="mean"):
        # HCCL supports sum/prod/max/min; emulate "mean" as sum / world_size.
        if op == "mean":
            r = _hops.hccl_all_reduce(self, "sum")
            return r / compile_extern.world_size
        return _hops.hccl_all_reduce(self, op)
    def _hccl_broadcast(self, root=0):
        return _hops.hccl_broadcast(self, root)
    core.Var.mpi_all_reduce = _hccl_all_reduce
    core.Var.mpi_broadcast = _hccl_broadcast
    _record_install("collectives.hccl")

# MPI-free NVIDIA multi-card: same env/file rendezvous as HCCL, route the
# collectives to NCCL ops so DDP works without mpirun on NVIDIA too. NCCL
# all_reduce does ncclSum; emulate "mean" as sum / world_size.
# NB: condition is "MPI didn't provide the collective" (hasattr), NOT "not
# has_mpi" -- on a box where MPI is installed but disabled (use_mpi=0 in the
# env/file mode) has_mpi is still True yet mpi_all_reduce is absent.
if compile_extern.nccl_ops is not None and not hasattr(core.Var, "mpi_all_reduce"):
    _nops = compile_extern.nccl_ops
    def _nccl_all_reduce(self, op="mean"):
        r = _nops.nccl_all_reduce(self)
        if op == "mean":
            return r / compile_extern.world_size
        return r
    def _nccl_broadcast(self, root=0):
        return _nops.nccl_broadcast(self, root)
    core.Var.mpi_all_reduce = _nccl_all_reduce
    core.Var.mpi_broadcast = _nccl_broadcast
    _record_install("collectives.nccl")



from .compat.runtime import compose as _compose_compat_runtime

_compat_composition_report = _compose_compat_runtime(
    _sys.modules[__name__],
    _core_flags,
    strict=_compat_is_truthy(
        _os.environ.get("JITTOR_TORCH_STRICT_BOOTSTRAP")
    ),
    preflight=_compat_preflight_result,
)
_record_install("compat.runtime_composition")

# Torch compatibility adds optimizer classes and module aliases after ``nn``
# consumed the native facade. Refresh explicit star exports to match the
# historical module behavior without exporting implementation modules.
optim._refresh_public_exports()
_record_install("optim.public_exports")

# ``flags`` is replaced by the compatibility proxy during composition. Moved
# core API functions resolve globals in ``core_api``, so share that proxy.
_core_api.flags = flags

# Every installer has run. Fail loudly here rather than let a half-patched
# runtime import cleanly and then behave like a different version of jittor.
_install_report = _install_order.verify()
del _record_install


_ROOT_EXPORTS = (
    "LOG", "__version__", "attention", "autograd", "baddbmm", "bmm",
    "bmm_transpose", "cat", "compile_custom_op", "compile_custom_ops",
    "compile_extern", "compiler", "concat", "contrib", "core", "cublas",
    "cudnn", "cufft", "curand", "cusparse", "dataset", "distributions",
    "dtype", "einsum", "fft", "gradfunctional", "has_cuda", "in_mpi",
    "init", "jittor_core", "kron", "linalg", "logsumexp", "lr_scheduler",
    "math_util", "matmul", "misc", "mkl_ops", "mpi", "mpi_ops", "nn",
    "numpy2cupy", "optim", "ops", "rank", "sparse", "tensordot",
    "world_size",
)

__all__ = tuple(sorted(set(
    _NATIVE_CORE_EXPORTS
    + _NATIVE_OP_EXPORTS
    + tuple(_core_api.__all__)
    + _MISC_EXPORTS
    + tuple(math_util.__all__)
    + _ROOT_EXPORTS
)))
