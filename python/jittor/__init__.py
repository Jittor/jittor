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

from jittor_utils import lock

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

with lock.lock_scope():
    ori_int = int
    ori_float = float
    ori_bool = bool
    from . import compiler
    from .compiler import LOG, has_cuda
    from .compiler import compile_custom_ops, compile_custom_op
    import jittor_core
    import jittor_core as core
    from jittor_core import *
    from jittor_core.ops import *
    _core_profiler = core.profiler
    from . import compile_extern
    from .compile_extern import mkl_ops, mpi, mpi_ops, in_mpi, rank, world_size
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
from ._runtime.core_api import *
from ._runtime.core_api import _core_flags

from . import nn
from . import lr_scheduler
from . import linalg
from .linalg import einsum
from .nn import matmul, \
    bmm, bmm_transpose, \
    baddbmm
from . import contrib
from . import numpy2cupy
from .contrib import concat, cat
from .misc import *
from . import sparse
from . import optim
from . import dataset
from . import init
from . import gradfunctional

dtype = NanoString

import jittor_utils

for backend in jittor_utils.backends:
    if hasattr(backend, "post_process"):
        backend.post_process()

# impl x.func(...) -> func_(...)
args = {"x", "input", "self"}
_white_list = {"mul", "add", "sub"}
for k,v in list(Var.__dict__.items()):
    if k.startswith("_"): continue
    if k.endswith("_"): continue
    if not callable(v): continue

    if k not in _white_list:
        if not hasattr(v, "__code__"): continue
        conames = v.__code__.co_varnames
        if len(conames) == 0: continue
        arg_name = conames[0]
        if arg_name not in args: continue

    new_k = k+"_"
    if hasattr(Var, new_k): continue
    def inplace_wrapper(new_k, prev_func):
        setattr(Var, new_k, lambda x, *args, **kw: x.assign(prev_func(x, *args, **kw)))
    inplace_wrapper(new_k, v)

from . import math_util
from .math_util import *
from . import distributions

if jt.compiler.has_acl:
    from jittor.extern.acl.acl_compiler import change_function
    change_function()

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



from .compat.runtime import compose as _compose_compat_runtime

_compat_composition_report = _compose_compat_runtime(
    _sys.modules[__name__],
    _core_flags,
    strict=_compat_is_truthy(
        _os.environ.get("JITTOR_TORCH_STRICT_BOOTSTRAP")
    ),
    preflight=_compat_preflight_result,
)

# Torch compatibility adds optimizer classes and module aliases after ``nn``
# consumed the native facade. Refresh explicit star exports to match the
# historical module behavior without exporting implementation modules.
optim._refresh_public_exports()

# ``flags`` is replaced by the compatibility proxy during composition. Moved
# core API functions resolve globals in ``core_api``, so share that proxy.
_core_api.flags = flags
