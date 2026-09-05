"""Core Python API composed after the native runtime bootstrap."""

import jittor as jt
from jittor import *
from jittor import _core_profiler
from jittor.compile_extern import distributed_state_getattr as __getattr__

from typing import List, Tuple
import functools as _functools
import contextlib
import numpy as np
import numbers
import itertools
from collections import OrderedDict
from collections.abc import Sequence, Mapping
import types
import pickle
import hashlib
import sys, os
import traceback
from .acl_clamp import dispatch_acl_clamp
from .registry import OpRegistry

_runtime_op_registry = OpRegistry.default()

if "SKEY" in os.environ:
    import jittor_utils.student_queue

def dfs_to_numpy(x):
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = dfs_to_numpy(x[i])
    elif isinstance(x, dict):
        for k in x:
            x[k] = dfs_to_numpy(x[k])
    elif isinstance(x, Var):
        return x.numpy()
    return x

def safepickle(obj, path):
    if path.endswith(".pth") or path.endswith(".pt") or path.endswith(".bin"):
        from jittor_utils.save_pytorch import save_pytorch
        save_pytorch(path, obj)
        return
    # Protocol version 4 was added in Python 3.4. It adds support for very large objects, pickling more kinds of objects, and some data format optimizations.
    # ref: <https://docs.python.org/3/library/pickle.html>
    # obj = dfs_to_numpy(obj)
    s = pickle.dumps(obj, 4)
    checksum = hashlib.sha1(s).digest()
    s += bytes(checksum)
    s += b"HCAJSLHD"
    with open(path, 'wb') as f:
        f.write(s)

def _load_pkl(s, path):
    try:
        return pickle.loads(s)
    except Exception as e:
        msg = str(e)
        msg += f"\nPath: \"{path}\""
        if "trunc" in msg:
            msg += "\nThis file maybe corrupted, please consider remove it" \
                 " and re-download."
        raise RuntimeError(msg)

def _upload(path, url, jk, tdir=""):
    tdir = tdir + '/' if tdir != "" else ""
    prefix = f"https://cg.cs.tsinghua.edu.cn/jittor/{tdir}assets"
    if url.startswith("jittorhub://"):
        url = url.replace("jittorhub://", prefix+"/build/checkpoints/")
    assert url.startswith(prefix)
    suffix = url[len(prefix):]
    dir_suffix = "/".join(suffix.split("/")[:-1])
    jkey = flags.cache_path+"/_jkey"
    with open(jkey, 'w') as f:
        f.write(jk)
    assert os.system(f"chmod 600 \"{jkey}\"") == 0
    print(dir_suffix)
    assert os.system(f"s""s""h"f" -i \"{jkey}\" jittor" "@" "166" f".111.68.30 mkdir -p Documents/jittor-blog/{tdir}assets{dir_suffix}") == 0
    assert os.system(f"s""c""p"+f" -i \"{jkey}\" \"{path}\" jittor" "@" "166" f".111.68.30:Documents/jittor-blog/{tdir}assets{suffix}") == 0
    assert os.system(f"s""s""h"f" -i \"{jkey}\" jittor" "@" "166" ".111.68.30 Documents/jittor-blog.git/hooks/post-update") == 0


def safeunpickle(path):
    if path.startswith("jittorhub://"):
        path = path.replace("jittorhub://", f"https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/")
    if path.startswith("https:") or path.startswith("http:"):
        base = path.split("/")[-1]
        fname = os.path.join(compiler.ck_path, base)
        from jittor_utils.misc import download_url_to_local
        download_url_to_local(path, base, compiler.ck_path, None)
        path = fname
        if not (path.endswith(".pth") or path.endswith(".pkl") or path.endswith(".pt")):
            return path
    if path.endswith(".pth") or path.endswith(".pt") or path.endswith(".bin") :
        from jittor_utils.load_pytorch import load_pytorch
        model_dict = load_pytorch(path)
        return model_dict
    with open(path, "rb") as f:
        s = f.read()
    if not s.endswith(b"HCAJSLHD"):
        return _load_pkl(s, path)
    checksum = s[-28:-8]
    s = s[:-28]
    if hashlib.sha1(s).digest() != checksum:
        raise ValueError("Pickle checksum does not match! path: "+path,
        " This file maybe corrupted, please consider remove it"
        " and re-download.")
    return _load_pkl(s, path)

class _call_no_record_scope:
    def __enter__(self): pass
    def __exit__(self, *exc): pass

    def _new_scope(self):
        """A fresh scope object for one decorated call.

        ``__call__`` used to close over ``self`` and re-enter that one instance
        on every call, which is half of why ``@jt.no_grad()`` leaked: a
        recursive call re-entered the same object. Scopes that keep no
        per-entry state can go on returning ``self``; :class:`flag_scope`
        overrides this.
        """
        return self

    def __call__(self, func):
        @_functools.wraps(func)
        def inner(*args, **kw):
            with self._new_scope():
                ret = func(*args, **kw)
            return ret
        return inner

class flag_scope(_call_no_record_scope):
    """Set jittor flags for the duration of a ``with`` block or a call.

    The saved values live on a **stack**, not on a single attribute. With one
    attribute, entering the same scope object twice without leaving it in
    between overwrote the outer entry's backup with the inner one's, and the
    outer ``__exit__`` then "restored" the inner scope's values -- permanently.

    That is not an exotic case: ``__call__`` decorates a function with one
    scope instance, so ``@jt.no_grad()`` on a **recursive** function hits it on
    the first recursive call. The outer frame saved ``no_grad=0``, the inner
    frame overwrote that backup with ``no_grad=1`` (already inside the scope),
    and on the way out the process was left with ``no_grad=1`` for good --
    every subsequent ``jt.grad`` silently returning zeros, no error, training
    loss simply not moving.
    """

    def __init__(self, **jt_flags):
        self.jt_flags = jt_flags
        # one entry per active __enter__, so nesting and recursion compose
        self._flags_bk_stack = []

    def _new_scope(self):
        # a decorated call gets its own scope object as well as its own stack
        # entry; both are needed for reentrancy across threads and generators
        return type(self)(**self.jt_flags)

    def _flush_if_device_changes(self, wanted):
        """Run everything still pending before the device flag moves.

        Execution is lazy, so an op built inside ``flag_scope(use_cuda=1)`` can
        still be pending when the scope restores the old value, and it then runs
        on the other device -- with an op that has no CPU version (a bfloat16
        cast, say) that is a hard failure, and with one that has both it is a
        silent device switch. Only a scope that actually moves ``use_cuda`` pays
        for this, and such a scope is a device boundary anyway.
        """
        # NB: `from jittor import *` above shadows the builtins `bool` and
        # `int` with jittor ops, so this compares without calling either.
        if "use_cuda" not in self.jt_flags:
            return
        current = getattr(flags, "use_cuda")
        if (current != 0) == (wanted != 0):
            return
        sync_all()

    def __enter__(self):
        flags_bk = {}
        # push BEFORE setting anything, so the __exit__ in the except branch
        # below pops this entry and not an enclosing scope's
        self._flags_bk_stack.append(flags_bk)
        try:
            if "use_cuda" in self.jt_flags:
                self._flush_if_device_changes(self.jt_flags["use_cuda"])
            for k,v in self.jt_flags.items():
                origin = getattr(flags, k)
                flags_bk[k] = origin
                # merge dict attrs
                if isinstance(origin, dict):
                    for ok, ov in origin.items():
                        if ok not in v:
                            v[ok] = ov
                setattr(flags, k, v)
        except:
            self.__exit__()
            raise

    def __exit__(self, *exc):
        if not self._flags_bk_stack:
            # __exit__ without a matching __enter__; nothing was saved
            return
        flags_bk = self._flags_bk_stack.pop()
        # Not while an exception is unwinding: the pending work is likely what
        # raised, and a second error here would bury the first one.
        unwinding = len(exc) > 0 and exc[0] is not None
        try:
            if "use_cuda" in flags_bk and not unwinding:
                self._flush_if_device_changes(flags_bk["use_cuda"])
        finally:
            # Restoring the flags is not optional: leaving the scope's values in
            # place because the flush raised would corrupt everything after it.
            for k,v in flags_bk.items():
                setattr(flags, k, v)

class no_grad(flag_scope):
    ''' no_grad scope, all variable created inside this
scope will stop grad.

Example::

    import jittor as jt

    with jt.no_grad():
        ...

    '''
    def __init__(self, **jt_flags):
        jt_flags["no_grad"] = 1
        super().__init__(**jt_flags)

class enable_grad(flag_scope):
    ''' enable_grad scope, all variable created inside this
scope will start grad.

Example::

    import jittor as jt

    with jt.enable_grad():
        ...

    '''
    def __init__(self, **jt_flags):
        jt_flags["no_grad"] = 0
        super().__init__(**jt_flags)


def _output_requires_grad(*values):
    """Whether an op built from ``values`` must preserve an autograd path.

    Process-wide grad mode is only half of that decision.  Outside a
    ``no_grad`` scope an operation whose tensor inputs are all stopped still
    produces a stopped output, so inference-only fused kernels are safe for
    it.  Containers are accepted because stack/cache dispatchers receive
    lists and dictionaries of tensors.
    """
    if flags.no_grad:
        return False
    pending = list(values)
    while pending:
        value = pending.pop()
        if isinstance(value, Var):
            if value.requires_grad:
                return True
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
    return False


def _stop_grad_outputs(value):
    """Mark an inference fusion's returned tensors as non-differentiable."""
    if isinstance(value, Var):
        value.stop_grad()
    elif isinstance(value, list):
        for item in value:
            _stop_grad_outputs(item)
    elif isinstance(value, tuple):
        for item in value:
            _stop_grad_outputs(item)
    elif isinstance(value, dict):
        for item in value.values():
            _stop_grad_outputs(item)
    return value

single_log_capture = None

class log_capture_scope(_call_no_record_scope):
    """log capture scope

    Example::

        with jt.log_capture_scope(log_v=0) as logs:
            LOG.v("...")
        print(logs)
    """
    def __init__(self, **jt_flags):
        jt_flags["use_parallel_op_compiler"] = 0
        self.fs = flag_scope(**jt_flags)

    def __enter__(self):
        global single_log_capture
        assert not single_log_capture
        single_log_capture = jt.single_log_capture = True
        self.logs = []
        LOG.log_capture_start()
        try:
            self.fs.__enter__()
            if "log_v" in self.fs.jt_flags:
                LOG.log_v = self.fs.jt_flags["log_v"]
            return self.logs
        except:
            LOG.log_capture_stop()
            single_log_capture = jt.single_log_capture = None
            raise

    def __exit__(self, *exc):
        global single_log_capture
        self.fs.__exit__(*exc)
        if "log_v" in self.fs.jt_flags:
            LOG.log_v = flags.log_v
        LOG.log_capture_stop()
        self.logs.extend(LOG.log_capture_read())
        single_log_capture = jt.single_log_capture = None


class profile_scope(_call_no_record_scope):
    """ profile scope

    example::

        with jt.profile_scope() as report:
            ......
        print(report)
    """
    def __init__(self, warmup=0, rerun=0, **jt_flags):
        self.fs = flag_scope(**jt_flags)
        self.warmup = warmup
        self.rerun = rerun

    def __enter__(self):
        assert not flags.profiler_enable
        self.report = []
        try:
            self.fs.__enter__()
            _core_profiler.start(self.warmup, self.rerun)
            return self.report
        except:
            _core_profiler.stop()
            raise

    def __exit__(self, *exc):
        _core_profiler.stop()
        self.report.extend(_core_profiler.report())
        self.fs.__exit__(*exc)


class profile_mark(_call_no_record_scope):
    def __init__(self, mark_name: str):
        ''' profiler mark is used for profiling part of code,

        Example::

        a = jt.rand(1000,1000)
        b = jt.rand(1000,1000)
        jt.sync_all()
        results = []
        with jt.profile_scope() as rep:
            results.append(jt.matmul(a, b))
            with jt.profile_mark("mark1"):
                results.append(jt.matmul(a, b))
                with jt.profile_mark("mark2"):
                    results.append(jt.matmul(a, b))
            with jt.profile_mark("mark3"):
                results.append(jt.matmul(a, b))
            results.append(jt.matmul(a, b))

        Output::

        Total time:    46.8ms
        Total Memory Access:    57.2MB
        [Mark mark3] time:       9ms
        [Mark mark2] time:    8.28ms
        [Mark mark1] time:    17.7ms

        '''
        self.mark_name = mark_name
    def __enter__(self):
        self.options = flags.compile_options
        new_options = flags.compile_options
        prev_marks = "_marks:"
        for x in self.options:
            if x.startswith(prev_marks):
                prev_marks = x
                del new_options[x]
        new_marks = prev_marks + self.mark_name + ','
        new_options[new_marks] = 1
        flags.compile_options = new_options

    def __exit__(self, *exc):
        flags.compile_options = self.options

class __single_process_scope:
    def __init__(self, rank=0):
        self.rank = rank

    def __enter__(self):
        # compile_extern owns in_mpi; this module used to keep its own copy
        # (pulled in by `from jittor import *` at import time) and mutate that
        # alongside, so a later correction to compile_extern.in_mpi left this
        # copy stale and mpi_param_broadcast() below silently did nothing. 6.B15.
        self.bk_in_mpi = compile_extern.in_mpi
        if mpi:
            self.bk_mpi_state = mpi.get_state()
        if not self.bk_in_mpi:
            return True

        ret = self.rank == mpi.world_rank()
        compile_extern.in_mpi = False
        mpi.set_state(False)
        return ret

    def __exit__(self, *exc):
        compile_extern.in_mpi = self.bk_in_mpi
        if mpi:
            mpi.set_state(self.bk_mpi_state)

def single_process_scope(rank=0):
    """ single_process_scope

    Code in this scope will only be executed by single process.

    All the mpi code inside this scope will have not affect.
    mpi.world_rank() and mpi.local_rank() will return 0, world_size() will return 1,

    example::

        @jt.single_process_scope(rank=0)
        def xxx():
            ...
    """
    def outer(func):
        def inner(*args, **kw):
            ret = None
            sync_all()
            with __single_process_scope(rank) as flag:
                if flag:
                    ret = func(*args, **kw)
            return ret
        return inner
    return outer

def clean():
    import gc
    # make sure python do a full collection
    gc.collect()
    core.gc()

cast = unary
Var.cast = Var.cast

def array(data, dtype=None):
    ''' Constructs a jittor Var from a number, List, numpy array or another jittor Var.

    :param data: The data to initialize the Var.
    :type data: number, list, numpy.ndarray, or jittor.Var.
    :param dtype: The data type of the Var. If None, the data type will be inferred from the data.
    :type dtype: str, jittor type-cast function, or None.

    ----------------

    Example::

        >>> jt.array(1)
        jt.Var([1], dtype=int32)
        >>> jt.array([0, 2.71, 3.14])
        jt.Var([0.   2.71 3.14], dtype=float32)
        >>> jt.array(np.arange(4, dtype=np.uint8))
        jt.Var([0 1 2 3], dtype=uint8)
    '''
    # torch accepts a range/generator in tensor(...); jittor's core array op
    # rejects them -> materialise to a list first (mmdet pisa_loss: jt.array(range(...))).
    import types as _types_arr
    if isinstance(data, (range, _types_arr.GeneratorType, map, filter, zip)):
        data = list(data)
    elif isinstance(data, core.NanoVector):     # e.g. jt.array(some_var.shape)
        data = list(data)                       # NB: `int` is shadowed by the dtype here
    if isinstance(data, core.Var):
        if dtype is None:
            ret = data.clone()
        else:
            ret = cast(data, dtype)
    elif dtype is not None:
        if isinstance(dtype, NanoString):
            dtype = str(dtype)
        # Torch-compatible dtype objects are callable str subclasses.  They are
        # dtype names here, not Jittor's historical cast functions.
        elif not isinstance(dtype, str) and callable(dtype):
            dtype = dtype.__name__
        with jt.flag_scope(auto_convert_64_to_32=0):
            ret = ops.array(np.array(data, dtype))
    else:
        ret = ops.array(data)
    # TODO: move those code to core
    return _amp_array_preference(ret)

class amp_flags:
    """Named bits of ``jt.flags.amp_reg``, the auto-mixed-precision register.

    The register is a bit field, and every reader in the tree used to spell its
    bits as bare integers -- ``amp_reg & 16``, ``amp_reg | 36``, ``amp_reg=4``
    -- across six files, with the meaning written down only in the flag's
    description string in ``src/var.cc``. ``36`` is ``keep_reduce |
    array_prefer``; you had to know that.

    These MUST match ``amp_prefer32`` .. ``amp_array_prefer`` in
    ``src/misc/nano_string.h`` and the ``auto_mixed_precision_level`` setter in
    ``src/var.cc``; ``tests/core/test_amp_reg_bits.py`` reads the header and
    fails if they drift.
    """

    #: force float32 for ops whose inputs are all non-scalar floats
    prefer32 = 1
    #: force float16 (bfloat16 if an input is bfloat16) for the same
    prefer16 = 2
    #: let a reduce keep its input's float type instead of accumulating in f32
    keep_reduce = 4
    #: let "white list" ops (exp, log, pow, ...) follow the preference too,
    #: instead of always computing in float32
    keep_white = 8
    #: apply the preference to array-like producers (jt.array, jt.random) too
    array_prefer = 16
    #: a float16 sum/mean does NOT use a float32 intermediate accumulator
    #: (read directly as ``amp_reg & 32`` in src/ops/reduce_op.cc)
    reduce16_no_fp32_acc = 32


def _amp_array_preference(ret):
    """Apply the array-like AMP preference to a freshly produced Var.

    ``array()`` and ``random()`` each carried their own copy of this, and the
    copies had drifted: ``array()`` skipped one-element and non-float results,
    ``random()`` did not. So under ``auto_mixed_precision_level=5``,
    ``jt.array([1.0])`` stayed float32 while ``jt.random((1,))`` came back
    float16 -- the same value, produced two ways, with two dtypes.

    ``array()``'s guards are the ones that survive. A Var of one element is a
    scalar as far as jittor's dtype inference is concerned (``dtype_infer``
    passes ``has_scalar`` and then skips the preference entirely), so casting
    one down here would make ``jt.array(1e-8) * x`` disagree with
    ``1e-8 * x``; and a non-float result has no float preference to apply.
    """
    amp_reg = jt.flags.amp_reg
    if not (amp_reg & amp_flags.array_prefer):
        return ret
    if ret.numel() == 1 or not ret.dtype.is_float():
        return ret
    if amp_reg & amp_flags.prefer32:
        return ret if ret.dtype == "float32" else ret.float32()
    if amp_reg & amp_flags.prefer16:
        return ret if ret.dtype == "float16" else ret.float16()
    return ret


def random(shape, dtype="float32", type="uniform"):
    ''' Constructs a random jittor Var.

    :param shape: The shape of the random Var.
    :type shape: list or tuple.
    :param dtype: The data type of the random Var.
    :type dtype: str, jittor type-cast function, or None.
    :param type: The random distribution, can be 'uniform' or 'normal'.
    :type type: str

    ----------------

    Example::

        >>> jt.random((2, 3))
        jt.Var([[0.96788853 0.28334728 0.30482838]
                [0.46107793 0.62798643 0.03457401]], dtype=float32)
    '''
    for dim in shape:
        if dim < 0:
            raise RuntimeError(f"Trying to create tensor with negative dimension {dim}: {shape}")
    if isinstance(dtype, NanoString):
        dtype = str(dtype)
    elif not isinstance(dtype, str) and callable(dtype):
        dtype = dtype.__name__
    if dtype in ("float16", "bfloat16"):
        # The CPU and accelerator random engines generate standard floating
        # types; low-precision outputs use their regular cast kernels.
        ret = ops.random(shape, "float32", type).cast(dtype)
    else:
        ret = ops.random(shape, dtype, type)
    return _amp_array_preference(ret)

_core_to_device = Var.to_device

def to_device(self, device):
    ''' Return this Var on CUDA device ``device`` -- an index, or anything
    with an ``index`` attribute such as a torch device. A Var already on that
    device is returned unchanged; otherwise the data is copied there by the
    ``device_copy`` op, whose gradient is a copy back.

    Devices are independent: an op takes its inputs' device and mixing two
    devices in one op is an error, as in torch. This is the only way data
    changes device. '''
    # `from jittor import *` shadows the builtin int with the cast op.
    device = ori_int(getattr(device, "index", device))
    if device < 0:
        raise RuntimeError("CUDA device index must be non-negative")
    # A host-resident Var remembers the CUDA device it belongs to.  Returning
    # it unchanged merely because the index matches would make ``x.cpu().cuda()``
    # stay on the host.  DeviceCopyOp has an explicit host-to-device path.
    if device == self.device_id and self.location() != "cpu":
        return self
    return _core_to_device(self, device)
Var.to_device = to_device

def _copy_to_cpu(self):
    '''Return a differentiable, independently allocated host copy.'''
    return _core_to_device(self, -1)
Var._copy_to_cpu = _copy_to_cpu

def float_auto(x):
    if jt.flags.amp_reg & amp_flags.prefer16:
        return x.float16()
    return x.float32()
Var.float_auto = float_auto

def array64(data, dtype=None):
    with jt.flag_scope(auto_convert_64_to_32=0):
        return array(data, dtype)

def grad(loss, targets, retain_graph=True):
    if type(targets) == core.Var:
        return core.grad(loss, [targets], retain_graph)[0]
    return core.grad(loss, targets, retain_graph)

def liveness_info():
    return {
        "hold_vars": core.number_of_hold_vars(),
        "lived_vars": core.number_of_lived_vars(),
        "lived_ops": core.number_of_lived_ops(),
    }

def ones(*shape, dtype="float32"):
    ''' Constructs a jittor Var with all elements set to 1.

    :param shape: The shape of the output Var.
    :type shape: list or tuple.
    :param dtype: The data type of the output Var.
    :type dtype: str, jittor type-cast function, or None.
    :return: The output Var.
    :rtype: jittor.Var
    '''
    if isinstance(shape, tuple) and isinstance(shape[-1], (str, NanoString)):
        dtype = shape[-1]
        shape = shape[:-1]
    if isinstance(shape, tuple) and isinstance(shape[0], (Sequence, NanoVector)):
        shape = shape[0]
    for dim in shape:
        if dim < 0:
            raise RuntimeError(f"Trying to create tensor with negative dimension {dim}: {shape}")
    return unary(1, dtype).broadcast(shape)

def new_ones(x, size):
    return ones(size, x.dtype)
Var.new_ones = new_ones

def ones_like(x):
    ''' Constructs a jittor Var with all elements set to 1 and shape same with x.

    :param x: The reference jittor Var.
    :type x: jt.Var
    :return: The output Var.
    :rtype: jittor.Var
    '''
    return ones(x.shape,x.dtype)

def zeros(*shape, dtype="float32"):
    ''' Constructs a jittor Var with all elements set to 0.

    :param shape: The shape of the output Var.
    :type shape: list or tuple.
    :param dtype: The data type of the output Var.
    :type dtype: str, jittor type-cast function, or None.
    :return: The output Var.
    :rtype: jittor.Var
    '''
    if isinstance(shape, tuple) and isinstance(shape[-1], (str, NanoString)):
        dtype = shape[-1]
        shape = shape[:-1]
    if isinstance(shape, tuple) and isinstance(shape[0], (Sequence, NanoVector)):
        shape = shape[0]
    for dim in shape:
        if dim < 0:
            raise RuntimeError(f"Trying to create tensor with negative dimension {dim}: {shape}")
    return unary(0, dtype).broadcast(shape)

def new_zeros(x, size):
    return zeros(size, x.dtype)
Var.new_zeros = new_zeros

def empty(*shape, dtype="float32"):
    if isinstance(shape, tuple) and isinstance(shape[-1], (str, NanoString)):
        dtype = shape[-1]
        shape = shape[:-1]
    if isinstance(shape, tuple) and isinstance(shape[0], (Sequence, NanoVector)):
        shape = shape[0]
    return ops.empty(shape, dtype)

def new_empty(x, size):
    return empty(size, x.dtype)
Var.new_empty = new_empty

def full(shape,val,dtype="float32"):
    ''' Constructs a jittor Var with all elements set to val.

    :param shape: The shape of the output Var.
    :type shape: list or tuple.
    :param val: The value of the output Var.
    :type val: number.
    :param dtype: The data type of the output Var. Defaults to jt.float32.
    :type dtype: str, jittor type-cast function, or None.
    :return: The output Var.
    :rtype: jittor.Var
    '''
    if not isinstance(shape, (NanoVector, Sequence)):
        shape = (shape,)
    for dim in shape:
        if dim < 0:
            raise RuntimeError(f"Trying to create tensor with negative dimension {dim}: {shape}")
    return unary(val, dtype).broadcast(shape)

def new_full(x, size, val):
    return full(size, val, x.dtype)
Var.new_full = new_full

def ne(x,y):
    return x!=y
Var.ne = ne

def full_like(x, val, dtype=None) -> Var:
    ''' Constructs a jittor Var with all elements set to val and shape same with x.

    :param x: The reference jittor Var.
    :type x: jt.Var.
    :param val: The value of the output Var.
    :type val: number.
    :param dtype: if None, the dtype of the output is the same as x.
        Otherwise, use the specified dtype. Defaults to None.
    :type dtype: str, optional
    :return: The output Var.
    :rtype: jittor.Var
    '''
    if dtype is None: dtype = x.dtype
    return full(x.shape, val, dtype)

def zeros_like(x, dtype=None) -> Var:
    ''' Constructs a jittor Var with all elements set to 0 and shape same with x.

    :param x: The reference jittor Var.
    :type x: jt.Var
    :param dtype: if None, the dtype of the output is the same as x.
        Otherwise, use the specified dtype. Defaults to None.
    :type dtype: str, optional
    :return: The output Var.
    :rtype: jittor.Var
    '''
    if dtype is None: dtype = x.dtype
    return zeros(x.shape, dtype)


_core_flags = core.Flags()
flags = _core_flags


class RuntimeContext:
    """Owner for execution state that is still backed by the native runtime.

    The context deliberately holds the native ``Flags`` object instead of
    copying individual values.  This gives the runtime a single migration
    seam while preserving the existing flag_scope and native setter semantics.
    """

    __slots__ = ("_flags",)

    def __init__(self, native_flags):
        self._flags = native_flags

    @property
    def sync_run(self):
        """Whether backend operators should synchronize after each launch."""
        return self._flags.sync_run

    @property
    def device_id(self):
        """Current device selected by the native runtime, or ``-1`` on CPU."""
        return getattr(self._flags, "device_id", -1)

    @property
    def use_cuda(self):
        """Whether the native runtime is configured to use CUDA."""
        return self._flags.use_cuda

    @property
    def lazy_execution(self):
        """Whether graph execution is deferred until an explicit flush."""
        return self._flags.lazy_execution

    @property
    def auto_flush_ops(self):
        """CUDA pipeline threshold for automatically submitting pending ops."""
        return self._flags.auto_flush_ops

    @property
    def no_grad(self):
        """Whether newly created operations are excluded from autograd."""
        return self._flags.no_grad

    @property
    def gopt_disable(self):
        """Whether graph optimization is disabled for execution."""
        return self._flags.gopt_disable

    @property
    def exec_called(self):
        """Number of executor synchronizations started by the runtime."""
        return self._flags.exec_called

    def snapshot(self):
        """Return an immutable snapshot of the fields owned by this context."""
        return {
            "sync_run": int(self.sync_run),
            "device_id": int(self.device_id),
            "use_cuda": int(self.use_cuda),
            "lazy_execution": int(self.lazy_execution),
            "auto_flush_ops": int(self.auto_flush_ops),
            "no_grad": int(self.no_grad),
            "gopt_disable": int(self.gopt_disable),
            "exec_called": int(self.exec_called),
        }


class RuntimeState:
    """Read-only Python view of the native :class:`RuntimeContext`.

    The view stores no state of its own.  In particular, ``flag_scope`` and
    direct native flag writes remain immediately visible through this object.
    """

    __slots__ = ("_context",)

    def __init__(self, context):
        self._context = context

    @property
    def sync_run(self):
        return self._context.sync_run

    @property
    def device_id(self):
        return self._context.device_id

    @property
    def use_cuda(self):
        return self._context.use_cuda

    @property
    def lazy_execution(self):
        return self._context.lazy_execution

    @property
    def auto_flush_ops(self):
        return self._context.auto_flush_ops

    @property
    def no_grad(self):
        return self._context.no_grad

    @property
    def gopt_disable(self):
        return self._context.gopt_disable

    @property
    def exec_called(self):
        return self._context.exec_called

    @property
    def context(self):
        """The state owner, exposed for diagnostics but not replacement."""
        return self._context


_runtime_context = RuntimeContext(flags)
runtime = RuntimeState(_runtime_context)


def var(x, dim=None, dims=None, unbiased=False, keepdims=False):
    """ return the sample variance. If unbiased is True, Bessel's correction will be used.

    :param x: the input jittor Var.
    :type x: jt.Var.
    :param dim: the dimension to compute the variance. If both dim and dims are None, the variance of the whole tensor will be computed.
    :type dim: int.
    :param dims: the dimensions to compute the variance. If both dim and dims are None, the variance of the whole tensor will be computed.
    :type dims: tuple of int.
    :param unbiased: if True, Bessel's correction will be used.
    :type unbiased: bool.
    :param keepdim: if True, the output shape is same as input shape except for the dimension in dim.
    :type keepdim: bool.

    Example::

        >>> a = jt.rand(3)
        >>> a
        jt.Var([0.79613626 0.29322362 0.19785859], dtype=float32)
        >>> a.var()
        jt.Var([0.06888353], dtype=float32)
        >>> a.var(unbiased=True)
        jt.Var([0.10332529], dtype=float32)
    """
    shape = x.shape
    new_shape = list(x.shape)

    assert dim is None or dims is None, "dim and dims can not be both set"
    if dim is None and dims is None:
        dims = list(range(len(shape)))
    elif dim is not None:
        dims = [dim]

    mean = jt.mean(x, dims, keepdims=True)
    mean = jt.broadcast(mean, shape)

    n = 1
    for d in dims:
        n *= shape[d]
        new_shape[d] = 1

    sqr = (x - mean) ** 2
    sqr = jt.sum(sqr, dims=dims, keepdims=False)
    if unbiased:
        n -= 1
    sqr /= n

    if keepdims:
        sqr = sqr.view(new_shape)
    return sqr
Var.var = var

def std(x, dim=None, keepdim=False):
    if dim is None:
        matsize=1
        for i in x.shape:
            matsize *= i
        out=(x-x.mean()).sqr().sum()
        out=out/(matsize-1)
        out=out.maximum(1e-6).sqrt()
        return out
    else:
        dimsize=x.size(dim)
        mean=jt.mean(x, dim, keepdim=True)
        out=(x - mean).sqr().sum(dim=dim, keepdim=keepdim)
        out=out/(dimsize-1)
        out=out.maximum(1e-6).sqrt()
        return out
Var.std = std

def norm(x, p=2, dim=-1, keepdims=False, eps=1e-30, keepdim=False):
    keepdim = keepdim or keepdims
    assert p==1 or p==2
    if p==1:
        return x.abs().sum(dim, keepdim)
    if p==2:
        return (x.sqr()).sum(dim, keepdim).maximum(eps).sqrt()
Var.norm = norm

origin_reshape = reshape
def reshape(x, *shape):
    if len(shape) == 1 and isinstance(shape[0], (Sequence, NanoVector)):
        shape = shape[0]
    # torch accepts 0-d int tensors / numpy ints as shape elements (e.g. longformer's
    # `_chunk` passes torch.div(size, n) into .view); jittor's core reshape needs plain
    # int64. Coerce only when a non-int element is present — plain-int shapes (the hot
    # path) are untouched, so this can't change existing behavior, only un-break it.
    # (NB: in this namespace `int`/`all`/`any` are shadowed by jittor's dtype/reductions,
    # so use an explicit loop and grab the genuine builtin int via `(0).__class__`.)
    pyint = (0).__class__
    coerce = False
    for s in shape:
        if type(s) is not pyint:
            coerce = True
            break
    if coerce:
        shape = tuple(pyint(s.item()) if isinstance(s, Var) else pyint(s) for s in shape)
    return origin_reshape(x, shape)
reshape.__doc__ = origin_reshape.__doc__
Var.view = Var.reshape = view = reshape

origin_transpose = transpose
def transpose(x, *dim):
    original_dim = dim
    if len(dim) == 1 and isinstance(dim[0], (Sequence, NanoVector)):
        dim = dim[0]
    elif len(dim) == 2:
        axes = list(range(x.ndim))
        a, b = dim
        axes[a], axes[b] = axes[b], axes[a]
        dim = axes
    # NumPy helpers such as np.argsort return numpy.integer axis values.  The
    # C++ transpose binding requires exact Python ints, while torch accepts any
    # integral sequence in Tensor.permute().
    pyint = (0).__class__
    coerce = False
    for d in dim:
        if type(d) is not pyint:
            coerce = True
            break
    if coerce:
        dim = tuple(pyint(d.item()) if isinstance(d, Var) else pyint(d) for d in dim)
    out = origin_transpose(x, dim)
    try:
        axes_tuple = tuple(pyint(i) for i in dim)
        last2 = list(range(x.ndim))
        if x.ndim >= 2:
            last2[-1], last2[-2] = last2[-2], last2[-1]
        if x.ndim >= 2 and axes_tuple == tuple(last2):
            out._jittor_transpose_base = x
            out._jittor_transpose_axes = axes_tuple
            out._jittor_transpose_last2 = True
        elif len(original_dim) == 2:
            a, b = pyint(original_dim[0]), pyint(original_dim[1])
            if x.ndim >= 2 and {a % x.ndim, b % x.ndim} == {x.ndim - 2, x.ndim - 1}:
                out._jittor_transpose_base = x
                out._jittor_transpose_axes = axes_tuple
                out._jittor_transpose_last2 = True
    except Exception:
        pass
    return out
transpose.__doc__ = origin_transpose.__doc__
Var.transpose = Var.permute = permute = transpose

def _flatten_cpu(input, start_dim=0, end_dim=-1):
    '''flatten dimentions by reshape'''
    in_shape = input.shape
    start_dim = len(in_shape) + start_dim if start_dim < 0 else start_dim
    end_dim = len(in_shape) + end_dim if end_dim < 0 else end_dim
    assert end_dim >= start_dim, "end_dim should be larger than or equal to start_dim for flatten function"
    if len(in_shape) <= end_dim:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-len(in_shape)}, {len(in_shape) - 1}], but got {end_dim})")
    out_shape = []
    for i in range(0,start_dim,1): out_shape.append(in_shape[i])
    dims = 1
    for i in range(start_dim, end_dim+1, 1): dims *= in_shape[i]
    out_shape.append(dims)
    for i in range(end_dim+1,len(in_shape),1): out_shape.append(in_shape[i])
    return input.reshape(out_shape)

_runtime_op_registry.register("flatten", "cpu", _flatten_cpu)

def flatten(input, start_dim=0, end_dim=-1):
    # CPU operations use the registry seam; CUDA keeps the pre-migration
    # implementation until its native provider is registered.
    if _runtime_op_registry.backends.backend_for(input) == "cpu":
        return _runtime_op_registry.dispatch_value(
            "flatten", input, start_dim, end_dim)
    return _flatten_cpu(input, start_dim, end_dim)

Var.flatten = flatten

Var.detach_inplace = Var.start_grad

def detach(x):
    return x.detach()

def unsqueeze(x, dim):
    shape = list(x.shape)
    if dim < 0: dim += len(shape) + 1
    assert dim <= len(shape)
    return x.reshape(shape[:dim] + [1] + shape[dim:])
Var.unsqueeze = unsqueeze

def squeeze(x, dim=None):
    shape = list(x.shape)
    if dim is None:
        # squeeze removes ONLY size-1 dims (size-0 dims must be kept, else an empty
        # tensor like [0,1] reshapes to the wrong size). jittor has no 0-dim tensors,
        # so an all-ones shape collapses to [1] (mmdet: nonzero(...).squeeze()).
        new_shape = [s for s in shape if s != 1]
        return x.reshape(new_shape if new_shape else [1])
    else:
        if dim < 0: dim += len(shape)
        assert dim < len(shape) and dim >= 0
        # torch (and numpy): squeeze(dim) is a no-op when that dim's size != 1,
        # not an error (canine's _downsample_attention_mask relies on this).
        if shape[dim] != 1:
            return x
        new_shape = shape[:dim] + shape[dim+1:]
        return x.reshape(new_shape if new_shape else [1])
Var.squeeze = squeeze

def _clamp_cpu(x, min_v=None, max_v=None):
    if x.shape[0]==0:
        return x
    # Torch allows tensor bounds and reversed scalar bounds. Applying the lower
    # then upper bound also gives Torch's all-max result when min_v > max_v.

    def prepare_bound(value, bound):
        if isinstance(bound, jt.Var):
            dtype = jt.binary_dtype_infer("add", value.dtype, bound.dtype)
            if value.dtype != dtype:
                value = value.cast(dtype)
            if bound.dtype != dtype:
                bound = bound.cast(dtype)
        elif "float" in str(value.dtype):
            bound = jt.unary(bound, value.dtype).stop_grad()
        elif isinstance(bound, numbers.Real) \
                and not isinstance(bound, numbers.Integral):
            value = value.cast("float32")
            bound = jt.unary(bound, "float32").stop_grad()
        else:
            bound = jt.unary(bound, value.dtype).stop_grad()
        return value, bound

    scalar_bounds = (
        min_v is not None
        and max_v is not None
        and not isinstance(min_v, jt.Var)
        and not isinstance(max_v, jt.Var)
        and min_v <= max_v
    )
    if min_v is not None:
        x, min_v = prepare_bound(x, min_v)
    if max_v is not None:
        x, max_v = prepare_bound(x, max_v)

    if scalar_bounds:
        result = dispatch_acl_clamp(x, min_v, max_v)
        if result is not None:
            return result

    def select_bound(value, bound, lower):
        keep = value >= bound if lower else value <= bound
        result = jt.ternary(keep, value, bound)
        if "float" in str(value.dtype):
            nan_value = value.clone().stop_grad()
            result = jt.ternary(value != value, nan_value, result)
        return result

    if min_v is not None:
        x = select_bound(x, min_v, True)
    if max_v is not None:
        x = select_bound(x, max_v, False)
    return x

_runtime_op_registry.register("clamp", "cpu", _clamp_cpu)

def clamp(x, min_v=None, max_v=None):
    # CPU operations now use the backend/op registry seam.  CUDA keeps the
    # existing implementation until its native provider is migrated.
    if _runtime_op_registry.backends.backend_for(x) == "cpu":
        return _runtime_op_registry.dispatch_value("clamp", x, min_v, max_v)
    return _clamp_cpu(x, min_v, max_v)

Var.clamp = clamp

def clamp_(x, min_v=None, max_v=None):
    ''' In-place version of clamp().

    Args:
        x (Jittor Var):
            the input var
        min_v ( Number or Var, optional) - lower-bound of clamp range
        max_v ( Number or Var, optional) - upper-bound of clamp range

    Return:
        x itself after clamp.

    '''
    return x.assign(x.clamp(min_v=min_v, max_v=max_v))
Var.clamp_ = clamp_


def _outer_cpu(x, y):
    ''' Returns the outer product of two 1-D vectors.

    :param x: the input Var.
    :type x: jt.Var, numpy array, or python sequence.
    :param y: the input Var.
    :type y: jt.Var, numpy array, or python sequence.


    Example::

    >>> x = jt.arange(3)
    >>> y = jt.arange(4)
    >>> jt.outer(x, y)
    jt.Var([[0 0 0 0]
            [0 1 2 3]
            [0 2 4 6]], dtype=int32)
    >>> x.outer(y)
    jt.Var([[0 0 0 0]
            [0 1 2 3]
            [0 2 4 6]], dtype=int32)
    '''
    return jt.multiply(x.unsqueeze(1), y.unsqueeze(0))


_runtime_op_registry.register("outer", "cpu", _outer_cpu)


def outer(x, y):
    # CPU operations use the registry seam; CUDA keeps the pre-migration
    # implementation until its native provider is registered.
    if _runtime_op_registry.backends.backend_for(x) == "cpu":
        return _runtime_op_registry.dispatch_value("outer", x, y)
    return _outer_cpu(x, y)


Var.outer = outer

def erfinv_(x):
    ''' In-place version of erfinv().
    '''
    return x.assign(x.erfinv())
Var.erfinv_ = erfinv_

def erf_(x):
    ''' In-place version of erf().
    '''
    return x.assign(x.erf())
Var.erf_ = erf_

def abs_(x):
    ''' In-place version of abs().
    '''
    return x.assign(x.abs())
Var.abs_ = abs_

def sigmoid_(x):
    ''' In-place version of sigmoid().
    '''
    return x.assign(x.sigmoid())
Var.sigmoid_ = sigmoid_

def sqrt_(x):
    ''' In-place version of sqrt().
    '''
    return x.assign(x.sqrt())
Var.sqrt_ = sqrt_

def add_(x, y):
    ''' In-place version of add().
    '''
    return x.assign(x.add(y))
Var.add_ = add_

def multiply_(x, y):
    ''' In-place version of multiply().
    '''
    return x.assign(x.multiply(y))
Var.multiply_ = multiply_

def type_as(a, b):
    return a.unary(op=b.dtype)
Var.type_as = type_as
Var.astype = Var.cast

def masked_fill(x, mask, value):
    return jt.ternary(mask, value, x)
Var.masked_fill = masked_fill


def sqr(x): return x*x
Var.sqr = sqr

def pow(x, y):
    ''' computes x^y, element-wise.

    This operation is equivalent to ``x ** y``.

    :param x: the first input.
    :type x: a python number or jt.Var.
    :param y: the second input.
    :type y: a python number or jt.Var.
    '''
    if isinstance(x,Var) and isinstance(y, (ori_int, ori_float)):
        if y == 2:
            return x.sqr()
        if y == 3 and str(x.dtype) == "float32":
            return x*x*x
    return core.ops.pow(x, y)
Var.pow = Var.__pow__ = pow

def argmax(x: Var, dim: int, keepdims:bool=False):
    ''' Returns the indices and values of the maximum elements along the specified dimension.

    :param x: the input Var.
    :type x: jt.Var, numpy array, or python sequence.
    :param dim: the dimension to reduce.
    :type dim: int.
    :param keepdims: whether the output Var has dim retained or not. Defaults to False
    :type keepdims: bool, optional

    Example::

        >>> a = jt.randn((2, 4))
        >>> a
        jt.Var([[-0.33272865 -0.4951588   1.4128606   0.13734372]
                [-1.633469    0.19593953 -0.7803732  -0.5260756 ]], dtype=float32)
        >>> a.argmax(dim=0)
        (jt.Var([0 1 0 0], dtype=int32), jt.Var([-0.33272865  0.19593953  1.4128606   0.13734372], dtype=float32))
        >>> a.argmax(dim=1)
        (jt.Var([2 1], dtype=int32), jt.Var([1.4128606  0.19593953], dtype=float32))
    '''
    if dim is None:
        dim = 0
        x = x.flatten()
    elif hasattr(x, "shape"):
        nd = len(x.shape)
        if not (-nd <= dim < nd):
            # clear error instead of the cryptic cutt_transpose "axes != xdim"
            raise IndexError(f"argmax: dim {dim} out of range for a {nd}-D "
                             f"input (expected dim in [{-nd}, {nd-1}])")
        # normalize negative dim: arg_reduce's internal transpose miscomputes the
        # axes for negative dims other than -1 -> cryptic cutt_transpose crash
        if dim < 0:
            dim += nd
    return jt.arg_reduce(x, "max", dim, keepdims)
Var.argmax = argmax

def argmin(x, dim: int, keepdims:bool=False):
    ''' Returns the indices and values of the minimum elements along the specified dimension.

    :param x: the input Var.
    :type x: jt.Var, numpy array, or python sequence.
    :param dim: the dimension to reduce.
    :type dim: int.
    :param keepdims: whether the output Var has dim retained or not. Defaults to False
    :type keepdims: bool, optional

    Example::

        >>> a = jt.randn((2, 4))
        >>> a
        jt.Var([[-0.33272865 -0.4951588   1.4128606   0.13734372]
                [-1.633469    0.19593953 -0.7803732  -0.5260756 ]], dtype=float32)
        >>> a.argmin(dim=0)
        (jt.Var([1 0 1 1], dtype=int32), jt.Var([-1.633469  -0.4951588 -0.7803732 -0.5260756], dtype=float32))
        >>> a.argmin(dim=1)
        (jt.Var([1 0], dtype=int32), jt.Var([-0.4951588 -1.633469 ], dtype=float32))
    '''
    if dim is not None and hasattr(x, "shape"):
        nd = len(x.shape)
        if not (-nd <= dim < nd):
            raise IndexError(f"argmin: dim {dim} out of range for a {nd}-D "
                             f"input (expected dim in [{-nd}, {nd-1}])")
        if dim < 0:
            dim += nd
    return jt.arg_reduce(x, "min", dim, keepdims)
Var.argmin = argmin

def randn(*size, dtype="float32", requires_grad=True) -> Var:
    ''' samples random numbers from a standard normal distribution.

    :param size: shape of the output.
    :type size: int or a sequence of int

    :param dtype: data type, defaults to "float32".
    :type dtype: str, optional

    :param requires_grad: whether to enable gradient back-propgation, defaults to True.
    :type requires_grad: bool, optional

    Example::

        >>> jt.randn(3)
        jt.Var([-1.019889   -0.30377278 -1.4948598 ], dtype=float32)
        >>> jt.randn(2, 3)
        jt.Var([[-0.15989183 -1.5010914   0.5476955 ]
         [-0.612632   -1.1471151  -1.1879086 ]], dtype=float32)
    '''
    if isinstance(size, tuple) and isinstance(size[0], (tuple, list, NanoVector)): size = size[0]
    for dim in size:
        if dim < 0:
            raise RuntimeError(f"Trying to create tensor with negative dimension {dim}: {size}")
    arr = jt.random(size, dtype, "normal")
    if not requires_grad: return arr.stop_grad()
    return arr

def rand(*size, dtype="float32", requires_grad=True) -> Var:
    ''' samples random numbers from a uniform distribution on the interval [0, 1).

    :param size: shape of the output.
    :type size: int or a sequence of int

    :param dtype: data type, defaults to "float32".
    :type dtype: str, optional

    :param requires_grad: whether to enable gradient back-propgation. defaults to True.
    :type requires_grad: bool, optional

    Example::

        >>> jt.rand(3)
        jt.Var([0.31005102 0.02765604 0.8150749 ], dtype=float32)
        >>> jt.rand(2, 3)
        jt.Var([[0.96414304 0.3519264  0.8268017 ]
         [0.05658621 0.04449705 0.86190987]], dtype=float32)
    '''
    if isinstance(size, tuple) and isinstance(size[0], (tuple, list, NanoVector)): size = size[0]
    arr = jt.random(size, dtype)
    if not requires_grad: return arr.stop_grad()
    return arr

def rand_like(x, dtype=None) -> Var:
    ''' samples random values from standard uniform distribution with the same shape as x.

    :param x: reference variable.
    :type x: jt.Var

    :param dtype: if None, the dtype of the output is the same as x.
        Otherwise, use the specified dtype. Defaults to None.
    :type dtype: str, optional

    Example::

        >>> x = jt.zeros((2, 3))
        >>> jt.rand_like(x)
        jt.Var([[0.6164821  0.21476883 0.61959815]
         [0.58626485 0.35345772 0.5638483 ]], dtype=float32)
    '''
    if dtype is None: dtype = x.dtype
    return jt.random(x.shape, dtype)

def randn_like(x, dtype=None) -> Var:
    ''' samples random values from standard normal distribution with the same shape as x.

    :param x: reference variable.
    :type x: jt.Var

    :param dtype: if None, the dtype of the output is the same as x.
        Otherwise, use the specified dtype. Defaults to None.
    :type dtype: str, optional

    Example::

        >>> x = jt.zeros((2, 3))
        >>> jt.randn_like(x)
        jt.Var([[-1.1647032   0.34847224 -1.3061888 ]
         [ 1.068085   -0.34366122  0.13172573]], dtype=float32)
    '''
    if dtype is None: dtype = x.dtype
    return jt.random(x.shape, dtype, "normal")

def randint(low, high=None, shape=(1,), dtype="int32") -> Var:
    ''' samples random integers from a uniform distribution on the interval [low, high).

    :param low: lowest intergers to be drawn from the distribution, defaults to 0.
    :type low: int, optional

    :param high: One above the highest integer to be drawn from the distribution.
    :type high: int

    :param shape: shape of the output size, defaults to (1,).
    :type shape: tuple, optional

    :param dtype: data type of the output, defaults to "int32".
    :type dtype: str, optional

    Example::

        >>> jt.randint(3, shape=(3, 3))
        jt.Var([[2 0 2]
                [2 1 2]
                [2 0 1]], dtype=int32)
        >>> jt.randint(1, 3, shape=(3, 3))
        jt.Var([[2 2 2]
                [1 1 2]
                [1 1 1]], dtype=int32)
    '''
    if high is None: low, high = 0, low
    for dim in shape:
        if dim < 0:
            raise RuntimeError(f"Trying to create tensor with negative dimension {dim}: {shape}")
    v = (jt.random(shape) * (high - low) + low).clamp(low, high-0.5)
    v = jt.floor_int(v)
    return v.astype(dtype)

def randint_like(x, low, high=None) -> Var:
    ''' samples random values from standard normal distribution with the same shape as x.

    :param x: reference variable.
    :type x: jt.Var

    :param low: lowest intergers to be drawn from the distribution, defaults to 0.
    :type low: int, optional

    :param high: One above the highest integer to be drawn from the distribution.
    :type high: int

    Example::

        >>> x = jt.zeros((2, 3))
        >>> jt.randint_like(x, 10)
        jt.Var([[9. 3. 4.]
                [4. 8. 5.]], dtype=float32)
        >>> jt.randint_like(x, 10, 20)
        jt.Var([[17. 11. 18.]
                [14. 17. 15.]], dtype=float32)
     '''

    return randint(low, high, x.shape, x.dtype)

def normal(mean, std, size=None, dtype="float32") -> Var:
    ''' samples random values from a normal distribution.

    :param mean: means of the normal distributions.
    :type mean: int or jt.Var

    :param std: standard deviations of the normal distributions.
    :type std: int or jt.Var

    :param size: shape of the output size. if not specified, the
        shape of the output is determined by mean or std. Exception will be
        raised if mean and std are all integers or have different shape in
        this case. Defaults to None
    :type size: tuple, optional

    :param dtype: data type of the output, defaults to "float32".
    :type dtype: str, optional

    Example::

        >>> jt.normal(5, 3, size=(2,3))
        jt.Var([[ 8.070848   7.654219  10.252696 ]
                [ 6.383718   7.8817277  3.0786133]], dtype=float32)
        >>> mean = jt.randint(low=0, high=10, shape=(10,))
        >>> jt.normal(mean, 0.1)
        jt.Var([1.9524184 1.0749301 7.9864206 5.9407325 8.1596155 4.824019  7.955083
                8.972998  6.0674286 8.88026  ], dtype=float32)
    '''
    if size is None:
        if isinstance(mean, Var) and isinstance(std, Var):
            assert mean.shape == std.shape
            size = mean.shape
        else:
            if isinstance(mean, Var): size = mean.shape
            if isinstance(std, Var): size = std.shape
    return jt.init.gauss(size, dtype, mean, std)

def attrs(var):
    return {
        "is_stop_fuse": var.is_stop_fuse(),
        "is_stop_grad": var.is_stop_grad(),
        "shape": var.shape,
        "dtype": var.dtype,
    }
Var.attrs = attrs

def fetch(*args):
    ''' Async fetch vars with function closure.

Example 1::

    for img,label in enumerate(your_dataset):
        pred = your_model(img)
        loss = critic(pred, label)
        acc = accuracy(pred, label)
        jt.fetch(acc, loss,
            lambda acc, loss:
                print(f"loss:{loss} acc:{acc}"
        )

Example 2::

    for i,(img,label) in enumerate(your_dataset):
        pred = your_model(img)
        loss = critic(pred, label)
        acc = accuracy(pred, label)
        # variable i will be bind into function closure
        jt.fetch(i, acc, loss,
            lambda i, acc, loss:
                print(f"#{i}, loss:{loss} acc:{acc}"
        )
    '''
    assert len(args)>=1
    func = args[-1]
    assert callable(func)
    args = list(args[:-1])
    if len(args)>0 and isinstance(args[0], Sequence) \
        and len(args[0])>=1 and isinstance(args[0][0], Var):
        raise TypeError("jt.Var should not inside a list or tuple.")

    var_map = []
    variables = []
    for i, v in enumerate(args):
        if isinstance(v, Var):
            variables.append(v)
            var_map.append(i)
            args[i] = None
    def callback(*results):
        for i,v in enumerate(results):
            args[var_map[i]] = v
        func(*args)
    core.ops.fetch(variables, callback)

Var.fetch = fetch

def display_memory_info():
    import inspect, os
    f = inspect.currentframe()
    fileline = inspect.getframeinfo(f.f_back)
    fileline = f"{os.path.basename(fileline.filename)}:{fileline.lineno}"
    core.display_memory_info(fileline)

def load(path: str):
    ''' loads an object from a file.
    '''
    model_dict = safeunpickle(path)
    return model_dict

def save(params_dict, path: str):
    ''' saves the parameter dictionary to a file.

    :param params_dict: parameters to be saved
    :type params_dict: list or dictionary
    :param path: file path
    :type path: str
    '''
    safepickle(params_dict, path)

def _uniq(x):
    a = set()
    b = []
    for i in x:
        j = id(i)
        if j not in a:
            a.add(j)
            b.append(i)
    return b

class _RemovableHandle:
    ''' torch-compatible handle returned by ``register_forward_hook`` etc.

    Calling ``.remove()`` (idempotent) detaches the hook. Also usable as a
    context manager, mirroring ``torch.utils.hooks.RemovableHandle``.
    '''
    def __init__(self, remove_fn):
        self._remove_fn = remove_fn
    def remove(self):
        if self._remove_fn is not None:
            self._remove_fn()
            self._remove_fn = None
    def __enter__(self):
        return self
    def __exit__(self, *a):
        self.remove()
        return False

class _WriteThroughDict(dict):
    ''' A dict view of a Module's Var attributes whose item-assignment writes back
    to the owning module. jittor's ``_parameters``/``_buffers`` are properties that
    build a fresh dict each access, so torch/accelerate's idiom
    ``module._parameters[name] = value`` (used by accelerate's
    set_module_tensor_to_device on the from_pretrained meta/low_cpu_mem_usage fast
    path) would write into a throwaway dict and be LOST -> the model keeps its
    construction-time weights instead of the checkpoint's. Writing through to
    ``setattr(owner, name, value)`` makes the assignment actually take effect. '''
    def __init__(self, owner, items):
        super().__init__(items)
        object.__setattr__(self, "_owner", owner)
    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        # Preserve the buffer/persistent classification of the attribute being
        # replaced, so a registered buffer stays a buffer (not reclassified as a
        # trainable parameter) when accelerate reassigns it from the checkpoint.
        old = getattr(self._owner, k, None)
        if old is not None and isinstance(v, Var):
            for _flag in ("is_buffer", "persistent"):
                if hasattr(old, _flag):
                    try: setattr(v, _flag, getattr(old, _flag))
                    except Exception: pass
        object.__setattr__(self._owner, k, v)
    def __delitem__(self, k):
        super().__delitem__(k)
        if hasattr(self._owner, k):
            object.__delattr__(self._owner, k)

# Jittor registers a parameter on assignment; torch registers one only for an
# nn.Parameter and leaves a plain tensor attribute unregistered. Under the shim a
# model tree mixes code written to each convention, so the two have to coexist:
# jittor's own nn.Linear declares its weight by plain assignment, while
# torch-authored code relies on `layer.foo = torch.tensor(...)` staying out of
# parameters()/state_dict() -- vLLM's attention layer does exactly that with its
# q/k/v_range scales, and its weight loader then reports them as checkpoint
# weights that were never initialised.
#
# Only a value carrying POSITIVE evidence of being a plain torch tensor is
# demoted, i.e. one the shim's own torch.tensor produced. Deciding it the other
# way round -- treat anything unmarked as plain -- reads far more assignments as
# non-parameters than intended, and the failure mode is a silently dropped weight:
# it demoted vLLM's `self.bias = Parameter(torch.empty(...))` (whose marker an
# adapter's replaced constructor had lost) until uninitialised memory reached a
# matmul, and it demoted `self.output_bias = jt.ones(...)` in jittor-style code
# that merely happened to run with the shim loaded. torch.ones/zeros cannot be
# used as the signal either: they ARE jittor's own, so marking them would demote
# the weights jittor's layers declare by assignment.
# The evidence is carried by the VALUE, not by a process-global mode bit. There
# used to be one (`_torch_registration_semantics`, set to True at the end of the
# shim's install), so that the meaning of `module.x = var` depended on whether
# some other import had run: the same assignment registered a parameter or did
# not, and nothing in the module tree said which. The marker below is only ever
# attached by the shim's own `torch.tensor`, so consulting it directly is exactly
# as narrow and no longer makes the kernel's behaviour a global.
def _is_plain_tensor(value):
    return value.__dict__.get("_jt_plain_tensor") is True


#: What a ``Var`` attribute of a Module is. Five methods -- parameters(),
#: named_parameters(), state_dict(), named_buffers() and _buffers -- used to
#: answer this question with five transcriptions of the same DFS, and the
#: answers had drifted: parameters() de-duplicated by ``id`` while
#: named_parameters() de-duplicated by NAME, so a model with tied weights told
#: the optimizer one parameter count and transformers/peft another; and
#: ``_parameters``/``_buffers`` both returned EVERY Var, so neither said
#: anything. ``Module._var_roles`` is now the single answer and all five are
#: views over one traversal (``Module._named_vars``).
_ROLE_PARAMETER = "parameter"
_ROLE_BUFFER = "buffer"
_ROLE_NON_PERSISTENT_BUFFER = "non_persistent_buffer"
_ROLE_PLAIN = "plain"

#: The roles each view reports. "state" is parameters plus persistent buffers,
#: which is torch's definition of a module's state.
_VIEW_ROLES = {
    "parameters": frozenset((_ROLE_PARAMETER,)),
    "buffers": frozenset((_ROLE_BUFFER, _ROLE_NON_PERSISTENT_BUFFER)),
    "state": frozenset((_ROLE_PARAMETER, _ROLE_BUFFER)),
}

class Module:
    def __init__(self, *args, **kw):
        pass
    def execute(self, *args, **kw):
        ''' Executes the module computation.

        Raises NotImplementedError if the subclass does not override the method.
        '''
        raise NotImplementedError("Please implement 'execute' method of "+str(type(self)))

    def __call__(self, *args, **kw):
        # Hooks are per-INSTANCE and are consulted here, so registering one no
        # longer rewrites the class. See ``_hooks``.
        if self._has_hooks():
            return self.__hooked_call__(*args, **kw)
        return self._dispatch_call(*args, **kw)

    def _dispatch_call(self, *args, **kw):
        """What ``__call__`` runs once the hooks have had their say.

        This, not ``__call__``, is the seam the torch compatibility layer
        replaces, so its dispatch (an instance-level ``forward``, fsdp,
        execution pipelining) keeps running INSIDE the hooks -- which is where
        it ran when a hook was installed by swapping ``cls.__call__``.
        """
        return self.execute(*args, **kw)
    def __repr__(self):
        return self.__str__()
    def _get_name(self):
        return self.__class__.__name__
    def __name__(self):
        pass

    def dfs(self, parents, k, callback, callback_leave=None, recurse=True):
        ''' An utility function to traverse the module. '''
        # One pass over ``__dict__`` rather than two. The count handed to the
        # callback and the set the recursion walks are the same children, and
        # this runs once per module on every parameters(), state_dict() and
        # named_modules() call -- the hottest Python in a training step, where a
        # module tree is walked more than once per iteration. ``ModuleList``
        # already overrides dfs in exactly this shape.
        children = [(key, value) for key, value in self.__dict__.items()
                    if isinstance(value, Module)]
        ret = callback(parents, k, self, len(children))
        if ret == False: return
        if recurse:
            parents.append(self)
            for key, value in children:
                value.dfs(parents, key, callback, callback_leave)
            parents.pop()
        if callback_leave:
            # ``k`` names this module. The previous loop bound its own key to the
            # same name, leaving the last entry of ``__dict__`` here instead.
            callback_leave(parents, k, self, len(children))

    def __str__(self):
        ss = []
        def callback(parents, k, v, n):
            # indent key:class_name(extra_repr)
            k = f"{k}: " if k is not None else ""
            s = f"{' '*(len(parents)*4)}{k}{v.__class__.__name__}"
            if n:
                s += '('
            else:
                s += f"({v.extra_repr()})"
            ss.append(s)
        def callback_leave(parents, k, v, n):
            if n:
                ss.append(' '*(len(parents)*4)+')')
        self.dfs([], None, callback, callback_leave)
        return "\n".join(ss)

    def _var_attrs(self):
        ''' The ``(key, Var)`` attributes this module owns, in declaration order.

        The one place that says where a module keeps its Vars. ``ParameterList``
        keeps them in ``self.params`` and overrides this; every traversal used to
        carry its own ``if isinstance(v, ParameterList): dc = v.params``.
        '''
        return [(k, v) for k, v in self.__dict__.items()
                if isinstance(v, Var) and not (type(k) is str and k[:1] == "_")]

    def _var_roles(self):
        ''' Classify every Var this module owns: ``[(key, var, role)]``.

        The single definition of "is this a parameter, a buffer, or neither".

        Registration is by NAME (``_buffer_names`` / ``_non_persistent_buffer_names``
        / ``_non_parameter_names``, the sets ``register_buffer`` and ``__setattr__``
        maintain) because a name survives what a per-Var tag does not: from_pretrained's
        dtype cast REPLACES the Var behind an attribute, and the fresh one carries no
        tag. The per-Var ``is_buffer``/``persistent`` tags are still honoured, for
        modules that set them directly.
        '''
        d = self.__dict__
        buffer_names = d.get("_buffer_names", ())
        non_persistent = d.get("_non_persistent_buffer_names", ())
        non_parameters = d.get("_non_parameter_names", ())
        # A second attribute pointing at a Var that register_buffer() already named
        # is an ALIAS of that buffer -- not a second buffer, and not a parameter.
        # torch has no such case at all: only the _buffers entry counts there.
        registered = ({id(d[n]) for n in buffer_names if isinstance(d.get(n), Var)}
                      if buffer_names else ())
        out = []
        for key, var in self._var_attrs():
            attrs = var.__dict__
            if key in buffer_names:
                role = (_ROLE_NON_PERSISTENT_BUFFER if key in non_persistent
                        else _ROLE_BUFFER)
            elif key in non_parameters:
                role = _ROLE_PLAIN
            elif attrs.get("is_buffer") is True:
                if id(var) in registered:
                    role = _ROLE_PLAIN
                elif attrs.get("persistent") is False:
                    role = _ROLE_NON_PERSISTENT_BUFFER
                else:
                    role = _ROLE_BUFFER
            elif attrs.get("persistent") is False:
                role = _ROLE_NON_PERSISTENT_BUFFER
            else:
                role = _ROLE_PARAMETER
            out.append((key, var, role))
        return out

    def _named_vars(self, kind="parameters", recurse=True, remove_duplicate=True):
        ''' One traversal of the module tree; every public view is a filter over it.

        :param kind: which roles to report -- ``"parameters"``, ``"buffers"`` or
            ``"state"`` (see ``_VIEW_ROLES``).
        :param remove_duplicate: keep only the FIRST name of a Var reachable under
            several names. That is torch's rule for parameters()/named_parameters()/
            named_buffers(), and it has to be by object identity: de-duplicating by
            name (which ``named_parameters`` did) does not de-duplicate a tied weight
            at all, since its two names differ. ``state_dict`` passes False, because
            torch writes a tied weight under every name it is registered as -- and
            de-duplicating there made the surviving key depend on ``__dict__`` order.

        The name is built from the traversal path and is NOT written back to the Var.
        parameters() and state_dict() used to call ``p.name(...)`` when the path they
        happened to be walking was longer than the name already stored, so a query
        mutated the model and the resulting checkpoint keys depended on which level
        of the tree someone had called parameters() from first.
        '''
        roles = _VIEW_ROLES[kind]
        out = []
        stack = []
        seen = set() if remove_duplicate else None
        def callback(parents, k, v, n):
            stack.append(str(k))
            prefix = ".".join(stack[1:])
            for key, var, role in v._var_roles():
                if role not in roles: continue
                if seen is not None:
                    if id(var) in seen: continue
                    seen.add(id(var))
                leaf = key if type(key) is str else str(key)
                out.append((prefix + "." + leaf if prefix else leaf, var))
        def callback_leave(parents, k, v, n):
            stack.pop()
        self.dfs([], None, callback, callback_leave, recurse)
        return out

    def parameters(self, recurse=True) -> List:
        ''' Returns a list of module parameters.

        A Var reachable under more than one name (a tied weight) is returned once.

        ----------------

        Example::

            >>> net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 2))
            >>> for name, p in net.named_parameters():
            ...     print(name)
            ...
            0.weight
            0.bias
            2.weight
            2.bias
        '''
        return [v for _, v in self._named_vars("parameters", recurse)]

    def state_dict(self, to=None, recurse=True, destination=None, prefix="",
                   keep_vars=None):
        ''' Returns a dictionary containing
        Jittor Var of the module and its descendants.

        Args:
            to: target type of var, canbe None or 'numpy' or 'torch'
            destination: optional mapping to write the entries into and return,
                matching ``torch.nn.Module.state_dict``. Wrapper modules such as
                ms-swift's tuners forward this through to the wrapped model.
            prefix: string prepended to every key, also matching Torch.
            keep_vars: Torch detaches its tensors when this is ``False``. Jittor
                has always returned live ``Var`` objects, so the default stays
                ``None`` (historical behaviour) and only an explicit ``False``
                detaches.

        Return:
            dictionary of module's states.

        Example::

            import jittor as jt
            from jittor.models import resnet50
            jittor_model = resnet50()
            dict = jittor_model.state_dict()
            jittor_model.load_state_dict(dict)

        Example2(export Jittor params to PyTorch)::

            import jittor as jt
            from jittor.models import resnet50
            jittor_model = resnet50()
            import torch
            from torchvision.models import resnet50
            torch_model = resnet50()
            torch_model.load_state_dict(jittor_model.state_dict(to="torch"))

        '''
        # A tied weight is written under EVERY name it is registered as, like
        # torch. De-duplicating by id kept only whichever name ``__dict__`` order
        # happened to reach first, so the checkpoint silently lost the other key.
        ps = dict(self._named_vars("state", recurse, remove_duplicate=False))
        if keep_vars is False:
            for k, v in ps.items():
                if isinstance(v, Var):
                    ps[k] = v.detach()
        if to == "numpy":
            for k,v in ps.items():
                if isinstance(v, Var):
                    ps[k] = v.numpy()
        elif to == "torch":
            import torch
            for k,v in ps.items():
                if isinstance(v, Var):
                    # from_numpy, not torch.Tensor(...): the latter is
                    # torch.FloatTensor, so it casts EVERY entry to float32.
                    # A state dict carries integer and bool buffers as well as
                    # float weights -- num_batches_tracked, attention masks,
                    # quantisation zero-points -- and load_state_dict on the
                    # torch side then either rejects them or silently keeps the
                    # float copy. from_numpy preserves the dtype, and the array
                    # v.numpy() returns is already a fresh copy, so sharing its
                    # storage costs nothing.
                    ps[k] = torch.from_numpy(v.numpy())
        if prefix:
            ps = {prefix + k: v for k, v in ps.items()}
        if destination is not None:
            destination.update(ps)
            return destination
        return ps

    def named_parameters(self, recurse=True) -> List[Tuple[str, Var]]:
        ''' Returns a list of module parameters and their names.

        The same Vars ``parameters()`` returns, in the same order, each under the
        first name the traversal reaches it by. The two used to disagree: this one
        de-duplicated by NAME, which does not de-duplicate a tied weight at all,
        so a model whose embedding and output projection share a weight reported
        one parameter count to the optimizer and a larger one here.

        ----------------

        Example::

            >>> net = nn.Linear(2, 5)
            >>> net.named_parameters()
            [('weight', jt.Var([[ 0.5964666  -0.3175258 ]
            [ 0.41493994 -0.66982657]
            [-0.32677156  0.49614117]
            [-0.24102807 -0.08656466]
            [ 0.15868133 -0.12468725]], dtype=float32)),
            ('bias', jt.Var([-0.38282675  0.36271113 -0.7063226   0.02899247  0.52210844], dtype=float32))]

        '''
        return self._named_vars("parameters", recurse)

    def load_state_dict(self, params) -> None:
        '''
        Loads the module's parameters from a dictionary.
        '''
        self.load_parameters(params)

    def _load_from_state_dict(self, state, prefix="", *args, **kw):
        if len(prefix):
            new_state = {}
            for k,v in state.items():
                if k.startswith(prefix):
                    new_state[k[len(prefix):]] = v
            state = new_state
        self.load_state_dict(state)

    def cuda(self, device=None):
        '''Move every parameter and buffer to a CUDA device, in place.'''
        return self._move_to_accelerator("cuda", device)

    def npu(self, device=None):
        '''Move every parameter and buffer to an NPU device, in place.'''
        return self._move_to_accelerator("npu", device)

    def _move_to_accelerator(self, method, device):
        if method == "npu" and not getattr(jt.compiler, "has_acl", False):
            raise RuntimeError("NPU backend is unavailable")
        values = self._named_vars("parameters") + self._named_vars("buffers")
        seen = set()
        for _, value in values:
            if id(value) in seen:
                continue
            seen.add(id(value))
            moved = getattr(value, method)(device)
            if moved is not value:
                # Module moves are in place: optimizers and state dictionaries
                # keep referring to the same Var object while its value moves.
                value.assign(moved)
        return self

    def modules(self) -> List:
        ''' Returns a list of sub-modules in the module recursively.

        ----------------

        Example::

            >>> net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 2))
            >>> net.modules()
            [Sequential(
                0: Linear(2, 10, float32[10,], None)
                1: relu()
                2: Linear(10, 2, float32[2,], None)
            ), Linear(2, 10, float32[10,], None), relu(), Linear(10, 2, float32[2,], None)]
        '''
        ms = []
        def callback(parents, k, v, n):
            if isinstance(v, Module):
                ms.append(v)
        self.dfs([], None, callback, None)
        return _uniq(ms)

    def named_modules(self):
        ''' Returns a list of sub-modules and their names recursively.

        ----------------

        Example::

            >>> net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 2))
            >>> net.named_modules()
            [('', Sequential(
                0: Linear(2, 10, float32[10,], None)
                1: relu()
                2: Linear(10, 2, float32[2,], None)
            )), ('0', Linear(2, 10, float32[10,], None)), ('1', relu()), ('2', Linear(10, 2, float32[2,], None))]
        '''
        ms = []
        stack = []
        def callback(parents, k, v, n):
            if isinstance(v, Module):
                stack.append(str(k))
                name = ".".join(stack[1:])
                ms.append((name, v))
        def callback_leave(parents, k, v, n):
            stack.pop()
        self.dfs([], "", callback, callback_leave)
        return ms

    def add_module(self, name, module):
        setattr(self, name ,module)
        return self

    @property
    def _modules(self):
        return { k:v for k,v in self.__dict__.items() if isinstance(v, Module) }

    @property
    def _parameters(self):
        # This module's own parameters, keyed by attribute name -- torch's
        # ``_parameters``. It used to return EVERY Var, exactly like ``_buffers``,
        # so accelerate's `is_buffer = name in module._buffers` was True for the
        # weights too. write-through: accelerate's `module._parameters[name] = value`
        # has to reach the attribute (see _WriteThroughDict).
        return _WriteThroughDict(self, self._named_vars("parameters", recurse=False))

    def requires_grad_(self, requires_grad=True):
        ''' Sets requires_grad for all parameters and sub-modules.

        torch semantics: this toggles every PARAMETER leaf's ``requires_grad`` and
        does NOT gate the module's forward. The previous jittor behavior of running
        the whole forward under ``no_grad`` whenever the module flag was False is
        incompatible with the freeze-then-unfreeze-a-subset pattern used by LoRA /
        adapters (peft freezes the base model with ``requires_grad_(False)`` and then
        re-enables only the adapter params): wrapping the forward in ``no_grad``
        severs the autograd graph, so the re-enabled adapter params -- and any
        upstream trainable tensors -- receive zero gradient. Toggling the leaves
        instead keeps frozen weights frozen while letting gradients flow through the
        module to whatever is still trainable.
        '''
        self._requires_grad = requires_grad
        # propagate to every parameter leaf (recurse through sub-modules), matching
        # torch.nn.Module.requires_grad_.
        for p in self.parameters():
            try:
                p.requires_grad = requires_grad
            except Exception:
                pass
        return self

    #: Hook ids come from one counter, so a handle names exactly one hook even
    #: after everything around it has been removed.
    _hook_serial = itertools.count()

    def _hooks(self, name, create=False):
        """This INSTANCE's ordered hook table for ``name``.

        Hooks used to be single attributes (``self.__fhook__ = func``), and
        installing one called ``_place_hooker``, which swapped ``__call__`` and
        ``__hooked_call__`` **on the class**. Four things followed, none of them
        announced:

        * a second ``register_forward_hook`` **replaced** the first. accelerate,
          peft and transformers all register several hooks on one module and
          rely on torch's ordered dict; the earlier ones simply stopped
          existing.
        * ``prepend`` and ``always_call`` were accepted and ignored, so a caller
          who asked for ordering got registration order, silently.
        * the swap was **class-level and permanent**: hooking one ``Linear``
          put every ``Linear`` in the process on the hook path forever, and
          ``handle.remove()`` could not undo it because it only deleted the
          attribute the wrapper looks for.
        * the guard was ``hasattr(cls, "__hooked__")``, which walks the MRO, so
          whether a subclass installed its own wrapper depended on whether some
          base had been hooked first.

        The table lives in ``__dict__`` directly: ``Module.__setattr__``
        classifies assignments into parameters and buffers, and a hook table is
        neither.
        """
        d = self.__dict__.get(name)
        if d is None and create:
            d = OrderedDict()
            self.__dict__[name] = d
        return d

    def _has_hooks(self):
        # NB: no bool() -- ``from jittor import *`` at the top of this module
        # rebinds the name to jittor's `bool` CAST OP, which raises on a dict.
        # The `or` chain already returns something falsy when every table is
        # missing or empty, which is all `__call__` asks.
        d = self.__dict__
        return (d.get("_forward_pre_hooks") or d.get("_forward_hooks")
                or d.get("_input_backward_hooks")
                or d.get("_output_backward_hooks"))

    def _add_hook(self, name, func, prepend=False, **info):
        """Append (or prepend) one hook and return the handle that removes it."""
        hooks = self._hooks(name, create=True)
        key = next(Module._hook_serial)
        hooks[key] = (func, info)
        if prepend:
            hooks.move_to_end(key, last=False)
        return _RemovableHandle(lambda: hooks.pop(key, None))

    def _run_forward_hooks(self, hooks, args, kw, ret):
        for func, info in hooks:
            if info.get("with_kwargs"):
                res = func(self, args, kw, ret)
            else:
                res = func(self, args, ret)
            if res is not None:
                ret = res
        return ret

    def __hooked_call__(self, *args, **kw):
        pre = self._hooks("_forward_pre_hooks")
        for func, info in list(pre.values()) if pre else ():
            # torch's forward_pre_hook convention:
            #   default:          hook(module, args) -> None | new_args
            #   with_kwargs=True: hook(module, args, kwargs) -> None | (new_args, new_kwargs)
            # When the hook was registered with_kwargs it must ALWAYS get the kwargs
            # arg (even if empty) -- ms-swift's VL pre_forward_hook has a 3-arg
            # signature and injects inputs_embeds via the kwargs dict.
            if info.get("with_kwargs") or len(kw):
                args_kw_result = func(self, args, kw)
            else:
                args_kw_result = func(self, args)
            if args_kw_result is None:
                continue
            if info.get("with_kwargs"):
                # with_kwargs: torch requires a (new_args, new_kwargs) pair.
                if isinstance(args_kw_result, tuple) and len(args_kw_result) == 2:
                    args, kw = args_kw_result
                else:
                    raise RuntimeError(
                        "forward pre-hook with kwargs must return None or a tuple "
                        f"of (new_args, new_kwargs), but got {args_kw_result}."
                    )
            else:
                # no kwargs: torch replaces args with the return value, wrapping a
                # single non-tuple return in a 1-tuple.
                if not isinstance(args_kw_result, tuple):
                    args_kw_result = (args_kw_result,)
                args = args_kw_result
        bihooks = self._hooks("_input_backward_hooks")
        if bihooks:
            if len(kw):
                LOG.w("backward hook not support kw")
            for func, _info in list(bihooks.values()):
                args = grad_hooker(args, func)
        # NB: do NOT wrap the forward in no_grad when `_requires_grad` is False.
        # torch's requires_grad_(False) freezes parameters, it does not stop the
        # forward from building the autograd graph; gating here severs grad for
        # re-enabled sub-params (LoRA adapters) and upstream trainables. Freezing is
        # enforced at the parameter level (see Module.requires_grad_).
        fhooks = self._hooks("_forward_hooks")
        fhooks = list(fhooks.values()) if fhooks else []
        try:
            ret = self._dispatch_call(*args, **kw)
        except BaseException:
            # torch's always_call=True: the hook runs even when the forward
            # raised. It gets None for the output there -- there is none.
            # accelerate's offload hooks use this to move weights back off the
            # GPU, so skipping it leaks device memory on every failed step.
            always = [(f, i) for f, i in fhooks if i.get("always_call")]
            if always:
                self._run_forward_hooks(always, args, kw, None)
            raise
        bohooks = self._hooks("_output_backward_hooks")
        if bohooks:
            if len(kw):
                LOG.w("backward hook not support kw")
            for func, _info in list(bohooks.values()):
                if isinstance(ret, Var):
                    ret = grad_hooker((ret,), func)[0]
                else:
                    ret = grad_hooker(ret, func)
        # Match torch's forward-hook calling convention:
        #   default:            hook(module, args, output)
        #   with_kwargs=True:   hook(module, args, kwargs, output)
        # (torch passes kwargs *before* output). Older jittor code called
        # the hook with 4 positional args whenever kwargs were present,
        # which breaks every plain 3-arg torch hook -- e.g. transformers'
        # output_hidden_states / output_attentions paths.
        return self._run_forward_hooks(fhooks, args, kw, ret)

    def register_forward_hook(self, func, *, prepend=False, with_kwargs=False, always_call=False):
        ''' Register a forward function hook that will be called after Module.execute.

        Follows torch's calling convention. By default the hook is called as::

            hook(module, input_args, output)

        If ``with_kwargs=True`` it is called as (torch passes kwargs before
        output)::

            hook(module, input_args, input_kwargs, output)

        If the hook returns a value it replaces the module output -- and the
        next hook sees the replacement. Any number of hooks may be registered;
        they run in registration order, or first if ``prepend=True``. With
        ``always_call=True`` the hook also runs when the forward raises (with
        ``None`` for the output). Returns a handle whose ``.remove()`` detaches
        this hook and only this hook.
        '''
        # NB: don't call bool() here -- the torch-compat layer rebinds the name
        # ``bool`` in this module's globals to a dtype object; use truthiness.
        return self._add_hook(
            "_forward_hooks", func, prepend=prepend,
            with_kwargs=True if with_kwargs else False,
            always_call=True if always_call else False)

    def remove_forward_hook(self):
        ''' Removes EVERY forward hook on this module.

        jittor's documented spelling, kept as-is. To drop one hook, use the
        handle its ``register_forward_hook`` returned.
        '''
        self.__dict__.pop("_forward_hooks", None)

    def register_pre_forward_hook(self, func):
        ''' Register a forward function hook that will be called before Module.execute.

        The hook function will be called with the following arguments::

            hook(module, input_args)
        or::
            hook(module, input_args, input_kwargs)

        Returns a removable handle, like ``register_forward_pre_hook``.
        '''
        return self._add_hook("_forward_pre_hooks", func, with_kwargs=False)

    def register_forward_pre_hook(self, func, *, prepend=False, with_kwargs=False):
        ''' torch-compatible alias of the pre-forward hook.

        transformers / peft / ms-swift call ``register_forward_pre_hook`` (torch's
        spelling) -- notably ms-swift's multimodal template registers one with
        ``with_kwargs=True`` to swap ``input_ids`` for ``inputs_embeds`` before the
        forward. Mirrors torch's signature and returns a ``.remove()``-able handle.
        With ``with_kwargs=True`` the hook is called as ``hook(module, args, kwargs)``
        and may return ``(new_args, new_kwargs)``; otherwise ``hook(module, args)``
        returning ``None`` or replacement args. Several may be registered; each
        sees the arguments the one before it returned.
        '''
        return self._add_hook("_forward_pre_hooks", func, prepend=prepend,
                              with_kwargs=True if with_kwargs else False)

    def remove_pre_forward_hook(self):
        ''' Removes EVERY pre-forward hook on this module. '''
        self.__dict__.pop("_forward_pre_hooks", None)

    def register_input_backward_hook(self, func):
        ''' Hook the gradients flowing into this module. Returns a handle. '''
        return self._add_hook("_input_backward_hooks", func)

    def remove_input_backward_hook(self):
        self.__dict__.pop("_input_backward_hooks", None)

    def register_output_backward_hook(self, func):
        ''' Hook the gradients flowing out of this module. Returns a handle. '''
        return self._add_hook("_output_backward_hooks", func)

    def remove_output_backward_hook(self):
        self.__dict__.pop("_output_backward_hooks", None)

    def register_backward_hook(self, func):
        ''' hook both input and output on backpropergation of this module.

Arguments of hook are defined as::

    hook(module, grad_input:tuple(jt.Var), grad_output:tuple(jt.Var)) -> tuple(jt.Var) or None

`grad_input` is the origin gradients of input of this module, `grad_input` is the  gradients of output of this module, return value is used to replace the gradient of input.

Returns a handle that removes both halves.
        '''
        _grad_output = None
        def bohook(grad_output):
            nonlocal _grad_output
            _grad_output = grad_output
        def bihook(grad_input):
            return func(self, grad_input, _grad_output)
        bi = self.register_input_backward_hook(bihook)
        bo = self.register_output_backward_hook(bohook)
        def _remove_both():
            bi.remove()
            bo.remove()
        return _RemovableHandle(_remove_both)

    def remove_backward_hook(self):
        ''' Removes the backward input and output hooks.
        '''
        self.remove_input_backward_hook()
        self.remove_output_backward_hook()

    def _place_hooker(self):
        """No longer needed; kept so out-of-tree callers do not break.

        This used to swap ``__call__`` and ``__hooked_call__`` on the class,
        which is what made hook installation class-wide and irreversible.
        ``__call__`` consults the instance's hook tables itself now, so there
        is nothing to place.
        """

    def children(self) -> List:
        ''' Returns an List of the children modules. '''
        cd = []
        def callback(parents, k, v, n):
            if len(parents) == 1 and isinstance(v, Module):
                cd.append(v)
                return False
        self.dfs([], None, callback, None)
        return cd

    def extra_repr(self):
        # Reentrancy guard: extra_repr introspects __init__ args and str()'s their
        # values. When a value is itself a sub-module (e.g. peft wraps a
        # `base_layer`, or passes ModuleDicts), str()'ing it re-enters the full
        # __str__/dfs traversal -- which calls extra_repr again -- re-walking the
        # subtree at every node => exponential blowup / RecursionError on deep
        # wrapped models. torch's extra_repr never recurses into sub-modules; mirror
        # that by short-circuiting any nested extra_repr triggered mid-render.
        if getattr(Module, "_in_extra_repr", False):
            return ""
        Module._in_extra_repr = True
        try:
            ss = []
            n = len(self.__init__.__code__.co_varnames)
            if self.__init__.__defaults__ is not None:
                n -= len(self.__init__.__defaults__)
            for i, k in enumerate(self.__init__.__code__.co_varnames[1:]):
                v = getattr(self, k) if hasattr(self, k) else None
                if isinstance(v, Var): v = v.peek()
                s = f"{k}={v}" if i >= n else str(v)
                ss.append(s)
        finally:
            Module._in_extra_repr = False
        return ", ".join(ss)

    def apply(self, func):
        ''' Applies a function to all sub-modules recursively. '''
        for m in self.modules():
            func(m)

    def load_parameters(self, params):
        ''' loads parameters to the Module.

        :param params: dictionary of parameter names and parameters.
        '''
        n_failed = 0
        for key in params.keys():
            v = self
            key_ = key.split('.')
            end = 0
            for k in key_:
                if isinstance(v, jt.nn.Sequential):
                    if (k in v.layers):
                        v = v[k]
                    elif k.isdigit() and (ori_int(k) in v.layers):
                        v = v[ori_int(k)]
                    else:
                        end=1
                        break
                else:
                    if hasattr(v, k):
                        v = getattr(v, k)
                        if v is None:
                            continue
                        assert isinstance(v, (Module, Var)), \
                            f"expect a jittor Module or Var, but got <{v.__class__.__name__}>, key: {key}"
                    else:
                        end = 1
                        break
            if end == 1:
                if not key.endswith("num_batches_tracked"):
                    n_failed += 1
                    LOG.w(f'load parameter {key} failed ...')
            else:
                assert isinstance(v, Var), \
                    f"expect a jittor Var, but got <{v.__class__.__name__}>, key: {key}"
                if isinstance(params[key], np.ndarray) or isinstance(params[key], list):
                    param = array(params[key])
                elif isinstance(params[key], Var):
                    param = params[key]
                else:
                    # assume is pytorch tensor
                    param = array(params[key].cpu().detach().numpy())
                if param.shape == v.shape:
                    LOG.v(f'load parameter {key} success ...')
                    v.update(param)
                    v.sync(False, False)
                else:
                    n_failed += 1
                    LOG.e(f'load parameter {key} failed: expect the shape of {key} to be {v.shape}, but got {param.shape}')
        if n_failed:
            LOG.w(f"load total {len(params)} params, {n_failed} failed")

    def save(self, path: str):
        ''' saves parameters to a file.

        :param path: path to save.
        :type path: str

        Example::

            >>> class Net(nn.Module):
            >>> ...
            >>> net = Net()
            >>> net.save('net.pkl')
            >>> net.load('net.pkl')
        '''
        params = self.state_dict()
        # Convert Vars to numpy before pickling. Pickling jittor Vars directly recurses
        # under the torch-compat layer (the Parameter/.grad bridge creates a reference
        # cycle), so model.save() RecursionError'd on torch-as-jittor. numpy values are
        # portable and load_state_dict/load() restore them; a fresh dict, model untouched.
        params = {k: (v.numpy() if isinstance(v, Var) else v) for k, v in params.items()}
        safepickle(params, path)

    def load(self, path: str):
        ''' loads parameters from a file.

        :param path: path to load.
        :type path: str

        Example::

            >>> class Net(nn.Module):
            >>> ...
            >>> net = Net()
            >>> net.save('net.pkl')
            >>> net.load('net.pkl')

        This method also supports loading a state dict from a pytorch .pth file.

        .. note::
            当载入的参数与模型定义不一致时, jittor 会输出错误信息, 但是不会抛出异常.
            若载入参数出现模型定义中没有的参数名, 则会输出如下信息, 并忽略此参数:

            >>> [w 0205 21:49:39.962762 96 __init__.py:723] load parameter w failed ...

            若载入参数的 shape 与模型定义不一致, 则会输出如下信息, 并忽略此参数:

            >>> [e 0205 21:49:39.962822 96 __init__.py:739] load parameter w failed: expect the shape of w to be [1000,100,], but got [3,100,100,]

            如载入过程中出现错误, jittor 会输出概要信息, 您需要仔细核对错误信息

            >>> [w 0205 21:49:39.962906 96 __init__.py:741] load total 100 params, 3 failed
        '''
        self.load_parameters(load(path))

    def _set_training(self, is_train):
        """Flip ``is_train`` on this module and every sub-module.

        That is ALL train()/eval() do, as in torch: the flag decides what
        BatchNorm and Dropout do, and nothing else. Freezing is a separate
        thing, spelled ``requires_grad``.

        ``eval()`` used to also call ``stop_grad()`` on every parameter and
        remember, in a dict keyed by ``id(p)``, which ones ``train()`` should
        later ``start_grad()``. Three things were wrong with that, all silent:

        * **torch's eval() does not freeze anything.** Evaluating a loss with
          gradients (adversarial examples, Grad-CAM, meta-learning inner loops,
          any "eval then backprop" script ported from torch) got no gradient at
          all after ``model.eval()``, with nothing said. And ``stop_grad()`` is
          documented in ``var_holder.h`` as *permanent* -- ``start_grad()``
          does not undo it, it detaches and swaps in a NEW Var node.

        * **``id(p)`` is not an identity.** CPython reuses the id of a
          collected object, so after a Var was replaced (a dtype cast, a
          checkpoint load -- ``from_pretrained`` does both) or garbage
          collected, ``train()`` looked up an unrelated Var's entry and
          restored the wrong answer. The dict also only ever grew.

        * **The backup lives on whichever module you called eval() on.**
          ``child.eval()`` then ``parent.train()`` found no backup on the
          parent, so the child's parameters stayed frozen for the rest of the
          process -- the model trained, that sub-tree did not, and the loss
          curve was the only hint.

        Deliberate freezing was destroyed too: ``requires_grad = False`` is a
        different, reversible flag, but ``eval()``'s ``stop_grad()`` clears it
        and records "was trainable", so an eval/train round trip silently
        UNFROZE a parameter the caller had frozen on purpose.

        For the memory that the old ``eval()`` incidentally saved, use what
        torch users use: ``with jt.no_grad():`` around inference.
        """
        def callback(parents, k, v, n):
            if isinstance(v, Module):
                v.is_train = is_train
        self.dfs([], None, callback, None)
        return self

    def eval(self):
        ''' Sets the module in evaluation mode.

        Only the training flag changes -- BatchNorm switches to its running
        statistics and Dropout becomes a no-op. Parameters are NOT frozen; use
        ``requires_grad_(False)`` to freeze, or ``with jt.no_grad():`` to skip
        building the graph. See ``_set_training``. '''
        return self._set_training(False)

    def train(self):
        ''' Sets the module in training mode.

        The mirror of ``eval()``: only the training flag changes. It does not
        unfreeze anything, so a parameter frozen with ``requires_grad_(False)``
        stays frozen. See ``_set_training``. '''
        return self._set_training(True)

    def is_training(self) -> bool:
        ''' Returns whether the module is in training mode.'''
        if not hasattr(self, "is_train"):
            self.is_train = True
        return self.is_train

    @property
    def training(self):
        if not hasattr(self, "is_train"):
            self.is_train = True
        return self.is_train

    @training.setter
    def training(self, value):
        self.is_train = value

    def mpi_param_broadcast(self, root=0):
        # Read through to the owner, never a module-level snapshot. 6.B15.
        if not compile_extern.in_mpi: return
        for p in self.parameters():
            p.update(p.mpi_broadcast(root))

    def __setattr__(self, key, value):
        if isinstance(value, Var):
            buffer_names = self.__dict__.get("_buffer_names", ())
            value_attrs = value.__dict__
            is_parameter = (
                not key.startswith("_")
                and key not in buffer_names
                and value_attrs.get("is_buffer") is not True
                and value_attrs.get("persistent") is not False
            )
            if is_parameter and _is_plain_tensor(value):
                non_params = self.__dict__.get("_non_parameter_names")
                if getattr(value, "_is_torch_parameter", False):
                    # nn.Parameter marks the Var itself, so re-registering a name
                    # that used to hold a plain tensor promotes it.
                    if non_params:
                        non_params.discard(key)
                elif not (isinstance(self.__dict__.get(key), Var)
                          and not (non_params and key in non_params)):
                    # Only the FIRST assignment decides. A name already holding a
                    # parameter stays one, so the dtype cast / weight load that
                    # replaces a Var with a plain one (from_pretrained does this)
                    # cannot silently demote a real weight out of the optimizer.
                    self.__dict__.setdefault("_non_parameter_names", set()).add(key)
                    is_parameter = False
            # Parameter identity belongs to the Var, not to its latest alias.
            # A private/helper alias must not erase an existing Parameter marker.
            if is_parameter:
                value._is_torch_parameter = True
        object.__setattr__(self, key, value)

    def __getattr__(self, key):
        return object.__getattribute__(self, key)

    def register_buffer(self, key, value, persistent=True):
        # torch allows registering a None buffer as a placeholder (e.g. vLLM's
        # FusedMoE expert_map when there is no expert parallelism). Don't try to
        # tag attributes on None.
        # Track buffer attribute NAMES on the module (like torch's _buffers dict).
        # The per-Var is_buffer/persistent tags are lost when from_pretrained's
        # dtype cast / weight-load REPLACES the buffer Var with a fresh one, so
        # parameters()/named_parameters() can no longer tell it's a buffer and it
        # leaks into the optimizer (then weight-decay corrupts e.g. rope inv_freq).
        # Name-based tracking survives any Var replacement -- the torch invariant.
        try:
            self.__dict__.setdefault("_buffer_names", set()).add(key)
            non_persistent = self.__dict__.setdefault(
                "_non_persistent_buffer_names", set()
            )
            if persistent:
                non_persistent.discard(key)
            else:
                non_persistent.add(key)
        except Exception:
            pass
        if value is not None:
            is_parameter = value.__dict__.get("_is_torch_parameter") is True
            if not is_parameter:
                value.persistent = persistent
                # Raw buffers remain non-Parameters even when the same Var is
                # later exposed through another public attribute.
                value.is_buffer = True
                value._is_torch_parameter = False
        object.__setattr__(self, key, value)
        return value

    @property
    def _buffers(self):
        # This module's own buffers, persistent and not, keyed by attribute name --
        # torch's ``_buffers``. write-through so accelerate's
        # `module._buffers[name] = value` (the is_buffer branch of
        # set_module_tensor_to_device) persists to the module attribute.
        return _WriteThroughDict(self, self._named_vars("buffers", recurse=False))

    def named_buffers(self, recurse=True):
        ''' Returns a list of (name, buffer) for all registered buffers.

        Like torch, recurse=True (default) descends into all child modules,
        prefixing names with the submodule path, and returns every registered
        buffer regardless of persistence. A buffer reachable under more than one
        name is returned once, under the first.
        '''
        return self._named_vars("buffers", recurse)

    def named_children(self,):
        childs = []
        for k,v in self.__dict__.items():
            if isinstance(v,Module):
                childs.append((k,v))
        return childs

    def _convert_float_vars(self, method):
        '''Convert every floating-point parameter and buffer in this module.'''
        seen = set()
        values = self._named_vars("parameters") + self._named_vars("buffers")
        for _, value in values:
            if id(value) in seen:
                continue
            seen.add(id(value))
            if value.dtype.is_float():
                value.assign(getattr(value, method)())
        return self

    def float64(self):
        '''convert all floating-point parameters and buffers to float64'''
        self._amp_level = 0
        return self._convert_float_vars("float64")

    def float32(self):
        '''convert all floating-point parameters and buffers to float32'''
        self._amp_level = 0
        return self._convert_float_vars("float32")

    def float16(self):
        '''convert all floating-point parameters and buffers to float16'''
        return self._convert_float_vars("float16")

    def bfloat16(self):
        '''convert all floating-point parameters and buffers to bfloat16'''
        return self._convert_float_vars("bfloat16")

    def half(self):
        '''convert all floating-point parameters and buffers to float16'''
        return self.float16()

    def float_auto(self):
        '''convert all parameters to float16 or float32 automatically
        by jt.flags.auto_mixed_precision_level and jt.flags.amp_reg'''
        self._amp_level = -1
        for p in self.parameters():
            if p.dtype.is_float():
                p.assign(p.float_auto())
        return self


class Function(Module):
    ''' Function Module for customized backward operations

Example 1 (Function can have multiple input and multiple output, and user
can store value for backward computation)::

    import jittor as jt
    from jittor import Function

    class MyFunc(Function):
        def execute(self, x, y):
            self.x = x
            self.y = y
            return x*y, x/y

        def grad(self, grad0, grad1):
            return grad0 * self.y, grad1 * self.x
    a = jt.array(3.0)
    b = jt.array(4.0)
    func = MyFunc.apply
    c,d = func(a, b)
    da, db = jt.grad(c+d*3, [a, b])
    assert da.data == 4
    assert db.data == 9

Example 2(Function can return None for no gradiant, and gradiant
can also be None)::

    import jittor as jt
    from jittor import Function

    class MyFunc(Function):
        def execute(self, x, y):
            self.x = x
            self.y = y
            return x*y, x/y

        def grad(self, grad0, grad1):
            assert grad1 is None
            return grad0 * self.y, None
    a = jt.array(3.0)
    b = jt.array(4.0)
    func = MyFunc.apply
    c,d = func(a, b)
    d.stop_grad()
    da, db = jt.grad(c+d*3, [a, b])
    assert da.data == 4
    assert db.data == 0

    '''
    def _new_call_context(self):
        """A one-shot object to run this call's ``execute``/``grad`` against.

        Everything a Function saves for its backward -- the user's
        ``self.x = x`` and the framework's own input/output masks -- used to
        live on the Function INSTANCE. Calling one instance twice therefore
        overwrote the first call's saved state, and the first call's backward
        then ran against the second call's tensors: a **wrong gradient with no
        warning**::

            f = Mul(); o1 = f(a, b); o2 = f(a, c)
            jt.grad(o1, a)      # used to give dc, not db

        ``MyFunc.apply(...)`` happened to be safe because it builds a new
        instance per call, and the examples above only show that spelling --
        but ``f = MyFunc(); f(x); f(y)`` is just as natural, and 50+ Function
        subclasses in this tree save state this way.

        The context starts as a shallow copy of the instance ``__dict__``, so
        anything ``__init__`` configured is visible to ``execute``, while
        writes made during the call land on the context and leave the shared
        instance alone. Binding ``ctx._grad`` into the tape keeps the context
        alive exactly as long as its backward might run.
        """
        ctx = object.__new__(type(self))
        ctx.__dict__.update(self.__dict__)
        return ctx

    @staticmethod
    def _reject_var_keywords(owner, kw):
        # Only positional arguments are taped, so a Var passed by keyword would
        # silently come back with no gradient. Say so instead.
        for k, v in kw.items():
            if isinstance(v, Var) and not v.is_stop_grad():
                raise TypeError(
                    f"{owner}: pass differentiable Var arguments positionally, "
                    f"not as the keyword {k!r}. Keyword arguments are not taped, "
                    "so this Var would silently receive no gradient.")

    def __call__(self, *args, **kw):
        # One context per call. `self` is only a factory from here on.
        return self._new_call_context()._run_call(*args, **kw)

    def _run_call(self, *args, **kw):
        """Run one call. ``self`` here is a one-shot context, not the instance.

        Split out of ``__call__`` so that a wrapper which needs per-call
        bookkeeping of its own can build the context, write onto it, and run
        the call against that same object::

            ctx = fn._new_call_context()
            ctx.my_state = ...          # visible to execute() and to grad()
            out = ctx._run_call(*args)
            ctx.more_state = ...        # still visible to grad()

        The torch compatibility layer does exactly this (it records
        ``needs_input_grad`` and the forward input/output shapes). Writing that
        bookkeeping onto the Function INSTANCE instead does not work, and fails
        in two different directions: whatever is written before the call is
        overwritten by the next call of the same instance, and whatever is
        written after the call never reaches the backward at all, because the
        context was copied from the instance when the call started.
        """
        self._reject_var_keywords(type(self).__name__, kw)
        if flags.no_grad:
            return self.execute(*args, **kw)
        backup = args
        args = list(args)
        taped_inputs = []
        taped_outputs = []
        input_mask = [-1] * len(args)
        for i,v in enumerate(args):
            if isinstance(v, Var):
                if v.is_stop_grad():
                    # -2 in input_mask represents it is stop_grad
                    input_mask[i] = -2
                    continue
                v = v.tape()
                input_mask[i] = len(taped_inputs)
                args[i] = v
                taped_inputs.append(v)
        ori_res = self.execute(*args, **kw)
        if not isinstance(ori_res, Sequence):
            res = [ori_res]
        else:
            res = list(ori_res)
        output_mask = [-1] * len(res)
        for i,v in enumerate(res):
            if isinstance(v, Var):
                v = v.tape()
                output_mask[i] = len(taped_outputs)
                res[i] = v
                taped_outputs.append(v)
        self.input_mask = input_mask
        self.output_mask = output_mask
        # tape output and input together so
        # backward treat them as one operator
        tape_together(taped_inputs, taped_outputs, self._grad)
        if isinstance(ori_res, Sequence):
            return res
        else:
            return res[0]

    def _grad(self, *args):
        new_args = ( (args[i] if i>=0 else None) for i in self.output_mask )
        ret = self.grad(*new_args)
        if not isinstance(ret, Sequence):
            ret = (ret,)
        new_ret = []
        for i, r in enumerate(ret):
            j = self.input_mask[i]
            if j<0:
                # -2 in input_mask represents it is stop_grad
                assert r is None or j==-2, f"{type(self)}'s {i}-th returned grad should be None, "\
                    "because the input value is not jittor variable."
            else:
                new_ret.append(r)
        return new_ret

    def dfs(self, parents, k, callback, callback_leave=None, recurse=True):
        pass

    @classmethod
    def apply(cls, *args, **kw):
        # Same contract as __call__, which now also accepts **kw (it used to
        # reject it outright, so `apply` could not actually forward keywords).
        func = cls()
        return func(*args, **kw)

class GradHooker(Function):
    def __init__(self, hook):
        self.hook = hook

    def execute(self, *args):
        return args

    def grad(self, *grad_input):
        ret = self.hook(grad_input)
        if ret: grad_input = ret
        return grad_input

def grad_hooker(args, hook):
    hooker = GradHooker(hook)
    return hooker(*args)

def register_hook(v, hook):
    """ register hook of any jittor Variables, if hook return not None,
the gradient of this variable will be alter,

    Example::

        x = jt.array([0.0, 0.0])
        y = x * [1,2]
        y.register_hook(lambda g: g*2)
        dx = jt.grad(y, x)
        print(dx)
        # will be [2, 4]

    Returns a handle whose ``.remove()`` detaches the hook, like torch's
    ``Tensor.register_hook``. It used to return the Var and offer no way to
    remove the hook at all -- while this same file already had
    ``_RemovableHandle`` for exactly this, used by every Module hook.

    The in-place ``swap`` stays: the hook has to BE a node in the graph, and
    the Var the caller is holding has to be the hooked one. So what
    ``remove()`` undoes is the hook *running*; the (now identity) node stays
    where it is, which is what keeps a graph already built on it intact.
    """
    live = [True]
    def _hook(grads):
        if not live[0]:
            return None
        g = hook(grads[0])
        if g is not None:
            return (g,)
        return None
    hooker = GradHooker(_hook)
    v.swap(hooker(v)[0])
    return _RemovableHandle(lambda: live.__setitem__(0, False))

Var.register_hook = register_hook

def make_module(func, exec_n_args=1):
    class MakeModule(Module):
        def __init__(self, *args, **kw):
            self.args = args
            self.kw = kw
        def execute(self, *args):
            return func(*args, *self.args, **self.kw)
        def __str__(self):
            return f"{func.__name__}({self.extra_repr()})"
        def extra_repr(self):
            return ",".join(map(str, self.args))
    MakeModule.__name__ = func.__name__
    return MakeModule


def dirty_fix_pytorch_runtime_error():
    ''' This funtion should be called before pytorch.

    Example::

        import jittor as jt
        jt.dirty_fix_pytorch_runtime_error()
        import torch
    '''
    import os, platform

    if platform.system() == 'Linux':
        import jittor_utils
        flags = os.RTLD_GLOBAL | os.RTLD_NOW | \
            getattr(os, "RTLD_DEEPBIND", 0)
        with jittor_utils.import_scope(flags):
            import torch


import atexit

class ExitHooks(object):
    def __init__(self):
        self.exit_code = None
        self.exception = None

    def hook(self):
        self._orig_exit = sys.exit
        sys.exit = self.exit
        sys.excepthook = self.exc_handler

    def exit(self, code=0):
        self.exit_code = code
        self._orig_exit(code)

    def exc_handler(self, exc_type, exc, *args):
        self.exception = exc
        traceback.print_exception(exc_type, exc, *args)

hooks = ExitHooks()
hooks.hook()

def jittor_exit():
    if hooks.exit_code is not None:
        pass
    elif hooks.exception is not None:
        pass
    else:
        pass
        # core.sync_all(True)
    core.cleanup()
atexit.register(jittor_exit)

def vtos(v):
    data_str = f"jt.Var({v.numpy()}, dtype={v.dtype})"
    data_str = data_str.replace("\n", "\n       ")
    return data_str

Var.__str__ = vtos
Var.__repr__ = vtos
Var.peek = lambda x: f"{x.dtype}{x.shape}"

def size(v, dim=None):
    if dim is None:
        return v.shape
    return v.shape[dim]
Var.size = size


def to_int(v):
    return ori_int(v.item())

def to_float(v):
    return ori_float(v.item())

def to_bool(v):
    assert v.dtype.is_int() or v.dtype.is_bool()
    return ori_bool(v.item())

Var.__int__ = to_int
Var.__float__ = to_float
Var.__bool__ = to_bool

def format(v, spec):
    return v.item().__format__(spec)
Var.__format__ = format

def get_len(var):
    return var.shape[0]

Var.__len__ = get_len
int = int32
Var.int = Var.int32
Var.long = Var.int32
float = float32
Var.float = Var.float32
double = float64
Var.double = Var.float64
half = float16
Var.half = Var.float16

def is_var(v):
    return isinstance(v, Var)

# __array__ interface is used for np.array(jt_var). numpy (and scipy, e.g. DETR's
# linear_sum_assignment) call __array__(dtype=None[, copy=None]); accept and apply.
def _var__array__(self, dtype=None, copy=None):
    a = self.numpy()
    if dtype is not None:
        a = a.astype(dtype)
    return a
Var.__array__ = _var__array__
Var.__array_priority__ = 2000
# __reduce__, __module__ is used for pickle.dump and pickle.load
Var.__module__ = "jittor"
Var.__reduce__ = lambda self: (Var, (self.data,))


__all__ = (
    "ExitHooks", "Function", "GradHooker", "Module", "abs_", "add_",
    "amp_flags", "argmax", "argmin", "array", "array64", "attrs", "cast",
    "clamp", "clamp_", "clean", "detach", "dfs_to_numpy",
    "dirty_fix_pytorch_runtime_error", "display_memory_info", "double",
    "empty", "enable_grad", "erf_", "erfinv_", "fetch", "flag_scope",
    "flags", "flatten", "float", "float_auto", "format", "full",
    "full_like", "get_len", "grad", "grad_hooker", "half", "hooks", "int",
    "is_var", "jittor_exit", "liveness_info", "load", "log_capture_scope",
    "make_module", "masked_fill", "multiply_", "ne", "new_empty",
    "new_full", "new_ones", "new_zeros", "no_grad", "norm", "normal",
    "ones", "ones_like", "origin_reshape", "origin_transpose", "outer",
    "permute", "pow", "profile_mark", "profile_scope", "rand", "rand_like",
    "randint", "randint_like", "randn", "randn_like", "random",
    "register_hook", "reshape", "safepickle", "safeunpickle", "save",
    "sigmoid_", "single_log_capture", "single_process_scope", "size", "sqr",
    "sqrt_", "squeeze", "std", "to_bool", "to_device", "to_float",
    "to_int", "transpose", "type_as", "unsqueeze", "var", "view", "vtos",
    "zeros", "zeros_like", "runtime",
)
