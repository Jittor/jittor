# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers:
#   Dun Liang <randonlang@gmail.com>.
#   Meng-Hao Guo <guomenghao1997@gmail.com>
#
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
__version__ = '1.2.2.62'
from . import lock
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
    from . import compile_extern
    from .compile_extern import mkl_ops, mpi, mpi_ops, in_mpi, rank
    if core.get_device_count() == 0:
        has_cuda = compile_extern.has_cuda = compiler.has_cuda = False
    if has_cuda:
        from .compile_extern import cudnn, curand, cublas
    from .init_cupy import numpy2cupy

import contextlib
import numpy as np
from collections import OrderedDict
from collections.abc import Sequence, Mapping
import types
import pickle
import hashlib
import sys, os
import traceback


def safepickle(obj, path):
    # Protocol version 4 was added in Python 3.4. It adds support for very large objects, pickling more kinds of objects, and some data format optimizations.
    # ref: <https://docs.python.org/3/library/pickle.html>
    s = pickle.dumps(obj, 4)
    checksum = hashlib.sha1(s).digest()
    s += bytes(checksum)
    s += b"HCAJSLHD"
    with open(path, 'wb') as f:
        f.write(s)

def safeunpickle(path):
    if path.startswith("jittorhub://"):
        path = path.replace("jittorhub://", "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/")
    if path.startswith("https:") or path.startswith("http:"):
        base = path.split("/")[-1]
        fname = os.path.join(compiler.ck_path, base)
        from jittor.utils.misc import download_url_to_local
        download_url_to_local(path, base, compiler.ck_path, None)
        path = fname
    if path.endswith(".pth"):
        try:
            dirty_fix_pytorch_runtime_error()
            import torch
        except:
            raise RuntimeError("pytorch need to be installed when load pth format.")
        model_dict = torch.load(path, map_location=torch.device('cpu'))
        return model_dict
    with open(path, "rb") as f:
        s = f.read()
    if not s.endswith(b"HCAJSLHD"):
        return pickle.loads(s)
    checksum = s[-28:-8]
    s = s[:-28]
    if hashlib.sha1(s).digest() != checksum:
        raise ValueError("Pickle checksum does not match! path: "+path)
    return pickle.loads(s)

class _call_no_record_scope:
    def __enter__(self): pass
    def __exit__(self, *exc): pass
    def __call__(self, func):
        def inner(*args, **kw):
            with self:
                ret = func(*args, **kw)
            return ret
        return inner

class flag_scope(_call_no_record_scope):
    def __init__(self, **jt_flags):
        self.jt_flags = jt_flags

    def __enter__(self):
        flags_bk = self.flags_bk = {}
        try:
            for k,v in self.jt_flags.items():
                flags_bk[k] = getattr(flags, k)
                setattr(flags, k, v)
        except:
            self.__exit__()
            raise

    def __exit__(self, *exc):
        for k,v in self.flags_bk.items():
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
        self.jt_flags = jt_flags
        jt_flags["no_grad"] = 1

class enable_grad(flag_scope):
    ''' enable_grad scope, all variable created inside this
scope will start grad.

Example::

    import jittor as jt

    with jt.enable_grad():
        ...

    '''
    def __init__(self, **jt_flags):
        self.jt_flags = jt_flags
        jt_flags["no_grad"] = 0

single_log_capture = None

class log_capture_scope(_call_no_record_scope):
    """log capture scope

    example::

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
        single_log_capture = True
        self.logs = []
        LOG.log_capture_start()
        try:
            self.fs.__enter__()
            if "log_v" in self.fs.jt_flags:
                LOG.log_v = self.fs.jt_flags["log_v"]
            return self.logs
        except:
            LOG.log_capture_stop()
            single_log_capture = None
            raise

    def __exit__(self, *exc):
        global single_log_capture
        self.fs.__exit__(*exc)
        if "log_v" in self.fs.jt_flags:
            LOG.log_v = flags.log_v
        LOG.log_capture_stop()
        self.logs.extend(LOG.log_capture_read())
        single_log_capture = None


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
        profiler.start(self.warmup, self.rerun)
        self.report = []
        try:
            self.fs.__enter__()
            return self.report
        except:
            profiler.stop()
            raise

    def __exit__(self, *exc):
        self.fs.__exit__(*exc)
        profiler.stop()
        self.report.extend(profiler.report())

class __single_process_scope:
    def __init__(self, rank=0):
        self.rank = rank

    def __enter__(self):
        global in_mpi
        self.bk_in_mpi = in_mpi
        if mpi:
            self.bk_mpi_state = mpi.get_state()
        if not in_mpi:
            return True
        
        ret = self.rank == mpi.world_rank()
        in_mpi = compile_extern.in_mpi = False
        mpi.set_state(False)
        return ret

    def __exit__(self, *exc):
        global in_mpi
        in_mpi = compile_extern.in_mpi = self.bk_in_mpi
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

cast = unary
Var.cast = Var.cast

def array(data, dtype=None):
    if isinstance(data, core.Var):
        if dtype is None:
            return data.clone()
        return cast(data, dtype)
    if dtype is not None:
        if isinstance(dtype, NanoString):
            dtype = str(dtype)
        elif callable(dtype):
            dtype = dtype.__name__
        return ops.array(np.array(data, dtype))
    return ops.array(data)

def grad(loss, targets):
    if type(targets) == core.Var:
        return core.grad(loss, [targets])[0]
    return core.grad(loss, targets)

def liveness_info():
    return {
        "hold_vars": core.number_of_hold_vars(),
        "lived_vars": core.number_of_lived_vars(),
        "lived_ops": core.number_of_lived_ops(),
    }

def ones(shape, dtype="float32"):
    if not isinstance(shape, (NanoVector, Sequence)):
        shape = (shape,)
    return unary(1, dtype).broadcast(shape)

def ones_like(x):
    return ones(x.shape,x.dtype)

def zeros(shape, dtype="float32"):
    if not isinstance(shape, (NanoVector, Sequence)):
        shape = (shape,)
    return unary(0, dtype).broadcast(shape)

def full(shape,val,dtype="float32"):
    if not isinstance(shape, (NanoVector, Sequence)):
        shape = (shape,)
    return unary(val, dtype).broadcast(shape)

def full_like(x,val):
    return full(x.shape,val,x.dtype)

def zeros_like(x):
    return zeros(x.shape,x.dtype)

flags = core.flags()

def std(x):
    matsize=1
    for i in x.shape:
        matsize *= i
    out=(x-x.mean()).sqr().sum()
    out=out/(matsize-1)
    out=out.maximum(1e-6).sqrt()
    return out
Var.std = std

def norm(x, k, dim, keepdim=False):
    assert k==1 or k==2
    if k==1:
        return x.abs().sum(dim, keepdim)
    if k==2:
        return (x.sqr()).sum(dim, keepdim).maximum(1e-6).sqrt()
Var.norm = norm

origin_reshape = reshape
def reshape(x, *shape):
    if len(shape) == 1 and isinstance(shape[0], (Sequence, NanoVector)):
        shape = shape[0]
    return origin_reshape(x, shape)
reshape.__doc__ = origin_reshape.__doc__
Var.view = Var.reshape = view = reshape

origin_transpose = transpose
def transpose(x, *dim):
    if len(dim) == 1 and isinstance(dim[0], (Sequence, NanoVector)):
        dim = dim[0]
    return origin_transpose(x, dim)
transpose.__doc__ = origin_transpose.__doc__
Var.transpose = Var.permute = permute = transpose

def flatten(input, start_dim=0, end_dim=-1):
    '''flatten dimentions by reshape'''
    in_shape = input.shape
    start_dim = len(in_shape) + start_dim if start_dim < 0 else start_dim
    end_dim = len(in_shape) + end_dim if end_dim < 0 else end_dim
    assert end_dim >= start_dim, "end_dim should be larger than or equal to start_dim for flatten function"
    out_shape = []
    for i in range(0,start_dim,1): out_shape.append(in_shape[i])
    dims = 1
    for i in range(start_dim, end_dim+1, 1): dims *= in_shape[i]
    out_shape.append(dims)
    for i in range(end_dim+1,len(in_shape),1): out_shape.append(in_shape[i])
    return input.reshape(out_shape)
Var.flatten = flatten

def start_grad(x):
    return x._update(x)
Var.detach_inplace = Var.start_grad = start_grad

def detach(x):
    return x.detach()

def unsqueeze(x, dim):
    shape = list(x.shape)
    if dim < 0: dim += len(shape) + 1
    assert dim <= len(shape)
    return x.reshape(shape[:dim] + [1] + shape[dim:])
Var.unsqueeze = unsqueeze

def squeeze(x, dim):
    shape = list(x.shape)
    if dim < 0: dim += len(shape)
    assert dim < len(shape) and dim >= 0
    assert shape[dim] == 1
    return x.reshape(shape[:dim] + shape[dim+1:])
Var.squeeze = squeeze

def clamp(x, min_v=None, max_v=None):
    if x.shape[0]==0:
        return x
    if min_v is not None and max_v is not None:
        assert min_v <= max_v
    if min_v is not None:
        x = x.maximum(min_v)
    if max_v is not None:
        x = x.minimum(max_v)
    return x

Var.clamp = clamp

def type_as(a, b):
    return a.unary(op=b.dtype)
Var.type_as = type_as
Var.astype = Var.cast

def masked_fill(x, mask, value):
    assert list(x.shape) == list(mask.shape)
    # TODO: assert mask = 0 or 1
    return x * (1 - mask) + mask * value
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
    if isinstance(y, (ori_int, ori_float)) and y == 2:
        return x.sqr()
    return core.ops.pow(x, y)
Var.pow = Var.__pow__ = pow

def argmax(x, dim, keepdims:bool=False):
    return x.arg_reduce("max", dim, keepdims)
Var.argmax = argmax

def argmin(x, dim, keepdims:bool=False):
    return x.arg_reduce("min", dim, keepdims)
Var.argmin = argmin

def randn(*size, dtype="float32", requires_grad=True) -> Var:
    ''' samples random numbers from a standard normal distribution.
    
    :param size: shape of the output.
    :type size: int or a sequence of int

    :param dtype: data type, defaults to "float32".
    :type dtype: str, optional

    :param requires_grad: whether to enable gradient back-propgation, defaults to True.
    :type requires_grad: bool, optional

    Example:

        >>> jt.randn(3)
        jt.Var([-1.019889   -0.30377278 -1.4948598 ], dtype=float32)
        >>> jt.randn(2, 3)
        jt.Var([[-0.15989183 -1.5010914   0.5476955 ]
         [-0.612632   -1.1471151  -1.1879086 ]], dtype=float32)
    '''
    if isinstance(size, tuple) and isinstance(size[0], (tuple, list, NanoVector)): size = size[0]
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

    Example:

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
    
    Example:
        
        >>> x = jt.zeros((2, 3))
        >>> jt.rand_like(x)
        jt.Var([[0.6164821  0.21476883 0.61959815]
         [0.58626485 0.35345772 0.5638483 ]], dtype=float32)
    ''' 
    if dtype is None: dtype = x.dtype
    return jt.random(x.shape, x.dtype)

def randn_like(x, dtype=None) -> Var:
    ''' samples random values from standard normal distribution with the same shape as x.

    :param x: reference variable.
    :type x: jt.Var
        
    :param dtype: if None, the dtype of the output is the same as x. 
        Otherwise, use the specified dtype. Defaults to None.
    :type dtype: str, optional

    Example:
        
        >>> x = jt.zeros((2, 3))
        >>> jt.randn_like(x)
        jt.Var([[-1.1647032   0.34847224 -1.3061888 ]
         [ 1.068085   -0.34366122  0.13172573]], dtype=float32)
    ''' 
    if dtype is None: dtype = x.dtype
    return jt.random(x.shape, x.dtype, "normal")

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

    Example:
        
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
    v = (jt.random(shape) * (high - low) + low).clamp(low, high-0.5)
    v = jt.floor(v)
    return v.astype(dtype)

def randint_like(x, low, high=None) -> Var:
    ''' samples random values from standard normal distribution with the same shape as x.

    :param x: reference variable.
    :type x: jt.Var
        
    :param low: lowest intergers to be drawn from the distribution, defaults to 0.
    :type low: int, optional
        
    :param high: One above the highest integer to be drawn from the distribution.
    :type high: int

    Example:

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

    Example:
    
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
    def dfs(x):
        if isinstance(x, list):
            for i in range(len(x)):
                x[i] = dfs(x[i])
        elif isinstance(x, dict):
            for k in x:
                x[k] = dfs(x[k])
        elif isinstance(x, Var):
            return x.numpy()
        return x
    safepickle(dfs(params_dict), path)

def _uniq(x):
    a = set()
    b = []
    for i in x:
        j = id(i)
        if j not in a:
            a.add(j)
            b.append(i)
    return b

class Module:
    def __init__(self, *args, **kw):
        pass
    def execute(self, *args, **kw):
        raise NotImplementedError("Please implement 'execute' method of "+str(type(self)))
    def __call__(self, *args, **kw):
        return self.execute(*args, **kw)
    def __repr__(self):
        return self.__str__()
    def _get_name(self):
        return self.__class__.__name__
    def __name__(self):
        pass

    def dfs(self, parents, k, callback, callback_leave=None):
        n_children = 0
        for v in self.__dict__.values():
            if isinstance(v, Module):
                n_children += 1
        ret = callback(parents, k, self, n_children)
        if ret == False: return
        for k,v in self.__dict__.items():
            if not isinstance(v, Module):
                continue
            parents.append(self)
            v.dfs(parents, k, callback, callback_leave)
            parents.pop()
        if callback_leave:
            callback_leave(parents, k, self, n_children)

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

    def parameters(self):
        ps = []
        stack = []
        def callback(parents, k, v, n):
            stack.append(str(k))
            for k2, p in v.__dict__.items():
                if k2.startswith("_"): continue
                if isinstance(p, Var):
                    ps.append(p)
                    p.name(".".join(stack[1:]+[str(k2)]))
        def callback_leave(parents, k, v, n):
            stack.pop()
        self.dfs([], None, callback, callback_leave)
        return _uniq(ps)

    def named_parameters(self):
        ps = self.parameters()
        return [ (p.name(), p) for p in ps ]

    def state_dict(self):
        ps = self.parameters()
        return { p.name(): p for p in ps }

    def load_state_dict(self, params):
        self.load_parameters(params)

    def modules(self):
        ms = []
        def callback(parents, k, v, n):
            if isinstance(v, Module):
                ms.append(v)
        self.dfs([], None, callback, None)
        return _uniq(ms)

    def named_modules(self):
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

    def register_forward_hook(self, func):
        cls = self.__class__
        self.__fhook__ = func
        if hasattr(cls, "__hooked__"):
            return
        cls.__hooked__ = True
        origin_call = cls.__call__
        def new_call(self, *args, **kw):
            ret = origin_call(self, *args, **kw)
            if hasattr(self, "__fhook__"):
                if len(kw):
                    self.__fhook__(self, args, ret, kw)
                else:
                    self.__fhook__(self, args, ret)
            return ret
        self.__class__.__call__ = new_call
    
    def remove_forward_hook(self):
        cls = self.__class__
        if hasattr(cls,"__hooked__"):
            delattr(cls,"__hooked__")
        if hasattr(self,"__fhook__"):
            delattr(self,"__fhook__")

    def register_pre_forward_hook(self, func):
        cls = self.__class__
        self.__fhook2__ = func
        if hasattr(cls, "__hooked2__"):
            return
        cls.__hooked2__ = True
        origin_call = cls.__call__
        def new_call(self, *args, **kw):
            if hasattr(self, "__fhook2__"):
                if len(kw):
                    self.__fhook2__(self, args, kw)
                else:
                    self.__fhook2__(self, args)
            return origin_call(self, *args, **kw)
        self.__class__.__call__ = new_call

    def remove_pre_forward_hook(self):
        cls = self.__class__
        if hasattr(cls,"__hooked2__"):
            delattr(cls,"__hooked2__")
        if hasattr(self,"__fhook2__"):
            delattr(self,"__fhook2__")

    def children(self):
        cd = []
        def callback(parents, k, v, n):
            if len(parents) == 1 and isinstance(v, Module):
                cd.append(v)
                return False
        self.dfs([], None, callback, None)
        return cd

    def extra_repr(self):
        ss = []
        n = len(self.__init__.__code__.co_varnames)
        if self.__init__.__defaults__ is not None:
            n -= len(self.__init__.__defaults__)
        for i, k in enumerate(self.__init__.__code__.co_varnames[1:]):
            v = getattr(self, k) if hasattr(self, k) else None
            if isinstance(v, Var): v = v.peek()
            s = f"{k}={v}" if i >= n else str(v)
            ss.append(s)
        return ", ".join(ss)

    def apply(self, func):
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
                if isinstance(v, nn.Sequential):
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
                else:
                    n_failed += 1
                    LOG.e(f'load parameter {key} failed: expect the shape of {key} to be {v.shape}, but got {param.shape}')
        if n_failed:
            LOG.w(f"load total {len(params)} params, {n_failed} failed")

    def save(self, path: str):
        ''' saves parameters to a file.

        :param path: path to save.
        :type path: str

        Example:

            >>> class Net(nn.Module):
            >>> ...
            >>> net = Net()
            >>> net.save('net.pkl')
            >>> net.load('net.pkl')
        '''
        params = self.parameters()
        params_dict = {}
        for p in params:
            params_dict[p.name()] = p.data
        safepickle(params_dict, path)

    def load(self, path: str):
        ''' loads parameters from a file.

        :param path: path to load.
        :type path: str
        
        Example:

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

    def eval(self):
        def callback(parents, k, v, n):
            if isinstance(v, Module):
                v.is_train = False
        self.dfs([], None, callback, None)

        # backup stop grad or not
        if not hasattr(self, "backup_grad_state"):
            self.backup_grad_state = {}
        for p in self.parameters():
            if id(p) not in self.backup_grad_state:
                self.backup_grad_state[id(p)] = not p.is_stop_grad()
            p.stop_grad()

    def train(self):
        def callback(parents, k, v, n):
            if isinstance(v, Module):
                v.is_train = True
        self.dfs([], None, callback, None)

        # backup stop grad or not
        if hasattr(self, "backup_grad_state"):
            for p in self.parameters():
                if id(p) in self.backup_grad_state and self.backup_grad_state[id(p)]:
                    p.start_grad()
    
    def is_training(self) -> bool:
        if not hasattr(self, "is_train"):
            self.is_train = True
        return self.is_train

    def mpi_param_broadcast(self, root=0):
        if not in_mpi: return
        for p in self.parameters():
            p.update(p.mpi_broadcast(root))

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
    func = MyFunc()
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
    func = MyFunc()
    c,d = func(a, b)
    d.stop_grad()
    da, db = jt.grad(c+d*3, [a, b])
    assert da.data == 4
    assert db.data == 0

    '''
    def __call__(self, *args):
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
        ori_res = self.execute(*args)
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

    def dfs(self, parents, k, callback, callback_leave=None):
        pass

    @classmethod
    def apply(cls, *args, **kw):
        func = cls()
        return func(*args, **kw)


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
    import os
    os.RTLD_GLOBAL = os.RTLD_GLOBAL | os.RTLD_DEEPBIND


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
        core.sync_all(True)
    core.cleanup()
atexit.register(jittor_exit)

def vtos(v):
    return f"jt.Var({v.data}, dtype={v.dtype})"

Var.__str__ = vtos
Var.__repr__ = vtos
Var.peek = lambda x: f"{x.dtype}{x.shape}"

def size(v, dim=None):
    if dim is None:
        return v.shape
    return v.shape[dim]
Var.size = size


def to_int(v):
    dtype = str(v.dtype)
    assert dtype.startswith("int")
    return v.item()

def to_float(v):
    dtype = str(v.dtype)
    assert dtype.startswith("float")
    return v.item()

def to_bool(v):
    dtype = str(v.dtype)
    assert dtype.startswith("int") or dtype=="bool"
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
float = float32
Var.float = Var.float32
double = float64
Var.double = Var.float64

# __array__ interface is used for np.array(jt_var)
Var.__array__ = Var.numpy
Var.__array_priority__ = 2000
# __reduce__, __module__ is used for pickle.dump and pickle.load
Var.__module__ = "jittor"
Var.__reduce__ = lambda self: (Var, (self.data,))

from . import nn
from . import attention
from . import lr_scheduler
from . import linalg
from .nn import matmul
from . import contrib
from . import numpy2cupy
from .contrib import concat
from .misc import *
from . import sparse
