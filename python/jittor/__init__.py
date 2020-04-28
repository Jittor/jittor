# ***************************************************************
# Copyright (c) 2020 Jittor. Authors:
#   Dun Liang <randonlang@gmail.com>.
#   Meng-Hao Guo <guomenghao1997@gmail.com>
#
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
from . import lock
with lock.lock_scope():
    from . import compiler
    from .compiler import LOG, has_cuda
    from .compiler import compile_custom_ops, compile_custom_op
    import jittor_core as core
    from jittor_core import *
    from jittor_core.ops import *
    from . import compile_extern
    from .compile_extern import mkl_ops, mpi, mpi_ops

import contextlib
import numpy as np
from collections import OrderedDict
from collections.abc import Sequence, Mapping
import types
import pickle
import sys
import traceback

def dfs(scope, vars):
    for v in scope.children.values():
        if type(v) == Scope:
            dfs(v, vars)
        else:
            vars.append(v)

def dfs_records(scope, records):
    for v in scope.children.values():
        if type(v) == Scope:
            dfs_records(v, records)
    for v in scope.records.values():
        records.append(v)

class Scope:
    def __init__(self, parent=None, name=None):
        self.children = OrderedDict()
        self.index = {}
        self.records = OrderedDict()
        if name == None:
            self.name = self.full_name = ""
        else:
            self.name = name
            self.full_name = parent.full_name + name + "/"

    def get_scope(self, name, unique=True):
        if not unique:
            index = self.index.get(name, 0)
            self.index[name] = index+1
            name = name + f'_{index}'
        if name not in self.children:
            sub_scope = Scope(self, name)
            self.children[name] = sub_scope
        else:
            sub_scope = self.children[name]
            assert type(sub_scope) == Scope, f"Name {name} is a Var: {sub_scope}"
        return sub_scope

    def make_var(self, shape, dtype, init, name, unique):
        if not unique:
            index = self.index.get(name, 0)
            self.index[name] = index+1
            name = name + f'_{index}'
        if name in self.children:
            var = self.children[name]
            assert type(var) == core.Var, f"Name {name} exist: {var}"
            assert (shape is None or var.shape == shape) and var.dtype == dtype, f"Shape or dtype not match {var} != {dtype}{shape}"
            return var
        else:
            full_name = self.full_name + name
            if type(init) != core.Var:
                if callable(init):
                    var = init(shape, dtype)
                    if type(var) != core.Var:
                        var = array(var)
                else:
                    assert init != None
                    var = array(init)
            else:
                var = init
            var.stop_fuse()
            self.children[name] = var
            var.name(full_name)
            return var

    def clean_index(self): self.index.clear()

    def clean(self):
        self.children.clear()
        self.records.clear()
        self.index.clear()

current_scope = Scope()
root_scope = current_scope

class _call_record_scope:
    def __enter__(self): pass
    def __exit__(self, *exc): pass
    def __call__(self, func):
        def inner(*args, **kw):
            with self:
                ret = func(*args, **kw)
                record_in_scope(ret, "output")
            return ret
        return inner

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

class var_scope(_call_record_scope):
    def __init__(self, name="scope", unique=False, **jt_flags):
        self.fs = flag_scope(**jt_flags)
        self.name = name
        self.unique = unique

    def __enter__(self):
        global current_scope
        self.prev = current_scope
        try:
            current_scope = current_scope.get_scope(self.name, self.unique)
            current_scope.clean_index()
            self.fs.__enter__()
        except:
            current_scope = self.prev
            del self.prev
            raise

    def __exit__(self, *exc):
        self.fs.__exit__(*exc)
        global current_scope
        current_scope = self.prev
        del self.prev

single_log_capture = None

class log_capture_scope(_call_no_record_scope):
    """log capture scope
    example:
        with jt.log_capture_scope(log_v=0) as logs:
            LOG.v("...")
        print(logs)
    """
    def __init__(self, **jt_flags):
        self.fs = flag_scope(**jt_flags)

    def __enter__(self):
        global single_log_capture
        assert not single_log_capture
        single_log_capture = True
        self.logs = []
        LOG.log_capture_start()
        try:
            self.fs.__enter__()
            return self.logs
        except:
            LOG.log_capture_stop()
            single_log_capture = None
            raise

    def __exit__(self, *exc):
        global single_log_capture
        self.fs.__exit__(*exc)
        LOG.log_capture_stop()
        self.logs.extend(LOG.log_capture_read())
        single_log_capture = None


class profile_scope(_call_no_record_scope):
    """ profile scope
    example:
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

def make_var(shape=None, dtype="float32", init=None, name='var', unique=False):
    return current_scope.make_var(shape, dtype, init, name, unique)

def find_vars(path=None):
    scope = current_scope
    if path is not None:
        assert isinstance(path, str)
        ns = path.split("/")
        if ns[-1] == "":
            ns.pop()
        for n in ns: scope = scope.children[n]
    if not isinstance(scope, Scope):
        return [scope]
    vars = []
    dfs(scope, vars)
    return vars

def find_var(path):
    scope = current_scope
    if path is not None:
        assert isinstance(path, str)
        ns = path.split("/")
        for n in ns: scope = scope.children[n]
    assert not isinstance(scope, Scope)
    return scope

def find_records(path=None):
    scope = current_scope
    if path is not None:
        assert isinstance(path, str)
        ns = path.split("/")
        if ns[-1] == "":
            ns.pop()
        for n in ns: scope = scope.children[n]
    assert isinstance(scope, Scope)
    records = []
    dfs_records(scope, records)
    return records

def find_record(path):
    scope = current_scope
    assert isinstance(path, str)
    ns = path.split("/")
    for n in ns[:-1]: scope = scope.children[n]
    assert isinstance(scope, Scope)
    return scope.records[ns[-1]]

def find_scope(path):
    scope = current_scope
    if path is not None:
        assert isinstance(path, str)
        ns = path.split("/")
        if ns[-1] == "":
            ns.pop()
        for n in ns: scope = scope.children[n]
    assert isinstance(scope, Scope)
    return scope

def record_in_scope(self, name):
    current_scope.records[name] = self
    if isinstance(self, Var):
        full_name = current_scope.full_name + name
        self.name(full_name)
    return self

Var.record_in_scope = record_in_scope

def clean():
    current_scope.clean()
    import gc
    # make sure python do a full collection
    gc.collect()

cast = unary

def array(data, dtype=None):
    if type(data) == core.Var:
        if dtype is None:
            return cast(data, data.dtype)
        return cast(data, dtype)
    if dtype != None:
        return ops.array(np.array(data, dtype))
    if type(data) == np.ndarray:
        if data.flags.c_contiguous:
            return ops.array(data)
        else:
            return ops.array(data.copy())
    return ops.array(np.array(data))

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
    return unary(1, dtype).broadcast(shape)

def zeros(shape, dtype="float32"):
    return unary(0, dtype).broadcast(shape)

flags = core.flags()

def detach(x):
    """return detached var"""
    return x.clone().stop_grad().clone()
Var.detach = detach

origin_reshape = reshape
def reshape(x, *shape):
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = shape[0]
    return origin_reshape(x, shape)
reshape.__doc__ = origin_reshape.__doc__
Var.view = Var.reshape = view = reshape

origin_transpose = transpose
def transpose(x, *dim):
    if len(dim) == 1 and isinstance(dim[0], Sequence):
        dim = dim[0]
    return origin_transpose(x, dim)
transpose.__doc__ = origin_transpose.__doc__
Var.transpose = Var.permute = permute = transpose

def flatten(input, start_dim=0, end_dim=-1):
    '''flatten dimentions by reshape'''
    in_shape = input.shape
    start_dim = len(in_shape) + start_dim if start_dim < 0 else start_dim
    end_dim = len(in_shape) + end_dim if end_dim < 0 else end_dim
    assert end_dim > start_dim, "end_dim should be larger than start_dim for flatten function"
    out_shape = []
    for i in range(0,start_dim,1): out_shape.append(in_shape[i])
    dims = 1
    for i in range(start_dim, end_dim+1, 1): dims *= in_shape[i]
    out_shape.append(dims)
    for i in range(end_dim+1,len(in_shape),1): out_shape.append(in_shape[i])
    return input.reshape(out_shape)
Var.flatten = flatten

def detach_inplace(x):
    return x.swap(x.stop_grad().clone())
Var.start_grad = Var.detach_inplace = detach_inplace

def unsqueeze(x, dim):
    shape = list(x.shape)
    assert dim <= len(shape)
    return x.reshape(shape[:dim] + [1] + shape[dim:])
Var.unsqueeze = unsqueeze

def squeeze(x, dim):
    shape = list(x.shape)
    assert dim < len(shape)
    assert shape[dim] == 1
    return x.reshape(shape[:dim] + shape[dim+1:])
Var.squeeze = squeeze

def clamp(x, min_v, max_v):
    # TODO: change to x.maximum(min_v).minimum(max_v)
    assert min_v <= max_v
    min_b = (x < min_v).int()
    max_b = (x > max_v).int()
    return x * (1 - min_b - max_b) + min_v * min_b + max_v * max_b
Var.clamp = clamp

def type_as(a, b):
    return a.unary(op=b.dtype)
Var.type_as = type_as

def masked_fill(x, mask, value):
    assert list(x.shape) == list(mask.shape)
    # TODO: assert mask = 0 or 1
    return x * (1 - mask) + mask * value
Var.masked_fill = masked_fill


def sqr(x): return x*x
Var.sqr = sqr

def attrs(var):
    return {
        "is_stop_fuse": var.is_stop_fuse(),
        "is_stop_grad": var.is_stop_grad(),
        "shape": var.shape,
        "dtype": var.dtype,
    }
Var.attrs = attrs

def fetch(vars, func, *args, **kw):
    core.fetch(vars, lambda *results: func(*results, *args, **kw))

def fetch_var(var, func, *args, **kw):
    core.fetch([var], lambda a: func(a, *args, **kw))
Var.fetch = fetch_var
del fetch_var

def import_vars(data):
    ''' Load variables into current scopes
    example:
        import_vars({"w":[1.0,2.0,3.0]})
        jt.get_var([3], "float64", name="w", gen_index=False)
    '''
    for k in data:
        v = data[k]
        if type(v) != core.Var:
            v = array(v).stop_fuse()
        scopes = k.split("/")
        scope = current_scope
        for i in range(len(scopes)-1):
            scope = scope.get_scope(scopes[i])
        vname = scopes[-1]
        assert vname not in scope.children, f"Var {k} exists. Please load_vars at the beginning"
        v.name(k)
        scope.children[vname] = v

def export_vars():
    ''' Export all vars into a dictionary
    return: a dictionary, key is var name, value is numpy array
    '''
    data = { v.name():v.fetch_sync() for v in find_vars() }
    return data

def load(path):
    pkl_file = open(path, 'rb')
    model_dict = pickle.load(pkl_file)
    return model_dict

class Module:
    def __init__(self, *args, **kw):
        __doc__ == 'doc'
    def execute(self, *args, **kw):
        pass
    def __call__(self, *args, **kw):
        return self.execute(*args, **kw)
    def __repr__(self):
        return self.__str__()
    def _get_name(self):
        return self.__class__.__name__
    def __doc__(self):
        pass
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
                if isinstance(p, Var):
                    ps.append(p)
                    p.name(".".join(stack[1:]+[str(k2)]))
        def callback_leave(parents, k, v, n):
            stack.pop()
        self.dfs([], None, callback, callback_leave)
        return ps

    def modules(self):
        ms = []
        def callback(parents, k, v, n):
            if isinstance(v, Module):
                ms.append(v)
        self.dfs([], None, callback, None)
        return ms

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

    def load_parameters(self, params):
        for key in params.keys():
            v = self
            key_ = key.split('.')
            end = 0
            for k in key_:
                if isinstance(v, nn.Sequential):
                    if np.int(k) >= len(v.layers):
                        end = 1
                        break
                    else:
                        v = v[np.int(k)]
                else:
                    if hasattr(v, k):
                        v = getattr(v, k)
                    else:
                        end = 1
                        break
            if end ==1:
                # print(f'init {key} fail ...')
                pass
            else:
                # print(f'init {key} success ...')
                if isinstance(params[key], np.ndarray) or isinstance(params[key], list):
                    v.assign(array(params[key]))
                elif isinstance(params[key], Var):
                    v.assign(params[key])
                else:
                    v.assign(array(params[key].cpu( ).detach().numpy()))
    def save(self, path):
        params = self.parameters()
        params_dict = {}
        for p in params:
            params_dict[p.name()] = p.data
        with open(path, 'wb') as f:
            pickle.dump(params_dict, f, pickle.HIGHEST_PROTOCOL)

    def eval(self):
        def callback(parents, k, v, n):
            if isinstance(v, Module) and hasattr(v, "is_train"):
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
            if isinstance(v, Module) and hasattr(v, "is_train"):
                v.is_train = True
        self.dfs([], None, callback, None)

        # backup stop grad or not
        if hasattr(self, "backup_grad_state"):
            for p in self.parameters():
                if id(p) in self.backup_grad_state and self.backup_grad_state[id(p)]:
                    p.start_grad()
    
    def mpi_param_broadcast(self, root=0):
        if mpi is None: return
        for p in self.parameters():
            p.assign(p.mpi_broadcast(root).detach())

def make_module(func, exec_n_args=1):
    class MakeModule(Module):
        def __init__(self, *args, **kw):
            self.args = args
            self.kw = kw
            self.__doc__ == 'doc'
        def execute(self, *args):
            return func(*args, *self.args, **self.kw)
        def __str__(self):
            return 'str'
        def __repr__(self):
            return self.__str__()
        def extra_repr(self):
            return ''

    return MakeModule


def dirty_fix_pytorch_runtime_error():
    ''' This funtion should be called before pytorch.
    Example:
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
atexit.register(jittor_exit)

Var.__str__ = lambda x: str(x.data)
Var.__repr__ = lambda x: f"jt.Var:{x.dtype}{x.uncertain_shape}"
Var.peek = lambda x: f"{x.dtype}{x.shape}"

from . import nn
from .nn import matmul
from . import contrib
