# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import sys
import contextlib
import os
import signal
import jittor as jt
jt.dirty_fix_pytorch_runtime_error()
import torch

class CallTree:
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
        self.children = []
        self.input = []
        self.output = []
        self.args = None
        if parent is not None:
            parent.children.append(self)

    def __str__(self):
        ss = []
        def dfs(v, depth):
            s = "    "*depth+f"{v.name} in:{v.input} out:{v.output}"
            if v.args is not None:
                s += f" args:{v.args}"
            ss.append(s)
            if len(v.children):
                for c in v.children:
                    dfs(c, depth+1)
                ss.append(s + " end")
        dfs(self, 0)
        return "\n".join(ss)

    def to_jt(self):
        defs = []
        template = {
            "add": "{0} + {1}",
            "mul": "{0} * {1}",
            "getitem": "{0}[{1}]",
            "gt": "{0} > {1}",
        }
        def dfs(v):
            if len(v.children)==0:
                return
            code = []
            code.append(f"def {v.name.split('.')[0]}({','.join(map(str,v.input))}):")
            for c in v.children:
                # parse the argument into jittor code
                # code.append(f"    # {c.args}")
                if c.name == "BatchNorm2d.forward":
                    bn = c.args["self"]
                    code.append(f"    {c.output[0]} = jt.nn.batch_norm({c.input[0]}, is_train={bn.training}, eps={bn.eps}, momentum={bn.momentum})")
                    continue
                if c.name == "ReLU.forward":
                    code.append(f"    {c.output[0]} = jt.nn.relu({c.input[0]})")
                    continue
                if c.name == "MaxPool2d.forward":
                    po = c.args["self"]
                    code.append(f"    {c.output[0]} = jt.nn.pool({c.input[0]}, size={po.kernel_size}, op='maximum', padding={po.padding}, stride={po.stride})")
                    continue
                if c.name == "Conv2d.forward":
                    mod = c.args["self"]
                    code.append(f"    # {mod}")
                    assert mod.kernel_size[0] == mod.kernel_size[1]
                    assert mod.padding[0] == mod.padding[1]
                    assert mod.stride[0] == mod.stride[1]
                    assert mod.bias == False                
                    code.append(f"    {c.output[0]} = nn.conv({c.output[0]}, {mod.in_channels}, {mod.out_channels}, {mod.kernel_size[0]}, {mod.padding[0]}, {mod.stride[0]})")
                    continue
                if c.name.startswith("inj"):
                    if c.name.endswith("__init__"):
                        code.append(f"    {c.args[0]} = jt.array({c.args[1]})")
                    else:
                        assert c.name.startswith("inj_torch_Tensor___") and \
                            c.name.endswith("__")
                        name = c.name[19:-2]
                        if name in template:
                            code.append(f"    {c.output[0]} = {template[name].format(*c.args)}")
                        else:
                            code.append(f"    {c.output[0]} = __{name}__({', '.join(map(str,c.args))})")
                else:
                    dfs(c)
                    out = ""
                    if len(c.output):
                        out = f"{','.join(map(str, c.output))} = "
                    code.append(f"    {out}{c.name.split('.')[0]}({','.join(map(str,c.input))})")
            if len(v.output):
                code.append(f"    return {','.join(map(str, v.output))}")
            defs.extend(code)
        dfs(self)
        return "\n".join(defs)

class TNode:
    def __init__(self, s, v):
        self.s = s
        self.v = v
    def __str__(self):
        return self.s
    def __repr__(self):
        return self.s

trace_depth = 0
stack = []
g_vars = {}
g_var_id = 0
g_func_names = []
call_tree = CallTree(None, "root")

def push_stack(name=None, input=[]):
    global trace_depth, call_tree
    trace_depth += 1
    if name is not None:
        # Do not re record functional
        if len(stack) and (
            stack[-1][1].startswith("functional.") or
            stack[-1][1].startswith("inj_")
        ):
            return
        call_tree = CallTree(call_tree, name)
        call_tree.input = input
        stack.append((trace_depth, name))
        return call_tree
    return None
    
def pop_stack(output=[]):
    global trace_depth, call_tree
    if len(stack) and stack[-1][0] == trace_depth:
        stack.pop()
        call_tree.output = output
        call_tree = call_tree.parent
    trace_depth -= 1

def trace_calls(frame, event, arg):
    def dfs(obj, func):
        if isinstance(obj, list):
            for i,v in enumerate(obj):
                dfs(v, func)
                if isinstance(v, torch.Tensor):
                    obj[i] = g_vars[id(v)]
        elif isinstance(obj, dict):
            for k,v in obj.items():
                if isinstance(v, tuple):
                    v = list(v)
                    obj[k] = v
                dfs(v, func)
                if isinstance(v, torch.Tensor):
                    obj[k] = g_vars[id(v)]
        elif isinstance(obj, torch.Tensor):
            func(obj)
    global g_var_id
    if event.endswith('call'):
        co = frame.f_code
        func_name = co.co_name
        func_line_no = frame.f_lineno
        func_filename = co.co_filename
        args = "???"
        t_values = []
        if event == "c_call":
            func_name = arg.__name__
        else:
            args = list(frame.f_locals.keys())
            if "self" in frame.f_locals:
                func_name = type(frame.f_locals["self"]).__name__ + "." + func_name
            
            val = {k:frame.f_locals[k] for k in args}
            def func(v):
                global g_var_id
                if id(v) not in g_vars:
                    if func_name.endswith("__init__"):
                        g_vars[id(v)] = TNode("array_"+str(g_var_id), v)
                    else:
                        g_vars[id(v)] = TNode("input_"+str(g_var_id), v)
                    g_var_id += 1
                t_values.append(g_vars[id(v)])
            dfs(val, func)
        
        # get arguments you want
        if func_name.endswith(".forward"):
            ct = push_stack(func_name, t_values)
            ct.args = val
        elif func_filename.endswith("functional.py"): # TODO: not stable
            push_stack("functional."+func_name, t_values)
        elif func_name.startswith("inj_"):
            ct = push_stack(func_name, t_values)
            ct.args = val["a"]
        elif func_name in g_func_names:
            push_stack(func_name, t_values)
        else:
            push_stack()
        jt.LOG.vvvv("----"*trace_depth+f"call: {func_name}({args}){t_values}     # {func_filename}:{func_line_no}")
    elif event.endswith('return'):
        ret = []
        if event == "c_return":
            jt.LOG.vvvv("----"*trace_depth+f"return {arg.__name__}: ???")
        else:
            co = frame.f_code
            func_name = co.co_name
            def func(arg):
                global g_var_id
                if id(arg) not in g_vars:
                    node = TNode(f"out_{g_var_id}", arg)
                    g_vars[id(arg)] = node
                else:
                    node = g_vars[id(arg)]
                ret.append(node)
                g_var_id += 1
            dfs(arg, func)
            if "self" in frame.f_locals:
                func_name = type(frame.f_locals["self"]).__name__ + "." + func_name
            jt.LOG.vvvv("----"*trace_depth+f"return {func_name}: {ret}")
        pop_stack(ret)
    return trace_calls

@contextlib.contextmanager
def trace_scope(func_names=[]):
    global g_func_names
    g_func_names = func_names
    with func_injection():
        try:
            global trace_depth, g_var_id
            sys.settrace(trace_calls)
            trace_depth = 1
            stack.clear()
            g_vars.clear()
            call_tree.children.clear()

            g_var_id = 0
            yield
        finally:
            sys.settrace(None)
            jt.LOG.v("="*20)
            jt.LOG.v(call_tree)


@contextlib.contextmanager
def func_injection():
    names = [
        "torch.Tensor.__init__",
        "torch.Tensor.__add__",
        "torch.Tensor.__mul__",
        "torch.Tensor.__sub__",
        "torch.Tensor.__truediv__",
        "torch.Tensor.__floordiv__",
        "torch.Tensor.__getitem__",
        # "torch.Tensor.__setitem__",
        "torch.Tensor.__pow__",
        "torch.Tensor.__mod__",
        "torch.Tensor.__lt__",
        "torch.Tensor.__le__",
        "torch.Tensor.__gt__",
        "torch.Tensor.__ge__",
        "torch.Tensor.__eq__",
        "torch.Tensor.__ne__",
        "torch.Tensor.__lshift__",
        "torch.Tensor.__rshift__",
        "torch.Tensor.__and__",
        "torch.Tensor.__or__",
        "torch.Tensor.__xor__",
        "torch.Tensor.__abs__",
        "torch.Tensor.__neg__",
    ]
    try:
        global inject_prevs
        inject_prevs = []
        for name in names:
            inject_prevs.append(eval(name))
        for i, name in enumerate(names):
            new_name = "inj_" + name.replace(".", "_")
            if name.endswith("__getitem__"):
                exec(f"def {new_name}(*a): return torch._C._TensorBase.__getitem__(a[0], a[1] if isinstance(a[1], tuple) else (a[1],))")
            elif name.endswith("__init__"):
                exec(f"def {new_name}(*a, **b): return None")
            else:
                exec(f"def {new_name}(*a, **b): return inject_prevs[{i}](*a, **b)")
            jt.LOG.v("inject", new_name)
            exec(f"{name} = {new_name}")
        yield
    finally:
        for i, name in enumerate(names):
            prev = inject_prevs[i]
            exec(f"{name} = prev")
        torch.Tensor.__getitem__ = \
            lambda s, a: torch._C._TensorBase.__getitem__(s, a if isinstance(a, tuple) else (a,))
