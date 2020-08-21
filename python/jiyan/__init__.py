# ***************************************************************
# Copyright (c) 2020 Jittor. All Rights Reserved.
# Authors:
#   Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
from jiyan.backend.numpy_backend import numpy_backend
import jiyan.ast as ja
import jittor as jt
from .pass_runner import PassRunner
from .autograd import autograd

def jiyan_compiler(func, backend, args, kw):
    func_ast = ja.AST.from_func(func)
    pr = PassRunner()
    pr.run(func_ast)
    

class JiyanFunction(jt.Function):
    def __init__(self, backend, func):
        self.backend = backend
        self.func = func

    def compile(self, args):
        self.args = args
        func_ast = ja.AST.from_func(func)
        pr = PassRunner()
        pr.run(func_ast)
        grad_ast = autograd(func_ast)
        self.forward, self.backward = ...
    
    def execute(self, *args):
        self.compile(args)
        # n args in, n args out
        return self.forward(*args)

    def grad(self, *args):
        # 2n args in, n args out
        return self.backward(*self.args, *args)

    def __call__(self, *args):
        ret = super().__call__(*args)
        assert len(ret) == len(args)
        for a, b in zip(ret, args):
            a.swap(b)


def jit(backend=numpy_backend):
    def inner(func):
        return JiyanFunction(backend, func)
    return inner

# def add(a, b):
#     ...

# def add_grad(a, b, grad_a, grad_b, ret_a, ret_b):
#     ...

# aa = a.copy()
# bb = b.copy()
# aa 

# directory organize
#  
# jiyan/
#     ast.py
#     pass_manager.py
#     autograd.py
#     pass/
#         split_tuple_assign.py
#         ......
#     backend/
#         numpy_backend.py
#         ......
#     test/
#         __main__.py
#         ......