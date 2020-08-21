# ***************************************************************
# Copyright (c) 2020 Jittor. All Rights Reserved.
# Authors:
#   Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import ast
from collections import Sequence
import astunparse
import inspect

def get_py_ast_from_func(func):
    try:
        return ast.parse(inspect.getsource(func)).body[0]
    except IndentationError as e:
        src = inspect.getsource(func)
        src = src.splitlines()
        line = src[0]
        i = 0
        for i,c in enumerate(line):
            if c != ' ':
                break
        src = [ line[i:] for line in src ]
        src = '\n'.join(src)
        return ast.parse(src).body[0]

class AST:
    def __init__(self, fa=None, fa_key=None):
        self.fa = fa
        self.fa_key = fa_key

    def to_py_ast(self):
        py_type = getattr(ast, type(self).__name__)
        py_ast = py_type()
        py_ast.__dict__.update(self.__dict__)
        return py_ast

    exclude_py_attrs = set(('lineno', 'col_offset', 'end_lineno', 'end_col_offset'))

    def children(self):
        for k,v in self.__dict__.items():
            if k == 'fa':
                continue
            if isinstance(v, AST):
                yield v
            elif isinstance(v, Sequence):
                for x in v:
                    if isinstance(x, AST):
                        yield x

    def dfs(self, func):
        func(self)
        for v in self.children():
            v.dfs(func)

    def __str__(self):
        if not isinstance(self, (stmt, expr)):
            return type(self).__name__
        pir = self.to_py_ast()
        return astunparse.unparse(pir)

    def fa_list(self):
        fa_list = getattr(self.fa, self.fa_key)
        assert isinstance(fa_list, Sequence)
        return fa_list

    def fa_list_with_index(self):
        fa_list = self.fa_list()
        return fa_list, fa_list.index(self)

    def maintain_key(self, k, v):
        if v.fa is not None and v.fa is not self:
            v = v.clone()
            setattr(self, k, v)
        v.fa = self
        v.fa_key = k

    def set_attr(self, k, v):
        setattr(self, k, v)
        self.maintain_key(k, v)

    def maintain_key_seq(self, k, v, i):
        x = v[i]
        if x.fa is not None and x.fa is not self:
            x = x.clone()
            v[i] = x
        x.fa = self
        x.fa_key = k
    
    def set_attr_seq(self, k, i, o):
        v = getattr(self, k)
        v[i] = o
        self.maintain_key_seq(k, v, i)

    def maintain(self):
        '''maintain all children and fix relationship'''
        for k,v in self.__dict__.items():
            if k == 'fa':
                continue
            if isinstance(v, AST):
                self.maintain_key(k, v)
            elif isinstance(v, Sequence):
                for i in range(len(v)):
                    if isinstance(v[i], AST):
                        self.maintain_key_seq(k, v, i)

    def clone(self):
        a = type(self)()
        a.__dict__.update(self.__dict__)
        a.maintain()
        return a

    def replace(self, code):
        if isinstance(code, Sequence):
            self.append(code)
            self.delete()
            return
        fa = getattr(self.fa, self.fa_key)
        if isinstance(fa, Sequence):
            i = fa.index(self)
            self.fa.set_attr_seq(self.fa_key, i, code)
        else:
            self.fa.set_attr(self.fa_key, code)

    def append(self, code):
        fa_list, i = self.fa_list_with_index()
        if isinstance(code, Sequence):
            new_list = fa_list[:i+1] + code + fa_list[i+1:]
            fa_list.clear()
            fa_list.extend(new_list)
        else:
            fa_list.insert(i+1, code)
        self.fa.maintain()
    
    def delete(self):
        fa_list, i = self.fa_list_with_index()
        del fa_list[i]

    def check_name_used(self, s):
        found = False
        def check(self):
            nonlocal found
            if isinstance(self, Name) and self.id == s:
                found = True
        self.dfs(check)
        return found

    def replace_name(self, map):
        def check(self):
            if isinstance(self, Name) and self.id in map:
                self.id = map[self.id]
        self.dfs(check)

    def get_tmp_name(self, name_hint='_'):
        # TODO: better get tmp name
        return name_hint

    @staticmethod
    def from_py_ast(py_ast, fa=None, fa_key=None):
        type_name = type(py_ast).__name__
        if type_name not in globals():
            raise TypeError(f"Not support type<{type_name}>,\nmro:{type(py_ast).__mro__}\ndict:{py_ast.__dict__}")
        jy_type = globals()[type_name]
        jy_ast = jy_type()
        jy_ast.fa = fa
        jy_ast.fa_key = fa_key
        for k,v in py_ast.__dict__.items():
            if k in AST.exclude_py_attrs:
                continue
            if k not in jy_ast.__dict__ and not (v is None or (isinstance(v,Sequence) and len(v)==0)):
                print(v)
                raise AttributeError(f"Not support attribute for <{type_name}>, expect {k}, but got {jy_ast.__dict__.keys()}")
            setattr(jy_ast, k, v)
        for k,v in jy_ast.__dict__.items():
            if isinstance(v, ast.AST):
                setattr(jy_ast, k, AST.from_py_ast(v, jy_ast, k))
            elif isinstance(v, Sequence):
                for i in range(len(v)):
                    if isinstance(v[i], ast.AST):
                        v[i] = AST.from_py_ast(v[i], jy_ast, k)
        return jy_ast

    @staticmethod
    def from_func(func):
        pir = get_py_ast_from_func(func)
        jir = AST.from_py_ast(pir)
        return jir

    
class stmt(AST):
    pass
    
class expr(AST):
    pass
    
class expr_context(AST):
    pass

class operator(AST):
    pass

class slice(AST):
    pass
    
class Store(expr_context): pass
class Load(expr_context): pass

class Add(operator): pass
class Sub(operator): pass
class Mult(operator): pass
class MatMult(operator): pass
class Div(operator): pass
class Mod(operator): pass
class Pow(operator): pass
class LShift(operator): pass            
class RShift(operator): pass
class BitOr(operator): pass
class BitXor(operator): pass
class BitAnd(operator): pass
class FloorDiv(operator): pass

class FunctionDef(stmt):
    def __init__(self, name=None, args=None, body=None, decorator_list=None, returns=None, type_comment=None):
        super().__init__()
        self.name = name
        self.args = args
        self.body = body
        self.decorator_list = decorator_list
        self.returns = returns
        self.type_comment = type_comment
        self.maintain()

class arguments(AST):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.maintain()

class arg(AST):
    def __init__(self, arg=None):
        super().__init__()
        self.arg = arg
        self.maintain()

class For(stmt):
    def __init__(self, target=None, iter=None, body=None):
        super().__init__()
        self.target = target
        self.iter = iter
        self.body = body
        self.maintain()

class Name(expr):
    def __init__(self, id=None, ctx=None):
        super().__init__()
        self.id = id
        self.ctx = ctx
        self.maintain()

class Call(expr):
    def __init__(self, func=None, args=None):
        super().__init__()
        self.func = func
        self.args = args
        self.maintain()

class Subscript(expr):
    def __init__(self, value=None, slice=None, ctx=None):
        super().__init__()
        if isinstance(slice, (int,float)):
            slice = Index(Constant(slice))
        elif isinstance(slice, Constant):
            slice = Index(slice)
        self.value = value
        self.slice = slice
        self.ctx = ctx
        self.maintain()

class Attribute(expr):
    def __init__(self, value=None, attr=None, ctx=None):
        super().__init__()
        self.value = value
        self.attr = attr
        self.ctx = ctx
        self.maintain()
        
class Index(slice):
    def __init__(self, value=None):
        super().__init__()
        self.value = value
        self.maintain()

class Constant(expr):
    def __init__(self, value=None, kind=None):
        super().__init__()
        self.value = value
        self.kind = kind
        self.maintain()

class Assign(stmt):
    def __init__(self, targets=None, value=None):
        super().__init__()
        if isinstance(targets, str):
            targets = [Name(targets)]
        elif isinstance(targets, AST):
            targets = [targets]
        self.targets = targets
        self.value = value
        self.maintain()

class Tuple(expr):
    def __init__(self, elts=None, ctx=None):
        super().__init__()
        self.elts = elts
        self.ctx = ctx
        self.maintain()

class List(expr):
    def __init__(self, elts=None, ctx=None):
        super().__init__()
        self.elts = elts
        self.ctx = ctx

class AugAssign(stmt):
    def __init__(self, target=None, op=None, value=None):
        super().__init__()
        self.target = target
        self.op = op
        self.value = value
        self.maintain()


class If(stmt):
    def __init__(self, test=None, body=None, orelse=None):
        super().__init__()
        self.test = test
        self.body = body
        self.orelse = orelse
        self.maintain()

class BinOp(expr):
    def __init__(self, left=None, op=None, right=None):
        super().__init__()
        self.left = left
        self.op = op
        self.right = right
        self.maintain()
