# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import ast, astunparse

def convert(code):
    a = ast.parse(code)
    dfs(a)
    a.body.insert(0, ast.parse('import jittor as jt').body[0])
    a.body.insert(1, ast.parse('from jittor import init').body[0])
    a.body.insert(2, ast.parse('from jittor import nn').body[0])
    return astunparse.unparse(a)

def replace(a):
    if hasattr(a, "attr"):
        if a.attr == "Conv2d": a.attr = "Conv"
        if a.attr == "BatchNorm2d": a.attr = "BatchNorm"
        if a.attr == "ReLU": a.attr = "Relu"
        if a.attr == "AvgPool2d": a.attr = "Pool"
        if a.attr == "MaxPool2d": a.attr = "Pool"
    if hasattr(a, "id"):
        if a.id == "Conv2d": a.id = "Conv"
        if a.id == "BatchNorm2d": a.id = "BatchNorm"
        if a.id == "ReLU": a.id = "Relu"
        if a.id == "AvgPool2d": a.id = "Pool"
        if a.id == "MaxPool2d": a.id = "Pool"

def dfs(a):
    if isinstance(a, ast.Import):
        if 'torch' in astunparse.unparse(a) and 'init' in astunparse.unparse(a):
            return ast.parse('from jittor import init').body[0]
        if 'torch' in astunparse.unparse(a) and 'nn' in astunparse.unparse(a):
            return ast.parse('from jittor import nn').body[0]
        if a.names[0].name == 'torch': 
            return 'delete'
    elif isinstance(a, ast.ImportFrom):
        if 'torch' in a.module: 
            return 'delete'
    elif isinstance(a, ast.Call):
        for idx, ag in enumerate(a.args): 
            ret = dfs(ag)
            if ret is not None:
                a.args[idx] = ret
        for idx, kw in enumerate(a.keywords): 
            ret = dfs(kw)
            if ret is not None:
                a.keywords[idx] = ret
        if ".load_state_dict" in astunparse.unparse(a.func):
            a.func.attr = 'load_parameters'
        if astunparse.unparse(a.func).startswith("torch.Tensor"):
            a.func.value.id = 'jt'
            a.func.attr = 'array'
        if ".cat" in astunparse.unparse(a.func):
            if len(a.args) == 1:
                dim = a.keywords[0].value.n
            else:
                dim = a.args[1].n
            if isinstance(a.args[0], (ast.List, ast.Tuple)):
                objs = []
                for elt in a.args[0].elts:
                    if isinstance(elt, ast.Call):
                        objs.append(astunparse.unparse(elt))
                con = 'jt.contrib.concat([' + ','.join(objs) + '], dim=' + str(dim) + ')'
            else:
                con = 'jt.contrib.concat(' + a.args[0].id + ', dim=' + str(dim) + ')'
            return ast.parse(con).body[0].value
        if "view" in astunparse.unparse(a.func):
            ags = [astunparse.unparse(ag).strip('\n') for ag in a.args]
            con = 'jt.reshape(' + a.func.value.id + ', [' + ','.join(ags) + '])'
            return ast.parse(con).body[0].value
        if "permute" in astunparse.unparse(a.func):
            ags = [astunparse.unparse(ag).strip('\n') for ag in a.func.value.args]
            con = 'jt.transpose(' + a.func.value.func.value.id + ', [' + ','.join(ags) + '])'
            return ast.parse(con).body[0].value
        if astunparse.unparse(a.func).strip('\n').endswith(".size"):
            ags = [astunparse.unparse(ag).strip('\n') for ag in a.args]
            if len(ags) != 0:
                con = astunparse.unparse(a.func).split('.size')[0] + '.shape[' + ','.join(ags) + ']'
            else:
                con = astunparse.unparse(a.func).replace('size', 'shape')
            return ast.parse(con).body[0].value
        if astunparse.unparse(a.func).startswith("F."):
            a.func.value.id = "nn"
            return a
        if "kaiming_normal_" in astunparse.unparse(a.func):
            ags = [astunparse.unparse(ag).strip('\n') for ag in a.args]
            kws_ = [astunparse.unparse(kw).strip('\n') for kw in a.keywords]
            kws = [kw for kw in kws_ if 'mode' in kw]
            con = 'init.relu_invariant_gauss_(' + ','.join(ags) + ',' + ','.join(kws) + ')'
            return ast.parse(con).body[0].value
        if "constant_" in astunparse.unparse(a.func):
            ags = [astunparse.unparse(ag).strip('\n') for ag in a.args]
            kws = [astunparse.unparse(kw).strip('\n') for kw in a.keywords]
            con = 'init.constant_(' + ','.join(ags) + ',' + ','.join(kws) + ')'
            return ast.parse(con).body[0].value
        if "normal_" in astunparse.unparse(a.func):
            ags = [astunparse.unparse(ag).strip('\n') for ag in a.args]
            kws = [astunparse.unparse(kw).strip('\n') for kw in a.keywords]
            con = 'init.gauss_(' + ','.join(ags) + ',' + ','.join(kws) + ')'
            return ast.parse(con).body[0].value
        if "flatten" in astunparse.unparse(a.func):
            ags = [astunparse.unparse(ag).strip('\n') for ag in a.args]
            kws = [astunparse.unparse(kw).strip('\n') for kw in a.keywords]
            if ags[1] == '1':
                con = 'jt.reshape(' + ags[0] + ',(' + ags[0] + '.shape[0],-1))'
                return ast.parse(con).body[0].value
        if ".ReLU" in astunparse.unparse(a.func):
            a.args.clear()
            a.keywords.clear()
        elif ".AvgPool2d" in astunparse.unparse(a.func):
            a.keywords.append(ast.keyword(arg='op', value=ast.Str(s='mean')))
        elif ".MaxPool2d" in astunparse.unparse(a.func):
            a.keywords.append(ast.keyword(arg='op', value=ast.Str(s='maximum')))
        for kw in a.keywords:
            if kw.arg in ['return_indices']:
                kw.value = ast.NameConstant(value=None)
            if kw.arg in ['groups']:
                kw.value = ast.NameConstant(value=1)
    elif isinstance(a, ast.Expr): pass
    elif isinstance(a, ast.Attribute) or isinstance(a, ast.Name): replace(a)
    elif isinstance(a, ast.FunctionDef):
        if a.name == 'forward': a.name = 'execute'
    if hasattr(a, '__dict__'):
        for k in a.__dict__.keys():
            if isinstance(a.__dict__[k], list):
                delete_flag = []
                for i,a_ in enumerate(a.__dict__[k]):
                    ret = dfs(a_)
                    if ret is 'delete':
                        delete_flag.append(True)
                        del a.__dict__[k][i]
                        continue
                    if ret is not None:
                        a.__dict__[k][i] = ret
                    delete_flag.append(False)
                tmp = [a_ for i,a_ in enumerate(a.__dict__[k]) if delete_flag[i] == False]
                a.__dict__[k] = tmp
            else:
                ret = dfs(a.__dict__[k])
                if ret is not None:
                    a.__dict__[k] = ret