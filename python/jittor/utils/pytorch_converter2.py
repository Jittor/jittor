# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import ast, astunparse

def convert(code):
    a = ast.parse(code)
    a.body.insert(0, ast.parse('import jittor as jt').body[0])
    a.body.insert(1, ast.parse('from jittor import init').body[0])
    dfs(a)
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
        if a.names[0].name == 'torch.nn' and a.names[0].asname == 'nn':
            a.names[0].name = 'jittor.nn'
            a.names[0].asname = 'nn'
    elif isinstance(a, ast.ImportFrom):
        if a.module == 'torch': 
            a.module = a.module.replace('torch', 'jittor')
            return a
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
            if isinstance(a.args[0], ast.List):
                objs = [elt.id for elt in a.args[0].elts]
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
            con = astunparse.unparse(a.func).split('.size')[0] + '.shape[' + ','.join(ags) + ']'
            return ast.parse(con).body[0].value
        if astunparse.unparse(a.func).startswith("F."):
            a.func.value.id = "nn"
            return a
        if "kaiming_normal_" in astunparse.unparse(a.func):
            ag = astunparse.unparse(a.args[0]).strip('\n')
            kws = {}
            for kw in a.keywords:
                tmp = astunparse.unparse(kw).split('=')
                kws[tmp[0]] = tmp[1].strip('\n')
            con = 'init.relu_invariant_gauss_(' + ag + ', mode=' + kws['mode'] + ')'
            return ast.parse(con).body[0].value
        if "constant_" in astunparse.unparse(a.func):
            ags = [astunparse.unparse(ag).strip('\n') for ag in a.args]
            con = 'init.constant_(' + ','.join(ags) + ')'
            return ast.parse(con).body[0].value
        if "ReLU" in astunparse.unparse(a.func):
            a.args.clear()
            a.keywords.clear()
        elif "Conv2d" in astunparse.unparse(a.func):
            pass
        elif "AvgPool2d" in astunparse.unparse(a.func):
            a.keywords.append(ast.keyword(arg='op', value=ast.Str(s='mean')))
        elif "MaxPool2d" in astunparse.unparse(a.func):
            a.keywords.append(ast.keyword(arg='op', value=ast.Str(s='maximum')))
        for kw in a.keywords:
            if kw.arg in ['return_indices', 'groups']:
                kw.value = ast.NameConstant(value=None)
    elif isinstance(a, ast.Expr): pass
    elif isinstance(a, ast.Attribute) or isinstance(a, ast.Name): replace(a)
    elif isinstance(a, ast.FunctionDef):
        if a.name == 'forward': a.name = 'execute'
    if hasattr(a, '__dict__'):
        for k in a.__dict__.keys():
            if isinstance(a.__dict__[k], list):
                for i,a_ in enumerate(a.__dict__[k]):
                    ret = dfs(a_)
                    if ret is not None:
                        a.__dict__[k][i] = ret
                            
            else:
                ret = dfs(a.__dict__[k])
                if ret is not None:
                    a.__dict__[k] = ret