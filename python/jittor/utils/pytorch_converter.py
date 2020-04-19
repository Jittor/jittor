# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import ast, astunparse
import numpy as np

pjmap = {
    # ***************************************************************
    # Module
    # ***************************************************************
    'Conv2d': {
        'pytorch': {
            'args': "in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'"
        },
        'jittor': {
            'module': 'nn',
            'name': 'Conv',
            'args': 'in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True'
        },
        'links': {},
        'extras': {},
    },
    'ConvTranspose2d': {
        'pytorch': {
            'args': "in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'"
        },
        'jittor': {
            'module': 'nn',
            'name': 'ConvTranspose',
            'args': 'in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1'
        },
        'links': {},
        'extras': {},
    },
    'MaxPool2d': {
        'pytorch': {
            'args': 'kernel_size, stride=None, padding=0, dilation=1, return_indices=False', 
        },
        'jittor': {
            'module': 'nn',
            'name': 'Pool',
            'args': 'kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, op="maximum"'
        },
        'links': {},
        'extras': {
            "op": "'maximum'",
        },
    },
    'AvgPool2d': {
        'pytorch': {
            'args': 'kernel_size, stride=None, padding=0, dilation=1, return_indices=False', 
        },
        'jittor': {
            'module': 'nn',
            'name': 'Pool',
            'args': 'kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, op="maximum"'
        },
        'links': {},
        'extras': {
            "op": "'mean'",
        },
    },
    'ReLU': {
        'pytorch': {
            'args': 'inplace=False', 
        },
        'jittor': {
            'module': 'nn',
            'name': 'ReLU',
            'args': ''
        },
        'links': {},
        'extras': {},
    },
    'BatchNorm2d': {
        'pytorch': {
            'args': 'num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True', 
        },
        'jittor': {
            'module': 'nn',
            'name': 'BatchNorm',
            'args': 'num_features, eps=1e-5, momentum=0.1, affine=None, is_train=True'
        },
        'links': {},
        'extras': {},
    },
    'LeakyReLU': {
        'pytorch': {
            'args': 'negative_slope=0.01, inplace=False', 
        },
        'jittor': {
            'module': 'nn',
            'name': 'Leaky_relu',
            'args': ''
        },
        'links': {},
        'extras': {},
    },
    'kaiming_normal_': {
        'pytorch': {
            'args': "tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'", 
        },
        'jittor': {
            'module': 'init',
            'name': 'relu_invariant_gauss_',
            'args': 'var, mode="fan_in"'
        },
        'links': {'tensor': 'var'},
        'extras': {},
    },
    'constant_': {
        'pytorch': {
            'args': "tensor, val", 
        },
        'jittor': {
            'module': 'init',
            'name': 'constant_',
            'args': 'var, value=0.0'
        },
        'links': {'tensor': 'var', 'val': 'value'},
        'extras': {},
    },
    'normal_': {
        'pytorch': {
            'args': "tensor, mean=0.0, std=1.0", 
        },
        'jittor': {
            'module': 'init',
            'name': 'gauss_',
            'args': 'var, mean=0.0, std=1.0'
        },
        'links': {'tensor': 'var'},
        'extras': {},
    },
    'cat': {
        'pytorch': {
            'args': "tensors, dim=0, out=None", 
        },
        'jittor': {
            'module': 'jt.contrib',
            'name': 'concat',
            'args': 'vars, dim=0'
        },
        'links': {'tensors': 'vars'},
        'extras': {},
    },
    # ***************************************************************
    # Convert format for function which can be writen as either torch.Tensor.xxx(...) or torch.xxx(torch.Tensor, ...)
    #       Example: x.reshape([2,3]) and torch.reshape(x, [2,3])
    # ***************************************************************
    'flatten': {
        'pytorch': {
            'prefix': ['torch'],
            'args_prefix': 'input, start_dim=0, end_dim=-1',
            'args': 'start_dim=0, end_dim=-1',
        },
        'jittor': {
            'prefix': 'jt',
            'module': '',
            'name': 'flatten',
            'args_prefix': 'input, start_dim=0, end_dim=-1',
            'args': 'start_dim=0, end_dim=-1'
        },
        'links': {'aaaaa': 'bbbb'},
        'extras': {},
    },
    'reshape': {
        'pytorch': {
            'prefix': ['torch'],
            'args_prefix': 'input, shape',
            'args': 'shape',
        },
        'jittor': {
            'prefix': 'jt',
            'module': '',
            'name': 'reshape',
            'args_prefix': 'input, shape',
            'args': 'shape'
        },
        'links': {},
        'extras': {},
    },
    'permute': {
        'pytorch': {
            'prefix': [],
            'args_prefix': '',
            'args': '*dim',
        },
        'jittor': {
            'prefix': '',
            'module': '',
            'name': 'permute',
            'args_prefix': '',
            'args': '*dim'
        },
        'links': {},
        'extras': {},
    },
    # 好像不需要如果一毛一样的话
    'view': {
        'pytorch': {
            'prefix': [],
            'args_prefix': '',
            'args': '*shape',
        },
        'jittor': {
            'prefix': '',
            'module': '',
            'name': 'view',
            'args_prefix': '',
            'args': '*shape'
        },
        'links': {},
        'extras': {},
    }
}

unsupport_ops = [
    # ***************************************************************
    # torch.nn
    # ***************************************************************
    'Parameter', 'ModuleList', 'ModuleDict', 'ParameterList', 'ParameterDict', 
    'Conv1d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose3d', 'Unfold', 'Fold', 
    'MaxPool1d', 'MaxPool3d', 'MaxUnpool1d', 'MaxUnpool2d', 'MaxUnpool3d', 'AvgPool1d', 'AvgPool3d', 'FractionalMaxPool2d', 'LPPool1d', 'LPPool2d', 'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d', 'AdaptiveAvgPool1d', 'AdaptiveAvgPool3d', 
    'ReflectionPad1d', 'ReflectionPad2d', 'ReplicationPad1d', 'ReplicationPad2d', 'ReplicationPad3d', 'ZeroPad2d', 'ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d', 'ELU', 'Hardshrink', 'Hardtanh', 'LeakyReLU', 'LogSigmoid', 'MultiheadAttention', 
    'PReLU', 'RReLU', 'SELU', 'CELU', 'GELU', 'Softplus', 'Softshrink', 'Softsign', 'Tanhshrink', 'Threshold', 'Softmin', 'Softmax2d', 'LogSoftmax', 'AdaptiveLogSoftmaxWithLoss', 'BatchNorm1d', 'BatchNorm3d', 'GroupNorm', 'SyncBatchNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d', 'LayerNorm', 'LocalResponseNorm', 'RNNBase', 'RNN', 'LSTM', 'GRU', 'RNNCell', 'LSTMCell', 'GRUCell', 'Transformer', 'TransformerEncoder', 'TransformerDecoder', 'TransformerEncoderLayer', 'TransformerDecoderLayer', 'Identity', 'Bilinear', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'Embedding', 'EmbeddingBag', 'CosineSimilarity', 'PairwiseDistance', 'L1Loss', 'MSELoss', 'CTCLoss', 'NLLLoss', 'PoissonNLLLoss', 'KLDivLoss', 'BCELoss', 'BCEWithLogitsLoss', 'MarginRankingLoss', 'HingeEmbeddingLoss', 'MultiLabelMarginLoss', 'SmoothL1Loss', 'SoftMarginLoss', 'MultiLabelSoftMarginLoss', 'CosineEmbeddingLoss', 'MultiMarginLoss', 'TripletMarginLoss', 'PixelShuffle', 'Upsample', 'UpsamplingNearest2d', 'UpsamplingBilinear2d', 'DataParallel', 'DistributedDataParallel', 'clip_grad_norm_', 'clip_grad_value_', 'parameters_to_vector', 'vector_to_parameters', 'BasePruningMethod', 'PruningContainer', 'Identity', 'RandomUnstructured', 'L1Unstructured', 'RandomStructured', 'LnStructured', 'CustomFromMask', 'identity', 'random_unstructured', 'l1_unstructured', 'random_structured', 'ln_structured', 'global_unstructured', 'custom_from_mask', 'remove', 'is_pruned', 'weight_norm', 'remove_weight_norm', 'spectral_norm', 'remove_spectral_norm', 'PackedSequence', 'pack_padded_sequence', 'pad_packed_sequence', 'pad_sequence', 'pack_sequence'
]

support_ops = {}
for key in pjmap.keys():
    module = pjmap[key]['jittor']['module']
    name = pjmap[key]['jittor']['name']
    if module == 'nn':
        support_ops[key] = name


def replace(a):
    if hasattr(a, "attr") and a.attr in unsupport_ops:
        raise RuntimeError(f'{a.attr} is not supported in Jittor yet. We will appreciate it if you code {a.attr} function and make pull request at https://github.com/Jittor/jittor.')
    
    if hasattr(a, "id") and a.id in unsupport_ops:
        raise RuntimeError(f'{a.id} is not supported in Jittor yet. We will appreciate it if you code {a.id} function and make pull request at https://github.com/Jittor/jittor.')

    if hasattr(a, "attr"):
        if a.attr in support_ops.keys(): a.attr = support_ops[a.attr]
        
    if hasattr(a, "id"):
        if a.id in support_ops.keys(): a.id = support_ops[a.id]

import_flag = []
def convert(code):
    a = ast.parse(code)
    dfs(a)
    a.body.insert(0, ast.parse('import jittor as jt').body[0])
    if 'init' not in import_flag:
        a.body.insert(1, ast.parse('from jittor import init').body[0])
    if 'nn' not in import_flag:
        a.body.insert(2, ast.parse('from jittor import nn').body[0])
    return astunparse.unparse(a)

def convert_(prefix, func_name, ags, kws):
    info = pjmap[func_name]
    p_prefix = info['pytorch']['prefix'] if 'prefix' in info['pytorch'].keys() else None
    if p_prefix is not None and prefix in p_prefix:
        p_ags = info['pytorch']['args_prefix']
        j_ags = info['jittor']['args_prefix']
    else:
        p_ags = info['pytorch']['args']
        j_ags = info['jittor']['args']
    j_prefix = info['jittor']['prefix'] if 'prefix' in info['jittor'].keys() else None
    j_module = info['jittor']['module']
    j_name = info['jittor']['name']
    links = info['links']
    extras = info['extras']
    jj_ags = []
    jj_kws = {}
    pp_ags = []
    pp_kws = {}
    if j_ags == '' and p_ags == '':
        # no args in Pytorch and Jittor.
        if p_prefix is None:
            return f"{j_module}.{j_name}()"
        else:
            if prefix in p_prefix:
                return f"{j_prefix}.{j_name}()"
            else:
                return f"{prefix}.{j_name}()"
    else:
        j_ags = j_ags.replace(' ','').split(',')
        for j_ag in j_ags:
            if '=' in j_ag:
                k,v = j_ag.split('=')
                jj_kws[k] = v
            else:
                jj_ags.append(j_ag)
        p_ags = p_ags.replace(' ','').split(',')
        for p_ag in p_ags:
            if '=' in p_ag:
                k,v = p_ag.split('=')
                pp_kws[k] = v
            else:
                pp_ags.append(p_ag)
        if len(jj_ags) == 0 and len(pp_ags) != 0:
            raise AttributeError(f"{func_name} in Jittor has no Attribute {pp_ags[0]}")
    if len(pp_ags) > len(ags) + len(kws):
        raise RuntimeError(f'There are needed {len(pp_ags) + len(list(pp_kws.keys()))} args in Pytorch {func_name} function, but you only provide {len(ags) + len(kws)}')
    ags_ = []
    for i in range(len(pp_ags)):
        if i < len(ags):
            if '*' in pp_ags[i]:
                ags_.append('(' + ', '.join(ags[i:]) + ')')
                ags = ags_
                break
            else:
                ags_.append(ags[i])
        else:
            break
    if len(pp_ags) + len(list(pp_kws.keys())) < len(ags) + len(kws):
        raise RuntimeError(f'There are only {len(pp_ags) + len(list(pp_kws.keys()))} args in Pytorch {func_name} function, but you provide {len(ags) + len(kws)}')
    j_ags_flag = np.zeros(len(jj_ags))
    j_ags_values = {}
    j_kws_values = {}
    for i,ag in enumerate(ags):
        if len(pp_ags) == 0:
            ag_name = list(pp_kws.keys())[i]
        elif i < len(pp_ags):
            ag_name = pp_ags[i]
        elif i >= len(pp_ags) and (i-len(pp_ags)) <= len(list(pp_kws.keys())):
            ag_name = list(pp_kws.keys())[i-len(pp_ags)]
        else:
            raise RuntimeError(f'The args number is not matc{func_name} in Jittor has no Attribute {ag_name}')
        if ag_name in links.keys():
            ag_name = links[ag_name]
        if ag_name in jj_ags:
            j_ags_flag[jj_ags.index(ag_name)] = 1
            j_ags_values[str(jj_ags.index(ag_name))] = ag
        elif ag_name in jj_kws.keys():
            j_kws_values[ag_name] = ag
        else:
            raise AttributeError(f'{func_name} in Jittor has no Attribute {ag_name}')
    for i,kw in enumerate(kws):
        kw_name, kw_value = kw.split('=')
        if kw_name in links.keys():
            kw_name = links[kw_name]
        if kw_name in jj_ags:
            j_ags_flag[jj_ags.index(kw_name)] = 1
            j_ags_values[str(jj_ags.index(kw_name))] = kw_value
        elif kw_name in jj_kws.keys():
            j_kws_values[kw_name] = kw_value
        else:
            raise AttributeError(f'{func_name} in Jittor has no Attribute {kw_name}')
    len_jj_ags = len(jj_ags) if len(jj_ags) == 0 or jj_ags[0] != '' else 0
    if j_ags_flag.sum() < len_jj_ags:
        missing_args = []
        for i in range(len(jj_ags)):
            if j_ags_flag[i] == 0:
                missing_args.append(jj_ags[i])
        raise AttributeError(f"the needed args of {func_name} in Jittor is {', '.join(jj_ags)}, so you need to give value of {', '.join(missing_args)}.")
    if extras:
        for k in extras.keys():
            if k in jj_ags:
                j_ags_values[str(jj_ags.index(k))] = extras[k]
            elif k in jj_kws.keys():
                j_kws_values[k] = extras[k]
            else:
                raise AttributeError(f"there is not attribute named {k} in Jittor {func_name}, you should delete it in {func_name} extras.")
    j_ags_ = [j_ags_values[str(i)] for i in range(len(list(j_ags_values.keys())))]
    j_kws_ = [key + "=" + j_kws_values[key] for key in j_kws_values.keys()]
    j_func = f"{j_module}.{j_name}({', '.join(j_ags_+j_kws_)})"
    if p_prefix is None:
        return f"{j_module}.{j_name}({', '.join(j_ags_+j_kws_)})"
    else:
        if prefix in p_prefix:
            return f"{j_prefix}.{j_name}({', '.join(j_ags_+j_kws_)})"
        else:
            return f"{prefix}.{j_name}({', '.join(j_ags_+j_kws_)})"
    return j_func

def dfs(a):
    if isinstance(a, ast.Import):
        if 'torch' in astunparse.unparse(a) and 'init' in astunparse.unparse(a):
            import_flag.append('init')
            return ast.parse('from jittor import init').body[0]
        if 'torch' in astunparse.unparse(a) and 'nn' in astunparse.unparse(a):
            import_flag.append('nn')
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
        func = astunparse.unparse(a.func).strip('\n').split('.')
        prefix = '.'.join(func[0:-1])
        func_name = func[-1]
        if func_name in unsupport_ops:
            raise RuntimeError(f'{func_name} is not supported in Jittor yet. We will appreciate it if you code {func_name} function and make pull request at https://github.com/Jittor/jittor.')
        if func_name in pjmap.keys():
            ags = [astunparse.unparse(ag).strip('\n') for ag in a.args]
            kws = [astunparse.unparse(kw).strip('\n') for kw in a.keywords]
            ret = convert_(prefix, func_name, ags, kws)
            return ast.parse(ret).body[0].value
        if ".load_state_dict" in astunparse.unparse(a.func):
            a.func.attr = 'load_parameters'
        if astunparse.unparse(a.func).strip('\n').endswith(".size"):
            ags = [astunparse.unparse(ag).strip('\n') for ag in a.args]
            if len(ags) != 0:
                con = astunparse.unparse(a.func).split('.size')[0] + '.shape[' + ','.join(ags) + ']'
            else:
                con = astunparse.unparse(a.func).replace('size', 'shape')
            return ast.parse(con).body[0].value
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