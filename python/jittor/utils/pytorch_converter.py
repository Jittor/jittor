# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
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
        'delete': ['inplace'],
    },
    'relu': {
        'pytorch': {
            'args': 'input', 
        },
        'jittor': {
            'module': 'nn',
            'name': 'relu',
            'args': 'x'
        },
        'links': {'input': 'x'},
        'extras': {},
        'delete': [],
    },
    'binary_cross_entropy_with_logits': {
        'pytorch': {
            'args': 'input, target, weight, size_average=True', 
        },
        'jittor': {
            'module': 'nn',
            'name': 'binary_cross_entropy_with_logits',
            'args': 'input, target, weight, size_average=True'
        },
        'links': {},
        'extras': {},
        'delete': [],
    },
    'ReLU6': {
        'pytorch': {
            'args': 'inplace=False', 
        },
        'jittor': {
            'module': 'nn',
            'name': 'ReLU6',
            'args': ''
        },
        'links': {},
        'extras': {},
        'delete': ['inplace'],
    },
    'PReLU': {
        'pytorch': {
            'args': 'num_parameters=1, init=0.25', 
        },
        'jittor': {
            'module': 'nn',
            'name': 'PReLU',
            'args': 'num_parameters=1, init_=0.25'
        },
        'links': {'init': 'init_'},
        'extras': {},
    },
    'LeakyReLU': {
        'pytorch': {
            'args': 'negative_slope=0.01, inplace=False', 
        },
        'jittor': {
            'module': 'nn',
            'name': 'LeakyReLU',
            'args': 'scale=0.01'
        },
        'links': {'negative_slope': 'scale'},
        'extras': {},
        'delete': ['inplace'],
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
    'BatchNorm1d': {
        'pytorch': {
            'args': "num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True"
        },
        'jittor': {
            'module': 'nn',
            'name': 'BatchNorm1d',
            'args': 'num_features, eps=1e-5, momentum=0.1, affine=None, is_train=True, sync=True',
        },
        'links': {},
        'extras': {'affine': 'None'},
        'delete': ['track_running_stats'],
    },
    'GroupNorm': {
        'pytorch': {
            'args': "num_groups, num_channels, eps=1e-05, momentum=0.1, affine=True"
        },
        'jittor': {
            'module': 'nn',
            'name': 'GroupNorm',
            'args': 'num_groups, num_channels, eps=1e-05, affine=None, is_train=True',
        },
        'links': {},
        'extras': {'affine': 'None'},
    },
    'Parameter':{
        'pytorch': {
            'args': "data,require_grad=True"
        },
        'jittor': {
            'module': 'jt',
            'name': 'array',
            'args': 'data,dtype=None',
        },
        'links': {},
        'extras': {},
    },
    'Dropout2d': {
        'pytorch': {
            'args': 'p=0.5, inplace=False', 
        },
        'jittor': {
            'module': 'nn',
            'name': 'Dropout',
            'args': 'p=0.5, is_train=False'
        },
        'links': {},
        'extras': {},
        'delete': ['inplace'],
    },
    'Upsample': {
        'pytorch': {
            'args': "size=None, scale_factor=None, mode='nearest', align_corners=None", 
        },
        'jittor': {
            'module': 'nn',
            'name': 'Upsample',
            'args': "scale_factor=None, mode='nearest'"
        },
        'links': {},
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
    'uniform_': {
        'pytorch': {
            'args': "tensor, a=0.0, b=1.0", 
        },
        'jittor': {
            'module': 'init',
            'name': 'uniform_',
            'args': 'var, low, high'
        },
        'links': {'tensor': 'var', 'a': 'low', 'b': 'high'},
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
        'links': {},
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
    'clamp': {
        'pytorch': {
            'prefix': ['torch'],
            'args_prefix': 'input, min, max, out=None',
            'args': 'min, max, out=None',
        },
        'jittor': {
            'prefix': 'jt',
            'module': '',
            'name': 'clamp',
            'args_prefix': 'x, min_v, max_v',
            'args': 'min_v, max_v'
        },
        'links': {'min': 'min_v', 'max': 'max_v'},
        'extras': {},
        'delete': ['out'],
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
    'ModuleDict', 'ParameterList', 'ParameterDict', 
    'Conv1d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose3d', 'Unfold', 'Fold', 
    'MaxPool1d', 'MaxUnpool1d', 'MaxUnpool2d', 'AvgPool1d', 
    'FractionalMaxPool2d', 'LPPool1d', 'LPPool2d', 'AdaptiveMaxPool1d', 
    'AdaptiveAvgPool1d',
    'ReflectionPad1d', 'ReplicationPad1d', 'ReplicationPad3d', 'ConstantPad1d', 'ConstantPad3d', 
    'ELU', 'Hardshrink', 'Hardtanh', 'LogSigmoid', 'MultiheadAttention', 
    'RReLU', 'SELU', 'CELU', 'Softshrink', 'Softsign', 'Tanhshrink', 
    'Threshold', 'Softmin', 'Softmax2d', 'LogSoftmax', 'AdaptiveLogSoftmaxWithLoss', 
    'BatchNorm3d', 'SyncBatchNorm', 'InstanceNorm1d', 'InstanceNorm3d', 'LocalResponseNorm', 
    # 'RNNBase', 'RNN', 'LSTM', 'GRU', 'RNNCell', 'LSTMCell', 'GRUCell', 'Transformer', 'TransformerEncoder', 
    'TransformerDecoder', 'TransformerEncoderLayer', 'TransformerDecoderLayer', # 'Identity', 'Bilinear', 
    'Dropout3d', 'AlphaDropout', 'EmbeddingBag', 'CosineSimilarity', 'PairwiseDistance', 'CTCLoss', 'NLLLoss', 'PoissonNLLLoss', 'KLDivLoss', 'BCEWithLogitsLoss', 
    'MarginRankingLoss', 'HingeEmbeddingLoss', 'MultiLabelMarginLoss', 'SmoothL1Loss', 'SoftMarginLoss', 
    'MultiLabelSoftMarginLoss', 'CosineEmbeddingLoss', 'MultiMarginLoss', 'TripletMarginLoss', # 'DataParallel', 'DistributedDataParallel', 
    'clip_grad_norm_', 'clip_grad_value_', 
    'parameters_to_vector', 'vector_to_parameters', 'BasePruningMethod', 'PruningContainer', 
    'RandomUnstructured', 'L1Unstructured', 'RandomStructured', 'LnStructured', 'CustomFromMask', 
    'random_unstructured', 'l1_unstructured', 'random_structured', 'ln_structured', 'global_unstructured', 
    'custom_from_mask', 'remove', 'is_pruned', 'weight_norm', 'remove_weight_norm', 'spectral_norm', 
    'remove_spectral_norm', 'PackedSequence', 'pack_padded_sequence', 'pad_packed_sequence', 'pad_sequence', 'pack_sequence'
]

def pjmap_append(pytorch_func_name, pytorch_args, jittor_func_module, jittor_func_name, jittor_args, extras=None, links=None, delete=None):
    ''' adding map to pjmap for converting new function, example: convert AvgPool2d to Pool
    args:
        * `pytorch_func_name`: Pytorch function name
        * `pytorch_args`: Pytorch parameter list
        * `jittor_func_module`: to which module the Jittor function belongs
        * `jittor_func_name`: Jittor function name
        * `jittor_args`: Jittor parameter list
        * `extras`: parameter assignment
        * `links`: connection parameters
        * `delete`: delete parameters

    example:
        from jittor.utils.pytorch_converter import pjmap_append
        pjmap_append(pytorch_func_name='AvgPool2d', 
                    pytorch_args='kernel_size, stride=None, padding=0, dilation=1, return_indices=False',
                    jittor_func_module='nn',
                    jittor_func_name='Pool',
                    jittor_args='kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, op="maximum"',
                    extras={"op": "'mean'"}) 
    '''
    if links == None: links = {}
    if extras == None: extras = {}
    if delete == None: delete = []
    assert isinstance(links, dict)
    assert isinstance(extras, dict)
    assert isinstance(delete, list)
    pjmap[pytorch_func_name] = {
        'pytorch': {
            'args': pytorch_args,
        },
        'jittor': {
            'module': jittor_func_module,
            'name': jittor_func_name,
            'args': jittor_args,
        },
        'links': links,
        'extras': extras,
        'delete': delete,
    }


def raise_unsupport(name, ori_src):
    ret = f"raise RuntimeError('''original source: <{ori_src.strip()}>, {name} is not supported in Jittor yet. We will appreciate it if you provide an implementation of {name} and make pull request at https://github.com/Jittor/jittor.''')"
    print(ret+'\n')
    ret = ast.parse(ret).body[0]
    return ret

class Converter:
    def __init__(self, ex_pjmap):
        import copy
        self.pjmap = copy.deepcopy(pjmap)
        if ex_pjmap:
            self.pjmap.update(ex_pjmap)
        self.unsupport_ops = set(unsupport_ops)
        support_ops = {}
        for key in self.pjmap.keys():
            module = self.pjmap[key]['jittor']['module']
            name = self.pjmap[key]['jittor']['name']
            if module == 'nn':
                support_ops[key] = name
            if key in self.unsupport_ops:
                self.unsupport_ops.remove(key)
        self.support_ops = support_ops
        self.import_flag = []

    def replace(self, a):
        if hasattr(a, "attr") and a.attr in self.unsupport_ops:
            ori_src = astunparse.unparse(a)
            return raise_unsupport(a.attr, ori_src)
        
        if hasattr(a, "id") and a.id in self.unsupport_ops:
            ori_src = astunparse.unparse(a)
            return raise_unsupport(a.id, ori_src)

        if hasattr(a, "attr"):
            if a.attr in self.support_ops.keys(): a.attr = self.support_ops[a.attr]
            
        if hasattr(a, "id"):
            if a.id in self.support_ops.keys(): a.id = self.support_ops[a.id]
            
        return None

    def convert_(self, prefix, func_name, ags, kws, ori_src):
        info = self.pjmap[func_name]
        p_prefix = info['pytorch']['prefix'] if 'prefix' in info['pytorch'].keys() else None
        if p_prefix is not None and prefix in p_prefix:
            p_ags = info['pytorch']['args_prefix']
            j_ags = info['jittor']['args_prefix']
        else:
            p_ags = info['pytorch']['args']
            j_ags = info['jittor']['args']
        if 'delete' in info.keys():
            delete = info['delete']
        else:
            delete = None
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
                return f"raise AttributeError('''origin source: <{ori_src.strip()}>, {func_name} in Jittor has no Attribute {pp_ags[0]}''')"
                # raise AttributeError(f"{func_name} in Jittor has no Attribute {pp_ags[0]}")
        if delete is not None:
            for d in delete:
                if d in pp_ags:
                    jj_ags.append(d)
                if d in pp_kws.keys():
                    jj_kws[d] = None
        if len(pp_ags) > len(ags) + len(kws):
            return f"raise RuntimeError('''origin source: <{ori_src.strip()}>, There are needed {len(pp_ags) + len(list(pp_kws.keys()))} args in Pytorch {func_name} function, but you only provide {len(ags) + len(kws)}''')"
            # raise RuntimeError(f'There are needed {len(pp_ags) + len(list(pp_kws.keys()))} args in Pytorch {func_name} function, but you only provide {len(ags) + len(kws)}')
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
            return f"raise RuntimeError('''origin source: <{ori_src.strip()}>,There are only {len(pp_ags) + len(list(pp_kws.keys()))} args in Pytorch {func_name} function, but you provide {len(ags) + len(kws)}''')"
            # raise RuntimeError(f'There are only {len(pp_ags) + len(list(pp_kws.keys()))} args in Pytorch {func_name} function, but you provide {len(ags) + len(kws)}')
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
                return f"raise RuntimeError('''origin source: <{ori_src.strip()}>,The args number is not matc{func_name} in Jittor has no Attribute {ag_name}''')"
                # raise RuntimeError(f'The args number is not matc{func_name} in Jittor has no Attribute {ag_name}')
            if ag_name in links.keys():
                ag_name = links[ag_name]
            if ag_name in jj_ags:
                j_ags_flag[jj_ags.index(ag_name)] = 1
                j_ags_values[str(jj_ags.index(ag_name))] = ag
            elif ag_name in jj_kws.keys():
                j_kws_values[ag_name] = ag
            else:
                return f"raise AttributeError('''origin source: <{ori_src.strip()}>, {func_name} in Jittor has no Attribute {ag_name}''')"
                # raise AttributeError(f'{func_name} in Jittor has no Attribute {ag_name}')
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
                return f"raise AttributeError('''origin source: <{ori_src.strip()}>, {func_name} in Jittor has no Attribute {kw_name}''')"
                # raise AttributeError(f'{func_name} in Jittor has no Attribute {kw_name}')
        len_jj_ags = len(jj_ags) if len(jj_ags) == 0 or jj_ags[0] != '' else 0
        if j_ags_flag.sum() < len_jj_ags:
            missing_args = []
            for i in range(len(jj_ags)):
                if j_ags_flag[i] == 0:
                    missing_args.append(jj_ags[i])
            return f"raise AttributeError('''origin source: <{ori_src.strip()}>, the needed args of {func_name} in Jittor is {', '.join(jj_ags)}, so you need to give value of {', '.join(missing_args)}.''')"
            # raise AttributeError(f"the needed args of {func_name} in Jittor is {', '.join(jj_ags)}, so you need to give value of {', '.join(missing_args)}.")
        if extras:
            for k in extras.keys():
                if k in jj_ags:
                    j_ags_values[str(jj_ags.index(k))] = extras[k]
                elif k in jj_kws.keys():
                    j_kws_values[k] = extras[k]
                else:
                    return f"raise AttributeError('''origin source: <{ori_src.strip()}>, there is not attribute named {k} in Jittor {func_name}, you should delete it in {func_name} extras.''')"
                    # raise AttributeError(f"there is not attribute named {k} in Jittor {func_name}, you should delete it in {func_name} extras.")
        if delete is not None:
            for d in delete:
                if d in j_ags_values:
                    del j_ags_values[d]
                if d in j_kws_values.keys():
                    j_kws_values.pop(d)
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

    def dfs(self, a):
        if isinstance(a, ast.Import):
            if 'torch' in astunparse.unparse(a) and 'init' in astunparse.unparse(a):
                self.import_flag.append('init')
                return ast.parse('from jittor import init').body[0]
            if 'torch' in astunparse.unparse(a) and a.names[0].asname == 'nn':
                self.import_flag.append('nn')
                return ast.parse('from jittor import nn').body[0]
            if 'torch' in a.names[0].name: 
                return 'delete'
        elif isinstance(a, ast.ImportFrom):
            if 'torch' in a.module:
                return 'delete'
        elif isinstance(a, ast.Call):
            for idx, ag in enumerate(a.args): 
                ret = self.dfs(ag)
                if ret is not None:
                    a.args[idx] = ret
            for idx, kw in enumerate(a.keywords): 
                ret = self.dfs(kw)
                if ret is not None:
                    a.keywords[idx] = ret
            ori_src = astunparse.unparse(a)
            func = astunparse.unparse(a.func).strip('\n').split('.')
            prefix = '.'.join(func[0:-1])
            func_name = func[-1]
            if func_name in self.unsupport_ops:
                ret = raise_unsupport(func_name, ori_src)
                return ret
            if func_name in self.pjmap:
                ags = [astunparse.unparse(ag).strip('\n') for ag in a.args]
                kws = [astunparse.unparse(kw).strip('\n') for kw in a.keywords]
                ret = self.convert_(prefix, func_name, ags, kws, ori_src)
                ret_tmp = ret
                ret = ast.parse(ret).body[0]
                if hasattr(ret,'value'):
                    return ret.value
                else:
                    print(ret_tmp+'\n')
                    return ret
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
        elif isinstance(a, ast.Attribute) or isinstance(a, ast.Name): 
            ret = self.replace(a)
            if ret is not None:
                print(ret)
                return ret
        elif isinstance(a, ast.FunctionDef):
            if a.name == 'forward': a.name = 'execute'
        if hasattr(a, '__dict__'):
            for k in a.__dict__.keys():
                if isinstance(a.__dict__[k], list):
                    delete_flag = []
                    for i,a_ in enumerate(a.__dict__[k]):
                        ret = self.dfs(a_)
                        if ret == 'delete':
                            delete_flag.append(True)
                            continue
                        if ret is not None:
                            a.__dict__[k][i] = ret
                        delete_flag.append(False)
                    tmp = [a_ for i,a_ in enumerate(a.__dict__[k]) if delete_flag[i] == False]
                    a.__dict__[k] = tmp
                else:
                    ret = self.dfs(a.__dict__[k])
                    if ret is not None:
                        a.__dict__[k] = ret


def convert(code, ex_pjmaps=None):
    ''' Model code converter, example:

    from jittor.utils.pytorch_converter import convert
    pytorch_code = """
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 10, 3)
            self.conv2 = nn.Conv2d(10, 32, 3)
            self.fc = nn.Linear(1200, 100)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    """
    jittor_code = convert(pytorch_code)
    print("## Generate Jittor code:", jittor_code)
    exec(jittor_code)
    model = Model()
    print("## Jittor model:", model)
    '''

    a = ast.parse(code)
    converter = Converter(ex_pjmaps)
    converter.dfs(a)
    a.body.insert(0, ast.parse('import jittor as jt').body[0])
    if 'init' not in converter.import_flag:
        a.body.insert(1, ast.parse('from jittor import init').body[0])
    if 'nn' not in converter.import_flag:
        a.body.insert(2, ast.parse('from jittor import nn').body[0])
    return astunparse.unparse(a)
