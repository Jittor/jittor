import os
from jittor_utils import env_or_try_find
import jittor_utils
import ctypes
import glob
import jittor.compiler as compiler
import jittor as jt
import math
import numpy as np

from typing import Union
from collections.abc import Sequence, Iterable

def getitem_cmd(name: str,
            inputs: list,
            output_dtypes: list = None,
            output_shapes: list = None,
            attr_code: str = "",
            attr_header: str = "",
            outputs: list = None):
    #         inputs: list,
    #         output_dtypes: list,
    #         output_shapes: list,
    #         attr_code: str = ""):
    # input_code = ''
    # for i in range(len(inputs)):
    #     input_code += f"op.add(in{i}, true);\n"

    # output_code = ''
    # for i in range(len(output_dtypes)):
    #     output_code += f"op.add(out{i}, false);\n"

    # # read the tmp_file.cpp to the cuda_header
    # with open(
    #         "/home/ma-user/work/zy/JittorHW/python/jittor/extern/acl/tmp_file.cpp",
    #         "r") as f:
    #     cuda_header = f.read()
    # import jittor as jt
    # return jt.code(output_shapes,
    #                output_dtypes,
    #                inputs,
    #                cuda_header=cuda_header,
    #                cuda_src=f"""
    attr_header = "\nnamespace jittor{" + attr_header + "}\n"

    # read the tmp_file.cpp to the cuda_header

    cuda_header = '#include "acl/aclops/aclops.h"'
    import jittor as jt
    outputs_ = []
    if outputs is not None:
        outputs_ = outputs
    else:
        assert output_dtypes is not None
        assert output_shapes is not None
        assert len(output_dtypes) == len(output_shapes)
        # print(f'{name } output_dtypes', output_dtypes)
        # print(f'{name } output_shapes', output_shapes)
        for i in range(len(output_shapes)):
            outputs_.append(jt.empty(output_shapes[i], output_dtypes[i]))
    # print(f'{name } outputs_', outputs_)
    input_code = ''
    for i in range(len(inputs)):
        input_code += f"op.add(in{i}, true);\n"

    output_code = ''
    for i in range(len(outputs_)):
        output_code += f"op.add(out{i}, false);\n"
    return jt.code(outputs=outputs_,
                   inputs=inputs,
                   cuda_header=attr_header + cuda_header,
                   cuda_src=f"""
   
    // aclop
    AclOpRunner op("{name}");
    {input_code}
    {output_code}
    {attr_code}
    op.run();""")

def getitem_forward(name: str,
                    inputs: list,
                    output_dtypes: list = None,
                    output_shapes: list = None,
                    attr_code: str = "",
                    attr_header: str = "",
                    outputs: list = None,
                    extra_data: dict = {}):
    attr_header = "\nnamespace jittor{" + attr_header + "}\n"

    cuda_header = '''
    #include "acl/aclops/aclops.h"
    '''
    import jittor as jt
    outputs_ = []
    if outputs is not None:
        outputs_ = outputs
    else:
        assert output_dtypes is not None
        assert output_shapes is not None
        assert len(output_dtypes) == len(output_shapes)
        # print(f'{name } output_dtypes', output_dtypes)
        # print(f'{name } output_shapes', output_shapes)
        for i in range(len(output_shapes)):
            outputs_.append(jt.empty(output_shapes[i], output_dtypes[i]))
    # print(f'{name } outputs_', outputs_)
    input_code = ''
    for i in range(len(inputs)):
        input_code += f"op.add(in{i}, true);\n"

    output_code = ''
    for i in range(len(outputs_)):
        output_code += f"op.add(out{i}, false);\n"

    return jt.code(outputs=outputs_,
                   inputs=inputs,
                   cuda_header=attr_header + cuda_header,
                   cuda_src=f"""
    // aclop
    AclOpRunner op("{name}");
    {input_code}
    {output_code}
    {attr_code}
    op.run();""",
                   data=extra_data)

def caculate_shape(tensors):
    if isinstance(tensors, jt.Var):
        # tensors = tensors[0]
        return tensors.shape
    elif isinstance(tensors, (int, float)):
        return []
    elif isinstance(tensors, (list, tuple)):
        # return [caculate_shape(tensor) for tensor in tensors]
        sub_shape = caculate_shape(tensors[0])
        return [len(tensors)] + sub_shape
    else:
        assert False, f"not implemented for {type(tensors)}"

def can_broadcast_and_shape(shape1, shape2):
    """
    检查两个张量是否可以广播，并返回广播后的形状。
    
    参数:
    - shape1: 第一个张量的形状（tuple 或 list）
    - shape2: 第二个张量的形状（tuple 或 list）
    
    返回:
    - can_broadcast: 布尔值，表示是否可以广播
    - broadcast_shape: 如果可以广播，返回广播后的形状；否则返回 None
    """
    # 将形状转换为元组，以防输入是列表
    shape1 = tuple(shape1)
    shape2 = tuple(shape2)

    # 使两个形状的长度一致，通过在前面补1
    len1, len2 = len(shape1), len(shape2)
    if len1 < len2:
        shape1 = (1, ) * (len2 - len1) + shape1
    elif len2 < len1:
        shape2 = (1, ) * (len1 - len2) + shape2

    broadcast_shape = []

    # 从最后一维开始检查每一维度
    for dim1, dim2 in zip(shape1, shape2):
        if dim1 == dim2:
            broadcast_shape.append(dim1)
        elif dim1 == 1:
            broadcast_shape.append(dim2)
        elif dim2 == 1:
            broadcast_shape.append(dim1)
        else:
            # 如果在某一维度上不兼容，则不能广播
            return False, None

    return True, tuple(broadcast_shape)

class GetItemACL(jt.Function):

    def __init__(self):
        self.type_ = 'notype'

    def stride(self, x, dim):
        stride = 1
        for i in range(dim + 1, len(x.shape)):
            stride *= x.shape[i]
        return stride

    def execute(self, x, slices, return_x=None):
        if isinstance(slices, jt.Var) and slices.dtype == 'bool':
            # assert False, "not support bool type now"
            #TODO:优化
            assert x.shape == slices.shape, "shape not match"
            output_len = slices.sum().item()
            # output = jt.empty((output_len,),dtype=x.dtype)
            x_len = x.numel()
            output = jt.empty((x_len), dtype=x.dtype)
            outputs = [output]
            inputs = [x, slices]
            # print(inputs,outputs)
            # print(output.shape)
            self.mask = slices
            self.type_ = 'mask'
            attr_code = f"""
            op.jt_name = "maskedselect";
            """
            result = getitem_cmd("MaskedSelect",
                                inputs=inputs,
                                outputs=outputs,
                                attr_code=attr_code)[0]
            result = result[:output_len]
            result.sync()
            return result
        self.x_shape = x.shape
        if not isinstance(slices, tuple):
            slices = (slices, )
        slices = list(slices)
        for i, s in enumerate(slices):
            if isinstance(s, int) and s < 0:
                slices[i] = s + x.shape[i]
        slices = tuple(slices)
        slices_list = list(slices)
        # if not isinstance(slices[0], slice):
        #check slices contains slice type
        contains_slice = False
        for s in slices:
            if not isinstance(s, jt.Var) and (isinstance(s, slice)
                                                or s == Ellipsis):
                contains_slice = True
                break
        if not contains_slice:
            indices = []
            output_shape = []
            slices_len = len(slices)
            boardcast_shape = caculate_shape(slices_list[0])
            for ii in range(1, len(slices)):
                dd, boardcast_shape = can_broadcast_and_shape(
                    boardcast_shape, caculate_shape(slices_list[ii]))
                assert dd is True, "can not broadcast"
            output_shape = boardcast_shape
            output_shape += x.shape[slices_len:]
            if output_shape == []:
                output_shape = [1]
            for ii in slices:
                indices.append(jt.Var(ii).int32())
            if isinstance(slices[0], jt.Var) or isinstance(
                    slices[0], int) or isinstance(
                        slices[0], list) or isinstance(slices[0], tuple):
                self.indices = indices
                inputs = [x] + indices
                attr_code = f"""
                op.jt_name = "index";
                """
                self.type_ = 'index'
                result = getitem_cmd("Index",
                                    inputs=inputs,
                                    output_dtypes=[x.dtype],
                                    output_shapes=[output_shape],
                                    attr_code=attr_code)[0]
                result.sync()
                return result
        assert contains_slice, "slice type error"
        x_dim = len(x.shape)
        slices = list(slices)
        for s in slices:
            if not isinstance(s, jt.Var) and s == Ellipsis:
                slices = slices[:slices.index(s)] + [
                    slice(None, None, None)
                ] * (x_dim - len(slices) + 1) + slices[slices.index(s) +
                                                        1:]
                break
        slices = tuple(slices)

        if len(slices) < x_dim:
            slices += (slice(None, None, None), ) * (x_dim - len(slices))
        inputs = [x]
        sizes = []
        begins = []
        ends = []
        steps = []
        dims = []
        squeeze_dims = []

        extra_data = {}
        if len(slices):
            extra_data["a"] = len(slices)
            for dim, s in enumerate(slices):
                if isinstance(s, int):
                    s = slice(s, s + 1, 1)
                    squeeze_dims.append(dim)
                if isinstance(s, jt.Var):
                    assert False, "jt.Var not supported"
                start, stop, step = s.indices(x.size(dim))
                size = (stop - start - 1) // step + 1
                # stride = self.stride(x, dim) * step
                sizes.append(size)
                extra_data[str(dim * 3)] = start
                extra_data[str(dim * 3 + 1)] = stop
                extra_data[str(dim * 3 + 2)] = step

                steps.append(step)
                begins.append(start)
                ends.append(stop)
                dims.append(dim)
        else:
            extra_data["a"] = -1
            sizes = [1]
            steps = [1]
        self.type_ = 'slicev2'
        # for backward
        self.begins = begins
        self.ends = ends
        self.steps = steps
        self.dims = dims

        self.slices = slices
        attr_code = """
        op.jt_name = "slicev2";
        StrideAttr *attr = new StrideAttr();
        
        int slice_dim = data["a"];
        
        if(slice_dim == -1) {
            attr->begins = {};
            attr->ends = {};
            attr->steps = {1};
            attr->axes = {};
        } else {
            vector<long int> begins;
            vector<long int> ends;
            vector<long int> steps;
            vector<long int> dims;
            for(int dim = 0; dim < slice_dim; dim++) {
                dims.push_back(dim);
                begins.push_back(data[std::to_string(dim*3)]);
                ends.push_back(data[std::to_string(dim*3+1)]);
                steps.push_back(data[std::to_string(dim*3+2)]);
            }
            attr->begins = begins;
            attr->ends = ends;
            attr->steps = steps;
            attr->axes = dims;
        }
        op.op_attr.reset(attr);
        """
        result = getitem_forward("SliceV2",
                                    inputs,
                                    output_dtypes=[x.dtype],
                                    output_shapes=[jt.empty(sizes).shape],
                                    attr_code=attr_code,
                                    extra_data=extra_data)[0]
        self.squeeze_dims = squeeze_dims
        for dim in squeeze_dims[::-1]:
            result = jt.squeeze(result, dim)
        result.sync()
        return result

    def grad(self, grad_output):
        if self.type_ == 'index':
            indices = self.indices
            inputs = [grad_output] + indices
            attr_code = f"""
            op.jt_name = "indexputimplaccumulate";
            """
            outputs = [jt.zeros(self.x_shape, dtype=grad_output.dtype)]
            # breakpoint()
            result = getitem_cmd("IndexPutImplAccumulate",
                                inputs=inputs,
                                outputs=outputs,
                                attr_code=attr_code)[0]
            result.sync()
            return result, None
        elif self.type_ == 'slicev2':
            begins = self.begins
            ends = self.ends
            steps = self.steps
            dims = self.dims
            slices = self.slices
            #注意前向的维数可能会被压缩，所以这里要还原
            for dim in self.squeeze_dims:
                grad_output = jt.unsqueeze(grad_output, dim)
            #适配华为奇怪的要求，最后一个维度的step必须是1
            expand_dim = False
            if isinstance(slices[-1], slice):
                if slices[-1].step is not None and slices[-1].step != 1:
                    slices = slices + (slice(None, None, None), )
                    expand_dim = True
            elif isinstance(slices[-1], int):
                #注意最后一个维度是数字
                slices = list(slices)
                slices[-1] = slice(slices[-1], slices[-1] + 1, 1)
                slices = tuple(slices)
                slices = slices + (slice(None, None, None), )
                expand_dim = True
            else:
                assert False, "not supported"
                # x = x.unsqueeze(-1)
            if expand_dim:
                grad_output = grad_output.unsqueeze(-1)
                self.x_shape = self.x_shape + (1, )
                sizes = []
                begins = []
                ends = []
                steps = []
                dims = []
                for dim, s in enumerate(slices):
                    if isinstance(s, int):
                        s = slice(s, s + 1, 1)
                        # squeeze_dims.append(dim)
                    if isinstance(s, jt.Var):
                        assert False, "jt.Var not supported"
                    start, stop, step = s.indices(self.x_shape[dim])
                    size = (stop - start - 1) // step + 1
                    # stride = self.stride(x, dim) * step
                    sizes.append(size)
                    steps.append(step)
                    begins.append(start)
                    ends.append(stop)
                    dims.append(dim)
                if not sizes:
                    sizes = [1]
                    steps = [1]
            attr_code = f"""
            op.jt_name = "stridedsliceassignv2";
            StrideAttr *attr = new StrideAttr();
            attr->begins = {{ {", ".join(map(str, begins))} }};
            attr->ends = {{ {", ".join(map(str, ends))} }};
            attr->steps = {{ {", ".join(map(str, steps))} }};
            attr->axes = {{ {", ".join(map(str, dims))} }};
            op.op_attr.reset(attr);
            """
            inputs = [grad_output]
            outputs = [jt.zeros(self.x_shape, dtype=grad_output.dtype)]
            result = getitem_cmd("StridedSliceAssignV2",
                                inputs=inputs,
                                outputs=outputs,
                                attr_code=attr_code)[0]
            result.sync()
            if expand_dim:
                result = result.squeeze(-1)
            return result, None
        elif self.type_ == 'mask':
            return self.mask.float()
            pass
        else:
            assert False, f"grad not implemented for {self.type_}"