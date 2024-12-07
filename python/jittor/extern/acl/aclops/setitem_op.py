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

def setitem_cmd(name: str,
            inputs: list,
            output_dtypes: list = None,
            output_shapes: list = None,
            attr_code: str = "",
            attr_header: str = "",
            outputs: list = None):
    attr_header = "\nnamespace jittor{" + attr_header + "}\n"

    cuda_header = '''
    #include "acl/aclops/aclops.h"
    '''
    outputs_ = []
    if outputs is not None:
        outputs_ = outputs
    else:
        assert output_dtypes is not None
        assert output_shapes is not None
        assert len(output_dtypes) == len(output_shapes)
        for i in range(len(output_shapes)):
            outputs_.append(jt.empty(output_shapes[i], output_dtypes[i]))
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
    {name}OpRunner op;
    {input_code}
    {output_code}
    {attr_code}
    op.run();""")

def setitem_forward(name: str,
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
    outputs_ = []
    if outputs is not None:
        outputs_ = outputs
    else:
        assert output_dtypes is not None
        assert output_shapes is not None
        assert len(output_dtypes) == len(output_shapes)
        for i in range(len(output_shapes)):
            outputs_.append(jt.empty(output_shapes[i], output_dtypes[i]))
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
    {name}OpRunner op;
    {input_code}
    op.add(out0, false);
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

class SetItemACL(jt.Function):

    def __init__(self):
        self.type_ = 'notype'
        self.value_var = True

    def stride(self, x, dim):
        stride = 1
        for i in range(dim + 1, len(x.shape)):
            stride *= x.shape[i]
        return stride

    def execute(self, x, slices, value):
        self.x_shape = x.shape
        self.input_slice = slices
        if not isinstance(value, jt.Var):
            self.value_var = False
        if isinstance(slices, jt.Var):
            if slices.dtype == "bool":
                slices_len = slices.sum().item()
                if slices_len == 0:
                    return x
                if isinstance(value, int) or isinstance(value, float):
                    value = jt.full((slices_len, ), value)
                assert slices.shape == x.shape, "setitem shape not match"
                assert len(value.shape) == 1, "value shape must be 1D"
                assert value.shape[
                    0] == slices_len, "value shape length must be equal to slices sum"
                self.type_ = 'mask'
                self.value_shape = value.shape
                inputs = [value, slices]
                outputs = [x.clone()]
                attr_code = f"""
                op.jt_name = "inplacemaskedscatter";
                """
                result = setitem_cmd("InplaceMaskedScatter",
                                    inputs=inputs,
                                    outputs=outputs,
                                    attr_code=attr_code)[0]
                return result

        # assert isinstance(value,jt.Var), "value must be jt.Var"
        # self.value_shape = value.shape
        if not isinstance(slices, tuple):
            slices = (slices, )
        slices = list(slices)
        for i, s in enumerate(slices):
            if isinstance(s, int) and s < 0:
                slices[i] = x.shape[i] + s
        slices = tuple(slices)
        slices_list = list(slices)
        #check slices contains slice type
        contains_slice = False
        for s in slices:
            if not isinstance(s, jt.Var) and (isinstance(s, slice)
                                                or s == Ellipsis):
                contains_slice = True
                break
        if not contains_slice:
            indices = []
            value_shape = []
            slices_len = len(slices)
            boardcast_shape = caculate_shape(slices_list[0])
            for ii in range(1, len(slices)):
                dd, boardcast_shape = can_broadcast_and_shape(
                    boardcast_shape, caculate_shape(slices_list[ii]))
                assert dd is True, "can not broadcast"
            value_shape = boardcast_shape
            value_shape += x.shape[slices_len:]
            if value_shape == []:
                value_shape = [1]
            if isinstance(value, int) or isinstance(value, float):
                value = jt.full(value_shape, value)
            self.value_shape = value_shape
            for ii in slices:
                indices.append(jt.Var(ii).int32())
            if isinstance(slices[0], jt.Var) or isinstance(
                    slices[0], int) or isinstance(
                        slices[0], list) or isinstance(slices[0], tuple):
                self.indices = indices
                self.type_ = 'index'
                attr_code = f"""
                op.jt_name = "indexputimpl";
                """
                inputs = [value] + indices
                outputs = [x.clone()]
                result = setitem_cmd("IndexPutImpl",
                                    inputs=inputs,
                                    outputs=outputs,
                                    attr_code=attr_code)[0]
                # result.sync()
                return result
            assert "not support"
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
        self.input_slice = slices
        if len(slices) < x_dim:
            slices += (slice(None, None, None), ) * (x_dim - len(slices))
        sizes = []
        #适配华为奇怪的要求，最后一个维度的step必须是1
        expand_dim = False
        if isinstance(slices[-1], slice):
            if slices[-1].step is not None and slices[-1].step != 1:
                slices = slices + (slice(None, None, None), )
                expand_dim = True

        elif isinstance(slices[-1], int):
            #注意最后一个维度是数字
            slices = slices + (slice(None, None, None), )
            expand_dim = True
            # value = value.unsqueeze(-1)
        else:
            assert False, "not supported"
        x_shape = list(x.shape)
        if expand_dim:
            x_shape.append(1)
            x = x.unsqueeze(-1)
            value = value.unsqueeze(-1)

        squeeze_dims = []
        if isinstance(value, jt.Var):
            for dim, s in enumerate(slices):
                if isinstance(s, int):
                    s = slice(s, s + 1, 1)
                    squeeze_dims.append(dim)

            for dim in squeeze_dims:
                value = value.unsqueeze(dim)

        extra_data = {}
        if len(slices):
            extra_data["a"] = len(slices)
            for dim, s in enumerate(slices):
                if isinstance(s, int):
                    s = slice(s, s + 1, 1)
                if isinstance(s, jt.Var):
                    assert False, "jt.Var not supported"
                start, stop, step = s.indices(x_shape[dim])
                size = (stop - start - 1) // step + 1
                sizes.append(size)
                extra_data[str(dim * 3)] = start
                extra_data[str(dim * 3 + 1)] = stop
                extra_data[str(dim * 3 + 2)] = step
        else:
            extra_data["a"] = -1
            sizes = [1]
            steps = [1]
        if isinstance(value, int) or isinstance(value, float):
            value = jt.full(sizes, value)
        self.type_ = 'slicev2'
        attr_code = """
        op.jt_name = "stridedsliceassignv2";
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
        self.value_shape = value.shape
        inputs = [value]
        outputs = [x.clone()]
        result = setitem_forward("StridedSliceAssignV2",
                                    inputs=inputs,
                                    outputs=outputs,
                                    attr_code=attr_code,
                                    extra_data=extra_data)[0]
        if expand_dim:
            result = result.squeeze(-1)
        # result.sync()
        return result

    def grad(self, grad_output):
        value_grad = None
        if self.value_var:
            value_grad = grad_output[self.input_slice]
        grad_output[self.input_slice] = jt.zeros(self.value_shape)
        return grad_output, None, value_grad