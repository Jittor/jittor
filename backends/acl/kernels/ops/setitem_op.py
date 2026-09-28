from ._code import code_with_attributes
from ._attributes import attribute_program, code_program, runner_for_alias
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
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


from ._code import acl_code as setitem_cmd


from ._code import acl_code as setitem_forward


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
        raise TypeError("setitem does not support shape metadata for {}".format(type(tensors)))


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
        shape1 = (1,) * (len2 - len1) + shape1
    elif len2 < len1:
        shape2 = (1,) * (len1 - len2) + shape2

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
        self.type_ = "notype"
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
            if _jittor_dtype_name(slices.dtype) == "bool":
                scalar_value = isinstance(value, (int, float)) or (
                    isinstance(value, jt.Var) and value.ndim == 0)
                self.mask_scalar = scalar_value
                if isinstance(value, int) or isinstance(value, float):
                    # ACL masked-scatter consumes only as many source elements
                    # as the mask selects. Avoid reducing the bool mask here:
                    # bool reductions are not reliable on ACL, and a wrong zero
                    # count would silently turn a real assignment into a no-op.
                    value = jt.full((x.numel(),), value, dtype=x.dtype)
                elif scalar_value:
                    value = value.broadcast((x.numel(),)).contiguous()
                if slices.shape != x.shape:
                    raise ValueError("setitem mask shape must match input shape")
                if len(value.shape) != 1:
                    raise ValueError("setitem mask value must be 1D")
                if self.value_var and not scalar_value:
                    slices_len = slices.int32().sum().item()
                    if value.shape[0] != slices_len:
                        raise ValueError("setitem value length must equal selected elements")
                self.type_ = "mask"
                self.value_shape = () if scalar_value else value.shape
                # base x is an explicit input so its data is materialized before
                # the in-place masked-scatter (the runner copies base->out first).
                inputs = [x.contiguous(), slices, value]
                outputs = [jt.empty(x.shape, x.dtype)]
                attr_code = f"""
                op.jt_name = "inplacemaskedscatter";
                """
                result = setitem_cmd(
                    "InplaceMaskedScatter", inputs=inputs, outputs=outputs, attr_code=attr_code
                )[0]
                return result

        # assert isinstance(value,jt.Var), "value must be jt.Var"
        # self.value_shape = value.shape
        if not isinstance(slices, tuple):
            slices = (slices,)
        slices = list(slices)
        for i, s in enumerate(slices):
            if isinstance(s, int) and s < 0:
                slices[i] = x.shape[i] + s
        slices = tuple(slices)
        slices_list = list(slices)
        # check slices contains slice type
        contains_slice = False
        for s in slices:
            if not isinstance(s, jt.Var) and (isinstance(s, slice) or s == Ellipsis):
                contains_slice = True
                break
        if all(isinstance(s, (int, slice)) or s is Ellipsis for s in slices):
            contains_slice = True
        if not contains_slice:
            indices = []
            value_shape = []
            slices_len = len(slices)
            boardcast_shape = caculate_shape(slices_list[0])
            for ii in range(1, len(slices)):
                dd, boardcast_shape = can_broadcast_and_shape(
                    boardcast_shape, caculate_shape(slices_list[ii])
                )
                if dd is not True:
                    raise ValueError("setitem indices cannot be broadcast")
            value_shape = boardcast_shape
            value_shape += x.shape[slices_len:]
            if value_shape == []:
                value_shape = [1]
            if isinstance(value, int) or isinstance(value, float):
                value = jt.full(value_shape, value)
            self.value_shape = value_shape
            for ii in slices:
                indices.append(jt.Var(ii).int32())
            if (
                isinstance(slices[0], jt.Var)
                or isinstance(slices[0], int)
                or isinstance(slices[0], list)
                or isinstance(slices[0], tuple)
            ):
                self.indices = indices
                self.type_ = "index"
                attr_code = f"""
                op.jt_name = "indexputimpl";
                """
                inputs = [value] + indices
                outputs = [x.clone()]
                result = setitem_cmd(
                    "IndexPutImpl", inputs=inputs, outputs=outputs, attr_code=attr_code
                )[0]
                # result.sync()
                return result
            raise NotImplementedError("ACL setitem index form is not supported")
        if not contains_slice:
            raise TypeError("setitem expects at least one slice index")
        x_dim = len(x.shape)
        slices = list(slices)
        for s in slices:
            if not isinstance(s, jt.Var) and s == Ellipsis:
                slices = (
                    slices[: slices.index(s)]
                    + [slice(None, None, None)] * (x_dim - len(slices) + 1)
                    + slices[slices.index(s) + 1 :]
                )
                break
        slices = tuple(slices)
        self.input_slice = slices
        if len(slices) < x_dim:
            slices += (slice(None, None, None),) * (x_dim - len(slices))
        sizes = []
        # 适配华为奇怪的要求，最后一个维度的step必须是1
        expand_dim = False
        if isinstance(slices[-1], slice):
            if slices[-1].step is not None and slices[-1].step != 1:
                slices = slices + (slice(None, None, None),)
                expand_dim = True

        elif isinstance(slices[-1], int):
            # 注意最后一个维度是数字
            slices = slices + (slice(None, None, None),)
            expand_dim = True
            # value = value.unsqueeze(-1)
        else:
            raise NotImplementedError("ACL setitem slice form is not supported")
        x_shape = list(x.shape)
        if expand_dim:
            x_shape.append(1)
            x = x.unsqueeze(-1)

        squeeze_dims = [dim for dim, s in enumerate(slices) if isinstance(s, int)]

        begins, ends, steps, dims = [], [], [], []
        if len(slices):
            for dim, s in enumerate(slices):
                if isinstance(s, int):
                    s = slice(s, s + 1, 1)
                if isinstance(s, jt.Var):
                    raise NotImplementedError("ACL setitem does not support Var slice indices")
                start, stop, step = s.indices(x_shape[dim])
                size = (stop - start - 1) // step + 1
                sizes.append(size)
                begins.append(start)
                ends.append(stop)
                steps.append(step)
                dims.append(dim)
        else:
            sizes = [1]
            steps = [1]
        if not isinstance(value, jt.Var):
            value = jt.array(value, dtype=x.dtype)
        self.value_shape = tuple(value.shape)
        # Broadcast in the public (integer-index axes removed) slice shape,
        # then restore singleton axes required by the CANN assignment ABI.
        logical_shape = tuple(size for dim, size in enumerate(sizes)
                              if dim not in squeeze_dims and not (expand_dim and dim == len(sizes) - 1))
        value = value.broadcast(logical_shape).reshape(sizes).contiguous()
        self.type_ = "slicev2"
        attr_code = attribute_program(
            "StridedSliceAssignV2",
            {
                "begins": begins,
                "ends": ends,
                "steps": steps,
                "axes": dims,
            },
        )
        # A partial writer must depend on the entire initialized base. A
        # write-only clone output loses this dependency and may be zero-strided.
        inputs = [value, x.contiguous()]
        outputs = [jt.empty(x.shape, x.dtype)]
        result = setitem_forward(
            "StridedSliceAssignV2", inputs=inputs, outputs=outputs, attr_code=attr_code
        )[0]
        if expand_dim:
            result = result.squeeze(-1)
        # result.sync()
        return result

    def grad(self, grad_output):
        if self.type_ == "mask":
            mask = self.input_slice
            value_grad = None
            if self.value_var:
                if self.mask_scalar:
                    # A masked reduction also handles an empty selection
                    # without launching a zero-length masked-select operation.
                    value_grad = jt.where(mask, grad_output, 0).sum().reshape(())
                else:
                    value_grad = grad_output[mask]
            return jt.where(mask, 0, grad_output), None, value_grad
        value_grad = None
        if self.value_var:
            value_grad = grad_output[self.input_slice]
            if self.type_ == "slicev2":
                padded = (1,) * (value_grad.ndim - len(self.value_shape)) + self.value_shape
                axes = tuple(i for i, (actual, target) in enumerate(zip(value_grad.shape, padded))
                             if target == 1 and actual != 1)
                if axes:
                    value_grad = value_grad.sum(axes, keepdims=True)
                value_grad = value_grad.reshape(self.value_shape)
        base_grad = grad_output.clone()
        base_grad[self.input_slice] = 0
        return base_grad, None, value_grad
