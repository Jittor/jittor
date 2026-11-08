from ._attributes import attribute_program, code_program, runner_for_alias
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

from ._code import acl_code as concat_cmd
from ._code import acl_emit, acl_program

#: Training programs by their split sizes and axis. Inference uses the mapped
#: native op above this cache lookup and never builds a SplitWithSize gradient.
_CONCAT_PROGRAMS = {}
_CONCAT_PROGRAM_LIMIT = 4096


def _build_concat_program(split_sizes, dim):
    grad_attr_code = code_program(
        [
            '\n        op.jt_name = "splitwithsize";\n        ',
            attribute_program(
                "SplitWithSize", {"splitSize": list(split_sizes), "dim": dim}, variable="op"
            ),
            "\n        ",
        ]
    )
    grad_outputs = "\n".join(f"op.add(out{i}, false);" for i in range(len(split_sizes)))
    return acl_program(
        "Concat",
        len(split_sizes),
        1,
        attributes={"tensorNum": len(split_sizes), "dim": dim},
        multi_grad_src=code_program(
            [
                "\n            // aclop\n            SplitWithSizeOpRunner op;\n            op.add(dout, true);\n            ",
                grad_outputs,
                "\n            ",
                grad_attr_code,
                "\n            op.run();\n            ",
            ]
        ),
    )


def _validate_concat_inputs(input_tensors, dim):
    """Retain the provider's public errors outside the native success path."""
    if not input_tensors:
        raise ValueError("Concat expects at least one input tensor")
    for tensor in input_tensors:
        if not (-tensor.ndim <= dim < tensor.ndim):
            raise ValueError("dim out of range")
    if dim < 0:
        dim += input_tensors[0].ndim
    for tensor in input_tensors:
        if tensor.dtype != input_tensors[0].dtype:
            raise ValueError("All input tensors must have the same dtype")
        if (tensor.shape[:dim] != input_tensors[0].shape[:dim]
                or tensor.shape[dim + 1:] != input_tensors[0].shape[dim + 1:]):
            raise ValueError("All input tensors must have the same shape")
    return dim


class ConcatACL:
    def __call__(self, input_tensors, dim=0):
        if not isinstance(input_tensors, (list, tuple)):
            raise TypeError("Concat expects a list or tuple of tensors")
        if not isinstance(dim, int):
            raise TypeError("Concat dimension must be an integer")
        return self.execute(input_tensors, dim)

    def execute(self, input_tensors, dim=0):
        if jt.flags.no_grad:
            try:
                return jt.core.ops.mapped_concat(input_tensors, dim)
            except RuntimeError:
                # Native construction checks metadata once. Revisit Python
                # validation only to retain ValueError for malformed requests;
                # a backend/runtime error remains the original RuntimeError.
                _validate_concat_inputs(input_tensors, dim)
                raise
        dim = _validate_concat_inputs(input_tensors, dim)
        self.input = input_tensors
        self.dim = dim
        split_sizes = tuple(int(tensor.shape[dim]) for tensor in input_tensors)
        key = (split_sizes, dim)
        program = _CONCAT_PROGRAMS.get(key)
        if program is None:
            program = _build_concat_program(split_sizes, dim)
            if len(_CONCAT_PROGRAMS) >= _CONCAT_PROGRAM_LIMIT:
                _CONCAT_PROGRAMS.clear()
            _CONCAT_PROGRAMS[key] = program
        return acl_emit(
            program,
            input_tensors,
            [input_tensors[0].dtype],
            [self.calculate_output_shape(input_tensors, dim)],
        )[0]

    def calculate_output_shape(self, input_tensors, axis):
        shape = list(input_tensors[0].shape)
        for tensor in input_tensors[1:]:
            shape[axis] += tensor.shape[axis]
        return tuple(shape)


class SplitWithSizeACL:
    def __call__(self, x, split_sizes, dim=0):
        return self.execute(x, split_sizes, dim)

    def execute(self, x, split_sizes, dim=0):
        output_shapes = []
        for size in split_sizes:
            shape = list(x.shape)
            shape[dim] = size
            output_shapes.append(shape)
        return tuple(
            concat_cmd(
                "SplitWithSize",
                [x],
                output_dtypes=[x.dtype] * len(split_sizes),
                output_shapes=output_shapes,
                attributes={"splitSize": list(split_sizes), "dim": dim},
            )
        )
