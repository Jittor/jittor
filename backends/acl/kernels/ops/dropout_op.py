from ._code import code_with_attributes
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


from ._code import acl_code as dropout_cmd


class DropoutACL(jt.Function):
    def __init__(self):
        super(DropoutACL, self).__init__()

    def execute(self, x, p=0.5, is_train=False):
        self.input = x
        self.p = p
        self.is_train = bool(is_train)
        num_elements = x.numel()
        aligned_elements = (num_elements + 127) // 128 * 128
        mask_shape = (aligned_elements // 8,)
        attributes = {
                "p": p,
                "train": bool(is_train),
                "seed": 0,
                "offset": 0,
            }
        result = dropout_cmd(
            "Dropout",
            [x],
            output_dtypes=[x.dtype, "uint8"],
            output_shapes=[x.shape, mask_shape],
            attributes=attributes,
        )
        self.maskout = result[1]
        return result[0]

    def grad(self, grad_output):
        # Inverted dropout divides surviving activations by (1 - p) in the
        # forward pass (confirmed empirically: a tensor of ones comes back
        # at 1/(1-p) on the elements the mask kept), so the chain rule needs
        # the same factor here -- grad_input = grad_output * mask / (1 - p).
        # A hardcoded scale=1.0 silently dropped that factor, under-scaling
        # every gradient that flows back through a dropout call whenever it
        # is actually active (is_train and p > 0).
        scale = 1.0 / (1.0 - self.p) if (self.is_train and self.p > 0.0) else 1.0
        grad_input = dropout_cmd(
            "DropoutBackward",
            [grad_output, self.maskout],
            output_dtypes=[grad_output.dtype],
            output_shapes=[grad_output.shape],
            attributes={"scale": scale},
        )[0]
        return grad_input
