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

from ._code import acl_code as embedding_cmd


class EmbeddingACL(jt.Function):
    def __init__(self, padding_idx=None, scale_grad_by_freq=False):
        super(EmbeddingACL, self).__init__()
        self.padding_idx = -1 if padding_idx is None else int(padding_idx)
        self.scale_grad_by_freq = bool(scale_grad_by_freq)

    def execute(
        self,
        indices,
        weight,
    ):
        inputs = [weight, indices]
        self.indices = indices
        self.weight_shape = weight.shape
        output_shape = list(indices.shape) + list(weight.shape[1:])
        outputs = [jt.empty(output_shape, weight.dtype)]
        attr_code = f"""
        op.jt_name = "embedding";
        """
        result = embedding_cmd("Embedding", inputs=inputs, outputs=outputs, attr_code=attr_code)[0]
        return result

    def grad(self, grad_output):
        inputs = [grad_output, self.indices]
        outputs = [jt.empty(self.weight_shape, grad_output.dtype)]
        attr_code = code_program(
            [
                '\n        op.jt_name = "embeddingbackward";\n        ',
                attribute_program(
                    "EmbeddingBackward",
                    {
                        "numEmbeddings": self.weight_shape[0],
                        "paddingIdx": self.padding_idx,
                        "scaleGradByFreq": bool(self.scale_grad_by_freq),
                    },
                    variable="op",
                ),
                "\n        ",
            ]
        )
        grad_weight = embedding_cmd(
            "EmbeddingBackward", inputs=inputs, outputs=outputs, attr_code=attr_code
        )[0]
        return None, grad_weight
