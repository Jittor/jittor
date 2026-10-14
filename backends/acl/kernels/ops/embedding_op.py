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
from .._soc import kernel_available


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
        if not kernel_available("EmbeddingBackward"):
            return None, self._grad_composite(grad_output)
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

    def _grad_composite(self, grad_output):
        """Embedding backward for SoCs whose CANN opp lacks the dense-grad
        kernel (Ascend310P): accumulate grad rows into a zero table through
        scatter_add, which the installed package does provide."""
        num_embeddings = int(self.weight_shape[0])
        flat_dim = 1
        for size in self.weight_shape[1:]:
            flat_dim *= int(size)
        flat_idx = self.indices.reshape(-1)
        rows = grad_output.reshape(-1, flat_dim)
        col_idx = flat_idx.reshape(-1, 1).broadcast(rows.shape).contiguous()
        grad_weight = jt.zeros(
            (num_embeddings, flat_dim), dtype=grad_output.dtype
        ).scatter_add(0, col_idx, rows)
        if self.scale_grad_by_freq:
            counts = jt.zeros(num_embeddings, dtype=grad_output.dtype).scatter_add(
                0, flat_idx, jt.full(flat_idx.shape, 1.0, dtype=grad_output.dtype)
            )
            scale = jt.where(counts > 0, 1.0 / counts, jt.zeros_like(counts))
            grad_weight = grad_weight * scale.reshape(-1, 1)
        if self.padding_idx >= 0:
            keep = jt.array(np.arange(num_embeddings, dtype=np.int64)) != self.padding_idx
            grad_weight = grad_weight * keep.reshape(-1, 1)
        return grad_weight.reshape(self.weight_shape)
