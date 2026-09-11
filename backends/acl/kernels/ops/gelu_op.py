"""Exact (erf) GELU through CANN's own kernel.

Jittor's portable definition is five elementwise nodes, and the ACL fused path
issues one aclnn launch per node, so every activation cost five launches on the
forward and more on the backward. aclnnGelu is the same function in one launch.
"""

import jittor as jt

from ._code import acl_code as gelu_cmd
from ._code import check_acl_float_dtype


class GeluACL:

    def __call__(self, x):
        return self.execute(x)

    def execute(self, x):
        check_acl_float_dtype(x, "gelu")
        attr_code = """
        op.jt_name = "gelu";
        """
        return gelu_cmd(
            "Gelu",
            inputs=[x],
            output_dtypes=[x.dtype],
            output_shapes=[x.shape],
            attr_code=attr_code,
            cuda_grad_src=['''
// aclop
GeluBackwardOpRunner op;
op.add(dout, true);
op.add(in0, true);
op.add(out0, false);
op.jt_name = "gelubackward";
op.run();
'''],
        )[0]
