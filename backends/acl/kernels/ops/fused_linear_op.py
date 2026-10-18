"""Inference-only ACL linear with bias folded into the GEMM accumulator.

vLLM's BF16 ``torch.nn.functional.linear`` uses CANN Addmm on Ascend. A
separate BF16 matmul followed by a BF16 bias addition rounds twice and can
change greedy token choices. This path is opt-in from the Torch shim until
model- and shape-wide correctness is established.
"""

import jittor as jt


_HEADER = r'''
#include "acl_jittor.h"
#include "aclops/aclops.h"
#include "aclnnop/aclnn_addmm.h"
namespace jittor {
class FusedLinearAddmmRunner : public BaseOpRunner {
public:
    // The registered MatMul family provides the launch bookkeeping. The
    // override below calls aclnnAddmm directly; it never reads that entry.
    FusedLinearAddmmRunner() : BaseOpRunner("MatMul") {}
protected:
    void setupInputDesc() override {
        for (size_t i = 0; i < in_.size(); ++i) {
            std::vector<int64_t> shape;
            for (int axis = 0; axis < in_[i]->shape.size(); ++axis)
                shape.push_back(in_[i]->shape[axis]);
            inputShapes.push_back(shape);
            inputTensors.push_back(nullptr);
            // Inputs are bias, activation, weight. Present weight transposed
            // via a descriptor, with no separate transpose kernel or copy.
            auto status = i == 2
                ? CreateFakeTransAclTensor(shape, in_[i]->mem_ptr, in_[i]->size,
                    get_dtype(in_[i]->dtype()), &inputTensors[i], use_nchw, in_[i])
                : CreateAclTensor(shape, in_[i]->mem_ptr, in_[i]->size,
                    get_dtype(in_[i]->dtype()), &inputTensors[i], use_nchw, in_[i]);
            if (status != ACL_SUCCESS)
                LOGf << "Fused linear input descriptor failed: " << status;
        }
    }
    void executeOp(AclOpRegistry::const_iterator &it) override {
        float one = 1.0f;
        aclScalar *beta = aclCreateScalar(&one, ACL_FLOAT);
        aclScalar *alpha = aclCreateScalar(&one, ACL_FLOAT);
        ret = aclnnAddmmGetWorkspaceSize(inputTensors[0], inputTensors[1],
            inputTensors[2], beta, alpha, outputTensors[0], 0,
            &workspaceSize, &executor);
        launch(ret, aclnnAddmm, true);
        aclDestroyScalar(beta);
        aclDestroyScalar(alpha);
    }
};
}
'''

_SOURCE = r'''
FusedLinearAddmmRunner runner;
runner.add(in0, true);
runner.add(in1, true);
runner.add(in2, true);
runner.add(out0, false);
runner.run();
'''


def fused_linear_acl(value, weight, bias):
    """BF16/FP16 inference linear for an ACL device, including a 1-D bias."""
    if value.ndim < 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("fused_linear_acl expects value rank >=2, weight rank 2, bias rank 1")
    if value.shape[-1] != weight.shape[1] or weight.shape[0] != bias.shape[0]:
        raise ValueError("fused_linear_acl incompatible dimensions")
    if str(value.dtype).rsplit(".", 1)[-1] not in ("bfloat16", "float16"):
        raise TypeError("fused_linear_acl supports BF16/FP16 inference")
    if value.dtype != weight.dtype or value.dtype != bias.dtype:
        raise TypeError("fused_linear_acl requires matching dtypes")
    flat = value.reshape((-1, value.shape[-1]))
    output = jt.empty((flat.shape[0], weight.shape[0]), value.dtype)
    result = jt.code(
        inputs=[bias, flat, weight], outputs=[output], backend="acl",
        cuda_header=_HEADER, cuda_src=_SOURCE,
    )[0]
    return result.reshape(list(value.shape[:-1]) + [weight.shape[0]])


__all__ = ["fused_linear_acl"]
