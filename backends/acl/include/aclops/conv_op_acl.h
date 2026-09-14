#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class Conv2dOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        // aclnn cubeMathType: 0 = KEEP_DTYPE (full fp32), 1 = ALLOW_FP32_DOWN_PRECISION
        // (HF32). Set from cuda_src per jt.acl_allow_hf32, the same knob matmul
        // and batched matmul already honour; torch_npu defaults its convolutions
        // to HF32, so without this the two frameworks ran different math.
        int cube_math_type = 0;
        Conv2dOpRunner();
    };

    class Conv2dBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;
        void setupOutputDesc() override;

    public:
        // aclnn cubeMathType: 0 = KEEP_DTYPE (full fp32), 1 = ALLOW_FP32_DOWN_PRECISION
        // (HF32). Set from cuda_src per jt.acl_allow_hf32, the same knob matmul
        // and batched matmul already honour; torch_npu defaults its convolutions
        // to HF32, so without this the two frameworks ran different math.
        int cube_math_type = 0;
        Conv2dBackwardOpRunner();
    };
}
