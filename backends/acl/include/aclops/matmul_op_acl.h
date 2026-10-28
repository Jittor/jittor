#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class MatMulOpRunner : public BaseOpRunner
    {

    protected:
        void setupInputDesc() override;
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        // aclnn cubeMathType: 0 = KEEP_DTYPE (full fp32, matches torch, default),
        // 1 = ALLOW_FP32_DOWN_PRECISION (HF32, faster ~5e-4 off). Set from cuda_src per
        // jt.acl_allow_hf32. Kept in the subclass (NOT BaseOpRunner) to avoid shifting
        // the shared base layout, which ABI-skews other cached ops (e.g. reduce).
        int cube_math_type = 0;
        MatMulOpRunner();
    };

    // The right operand is an explicitly packed FP16 weight snapshot. Its
    // physical NZ dimensions are distinct from the logical transposed matrix.
    class PackedMatMulOpRunner : public BaseOpRunner
    {
    protected:
        void setupInputDesc() override;
        void setupOutputDesc() override;
        void executeOp(AclOpRegistry::const_iterator &it) override;
    public:
        PackedMatMulOpRunner();
    };
}
