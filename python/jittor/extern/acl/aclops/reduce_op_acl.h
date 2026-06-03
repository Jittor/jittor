#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct ReduceOpRunner : public BaseOpRunner
    {
        int op_idx; // Specific to reduce operations

        ReduceOpRunner();

    protected:
        ReduceAttr *attr;
        aclIntArray *dim;
        bool keepdims;
        // On Ascend/ACL, aclnnAmax/Amin (and other reduce kernels) reject 1-D
        // inputs and return ERROR 161002, silently producing wrong results.
        // We pad a 1-D input to 2-D (1, N) in setupInputDesc and shift the
        // reduce axes accordingly in setupOutputDesc.
        bool input_padded_1d = false;
        // backing storage for the (possibly shifted) reduce axes; must outlive
        // the aclIntArray *dim created from it.
        std::vector<int64_t> shifted_axes_;

        void setupInputDesc() override;
        void setupOutputDesc() override;
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    };
}