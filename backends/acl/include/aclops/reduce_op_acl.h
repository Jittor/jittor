#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct ReduceOpRunner : public BaseOpRunner
    {
        int op_idx; // Specific to reduce operations

        ReduceOpRunner();
        ~ReduceOpRunner() override;

    protected:
        ReduceAttr *attr;
        aclIntArray *dim = nullptr;
        bool keepdims;
        // On Ascend/ACL, aclnnAmax/Amin (and other reduce kernels) reject 1-D
        // inputs and return ERROR 161002, silently producing wrong results.
        // We pad a 1-D input to 2-D (1, N) in setupInputDesc and shift the
        // reduce axes accordingly in setupOutputDesc.
        bool input_padded_1d = false;
        // backing storage for the (possibly shifted) reduce axes; must outlive
        // the aclIntArray *dim created from it.
        std::vector<int64_t> shifted_axes_;
        // aclnnAmax/Amin require their output dtype to equal the input dtype.
        // A fused AMP graph may legally request a narrower ReduceOp output,
        // so those two kernels reduce into this temporary and cast afterwards.
        aclTensor *extrema_output_tensor_ = nullptr;
        void *extrema_output_buffer_ = nullptr;

        void setupInputDesc() override;
        void setupOutputDesc() override;
        void executeOp(AclOpRegistry::const_iterator &it) override;
    };
}
