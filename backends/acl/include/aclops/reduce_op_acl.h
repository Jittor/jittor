#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct ReduceOpRunner : public BaseOpRunner
    {
        int op_idx; // Specific to reduce operations
        bool use_inline_attr = false;
        int64_t inline_axes[16] = {};
        int inline_axes_len = 0;
        bool inline_keepdims = false;

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

        void setupInputDesc() override;
        void setupOutputDesc() override;
        void executeOp(AclOpRegistry::const_iterator &it) override;
    };

    class HistcOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;
    public:
        int64_t bins = 100;
        double min_val = 0.0;
        double max_val = 0.0;
        HistcOpRunner();
    };
}
