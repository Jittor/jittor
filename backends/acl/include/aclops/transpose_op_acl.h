#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class TransposeOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        TransposeOpRunner();
    };

    // Fast path for ContiguousOp when the input view is a pure permutation of
    // contiguous storage: wraps the base buffer with dense shape `base_shape`
    // and launches aclnnPermute directly instead of strided aclnnCast.
    class ContiguousPermuteOpRunner : public BaseOpRunner
    {
    protected:
        void setupInputDesc() override;
        void setupOutputDesc() override;
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        int64_t base_shape[16] = {};
        int64_t out_shape[16] = {};
        int64_t perm[16] = {};
        int perm_rank = 0;
        ContiguousPermuteOpRunner();
    };

    // Fast path for non-permutation strided ContiguousOp views (e.g., narrow /
    // unbind / rotate_half slices): launches aclnnInplaceCopy directly on the
    // strided input descriptor instead of strided aclnnCast (~6x faster).
    class ContiguousCopyOpRunner : public BaseOpRunner
    {
    protected:
        void setupInputDesc() override;
        void setupOutputDesc() override;
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        ContiguousCopyOpRunner();
    };

}
