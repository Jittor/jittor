#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct ArgReduceOpRunner : public BaseOpRunner
    {
        ArgReduceOpRunner(bool is_max, int64_t dim, bool keepdims);

    protected:
        bool is_max;
        int64_t dim;
        bool keepdims;

        void setupOutputDesc() override;
        void executeOp(AclOpRegistry::const_iterator &it) override;
    };
}
