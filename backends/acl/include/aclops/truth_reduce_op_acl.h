#pragma once
#include "base_op.h"
#include "utils.h"

namespace jittor
{
    struct TruthReduceOpRunner : public BaseOpRunner
    {
        explicit TruthReduceOpRunner(bool reduce_all);

    protected:
        bool reduce_all;
        ReduceAttr *attr;

        void setupOutputDesc() override;
        void executeOp(
            AclOpRegistry::const_iterator &it) override;
    };
}
