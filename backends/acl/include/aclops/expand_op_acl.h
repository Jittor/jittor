#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct ExpandOpRunner : public BaseOpRunner
    {
        ExpandOpRunner();

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;
    };
}
