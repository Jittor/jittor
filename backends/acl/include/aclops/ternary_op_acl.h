#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct TernaryOpRunner : public BaseOpRunner
    {
        TernaryOpRunner();

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;
    };
}
