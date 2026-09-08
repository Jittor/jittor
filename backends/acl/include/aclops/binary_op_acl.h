#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct BinaryOpRunner : public BaseOpRunner
    {
        BinaryOpRunner();

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;
    };
}
