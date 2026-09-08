#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct UnaryOpRunner : public BaseOpRunner
    {
        UnaryOpRunner();

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;
    };
}
