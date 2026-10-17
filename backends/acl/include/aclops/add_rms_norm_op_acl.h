#pragma once
#include "base_op.h"

namespace jittor
{
    class AddRmsNormOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;
    public:
        AddRmsNormOpRunner();
    };
}
