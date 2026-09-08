#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class StackOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        StackOpRunner();
    };

}
