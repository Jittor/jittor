#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class DropoutOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        DropoutOpRunner();
    };

    class DropoutBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        DropoutBackwardOpRunner();
    };

}
