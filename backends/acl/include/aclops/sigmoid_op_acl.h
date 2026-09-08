#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class SigmoidOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        SigmoidOpRunner();
    };

    class SigmoidBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        SigmoidBackwardOpRunner();
    };

}
