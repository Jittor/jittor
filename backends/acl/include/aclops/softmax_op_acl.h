#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class SoftmaxOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        SoftmaxOpRunner();
    };

    class SoftmaxBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        SoftmaxBackwardOpRunner();
    };

}
