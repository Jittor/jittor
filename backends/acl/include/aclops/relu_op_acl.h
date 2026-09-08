#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class LeakyReLUOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        LeakyReLUOpRunner();
    };

    class LeakyReLUBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        LeakyReLUBackwardOpRunner();
    };

}
