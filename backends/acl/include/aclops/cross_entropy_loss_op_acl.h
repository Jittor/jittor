#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class CrossEntropyLossOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        CrossEntropyLossOpRunner();
    };

    class CrossEntropyLossBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        CrossEntropyLossBackwardOpRunner();
    };
}
