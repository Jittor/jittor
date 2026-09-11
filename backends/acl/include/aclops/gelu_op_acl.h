#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class GeluOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        GeluOpRunner();
    };

    class GeluBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        GeluBackwardOpRunner();
    };
}
