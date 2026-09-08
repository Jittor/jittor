#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class Conv2dOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        Conv2dOpRunner();
    };

    class Conv2dBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;
        void setupOutputDesc() override;

    public:
        Conv2dBackwardOpRunner();
    };
}
