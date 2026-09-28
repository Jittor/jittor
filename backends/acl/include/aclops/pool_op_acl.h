#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class AdaptiveAvgPool2dOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;
    public:
        AdaptiveAvgPool2dOpRunner();
    };

    class AdaptiveAvgPool2dBackwardOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;
    public:
        AdaptiveAvgPool2dBackwardOpRunner();
    };

    class MaxpoolOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        MaxpoolOpRunner();
    };

    class AvgpoolOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        AvgpoolOpRunner();
    };

    class MaxpoolBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        MaxpoolBackwardOpRunner();
    };

    class AvgpoolBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        AvgpoolBackwardOpRunner();
    };
}
