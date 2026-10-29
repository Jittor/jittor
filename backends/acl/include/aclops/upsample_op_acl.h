#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class UpsampleNearest2dOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        UpsampleNearest2dOpRunner();
    };

    class UpsampleBicubic2dOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        UpsampleBicubic2dOpRunner();
    };

    class UpsampleBilinear2dOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        UpsampleBilinear2dOpRunner();
    };

    class UpsampleNearest2dBackwardOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        UpsampleNearest2dBackwardOpRunner();
    };
}
