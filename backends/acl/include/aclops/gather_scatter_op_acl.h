#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class GatherOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        GatherOpRunner();
    };

    class IndexCopyOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        IndexCopyOpRunner();
    };

    class ScatterOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        ScatterOpRunner();
    };
}
