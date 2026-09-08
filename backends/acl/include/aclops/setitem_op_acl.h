#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class InplaceMaskedScatterOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        InplaceMaskedScatterOpRunner();
    };

    class IndexPutImplOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        IndexPutImplOpRunner();
    };
}
