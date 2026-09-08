#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class CumsumOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        CumsumOpRunner();
    };

}
