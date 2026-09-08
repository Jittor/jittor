#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class FloorOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        FloorOpRunner();
    };
}
