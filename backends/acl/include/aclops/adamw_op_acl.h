#pragma once

#include "base_op.h"
#include "utils.h"

namespace jittor
{
    class AdamWListOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        AdamWListOpRunner();
    };
}
