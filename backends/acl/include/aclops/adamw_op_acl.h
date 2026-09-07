#pragma once

#include "base_op.h"
#include "utils.h"

namespace jittor
{
    class AdamWListOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        AdamWListOpRunner();
    };
}
