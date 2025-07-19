#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct ExpandOpRunner : public BaseOpRunner
    {
        ExpandOpRunner();

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    };
}