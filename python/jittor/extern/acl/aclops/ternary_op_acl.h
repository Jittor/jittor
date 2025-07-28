#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct TernaryOpRunner : public BaseOpRunner
    {
        TernaryOpRunner();

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    };
}