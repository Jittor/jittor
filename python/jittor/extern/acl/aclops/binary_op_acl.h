#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct BinaryOpRunner : public BaseOpRunner
    {
        BinaryOpRunner();

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    };
}