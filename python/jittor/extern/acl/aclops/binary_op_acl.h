#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct BinaryOpRunner : public BaseOpRunner
    {
        BinaryOpRunner();

    protected:
        bool is_group_op = true;
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    };
}