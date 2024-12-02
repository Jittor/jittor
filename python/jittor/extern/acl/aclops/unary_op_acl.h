#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct UnaryOpRunner : public BaseOpRunner
    {
        UnaryOpRunner();

    protected:
        bool is_group_op = true;
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    };
}