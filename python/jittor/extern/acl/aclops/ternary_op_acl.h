#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct TernaryOpRunner : public BaseOpRunner
    {
        TernaryOpRunner();
        
    protected:
        bool is_group_op = false;
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    };
}