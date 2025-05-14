#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct UnaryOpRunner : public BaseOpRunner
    {
        UnaryOpRunner();

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    };
}