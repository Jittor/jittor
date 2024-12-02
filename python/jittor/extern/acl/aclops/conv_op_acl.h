#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct ConvOpRunner : public BaseOpRunner
    {
        ConvOpRunner();

    protected:
        bool use_nchw = true;
        bool is_group_op = false;

        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    };
}