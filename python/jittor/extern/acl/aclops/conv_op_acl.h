#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class Conv2dOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        Conv2dOpRunner();
    };

    class Conv2dBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        Conv2dBackwardOpRunner();
    };
}