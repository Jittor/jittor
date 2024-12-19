#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class DropoutOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        DropoutOpRunner();
    };

    class DropoutBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        DropoutBackwardOpRunner();
    };

}