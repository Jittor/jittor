#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class WhereOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        WhereOpRunner();
    };

    class NonzeroOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        NonzeroOpRunner();
    };
}