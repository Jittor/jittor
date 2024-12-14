#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class GatherOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        GatherOpRunner();
    };


    class ScatterOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        ScatterOpRunner();
    };
}