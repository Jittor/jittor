#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class MaxpoolOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        MaxpoolOpRunner();
    };

    class AvgpoolOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        AvgpoolOpRunner();
    };

    class MaxpoolBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        MaxpoolBackwardOpRunner();
    };

    class AvgpoolBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;

    public:
        AvgpoolBackwardOpRunner();
    };
}