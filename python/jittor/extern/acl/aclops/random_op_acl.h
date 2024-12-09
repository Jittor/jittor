#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class RandomOpRunner : public BaseOpRunner
    {

    protected:
        string name;
        void executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it) override;
    public:
        RandomOpRunner();
        RandomOpRunner(const string &name);
    };
}