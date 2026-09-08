#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class SiLUOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        SiLUOpRunner();
    };

    class SiLUBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        SiLUBackwardOpRunner();
    };

    class SwishOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        SwishOpRunner();
    };

    class SwishBackwardOpRunner : public BaseOpRunner
    {

    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        SwishBackwardOpRunner();
    };

    class SwiGluOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        int64_t dim = -1;
        SwiGluOpRunner();
    };

}
