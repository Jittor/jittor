#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class DynamicBlockQuantOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        DynamicBlockQuantOpRunner();
    };

    class DynamicMxQuantOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        DynamicMxQuantOpRunner();
    };

    class QuantMatmulOpRunner : public BaseOpRunner
    {
    protected:
        void setupInputDesc() override;
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        QuantMatmulOpRunner();
    };

    class Fp8LinearOpRunner : public BaseOpRunner
    {
    protected:
        void setupInputDesc() override;
        void setupOutputDesc() override;
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        int64_t mode = 0;
        int64_t block_size = 128;
        Var *qx_scratch = nullptr;
        Var *sx_scratch = nullptr;
        Fp8LinearOpRunner();
    };
} // namespace jittor
