#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    class DirectConv2dOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        int64_t stride_h = 1;
        int64_t stride_w = 1;
        int64_t pad_h = 0;
        int64_t pad_w = 0;
        int64_t dil_h = 1;
        int64_t dil_w = 1;
        int64_t groups = 1;
        int cube_math_type = 0;
        DirectConv2dOpRunner();
    };

    class DirectConv2dBackwardOpRunner : public BaseOpRunner
    {
    protected:
        void setupOutputDesc() override;
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        int64_t stride_h = 1;
        int64_t stride_w = 1;
        int64_t pad_h = 0;
        int64_t pad_w = 0;
        int64_t dil_h = 1;
        int64_t dil_w = 1;
        int64_t groups = 1;
        int cube_math_type = 0;
        DirectConv2dBackwardOpRunner();
    };

    class DirectBatchNormOpRunner : public BaseOpRunner
    {
    protected:
        void setupInputDesc() override;
        void setupOutputDesc() override;
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        bool is_train = false;
        double momentum = 0.1;
        double eps = 1e-5;
        DirectBatchNormOpRunner();
    };

    class DirectBatchNormBackwardOpRunner : public BaseOpRunner
    {
    protected:
        void setupInputDesc() override;
        void setupOutputDesc() override;
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        bool is_train = true;
        double eps = 1e-5;
        DirectBatchNormBackwardOpRunner();
    };

    class DirectLayerNormOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        int normalized_ndim = 1;
        double eps = 1e-5;
        DirectLayerNormOpRunner();
    };

    class DirectLayerNormBackwardOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        int normalized_ndim = 1;
        DirectLayerNormBackwardOpRunner();
    };

    class DirectUnaryGradOpRunner : public BaseOpRunner
    {
    protected:
        void executeOp(AclOpRegistry::const_iterator &it) override;

    public:
        int kind = 8;
        int64_t approximate = 0;
        DirectUnaryGradOpRunner();
    };
}
