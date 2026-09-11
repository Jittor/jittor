// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include "aclnn/aclnn.h"
#include <acl/acl.h>
#include "acl_runtime.h"
#include "acl_workspace.h"
#include "acl_op_registry.h"

#define aclstream acl_current_stream()
#define workspaceAddr acl_workspace_address()

std::string acl_error_to_string(aclError error);

namespace jittor
{

    inline aclError acl_jittor_get_device_count(int *count)
    {
        uint32_t acl_count = 0;
        aclError status = aclrtGetDeviceCount(&acl_count);
        *count = static_cast<int>(acl_count);
        return status;
    }

    struct AclOpAttr
    {
        virtual ~AclOpAttr() {}
    };

    struct AdamWAttr : AclOpAttr
    {
        int64_t tensorCount;
        float lr;
        float beta1;
        float beta2;
        float weightDecay;
        float eps;
    };

    struct FusedSgdAttr : AclOpAttr
    {
        int64_t tensorCount;
        float lr;
        float momentum;
        float weightDecay;
        float dampening;
        bool nesterov;
        bool maximize;
        bool isFirstStep;
    };

    struct ConvAttr : AclOpAttr
    {
        vector<int64_t> convStrides;
        vector<int64_t> convPads;
        vector<int64_t> convOutPads;
        vector<int64_t> convDilations;
        bool convWithBias;
        bool is_transposed;
        int64_t group;

        // 析构函数
        ~ConvAttr()
        {
            convStrides.clear();
            convPads.clear();
            convOutPads.clear();
            convDilations.clear();
        }
    };

    struct ReduceAttr : AclOpAttr
    {
        vector<int64_t> axes;
        // for proddim
        int64_t prod_dim;
        bool keepdims;

        ~ReduceAttr()
        {
            axes.clear();
        }
    };

    struct RandomAttr : AclOpAttr
    {
        int64_t seed, offset;

        ~RandomAttr()
        {
        }
    };

    struct TriuAttr : AclOpAttr
    {
        int64_t diagonal;

        ~TriuAttr()
        {
        }
    };

    struct PoolAttr : AclOpAttr
    {
        vector<int64_t> kernel_size;
        vector<int64_t> poolStrides;
        vector<int64_t> poolPads;
        vector<int64_t> poolDilations;
        bool poolCeil;
        bool countIncludePad;

        // divisorOverride(const int64_t，计算输入): 表示取平均的除数。数据类型支持INT64。divisorOverride配置为默认值0时表示功能不使能。
        // https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/apiref/appdevgapi/context/aclnnAvgPool2d.md
        int64_t divisorOverride = 0;

        // cubeMathType(int8_t，计算输入): host侧的整型，判断Cube单元应该使用哪种计算逻辑进行运算，数据类型支持INT8。对于无特殊说明的数据类型，均保持原始输入数据类型计算。支持的枚举值如下：
        //    0:KEEP_DTYPE，保持输入的数据类型进行计算。当输入是FLOAT，Atlas 训练系列产品和Atlas 推理系列产品（Ascend 310P处理器）暂不支持，取0时会报错。
        //    1:ALLOW_FP32_DOWN_PRECISION，允许将输入数据降精度计算。当输入是FLOAT，Atlas 训练系列产品和Atlas 推理系列产品（Ascend 310P处理器）允许转换为FLOAT16计算。
        //    2:USE_FP16，允许转换为数据类型FLOAT16进行计算。当输入数据类型是FLOAT，转换为FLOAT16计算。
        //    3:USE_HF32，允许转换为数据类型HFLOAT32计算。当输入是FLOAT，Atlas 训练系列产品、Atlas 推理系列产品（Ascend 310P处理器）和Atlas A2训练系列产品/Atlas 800I A2推理产品暂不支持，取3时会报错。
        // https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/apiref/appdevgapi/context/aclnnAvgPool2d.md
        int8_t cubeMathType = 0;

        // 析构函数
        ~PoolAttr()
        {
            kernel_size.clear();
            poolStrides.clear();
            poolPads.clear();
            poolDilations.clear();
        }
    };

    struct UpsampleNearest2dAttr : AclOpAttr
    {
        vector<int64_t> outputSize;
        vector<int64_t> inputSize;

        ~UpsampleNearest2dAttr()
        {
            outputSize.clear();
            inputSize.clear();
        }
    };

    struct ConcatAttr : AclOpAttr
    {
        int64_t tensorNum;
        int64_t dim;

        ~ConcatAttr()
        {
        }
    };

    struct GatherAttr : AclOpAttr
    {
        int64_t dim;

        ~GatherAttr()
        {
        }
    };

    struct ScatterAttr : AclOpAttr
    {
        int64_t axis;
        int64_t reduction;

        ~ScatterAttr()
        {
        }
    };

    struct StrideAttr : AclOpAttr
    {
        vector<int64_t> begins;
        vector<int64_t> ends;
        vector<int64_t> steps;
        vector<int64_t> axes;
        ~StrideAttr()
        {
            begins.clear();
            ends.clear();
            steps.clear();
            axes.clear();
        }
    };

    struct RangeAttr : AclOpAttr
    {
        int64_t start;
        int64_t end;
        int64_t step;

        ~RangeAttr()
        {
        }
    };

    struct LeakyReluAttr : AclOpAttr
    {
        float negativeSlope;
        bool selfIsResult;

        ~LeakyReluAttr()
        {
        }
    };

    struct DropoutAttr : AclOpAttr
    {
        float p;
        bool train;
        int64_t seed;
        int64_t offset;
        float scale;

        ~DropoutAttr()
        {
        }
    };

    struct EmbeddingAttr : AclOpAttr
    {
        int64_t numEmbeddings;
        // int64_t embeddingDim;
        int64_t paddingIdx;
        bool scaleGradByFreq;
        // bool sparse;
        // bool isSparse;
        // bool isDense;

        ~EmbeddingAttr()
        {
        }
    };

    struct SplitWithSizeAttr : AclOpAttr
    {
        vector<int64_t> splitSize;
        int64_t dim;
        ~SplitWithSizeAttr()
        {
            splitSize.clear();
        }
    };

    struct SoftmaxAttr : AclOpAttr
    {
        int64_t dim;
        ~SoftmaxAttr()
        {
        }
    };

    struct BatchNormAttr : AclOpAttr
    {
        bool is_train;
        float momentum;
        float eps;
        ~BatchNormAttr()
        {
        }
    };

    struct LayerNormAttr : AclOpAttr
    {
        float eps;
        vector<int64_t> normalizedShape;
        int64_t size;
        ~LayerNormAttr()
        {
            normalizedShape.clear();
        }
    };

    struct GroupNormAttr : AclOpAttr
    {
        int64_t batch;
        int64_t channels;
        int64_t spatialSize;
        int64_t groups;
        float eps;
        ~GroupNormAttr()
        {
        }
    };

    struct RmsNormAttr : AclOpAttr
    {
        float eps;
        ~RmsNormAttr()
        {
        }
    };

    struct FlashAttentionAttr : AclOpAttr
    {
        vector<int64_t> prefix;
        vector<int64_t> qStartIdx;
        vector<int64_t> kvStartIdx;
        float scale;
        float keepProb;
        int64_t preToken;
        int64_t nextToken;
        int64_t headNum;
        string inputLayout;
        int64_t innerPrecise;
        int64_t sparseMode;
        int64_t psetype;
        bool hasRealshift;
        bool hasDropmask;
        bool hasPaddingmask;
        bool hasAttentmask;

        ~FlashAttentionAttr()
        {
            prefix.clear();
            qStartIdx.clear();
            kvStartIdx.clear();
        }
    };

    struct IncreFlashAttentionAttr : AclOpAttr
    {
        double scale;
        int64_t headNum;
        int64_t keyValueHeadNum;
        string inputLayout;
        int64_t innerPrecise;
        int64_t blockSize = 0;
        bool hasBlockTable = false;
        vector<int64_t> actualSeqLengths;
    };

    struct KVCacheMemcpyAttr : AclOpAttr
    {
        int64_t blockSize;
        vector<int64_t> slots;
    };

    struct NanToNumAttr : AclOpAttr
    {
        float nan;
        float posinf;
        float neginf;
        ~NanToNumAttr()
        {
        }
    };
}
