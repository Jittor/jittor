#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>

namespace jittor
{

    static std::unordered_map<string,int>op_idx_map = 
    {
        {"Add",1},
        {"Sub",2},
        {"Expand",3},
        {"Cast",4},
        {"Unary",5},
        {"Binary",6},
        {"BatchMatMul",7},
        {"MatMul",8},
        {"ReduceSum",9},
        {"ReduceMean",10},
        {"ReduceMax",11},
        {"ReduceMin",12},
        {"RandomUniform",13},
        {"RandomNormal",14},
        {"Nonzero",15},
        {"Select",16},
        {"Where",17},
        {"Triu",18},
        {"Transpose",19},
        {"Conv2d",20},
        {"Conv2dBackward",21},
        {"Maxpool",22},
        {"MaxpoolBackward",23},
        {"Avgpool",24},
        {"AvgpoolBackward",25},
        {"Flip",26},
        {"Concat",27},
        {"Gather",28},
        {"Cumsum",29},
        {"Scatter",30},
        {"Floor",31},
        {"Index",32},
        {"SliceV2",33},
        {"IndexPutImpl",34},
        {"IndexPutImplAccumulate",35},
        {"StridedSliceAssignV2",36},
        {"Range",37},
        {"LeakyReLU",38},
        {"LeakyReLUBackward",39},
        {"Dropout",40},
        {"DropoutBackward",41},
        {"SiLU",42},
        {"SiLUBackward",43},
        {"Sigmoid",44},
        {"SigmoidBackward",45},
        {"Embedding",46},
        {"EmbeddingBackward",47},
        {"InplaceMaskedScatter",48},
        {"MaskedSelect",49},
        {"SplitWithSize",50},
        {"FlashAttention",51},
        {"FlashAttentionBackward",52} 
    };
    int CreateAclTensor(const std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                        aclDataType dataType, aclTensor **tensor, bool use_nchw = false)
    {
        // 计算连续tensor的strides
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--)
        {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        if (shape.size() == 0)
            strides = {};
        // 调用aclCreateTensor接口创建aclTensor
        if (use_nchw)
            *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                      shape.data(), shape.size(), deviceAddr);
        else
            *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                      shape.data(), shape.size(), deviceAddr);
        return 0;
    }

    int CreateFakeTransAclTensor(std::vector<int64_t> &shape, void *deviceAddr, int64_t size,
                                 aclDataType dataType, aclTensor **tensor, bool use_nchw = false)
    {
        // 计算连续tensor的strides
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--)
        {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        if (shape.size() == 0)
            strides = {};
        int n = shape.size();
        if (n > 1)
        {
            std::swap(shape[n - 1], shape[n - 2]);
            std::swap(strides[n - 1], strides[n - 2]);
        }
        // 调用aclCreateTensor接口创建aclTensor
        if (use_nchw)
            *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                      shape.data(), shape.size(), deviceAddr);
        else
            *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                      shape.data(), shape.size(), deviceAddr);
        return 0;
    }

    struct AclOpRunner
    {
        string name;
        string jt_name;
        vector<Var *> in_;
        vector<Var *> out_;
        std::unique_ptr<AclOpAttr> op_attr;

        AclOpRunner(const string &name) : name(name)
        {
        }

        ~AclOpRunner()
        {
        }

        aclDataType get_dtype(NanoString s)
        {
            if (s == ns_float32)
                return ACL_FLOAT;
            if (s == ns_float16)
                return ACL_FLOAT16;
            if (s == ns_int64)
                return ACL_INT64;
            if (s == ns_int32)
                return ACL_INT32;
            if (s == ns_int8)
                return ACL_INT8;
            if (s == ns_int16)
                return ACL_INT16;
            if (s == ns_uint8)
                return ACL_UINT8;
            if (s == ns_uint16)
                return ACL_UINT16;
            if (s == ns_uint32)
                return ACL_UINT32;
            if (s == ns_bool)
                return ACL_BOOL;
            LOGf << "Not supported dtype: " << s;
            return ACL_FLOAT;
        }

        void add(Var *v, bool is_input)
        {

            if (is_input)
            {
                in_.push_back(v);
            }
            else
            {
                out_.push_back(v);
            }
            return;
        }

        template <typename T>
        std::vector<T> createVector(int64_t size)
        {
            return std::vector<T>(size, 0);
        }

        void run()
        {
            //LOGir << name << " " << jt_name;
            auto it = aclOpFuncMap.find(name);
            if (it == aclOpFuncMap.end())
            {
                LOGir << "aclOpFuncMap Not supported op: " << name;
                throw std::runtime_error("Unsupported operation type.");
            }

            // 0. 算子的输入、输出、需要的attr定义
            std::vector<std::vector<int64_t>> inputShapes;
            std::vector<std::vector<int64_t>> outputShapes;

            // for reduce
            std::vector<int64_t> axes;
            aclIntArray *dim = nullptr;
            bool keepdims;

            bool use_nchw = false;

            auto input_num = in_.size();
            auto output_num = out_.size();

            for (int input_idx = 0; input_idx < input_num; input_idx++)
            {
                std::vector<int64_t> shape;
                for (int j = 0; j < in_[input_idx]->shape.size(); j++)
                {
                    shape.push_back(in_[input_idx]->shape[j]);
                }
                inputShapes.push_back(shape);
            }
            for (int output_idx = 0; output_idx < output_num; output_idx++)
            {
                std::vector<int64_t> shape;
                for (int j = 0; j < out_[output_idx]->shape.size(); j++)
                {
                    shape.push_back(out_[output_idx]->shape[j]);
                }
                outputShapes.push_back(shape);
            }

            // 1. 创建aclTensor和aclScalar，不同算子可能不一样，需要根据具体API的接口定义修改
            std::vector<aclTensor *> inputTensors;
            std::vector<aclTensor *> outputTensors;

            // for add and sub
            aclScalar *alpha = nullptr;

            // for expand
            aclIntArray *size = nullptr;

            // for conv
            aclIntArray *strides = nullptr;
            aclIntArray *pads = nullptr;
            aclIntArray *outPads = nullptr;
            aclIntArray *dilations = nullptr;
            int ret = -1;

            // for maxpool
            aclIntArray *kernel_size = nullptr;

            // for range
            aclScalar *start = nullptr;
            aclScalar *end = nullptr;
            aclScalar *step = nullptr;

            // for leaky_relu
            aclScalar *negativeSlope = nullptr;

            if (name == string("Add") || name == string("Sub"))
            {

                if (get_dtype(in_[0]->dtype()) == ACL_FLOAT)
                {
                    float alphaValue = 1.0;
                    alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
                }
                else if (get_dtype(in_[0]->dtype()) == ACL_FLOAT16)
                {
                    __fp16 alphaValue = 1.0;
                    alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
                }
                else if (get_dtype(in_[0]->dtype()) == ACL_INT64)
                {
                    int64_t alphaValue = 1;
                    alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
                }
                else if (get_dtype(in_[0]->dtype()) == ACL_INT32)
                {
                    int alphaValue = 1;
                    alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
                }
                else if (get_dtype(in_[0]->dtype()) == ACL_INT8)
                {
                    int8_t alphaValue = 1;
                    alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
                }
                else if (get_dtype(in_[0]->dtype()) == ACL_INT16)
                {
                    int16_t alphaValue = 1;
                    alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
                }
                else if (get_dtype(in_[0]->dtype()) == ACL_UINT8)
                {
                    uint8_t alphaValue = 1;
                    alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
                }
                else if (get_dtype(in_[0]->dtype()) == ACL_UINT16)
                {
                    uint16_t alphaValue = 1;
                    alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
                }
                else if (get_dtype(in_[0]->dtype()) == ACL_UINT32)
                {
                    uint32_t alphaValue = 1;
                    alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
                }
                else if (get_dtype(in_[0]->dtype()) == ACL_BOOL)
                {
                    bool alphaValue = true;
                    alpha = aclCreateScalar(&alphaValue, get_dtype(in_[0]->dtype()));
                }
                else
                {
                    LOGf << "Not supported dtype: " << in_[0]->dtype();
                }

                CHECK_RET(alpha != nullptr, return);
            }

            if (jt_name == "conv" || jt_name == "conv2d" || jt_name == "conv2dbackward" || jt_name == "maxpool" || jt_name == "maxpoolbackward" || jt_name == "avgpool" || jt_name == "avgpoolbackward")
                use_nchw = true;

            for (int idx = 0; idx < input_num; idx++)
            {
                inputTensors.push_back(nullptr);
                if ((jt_name == "matmul_trans_1" && idx == 1) || (jt_name == "bmm_trans_1" && idx == 1) || (jt_name == "matmul_trans_0" && idx == 0) || (jt_name == "bmm_trans_0" && idx == 0))
                {
                    auto ret = CreateFakeTransAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw);
                    CHECK_RET(ret == ACL_SUCCESS, return);
                }
                else
                {
                    auto ret = CreateAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw);
                    CHECK_RET(ret == ACL_SUCCESS, return);
                }
            }

            if (jt_name == "reduce" || jt_name == "transpose")
            {
                auto attr = dynamic_cast<ReduceAttr *>(op_attr.get());
                dim = aclCreateIntArray(attr->axes.data(), attr->axes.size());
                keepdims = attr->keepdims;
                if (name == string("ReduceMax") || name == string("ReduceMin") || name == string("ReduceMean") || name == string("ReduceProd"))
                {
                    if (attr->axes.size() == in_[0]->shape.size())
                        outputShapes[0] = {};
                }
            }

            if (jt_name == "range")
            {
                auto attr = dynamic_cast<RangeAttr *>(op_attr.get());
                int64_t startValue = attr->start;
                int64_t endValue = attr->end;
                int64_t stepValue = attr->step;
                start = aclCreateScalar(&startValue, aclDataType::ACL_INT64);
                end = aclCreateScalar(&endValue, aclDataType::ACL_INT64);
                step = aclCreateScalar(&stepValue, aclDataType::ACL_INT64);
            }

            if (jt_name == "conv2dbackward")
            {
                for (int idx = 0; idx < 2; idx++)
                {
                    outputTensors.push_back(nullptr);
                    auto ret = CreateAclTensor(outputShapes[idx], out_[idx]->mem_ptr, out_[idx]->size, get_dtype(out_[idx]->dtype()), &outputTensors[idx], use_nchw);
                    CHECK_RET(ret == ACL_SUCCESS, return);
                }
                // biasgrad nd format
                {
                    outputTensors.push_back(nullptr);
                    auto ret = CreateAclTensor(outputShapes[2], out_[2]->mem_ptr, out_[2]->size, get_dtype(out_[2]->dtype()), &outputTensors[2], false);
                    CHECK_RET(ret == ACL_SUCCESS, return);
                }
            }
            else
            {
                for (int idx = 0; idx < output_num; idx++)
                {
                    outputTensors.push_back(nullptr);
                    auto ret = CreateAclTensor(outputShapes[idx], out_[idx]->mem_ptr, out_[idx]->size, get_dtype(out_[idx]->dtype()), &outputTensors[idx], use_nchw);
                    CHECK_RET(ret == ACL_SUCCESS, return);
                }
            }

            // 2. 调用CANN算子库aclnnxxxGetWorkspaceSize的接口，两段式接口的第一个
            uint64_t workspaceSize = 0;
            aclOpExecutor *executor;
            int op_idx;
            if(jt_name == "binary" && name != "Add" && name != "Sub") 
                op_idx = 6;
            else if(jt_name == "unary" && name != "Cast")
                op_idx = 5;
            else
                op_idx = op_idx_map.find(name)->second;

            //LOGir << name << " " << jt_name;
            //LOGir<<op_idx; 
            switch(op_idx)
            {
                
                case 1 :{
                    ret = it->second.getWorkspaceSizeFuncAdd(inputTensors[0], inputTensors[1], alpha, outputTensors[0], &workspaceSize, &executor);
                    break;}
                
                case 2:{
                    ret = it->second.getWorkspaceSizeFuncAdd(inputTensors[0], inputTensors[1], alpha, outputTensors[0], &workspaceSize, &executor);
                    break;}
                
                case 3:{ 
                    size = aclCreateIntArray(&outputShapes[0][0], outputShapes[0].size());
                    ret = it->second.getWorkspaceSizeFuncExpand(inputTensors[0], size, outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 4:{
                    ret = it->second.getWorkspaceSizeFuncCast(inputTensors[0], get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
                    break;}  
                case 5:{
                    ret = it->second.getWorkspaceSizeFuncUnaryNonzero(inputTensors[0], outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 6:{
                    ret = it->second.getWorkspaceSizeFuncBinary(inputTensors[0], inputTensors[1], outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 7:{
                    ret = it->second.getWorkspaceSizeFuncMatmul(inputTensors[0], inputTensors[1], outputTensors[0], 1, &workspaceSize, &executor);
                    break;}
                case 8:{
                    ret = it->second.getWorkspaceSizeFuncMatmul(inputTensors[0], inputTensors[1], outputTensors[0], 1, &workspaceSize, &executor);
                    break;}  
                case 9 :{    
                    ret = it->second.getWorkspaceSizeFuncReduceSum(inputTensors[0], dim, keepdims, get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
                    break;}
                case 10:{    
                    ret = it->second.getWorkspaceSizeFuncReduceSum(inputTensors[0], dim, keepdims, get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
                    break;}  
                case 11 :{
                    ret = it->second.getWorkspaceSizeFuncAmax(inputTensors[0], dim, keepdims, outputTensors[0], &workspaceSize, &executor);
                    break;}
                case 12:{
                    ret = it->second.getWorkspaceSizeFuncAmax(inputTensors[0], dim, keepdims, outputTensors[0], &workspaceSize, &executor);
                    break;}    
                
                case 13 :{    
                    auto attr = dynamic_cast<RandomAttr *>(op_attr.get());
                    ret = it->second.getWorkspaceSizeFuncRandom(outputTensors[0], 0.0, 1.0, attr->seed, attr->offset, &workspaceSize, &executor);
                    break;}
                case 14:{    
                    auto attr = dynamic_cast<RandomAttr *>(op_attr.get());
                    ret = it->second.getWorkspaceSizeFuncRandom(outputTensors[0], 0.0, 1.0, attr->seed, attr->offset, &workspaceSize, &executor);
                    break;} 
                case 15:{ 
                    ret = it->second.getWorkspaceSizeFuncUnaryNonzero(inputTensors[0], outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 16 :{
                    ret = it->second.getWorkspaceSizeFuncSelect(inputTensors[0], inputTensors[1], inputTensors[2], outputTensors[0], &workspaceSize, &executor);
                    break;}
                case 17:{
                    ret = it->second.getWorkspaceSizeFuncSelect(inputTensors[0], inputTensors[1], inputTensors[2], outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 18:{
                    auto attr = dynamic_cast<TriuAttr *>(op_attr.get());
                    ret = it->second.getWorkspaceSizeFuncCast(inputTensors[0], aclDataType(attr->diagonal), outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 19:{
                    ret = it->second.getWorkspaceSizeFuncExpand(inputTensors[0], dim, outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 20:{
                    auto attr = dynamic_cast<ConvAttr *>(op_attr.get());
                    strides = aclCreateIntArray(attr->convStrides.data(), 2);
                    pads = aclCreateIntArray(attr->convPads.data(), 2);
                    outPads = aclCreateIntArray(attr->convOutPads.data(), 2);
                    dilations = aclCreateIntArray(attr->convDilations.data(), 2);
                    aclTensor *bias = nullptr;
                    if (input_num == 3)
                        bias = inputTensors[2];

                    ret = it->second.getWorkspaceSizeFuncConv(inputTensors[0], inputTensors[1], bias, strides, pads, dilations, false, outPads, attr->group, outputTensors[0], 0, &workspaceSize, &executor);
                    break;} 
                case 21:{
                    auto attr = dynamic_cast<ConvAttr *>(op_attr.get());
                    strides = aclCreateIntArray(attr->convStrides.data(), 2);
                    pads = aclCreateIntArray(attr->convPads.data(), 2);
                    outPads = aclCreateIntArray(attr->convOutPads.data(), 2);
                    dilations = aclCreateIntArray(attr->convDilations.data(), 2);
                    bool outputMask[3] = {true, true, true};
                    if (input_num == 3)
                    {
                        outputMask[2] = false;
                    }
                    aclBoolArray *outMask = aclCreateBoolArray(outputMask, 3);
                    auto biasSizes = aclCreateIntArray(&outputShapes[2][0], outputShapes[2].size());
                    ret = it->second.getWorkspaceSizeFuncConvBackward(inputTensors[0], inputTensors[1], inputTensors[2], biasSizes, strides, pads, dilations, false, outPads, attr->group, outMask, 0, outputTensors[0], outputTensors[1], outputTensors[2], &workspaceSize, &executor);
                    break;} 
                case 22:{
                    auto attr = dynamic_cast<PoolAttr *>(op_attr.get());
                    kernel_size = aclCreateIntArray(attr->kernel_size.data(), 2);
                    strides = aclCreateIntArray(attr->poolStrides.data(), 2);
                    pads = aclCreateIntArray(attr->poolPads.data(), 2);
                    dilations = aclCreateIntArray(attr->poolDilations.data(), 2);
                    ret = it->second.getWorkspaceSizeFuncMaxPool(inputTensors[0], kernel_size, strides, pads, dilations, attr->poolCeil, outputTensors[0], outputTensors[1], &workspaceSize, &executor);
                    break;}  
                case 23:{
                    auto attr = dynamic_cast<PoolAttr *>(op_attr.get());
                    kernel_size = aclCreateIntArray(attr->kernel_size.data(), 2);
                    strides = aclCreateIntArray(attr->poolStrides.data(), 2);
                    pads = aclCreateIntArray(attr->poolPads.data(), 2);
                    dilations = aclCreateIntArray(attr->poolDilations.data(), 2);
                    ret = it->second.getWorkspaceSizeFuncMaxPoolBackward(inputTensors[0], inputTensors[1], inputTensors[2], kernel_size, strides, pads, dilations, attr->poolCeil, outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 24:{
                    auto attr = dynamic_cast<PoolAttr *>(op_attr.get());
                    kernel_size = aclCreateIntArray(attr->kernel_size.data(), 2);
                    strides = aclCreateIntArray(attr->poolStrides.data(), 2);
                    pads = aclCreateIntArray(attr->poolPads.data(), 2);
                    ret = it->second.getWorkspaceSizeFuncAvgPool(inputTensors[0], kernel_size, strides, pads, attr->poolCeil, attr->countIncludePad, attr->divisorOverride, attr->divisorOverride, outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 25:{
                    auto attr = dynamic_cast<PoolAttr *>(op_attr.get());
                    kernel_size = aclCreateIntArray(attr->kernel_size.data(), 2);
                    strides = aclCreateIntArray(attr->poolStrides.data(), 2);
                    pads = aclCreateIntArray(attr->poolPads.data(), 2);
                    ret = it->second.getWorkspaceSizeFuncAvgPoolBackward(inputTensors[0], inputTensors[1], kernel_size, strides, pads, attr->countIncludePad, attr->divisorOverride, attr->divisorOverride, attr->poolCeil, outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 26:{
                    auto attr = dynamic_cast<ReduceAttr *>(op_attr.get());
                    dim = aclCreateIntArray(attr->axes.data(), attr->axes.size());
                    ret = it->second.getWorkspaceSizeFuncExpand(inputTensors[0], dim, outputTensors[0], &workspaceSize, &executor);
                    break;}  
                case 27:{
                    std::vector<aclTensor *> concatTensorList = {};
                    for (int i = 0; i < input_num; i++)
                    {
                        concatTensorList.push_back(inputTensors[i]);
                    }
                    auto concatTensorListInput = aclCreateTensorList(&concatTensorList[0], input_num);
                    auto attr = dynamic_cast<ConcatAttr *>(op_attr.get());
                    ret = it->second.getWorkspaceSizeFuncConcat(concatTensorListInput, attr->dim, outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 28:{
                    auto attr = dynamic_cast<GatherAttr *>(op_attr.get());
                    ret = it->second.getWorkspaceSizeFuncGather(inputTensors[0], attr->dim, inputTensors[1], outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 29:{
                    auto attr = dynamic_cast<GatherAttr *>(op_attr.get());
                    ret = it->second.getWorkspaceSizeFuncCumsum(inputTensors[0], attr->dim, get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 30:{
                    auto attr = dynamic_cast<ScatterAttr *>(op_attr.get());
                    ret = it->second.getWorkspaceSizeFuncScatter(inputTensors[0], attr->axis, inputTensors[1], inputTensors[2], attr->reduction, outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 31:{
                    ret = it->second.getWorkspaceSizeFuncUnaryNonzero(inputTensors[0], outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 32:{
                    auto indexTensorList = aclCreateTensorList(&inputTensors[1], input_num - 1);
                    ret = it->second.getWorkspaceSizeFuncIndex(inputTensors[0], indexTensorList, outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 33:{
                    auto attr = dynamic_cast<StrideAttr *>(op_attr.get());
                    auto begins = aclCreateIntArray(attr->begins.data(), attr->begins.size());
                    auto ends = aclCreateIntArray(attr->ends.data(), attr->ends.size());
                    auto steps = aclCreateIntArray(attr->steps.data(), attr->steps.size());
                    auto axes = aclCreateIntArray(attr->axes.data(), attr->axes.size());
                    ret = it->second.getWorkspaceSizeFuncSliceV2(inputTensors[0], begins, ends, axes, steps, outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 34:{
                    std::vector<aclTensor *> indexTensorList = {};
                    for (int i = 1; i < input_num; i++)
                    {
                        indexTensorList.push_back(inputTensors[i]);
                    }
                    auto indexTensorListInput = aclCreateTensorList(&indexTensorList[0], input_num - 1);
                    ret = it->second.getWorkspaceSizeFuncIndexPutImpl(outputTensors[0], indexTensorListInput, inputTensors[0], false, true, &workspaceSize, &executor);
                    break;} 
                case 35:{
                    std::vector<aclTensor *> indexTensorList = {};
                    for (int i = 1; i < input_num; i++)
                    {
                        indexTensorList.push_back(inputTensors[i]);
                    }
                    auto indexTensorListInput = aclCreateTensorList(&indexTensorList[0], input_num - 1);
                    ret = it->second.getWorkspaceSizeFuncIndexPutImpl(outputTensors[0], indexTensorListInput, inputTensors[0], true, true, &workspaceSize, &executor);
                    break;} 
                case 36:{
                    auto attr = dynamic_cast<StrideAttr *>(op_attr.get());
                    auto begins = aclCreateIntArray(attr->begins.data(), attr->begins.size());
                    auto ends = aclCreateIntArray(attr->ends.data(), attr->ends.size());
                    auto steps = aclCreateIntArray(attr->steps.data(), attr->steps.size());
                    auto axes = aclCreateIntArray(attr->axes.data(), attr->axes.size());
                    ret = it->second.getWorkspaceSizeFuncStridedSliceAssignV2(outputTensors[0], inputTensors[0], begins, ends, steps, axes, &workspaceSize, &executor);
                    break;} 
                case 37:{
                    ret = it->second.getWorkspaceSizeFuncRange(start, end, step, outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 38:{
                    auto attr = dynamic_cast<LeakyReluAttr *>(op_attr.get());
                    negativeSlope = aclCreateScalar(&attr->negativeSlope, aclDataType::ACL_FLOAT);
                    ret = it->second.getWorkspaceSizeFuncLeakyRelu(inputTensors[0], negativeSlope, outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 39:{
                    auto attr = dynamic_cast<LeakyReluAttr *>(op_attr.get());
                    negativeSlope = aclCreateScalar(&attr->negativeSlope, aclDataType::ACL_FLOAT);
                    ret = it->second.getWorkspaceSizeFuncLeakyReluBackward(inputTensors[0], inputTensors[1], negativeSlope, attr->selfIsResult, outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case  40:{
                    auto attr = dynamic_cast<DropoutAttr *>(op_attr.get());
                    ret = it->second.getWorkspaceSizeFuncDropout(inputTensors[0], attr->p, attr->train, attr->seed, attr->offset, outputTensors[0], outputTensors[1], &workspaceSize, &executor);
                    break;} 
                case 41:{
                    auto attr = dynamic_cast<DropoutAttr *>(op_attr.get());
                    ret = it->second.getWorkspaceSizeFuncDropoutBackward(inputTensors[0], inputTensors[1], attr->scale, outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 42:{
                    ret = it->second.getWorkspaceSizeFuncUnaryNonzero(inputTensors[0], outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 43:{
                    ret = it->second.getWorkspaceSizeFuncBinary(inputTensors[0], inputTensors[1], outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 44:{
                    ret = it->second.getWorkspaceSizeFuncUnaryNonzero(inputTensors[0], outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 45:{
                    ret = it->second.getWorkspaceSizeFuncBinary(inputTensors[0], inputTensors[1], outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 46:{
                    ret = it->second.getWorkspaceSizeFuncBinary(inputTensors[0], inputTensors[1], outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 47:{
                    auto attr = dynamic_cast<EmbeddingAttr *>(op_attr.get());
                    auto numEmbeddings = attr->numEmbeddings;
                    ret = it->second.getWorkspaceSizeFuncEmbeddingBackward(inputTensors[0], inputTensors[1], numEmbeddings, 0, false, outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 48:{
                    ret = it->second.getWorkspaceSizeFuncBinary(outputTensors[0], inputTensors[1], inputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 49:{
                    ret = it->second.getWorkspaceSizeFuncBinary(inputTensors[0], inputTensors[1], outputTensors[0], &workspaceSize, &executor);
                    break;} 
                case 50:{
                    auto attr = dynamic_cast<SplitWithSizeAttr *>(op_attr.get());
                    auto splitSize = aclCreateIntArray(attr->splitSize.data(), attr->splitSize.size());
                    auto tensorList = aclCreateTensorList(&outputTensors[0], output_num);
                    ret = it->second.getWorkspaceSizeFuncSplitWithSize(inputTensors[0], splitSize, attr->dim, tensorList, &workspaceSize, &executor);
                    break;} 
                case 51:{
                    auto attr = dynamic_cast<FlashAttentionAttr *>(op_attr.get());
                    auto prefix = aclCreateIntArray(attr->prefix.data(), attr->prefix.size());
                    auto qstart = aclCreateIntArray(attr->qStartIdx.data(), attr->qStartIdx.size());
                    auto kvstart = aclCreateIntArray(attr->kvStartIdx.data(), attr->kvStartIdx.size());
                    char *layout = const_cast<char *>(attr->inputLayout.data());
                    ret = it->second.getWorkspaceSizeFuncFalshAttention(inputTensors[0], inputTensors[1], inputTensors[2], attr->hasRealshift ? inputTensors[3] : nullptr, attr->hasDropmask ? inputTensors[4] : nullptr, nullptr, attr->hasAttentmask ? inputTensors[6] : nullptr, prefix, qstart, kvstart, attr->scale, attr->keepProb, attr->preToken, attr->nextToken, attr->headNum, layout, attr->innerPrecise, attr->sparseMode, attr->psetype, outputTensors[0], outputTensors[1], nullptr, outputTensors[2], &workspaceSize, &executor);
                    break;}  
                case 52:{
                    auto attr = dynamic_cast<FlashAttentionAttr *>(op_attr.get());
                    auto prefix = aclCreateIntArray(attr->prefix.data(), attr->prefix.size());
                    auto qstart = aclCreateIntArray(attr->qStartIdx.data(), attr->qStartIdx.size());
                    auto kvstart = aclCreateIntArray(attr->kvStartIdx.data(), attr->kvStartIdx.size());
                    char *layout = const_cast<char *>(attr->inputLayout.data());
                    ret = it->second.getWorkspaceSizeFuncFalshAttentionBackward(inputTensors[0], inputTensors[1], inputTensors[2], inputTensors[3], attr->hasRealshift ? inputTensors[4] : nullptr, attr->hasDropmask ? inputTensors[5] : nullptr, nullptr, attr->hasAttentmask ? inputTensors[7] : nullptr, inputTensors[8], inputTensors[9], nullptr, inputTensors[10], prefix, qstart, kvstart, attr->scale, attr->keepProb, attr->preToken, attr->nextToken, attr->headNum, layout, attr->innerPrecise, attr->sparseMode, attr->psetype, outputTensors[0], outputTensors[1], outputTensors[2], nullptr, &workspaceSize, &executor);
                    break;} 
                default:{
                    LOGir << "not supported op: " << name;
                    break;
                }
                // for debug
                if (ret != ACL_SUCCESS)
                {
                    auto tmp_err_msg = aclGetRecentErrMsg();
                    LOGir << name << ", " << tmp_err_msg;
                }
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxxGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);
            }
            // 4. 根据第一段接口计算出的workspaceSize申请device内存
            if (workspaceSize > 0)
            {
                mallocWorkSpace(workspaceSize);
            }

            // 5. 调用aclnnxx第二段接口
            ret = it->second.executeFunc(workspaceAddr, workspaceSize, executor, aclstream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxx failed. ERROR: %d\n", name.c_str(), ret); return);

            // 6. （固定写法）同步等待任务执行结束
            // ret = aclrtSynchronizeStream(aclstream);
            // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclrtSynchronizeStream failed. ERROR: %d\n", name.c_str(), ret); return);

            // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
            // destroy tensor
            for (int idx = 0; idx < input_num; idx++)
            {
                aclDestroyTensor(inputTensors[idx]);
            }
            for (int idx = 0; idx < output_num; idx++)
            {
                aclDestroyTensor(outputTensors[idx]);
            }
            // destroy scalar
            aclDestroyScalar(alpha);
            aclDestroyScalar(start);
            aclDestroyScalar(end);
            aclDestroyScalar(step);
            aclDestroyScalar(negativeSlope);

            // destroy IntArray
            aclDestroyIntArray(size);
            aclDestroyIntArray(dim);
            aclDestroyIntArray(strides);
            aclDestroyIntArray(pads);
            aclDestroyIntArray(outPads);
            aclDestroyIntArray(dilations);
            aclDestroyIntArray(kernel_size);

            return;
        }
    };
}