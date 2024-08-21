#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>

namespace jittor
{
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
            // LOGir << name << " " << jt_name;
            auto it = aclOpFuncMap.find(name);
            if (it == aclOpFuncMap.end())
            {
                LOGir << "Not supported op: " << name;
                throw std::runtime_error("Unsupported operation type.");
            }

            // 0. 算子的输入、输出、需要的attr定义
            std::vector<std::vector<int64_t>> inputShapes;
            std::vector<std::vector<int64_t>> outputShapes;

            // for reduce
            std::vector<int64_t> axes;
            aclIntArray *dim = nullptr;

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

            // for add and sub
            float alphaValue = 1.0f;

            // for conv
            aclIntArray *strides = nullptr;
            aclIntArray *pads = nullptr;
            aclIntArray *outPads = nullptr;
            aclIntArray *dilations = nullptr;
            int ret = -1;

            if (name == string("Add") || name == string("Sub"))
            {
                alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
                CHECK_RET(alpha != nullptr, return);
            }

            if (jt_name == "conv" || jt_name == "conv2d" || jt_name == "conv2dbackward")
                use_nchw = true;

            for (int idx = 0; idx < input_num; idx++)
            {
                inputTensors.push_back(nullptr);
                auto ret = CreateAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw);
                CHECK_RET(ret == ACL_SUCCESS, return);
            }

            if (jt_name == "reduce")
            {
                auto attr = dynamic_cast<ReduceAttr *>(op_attr.get());
                dim = aclCreateIntArray(attr->axes.data(), attr->axes.size());

                if (name == string("ReduceMax") || name == string("ReduceMin") || name == string("ReduceMean") || name == string("ReduceProd"))
                {
                    if (attr->axes.size() == in_[0]->shape.size())
                        outputShapes[0] = {};
                }
            }
            for (int idx = 0; idx < output_num; idx++)
            {
                outputTensors.push_back(nullptr);
                auto ret = CreateAclTensor(outputShapes[idx], out_[idx]->mem_ptr, out_[idx]->size, get_dtype(out_[idx]->dtype()), &outputTensors[idx], use_nchw);
                CHECK_RET(ret == ACL_SUCCESS, return);
            }

            // 2. 调用CANN算子库aclnnxxxGetWorkspaceSize的接口，两段式接口的第一个
            uint64_t workspaceSize = 0;
            aclOpExecutor *executor;

            if (name == string("Add") || name == string("Sub"))
                ret = it->second.getWorkspaceSizeFuncAdd(inputTensors[0], inputTensors[1], alpha, outputTensors[0], &workspaceSize, &executor);
            else if (name == string("Expand"))
            {
                size = aclCreateIntArray(&outputShapes[0][0], outputShapes[0].size());
                ret = it->second.getWorkspaceSizeFuncExpand(inputTensors[0], size, outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Cast"))
                ret = it->second.getWorkspaceSizeFuncCast(inputTensors[0], get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
            else if (jt_name == "unary")
                ret = it->second.getWorkspaceSizeFuncUnary(inputTensors[0], outputTensors[0], &workspaceSize, &executor);
            else if (jt_name == "binary")
                ret = it->second.getWorkspaceSizeFuncBinary(inputTensors[0], inputTensors[1], outputTensors[0], &workspaceSize, &executor);
            else if (jt_name == "bmm" || jt_name == "matmul")
                ret = it->second.getWorkspaceSizeFuncMatmul(inputTensors[0], inputTensors[1], outputTensors[0], 1, &workspaceSize, &executor);
            else if (name == string("ReduceSum") || name == string("ReduceMean"))
            {
                ret = it->second.getWorkspaceSizeFuncReduceSum(inputTensors[0], dim, false, get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("ReduceMax") || name == string("ReduceMin"))
            {
                ret = it->second.getWorkspaceSizeFuncAmax(inputTensors[0], dim, false, outputTensors[0], &workspaceSize, &executor);
            }
            // else if (name == string("ReduceProd"))
            // {
            //     ret = it->second.getWorkspaceSizeFuncReduceProd(inputTensors[0], dim, false, outputTensors[0], &workspaceSize, &executor);
            // }
            else if (name == string("Select"))
            {
                ret = it->second.getWorkspaceSizeFuncSelect(inputTensors[0], inputTensors[1], inputTensors[2], outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Triu"))
            {
                auto attr = dynamic_cast<TriuAttr *>(op_attr.get());
                ret = it->second.getWorkspaceSizeFuncCast(inputTensors[0], aclDataType(attr->diagonal), outputTensors[0], &workspaceSize, &executor);
            }
            else if (name == string("Conv2d"))
            {
                auto attr = dynamic_cast<ConvAttr *>(op_attr.get());
                strides = aclCreateIntArray(attr->convStrides.data(), 2);
                pads = aclCreateIntArray(attr->convPads.data(), 2);
                outPads = aclCreateIntArray(attr->convOutPads.data(), 2);
                dilations = aclCreateIntArray(attr->convDilations.data(), 2);

                ret = it->second.getWorkspaceSizeFuncConv(inputTensors[0], inputTensors[1], nullptr, strides, pads, dilations, false, outPads, attr->group, outputTensors[0], 0, &workspaceSize, &executor);
            }
            else if (name == string("Conv2dBackward"))
            {
                auto attr = dynamic_cast<ConvAttr *>(op_attr.get());
                strides = aclCreateIntArray(attr->convStrides.data(), 2);
                pads = aclCreateIntArray(attr->convPads.data(), 2);
                outPads = aclCreateIntArray(attr->convOutPads.data(), 2);
                dilations = aclCreateIntArray(attr->convDilations.data(), 2);
                bool outputMask[3] = {true, true, false};
                aclBoolArray *outMask = aclCreateBoolArray(outputMask, 3);
                ret = it->second.getWorkspaceSizeFuncConvBackward(inputTensors[0], inputTensors[1], inputTensors[2], nullptr, strides, pads, dilations, false, outPads, attr->group, outMask, 0, outputTensors[0], outputTensors[1], nullptr, &workspaceSize, &executor);
            }
            else
                LOGf << "not supported op " << jt_name;

            // for debug
            if (ret != ACL_SUCCESS)
            {
                auto tmp_err_msg = aclGetRecentErrMsg();
                LOGir << tmp_err_msg;
            }

            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxxGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);

            // 4. 根据第一段接口计算出的workspaceSize申请device内存
            void *workspaceAddr = nullptr;
            if (workspaceSize > 0)
            {
                ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: allocate workspace failed. ERROR: %d\n", name.c_str(), ret); return);
            }

            // 5. 调用aclnnxx第二段接口
            ret = it->second.executeFunc(workspaceAddr, workspaceSize, executor, aclstream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxx failed. ERROR: %d\n", name.c_str(), ret); return);

            // 6. （固定写法）同步等待任务执行结束
            ret = aclrtSynchronizeStream(aclstream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclrtSynchronizeStream failed. ERROR: %d\n", name.c_str(), ret); return);

            // 7. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
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

            // destroy IntArray
            aclDestroyIntArray(size);
            aclDestroyIntArray(dim);
            aclDestroyIntArray(strides);
            aclDestroyIntArray(pads);
            aclDestroyIntArray(outPads);
            aclDestroyIntArray(dilations);

            // 8. 释放device资源
            if (workspaceSize > 0)
            {
                aclrtFree(workspaceAddr);
            }
            return;
        }
    };
}