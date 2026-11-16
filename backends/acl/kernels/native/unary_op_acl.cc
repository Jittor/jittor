#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include <algorithm>
#include <queue>
#include <set>
#include "core/common.h"
#include "core/op.h"
#include "acl_jittor.h"
#include "ops/composite/random_op.h"
#include "ops/reduce_op.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/composite/transpose_op.h"
#include "ops/composite/array_op.h"
#include "ops/composite/code_op.h"
#include "core/fused_op.h"
#include "ops/unary_op.h"
#include "ops/ternary_op.h"
#include "core/executor.h"
#include "runtime/device.h"
#include "mem/allocator.h"
#include "codegen/op_compiler.h"
#include "ops/op_register.h"
#include "codegen/opt/tuner_manager.h"
#include "utils/str_utils.h"
#include "aclnn/aclnn.h"

#include "unary_op_acl.h"

namespace jittor
{
    UnaryOpRunner::UnaryOpRunner() : BaseOpRunner("unary")
    {
        use_nchw = false;
        is_group_op = true;
    }

    bool UnaryOpRunner::stridedHighRankCast() const
    {
        return name == "Cast" && in_[0]->shape.size() > 8
            && (!in_[0]->is_contiguous() || !out_[0]->is_contiguous());
    }

    void UnaryOpRunner::setupInputDesc()
    {
        if (!stridedHighRankCast()) { BaseOpRunner::setupInputDesc(); return; }
        inputTensors.resize(1, nullptr);
    }

    void UnaryOpRunner::setupOutputDesc()
    {
        if (!stridedHighRankCast()) { BaseOpRunner::setupOutputDesc(); return; }
        outputTensors.resize(1, nullptr);
    }

    void UnaryOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto execute = [&] {
            AclWorkspaceArguments args;
            args.x = inputTensors[0];
            args.output = outputTensors[0];
            if (name == "Cast") args.dtype = get_dtype(out_[0]->dtype());
            ret = it->second.workspace(args, &workspaceSize, &executor);
            launch(ret, it->second.launcher(), false);
        };
        if (!stridedHighRankCast()) { execute(); return; }

        // CANN limits Cast to eight logical axes. Flattening a view would
        // discard its strides. Iterate its leading coordinates instead,
        // retaining the last eight axes and both operands' physical strides.
        // All chunks stay on the same device stream; no host tensor copy or
        // graph replay is involved. Dense high-rank tensors keep one launch.
        auto* input = in_[0];
        auto* output = out_[0];
        if (!input->num) return;
        const int prefix = input->shape.size() - 8;
        vector<int64_t> shape(8), input_strides(8), output_strides(8);
        int64_t elements = 1;
        for (int i = 0; i < 8; ++i) {
            shape[i] = input->shape[prefix + i];
            input_strides[i] = input->storage_stride(prefix + i);
            output_strides[i] = output->storage_stride(prefix + i);
            elements *= shape[i];
        }
        for (int64_t chunk = 0; chunk < input->num / elements; ++chunk) {
            int64_t remaining = chunk, input_offset = 0, output_offset = 0;
            for (int i = prefix - 1; i >= 0; --i) {
                auto coordinate = remaining % input->shape[i];
                remaining /= input->shape[i];
                input_offset += coordinate * input->storage_stride(i);
                output_offset += coordinate * output->storage_stride(i);
            }
            int64_t input_span = input->storage_span_bytes() / input->dsize() - input_offset;
            int64_t output_span = output->storage_span_bytes() / output->dsize() - output_offset;
            inputTensors[0] = aclCreateTensor(shape.data(), 8, get_dtype(input->dtype()),
                input_strides.data(), 0, aclFormat::ACL_FORMAT_ND, &input_span, 1,
                static_cast<char*>(input->mem_ptr) + input_offset * input->dsize());
            outputTensors[0] = aclCreateTensor(shape.data(), 8, get_dtype(output->dtype()),
                output_strides.data(), 0, aclFormat::ACL_FORMAT_ND, &output_span, 1,
                static_cast<char*>(output->mem_ptr) + output_offset * output->dsize());
            USER_CHECK(inputTensors[0] && outputTensors[0])
                << "ACL Cast could not describe a strided high-rank chunk";
            execute();
            cleanupDesc();
        }
    }
}
