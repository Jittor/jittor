#include <algorithm>
#include <vector>

#include "acl_jittor.h"
#include "core/var.h"
#include "fused_sgd_op_acl.h"

namespace jittor
{
    FusedSgdOpRunner::FusedSgdOpRunner() : ForeachOpRunner("FusedSgd")
    {
    }

    void FusedSgdOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<FusedSgdAttr *>(op_attr.get());
        CHECK(attr != nullptr);
        const int64_t count = attr->tensorCount;
        CHECK(count > 0);
        // inputs: parameters, velocities, gradients; outputs: new parameters,
        // new velocities. The outputs share storage with their inputs, so the
        // copies below are normally skipped.
        CHECK(inputTensors.size() == size_t(count * 3));
        CHECK(outputTensors.size() == size_t(count * 2));

        for (int64_t slot = 0; slot < count * 2; ++slot)
        {
            CHECK(in_[slot]->size == out_[slot]->size);
            if (in_[slot]->mem_ptr == out_[slot]->mem_ptr)
                continue;
            ret = aclrtMemcpyAsync(out_[slot]->mem_ptr, out_[slot]->size,
                                   in_[slot]->mem_ptr, in_[slot]->size,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused SGD D2D copy failed: " + acl_error_to_string(ret));
        }

        // The portable update is
        //     dp = sign * grad + weight_decay * param
        //     v  = momentum * v + (1 - dampening) * dp
        //     p  = p - lr * (nesterov ? dp + momentum * v : v)
        // and every coefficient below is one term of it folded so that no
        // temporary the size of the parameter list is needed: `dp` never
        // materialises, its two halves are accumulated into `v` (and, for
        // Nesterov, into `p`) separately.
        const float sign = attr->maximize ? -1.0f : 1.0f;
        const float retained = float(1.0 - attr->dampening);
        const float lr = float(attr->lr);
        const float momentum = float(attr->momentum);
        const bool decays = attr->weightDecay != 0;

        std::vector<float> coefficients;
        coefficients.push_back(momentum);                              // v <- momentum * v
        coefficients.push_back(retained * sign);                       // v += . * grad
        if (decays)
            coefficients.push_back(retained * float(attr->weightDecay)); // v += . * param
        if (!attr->nesterov)
        {
            coefficients.push_back(-lr);                               // p += . * v
        }
        else
        {
            if (decays)
                coefficients.push_back(1.0f - lr * float(attr->weightDecay)); // p *= .
            coefficients.push_back(-lr * sign);                        // p += . * grad
            coefficients.push_back(-lr * momentum);                    // p += . * v
        }
        stageCoefficients(coefficients);

        // The coefficients are the same for every slice of the parameter list,
        // so they are staged once and read again per slice.
        for (int64_t base = 0; base < count; base += list_limit)
        {
            const int64_t span = std::min(list_limit, count - base);
            const bool last = base + span >= count;
            const aclTensorList *parameters = outputList(base, span);
            const aclTensorList *velocities = outputList(count + base, span);
            const aclTensorList *gradients = inputList(count * 2 + base, span);

            size_t next = 0;
            foreachMulScalar(velocities, coefficient(next++), velocities, false);
            foreachAddList(velocities, gradients, coefficient(next++), velocities, false);
            if (decays)
                foreachAddList(velocities, parameters, coefficient(next++), velocities, false);
            if (!attr->nesterov)
            {
                foreachAddList(parameters, velocities, coefficient(next++), parameters, last);
            }
            else
            {
                if (decays)
                    foreachMulScalar(parameters, coefficient(next++), parameters, false);
                foreachAddList(parameters, gradients, coefficient(next++), parameters, false);
                foreachAddList(parameters, velocities, coefficient(next++), parameters, last);
            }
        }
    }
}
