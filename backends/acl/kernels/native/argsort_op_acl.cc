#include <limits>
#include <memory>
#include "core/var.h"
#include "core/executor.h"
#include "aclops/argsort_op_acl.h"
#include "aclnnop/aclnn_sort.h"
#include "aclnnop/aclnn_cast.h"

namespace jittor {

ArgsortOpRunner::ArgsortOpRunner(int64_t dim, bool descending)
    : BaseOpRunner("Sort"), dim(dim), descending(descending) {}

void ArgsortOpRunner::executeOp(AclOpRegistry::const_iterator &) {
    if (out_[0]->num == 0) return;
    auto *indices = outputTensors[0];
    std::unique_ptr<Allocation> temporary;
    std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)> descriptor(
        nullptr, aclDestroyTensor);
    if (out_[0]->dtype() == ns_int32) {
        USER_CHECK(in_[0]->shape[dim] <= std::numeric_limits<int32_t>::max())
            << "ACL argsort axis does not fit requested int32 indices";
        USER_CHECK(uint64_t(out_[0]->num) <=
            std::numeric_limits<size_t>::max() / sizeof(int64_t))
            << "ACL argsort index storage size overflow";
        temporary.reset(new Allocation(
            runtime_executor().temp_allocator, size_t(out_[0]->num) * sizeof(int64_t)));
        std::vector<int64_t> shape;
        for (int i = 0; i < out_[0]->shape.size(); ++i)
            shape.push_back(out_[0]->shape[i]);
        aclTensor *raw = nullptr;
        auto status = CreateAclTensor(shape, temporary->ptr, temporary->size,
                                      ACL_INT64, &raw);
        descriptor.reset(raw);
        USER_CHECK(status == ACL_SUCCESS)
            << "ACL argsort index descriptor failed: " << status;
        indices = descriptor.get();
    }
    try {
        // The core API has no stable flag. Keep its default unordered-tie
        // policy; CANN returns the full permutation on the requested axis.
        ret = aclnnSortGetWorkspaceSize(
            inputTensors[0], false, dim, descending, outputTensors[1], indices,
            &workspaceSize, &executor);
        launch(ret, aclnnSort, true);
        if (temporary) {
            ret = aclnnCastGetWorkspaceSize(
                indices, ACL_INT32, outputTensors[0], &workspaceSize, &executor);
            launch(ret, aclnnCast, true);
            // BaseOpRunner::launch(true) obeys the optional sync_run flag.
            // The pool releases Allocation immediately, so this local buffer
            // requires unconditional completion before returning its owner.
            auto status = aclrtSynchronizeStream(aclstream);
            USER_CHECK(status == ACL_SUCCESS)
                << "ACL argsort temporary-index synchronization failed: " << status;
        }
    } catch (...) {
        // A later workspace query may fail after Sort was queued. Complete
        // that work before stack unwinding releases the temporary allocation.
        if (temporary) aclrtSynchronizeStream(aclstream);
        throw;
    }
}

} // namespace jittor
