#include "runtime/dispatch_context.h"
#include "runtime/backend.h"
#include "runtime/device.h"
#include "var.h"

namespace jittor {
namespace {

bool pending_scalar(Var* value) {
    return !value->is_finished() && value->flag(VarFlags::_is_scalar);
}

// Read-only counterpart of Op::propagate_device's bounded scalar retargeting.
bool can_retarget_pending(Var* value, int device) {
    if (value->device_id == device) return true;
    if (value->is_finished()) return false;
    vector<Var*> pending;
    vector<Node*> queue{value};
    for (size_t index = 0; index < queue.size(); ++index) {
        auto* node = queue[index];
        if (node->is_var()) {
            auto* var = node->var();
            if (var->is_finished()) {
                if (var->device_id != device) return false;
                continue;
            }
            pending.push_back(var);
        }
        if (pending.size() > 32) return false;
        for (const auto& edge : node->_inputs) {
            bool duplicate = false;
            for (auto* queued : queue)
                if (queued == edge.node) { duplicate = true; break; }
            if (!duplicate) queue.push_back(edge.node);
        }
    }
    return true;
}

} // namespace

DispatchContext query_dispatch_context(const vector<Var*>& inputs) {
    int device = -1;
    for (auto* value : inputs) {
        USER_CHECK(value) << "dispatch_context requires non-null tensor inputs";
        if (value->device_id < 0 || pending_scalar(value)) continue;
        if (device < 0) device = value->device_id;
        USER_CHECK(device == value->device_id)
            << "Expected all inputs to be on the same device, but found"
            << device << "and" << value->device_id << "in dispatch_context";
    }
    if (device < 0)
        for (auto* value : inputs)
            if (value->device_id >= 0) { device = value->device_id; break; }
    for (auto* value : inputs) {
        if (value->device_id < 0 || value->device_id == device) continue;
        USER_CHECK(pending_scalar(value) && can_retarget_pending(value, device))
            << "Expected all inputs to be on the same device, but found"
            << device << "and" << value->device_id << "in dispatch_context";
    }
    // Like Op::execution_backend for dual-source ops, runtime policy selects
    // the implementation even when an input is pending or staged on the host.
    if (!runtime_use_cuda()) return {backend_ops(BackendId::Cpu).name, -1};
    if (device < 0) device = runtime_device_state().current_device;
    USER_CHECK(device >= 0)
        << "Accelerator dispatch requires an initialized current device";
    return {backend_ops(accelerator_backend_id()).name, device};
}

} // namespace jittor
