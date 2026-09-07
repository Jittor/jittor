#include "native_indexing_op_acl.h"
#include "acl_jittor.h"
#include "ops/composite/getitem_op.h"
#include "ops/composite/setitem_op.h"
#include "runtime/backend.h"
#include "runtime/device_state.h"

namespace jittor {
namespace {

using acl_indexing::CopyPlan;
using acl_indexing::Selection;
using acl_indexing::SliceKind;
using acl_indexing::SliceSpec;

struct PreparedIndex {
    CopyPlan copies;
    bool skip_selection = false;
    bool skip_base = false;
};

string shape_of(const Var* value, std::vector<size_t>& shape) {
    shape.clear();
    for (int axis = 0; axis < value->shape.size(); ++axis) {
        const auto size = value->shape[axis];
        if (size < 0) return "unresolved dynamic index dimension";
        shape.push_back(static_cast<size_t>(size));
    }
    size_t elements, bytes;
    if (!acl_indexing::product(shape, elements) ||
        !acl_indexing::multiply(elements, value->dtype().dsize(), bytes))
        return "tensor byte size overflow";
    if (value->num < 0 || value->size < 0 || elements != static_cast<size_t>(value->num) ||
        bytes != static_cast<size_t>(value->size)) return "inconsistent tensor size metadata";
    return {};
}

string selection_of(Var* input, const VarSlices& slices, Selection& selection) {
    std::vector<size_t> shape;
    auto reason = shape_of(input, shape);
    if (!reason.empty()) return reason;
    if (slices.n < 0 || (slices.n && !slices.slices)) return "invalid slice metadata";
    std::vector<SliceSpec> specs;
    specs.reserve(slices.n);
    for (int i = 0; i < slices.n; ++i) {
        const auto& slice = slices.slices[i];
        SliceSpec spec;
        if (slice.is_var()) spec.kind = SliceKind::Advanced;
        else if (slice.is_str()) spec.kind = SliceKind::Expression;
        else if (slice.is_none()) spec.kind = SliceKind::NewAxis;
        else if (slice.is_ellipsis()) spec.kind = SliceKind::Ellipsis;
        else if (slice.is_int()) {
            spec.kind = SliceKind::Integer;
            spec.start = slice.i;
        } else if (slice.is_slice()) {
            if (slice.slice.mask == 7) spec.kind = SliceKind::Full;
            else if (input->num == 0 && slice.slice.mask != 0) {
                // infer_slices intentionally leaves bounds unfilled on a
                // zero-size input axis. Its selection is empty for any step.
                if (!(slice.slice.mask & 4) && slice.slice.step <= 0)
                    return "zero or negative slice step is not implemented";
                spec.kind = SliceKind::Full;
            } else {
                if (slice.slice.mask != 0) return "slice metadata has not been normalized";
                spec.kind = SliceKind::Range;
                spec.start = slice.slice.start;
                spec.stop = slice.slice.stop;
                spec.step = slice.slice.step;
            }
        } else return "unknown slice variant";
        specs.push_back(spec);
    }
    return acl_indexing::make_selection(shape, specs, input->dtype().dsize(), selection);
}

uintptr_t address(const Var* value) {
    return reinterpret_cast<uintptr_t>(value->mem_ptr);
}

bool shared_full_storage(Var* left, Var* right) {
    return left->size == right->size && left->mem_ptr == right->mem_ptr &&
        left->shares_allocation_with(right);
}

string acl_storage(Var* value, int& device) {
    if (!value->size) return {};
    if (!value->mem_ptr || !value->allocator) return "indexing storage is not materialized";
    uintptr_t end;
    if (!acl_indexing::address_end(address(value), static_cast<size_t>(value->size), end))
        return "storage address range overflow";
    const auto location = allocation_device(value->allocator);
    if (location.backend != BackendId::Acl) return "native ACL indexing requires device-resident storage";
    if (device >= 0 && location.index != device) return "native ACL indexing requires one device";
    device = location.index;
    return {};
}

string prepare_getitem(GetitemOp* op, PreparedIndex& prepared) {
    if (op->inputs().size() != 1 || op->outputs().size() == 0 || op->outputs().size() > 2)
        return "advanced or invalid getitem arity";
    auto* input = op->input(0);
    auto* output = op->output(0);
    if (input->dtype() != output->dtype()) return "getitem dtype mismatch";
    Selection selection;
    auto reason = selection_of(input, op->vs, selection);
    if (!reason.empty()) return reason;
    std::vector<size_t> shape;
    reason = shape_of(output, shape);
    if (!reason.empty()) return reason;
    reason = acl_indexing::make_get_plan(selection, shape, prepared.copies);
    if (!reason.empty()) return reason;
    if (!selection.elements && op->outputs().size() == 1) return {};
    int device = -1;
    for (auto* value : {input, output}) {
        reason = acl_storage(value, device);
        if (!reason.empty()) return reason;
    }
    const auto& plan = prepared.copies;
    if (plan.elements) {
        prepared.skip_selection = op->ns.get(GetitemOp::_inplace) &&
            input->shares_allocation_with(output) &&
            acl_indexing::identical_mapping(plan, address(input), address(output));
        if (!prepared.skip_selection && acl_indexing::overlaps(
                address(input), 0, input->size,
                address(output), plan.target_offset, plan.target_end))
            return "overlapping getitem storage is not an exact shared view";
    }
    if (op->outputs().size() == 2) {
        auto* original = op->output(1);
        reason = shape_of(original, shape);
        if (!reason.empty()) return reason;
        if (original->shape != input->shape || original->dtype() != input->dtype())
            return "getitem return_x metadata differs from its input";
        reason = acl_storage(original, device);
        if (!reason.empty()) return reason;
        prepared.skip_base = shared_full_storage(input, original);
        if (!prepared.skip_base && input->size && acl_indexing::overlaps(
                address(input), 0, input->size, address(original), 0, original->size))
            return "overlapping getitem return_x storage";
        if (!prepared.skip_base && plan.elements && acl_indexing::overlaps(
                address(output), 0, output->size, address(original), 0, original->size))
            return "getitem outputs overlap without the original shared allocation";
    }
    return {};
}

string prepare_setitem(SetitemOp* op, PreparedIndex& prepared) {
    if (op->inputs().size() != 2 || op->outputs().size() != 1)
        return "advanced or invalid setitem arity";
    if (op->op != ns_void) return "native ACL setitem reduction is not implemented";
    auto* input = op->input(0);
    auto* value = op->input(1);
    auto* output = op->output(0);
    if (input->dtype() != output->dtype() || input->shape != output->shape)
        return "setitem output metadata differs from its input";
    Selection selection;
    auto reason = selection_of(input, op->vs, selection);
    if (!reason.empty()) return reason;
    if (selection.elements && value->dtype() != input->dtype())
        return "native ACL setitem requires an explicit value cast";
    std::vector<size_t> value_shape, output_shape;
    reason = shape_of(value, value_shape);
    if (!reason.empty()) return reason;
    reason = shape_of(output, output_shape);
    if (!reason.empty()) return reason;
    reason = acl_indexing::make_set_plan(selection, value_shape, prepared.copies);
    if (!reason.empty()) return reason;
    int device = -1;
    for (auto* tensor : {input, output}) {
        reason = acl_storage(tensor, device);
        if (!reason.empty()) return reason;
    }
    prepared.skip_base = shared_full_storage(input, output);
    if (!prepared.skip_base && input->size && acl_indexing::overlaps(
            address(input), 0, input->size, address(output), 0, output->size))
        return "setitem base copy has overlapping storage";
    const auto& plan = prepared.copies;
    if (!plan.elements) return {};
    reason = acl_storage(value, device);
    if (!reason.empty()) return reason;
    prepared.skip_selection = op->ns.get(GetitemOp::_inplace) &&
        input->shares_allocation_with(value) &&
        acl_indexing::identical_mapping(plan, address(value), address(input));
    if (prepared.skip_selection) return {};
    const size_t target_begin = prepared.skip_base ? plan.target_offset : 0;
    const size_t target_end = prepared.skip_base ? plan.target_end : static_cast<size_t>(output->size);
    if (acl_indexing::overlaps(address(value), plan.source_offset, plan.source_end,
                               address(output), target_begin, target_end))
        return "setitem output would overwrite unread assignment values";
    return {};
}

void copy_region(Var* source, Var* target, size_t source_offset, size_t target_offset, size_t bytes) {
    if (!bytes) return;
    const auto device = allocation_device(target->allocator);
    // ACL producers and consumers use aclstream, not the CUDA-shaped default
    // stream used by the generic ordered-copy helper.
    backend_copy_async(static_cast<char*>(target->mem_ptr) + target_offset,
                       device, static_cast<const char*>(source->mem_ptr) + source_offset,
                       allocation_device(source->allocator), bytes,
                       BackendStream{device, reinterpret_cast<void*>(aclstream)});
}

void finish_copies(Var* output, bool copied) {
    if (copied && runtime_device_state().sync_run)
        backend_synchronize(allocation_device(output->allocator));
}

} // namespace

std::string acl_getitem_unsupported_reason(Op* op) {
    PreparedIndex prepared;
    return prepare_getitem(static_cast<GetitemOp*>(op), prepared);
}

std::string acl_setitem_unsupported_reason(Op* op) {
    PreparedIndex prepared;
    return prepare_setitem(static_cast<SetitemOp*>(op), prepared);
}

void exec_native_acl_getitem(Op* base) {
    auto* op = static_cast<GetitemOp*>(base);
    PreparedIndex prepared;
    auto reason = prepare_getitem(op, prepared);
    USER_CHECK(reason.empty()) << "Unsupported native ACL getitem:" << reason;
    bool copied = false;
    if (!prepared.skip_selection)
        acl_indexing::for_each_copy(prepared.copies, [&](size_t source, size_t target, size_t bytes) {
            copy_region(op->input(0), op->output(0), source, target, bytes);
            copied = true;
        });
    if (op->outputs().size() == 2 && !prepared.skip_base && op->input(0)->size) {
        copy_region(op->input(0), op->output(1), 0, 0, op->input(0)->size);
        copied = true;
    }
    finish_copies(op->output(op->outputs().size() == 2 ? 1 : 0), copied);
}

void exec_native_acl_setitem(Op* base) {
    auto* op = static_cast<SetitemOp*>(base);
    PreparedIndex prepared;
    auto reason = prepare_setitem(op, prepared);
    USER_CHECK(reason.empty()) << "Unsupported native ACL setitem:" << reason;
    bool copied = false;
    if (!prepared.skip_base && op->input(0)->size) {
        copy_region(op->input(0), op->output(0), 0, 0, op->input(0)->size);
        copied = true;
    }
    if (!prepared.skip_selection)
        acl_indexing::for_each_copy(prepared.copies, [&](size_t source, size_t target, size_t bytes) {
            copy_region(op->input(1), op->output(0), source, target, bytes);
            copied = true;
        });
    finish_copies(op->output(0), copied);
}

} // namespace jittor
