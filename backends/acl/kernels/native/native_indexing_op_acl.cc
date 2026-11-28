#include "native_indexing_op_acl.h"
#include "acl_jittor.h"
#include "acl_workspace.h"
#include "aclnn/aclnn.h"
#include "utils.h"
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
    if (!value->is_contiguous())
        return "ACL indexed-copy planner requires contiguous storage; use contiguous() before indexed copy";
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
    if (left->size != right->size || left->mem_ptr != right->mem_ptr)
        return false;
    return left->shares_allocation_with(right)
        || (left->allocator && left->allocator == right->allocator
            && left->allocation == right->allocation);
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

bool is_inplace_add_dtype(NanoString ns) {
    return ns == ns_float32 || ns == ns_float16 || ns == ns_bfloat16
        || ns == ns_int8 || ns == ns_int16 || ns == ns_int32 || ns == ns_int64
        || ns == ns_uint8;
}

string prepare_setitem(SetitemOp* op, PreparedIndex& prepared) {
    if (op->inputs().size() != 2 || op->outputs().size() != 1)
        return "advanced or invalid setitem arity";
    const bool is_add = (op->op == ns_add);
    if (op->op != ns_void && !is_add)
        return "native ACL setitem reduction is not implemented";
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
    if (is_add && prepared.copies.elements) {
        if (!is_inplace_add_dtype(output->dtype()))
            return "native ACL setitem add does not support dtype";
        const size_t dsize = output->dtype().dsize();
        const auto& plan = prepared.copies;
        if (!dsize || plan.shape.size() + 1 > 16
                || (plan.block_bytes % dsize) != 0
                || (plan.source_offset % dsize) != 0
                || (plan.target_offset % dsize) != 0)
            return "native ACL setitem add layout is not supported";
        for (size_t i = 0; i < plan.shape.size(); ++i) {
            if ((plan.source_strides[i] % dsize) != 0
                    || (plan.target_strides[i] % dsize) != 0)
                return "native ACL setitem add stride is not aligned";
            if (plan.shape[i] > 1 && plan.target_strides[i] == 0)
                return "native ACL setitem add target has duplicate positions";
        }
    }
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
    prepared.skip_selection = !is_add && op->ns.get(GetitemOp::_inplace) &&
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

bool try_strided_inplace_copy(Var* source, Var* target, const CopyPlan& plan) {
    if (!plan.elements || plan.shape.empty()) return false;
    const size_t dsize = target->dtype().dsize();
    if (!dsize || (plan.block_bytes % dsize) != 0
            || (plan.source_offset % dsize) != 0
            || (plan.target_offset % dsize) != 0)
        return false;
    const size_t rank = plan.shape.size() + 1;
    if (rank > 16) return false;
    const NanoString ns = target->dtype();
    if (ns != ns_float32 && ns != ns_float16 && ns != ns_bfloat16
            && ns != ns_int8 && ns != ns_int16 && ns != ns_int32 && ns != ns_int64
            && ns != ns_uint8 && ns != ns_uint16 && ns != ns_uint32 && ns != ns_bool)
        return false;
    int64_t view_dims[16];
    int64_t src_strides[16];
    int64_t dst_strides[16];
    for (size_t i = 0; i < plan.shape.size(); ++i) {
        if ((plan.source_strides[i] % dsize) != 0 || (plan.target_strides[i] % dsize) != 0)
            return false;
        view_dims[i] = static_cast<int64_t>(plan.shape[i]);
        src_strides[i] = static_cast<int64_t>(plan.source_strides[i] / dsize);
        dst_strides[i] = static_cast<int64_t>(plan.target_strides[i] / dsize);
    }
    view_dims[rank - 1] = static_cast<int64_t>(plan.block_bytes / dsize);
    src_strides[rank - 1] = 1;
    dst_strides[rank - 1] = 1;

    const aclDataType dt = get_dtype(ns);
    const int64_t src_storage = static_cast<int64_t>(source->size / dsize);
    const int64_t dst_storage = static_cast<int64_t>(target->size / dsize);
    const int64_t src_off = static_cast<int64_t>(plan.source_offset / dsize);
    const int64_t dst_off = static_cast<int64_t>(plan.target_offset / dsize);

    aclTensor* dst_t = aclCreateTensor(
        view_dims, rank, dt, dst_strides, dst_off,
        aclFormat::ACL_FORMAT_ND, &dst_storage, 1, target->mem_ptr);
    if (!dst_t) return false;
    aclTensor* src_t = aclCreateTensor(
        view_dims, rank, dt, src_strides, src_off,
        aclFormat::ACL_FORMAT_ND, &src_storage, 1, source->mem_ptr);
    if (!src_t) {
        aclDestroyTensor(dst_t);
        return false;
    }
    uint64_t ws_size = 0;
    aclOpExecutor* exec = nullptr;
    auto st = aclnnInplaceCopyGetWorkspaceSize(dst_t, src_t, &ws_size, &exec);
    if (st != ACL_SUCCESS) {
        aclDestroyTensor(src_t);
        aclDestroyTensor(dst_t);
        return false;
    }
    void* ws = ws_size ? mallocWorkSpace(ws_size) : nullptr;
    st = aclnnInplaceCopy(ws, ws_size, exec, aclstream);
    aclDestroyTensor(src_t);
    aclDestroyTensor(dst_t);
    USER_CHECK(st == ACL_SUCCESS)
        << "ACL strided InplaceCopy failed, status:" << st << acl_error_to_string(st);
    return true;
}

bool try_strided_inplace_add(Var* source, Var* target, const CopyPlan& plan) {
    if (!plan.elements) return true;
    const size_t dsize = target->dtype().dsize();
    if (!dsize || (plan.block_bytes % dsize) != 0
            || (plan.source_offset % dsize) != 0
            || (plan.target_offset % dsize) != 0)
        return false;
    const size_t rank = plan.shape.size() + 1;
    if (rank > 16) return false;
    const NanoString ns = target->dtype();
    if (!is_inplace_add_dtype(ns))
        return false;
    int64_t view_dims[16];
    int64_t src_strides[16];
    int64_t dst_strides[16];
    for (size_t i = 0; i < plan.shape.size(); ++i) {
        if ((plan.source_strides[i] % dsize) != 0 || (plan.target_strides[i] % dsize) != 0)
            return false;
        if (plan.shape[i] > 1 && plan.target_strides[i] == 0)
            return false;
        view_dims[i] = static_cast<int64_t>(plan.shape[i]);
        src_strides[i] = static_cast<int64_t>(plan.source_strides[i] / dsize);
        dst_strides[i] = static_cast<int64_t>(plan.target_strides[i] / dsize);
    }
    view_dims[rank - 1] = static_cast<int64_t>(plan.block_bytes / dsize);
    src_strides[rank - 1] = 1;
    dst_strides[rank - 1] = 1;

    const aclDataType dt = get_dtype(ns);
    const int64_t src_storage = static_cast<int64_t>(source->size / dsize);
    const int64_t dst_storage = static_cast<int64_t>(target->size / dsize);
    const int64_t src_off = static_cast<int64_t>(plan.source_offset / dsize);
    const int64_t dst_off = static_cast<int64_t>(plan.target_offset / dsize);

    aclTensor* dst_t = aclCreateTensor(
        view_dims, rank, dt, dst_strides, dst_off,
        aclFormat::ACL_FORMAT_ND, &dst_storage, 1, target->mem_ptr);
    if (!dst_t) return false;
    aclTensor* src_t = aclCreateTensor(
        view_dims, rank, dt, src_strides, src_off,
        aclFormat::ACL_FORMAT_ND, &src_storage, 1, source->mem_ptr);
    if (!src_t) {
        aclDestroyTensor(dst_t);
        return false;
    }
    aclScalar* alpha = nullptr;
    if (ns == ns_float32 || ns == ns_float16 || ns == ns_bfloat16) {
        float one = 1.0f;
        alpha = aclCreateScalar(&one, aclDataType::ACL_FLOAT);
    } else {
        int64_t one = 1;
        alpha = aclCreateScalar(&one, aclDataType::ACL_INT64);
    }
    if (!alpha) {
        aclDestroyTensor(src_t);
        aclDestroyTensor(dst_t);
        return false;
    }
    uint64_t ws_size = 0;
    aclOpExecutor* exec = nullptr;
    auto st = aclnnInplaceAddGetWorkspaceSize(dst_t, src_t, alpha, &ws_size, &exec);
    if (st != ACL_SUCCESS) {
        aclDestroyScalar(alpha);
        aclDestroyTensor(src_t);
        aclDestroyTensor(dst_t);
        return false;
    }
    void* ws = ws_size ? mallocWorkSpace(ws_size) : nullptr;
    st = aclnnInplaceAdd(ws, ws_size, exec, aclstream);
    aclDestroyScalar(alpha);
    aclDestroyTensor(src_t);
    aclDestroyTensor(dst_t);
    USER_CHECK(st == ACL_SUCCESS)
        << "ACL strided InplaceAdd failed, status:" << st << acl_error_to_string(st);
    return true;
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
    if (!prepared.skip_selection) {
        if (try_strided_inplace_copy(op->input(0), op->output(0), prepared.copies)) {
            copied = true;
        } else {
            acl_indexing::for_each_copy(prepared.copies, [&](size_t source, size_t target, size_t bytes) {
                copy_region(op->input(0), op->output(0), source, target, bytes);
                copied = true;
            });
        }
    }
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
    if (!prepared.skip_selection) {
        if (op->op == ns_add) {
            USER_CHECK(try_strided_inplace_add(op->input(1), op->output(0), prepared.copies))
                << "Unsupported native ACL setitem add";
            copied = true;
        } else if (try_strided_inplace_copy(op->input(1), op->output(0), prepared.copies)) {
            copied = true;
        } else {
            acl_indexing::for_each_copy(prepared.copies, [&](size_t source, size_t target, size_t bytes) {
                copy_region(op->input(1), op->output(0), source, target, bytes);
                copied = true;
            });
        }
    }
    finish_copies(op->output(0), copied);
}

} // namespace jittor
