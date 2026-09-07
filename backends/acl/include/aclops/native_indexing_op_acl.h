#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace jittor {
struct Op;

std::string acl_getitem_unsupported_reason(Op* op);
std::string acl_setitem_unsupported_reason(Op* op);
void exec_native_acl_getitem(Op* op);
void exec_native_acl_setitem(Op* op);

namespace acl_indexing {

// Byte-copy plans for typed native basic indexing. Tensor/expression indices,
// negative steps and reductions remain explicit unsupported variants. Scalar
// broadcasts retain correctness through repeated device copies, not a tuned kernel.
enum class SliceKind { Full, Integer, Range, NewAxis, Ellipsis, Advanced, Expression };

struct SliceSpec {
    SliceKind kind = SliceKind::Full;
    int64_t start = 0;
    int64_t stop = 0;
    int64_t step = 1;
};

struct Selection {
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t offset = 0;
    size_t elements = 0;
    size_t storage_bytes = 0;
    size_t element_bytes = 0;
};

struct CopyPlan {
    std::vector<size_t> shape;
    std::vector<size_t> source_strides;
    std::vector<size_t> target_strides;
    size_t source_offset = 0;
    size_t target_offset = 0;
    size_t source_end = 0;
    size_t target_end = 0;
    size_t block_bytes = 0;
    size_t elements = 0;
};

inline bool multiply(size_t a, size_t b, size_t& result) {
    if (b && a > std::numeric_limits<size_t>::max() / b) return false;
    result = a * b;
    return true;
}

inline bool add(size_t a, size_t b, size_t& result) {
    if (a > std::numeric_limits<size_t>::max() - b) return false;
    result = a + b;
    return true;
}

inline bool product(const std::vector<size_t>& shape, size_t& result) {
    if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
        result = 0;
        return true;
    }
    result = 1;
    for (auto dim : shape) if (!multiply(result, dim, result)) return false;
    return true;
}

inline bool dense_strides(const std::vector<size_t>& shape, size_t bytes,
                          std::vector<size_t>& strides) {
    strides.resize(shape.size());
    for (size_t axis = shape.size(); axis-- > 0;) {
        strides[axis] = bytes;
        if (!multiply(bytes, shape[axis], bytes)) return false;
    }
    return true;
}

inline std::string make_selection(const std::vector<size_t>& input_shape,
                                  const std::vector<SliceSpec>& slices,
                                  size_t element_bytes, Selection& result) {
    result = Selection{};
    result.element_bytes = element_bytes;
    if (!element_bytes) return "zero dtype width";
    size_t input_elements;
    if (!product(input_shape, input_elements) ||
        !multiply(input_elements, element_bytes, result.storage_bytes))
        return "input byte size overflow";
    std::vector<size_t> input_strides(input_shape.size(), 0);
    if (input_elements && !dense_strides(input_shape, element_bytes, input_strides))
        return "input stride overflow";
    size_t consumed = 0, ellipses = 0;
    for (const auto& slice : slices) {
        if (slice.kind == SliceKind::Advanced) return "advanced tensor indexing is not implemented";
        if (slice.kind == SliceKind::Expression) return "string indexing is not implemented";
        if (slice.kind == SliceKind::Ellipsis) ++ellipses;
        else if (slice.kind != SliceKind::NewAxis) ++consumed;
        if (slice.kind == SliceKind::Range && slice.step <= 0)
            return "zero or negative slice step is not implemented";
    }
    if (ellipses > 1 || consumed > input_shape.size()) return "invalid slice rank";
    size_t input_axis = 0;
    auto full_axis = [&]() {
        result.shape.push_back(input_shape[input_axis]);
        result.strides.push_back(input_strides[input_axis++]);
    };
    for (const auto& slice : slices) {
        if (slice.kind == SliceKind::NewAxis) {
            result.shape.push_back(1);
            result.strides.push_back(0);
            continue;
        }
        if (slice.kind == SliceKind::Ellipsis) {
            for (size_t i = consumed; i < input_shape.size(); ++i) full_axis();
            continue;
        }
        if (input_axis >= input_shape.size()) return "too many index dimensions";
        const size_t size = input_shape[input_axis];
        if (size > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
            return "dimension does not fit signed indexing";
        const size_t stride = input_strides[input_axis];
        if (slice.kind == SliceKind::Full) {
            full_axis();
            continue;
        }
        int64_t start = slice.start;
        if (slice.kind == SliceKind::Integer && start < 0) start += static_cast<int64_t>(size);
        if (slice.kind == SliceKind::Integer) {
            if (start < 0 || static_cast<size_t>(start) >= size) return "integer index is out of bounds";
        } else {
            if (start < 0 || slice.stop < 0 || static_cast<size_t>(start) > size ||
                static_cast<size_t>(slice.stop) > size) return "slice bounds are not normalized";
            const size_t count = slice.stop <= start ? 0
                : static_cast<size_t>((slice.stop - start - 1) / slice.step + 1);
            size_t step_bytes;
            if (!multiply(static_cast<size_t>(slice.step), stride, step_bytes))
                return "slice stride overflow";
            result.shape.push_back(count);
            result.strides.push_back(step_bytes);
        }
        size_t offset;
        if (!multiply(static_cast<size_t>(start), stride, offset) ||
            !add(result.offset, offset, result.offset)) return "slice offset overflow";
        ++input_axis;
    }
    while (input_axis < input_shape.size()) full_axis();
    if (!product(result.shape, result.elements)) return "selection size overflow";
    return {};
}

inline bool last_byte(size_t offset, const std::vector<size_t>& shape,
                      const std::vector<size_t>& strides, size_t bytes, size_t& end) {
    end = offset;
    for (size_t i = 0; i < shape.size(); ++i) {
        size_t increment;
        if (!multiply(shape[i] - 1, strides[i], increment) || !add(end, increment, end)) return false;
    }
    return add(end, bytes, end);
}

inline std::string finish_plan(CopyPlan& plan, size_t source_bytes, size_t target_bytes) {
    if (!plan.elements) return {};
    if (!last_byte(plan.source_offset, plan.shape, plan.source_strides,
                   plan.block_bytes, plan.source_end) || plan.source_end > source_bytes)
        return "source interval exceeds its allocation";
    if (!last_byte(plan.target_offset, plan.shape, plan.target_strides,
                   plan.block_bytes, plan.target_end) || plan.target_end > target_bytes)
        return "target interval exceeds its allocation";
    // Collapse the contiguous suffix once, before any device operation.
    while (!plan.shape.empty()) {
        const size_t count = plan.shape.back();
        if (count > 1 && (plan.source_strides.back() != plan.block_bytes ||
                          plan.target_strides.back() != plan.block_bytes)) break;
        if (!multiply(plan.block_bytes, count, plan.block_bytes)) return "copy block size overflow";
        plan.shape.pop_back();
        plan.source_strides.pop_back();
        plan.target_strides.pop_back();
    }
    return {};
}

inline std::string make_get_plan(const Selection& selection,
                                 const std::vector<size_t>& output_shape, CopyPlan& plan) {
    if (selection.shape != output_shape) return "getitem output shape differs from its selection";
    plan = CopyPlan{};
    plan.shape = selection.shape;
    plan.source_strides = selection.strides;
    plan.source_offset = selection.offset;
    plan.block_bytes = selection.element_bytes;
    plan.elements = selection.elements;
    size_t output_bytes;
    if (!multiply(plan.elements, plan.block_bytes, output_bytes)) return "getitem byte size overflow";
    if (!plan.elements) return {};
    if (!dense_strides(output_shape, plan.block_bytes, plan.target_strides)) return "output stride overflow";
    return finish_plan(plan, selection.storage_bytes, output_bytes);
}

inline std::string make_set_plan(const Selection& selection,
                                 const std::vector<size_t>& value_shape, CopyPlan& plan) {
    if (value_shape.size() > selection.shape.size()) return "assignment rank is not broadcastable";
    plan = CopyPlan{};
    plan.shape = selection.shape;
    plan.target_strides = selection.strides;
    plan.target_offset = selection.offset;
    plan.source_strides.resize(plan.shape.size(), 0);
    plan.block_bytes = selection.element_bytes;
    plan.elements = selection.elements;
    size_t value_elements, value_bytes;
    if (!product(value_shape, value_elements) ||
        !multiply(value_elements, plan.block_bytes, value_bytes)) return "assignment byte size overflow";
    std::vector<size_t> value_strides;
    if (value_elements && !dense_strides(value_shape, plan.block_bytes, value_strides))
        return "assignment stride overflow";
    const size_t leading = plan.shape.size() - value_shape.size();
    for (size_t i = 0; i < value_shape.size(); ++i) {
        if (value_shape[i] != 1 && value_shape[i] != plan.shape[leading + i])
            return "assignment shape is not broadcastable";
        if (plan.elements && value_shape[i] != 1) plan.source_strides[leading + i] = value_strides[i];
    }
    return finish_plan(plan, value_bytes, selection.storage_bytes);
}

inline bool address_end(uintptr_t base, size_t offset, uintptr_t& result) {
    if (base > std::numeric_limits<uintptr_t>::max() - offset) return false;
    result = base + offset;
    return true;
}

inline bool identical_mapping(const CopyPlan& plan, uintptr_t source, uintptr_t target) {
    uintptr_t left, right;
    if (!address_end(source, plan.source_offset, left) ||
        !address_end(target, plan.target_offset, right) || left != right) return false;
    for (size_t i = 0; i < plan.shape.size(); ++i)
        if (plan.shape[i] > 1 && plan.source_strides[i] != plan.target_strides[i]) return false;
    return true;
}

inline bool overlaps(uintptr_t left, size_t left_begin, size_t left_end,
                     uintptr_t right, size_t right_begin, size_t right_end) {
    uintptr_t lb, le, rb, re;
    if (!address_end(left, left_begin, lb) || !address_end(left, left_end, le) ||
        !address_end(right, right_begin, rb) || !address_end(right, right_end, re)) return true;
    return lb < re && rb < le;
}

template<class Copy>
inline void for_each_copy(const CopyPlan& plan, Copy copy) {
    if (!plan.elements) return;
    std::vector<size_t> indices(plan.shape.size(), 0);
    while (true) {
        size_t source = plan.source_offset, target = plan.target_offset;
        for (size_t i = 0; i < indices.size(); ++i) {
            source += indices[i] * plan.source_strides[i];
            target += indices[i] * plan.target_strides[i];
        }
        copy(source, target, plan.block_bytes);
        size_t axis = indices.size();
        while (axis && ++indices[axis - 1] == plan.shape[axis - 1]) indices[--axis] = 0;
        if (!axis) break;
    }
}

} // namespace acl_indexing
} // namespace jittor
