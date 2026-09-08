#pragma once
#include "acl_jittor.h"
#include "acl_code_data.h"
#include <memory>

namespace jittor {

// The schema is independent of tensor storage and CANN descriptor creation.
// Values come from CodeOp::data; generated source only names the runner.
inline acl_data::AclAttrSchema acl_code_attribute_schema(const string& name) {
    using Type = acl_data::AclDataType;
    auto required = [](Type type) {
        acl_data::AclAttrField field;
        field.type = type;
        return field;
    };
    if (name == "Softmax" || name == "SoftmaxBackward" ||
        name == "Cumsum" || name == "Gather")
        return {{"dim", required(Type::int64)}};
    if (name == "Triu") return {{"diagonal", required(Type::int64)}};
    if (name == "Flip") return {{"axes", required(Type::int64_vector)}};
    if (name == "Scatter")
        return {{"axis", required(Type::int64)}, {"reduction", required(Type::int64)}};
    acl_data::internal_error("ACL runner has no attribute schema: " + name);
    return {};
}

template<class Runner, class Map>
void apply_acl_code_attributes(Runner& op, const Map& data) {
    const auto decoded = acl_data::decode_code_data(data, op.name, acl_code_attribute_schema(op.name));
    const auto& fields = decoded.fields;
    if (op.name == "Softmax" || op.name == "SoftmaxBackward") {
        auto attr = std::unique_ptr<SoftmaxAttr>(new SoftmaxAttr());
        attr->dim = fields.at("dim").int_value;
        op.op_attr = std::move(attr);
        op.jt_name = "softmax";
    } else if (op.name == "Triu") {
        auto attr = std::unique_ptr<TriuAttr>(new TriuAttr());
        attr->diagonal = fields.at("diagonal").int_value;
        op.op_attr = std::move(attr);
        op.jt_name = "triu";
    } else if (op.name == "Flip") {
        auto attr = std::unique_ptr<ReduceAttr>(new ReduceAttr());
        attr->axes = fields.at("axes").int_values;
        attr->prod_dim = attr->axes.size();
        attr->keepdims = false;
        op.op_attr = std::move(attr);
        op.jt_name = "flip";
    } else if (op.name == "Cumsum" || op.name == "Gather") {
        auto attr = std::unique_ptr<GatherAttr>(new GatherAttr());
        attr->dim = fields.at("dim").int_value;
        op.op_attr = std::move(attr);
        op.jt_name = op.name == "Cumsum" ? "cumsum" : "gather";
    } else if (op.name == "Scatter") {
        auto attr = std::unique_ptr<ScatterAttr>(new ScatterAttr());
        attr->axis = fields.at("axis").int_value;
        attr->reduction = fields.at("reduction").int_value;
        op.op_attr = std::move(attr);
        op.jt_name = "scatter";
    }
}

} // namespace jittor
