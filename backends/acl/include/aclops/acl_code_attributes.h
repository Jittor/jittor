#pragma once
#include "acl_jittor.h"
#include "acl_code_data.h"
#include <memory>
namespace jittor {
inline string acl_code_text(const acl_data::AclDataValue& value, bool layout) {
    string text;
    for (auto byte : value.int_values) {
        if (byte < 0 || byte > 255) acl_data::user_error("ACL text attribute byte out of range");
        text.push_back(static_cast<char>(byte));
    }
    if (layout ? (text != "BSH" && text != "SBH" && text != "BSND" && text != "BNSD") : text != "ReLU")
        acl_data::user_error("unsupported ACL text attribute: " + text);
    return text;
}
template<class Runner> auto acl_set_cube(Runner& op, int64_t value, int)
    -> decltype(op.cube_math_type = 0, void()) {
    if (value < 0 || value > 3) acl_data::user_error("invalid cube math type");
    op.cube_math_type = static_cast<int>(value);
}
template<class Runner> void acl_set_cube(Runner&, int64_t, long) {
    acl_data::internal_error("runner has no cube math field");
}
template<class Runner> auto acl_set_roll(Runner& op, const vector<int64_t>& shifts, const vector<int64_t>& dims, int)
    -> decltype(op.shifts = shifts, op.dims = dims, void()) { op.shifts = shifts; op.dims = dims; }
template<class Runner> void acl_set_roll(Runner&, const vector<int64_t>&, const vector<int64_t>&, long) {
    acl_data::internal_error("runner has no roll fields");
}
inline acl_data::AclAttrSchema acl_code_attribute_schema(const string& name) {
    using Type = acl_data::AclDataType;
    auto required = [](Type type) { acl_data::AclAttrField field; field.type=type; return field; };
    auto defaulted = [](acl_data::AclDataValue value) { acl_data::AclAttrField field; field.type=value.type;
        field.required=false; field.has_default=true; field.default_value=std::move(value); return field; };

    if (name == "Softmax") return {{"dim", required(Type::int64)}};
    if (name == "SoftmaxBackward") return {{"dim", required(Type::int64)}};
    if (name == "Triu") return {{"diagonal", required(Type::int64)}};
    if (name == "Flip") return {{"axes", required(Type::int64_vector)}};
    if (name == "Transpose") return {{"axes", required(Type::int64_vector)}};
    if (name == "Cumsum") return {{"dim", required(Type::int64)}};
    if (name == "Gather") return {{"dim", required(Type::int64)}};
    if (name == "Scatter") return {{"axis", required(Type::int64)}, {"reduction", required(Type::int64)}};
    if (name == "Conv2d") return {{"convStrides", required(Type::int64_vector)}, {"convPads", required(Type::int64_vector)}, {"convDilations", required(Type::int64_vector)}, {"group", required(Type::int64)}, {"convOutPads", required(Type::int64_vector)}, {"cube_math_type", required(Type::int64)}};
    if (name == "Conv2dBackward") return {{"convStrides", required(Type::int64_vector)}, {"convPads", required(Type::int64_vector)}, {"convDilations", required(Type::int64_vector)}, {"group", required(Type::int64)}, {"convOutPads", required(Type::int64_vector)}, {"cube_math_type", required(Type::int64)}};
    if (name == "BatchNorm") return {{"is_train", required(Type::boolean)}, {"momentum", required(Type::float64)}, {"eps", required(Type::float64)}};
    if (name == "BatchNormBackward") return {{"is_train", required(Type::boolean)}, {"momentum", required(Type::float64)}, {"eps", required(Type::float64)}};
    if (name == "LayerNorm") return {{"eps", required(Type::float64)}, {"normalizedShape", required(Type::int64_vector)}};
    if (name == "LayerNormBackward") return {{"eps", required(Type::float64)}, {"normalizedShape", required(Type::int64_vector)}};
    if (name == "GroupNorm") return {{"batch", required(Type::int64)}, {"channels", required(Type::int64)}, {"spatialSize", required(Type::int64)}, {"groups", required(Type::int64)}, {"eps", required(Type::float64)}};
    if (name == "GroupNormBackward") return {{"batch", required(Type::int64)}, {"channels", required(Type::int64)}, {"spatialSize", required(Type::int64)}, {"groups", required(Type::int64)}, {"eps", required(Type::float64)}};
    if (name == "RmsNorm") return {{"eps", required(Type::float64)}};
    if (name == "Maxpool") return {{"kernel_size", required(Type::int64_vector)}, {"poolStrides", required(Type::int64_vector)}, {"poolPads", required(Type::int64_vector)}, {"poolDilations", required(Type::int64_vector)}, {"poolCeil", required(Type::boolean)}, {"countIncludePad", required(Type::boolean)}};
    if (name == "Avgpool") return {{"kernel_size", required(Type::int64_vector)}, {"poolStrides", required(Type::int64_vector)}, {"poolPads", required(Type::int64_vector)}, {"poolDilations", required(Type::int64_vector)}, {"poolCeil", required(Type::boolean)}, {"countIncludePad", required(Type::boolean)}};
    if (name == "MaxpoolBackward") return {{"kernel_size", required(Type::int64_vector)}, {"poolStrides", required(Type::int64_vector)}, {"poolPads", required(Type::int64_vector)}, {"poolDilations", required(Type::int64_vector)}, {"poolCeil", required(Type::boolean)}, {"countIncludePad", required(Type::boolean)}};
    if (name == "AvgpoolBackward") return {{"kernel_size", required(Type::int64_vector)}, {"poolStrides", required(Type::int64_vector)}, {"poolPads", required(Type::int64_vector)}, {"poolDilations", required(Type::int64_vector)}, {"poolCeil", required(Type::boolean)}, {"countIncludePad", required(Type::boolean)}};
    if (name == "UpsampleNearest2d") return {{"outputSize", required(Type::int64_vector)}, {"inputSize", required(Type::int64_vector)}};
    if (name == "UpsampleNearest2dBackward") return {{"outputSize", required(Type::int64_vector)}, {"inputSize", required(Type::int64_vector)}};
    if (name == "Concat") return {{"tensorNum", required(Type::int64)}, {"dim", required(Type::int64)}};
    if (name == "Stack") return {{"tensorNum", required(Type::int64)}, {"dim", required(Type::int64)}};
    if (name == "SplitWithSize") return {{"splitSize", required(Type::int64_vector)}, {"dim", required(Type::int64)}};
    if (name == "SliceV2") return {{"begins", required(Type::int64_vector)}, {"ends", required(Type::int64_vector)}, {"steps", required(Type::int64_vector)}, {"axes", required(Type::int64_vector)}};
    if (name == "StridedSliceAssign") return {{"begins", required(Type::int64_vector)}, {"ends", required(Type::int64_vector)}, {"steps", required(Type::int64_vector)}, {"axes", required(Type::int64_vector)}};
    if (name == "StridedSliceAssignV2") return {{"begins", required(Type::int64_vector)}, {"ends", required(Type::int64_vector)}, {"steps", required(Type::int64_vector)}, {"axes", required(Type::int64_vector)}};
    if (name == "Range") return {{"start", required(Type::int64)}, {"end", required(Type::int64)}, {"step", required(Type::int64)}};
    if (name == "LeakyReLU") return {{"negativeSlope", required(Type::float64)}, {"selfIsResult", required(Type::boolean)}};
    if (name == "LeakyReLUBackward") return {{"negativeSlope", required(Type::float64)}, {"selfIsResult", required(Type::boolean)}};
    if (name == "Dropout") return {{"p", required(Type::float64)}, {"train", required(Type::boolean)}, {"seed", required(Type::int64)}, {"offset", required(Type::int64)}};
    if (name == "DropoutBackward") return {{"scale", required(Type::float64)}};
    if (name == "EmbeddingBackward") return {{"numEmbeddings", required(Type::int64)}, {"paddingIdx", required(Type::int64)}, {"scaleGradByFreq", required(Type::boolean)}};
    if (name == "NanToNum") return {{"nan", required(Type::float64)}, {"posinf", required(Type::float64)}, {"neginf", required(Type::float64)}};
    if (name == "FlashAttention") return {{"scale", required(Type::float64)}, {"keepProb", required(Type::float64)}, {"preToken", required(Type::int64)}, {"nextToken", required(Type::int64)}, {"headNum", required(Type::int64)}, {"inputLayout", required(Type::int64_vector)}, {"innerPrecise", required(Type::int64)}, {"sparseMode", required(Type::int64)}, {"psetype", required(Type::int64)}, {"prefix", required(Type::int64_vector)}, {"qStartIdx", required(Type::int64_vector)}, {"kvStartIdx", required(Type::int64_vector)}, {"hasRealshift", required(Type::boolean)}, {"hasDropmask", required(Type::boolean)}, {"hasPaddingmask", required(Type::boolean)}, {"hasAttentmask", required(Type::boolean)}};
    if (name == "FlashAttentionBackward") return {{"scale", required(Type::float64)}, {"keepProb", required(Type::float64)}, {"preToken", required(Type::int64)}, {"nextToken", required(Type::int64)}, {"headNum", required(Type::int64)}, {"inputLayout", required(Type::int64_vector)}, {"innerPrecise", required(Type::int64)}, {"sparseMode", required(Type::int64)}, {"psetype", required(Type::int64)}, {"prefix", required(Type::int64_vector)}, {"qStartIdx", required(Type::int64_vector)}, {"kvStartIdx", required(Type::int64_vector)}, {"hasRealshift", required(Type::boolean)}, {"hasDropmask", required(Type::boolean)}, {"hasPaddingmask", required(Type::boolean)}, {"hasAttentmask", required(Type::boolean)}};
    if (name == "IncreFlashAttention") return {{"scale", required(Type::float64)}, {"headNum", required(Type::int64)}, {"keyValueHeadNum", required(Type::int64)}, {"inputLayout", required(Type::int64_vector)}, {"innerPrecise", required(Type::int64)}, {"blockSize", defaulted(acl_data::AclDataValue::int64_value(0))}, {"hasBlockTable", defaulted(acl_data::AclDataValue::bool_value_of(false))}, {"actualSeqLengths", defaulted(acl_data::AclDataValue::int64_vector({}))}};
    if (name == "KVCacheMemcpy") return {{"blockSize", required(Type::int64)}, {"slots", required(Type::int64_vector)}};
    if (name == "Roll") return {{"shifts", required(Type::int64_vector)}, {"dims", required(Type::int64_vector)}};
    if (name == "MatMul") return {{"mode", required(Type::int64)}, {"cube_math_type", required(Type::int64)}};
    if (name == "BatchMatMul") return {{"mode", required(Type::int64)}, {"cube_math_type", required(Type::int64)}};
    if (name == "Unary") return {{"operation", required(Type::int64_vector)}};
    if (name == "TruthReduce") return {{"axes", required(Type::int64_vector)}, {"keepdims", required(Type::boolean)}, {"reduce_all", required(Type::boolean)}};
    acl_data::internal_error("ACL runner has no attribute schema: " + name); return {};
}
template<class Runner>
void assign_acl_code_attributes(Runner& op, const acl_data::AclDecodedData& decoded) {
    const auto& name=decoded.op;
    const auto& fields=decoded.fields;
    if (op.name != name && !(name == "Unary" && op.name == "unary") &&
        !(name == "TruthReduce" && (op.name == "All" || op.name == "Any")))
        acl_data::internal_error("attribute schema does not belong to actual runner: " + op.name + " / " + name);

    if (name == "Softmax") {
        auto attr=std::unique_ptr<SoftmaxAttr>(new SoftmaxAttr());
        attr->dim = fields.at("dim").int_value;
        op.op_attr=std::move(attr);
        op.jt_name="softmax";
        return;
    }
    if (name == "SoftmaxBackward") {
        auto attr=std::unique_ptr<SoftmaxAttr>(new SoftmaxAttr());
        attr->dim = fields.at("dim").int_value;
        op.op_attr=std::move(attr);
        op.jt_name="softmax";
        return;
    }
    if (name == "Triu") {
        auto attr=std::unique_ptr<TriuAttr>(new TriuAttr());
        attr->diagonal = fields.at("diagonal").int_value;
        op.op_attr=std::move(attr);
        op.jt_name="triu";
        return;
    }
    if (name == "Flip") {
        auto attr=std::unique_ptr<ReduceAttr>(new ReduceAttr());
        attr->axes = fields.at("axes").int_values;
        attr->prod_dim=attr->axes.size();
        attr->keepdims=false;
        op.op_attr=std::move(attr);
        op.jt_name="flip";
        return;
    }
    if (name == "Transpose") {
        auto attr=std::unique_ptr<ReduceAttr>(new ReduceAttr());
        attr->axes = fields.at("axes").int_values;
        attr->prod_dim=attr->axes.size();
        attr->keepdims=false;
        op.op_attr=std::move(attr);
        op.jt_name="transpose";
        return;
    }
    if (name == "Cumsum") {
        auto attr=std::unique_ptr<GatherAttr>(new GatherAttr());
        attr->dim = fields.at("dim").int_value;
        op.op_attr=std::move(attr);
        op.jt_name="cumsum";
        return;
    }
    if (name == "Gather") {
        auto attr=std::unique_ptr<GatherAttr>(new GatherAttr());
        attr->dim = fields.at("dim").int_value;
        op.op_attr=std::move(attr);
        op.jt_name="gather";
        return;
    }
    if (name == "Scatter") {
        auto attr=std::unique_ptr<ScatterAttr>(new ScatterAttr());
        attr->axis = fields.at("axis").int_value;
        attr->reduction = fields.at("reduction").int_value;
        op.op_attr=std::move(attr);
        op.jt_name="scatter";
        return;
    }
    if (name == "Conv2d") {
        auto attr=std::unique_ptr<ConvAttr>(new ConvAttr());
        attr->convStrides = fields.at("convStrides").int_values;
        attr->convPads = fields.at("convPads").int_values;
        attr->convDilations = fields.at("convDilations").int_values;
        attr->group = fields.at("group").int_value;
        attr->convOutPads = fields.at("convOutPads").int_values;
        acl_set_cube(op, fields.at("cube_math_type").int_value, 0);
        op.op_attr=std::move(attr);
        op.jt_name="conv2d";
        return;
    }
    if (name == "Conv2dBackward") {
        auto attr=std::unique_ptr<ConvAttr>(new ConvAttr());
        attr->convStrides = fields.at("convStrides").int_values;
        attr->convPads = fields.at("convPads").int_values;
        attr->convDilations = fields.at("convDilations").int_values;
        attr->group = fields.at("group").int_value;
        attr->convOutPads = fields.at("convOutPads").int_values;
        acl_set_cube(op, fields.at("cube_math_type").int_value, 0);
        op.op_attr=std::move(attr);
        op.jt_name="conv2dbackward";
        return;
    }
    if (name == "BatchNorm") {
        auto attr=std::unique_ptr<BatchNormAttr>(new BatchNormAttr());
        attr->is_train = fields.at("is_train").bool_value;
        attr->momentum = fields.at("momentum").float_value;
        attr->eps = fields.at("eps").float_value;
        op.op_attr=std::move(attr);
        op.jt_name="batchnorm";
        return;
    }
    if (name == "BatchNormBackward") {
        auto attr=std::unique_ptr<BatchNormAttr>(new BatchNormAttr());
        attr->is_train = fields.at("is_train").bool_value;
        attr->momentum = fields.at("momentum").float_value;
        attr->eps = fields.at("eps").float_value;
        op.op_attr=std::move(attr);
        op.jt_name="batchnormbackward";
        return;
    }
    if (name == "LayerNorm") {
        auto attr=std::unique_ptr<LayerNormAttr>(new LayerNormAttr());
        attr->eps = fields.at("eps").float_value;
        attr->normalizedShape = fields.at("normalizedShape").int_values;
        attr->size=attr->normalizedShape.size();
        op.op_attr=std::move(attr);
        op.jt_name="layernorm";
        return;
    }
    if (name == "LayerNormBackward") {
        auto attr=std::unique_ptr<LayerNormAttr>(new LayerNormAttr());
        attr->eps = fields.at("eps").float_value;
        attr->normalizedShape = fields.at("normalizedShape").int_values;
        attr->size=attr->normalizedShape.size();
        op.op_attr=std::move(attr);
        op.jt_name="layernormbackward";
        return;
    }
    if (name == "GroupNorm") {
        auto attr=std::unique_ptr<GroupNormAttr>(new GroupNormAttr());
        attr->batch = fields.at("batch").int_value;
        attr->channels = fields.at("channels").int_value;
        attr->spatialSize = fields.at("spatialSize").int_value;
        attr->groups = fields.at("groups").int_value;
        attr->eps = fields.at("eps").float_value;
        op.op_attr=std::move(attr);
        op.jt_name="groupnorm";
        return;
    }
    if (name == "GroupNormBackward") {
        auto attr=std::unique_ptr<GroupNormAttr>(new GroupNormAttr());
        attr->batch = fields.at("batch").int_value;
        attr->channels = fields.at("channels").int_value;
        attr->spatialSize = fields.at("spatialSize").int_value;
        attr->groups = fields.at("groups").int_value;
        attr->eps = fields.at("eps").float_value;
        op.op_attr=std::move(attr);
        op.jt_name="groupnorm";
        return;
    }
    if (name == "RmsNorm") {
        auto attr=std::unique_ptr<RmsNormAttr>(new RmsNormAttr());
        attr->eps = fields.at("eps").float_value;
        op.op_attr=std::move(attr);
        op.jt_name="rmsnorm";
        return;
    }
    if (name == "Maxpool") {
        auto attr=std::unique_ptr<PoolAttr>(new PoolAttr());
        attr->kernel_size = fields.at("kernel_size").int_values;
        attr->poolStrides = fields.at("poolStrides").int_values;
        attr->poolPads = fields.at("poolPads").int_values;
        attr->poolDilations = fields.at("poolDilations").int_values;
        attr->poolCeil = fields.at("poolCeil").bool_value;
        attr->countIncludePad = fields.at("countIncludePad").bool_value;
        op.op_attr=std::move(attr);
        op.jt_name="maxpool";
        return;
    }
    if (name == "Avgpool") {
        auto attr=std::unique_ptr<PoolAttr>(new PoolAttr());
        attr->kernel_size = fields.at("kernel_size").int_values;
        attr->poolStrides = fields.at("poolStrides").int_values;
        attr->poolPads = fields.at("poolPads").int_values;
        attr->poolDilations = fields.at("poolDilations").int_values;
        attr->poolCeil = fields.at("poolCeil").bool_value;
        attr->countIncludePad = fields.at("countIncludePad").bool_value;
        op.op_attr=std::move(attr);
        op.jt_name="avgpool";
        return;
    }
    if (name == "MaxpoolBackward") {
        auto attr=std::unique_ptr<PoolAttr>(new PoolAttr());
        attr->kernel_size = fields.at("kernel_size").int_values;
        attr->poolStrides = fields.at("poolStrides").int_values;
        attr->poolPads = fields.at("poolPads").int_values;
        attr->poolDilations = fields.at("poolDilations").int_values;
        attr->poolCeil = fields.at("poolCeil").bool_value;
        attr->countIncludePad = fields.at("countIncludePad").bool_value;
        op.op_attr=std::move(attr);
        op.jt_name="maxpoolbackward";
        return;
    }
    if (name == "AvgpoolBackward") {
        auto attr=std::unique_ptr<PoolAttr>(new PoolAttr());
        attr->kernel_size = fields.at("kernel_size").int_values;
        attr->poolStrides = fields.at("poolStrides").int_values;
        attr->poolPads = fields.at("poolPads").int_values;
        attr->poolDilations = fields.at("poolDilations").int_values;
        attr->poolCeil = fields.at("poolCeil").bool_value;
        attr->countIncludePad = fields.at("countIncludePad").bool_value;
        op.op_attr=std::move(attr);
        op.jt_name="avgpoolbackward";
        return;
    }
    if (name == "UpsampleNearest2d") {
        auto attr=std::unique_ptr<UpsampleNearest2dAttr>(new UpsampleNearest2dAttr());
        attr->outputSize = fields.at("outputSize").int_values;
        attr->inputSize = fields.at("inputSize").int_values;
        op.op_attr=std::move(attr);
        op.jt_name="upsample_nearest2d";
        return;
    }
    if (name == "UpsampleNearest2dBackward") {
        auto attr=std::unique_ptr<UpsampleNearest2dAttr>(new UpsampleNearest2dAttr());
        attr->outputSize = fields.at("outputSize").int_values;
        attr->inputSize = fields.at("inputSize").int_values;
        op.op_attr=std::move(attr);
        op.jt_name="upsample_nearest2d";
        return;
    }
    if (name == "Concat") {
        auto attr=std::unique_ptr<ConcatAttr>(new ConcatAttr());
        attr->tensorNum = fields.at("tensorNum").int_value;
        attr->dim = fields.at("dim").int_value;
        op.op_attr=std::move(attr);
        op.jt_name="concat";
        return;
    }
    if (name == "Stack") {
        auto attr=std::unique_ptr<ConcatAttr>(new ConcatAttr());
        attr->tensorNum = fields.at("tensorNum").int_value;
        attr->dim = fields.at("dim").int_value;
        op.op_attr=std::move(attr);
        op.jt_name="stack";
        return;
    }
    if (name == "SplitWithSize") {
        auto attr=std::unique_ptr<SplitWithSizeAttr>(new SplitWithSizeAttr());
        attr->splitSize = fields.at("splitSize").int_values;
        attr->dim = fields.at("dim").int_value;
        op.op_attr=std::move(attr);
        op.jt_name="splitwithsize";
        return;
    }
    if (name == "SliceV2") {
        auto attr=std::unique_ptr<StrideAttr>(new StrideAttr());
        attr->begins = fields.at("begins").int_values;
        attr->ends = fields.at("ends").int_values;
        attr->steps = fields.at("steps").int_values;
        attr->axes = fields.at("axes").int_values;
        op.op_attr=std::move(attr);
        op.jt_name="slicev2";
        return;
    }
    if (name == "StridedSliceAssign") {
        auto attr=std::unique_ptr<StrideAttr>(new StrideAttr());
        attr->begins = fields.at("begins").int_values;
        attr->ends = fields.at("ends").int_values;
        attr->steps = fields.at("steps").int_values;
        attr->axes = fields.at("axes").int_values;
        op.op_attr=std::move(attr);
        op.jt_name="stridedsliceassign";
        return;
    }
    if (name == "StridedSliceAssignV2") {
        auto attr=std::unique_ptr<StrideAttr>(new StrideAttr());
        attr->begins = fields.at("begins").int_values;
        attr->ends = fields.at("ends").int_values;
        attr->steps = fields.at("steps").int_values;
        attr->axes = fields.at("axes").int_values;
        op.op_attr=std::move(attr);
        // The gradient variant zero-initializes the destination before writing
        // its slice. Keep this structural execution choice separate from data.
        if (op.jt_name != "stridedsliceassignv2_grad")
            op.jt_name="stridedsliceassignv2";
        return;
    }
    if (name == "Range") {
        auto attr=std::unique_ptr<RangeAttr>(new RangeAttr());
        attr->start = fields.at("start").int_value;
        attr->end = fields.at("end").int_value;
        attr->step = fields.at("step").int_value;
        op.op_attr=std::move(attr);
        op.jt_name="range";
        return;
    }
    if (name == "LeakyReLU") {
        auto attr=std::unique_ptr<LeakyReluAttr>(new LeakyReluAttr());
        attr->negativeSlope = fields.at("negativeSlope").float_value;
        attr->selfIsResult = fields.at("selfIsResult").bool_value;
        op.op_attr=std::move(attr);
        op.jt_name="leakyrelu";
        return;
    }
    if (name == "LeakyReLUBackward") {
        auto attr=std::unique_ptr<LeakyReluAttr>(new LeakyReluAttr());
        attr->negativeSlope = fields.at("negativeSlope").float_value;
        attr->selfIsResult = fields.at("selfIsResult").bool_value;
        op.op_attr=std::move(attr);
        op.jt_name="leakyrelubackward";
        return;
    }
    if (name == "Dropout") {
        auto attr=std::unique_ptr<DropoutAttr>(new DropoutAttr());
        attr->p = fields.at("p").float_value;
        attr->train = fields.at("train").bool_value;
        attr->seed = fields.at("seed").int_value;
        attr->offset = fields.at("offset").int_value;
        op.op_attr=std::move(attr);
        op.jt_name="dropout";
        return;
    }
    if (name == "DropoutBackward") {
        auto attr=std::unique_ptr<DropoutAttr>(new DropoutAttr());
        attr->scale = fields.at("scale").float_value;
        op.op_attr=std::move(attr);
        op.jt_name="dropoutbackward";
        return;
    }
    if (name == "EmbeddingBackward") {
        auto attr=std::unique_ptr<EmbeddingAttr>(new EmbeddingAttr());
        attr->numEmbeddings = fields.at("numEmbeddings").int_value;
        attr->paddingIdx = fields.at("paddingIdx").int_value;
        attr->scaleGradByFreq = fields.at("scaleGradByFreq").bool_value;
        op.op_attr=std::move(attr);
        op.jt_name="embeddingbackward";
        return;
    }
    if (name == "NanToNum") {
        auto attr=std::unique_ptr<NanToNumAttr>(new NanToNumAttr());
        attr->nan = fields.at("nan").float_value;
        attr->posinf = fields.at("posinf").float_value;
        attr->neginf = fields.at("neginf").float_value;
        op.op_attr=std::move(attr);
        op.jt_name="NanToNum";
        return;
    }
    if (name == "FlashAttention") {
        auto attr=std::unique_ptr<FlashAttentionAttr>(new FlashAttentionAttr());
        attr->scale = fields.at("scale").float_value;
        attr->keepProb = fields.at("keepProb").float_value;
        attr->preToken = fields.at("preToken").int_value;
        attr->nextToken = fields.at("nextToken").int_value;
        attr->headNum = fields.at("headNum").int_value;
        attr->inputLayout = acl_code_text(fields.at("inputLayout"), true);
        attr->innerPrecise = fields.at("innerPrecise").int_value;
        attr->sparseMode = fields.at("sparseMode").int_value;
        attr->psetype = fields.at("psetype").int_value;
        attr->prefix = fields.at("prefix").int_values;
        attr->qStartIdx = fields.at("qStartIdx").int_values;
        attr->kvStartIdx = fields.at("kvStartIdx").int_values;
        attr->hasRealshift = fields.at("hasRealshift").bool_value;
        attr->hasDropmask = fields.at("hasDropmask").bool_value;
        attr->hasPaddingmask = fields.at("hasPaddingmask").bool_value;
        attr->hasAttentmask = fields.at("hasAttentmask").bool_value;
        op.op_attr=std::move(attr);
        op.jt_name="flashattention";
        return;
    }
    if (name == "FlashAttentionBackward") {
        auto attr=std::unique_ptr<FlashAttentionAttr>(new FlashAttentionAttr());
        attr->scale = fields.at("scale").float_value;
        attr->keepProb = fields.at("keepProb").float_value;
        attr->preToken = fields.at("preToken").int_value;
        attr->nextToken = fields.at("nextToken").int_value;
        attr->headNum = fields.at("headNum").int_value;
        attr->inputLayout = acl_code_text(fields.at("inputLayout"), true);
        attr->innerPrecise = fields.at("innerPrecise").int_value;
        attr->sparseMode = fields.at("sparseMode").int_value;
        attr->psetype = fields.at("psetype").int_value;
        attr->prefix = fields.at("prefix").int_values;
        attr->qStartIdx = fields.at("qStartIdx").int_values;
        attr->kvStartIdx = fields.at("kvStartIdx").int_values;
        attr->hasRealshift = fields.at("hasRealshift").bool_value;
        attr->hasDropmask = fields.at("hasDropmask").bool_value;
        attr->hasPaddingmask = fields.at("hasPaddingmask").bool_value;
        attr->hasAttentmask = fields.at("hasAttentmask").bool_value;
        op.op_attr=std::move(attr);
        op.jt_name="flashattentionbackward";
        return;
    }
    if (name == "IncreFlashAttention") {
        auto attr=std::unique_ptr<IncreFlashAttentionAttr>(new IncreFlashAttentionAttr());
        attr->scale = fields.at("scale").float_value;
        attr->headNum = fields.at("headNum").int_value;
        attr->keyValueHeadNum = fields.at("keyValueHeadNum").int_value;
        attr->inputLayout = acl_code_text(fields.at("inputLayout"), true);
        attr->innerPrecise = fields.at("innerPrecise").int_value;
        attr->blockSize = fields.at("blockSize").int_value;
        attr->hasBlockTable = fields.at("hasBlockTable").bool_value;
        attr->actualSeqLengths = fields.at("actualSeqLengths").int_values;
        op.op_attr=std::move(attr);
        op.jt_name="increflashattention";
        if (fields.at("hasBlockTable").bool_value) op.jt_name="paged_increflashattention";
        return;
    }
    if (name == "KVCacheMemcpy") {
        auto attr=std::unique_ptr<KVCacheMemcpyAttr>(new KVCacheMemcpyAttr());
        attr->blockSize = fields.at("blockSize").int_value;
        attr->slots = fields.at("slots").int_values;
        op.op_attr=std::move(attr);
        op.jt_name="kv_cache_memcpy";
        return;
    }
    if (name == "Roll") {
        op.jt_name="roll";
        acl_set_roll(op, fields.at("shifts").int_values, fields.at("dims").int_values, 0);
        return;
    }
    if (name == "MatMul") {
        op.jt_name="matmul";
        auto mode=fields.at("mode").int_value;
        if (mode<0 || mode>2) acl_data::user_error("invalid matmul transpose mode");
        op.jt_name=mode==0 ? "matmul" : mode==1 ? "matmul_trans_1" : "matmul_trans_0";
        acl_set_cube(op, fields.at("cube_math_type").int_value, 0);
        return;
    }
    if (name == "BatchMatMul") {
        op.jt_name="batchmatmul";
        auto mode=fields.at("mode").int_value;
        if (mode<0 || mode>2) acl_data::user_error("invalid matmul transpose mode");
        op.jt_name=mode==0 ? "bmm" : mode==1 ? "bmm_trans_1" : "bmm_trans_0";
        acl_set_cube(op, fields.at("cube_math_type").int_value, 0);
        return;
    }
    if (name == "Unary") {
        op.jt_name="unary";
        op.name=acl_code_text(fields.at("operation"), false);
        return;
    }
    if (name == "TruthReduce") {
        auto attr=std::unique_ptr<ReduceAttr>(new ReduceAttr());
        attr->axes = fields.at("axes").int_values;
        attr->keepdims = fields.at("keepdims").bool_value;
        attr->prod_dim=attr->axes.size();
        op.op_attr=std::move(attr);
        op.jt_name="truthreduce";
        bool all=fields.at("reduce_all").bool_value;
        if (op.name != (all ? "All" : "Any")) acl_data::internal_error("truth reducer construction disagrees with data");
        op.jt_name=all ? "all" : "any";
        return;
    }
    acl_data::internal_error("ACL runner has no attribute schema: " + name);
}
template<class Runner, class Map>
void apply_acl_code_attributes(Runner& op, const Map& data,
                               const string& prefix="acl_attr.", const string& schema_name="") {
    const string name=schema_name.empty() ? op.name : schema_name;
    auto decoded=acl_data::decode_code_data(data, name, acl_code_attribute_schema(name), prefix);
    assign_acl_code_attributes(op, decoded);
}
} // namespace jittor
