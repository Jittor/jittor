"""Schemas for production ACL runners using the CodeOp data channel."""

from .acl_data import AclDataInternalError, SCHEMA_VERSION, encode_code_data


SCHEMAS = {
    "Softmax": {"dim": {"type": "int64"}},
    "SoftmaxBackward": {"dim": {"type": "int64"}},
    "Triu": {"diagonal": {"type": "int64"}},
    "Flip": {"axes": {"type": "int64[]"}},
    "Transpose": {"axes": {"type": "int64[]"}},
    "Cumsum": {"dim": {"type": "int64"}},
    "Gather": {"dim": {"type": "int64"}},
    "Scatter": {"axis": {"type": "int64"}, "reduction": {"type": "int64"}},
    "Conv2d": {
        "convStrides": {"type": "int64[]"},
        "convPads": {"type": "int64[]"},
        "convDilations": {"type": "int64[]"},
        "group": {"type": "int64"},
        "convOutPads": {"type": "int64[]"},
    },
    "Conv2dBackward": {
        "convStrides": {"type": "int64[]"},
        "convPads": {"type": "int64[]"},
        "convDilations": {"type": "int64[]"},
        "group": {"type": "int64"},
        "convOutPads": {"type": "int64[]"},
    },
    "BatchNorm": {
        "is_train": {"type": "bool"},
        "momentum": {"type": "float64"},
        "eps": {"type": "float64"},
    },
    "BatchNormBackward": {
        "is_train": {"type": "bool"},
        "momentum": {"type": "float64"},
        "eps": {"type": "float64"},
    },
    "LayerNorm": {"eps": {"type": "float64"}, "normalizedShape": {"type": "int64[]"}},
    "LayerNormBackward": {"eps": {"type": "float64"}, "normalizedShape": {"type": "int64[]"}},
    "GroupNorm": {
        "batch": {"type": "int64"},
        "channels": {"type": "int64"},
        "spatialSize": {"type": "int64"},
        "groups": {"type": "int64"},
        "eps": {"type": "float64"},
    },
    "GroupNormBackward": {
        "batch": {"type": "int64"},
        "channels": {"type": "int64"},
        "spatialSize": {"type": "int64"},
        "groups": {"type": "int64"},
        "eps": {"type": "float64"},
    },
    "RmsNorm": {"eps": {"type": "float64"}},
    "Maxpool": {
        "kernel_size": {"type": "int64[]"},
        "poolStrides": {"type": "int64[]"},
        "poolPads": {"type": "int64[]"},
        "poolDilations": {"type": "int64[]"},
        "poolCeil": {"type": "bool"},
        "countIncludePad": {"type": "bool"},
    },
    "Avgpool": {
        "kernel_size": {"type": "int64[]"},
        "poolStrides": {"type": "int64[]"},
        "poolPads": {"type": "int64[]"},
        "poolDilations": {"type": "int64[]"},
        "poolCeil": {"type": "bool"},
        "countIncludePad": {"type": "bool"},
    },
    "MaxpoolBackward": {
        "kernel_size": {"type": "int64[]"},
        "poolStrides": {"type": "int64[]"},
        "poolPads": {"type": "int64[]"},
        "poolDilations": {"type": "int64[]"},
        "poolCeil": {"type": "bool"},
        "countIncludePad": {"type": "bool"},
    },
    "AvgpoolBackward": {
        "kernel_size": {"type": "int64[]"},
        "poolStrides": {"type": "int64[]"},
        "poolPads": {"type": "int64[]"},
        "poolDilations": {"type": "int64[]"},
        "poolCeil": {"type": "bool"},
        "countIncludePad": {"type": "bool"},
    },
    "UpsampleNearest2d": {"outputSize": {"type": "int64[]"}, "inputSize": {"type": "int64[]"}},
    "UpsampleNearest2dBackward": {
        "outputSize": {"type": "int64[]"},
        "inputSize": {"type": "int64[]"},
    },
    "Concat": {"tensorNum": {"type": "int64"}, "dim": {"type": "int64"}},
    "Stack": {"tensorNum": {"type": "int64"}, "dim": {"type": "int64"}},
    "SplitWithSize": {"splitSize": {"type": "int64[]"}, "dim": {"type": "int64"}},
    "SliceV2": {
        "begins": {"type": "int64[]"},
        "ends": {"type": "int64[]"},
        "steps": {"type": "int64[]"},
        "axes": {"type": "int64[]"},
    },
    "StridedSliceAssign": {
        "begins": {"type": "int64[]"},
        "ends": {"type": "int64[]"},
        "steps": {"type": "int64[]"},
        "axes": {"type": "int64[]"},
    },
    "StridedSliceAssignV2": {
        "begins": {"type": "int64[]"},
        "ends": {"type": "int64[]"},
        "steps": {"type": "int64[]"},
        "axes": {"type": "int64[]"},
    },
    "Range": {"start": {"type": "int64"}, "end": {"type": "int64"}, "step": {"type": "int64"}},
    "LeakyReLU": {"negativeSlope": {"type": "float64"}, "selfIsResult": {"type": "bool"}},
    "LeakyReLUBackward": {"negativeSlope": {"type": "float64"}, "selfIsResult": {"type": "bool"}},
    "Dropout": {
        "p": {"type": "float64"},
        "train": {"type": "bool"},
        "seed": {"type": "int64"},
        "offset": {"type": "int64"},
    },
    "DropoutBackward": {"scale": {"type": "float64"}},
    "EmbeddingBackward": {
        "numEmbeddings": {"type": "int64"},
        "paddingIdx": {"type": "int64"},
        "scaleGradByFreq": {"type": "bool"},
    },
    "NanToNum": {
        "nan": {"type": "float64"},
        "posinf": {"type": "float64"},
        "neginf": {"type": "float64"},
    },
    "FlashAttention": {
        "scale": {"type": "float64"},
        "keepProb": {"type": "float64"},
        "preToken": {"type": "int64"},
        "nextToken": {"type": "int64"},
        "headNum": {"type": "int64"},
        "inputLayout": {"type": "int64[]", "encoding": "utf8"},
        "innerPrecise": {"type": "int64"},
        "sparseMode": {"type": "int64"},
        "psetype": {"type": "int64"},
        "prefix": {"type": "int64[]"},
        "qStartIdx": {"type": "int64[]"},
        "kvStartIdx": {"type": "int64[]"},
        "hasRealshift": {"type": "bool"},
        "hasDropmask": {"type": "bool"},
        "hasPaddingmask": {"type": "bool"},
        "hasAttentmask": {"type": "bool"},
    },
    "FlashAttentionBackward": {
        "scale": {"type": "float64"},
        "keepProb": {"type": "float64"},
        "preToken": {"type": "int64"},
        "nextToken": {"type": "int64"},
        "headNum": {"type": "int64"},
        "inputLayout": {"type": "int64[]", "encoding": "utf8"},
        "innerPrecise": {"type": "int64"},
        "sparseMode": {"type": "int64"},
        "psetype": {"type": "int64"},
        "prefix": {"type": "int64[]"},
        "qStartIdx": {"type": "int64[]"},
        "kvStartIdx": {"type": "int64[]"},
        "hasRealshift": {"type": "bool"},
        "hasDropmask": {"type": "bool"},
        "hasPaddingmask": {"type": "bool"},
        "hasAttentmask": {"type": "bool"},
    },
    "IncreFlashAttention": {
        "scale": {"type": "float64"},
        "headNum": {"type": "int64"},
        "keyValueHeadNum": {"type": "int64"},
        "inputLayout": {"type": "int64[]", "encoding": "utf8"},
        "innerPrecise": {"type": "int64"},
        "blockSize": {"type": "int64", "default": 0},
        "hasBlockTable": {"type": "bool", "default": False},
        "actualSeqLengths": {"type": "int64[]", "default": []},
    },
    "KVCacheMemcpy": {"blockSize": {"type": "int64"}, "slots": {"type": "int64[]"}},
    "Roll": {"shifts": {"type": "int64[]"}, "dims": {"type": "int64[]"}},
    "MatMul": {"mode": {"type": "int64"}, "cube_math_type": {"type": "int64"}},
    "BatchMatMul": {"mode": {"type": "int64"}, "cube_math_type": {"type": "int64"}},
    "Unary": {"operation": {"type": "int64[]", "encoding": "utf8"}},
    "TruthReduce": {
        "axes": {"type": "int64[]"},
        "keepdims": {"type": "bool"},
        "reduce_all": {"type": "bool"},
    },
}


def attribute_data(name, attributes, *, prefix="acl_attr."):
    try:
        schema = SCHEMAS[name]
    except KeyError:
        raise AclDataInternalError("ACL runner has no attribute schema: " + name) from None
    unknown = set(attributes) - set(schema)
    if unknown:
        raise AclDataInternalError("undeclared ACL runner attributes: " + repr(sorted(unknown)))

    def owner_value(field, value):
        declaration = schema[field]
        if declaration.get("encoding") == "utf8":
            if not isinstance(value, str):
                raise TypeError("ACL text attribute must be a string: " + field)
            return list(value.encode("utf-8"))
        if declaration["type"].endswith("[]"):
            return list(value)
        if declaration["type"] == "bool":
            return bool(value)
        return value

    record = {
        "schema_version": SCHEMA_VERSION,
        "op": name,
        "fields": {
            field: {"type": schema[field]["type"], "value": owner_value(field, value)}
            for field, value in attributes.items()
        },
    }
    return encode_code_data(record, expected_op=name, schema=schema, prefix=prefix)


def attribute_payloads(payloads):
    result = {}
    for slot, (name, attributes) in payloads.items():
        if not isinstance(slot, str) or not slot.isidentifier():
            raise AclDataInternalError("ACL payload slot must be an identifier")
        result.update(attribute_data(name, attributes, prefix="acl_payload." + slot + "."))
    return result


class AttributeCode:
    """Structural source plus typed data; never converts values back to C++."""

    def __init__(self, source, data):
        self.source = source
        self.data = data


def code_program(parts):
    import struct

    source, data = [], {}
    for part in parts:
        if isinstance(part, AttributeCode):
            source.append(part.source)
            for key, value in part.data.items():
                if key in data and struct.pack("d", data[key]) != struct.pack("d", value):
                    raise AclDataInternalError("conflicting ACL attribute payload: " + key)
                data[key] = value
        else:
            source.append(str(part))
    return AttributeCode("".join(source), data)


def attribute_program(name, attributes, *, variable="op", slot=None):
    if not isinstance(variable, str) or not variable.isidentifier():
        raise AclDataInternalError("ACL runner variable must be an identifier")
    slot = slot or name + "_" + variable
    if not slot.isidentifier():
        raise AclDataInternalError("ACL payload slot must be an identifier")
    prefix = "acl_payload." + slot + "."
    return AttributeCode(
        "apply_acl_code_attributes(" + variable + ', data, "' + prefix + '", "' + name + '");',
        attribute_data(name, attributes, prefix=prefix),
    )


_RUNNER_ALIASES = {
    "softmax": "Softmax",
    "triu": "Triu",
    "flip": "Flip",
    "transpose": "Transpose",
    "cumsum": "Cumsum",
    "gather": "Gather",
    "scatter": "Scatter",
    "conv2d": "Conv2d",
    "conv2dbackward": "Conv2dBackward",
    "batchnorm": "BatchNorm",
    "batchnormbackward": "BatchNormBackward",
    "layernorm": "LayerNorm",
    "layernormbackward": "LayerNormBackward",
    "groupnorm": "GroupNorm",
    "rmsnorm": "RmsNorm",
    "maxpool": "Maxpool",
    "avgpool": "Avgpool",
    "maxpoolbackward": "MaxpoolBackward",
    "avgpoolbackward": "AvgpoolBackward",
    "upsample_nearest2d": "UpsampleNearest2d",
    "concat": "Concat",
    "stack": "Stack",
    "splitwithsize": "SplitWithSize",
    "slicev2": "SliceV2",
    "stridedsliceassign": "StridedSliceAssign",
    "stridedsliceassignv2": "StridedSliceAssignV2",
    "range": "Range",
    "leakyrelu": "LeakyReLU",
    "leakyrelubackward": "LeakyReLUBackward",
    "dropout": "Dropout",
    "dropoutbackward": "DropoutBackward",
    "embeddingbackward": "EmbeddingBackward",
    "NanToNum": "NanToNum",
    "flashattention": "FlashAttention",
    "flashattentionbackward": "FlashAttentionBackward",
    "increflashattention": "IncreFlashAttention",
    "kv_cache_memcpy": "KVCacheMemcpy",
    "roll": "Roll",
    "matmul": "MatMul",
    "batchmatmul": "BatchMatMul",
    "unary": "Unary",
    "truthreduce": "TruthReduce",
    "stridedsliceassignv2_grad": "StridedSliceAssignV2",
    "relubackward": "LeakyReLUBackward",
    "contiguous_slice_grad": "Concat",
    "expand_rotary_cache": "SplitWithSize",
}


def runner_for_alias(name):
    try:
        return _RUNNER_ALIASES[name]
    except KeyError:
        raise AclDataInternalError("unregistered ACL runner alias: " + str(name)) from None
