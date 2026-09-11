"""Schemas for production ACL runners using the CodeOp data channel."""

from types import MappingProxyType

from .acl_data import AclDataInternalError, SCHEMA_VERSION, encode_code_data


SCHEMAS = {
    "Softmax": {"dim": {"type": "int64"}},
    "SoftmaxBackward": {"dim": {"type": "int64"}},
    "SwiGlu": {"dim": {"type": "int64"}},
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
        "cube_math_type": {"type": "int64"},
    },
    "Conv2dBackward": {
        "convStrides": {"type": "int64[]"},
        "convPads": {"type": "int64[]"},
        "convDilations": {"type": "int64[]"},
        "group": {"type": "int64"},
        "convOutPads": {"type": "int64[]"},
        "cube_math_type": {"type": "int64"},
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


def _cache_key(value):
    """Hashable canonical form of an attribute value, or None when unhashable.

    Values reaching here are ints, bools, floats, short strings and small
    integer sequences, so the key is cheap next to re-encoding the payload.
    """
    if isinstance(value, (bool, int, float, str)):
        return value
    item = getattr(value, "item", None)
    if item is not None and getattr(value, "shape", None) == ():
        return item()
    if isinstance(value, (list, tuple)) or getattr(value, "shape", None) is not None:
        parts = []
        for element in value:
            key = _cache_key(element)
            if key is None:
                return None
            parts.append(key)
        return (tuple, tuple(parts))
    return None


#: Encoded payloads keyed by (runner, prefix, attribute values). The encoding is
#: a pure function of those, and it ran three times per matmul -- once for the
#: forward program and once for each gradient program -- on every single call.
#: The stored mappings are handed out read-only; every consumer copies entries
#: out of them.
_ENCODED_ATTRIBUTES = {}


def attribute_data(name, attributes, *, prefix="acl_attr."):
    key = None
    if isinstance(attributes, dict):
        fields = []
        for field in sorted(attributes):
            value = _cache_key(attributes[field])
            if value is None:
                fields = None
                break
            fields.append((field, value))
        if fields is not None:
            key = (name, prefix, tuple(fields))
            cached = _ENCODED_ATTRIBUTES.get(key)
            if cached is not None:
                return cached
    encoded = _attribute_data_uncached(name, attributes, prefix=prefix)
    if key is not None:
        encoded = MappingProxyType(encoded)
        _ENCODED_ATTRIBUTES[key] = encoded
    return encoded


def _attribute_data_uncached(name, attributes, *, prefix="acl_attr."):
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
    """Structural source plus typed data; never converts values back to C++.

    ``key`` identifies a fragment that was built from cached, immutable inputs,
    which lets code_program memoise the programs assembled out of them.
    """

    def __init__(self, source, data, key=None):
        self.source = source
        self.data = data
        self.key = key


#: Assembled programs keyed by their fragments. A runner program is rebuilt on
#: every op construction, and joining the sources plus merging ~20 data lanes
#: was the largest remaining python cost per ACL operator.
_ASSEMBLED_PROGRAMS = {}


def _program_key(parts):
    key = []
    for part in parts:
        if isinstance(part, AttributeCode):
            if part.key is None:
                return None
            key.append(part.key)
        else:
            key.append(str(part))
    return tuple(key)


def code_program(parts):
    parts = list(parts)
    key = _program_key(parts)
    if key is not None:
        cached = _ASSEMBLED_PROGRAMS.get(key)
        if cached is not None:
            return cached
    program = _code_program_uncached(parts)
    if key is not None:
        program.key = key
        _ASSEMBLED_PROGRAMS[key] = program
    return program


def _code_program_uncached(parts):
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


_ATTRIBUTE_PROGRAMS = {}


def attribute_program(name, attributes, *, variable="op", slot=None):
    key = None
    if isinstance(attributes, dict):
        values = []
        for field in sorted(attributes):
            value = _cache_key(attributes[field])
            if value is None:
                values = None
                break
            values.append((field, value))
        if values is not None:
            key = ("program", name, variable, slot, tuple(values))
            cached = _ATTRIBUTE_PROGRAMS.get(key)
            if cached is not None:
                return cached
    program = _attribute_program_uncached(name, attributes, variable=variable, slot=slot)
    if key is not None:
        program.key = key
        _ATTRIBUTE_PROGRAMS[key] = program
    return program


def _attribute_program_uncached(name, attributes, *, variable="op", slot=None):
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
