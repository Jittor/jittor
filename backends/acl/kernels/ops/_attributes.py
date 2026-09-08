"""Schemas for production ACL runners using the CodeOp data channel."""
from .acl_data import AclDataInternalError, SCHEMA_VERSION, encode_code_data


_DIM = {"dim": {"type": "int64"}}
SCHEMAS = {
    "Softmax": _DIM,
    "SoftmaxBackward": _DIM,
    "Triu": {"diagonal": {"type": "int64"}},
    "Flip": {"axes": {"type": "int64[]"}},
    "Cumsum": _DIM,
    "Gather": _DIM,
    "Scatter": {"axis": {"type": "int64"}, "reduction": {"type": "int64"}},
}


def attribute_data(name, attributes):
    try:
        schema = SCHEMAS[name]
    except KeyError:
        raise AclDataInternalError("ACL runner has no attribute schema: " + name) from None
    unknown = set(attributes) - set(schema)
    if unknown:
        raise AclDataInternalError("undeclared ACL runner attributes: " + repr(sorted(unknown)))
    record = {
        "schema_version": SCHEMA_VERSION,
        "op": name,
        "fields": {
            field: {"type": schema[field]["type"], "value": value}
            for field, value in attributes.items()
        },
    }
    return encode_code_data(record, expected_op=name, schema=schema)
