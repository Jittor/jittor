from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs/guides/acl-structure-boundary.md"


def test_acl_structure_boundary_names_atomic_migrations_and_hardware_gate():
    text = GUIDE.read_text(encoding="utf-8")
    for required in (
        "AclOpFunctions",
        "Attribute data plumbing",
        "Descriptor caching",
        "KVCacheMemcpy",
        "BaseOpRunner::launch",
        "Ascend 910B3/CANN",
        "no CPU fallback",
        "triu.diagonal",
        "softmax.dim",
        "flip.axes",
        "data-channel",
        "schema",
        "Migration order",
        "softmax.dim",
        "triu.diagonal",
        "npu-smi info",
        "fallback cpu",
        "schema_version",
        "cache_key",
        "type_tag",
        "BaseOpRunner` helper",
        "AclDecodedData decode_acl_data",
        "AclDataRecord& record",
        "UserError",
        "InternalInvariantError",
        "canonical_cache_key",
        "non-canonical vector",
    ):
        assert required in text


def test_acl_data_schema_contract_for_future_decoder():
    text = GUIDE.read_text(encoding="utf-8")
    assert "currently `1`, required" in text
    assert "typed `int64`, `float64`, or `bool`" in text
    assert "typed homogeneous `int64[]`/`float64[]`/`bool[]`" in text
    assert "pointer addresses and Python object ids are forbidden" in text
    assert "validates the\noperator name, schema version, type tag, and required fields" in text
    assert "host-only C++ decoder boundary" in text
