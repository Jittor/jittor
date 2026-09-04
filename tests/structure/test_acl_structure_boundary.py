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
    ):
        assert required in text
