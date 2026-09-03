from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIGRATED_DIMENSION_BOUNDARIES = {
    "python/jittor/src/ops/arg_reduce_op.cc": 2,
    "python/jittor/src/ops/argsort_op.cc": 2,
    "python/jittor/src/ops/reduce_op.cc": 1,
    "python/jittor/src/ops/broadcast_to_op.cc": 2,
}

MIGRATED_SHAPE_CARDINALITY_BOUNDARIES = {
    "python/jittor/src/ops/code_op.cc": 3,
    "python/jittor/src/ops/numpy_code_op.cc": 4,
    "python/jittor/src/ops/reindex_op.cc": 1,
    "python/jittor/src/ops/reindex_reduce_op.cc": 2,
}

MIGRATED_VIEW_SHAPE_BOUNDARIES = {
    "python/jittor/src/ops/transpose_op.cc": 3,
    "python/jittor/src/ops/fuse_transpose_op.cc": 3,
    "python/jittor/src/ops/reshape_op.cc": 3,
}

MIGRATED_TERNARY_SHAPE_BOUNDARIES = {
    "python/jittor/src/ops/ternary_op.cc": 2,
}


def test_typed_error_entry_points_are_distinct():
    source = (ROOT / "python/jittor/src/utils/log.h").read_text()
    assert "struct UserError : JittorError" in source
    assert "struct InternalInvariantError : JittorError" in source
    for entry in ("USER_CHECK", "USER_CHECKop", "INTERNAL_ASSERT",
                  "INTERNAL_ASSERTop"):
        assert "#define " + entry in source


def test_public_dimension_boundary_migration_is_explicit_and_bounded():
    counts = {}
    for relative, expected in MIGRATED_DIMENSION_BOUNDARIES.items():
        source = (ROOT / relative).read_text()
        actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
        counts[relative] = actual
        assert actual == expected, (relative, actual, expected)
    assert sum(counts.values()) == 7


def test_public_shape_cardinality_migration_is_explicit_and_bounded():
    counts = {}
    for relative, expected in MIGRATED_SHAPE_CARDINALITY_BOUNDARIES.items():
        source = (ROOT / relative).read_text()
        actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
        counts[relative] = actual
        assert actual == expected, (relative, actual, expected)
    assert sum(counts.values()) == 10


def test_public_view_shape_migration_is_explicit_and_bounded():
    counts = {}
    for relative, expected in MIGRATED_VIEW_SHAPE_BOUNDARIES.items():
        source = (ROOT / relative).read_text()
        actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
        counts[relative] = actual
        assert actual == expected, (relative, actual, expected)
    assert sum(counts.values()) == 9


def test_public_ternary_shape_migration_is_explicit_and_bounded():
    counts = {}
    for relative, expected in MIGRATED_TERNARY_SHAPE_BOUNDARIES.items():
        source = (ROOT / relative).read_text()
        actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
        counts[relative] = actual
        assert actual == expected, (relative, actual, expected)
    assert sum(counts.values()) == 2
