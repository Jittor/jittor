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

MIGRATED_BROADCAST_SHAPE_BOUNDARIES = {
    "python/jittor/src/ops/broadcast_to_op.cc": 5,
}

MIGRATED_REINTERPRET_VIEW_BOUNDARIES = {
    "python/jittor/src/ops/reinterpret_view_op.cc": 6,
}

MIGRATED_BINARY_SHAPE_BOUNDARIES = {
    "python/jittor/src/ops/binary_op.cc": 1,
}

MIGRATED_SETITEM_SHAPE_BOUNDARIES = {
    "python/jittor/src/ops/setitem_op.cc": 2,
}

MIGRATED_GETITEM_SHAPE_BOUNDARIES = {
    "python/jittor/src/ops/getitem_op.cc": 3,
}

MIGRATED_PY_CONVERTER_USER_BOUNDARIES = {
    "python/jittor/src/pyjt/py_converter.h": 1,
}

MIGRATED_DEVICE_COPY_USER_BOUNDARIES = {
    "python/jittor/src/ops/device_copy_op.cc": 1,
}

MIGRATED_NUMPY_TYPE_BOUNDARIES = {
    "python/jittor/src/pyjt/numpy.h": 1,
}

MIGRATED_VAR_SLICES_USER_BOUNDARIES = {
    "python/jittor/src/var_slices.h": 1,
}

MIGRATED_SET_DATA_USER_BOUNDARIES = {
    "python/jittor/src/var_holder.cc": 2,
}

MIGRATED_FUSED_ADAMW_CARDINALITY_BOUNDARIES = {
    "python/jittor/src/ops/fused_adamw_op.cc": 4,
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


def test_broadcast_shape_migration_is_explicit_and_bounded():
    source = (ROOT / "python/jittor/src/ops/broadcast_to_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_BROADCAST_SHAPE_BOUNDARIES[
        "python/jittor/src/ops/broadcast_to_op.cc"]


def test_reinterpret_view_migration_is_explicit_and_bounded():
    source = (ROOT / "python/jittor/src/ops/reinterpret_view_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_REINTERPRET_VIEW_BOUNDARIES[
        "python/jittor/src/ops/reinterpret_view_op.cc"]


def test_binary_shape_migration_is_explicit_and_bounded():
    source = (ROOT / "python/jittor/src/ops/binary_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_BINARY_SHAPE_BOUNDARIES[
        "python/jittor/src/ops/binary_op.cc"]


def test_setitem_shape_migration_is_explicit_and_bounded():
    source = (ROOT / "python/jittor/src/ops/setitem_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_SETITEM_SHAPE_BOUNDARIES[
        "python/jittor/src/ops/setitem_op.cc"]


def test_getitem_shape_migration_is_explicit_and_bounded():
    source = (ROOT / "python/jittor/src/ops/getitem_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_GETITEM_SHAPE_BOUNDARIES[
        "python/jittor/src/ops/getitem_op.cc"]


def test_py_converter_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "python/jittor/src/pyjt/py_converter.h").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_PY_CONVERTER_USER_BOUNDARIES[
        "python/jittor/src/pyjt/py_converter.h"]


def test_device_copy_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "python/jittor/src/ops/device_copy_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_DEVICE_COPY_USER_BOUNDARIES[
        "python/jittor/src/ops/device_copy_op.cc"]


def test_numpy_type_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "python/jittor/src/pyjt/numpy.h").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_NUMPY_TYPE_BOUNDARIES[
        "python/jittor/src/pyjt/numpy.h"]


def test_var_slices_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "python/jittor/src/var_slices.h").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_VAR_SLICES_USER_BOUNDARIES[
        "python/jittor/src/var_slices.h"]


def test_set_data_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "python/jittor/src/var_holder.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_SET_DATA_USER_BOUNDARIES[
        "python/jittor/src/var_holder.cc"]


def test_fused_adamw_cardinality_migration_is_explicit_and_bounded():
    source = (ROOT / "python/jittor/src/ops/fused_adamw_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_FUSED_ADAMW_CARDINALITY_BOUNDARIES[
        "python/jittor/src/ops/fused_adamw_op.cc"]


def test_public_ternary_shape_migration_is_explicit_and_bounded():
    counts = {}
    for relative, expected in MIGRATED_TERNARY_SHAPE_BOUNDARIES.items():
        source = (ROOT / relative).read_text()
        actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
        counts[relative] = actual
        assert actual == expected, (relative, actual, expected)
    assert sum(counts.values()) == 2
