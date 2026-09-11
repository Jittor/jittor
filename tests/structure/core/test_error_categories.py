from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MIGRATED_DIMENSION_BOUNDARIES = {
    "src/ops/composite/arg_reduce_op.cc": 2,
    "src/ops/composite/argsort_op.cc": 2,
    # 5, not 1. Two of the three that were already there predate this ledger
    # entry and were never counted: `ns.is_binary()` appears in each of the two
    # real constructors (the third delegates). The two added on 2026-09-11 are
    # the integral-only dtype guard, one per constructor -- before it, a bitwise
    # reduction over a float was accepted and the mistake surfaced as a g++ wall
    # from the generated kernel. See tests/ops/test_bitwise_dtype_guard.py.
    "src/ops/reduce_op.cc": 5,
    "src/ops/broadcast_to_op.cc": 2,
}

MIGRATED_SHAPE_CARDINALITY_BOUNDARIES = {
    "src/ops/composite/code_op.cc": 5,
    "src/ops/composite/numpy_code_op.cc": 4,
    "src/ops/reindex_op.cc": 2,
    "src/ops/reindex_reduce_op.cc": 3,
}

MIGRATED_VIEW_SHAPE_BOUNDARIES = {
    "src/ops/composite/transpose_op.cc": 3,
    "src/ops/composite/fuse_transpose_op.cc": 3,
    "src/ops/composite/reshape_op.cc": 3,
}

MIGRATED_BROADCAST_SHAPE_BOUNDARIES = {
    "src/ops/broadcast_to_op.cc": 5,
}

MIGRATED_REINTERPRET_VIEW_BOUNDARIES = {
    "src/ops/composite/reinterpret_view_op.cc": 6,
}

MIGRATED_BINARY_SHAPE_BOUNDARIES = {
    "src/ops/binary_op.cc": 1,
}

# 2026-09-10: both counts were one short of the source before the index bounds
# check was added, so these two cases had been failing without anyone reading
# them. Every entry below was checked by hand and is a genuine caller boundary
# -- a shape that does not match, an ellipsis with too many indices after it, a
# `int` index outside the dimension, too many slices, a setitem operation that
# is neither void nor binary, a data dimension larger than the target. The new
# one is the Var index the kernel now range-checks (KI-OPS-010); it is
# `USER_CHECK` rather than `CHECK` because an out-of-range index is exactly the
# case `USER_ERROR` in `src/utils/log.h` describes -- "a caller supplied an
# unsupported value, shape, dtype, or index" -- and callers should be able to
# catch it as one.
MIGRATED_SETITEM_SHAPE_BOUNDARIES = {
    "src/ops/composite/setitem_op.cc": 4,
}

MIGRATED_GETITEM_SHAPE_BOUNDARIES = {
    "src/ops/composite/getitem_op.cc": 5,
}

MIGRATED_PY_CONVERTER_USER_BOUNDARIES = {
    # 1 bool-slice input + 3 on the value Function.grad hands back.
    "src/bindings/pyjt/py_converter.h": 4,
}

MIGRATED_DEVICE_COPY_USER_BOUNDARIES = {
    "src/ops/composite/device_copy_op.cc": 1,
}

MIGRATED_NUMPY_TYPE_BOUNDARIES = {
    "src/bindings/pyjt/numpy.h": 1,
}

MIGRATED_VAR_SLICES_USER_BOUNDARIES = {
    "src/core/var_slices.h": 1,
}

MIGRATED_SET_DATA_USER_BOUNDARIES = {
    "src/core/var_holder.cc": 2,
}

MIGRATED_PY_ARRAY_USER_BOUNDARIES = {
    "src/bindings/pyjt/py_array_op.cc": 2,
}

MIGRATED_RANDOM_TYPE_USER_BOUNDARIES = {
    "src/ops/composite/random_op.cc": 1,
}

MIGRATED_PY_CALLER_USER_BOUNDARIES = {
    "src/bindings/pyjt/py_caller.cc": 1,
}

MIGRATED_UNARY_OP_USER_BOUNDARIES = {
    # 2 since 2026-09-11: `bitwise_not` on a float now says so instead of
    # reaching `~x` in the generated kernel and letting g++ answer "wrong type
    # argument to bit-complement". See tests/ops/test_bitwise_dtype_guard.py.
    "src/ops/unary_op.cc": 2,
}

MIGRATED_CURAND_USER_BOUNDARIES = {
    "backends/cuda/kernels/curand/curand_random_op.cc": 2,
}

MIGRATED_CUFFT_DTYPE_USER_BOUNDARIES = {
    "backends/cuda/kernels/cufft/cufft_fft_op.cc": 1,
}

MIGRATED_CUDNN_RNN_DTYPE_USER_BOUNDARIES = {
    "backends/cuda/libraries/cudnn/include/cudnn_rnn_descriptor.h": 1,
}

MIGRATED_CUDNN_RNN_DESCRIPTOR_MODE_USER_BOUNDARIES = {
    "backends/cuda/libraries/cudnn/include/cudnn_rnn_descriptor.h": 2,
}

MIGRATED_CUDNN_RNN_OP_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_rnn_op.cc": 7,
}

MIGRATED_CUDNN_RNN_BWD_X_MODE_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_rnn_backward_x_op.cc": 4,
}

MIGRATED_CUTT_TRANSPOSE_AXES_USER_BOUNDARIES = {
    "backends/cuda/kernels/cutt/cutt_transpose_op.cc": 2,
}

MIGRATED_CUTT_TRANSPOSE_RANK_USER_BOUNDARIES = {
    "backends/cuda/kernels/cutt/cutt_transpose_op.cc": 1,
}

MIGRATED_CUBLAS_MATMUL_DTYPE_USER_BOUNDARIES = {
    "backends/cuda/kernels/cublas/cublas_matmul_op.cc": 4,
}

MIGRATED_CUBLAS_MATMUL_RANK_USER_BOUNDARIES = {
    "backends/cuda/kernels/cublas/cublas_matmul_op.cc": 1,
}

MIGRATED_CUBLAS_BATCHED_MATMUL_DTYPE_USER_BOUNDARIES = {
    "backends/cuda/kernels/cublas/cublas_batched_matmul_op.cc": 2,
}

MIGRATED_CUBLAS_BATCHED_MATMUL_RANK_USER_BOUNDARIES = {
    "backends/cuda/kernels/cublas/cublas_batched_matmul_op.cc": 3,
}

MIGRATED_CUBLAS_BATCHED_MATMUL_INNER_DIM_USER_BOUNDARIES = {
    "backends/cuda/kernels/cublas/cublas_batched_matmul_op.cc": 1,
}

MIGRATED_CUBLAS_BATCHED_MATMUL_BATCH_SHAPE_USER_BOUNDARIES = {
    "backends/cuda/kernels/cublas/cublas_batched_matmul_op.cc": 1,
}

MIGRATED_CUBLAS_ACC_MATMUL_DTYPE_USER_BOUNDARIES = {
    "backends/cuda/kernels/cublas/cublas_acc_matmul_op.cc": 2,
}

MIGRATED_CUBLAS_ACC_MATMUL_RANK_USER_BOUNDARIES = {
    "backends/cuda/kernels/cublas/cublas_acc_matmul_op.cc": 2,
}

MIGRATED_CUBLAS_ACC_MATMUL_INNER_DIM_USER_BOUNDARIES = {
    "backends/cuda/kernels/cublas/cublas_acc_matmul_op.cc": 1,
}

MIGRATED_CUSPARSE_SPMMCSR_DTYPE_USER_BOUNDARIES = {
    "backends/cuda/kernels/cusparse/cusparse_spmmcsr_op.cc": 2,
}

MIGRATED_CUSPARSE_SPMMCSR_SHAPE_USER_BOUNDARIES = {
    "backends/cuda/kernels/cusparse/cusparse_spmmcsr_op.cc": 2,
}

MIGRATED_CUSPARSE_SPMMCOO_DTYPE_USER_BOUNDARIES = {
    "backends/cuda/kernels/cusparse/cusparse_spmmcoo_op.cc": 2,
}

MIGRATED_CUSPARSE_SPMMCOO_SHAPE_USER_BOUNDARIES = {
    "backends/cuda/kernels/cusparse/cusparse_spmmcoo_op.cc": 2,
}

MIGRATED_NCCL_REDUCE_SCATTER_SHAPE_USER_BOUNDARIES = {
    "backends/comm/nccl/ops/nccl_reduce_scatter_op.cc": 2,
}

MIGRATED_CUB_CUMSUM_RANK_USER_BOUNDARIES = {
    "backends/cuda/kernels/cub/cub_cumsum_op.cc": 1,
}

MIGRATED_CUB_OP_USER_BOUNDARIES = {
    "backends/cuda/kernels/cub/cub_argsort_op.cc": 5,
    "backends/cuda/kernels/cub/cub_arg_reduce_op.cc": 3,
}

MIGRATED_CUDNN_CONV_FORMAT_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv_op.cc": 4,
}

MIGRATED_CUDNN_CONV_FORMAT_COMPARE_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv_op.cc": 1,
}

MIGRATED_CUDNN_CONV_BWD_X_FORMAT_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv_backward_x_op.cc": 3,
}

MIGRATED_CUDNN_CONV_BWD_X_FORMAT_COMPARE_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv_backward_x_op.cc": 1,
}

MIGRATED_CUDNN_CONV_BWD_W_FORMAT_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv_backward_w_op.cc": 3,
}

MIGRATED_CUDNN_CONV_BWD_W_FORMAT_COMPARE_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv_backward_w_op.cc": 1,
}

INTERNAL_BACKEND_ASSERTION_CONTRACTS = {
    "backends/cuda/kernels/cudnn/cudnn_conv_op.cc": "ASSERT(best_algo_idx!=-1)",
    "backends/cuda/kernels/cudnn/cudnn_conv3d_op.cc": "ASSERT(best_algo_idx!=-1)",
    "backends/cuda/kernels/cudnn/cudnn_conv_backward_x_op.cc": "ASSERT(best_algo_idx!=-1)",
    "backends/cuda/kernels/cudnn/cudnn_conv_backward_w_op.cc": "ASSERT(best_algo_idx!=-1)",
    "backends/cuda/kernels/cudnn/cudnn_conv3d_backward_x_op.cc": "ASSERT(best_algo_idx!=-1)",
    "backends/cuda/kernels/cudnn/cudnn_conv3d_backward_w_op.cc": "ASSERT(best_algo_idx!=-1)",
    "backends/cuda/libraries/cudnn/include/cudnn_conv_plan.h": "ASSERT(ok)",
    "backends/cuda/libraries/cudnn/src/cudnn_rnn_descriptor.cc": "ASSERT(linLayerMat)",
    "backends/cuda/libraries/cutt/src/cutt_wrapper.cc": "CHECK(ret == CUTT_SUCCESS)",
    "backends/cuda/kernels/cub/cub_test_op.cc": "ASSERT(cub_test_entry",
    "backends/cuda/kernels/cublas/cublas_test_op.cc": "ASSERT(cublas_test_entry",
    "backends/cuda/kernels/cudnn/cudnn_test_op.cc": "ASSERT(cudnn_test_entry",
}

MIGRATED_CUDNN_CONV3D_X_RANK_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv3d_op.cc": 3,
}

MIGRATED_CUDNN_CONV3D_BWD_X_W_RANK_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv3d_backward_x_op.cc": 2,
}

MIGRATED_CUDNN_CONV3D_BWD_W_X_RANK_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv3d_backward_w_op.cc": 2,
}

MIGRATED_FUSED_ADAMW_CARDINALITY_BOUNDARIES = {
    "src/ops/composite/fused_adamw_op.cc": 4,
}

MIGRATED_TERNARY_SHAPE_BOUNDARIES = {
    "src/ops/ternary_op.cc": 2,
}

MIGRATED_ITEM_USER_BOUNDARIES = {
    "src/core/var_holder.cc": 1,
}

MIGRATED_GRAD_DTYPE_USER_BOUNDARIES = {
    "src/core/grad.cc": 2,
}


def test_typed_error_entry_points_are_distinct():
    source = (ROOT / "src/utils/log.h").read_text()
    assert "struct UserError : JittorError" in source
    assert "struct InternalInvariantError : JittorError" in source
    for entry in ("USER_CHECK", "USER_CHECKop", "INTERNAL_ASSERT",
                  "INTERNAL_ASSERTop"):
        assert "#define " + entry in source


def test_item_size_boundary_is_a_user_error():
    source = (ROOT / "src/core/var_holder.cc").read_text()
    assert "USER_CHECK(var->num==1)" in source
    assert "\n    CHECK(var->num==1)" not in source
    actual = source.count("USER_CHECK(var->num==1)")
    assert actual == MIGRATED_ITEM_USER_BOUNDARIES[
        "src/core/var_holder.cc"]


def test_grad_dtype_boundaries_are_user_errors():
    source = (ROOT / "src/core/grad.cc").read_text()
    assert 'USER_CHECK(loss->is_float())' in source
    assert 'USER_CHECK(var->is_float() || var->dtype().is_complex())' in source
    assert '\n    CHECK(loss->is_float())' not in source
    assert '\n        CHECK(var->is_float() || var->dtype().is_complex())' not in source
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_GRAD_DTYPE_USER_BOUNDARIES[
        "src/core/grad.cc"]


def test_public_dimension_boundary_migration_is_explicit_and_bounded():
    counts = {}
    for relative, expected in MIGRATED_DIMENSION_BOUNDARIES.items():
        source = (ROOT / relative).read_text()
        # ``broadcast_to_op.cc`` also owns the five shape checks asserted by
        # ``MIGRATED_BROADCAST_SHAPE_BOUNDARIES``.  Count its two dimension
        # checks by their diagnostic so the independent ledgers do not overlap.
        if relative == "src/ops/broadcast_to_op.cc":
            actual = source.count('USER_CHECK(dim>=0 && dim<ydim)')
        else:
            actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
        counts[relative] = actual
        assert actual == expected, (relative, actual, expected)
    # 11 = 2 + 2 + 5 + 2, the four entries above. It read 7 while
    # ``reduce_op.cc`` was recorded as 1 and actually held 3, so this total had
    # been failing before the two guards of 2026-09-11 were added; both halves
    # are corrected together here rather than leaving the sum stale.
    assert sum(counts.values()) == 11


def test_public_shape_cardinality_migration_is_explicit_and_bounded():
    counts = {}
    for relative, expected in MIGRATED_SHAPE_CARDINALITY_BOUNDARIES.items():
        source = (ROOT / relative).read_text()
        actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
        if relative == "src/ops/composite/code_op.cc":
            # Backend provenance is independent of the shape/cardinality ledger.
            for guard in ("USER_CHECK(backend.empty()", "USER_CHECK(execution_backend() == expected)"):
                assert source.count(guard) == 1
                actual -= 1
        counts[relative] = actual
        assert actual == expected, (relative, actual, expected)
    assert sum(counts.values()) == 14


def test_public_view_shape_migration_is_explicit_and_bounded():
    counts = {}
    for relative, expected in MIGRATED_VIEW_SHAPE_BOUNDARIES.items():
        source = (ROOT / relative).read_text()
        actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
        counts[relative] = actual
        assert actual == expected, (relative, actual, expected)
    assert sum(counts.values()) == 9


def test_broadcast_shape_migration_is_explicit_and_bounded():
    source = (ROOT / "src/ops/broadcast_to_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_BROADCAST_SHAPE_BOUNDARIES[
        "src/ops/broadcast_to_op.cc"]


def test_reinterpret_view_migration_is_explicit_and_bounded():
    source = (ROOT / "src/ops/composite/reinterpret_view_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_REINTERPRET_VIEW_BOUNDARIES[
        "src/ops/composite/reinterpret_view_op.cc"]


def test_binary_shape_migration_is_explicit_and_bounded():
    source = (ROOT / "src/ops/binary_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_BINARY_SHAPE_BOUNDARIES[
        "src/ops/binary_op.cc"]


def test_setitem_shape_migration_is_explicit_and_bounded():
    source = (ROOT / "src/ops/composite/setitem_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_SETITEM_SHAPE_BOUNDARIES[
        "src/ops/composite/setitem_op.cc"]


def test_getitem_shape_migration_is_explicit_and_bounded():
    source = (ROOT / "src/ops/composite/getitem_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_GETITEM_SHAPE_BOUNDARIES[
        "src/ops/composite/getitem_op.cc"]


def test_py_converter_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "src/bindings/pyjt/py_converter.h").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_PY_CONVERTER_USER_BOUNDARIES[
        "src/bindings/pyjt/py_converter.h"]
    negative = (ROOT / "tests/autograd/test_function.py").read_text()
    assert "test_returning_the_wrong_number_of_grads_is_a_catchable_user_error" in negative
    assert "test_returning_a_non_var_grad_is_a_catchable_user_error" in negative


def test_py_converter_conversion_checks_stay_internal_invariants():
    """The rest of py_converter.h is not a user boundary.

    ``from_py_object`` only runs after the generated argument parser has
    accepted the call through ``is_type``; a disagreement between those two
    halves of the same generated binding is the framework contradicting
    itself, not bad input. Bad input is rejected earlier, by the parser, with
    a catchable RuntimeError that names the op and the types it was given.
    """
    source = (ROOT / "src/bindings/pyjt/py_converter.h").read_text()
    conversion_guards = source.count("CHECK(is_type<")
    assert conversion_guards >= 3, source.count("CHECK(is_type<")
    assert "USER_CHECK(is_type<" not in source


def test_device_copy_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "src/ops/composite/device_copy_op.cc").read_text()
    # The original migration owns the device-index check. Scalar placement
    # normalization is a new constructor-input boundary, not a second migrated
    # assertion; keep the two scopes explicit rather than inflating the ledger.
    copy_impl, marker, scalar_adapter = source.partition("\nvoid adapt_cpu_scalar_operands(")
    assert marker
    actual = copy_impl.count("USER_CHECK(") + copy_impl.count("USER_CHECKop(")
    assert actual == MIGRATED_DEVICE_COPY_USER_BOUNDARIES[
        "src/ops/composite/device_copy_op.cc"]
    assert scalar_adapter.count("USER_CHECK(") == 1
    assert "USER_CHECK(!target.explicit_backend || target == value->placement)" in scalar_adapter
    assert "Expected all tensor inputs on the same backend and device" in scalar_adapter


def test_numpy_type_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "src/bindings/pyjt/numpy.h").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_NUMPY_TYPE_BOUNDARIES[
        "src/bindings/pyjt/numpy.h"]


def test_var_slices_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "src/core/var_slices.h").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_VAR_SLICES_USER_BOUNDARIES[
        "src/core/var_slices.h"]


def test_set_data_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "src/core/var_holder.cc").read_text()
    # ``item()`` has an independent user boundary in the same translation
    # unit; count only the two set_data predicates here.
    actual = source.count("USER_CHECK(array.dtype.dsize()")
    actual += source.count("USER_CHECK(size==var->size)")
    assert actual == MIGRATED_SET_DATA_USER_BOUNDARIES[
        "src/core/var_holder.cc"]


def test_py_array_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "src/bindings/pyjt/py_array_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_PY_ARRAY_USER_BOUNDARIES[
        "src/bindings/pyjt/py_array_op.cc"]


def test_random_type_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "src/ops/composite/random_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_RANDOM_TYPE_USER_BOUNDARIES[
        "src/ops/composite/random_op.cc"]


def test_py_caller_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "src/bindings/pyjt/py_caller.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_PY_CALLER_USER_BOUNDARIES[
        "src/bindings/pyjt/py_caller.cc"]


def test_unary_op_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "src/ops/unary_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_UNARY_OP_USER_BOUNDARIES[
        "src/ops/unary_op.cc"]


def test_curand_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/curand/curand_random_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_CURAND_USER_BOUNDARIES[
        "backends/cuda/kernels/curand/curand_random_op.cc"]
    assert "curand_random supports float32 and float64 only" in source
    assert "type == ns_normal || type == ns_uniform" in source
    negative = (ROOT / "tests/backends/cuda/test_curand_odd_length.py").read_text()
    assert "test_unsupported_dtype_is_rejected_clearly" in negative


def test_cufft_dtype_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cufft/cufft_fft_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_CUFFT_DTYPE_USER_BOUNDARIES[
        "backends/cuda/kernels/cufft/cufft_fft_op.cc"]
    assert "not supported fft dtype" in source


def test_cudnn_rnn_dtype_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/libraries/cudnn/include/cudnn_rnn_descriptor.h").read_text()
    actual = source.count("USER_CHECK(")
    assert actual == MIGRATED_CUDNN_RNN_DTYPE_USER_BOUNDARIES[
        "backends/cuda/libraries/cudnn/include/cudnn_rnn_descriptor.h"]
    assert "cudnn rnn supports float16, float32 and float64" in source
    negative = (ROOT / "tests/backends/cuda/test_cudnn_rnn_dtype.py").read_text()
    assert "test_unsupported_dtype_names_itself" in negative


def test_cudnn_rnn_descriptor_mode_is_a_catchable_user_error():
    source = (ROOT / "backends/cuda/libraries/cudnn/include/cudnn_rnn_descriptor.h").read_text()
    marker = 'USER_CHECKop(mode,==,"gru")'
    assert source.count(marker) == MIGRATED_CUDNN_RNN_DESCRIPTOR_MODE_USER_BOUNDARIES[
        "backends/cuda/libraries/cudnn/include/cudnn_rnn_descriptor.h"]
    assert 'ASSERT(mode == "gru")' not in source
    assert "mode must be relu, tanh, lstm, or gru" in source


def test_cudnn_rnn_op_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cudnn/cudnn_rnn_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_CUDNN_RNN_OP_USER_BOUNDARIES[
        "backends/cuda/kernels/cudnn/cudnn_rnn_op.cc"]
    negative = (ROOT / "tests/backends/cuda/test_cudnn_rnn_dtype.py").read_text()
    assert "test_mixed_input_weight_dtype_is_rejected_clearly" in negative


def test_cudnn_rnn_bwd_x_mode_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cudnn/cudnn_rnn_backward_x_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_CUDNN_RNN_BWD_X_MODE_USER_BOUNDARIES[
        "backends/cuda/kernels/cudnn/cudnn_rnn_backward_x_op.cc"]


def test_cutt_transpose_axes_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cutt/cutt_transpose_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    actual -= source.count("USER_CHECK(xdim)")
    assert actual == MIGRATED_CUTT_TRANSPOSE_AXES_USER_BOUNDARIES[
        "backends/cuda/kernels/cutt/cutt_transpose_op.cc"]
    negative = (ROOT / "tests/backends/cuda/test_cutt_transpose_op.py").read_text()
    assert "test_axes_length_is_a_catchable_user_error" in negative
    assert "test_duplicate_axes_are_a_catchable_user_error" in negative


def test_cutt_transpose_scalar_rank_is_a_catchable_user_error():
    source = (ROOT / "backends/cuda/kernels/cutt/cutt_transpose_op.cc").read_text()
    assert 'USER_CHECK(xdim) << "cutt transpose requires a non-scalar input"' in source
    assert "CHECK(xdim);" not in source
    assert source.count("USER_CHECK(xdim)") == MIGRATED_CUTT_TRANSPOSE_RANK_USER_BOUNDARIES[
        "backends/cuda/kernels/cutt/cutt_transpose_op.cc"]


def test_cublas_matmul_dtype_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cublas/cublas_matmul_op.cc").read_text()
    # Keep the new b-rank cohort independent from the existing dtype/rank/inner
    # ledger represented by this historical count.
    actual = source.count("USER_CHECK(")
    actual += source.count("USER_CHECKop(a->shape.size(),==,2)")
    actual += source.count("USER_CHECKop(m,==,m_)")
    assert actual == MIGRATED_CUBLAS_MATMUL_DTYPE_USER_BOUNDARIES[
        "backends/cuda/kernels/cublas/cublas_matmul_op.cc"]
    negative = (ROOT / "tests/backends/cuda/test_cublas_matmul_grad.py").read_text()
    assert "test_non_float_inputs_are_rejected_clearly" in negative
    assert "test_mixed_input_dtypes_are_rejected_clearly" in negative


def test_cublas_matmul_b_rank_is_a_catchable_user_error():
    source = (ROOT / "backends/cuda/kernels/cublas/cublas_matmul_op.cc").read_text()
    marker = "USER_CHECKop(b->shape.size(),==,2)"
    assert marker in source
    assert "ASSERTop(b->shape.size(),==,2)" not in source
    assert "rank-2 input b" in source
    assert source.count(marker) == MIGRATED_CUBLAS_MATMUL_RANK_USER_BOUNDARIES[
        "backends/cuda/kernels/cublas/cublas_matmul_op.cc"]


def test_cublas_matmul_inner_dimension_is_a_catchable_user_error():
    source = (ROOT / "backends/cuda/kernels/cublas/cublas_matmul_op.cc").read_text()
    assert "USER_CHECKop(m,==,m_)" in source
    assert "ASSERTop(m,==,m_)" not in source
    assert "inner dimensions must match" in source


def test_cublas_batched_matmul_dtype_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cublas/cublas_batched_matmul_op.cc").read_text()
    # Rank checks have a separate ledger below; count only the two constructor
    # dtype checks here to keep the cohorts independent.
    actual = source.count("USER_CHECK(")
    assert actual == MIGRATED_CUBLAS_BATCHED_MATMUL_DTYPE_USER_BOUNDARIES[
        "backends/cuda/kernels/cublas/cublas_batched_matmul_op.cc"]
    negative = (ROOT / "tests/backends/cuda/test_cublas_matmul_grad.py").read_text()
    assert "test_batched_non_float_inputs_are_rejected_clearly" in negative
    assert "test_batched_mixed_input_dtypes_are_rejected_clearly" in negative


def test_cublas_batched_matmul_rank_is_a_catchable_user_error():
    source = (ROOT / "backends/cuda/kernels/cublas/cublas_batched_matmul_op.cc").read_text()
    actual = source.count("USER_CHECKop(adim,>=,3)")
    actual += source.count("USER_CHECKop(bdim,>=,3)")
    actual += source.count("USER_CHECKop(adim,==,bdim)")
    assert actual == MIGRATED_CUBLAS_BATCHED_MATMUL_RANK_USER_BOUNDARIES[
        "backends/cuda/kernels/cublas/cublas_batched_matmul_op.cc"]
    markers = (
        "USER_CHECKop(adim,>=,3)",
        "USER_CHECKop(bdim,>=,3)",
        "USER_CHECKop(adim,==,bdim)",
    )
    assert all(marker in source for marker in markers)
    assert "ASSERTop(adim,>=,3)" not in source
    assert "ASSERTop(bdim,>=,3)" not in source
    assert "ASSERTop(adim,==,bdim)" not in source


def test_cublas_batched_matmul_inner_dimension_is_a_catchable_user_error():
    source = (ROOT / "backends/cuda/kernels/cublas/cublas_batched_matmul_op.cc").read_text()
    assert "USER_CHECKop(m,==,m_)" in source
    assert "ASSERTop(m,==,m_)" not in source
    assert "inner dimensions must match" in source
    assert source.count("USER_CHECKop(m,==,m_)") == MIGRATED_CUBLAS_BATCHED_MATMUL_INNER_DIM_USER_BOUNDARIES[
        "backends/cuda/kernels/cublas/cublas_batched_matmul_op.cc"]


def test_cublas_batched_matmul_batch_shape_is_a_catchable_user_error():
    source = (ROOT / "backends/cuda/kernels/cublas/cublas_batched_matmul_op.cc").read_text()
    marker = "USER_CHECKop(a->shape[i],==,b->shape[i])"
    assert marker in source
    assert "ASSERTop(a->shape[i],==,b->shape[i])" not in source
    assert "batch dimensions must match" in source
    assert source.count(marker) == MIGRATED_CUBLAS_BATCHED_MATMUL_BATCH_SHAPE_USER_BOUNDARIES[
        "backends/cuda/kernels/cublas/cublas_batched_matmul_op.cc"]


def test_cublas_acc_matmul_dtype_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cublas/cublas_acc_matmul_op.cc").read_text()
    actual = source.count("USER_CHECK(")
    assert actual == MIGRATED_CUBLAS_ACC_MATMUL_DTYPE_USER_BOUNDARIES[
        "backends/cuda/kernels/cublas/cublas_acc_matmul_op.cc"]
    negative = (ROOT / "tests/backends/cuda/test_cublas_matmul_grad.py").read_text()
    assert "test_acc_non_float_inputs_are_rejected_clearly" in negative
    assert "test_acc_mixed_input_dtypes_are_rejected_clearly" in negative


def test_cublas_acc_matmul_rank_is_a_catchable_user_error():
    source = (ROOT / "backends/cuda/kernels/cublas/cublas_acc_matmul_op.cc").read_text()
    markers = (
        "USER_CHECKop(a->shape.size(),==,2)",
        "USER_CHECKop(b->shape.size(),==,2)",
    )
    assert all(marker in source for marker in markers)
    assert "ASSERTop(a->shape.size(),==,2)" not in source
    assert "ASSERTop(b->shape.size(),==,2)" not in source
    actual = sum(source.count(marker) for marker in markers)
    assert actual == MIGRATED_CUBLAS_ACC_MATMUL_RANK_USER_BOUNDARIES[
        "backends/cuda/kernels/cublas/cublas_acc_matmul_op.cc"]


def test_cublas_acc_matmul_inner_dimension_is_a_catchable_user_error():
    source = (ROOT / "backends/cuda/kernels/cublas/cublas_acc_matmul_op.cc").read_text()
    marker = "USER_CHECKop(m,==,m_)"
    assert marker in source
    assert "ASSERTop(m,==,m_)" not in source
    assert "inner dimensions must match" in source
    assert source.count(marker) == MIGRATED_CUBLAS_ACC_MATMUL_INNER_DIM_USER_BOUNDARIES[
        "backends/cuda/kernels/cublas/cublas_acc_matmul_op.cc"]


def test_cusparse_spmmcsr_dtype_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cusparse/cusparse_spmmcsr_op.cc").read_text()
    actual = source.count("USER_CHECK(")
    assert actual == MIGRATED_CUSPARSE_SPMMCSR_DTYPE_USER_BOUNDARIES[
        "backends/cuda/kernels/cusparse/cusparse_spmmcsr_op.cc"]
    negative = (ROOT / "tests/backends/cuda/test_cusparse_dtype.py").read_text()
    assert "test_csr_rejects_non_float_input" in negative
    assert "test_csr_rejects_mixed_input_dtypes" in negative


def test_cusparse_spmmcsr_shape_is_a_catchable_user_error():
    source = (ROOT / "backends/cuda/kernels/cusparse/cusparse_spmmcsr_op.cc").read_text()
    markers = (
        "USER_CHECKop(xs,==,os)",
        "USER_CHECKop(A_col,==,xs[0])",
    )
    assert all(marker in source for marker in markers)
    assert "ASSERT(xs==os)" not in source
    assert "ASSERT(A_col==xs[0])" not in source
    assert "sizes must match" in source
    assert "columns must match" in source
    assert sum(source.count(marker) for marker in markers) == MIGRATED_CUSPARSE_SPMMCSR_SHAPE_USER_BOUNDARIES[
        "backends/cuda/kernels/cusparse/cusparse_spmmcsr_op.cc"]


def test_cusparse_spmmcoo_dtype_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cusparse/cusparse_spmmcoo_op.cc").read_text()
    actual = source.count("USER_CHECK(")
    assert actual == MIGRATED_CUSPARSE_SPMMCOO_DTYPE_USER_BOUNDARIES[
        "backends/cuda/kernels/cusparse/cusparse_spmmcoo_op.cc"]
    negative = (ROOT / "tests/backends/cuda/test_cusparse_dtype.py").read_text()
    assert "test_coo_rejects_non_float_input" in negative
    assert "test_coo_rejects_mixed_input_dtypes" in negative


def test_cusparse_spmmcoo_shape_is_a_catchable_user_error():
    source = (ROOT / "backends/cuda/kernels/cusparse/cusparse_spmmcoo_op.cc").read_text()
    markers = (
        "USER_CHECKop(xs,==,os)",
        "USER_CHECKop(A_col,==,xs[0])",
    )
    assert all(marker in source for marker in markers)
    assert "ASSERT(xs==os)" not in source
    assert "ASSERT(A_col==xs[0])" not in source
    assert "sizes must match" in source
    assert "columns must match" in source
    assert sum(source.count(marker) for marker in markers) == MIGRATED_CUSPARSE_SPMMCOO_SHAPE_USER_BOUNDARIES[
        "backends/cuda/kernels/cusparse/cusparse_spmmcoo_op.cc"]


def test_nccl_reduce_scatter_shape_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/comm/nccl/ops/nccl_reduce_scatter_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_NCCL_REDUCE_SCATTER_SHAPE_USER_BOUNDARIES[
        "backends/comm/nccl/ops/nccl_reduce_scatter_op.cc"]
    assert "nccl_reduce_scatter expects dim0 divisible by process-group size" in source


def test_cub_cumsum_rank_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cub/cub_cumsum_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_CUB_CUMSUM_RANK_USER_BOUNDARIES[
        "backends/cuda/kernels/cub/cub_cumsum_op.cc"]
    negative = (ROOT / "tests/backends/cuda/test_cub_cumsum.py").read_text()
    assert "test_rank_three_is_rejected_clearly" in negative


def test_cub_op_user_boundary_migration_is_explicit_and_bounded():
    for relative, expected in MIGRATED_CUB_OP_USER_BOUNDARIES.items():
        source = (ROOT / relative).read_text()
        actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
        assert actual == expected, (relative, actual, expected)
    argsort_negative = (ROOT / "tests/ops/test_argsort_op.py").read_text()
    arg_reduce_negative = (ROOT / "tests/ops/test_arg_reduce_op.py").read_text()
    assert "test_cub_rejects_non_int32_offsets" in argsort_negative
    assert "test_cub_rejects_non_int32_offsets" in arg_reduce_negative


def test_cudnn_conv_format_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cudnn/cudnn_conv_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_CUDNN_CONV_FORMAT_USER_BOUNDARIES[
        "backends/cuda/kernels/cudnn/cudnn_conv_op.cc"]
    assert "Not a valid format" in source


def test_cudnn_conv_format_compare_is_a_catchable_user_error():
    source = (ROOT / "backends/cuda/kernels/cudnn/cudnn_conv_op.cc").read_text()
    marker = "USER_CHECKop(c,==,format[3])"
    assert marker in source
    assert "USER_CHECK(c==format[3])" not in source
    assert source.count(marker) == MIGRATED_CUDNN_CONV_FORMAT_COMPARE_USER_BOUNDARIES[
        "backends/cuda/kernels/cudnn/cudnn_conv_op.cc"]


def test_cudnn_conv_bwd_x_format_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cudnn/cudnn_conv_backward_x_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_CUDNN_CONV_BWD_X_FORMAT_USER_BOUNDARIES[
        "backends/cuda/kernels/cudnn/cudnn_conv_backward_x_op.cc"]
    assert "Not a valid format" in source


def test_cudnn_conv_bwd_x_format_compare_is_a_catchable_user_error():
    source = (ROOT / "backends/cuda/kernels/cudnn/cudnn_conv_backward_x_op.cc").read_text()
    marker = "USER_CHECKop(c,==,format[3])"
    assert marker in source
    assert "USER_CHECK(c==format[3])" not in source
    assert source.count(marker) == MIGRATED_CUDNN_CONV_BWD_X_FORMAT_COMPARE_USER_BOUNDARIES[
        "backends/cuda/kernels/cudnn/cudnn_conv_backward_x_op.cc"]


def test_cudnn_conv_bwd_w_format_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cudnn/cudnn_conv_backward_w_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_CUDNN_CONV_BWD_W_FORMAT_USER_BOUNDARIES[
        "backends/cuda/kernels/cudnn/cudnn_conv_backward_w_op.cc"]
    assert "Not a valid format" in source


def test_cudnn_conv_bwd_w_format_compare_is_a_catchable_user_error():
    source = (ROOT / "backends/cuda/kernels/cudnn/cudnn_conv_backward_w_op.cc").read_text()
    marker = "USER_CHECKop(c,==,format[3])"
    assert marker in source
    assert "USER_CHECK(c==format[3])" not in source
    assert source.count(marker) == MIGRATED_CUDNN_CONV_BWD_W_FORMAT_COMPARE_USER_BOUNDARIES[
        "backends/cuda/kernels/cudnn/cudnn_conv_backward_w_op.cc"]


def test_backend_internal_assertion_classification_is_explicit():
    for relative, marker in INTERNAL_BACKEND_ASSERTION_CONTRACTS.items():
        source = (ROOT / relative).read_text()
        assert marker in source, (relative, marker)
    descriptor = (ROOT / "backends/cuda/libraries/cudnn/src/cudnn_rnn_descriptor.cc").read_text()
    assert descriptor.count("ASSERT(linLayerMat)") == 1
    assert "ASSERT(linLayerBias)" in descriptor
    assert descriptor.count("ASSERT(linLayerBias)") == 1
    plan = (ROOT / "backends/cuda/libraries/cudnn/include/cudnn_conv_plan.h").read_text()
    assert plan.count("ASSERT(ok)") == 1
    cub_test = (ROOT / "backends/cuda/kernels/cub/cub_test_op.cc").read_text()
    assert cub_test.count("ASSERT(cub_test_entry") == 1
    assert cub_test.count("ASSERT(status == cudaSuccess)") == 1
    cublas_test = (ROOT / "backends/cuda/kernels/cublas/cublas_test_op.cc").read_text()
    assert cublas_test.count("ASSERT(cublas_test_entry") == 1
    cudnn_test = (ROOT / "backends/cuda/kernels/cudnn/cudnn_test_op.cc").read_text()
    assert cudnn_test.count("ASSERT(cudnn_test_entry") == 1
    cudnn_conv = (ROOT / "backends/cuda/kernels/cudnn/cudnn_conv_op.cc").read_text()
    assert cudnn_conv.count("ASSERT(best_algo_idx!=-1)") == 1
    cudnn_conv3d = (ROOT / "backends/cuda/kernels/cudnn/cudnn_conv3d_op.cc").read_text()
    assert cudnn_conv3d.count("ASSERT(best_algo_idx!=-1)") == 1
    cudnn_conv_bwd_x = (ROOT / "backends/cuda/kernels/cudnn/cudnn_conv_backward_x_op.cc").read_text()
    assert cudnn_conv_bwd_x.count("ASSERT(best_algo_idx!=-1)") == 1
    cutt_wrapper = (ROOT / "backends/cuda/libraries/cutt/src/cutt_wrapper.cc").read_text()
    assert cutt_wrapper.count("CHECK(ret == CUTT_SUCCESS)") == 2


def test_cudnn_conv3d_x_rank_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cudnn/cudnn_conv3d_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_CUDNN_CONV3D_X_RANK_USER_BOUNDARIES[
        "backends/cuda/kernels/cudnn/cudnn_conv3d_op.cc"]


def test_cudnn_conv3d_bwd_x_w_rank_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cudnn/cudnn_conv3d_backward_x_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_CUDNN_CONV3D_BWD_X_W_RANK_USER_BOUNDARIES[
        "backends/cuda/kernels/cudnn/cudnn_conv3d_backward_x_op.cc"]


def test_cudnn_conv3d_bwd_w_x_rank_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cudnn/cudnn_conv3d_backward_w_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_CUDNN_CONV3D_BWD_W_X_RANK_USER_BOUNDARIES[
        "backends/cuda/kernels/cudnn/cudnn_conv3d_backward_w_op.cc"]


def test_fused_adamw_cardinality_migration_is_explicit_and_bounded():
    source = (ROOT / "src/ops/composite/fused_adamw_op.cc").read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_FUSED_ADAMW_CARDINALITY_BOUNDARIES[
        "src/ops/composite/fused_adamw_op.cc"]


def test_public_ternary_shape_migration_is_explicit_and_bounded():
    counts = {}
    for relative, expected in MIGRATED_TERNARY_SHAPE_BOUNDARIES.items():
        source = (ROOT / relative).read_text()
        actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
        counts[relative] = actual
        assert actual == expected, (relative, actual, expected)
    assert sum(counts.values()) == 2
