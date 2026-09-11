from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
# 2026-09-11: `reduce_op.cc` went 1 -> 2. `0f5eab25e` reclassified
# `ASSERT(ns.is_binary())` in the *public* `ReduceOp(x, op, dims, keepdims)`
# constructor to `USER_CHECK`, and that is right: `op` there is the string a
# caller passed to `jt.reduce`, so `jt.reduce(x, "sqrt")` is bad input, not a
# framework fault. The same commit reclassified the copy in the second
# constructor too; that one has been put back to `ASSERT` -- see
# `test_reduce_mask_constructor_op_stays_an_internal_invariant`.
MIGRATED_DIMENSION_BOUNDARIES = {
    "src/ops/composite/arg_reduce_op.cc": 2,
    "src/ops/composite/argsort_op.cc": 2,
    "src/ops/reduce_op.cc": 2,
    "src/ops/broadcast_to_op.cc": 2,
}

# 2026-09-11: `code_op.cc` went 5 -> 14 (the two backend-provenance guards are
# still subtracted below, so the raw file count is 16). Nine sites joined the
# ledger and every one of them is `jt.code(...)` input:
#   * `7e83d6da4` added `USER_CHECK(output->is_contiguous())`;
#   * `0f5eab25e` reclassified six `CHECK`/`CHECKop` to `USER_CHECK` --
#     `cpu_grad_src.size() || cuda_grad_src.size()` (asking for a multi-output
#     gradient without giving a gradient source), `output_index >= 0`,
#     `output_index < _outputs.size()`, `input_count > 0`,
#     `input_count <= _inputs.size()` and `src.size()` (running on a backend
#     the caller gave no source for);
#   * and added two validations of the caller's `data` map, for
#     `multi_grad_output` and `multi_grad_input_count`.
MIGRATED_SHAPE_CARDINALITY_BOUNDARIES = {
    "src/ops/composite/code_op.cc": 14,
    # 2026-09-11: 4 -> 17. `0f5eab25e` reclassified thirteen sites here, all
    # on `jt.numpy_code(shapes, dtypes, inputs, forward, backward)`
    # arguments: five `CHECKop(_inputs.size(),<=,10)` and two
    # `CHECKop(_outputs.size(),<=,10)` (the op's fixed ten-slot input and
    # output arrays -- the caller chose those list lengths), five
    # `ASSERT(_outputs[i]->num >= 0)` (the extent came from the caller's
    # `shape`/`shapes`), and one new check that the caller supplied a
    # `backward` callback for the input being differentiated.
    "src/ops/composite/numpy_code_op.cc": 17,
    # 2026-09-11: 2 -> 8. `0f5eab25e` reclassified five `ASSERT`/`ASSERTop` in
    # `reindex_var` to `USER_CHECK` and gave each a sentence: one index tensor
    # per input dimension, at least one index tensor, at most ten dimensions,
    # matching index-tensor ranks and matching index-tensor shapes -- every one
    # of them reading `jt.reindex`'s `indexes`/`shape` arguments. It also
    # reclassified `ASSERT(extras.size())` (reindex without an explicit shape
    # needs the caller to pass an overflow extras tensor).
    "src/ops/reindex_op.cc": 8,
    # 2026-09-11: 3 -> 4. `0f5eab25e` reclassified
    # `ASSERT((ns.is_binary() && ns!=ns_mean) || ns == ns_void)`: `op` is the
    # reduction name the caller passed to `jt.reindex_reduce`.
    "src/ops/reindex_reduce_op.cc": 4,
}

# 2026-09-11: `reshape_op.cc` went 3 -> 5. `7e83d6da4` gave `infer_shape` a
# stride-aware path for non-contiguous inputs and guarded it with two new
# `USER_CHECK`s ("view shape is incompatible with storage strides; call
# contiguous() first" and its `vd == -1` companion). Both name a shape the
# caller asked for and tell the caller what to do, so both are user
# boundaries.
MIGRATED_VIEW_SHAPE_BOUNDARIES = {
    "src/ops/composite/transpose_op.cc": 3,
    "src/ops/composite/fuse_transpose_op.cc": 3,
    "src/ops/composite/reshape_op.cc": 5,
}

MIGRATED_BROADCAST_SHAPE_BOUNDARIES = {
    "src/ops/broadcast_to_op.cc": 5,
}

# 2026-09-11: 6 -> 8. `7e83d6da4` added two constructor boundaries --
# `x->is_contiguous()` ("call contiguous() explicitly") and
# `storage_offset_bytes % dtype.dsize() == 0`. Both describe the tensor the
# caller handed in. The two `CHECK`s further down (`x->num >= 0`,
# `ydsize > 0`) stay internal: they restate what `is_dtype()` and the Var
# invariants already guarantee.
MIGRATED_REINTERPRET_VIEW_BOUNDARIES = {
    "src/ops/composite/reinterpret_view_op.cc": 8,
}

# 2026-09-11: 1 -> 3. Besides the shape check this entry was named for,
# `0f5eab25e` reclassified two more sites in the same constructor, both from
# a caller's arguments: `ASSERT(ns.is_binary())` -> `USER_CHECK` (the op name
# passed to `jt.binary`) and `CHECK(x_ok && y_ok)` -> `USER_CHECK` (a bitwise
# or shift op on float dtypes).
MIGRATED_BINARY_SHAPE_BOUNDARIES = {
    "src/ops/binary_op.cc": 3,
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
    "src/ops/unary_op.cc": 1,
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

# 2026-09-11: the two inline shape predicates these entries named are gone
# from both op files. `0f5eab25e` replaced them with one shared
# `cusparse_check_spmm_metadata()` in `cusparse_user_checks.h`, whose nine
# `USER_CHECK`s subsume them: `USER_CHECKop(A_col,==,xs[0])` became
# `USER_CHECK(k == b_rows)` (and now accounts for `trans_a`/`trans_b`), and
# `USER_CHECKop(xs,==,os)` became the output-shape check against the
# requested product. Still user boundaries, still catchable, now stated once
# instead of twice; the ledger follows them to the header.
MIGRATED_CUSPARSE_SPMM_SHAPE_USER_BOUNDARIES = {
    "backends/cuda/libraries/cusparse/include/cusparse_user_checks.h": 9,
}

MIGRATED_CUSPARSE_SPMMCOO_DTYPE_USER_BOUNDARIES = {
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

# 2026-09-11: 3 -> 4. `0f5eab25e` added
# `USER_CHECK(height >= 0 && width >= 0)` to the constructor. The requested
# output spatial size is a caller argument (`nn.conv_transpose` passes it
# through), and a negative one used to reach cuDNN as a descriptor error.
MIGRATED_CUDNN_CONV_BWD_X_FORMAT_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv_backward_x_op.cc": 4,
}

MIGRATED_CUDNN_CONV_BWD_X_FORMAT_COMPARE_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv_backward_x_op.cc": 1,
}

# 2026-09-11: 3 -> 4. `0f5eab25e` added `USER_CHECK(kh > 0 && kw > 0)`; the
# kernel size is a caller argument on this op just as it is on the forward.
MIGRATED_CUDNN_CONV_BWD_W_FORMAT_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv_backward_w_op.cc": 4,
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
    # 2026-09-11: was `CHECK(ret == CUTT_SUCCESS)`, twice. `221bc8ada` rewrote
    # the plan cache: the destroy path moved into `DestroyCuttPlan::destroy`,
    # where a failure is logged and counted (`cutt_plan_destroy_failures`)
    # instead of being fatal, and the surviving build-path check renamed
    # `ret` to `status`. One site now, same category.
    "backends/cuda/libraries/cutt/src/cutt_wrapper.cc": "CHECK(status == CUTT_SUCCESS)",
    "backends/cuda/kernels/cub/cub_test_op.cc": "ASSERT(cub_test_entry",
    "backends/cuda/kernels/cublas/cublas_test_op.cc": "ASSERT(cublas_test_entry",
    "backends/cuda/kernels/cudnn/cudnn_test_op.cc": "ASSERT(cudnn_test_entry",
}

# 2026-09-11: the three conv3d entries below all grew, all from `0f5eab25e`,
# all on constructor arguments: each gained the `xformat` check ("Not a valid
# format for cuDNN conv3d") that the 2-D ops already had, and the two
# backward ops additionally gained the spatial/kernel positivity checks their
# forward counterpart carries. 3 -> 4 here; 2 -> 5 for each backward op.
MIGRATED_CUDNN_CONV3D_X_RANK_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv3d_op.cc": 4,
}

MIGRATED_CUDNN_CONV3D_BWD_X_W_RANK_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv3d_backward_x_op.cc": 5,
}

MIGRATED_CUDNN_CONV3D_BWD_W_X_RANK_USER_BOUNDARIES = {
    "backends/cuda/kernels/cudnn/cudnn_conv3d_backward_w_op.cc": 5,
}

# 2026-09-11: 4 -> 10. The four cardinality checks this entry was named for
# are unchanged; `0f5eab25e` reclassified the six per-index `CHECK`s in the
# same constructor -- parameters vs moments/variances/gradients, by shape and
# by dtype -- to `USER_CHECK` and gave each a message. `jt.fused_adamw` takes
# those four lists straight from the caller, so a mismatch between them is
# caller input. (The same commit also turned the unmapped-backend `LOGf` in
# `jit_run` into `USER_ERROR`, which this count does not see.)
MIGRATED_FUSED_ADAMW_CARDINALITY_BOUNDARIES = {
    "src/ops/composite/fused_adamw_op.cc": 10,
}

MIGRATED_TERNARY_SHAPE_BOUNDARIES = {
    "src/ops/ternary_op.cc": 2,
}

MIGRATED_ITEM_USER_BOUNDARIES = {
    "src/core/var_holder.cc": 1,
}

# 2026-09-11: 2 -> 3. `0f5eab25e` reclassified
# `CHECK(!loss->flag(VarFlags::_first_order_only))` to `USER_CHECK`: asking
# for a gradient of a first-order-only result is something the caller's
# program did, and the message already tells them so. The same commit moved
# two `LOGf` in this file to `USER_ERROR`, which this count does not see.
MIGRATED_GRAD_DTYPE_USER_BOUNDARIES = {
    "src/core/grad.cc": 3,
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
    assert source.count("USER_CHECK(!loss->flag(VarFlags::_first_order_only))") == 1
    assert "\n    CHECK(!loss->flag(VarFlags::_first_order_only))" not in source
    assert '\n        CHECK(var->is_float() || var->dtype().is_complex())' not in source
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_GRAD_DTYPE_USER_BOUNDARIES[
        "src/core/grad.cc"]


def test_public_dimension_boundary_migration_is_explicit_and_bounded():
    counts = {}
    _LEDGER = MIGRATED_DIMENSION_BOUNDARIES
    for relative in _LEDGER:
        source = (ROOT / relative).read_text()
        # ``broadcast_to_op.cc`` also owns the five shape checks asserted by
        # ``MIGRATED_BROADCAST_SHAPE_BOUNDARIES``.  Count its two dimension
        # checks by their diagnostic so the independent ledgers do not overlap.
        if relative == "src/ops/broadcast_to_op.cc":
            actual = source.count('USER_CHECK(dim>=0 && dim<ydim)')
        else:
            actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
        counts[relative] = actual
    # Report every disagreement, not just the first. Three entries went stale
    # behind one that failed earlier in iteration order and stayed invisible
    # until that earlier one was fixed (2026-09-11).
    assert counts == dict(_LEDGER), (counts, dict(_LEDGER))
    assert sum(counts.values()) == 8


def test_public_shape_cardinality_migration_is_explicit_and_bounded():
    counts = {}
    _LEDGER = MIGRATED_SHAPE_CARDINALITY_BOUNDARIES
    for relative in _LEDGER:
        source = (ROOT / relative).read_text()
        actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
        if relative == "src/ops/composite/code_op.cc":
            # Backend provenance is independent of the shape/cardinality ledger.
            for guard in ("USER_CHECK(backend.empty()", "USER_CHECK(execution_backend() == expected)"):
                assert source.count(guard) == 1
                actual -= 1
        counts[relative] = actual
    # Report every disagreement, not just the first. Three entries went stale
    # behind one that failed earlier in iteration order and stayed invisible
    # until that earlier one was fixed (2026-09-11).
    assert counts == dict(_LEDGER), (counts, dict(_LEDGER))
    assert sum(counts.values()) == 43


def test_public_view_shape_migration_is_explicit_and_bounded():
    counts = {}
    _LEDGER = MIGRATED_VIEW_SHAPE_BOUNDARIES
    for relative in _LEDGER:
        source = (ROOT / relative).read_text()
        actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
        counts[relative] = actual
    # Report every disagreement, not just the first. Three entries went stale
    # behind one that failed earlier in iteration order and stayed invisible
    # until that earlier one was fixed (2026-09-11).
    assert counts == dict(_LEDGER), (counts, dict(_LEDGER))
    assert sum(counts.values()) == 11


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
    # Named, so the count above cannot drift into meaning something else.
    assert source.count("USER_CHECKop(xshape,==,yshape)") == 1
    assert source.count("USER_CHECK(ns.is_binary())") == 1
    assert "ASSERT(ns.is_binary())" not in source
    assert source.count("USER_CHECK(x_ok && y_ok)") == 1


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


def test_cusparse_spmm_shape_is_a_catchable_user_error():
    relative = "backends/cuda/libraries/cusparse/include/cusparse_user_checks.h"
    source = (ROOT / relative).read_text()
    actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert actual == MIGRATED_CUSPARSE_SPMM_SHAPE_USER_BOUNDARIES[relative]
    # The two successors of the predicates the op files used to carry inline.
    assert source.count("USER_CHECK(k == b_rows)") == 1
    assert source.count("USER_CHECK(output->shape[0] == m && output->shape[1] == n)") == 1
    assert "cuSPARSE inner matrix dimensions must match" in source
    assert "cuSPARSE output shape does not match the requested matrix product" in source
    assert "ASSERT" not in source
    # Both kernels have to reach it, or the boundary exists only on paper.
    for op in ("cusparse_spmmcsr_op.cc", "cusparse_spmmcoo_op.cc"):
        kernel = (ROOT / "backends/cuda/kernels/cusparse" / op).read_text()
        assert kernel.count("cusparse_check_spmm_metadata(") == 1
        assert "USER_CHECKop(xs,==,os)" not in kernel
        assert "USER_CHECKop(A_col,==,xs[0])" not in kernel


def test_cusparse_spmmcoo_dtype_user_boundary_migration_is_explicit_and_bounded():
    source = (ROOT / "backends/cuda/kernels/cusparse/cusparse_spmmcoo_op.cc").read_text()
    actual = source.count("USER_CHECK(")
    assert actual == MIGRATED_CUSPARSE_SPMMCOO_DTYPE_USER_BOUNDARIES[
        "backends/cuda/kernels/cusparse/cusparse_spmmcoo_op.cc"]
    negative = (ROOT / "tests/backends/cuda/test_cusparse_dtype.py").read_text()
    assert "test_coo_rejects_non_float_input" in negative
    assert "test_coo_rejects_mixed_input_dtypes" in negative


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
    counts = {}
    for relative in MIGRATED_CUB_OP_USER_BOUNDARIES:
        source = (ROOT / relative).read_text()
        counts[relative] = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
    assert counts == dict(MIGRATED_CUB_OP_USER_BOUNDARIES), counts
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
    assert cutt_wrapper.count("CHECK(status == CUTT_SUCCESS)") == 1
    assert "CHECK(ret == CUTT_SUCCESS)" not in cutt_wrapper
    assert cutt_wrapper.count("cuttDestroy failed with") == 1


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
    # Four cardinality checks, then six per-index ones; naming the split
    # keeps the single number above from hiding a swap between the cohorts.
    assert source.count("USER_CHECKop(parameters.size(),") == 4
    for other in ("moments", "variances", "gradients"):
        assert source.count("USER_CHECK(parameters[i]->shape == %s[i]->shape)" % other) == 1
        assert source.count("USER_CHECK(parameters[i]->dtype() == %s[i]->dtype())" % other) == 1
    assert "\n        CHECK(parameters[i]->" not in source


def test_public_ternary_shape_migration_is_explicit_and_bounded():
    counts = {}
    _LEDGER = MIGRATED_TERNARY_SHAPE_BOUNDARIES
    for relative in _LEDGER:
        source = (ROOT / relative).read_text()
        actual = source.count("USER_CHECK(") + source.count("USER_CHECKop(")
        counts[relative] = actual
    # Report every disagreement, not just the first. Three entries went stale
    # behind one that failed earlier in iteration order and stayed invisible
    # until that earlier one was fixed (2026-09-11).
    assert counts == dict(_LEDGER), (counts, dict(_LEDGER))
    assert sum(counts.values()) == 2


def test_reduce_mask_constructor_op_stays_an_internal_invariant():
    """Only one of the two `ns.is_binary()` checks guards caller input.

    `ReduceOp(x, op, dims, keepdims)` gets `op` from `jt.reduce(x, "sqrt")`, so
    its check is a user boundary. `ReduceOp(x, op, dims_mask, keepdims_mask)` is
    `@pybind(None)`: the generic overload is not generated for it, the aliases
    that are (`reduce_add`, `reduce_maximum`, ...) bind `op` to a compile-time
    constant, and its only in-tree callers pass `ns_add` or forward an `op` the
    other constructor already accepted. A failure there is the framework
    contradicting itself, so it is an `ASSERT` and reads as one.
    """
    source = (ROOT / "src/ops/reduce_op.cc").read_text()
    assert source.count("USER_CHECK(ns.is_binary())") == 1
    assert source.count("ASSERT(ns.is_binary())") == 1
    assert "// @pybind(None)" in (ROOT / "src/ops/reduce_op.h").read_text()


def test_library_plan_caches_do_not_blame_the_caller_for_their_own_key():
    """`device == key.device` is the framework checking itself.

    Each caller fills the key's device from a `cudaGetDevice` or placement read
    on the line before the call, so the predicate cannot be reached by anything
    a caller passed. As `USER_CHECK` it raised `UserError`, which invites a
    caller to catch and handle a framework bug.
    """
    cutt = (ROOT / "backends/cuda/libraries/cutt/src/cutt_wrapper.cc").read_text()
    assert cutt.count("ASSERT(device == key.device)") == 1
    assert "USER_CHECK(device == key.device)" not in cutt
    cufft = (ROOT / "backends/cuda/libraries/cufft/src/cufft_wrapper.cc").read_text()
    assert cufft.count("ASSERT(device == key.device)") == 1
    assert "USER_CHECK(device == key.device)" not in cufft
    hipblas = (ROOT / "backends/rocm/libraries/hipblas/hipblas_wrapper.cc").read_text()
    assert hipblas.count("ASSERTop(backend.current_device(), ==, device)") == 1
    assert "USER_CHECKop(backend.current_device()" not in hipblas
    for relative in ("backends/cuda/kernels/cutt/cutt_transpose_op.cc",
                     "backends/cuda/kernels/cufft/cufft_fft_op.cc"):
        assert "key.device = device;" in (ROOT / relative).read_text(), relative


def test_rocm_library_status_checks_are_not_user_errors():
    """A hipBLAS/HIP status says the library failed, not that input was bad.

    These three were `USER_CHECK`, so a driver fault, an allocation failure or a
    missing device surfaced as `UserError` -- "a caller supplied an unsupported
    value". Plain `CHECK`, which is what the cuTT wrapper already uses for
    `cuttPlan`, states the fault without assigning it to the caller.
    """
    header = (ROOT / "backends/rocm/libraries/hipblas/hipblas_wrapper.h").read_text()
    assert header.count("CHECK(status == HIPBLAS_STATUS_SUCCESS)") == 1
    assert "USER_CHECK(status == HIPBLAS_STATUS_SUCCESS)" not in header
    matmul = (ROOT / "backends/rocm/libraries/hipblas/hipblas_matmul_op.cc").read_text()
    assert matmul.count("CHECK(status == hipSuccess)") == 1
    assert "USER_CHECK(status == hipSuccess)" not in matmul
    cumsum = (ROOT / "backends/rocm/libraries/rocprim/rocprim_cumsum_op.cc").read_text()
    assert cumsum.count("CHECK(status == hipSuccess)") == 1
    assert "USER_CHECK(status == hipSuccess)" not in cumsum
    # The input boundary in the same file stays a user boundary.
    assert cumsum.count("USER_CHECK(x->shape.size() == 1 || x->shape.size() == 2)") == 1


def test_mpi_reduction_op_name_is_a_catchable_user_error():
    """`mpi_reduce(x, op="max")` is bad input, and used to read as a crash.

    `op` is a public argument of `mpi_reduce(x, op="add", root=0)` and
    `mpi_all_reduce(x, op="add")`. Under `ASSERT` an unsupported name produced
    the internal-invariant text -- "not an error in your program... please
    report it" -- for a spelling the caller chose.
    """
    for relative in ("backends/comm/mpi/ops/mpi_reduce_op.cc",
                     "backends/comm/mpi/ops/mpi_all_reduce_op.cc"):
        source = (ROOT / relative).read_text()
        assert source.count("USER_CHECK(op == ns_add)") == 1, relative
        assert "ASSERT(op == ns_add)" not in source, relative
        assert "Not supported MPI op" in source, relative
