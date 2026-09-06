# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""One operator matrix, generated from the registry, run against every backend.

Why generated and not written down
----------------------------------
``tests/backends/parity/test_device_parity.py`` already compares CPU against
*an* accelerator, but its two axes both come from outside the registry: the
operator axis is ``opinfo.database.op_db`` (hand-maintained, user-level names
like ``sum``), and the backend axis is one ``_ACCEL`` string picked by probing
the build. Neither axis can notice a backend or an operator that the registry
knows about and the battery does not. The registry is now the thing that
decides what a backend claims -- ``BackendRegistry`` publishes the backend rows
and ``NativeOpRegistry::supported_ops`` publishes each backend's declared
implementations -- so the matrix is built by *reading* those two, and a cell
only exists because some backend declared it.

Three things follow from generating it, none of which a written-down list gives:

* An operator that enters ``backend_supported_ops`` with no probe and no
  recorded reason fails the gate. It cannot arrive unverified and unnoticed.
* A backend the core declares but this build did not register gets an explicit
  ``NOT_BUILT`` row, and a registered backend with no device gets
  ``NO_DEVICE``. Neither is a pytest skip. A skip is one line in a summary that
  reads identically to a pass (0.24); an unverified cell is a value in a table
  that the gate prints and asserts about.
* The backends' declared names do not have to agree for their *capabilities* to
  be compared. CPU declares ``argsort``/``arg_reduce``/``random`` while CUDA
  declares ``cub_argsort``/``cub_arg_reduce``/``curand_random`` -- different
  implementations of one contract -- and ``backend_supported_capabilities``
  is the axis on which those are the same cell.

The lazy-library trap
---------------------
The registry is not fully populated at import. Optional libraries are
registered by loaders (``jittor._runtime.backend_libraries``) that fire on
first use, so a snapshot taken at import time reports a *smaller* contract than
the build actually has: MKL's ``mkl_matmul``/``mkl_conv`` are absent from
``backend_supported_ops("cpu")`` until something calls
``nn.functional.matrix``, and cuTT's ``cutt_transpose`` is absent until
``setup_cutt`` runs. A matrix built on an unloaded registry would silently omit
exactly the entries whose verification is in question -- the same shape of
failure as ``setup_cutt()`` having no call site at all, which left six cuTT
cases permanently skipped. So ``load_optional_libraries()`` runs first and
records, per library, whether it loaded and why not.
"""

import os
import platform

import jittor as jt

from jittor._runtime.backend_libraries import LIBRARY_NAMES, get_library


#: A cell that ran and agreed with the oracle.
PASSED = "passed"
#: A cell that ran and disagreed. Carries the failure text.
FAILED = "failed"
#: This backend does not declare this operator, so there is nothing to verify.
#: Legitimately empty: CUDA has no ``argsort`` because it has ``cub_argsort``.
NOT_DECLARED = "not-declared"
#: The backend is declared by the core but this build registered no descriptor
#: for it. Verifying it needs a different build, not a different machine.
NOT_BUILT = "unverified:not-built"
#: The backend registered a descriptor but reports no device. Verifying it
#: needs hardware this machine does not have.
NO_DEVICE = "unverified:no-device"
#: The backend declares the operator and this build can reach it, but no probe
#: exists. Every one of these must appear in ``UNPROBED_REASONS``.
UNPROBED = "unverified:no-probe"

#: The statuses that mean "this cell was not verified here". Kept as a set so a
#: report can count them without re-listing the spellings.
UNVERIFIED = frozenset((NOT_BUILT, NO_DEVICE, UNPROBED))


def load_optional_libraries():
    """Force every optional backend library to register, before any snapshot.

    Returns ``{name: None}`` when the library loaded and ``{name: reason}``
    when it did not. A loader that raises is *recorded*, not propagated: a
    missing cuDNN on a CPU box must not stop the CPU column from being checked,
    and the reason is what turns "absent from the matrix" into "absent for a
    stated cause".
    """
    outcome = {}
    for name in LIBRARY_NAMES:
        try:
            module = get_library(name, load=True)
        except BaseException as error:            # noqa: BLE001 - recorded below
            outcome[name] = "%s: %s" % (type(error).__name__, str(error)[:200])
            continue
        outcome[name] = None if module is not None else "loader produced no module"
    return outcome


class BackendRow:
    """One backend's place in the matrix, with the reason if it has no cells."""

    def __init__(self, name, registered, devices, ops, capabilities):
        self.name = name
        self.registered = registered
        self.devices = devices
        self.ops = frozenset(ops)
        self.capabilities = frozenset(capabilities)

    @property
    def runnable(self):
        return self.registered and self.devices > 0

    @property
    def unverified_status(self):
        if not self.registered:
            return NOT_BUILT
        if self.devices <= 0:
            return NO_DEVICE
        return None

    def __repr__(self):
        return "BackendRow(%r, registered=%r, devices=%r, ops=%d, caps=%d)" % (
            self.name, self.registered, self.devices,
            len(self.ops), len(self.capabilities))


def backend_rows():
    """Every backend the core declares, in the order the core declares them.

    ``known_backends()`` is the universe and ``registered_backends()`` is the
    subset this build has. A name in the second and not the first would mean a
    descriptor published a spelling the enum does not know -- which is true
    today for ACL, whose descriptor registers as ``acl_legacy`` -- so those are
    carried as extra rows rather than dropped.
    """
    known = list(jt.core.known_backends())
    registered = list(jt.core.registered_backends())
    rows = []
    for name in known + [n for n in registered if n not in known]:
        if name in registered:
            devices = jt.core.backend_device_count(name)
            ops = jt.core.backend_supported_ops(name)
            capabilities = jt.core.backend_supported_capabilities(name)
        else:
            devices, ops, capabilities = 0, (), ()
        rows.append(BackendRow(name, name in registered, devices, ops, capabilities))
    return rows


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------
# A probe is the smallest graph that reaches one declared implementation, and
# it returns numbers so the cell is a *comparison* and not a smoke test. Each
# takes the ``jittor`` module and returns a numpy array; the caller runs it once
# per backend and compares against the CPU result.
#
# Probes are keyed by the registry's own spelling. That is the whole point: the
# key has to be the thing ``backend_supported_ops`` returns, or the matrix
# cannot tell whether a declared implementation is covered.


def _fixed(shape, offset=0.0, dtype="float32"):
    import numpy as np

    size = 1
    for dimension in shape:
        size *= dimension
    values = (np.arange(size, dtype="float64") * 0.37 + offset) % 7.0 - 3.0
    return np.asarray(values.reshape(shape), dtype=dtype)


def _probe_binary(jt_):
    return (jt_.array(_fixed((4, 5))) * jt_.array(_fixed((4, 5), 1.0))).numpy()


def _probe_unary(jt_):
    return jt_.array(_fixed((4, 5))).abs().sqrt().numpy()


def _probe_ternary(jt_):
    condition = jt_.array(_fixed((4, 5)) > 0)
    return condition.ternary(jt_.array(_fixed((4, 5))),
                             jt_.array(_fixed((4, 5), 2.0))).numpy()


def _probe_reduce(jt_):
    return jt_.array(_fixed((4, 5, 3))).sum(1).numpy()


def _probe_broadcast_to(jt_):
    return jt_.array(_fixed((4, 1, 3))).broadcast([4, 5, 3], [1]).numpy()


def _probe_array(jt_):
    return jt_.array(_fixed((3, 4))).numpy()


def _probe_getitem(jt_):
    return jt_.array(_fixed((6, 5)))[1:5, 2:4].numpy()


def _probe_setitem(jt_):
    value = jt_.array(_fixed((6, 5)))
    value[1:5, 2:4] = jt_.array(_fixed((4, 2), 3.0))
    return value.numpy()


def _probe_index(jt_):
    return jt_.index((4, 5), 1).numpy()


def _probe_reindex(jt_):
    return jt_.array(_fixed((4, 5))).reindex([4, 5, 2], ["i0", "i1"]).numpy()


def _probe_reindex_reduce(jt_):
    return jt_.array(_fixed((4, 5))).reindex_reduce("add", [4], ["i0"]).numpy()


def _probe_transpose(jt_):
    return jt_.array(_fixed((4, 5, 3))).transpose(2, 0, 1).numpy()


def _probe_fuse_transpose(jt_):
    # The fused spelling the optimizer produces: a transpose consumed by a
    # reduction, which is what puts `fuse_transpose` in the executed graph.
    return jt_.array(_fixed((4, 5, 3))).transpose(1, 2, 0).sum(0).numpy()


#: The three probes below go through ``jt.ops.*`` rather than the module-level
#: wrappers, and that is not a style choice. This matrix is keyed on registry
#: implementation names, so the call has to be the registry's own entry point:
#: the module-level ``jt.argsort`` is re-bound in Torch-compatibility mode,
#: where it returns a different arity, and a probe written against it measures
#: the shim rather than the backend. ``jt.ops.argsort`` is the same operator in
#: both process modes.


def _probe_where(jt_):
    coordinates = jt_.ops.where(jt_.array(_fixed((4, 5)) > 0), "int64")
    return coordinates[0].numpy()


def _probe_argsort(jt_):
    indices, values = jt_.ops.argsort(jt_.array(_fixed((4, 7))), 1, False, "int32")
    return values.numpy()


def _probe_arg_reduce(jt_):
    indices, values = jt_.ops.arg_reduce(jt_.array(_fixed((4, 7))), "max", 1, False)
    return values.numpy()


def _probe_safe_clip(jt_):
    return jt_.safe_clip(jt_.array(_fixed((4, 5))), -1.0, 1.0).numpy()


def _probe_clone(jt_):
    return jt_.array(_fixed((4, 5))).clone().numpy()


def _probe_copy(jt_):
    return jt_.array(_fixed((4, 5))).copy().numpy()


def _probe_reshape(jt_):
    return jt_.array(_fixed((4, 6))).reshape([3, 8]).numpy()


def _probe_reinterpret_view(jt_):
    return jt_.array(_fixed((4, 6))).view(3, 8).numpy()


def _probe_empty(jt_):
    # An allocation has no value to compare, so compare what it *does* promise:
    # shape and dtype. A cell still runs on every backend.
    import numpy as np

    value = jt_.empty((4, 5), "float32")
    return np.asarray([value.shape[0], value.shape[1],
                       float(str(value.dtype) == "float32")], dtype="float32")


def _probe_code(jt_):
    # One source per backend computing the same thing, which is exactly the
    # contract a hand-written kernel has to honour: `code` is the operator
    # whose CPU and accelerator bodies are written separately and can therefore
    # disagree without anything noticing.
    x = jt_.array(_fixed((4, 5)))
    return jt_.code(x.shape, x.dtype, [x],
                    cpu_src="""
                        for (int i = 0; i < in0_shape0; i++)
                        for (int j = 0; j < in0_shape1; j++)
                            @out(i, j) = @in0(i, j) * 2 + 1;
                    """,
                    cuda_src="""
                        __global__ static void probe(@ARGS_DEF) {
                            @PRECALC
                            for (int i = blockIdx.x; i < in0_shape0; i += gridDim.x)
                            for (int j = threadIdx.x; j < in0_shape1; j += blockDim.x)
                                @out(i, j) = @in0(i, j) * 2 + 1;
                        }
                        probe<<<4, 32>>>(@ARGS);
                    """).numpy()


def _probe_numpy_code(jt_):
    def forward(np, data):
        source = data["inputs"][0]
        target = data["outputs"][0]
        np.multiply(source, 2, out=target)
        np.add(target, 1, out=target)

    x = jt_.array(_fixed((4, 5)))
    return jt_.numpy_code(x.shape, x.dtype, [x], forward).numpy()


def _probe_fused(jt_):
    # Fusion is not requested, it is produced: two elementwise ops and a
    # reduction over one graph become one fused kernel.
    x = jt_.array(_fixed((8, 9)))
    return ((x * 2 + 1).abs().sum(1)).numpy()


def _probe_candidate(jt_):
    x = jt_.array(_fixed((16,)))
    return jt_.candidate(x.reshape([16, 1]), "(@x(j, 0) > @x(i, 0))").numpy()


def _probe_fetch(jt_):
    import numpy as np

    captured = []
    jt_.fetch(jt_.array(_fixed((4, 5))), lambda value: captured.append(value))
    jt_.sync_all(True)
    assert captured, "fetch produced no callback"
    return np.asarray(captured[0], dtype="float32")


def _probe_device_copy(jt_):
    # The op that moves storage between the host and a device. On CPU it is the
    # identity copy; on an accelerator it is the transfer itself.
    value = jt_.array(_fixed((4, 5)))
    value.sync()
    return value.numpy()


def _probe_tape(jt_):
    x = jt_.array(_fixed((4, 5)))
    x.requires_grad = True
    taped = jt_.tape(x)
    return jt_.grad((taped * taped).sum(), x).numpy()


def _probe_random(jt_):
    # A random draw cannot be compared value-for-value across two generators
    # (CPU uses its own PRNG, CUDA uses cuRAND), so the cell compares the
    # distribution contract the backends *do* share. The comparison is still
    # numeric and still fails on a broken kernel: a generator returning zeros,
    # constants, or values outside [0, 1) moves these four numbers.
    import numpy as np

    jt_.set_seed(20260906)
    samples = jt_.random((65536,), "float32", "uniform").numpy()
    assert np.isfinite(samples).all(), "random produced non-finite values"
    return np.asarray([
        float(np.all((samples >= 0) & (samples < 1))),
        round(float(np.mean(samples)), 2),
        round(float(np.var(samples)), 2),
        float(len(np.unique(samples)) > 60000),
    ], dtype="float32")


def _probe_matmul(jt_):
    a = jt_.array(_fixed((16, 24)))
    b = jt_.array(_fixed((24, 12), 5.0))
    return jt_.matmul(a, b).numpy()


def _probe_batched_matmul(jt_):
    a = jt_.array(_fixed((3, 16, 24)))
    b = jt_.array(_fixed((3, 24, 12), 5.0))
    return jt_.matmul(a, b).numpy()


def _probe_conv2d(jt_):
    x = jt_.array(_fixed((2, 3, 9, 10)))
    w = jt_.array(_fixed((4, 3, 3, 3), 2.0))
    return jt_.nn.conv2d(x, w, stride=1, padding=1).numpy()


def _probe_conv2d_backward(jt_):
    import numpy as np

    x = jt_.array(_fixed((2, 3, 9, 10)))
    w = jt_.array(_fixed((4, 3, 3, 3), 2.0))
    x.requires_grad = True
    w.requires_grad = True
    loss = (jt_.nn.conv2d(x, w, stride=1, padding=1) * 0.5).sum()
    grad_x, grad_w = jt_.grad(loss, [x, w])
    return np.concatenate([grad_x.numpy().ravel(), grad_w.numpy().ravel()])


def _probe_cumsum(jt_):
    return jt_.cumsum(jt_.array(_fixed((4, 9))), dim=1).numpy()


#: ``registry op name -> probe``.
#:
#: The keys are what ``backend_supported_ops`` returns. Where two backends give
#: one contract two implementation names, both names map to the same probe --
#: that is how the ``argsort``/``cub_argsort`` pair becomes one comparable cell
#: instead of two half-covered ones.
PROBES = {
    "array": _probe_array,
    "arg_reduce": _probe_arg_reduce,
    "cub_arg_reduce": _probe_arg_reduce,
    "argsort": _probe_argsort,
    "cub_argsort": _probe_argsort,
    "binary": _probe_binary,
    "broadcast_to": _probe_broadcast_to,
    "candidate": _probe_candidate,
    "clone": _probe_clone,
    "code": _probe_code,
    "copy": _probe_copy,
    "cub_cumsum": _probe_cumsum,
    "cublas_matmul": _probe_matmul,
    "mkl_matmul": _probe_matmul,
    "cublas_batched_matmul": _probe_batched_matmul,
    "mkl_batched_matmul": _probe_batched_matmul,
    "cudnn_conv": _probe_conv2d,
    "mkl_conv": _probe_conv2d,
    "cudnn_conv_backward_x": _probe_conv2d_backward,
    "cudnn_conv_backward_w": _probe_conv2d_backward,
    "mkl_conv_backward_x": _probe_conv2d_backward,
    "mkl_conv_backward_w": _probe_conv2d_backward,
    "curand_random": _probe_random,
    "random": _probe_random,
    "cub_where": _probe_where,
    "where": _probe_where,
    "device_copy": _probe_device_copy,
    "empty": _probe_empty,
    "fetch": _probe_fetch,
    "fuse_transpose": _probe_fuse_transpose,
    "fused": _probe_fused,
    "getitem": _probe_getitem,
    "index": _probe_index,
    "numpy_code": _probe_numpy_code,
    "reduce": _probe_reduce,
    "reindex": _probe_reindex,
    "reindex_reduce": _probe_reindex_reduce,
    "reinterpret_view": _probe_reinterpret_view,
    "reshape": _probe_reshape,
    "safe_clip": _probe_safe_clip,
    "setitem": _probe_setitem,
    "tape": _probe_tape,
    "transpose": _probe_transpose,
    "cutt_transpose": _probe_transpose,
    "ternary": _probe_ternary,
    "unary": _probe_unary,
}


#: ``registry op name -> why this declared implementation has no probe``.
#:
#: This is the honest half of the matrix and the reason it can be a ratchet: an
#: operator is either probed or listed here with a cause, and
#: ``test_backend_contract_matrix.py`` fails on a declared operator that is in
#: neither. "Not probed" is then a decision somebody wrote down, not the
#: default state of anything new.
UNPROBED_REASONS = {
    # Multi-rank: a single-process matrix cannot form a communicator, and the
    # collectives have their own launcher-based gate (`nox -s mpi` / `nccl`).
    "mpi_all_reduce": "needs an MPI communicator; covered by the mpi/nccl gates",
    "mpi_broadcast": "needs an MPI communicator; covered by the mpi/nccl gates",
    "mpi_reduce": "needs an MPI communicator; covered by the mpi/nccl gates",
    "mpi_test": "needs an MPI communicator; covered by the mpi/nccl gates",
    # Library self-tests: these exist to prove a library links and are not
    # numerical contracts, so there is no cross-backend result to compare.
    "cub_test": "library link self-test, not a numerical contract",
    "cublas_test": "library link self-test, not a numerical contract",
    "cudnn_test": "library link self-test, not a numerical contract",
    "mkl_test": "library link self-test, not a numerical contract",
    "cutt_test": "library link self-test, not a numerical contract",
    # Declared only by an accelerator, and their CPU counterpart is a composed
    # graph rather than a declared implementation -- so there is no second cell
    # to compare against and the pairing has to be built before probing.
    "cublas_acc_matmul": "accumulating matmul has no declared CPU counterpart to compare against",
    "cudnn_conv3d": "3d convolution has no declared CPU counterpart to compare against",
    "cudnn_conv3d_backward_x": "3d convolution has no declared CPU counterpart to compare against",
    "cudnn_conv3d_backward_w": "3d convolution has no declared CPU counterpart to compare against",
    "cudnn_rnn": "recurrent kernel has no declared CPU counterpart to compare against",
    "cudnn_rnn_backward_x": "recurrent kernel has no declared CPU counterpart to compare against",
    "cusparse_spmmcoo": "sparse matmul has no declared CPU counterpart to compare against",
    "cusparse_spmmcsr": "sparse matmul has no declared CPU counterpart to compare against",
    # `jittor.nn.legacy_complex._fft2` is the only caller and it is guarded by
    # `has_cuda`: there is no CPU FFT implementation in the registry, so a
    # cross-backend cell cannot be formed. Its own coverage is tests/ops/test_fft_op.py.
    "cufft_fft": "no declared CPU FFT to compare against; covered by tests/ops/test_fft_op.py",
    "fused_adamw": "fused optimizer step is compared by tests/optim, not by an op cell",
    # Graph plumbing with no standalone value: `tapes` is only reachable from a
    # multi-output custom gradient, which `tape` already exercises.
    "tapes": "only reachable through a multi-output custom gradient; `tape` covers the mechanism",
}


def cell_status(row, op, evaluate):
    """One matrix cell: the status, and the detail a report needs.

    ``evaluate(row, op)`` runs the probe on this backend and compares it
    against the oracle, returning ``(ok, detail)``. It is called only for a
    cell that reaches ``PASSED``/``FAILED``.

    Ordering here *is* the honesty rule. A backend that cannot run gets its own
    unverified status even for operators it declares, so a missing device can
    never be reported as a pass; and an operator with no probe is unverified
    rather than absent, so it has to be excused in writing.
    """
    if op not in row.ops:
        return NOT_DECLARED, ""
    unverified = row.unverified_status
    if unverified is not None:
        return unverified, "backend %s: %s" % (row.name, unverified)
    if op not in PROBES:
        return UNPROBED, UNPROBED_REASONS.get(op, "")
    ok, detail = evaluate(row, op)
    return (PASSED if ok else FAILED), detail


def build_matrix(rows, ops, evaluate):
    """``{(backend, op): (status, detail)}`` for the whole matrix.

    Every backend gets a cell for every operator, including the backends that
    cannot run and the operators they do not declare. A dense matrix is the
    point: a sparse one cannot distinguish "checked and fine" from "never
    considered", which is the distinction the whole gate exists to make.
    """
    return {
        (row.name, op): cell_status(row, op, evaluate)
        for row in rows
        for op in ops
    }


def declared_ops(rows):
    """Every operator name any backend declares, sorted.

    The operator axis of the matrix. Generated, so it grows when a backend
    starts declaring something and the gate then demands a probe or a reason.
    """
    names = set()
    for row in rows:
        names |= set(row.ops)
    return sorted(names)


def format_matrix(rows, statuses, library_outcome=None):
    """The table the gate prints, so an unverified cell is visible.

    A pytest summary reports skips as a count without saying what went
    unchecked. This prints the backend rows with their per-status totals, then
    names the unverified operators, so a green run still says which parts of
    the contract this machine could not answer for.
    """
    lines = ["", "cross-backend contract matrix"]
    lines.append("  %-8s %-24s %s" % ("backend", "state", "cells"))
    for row in rows:
        counts = {}
        for (backend, _op), (status, _detail) in statuses.items():
            if backend == row.name:
                counts[status] = counts.get(status, 0) + 1
        state = row.unverified_status or ("registered, %d device(s)" % row.devices)
        summary = ", ".join("%s=%d" % item for item in sorted(counts.items())
                            if item[0] != NOT_DECLARED)
        lines.append("  %-8s %-24s %s" % (row.name, state, summary or "-"))
    unverified = sorted(
        (backend, op, status, detail)
        for (backend, op), (status, detail) in statuses.items()
        if status in UNVERIFIED)
    lines.append("  unverified cells: %d" % len(unverified))
    for backend, op, status, detail in unverified:
        if status == UNPROBED:
            lines.append("    %s/%s %s -- %s" % (backend, op, status, detail))
    if library_outcome:
        missing = sorted(name for name, reason in library_outcome.items() if reason)
        lines.append("  optional libraries not loaded: %s"
                     % (", ".join(missing) if missing else "none"))
        # The reason and not just the name: "cutt is not loaded" reads like a
        # fact about the machine, while the reason said it was a compile error
        # in the wrapper -- a build breakage wearing a missing-hardware
        # costume. Printing the cause is what tells those two apart.
        for name in missing:
            lines.append("    %s -- %s" % (name, library_outcome[name]))
    lines.append("  host: %s %s" % (platform.system(), platform.machine()))
    lines.append("  JITTOR_HOME set: %s" % bool(os.environ.get("JITTOR_HOME")))
    return "\n".join(lines)
