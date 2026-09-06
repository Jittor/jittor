# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The cross-backend contract gate: one matrix, generated from the registry.

What this adds that ``test_device_parity.py`` cannot
----------------------------------------------------
That battery compares CPU against one accelerator over a hand-maintained
``op_db``. Both of its axes are outside the registry, so it cannot notice a
backend or a declared implementation that the registry knows about and it does
not. Here both axes are read from the registry -- ``known_backends()`` and
``registered_backends()`` for the backend axis, ``backend_supported_ops()`` for
the operator axis -- which buys three checks the parity battery has no way to
make:

* ``test_every_declared_operator_is_probed_or_excused``: a new implementation
  entering ``backend_supported_ops`` fails this gate until somebody either
  probes it or writes down why it cannot be probed.
* ``test_every_known_backend_appears_with_an_explicit_state``: a backend the
  core declares and this build did not register is a printed
  ``unverified:not-built`` row, not an absence.
* ``test_a_cuda_build_actually_executed_its_cuda_column``: a build that *has*
  CUDA may not report the CUDA column as unverified. This is the check that
  114 consecutive hand-offs needed and did not have: they recorded "this
  machine has no CUDA" for a machine with eight usable GPUs, and no gate
  contradicted them, because a skip and a pass look the same in a summary.

The whole file therefore runs on a CPU-only box as well: the matrix is still
built, the CPU column is still executed and compared, and the accelerator
column is reported as unverified with a reason. It does not skip itself, so it
cannot become another entry that is only ever green because it did nothing.
"""

import numpy as np
import pytest

import jittor as jt

from _helpers.common import net_scaled_max_err, per_element_max_rel_err
from backends.parity import backend_contract_matrix as contract


#: Same shape of tolerance as the device-parity battery: accumulation-order
#: round-off between two kernels is allowed, anything larger is a kernel bug.
#: Two metrics because they fail to different bug classes -- the net-scaled one
#: catches gross divergence, the per-element one catches a single wrong
#: coordinate whose magnitude is far below the peak (the scatter/int-reduce
#: silent-wrong class).
VALUE_TOL = 2e-4
PER_ELEMENT_TOL = 2e-3
PER_ELEMENT_ATOL = 1e-3


def _run_probe(probe, use_cuda):
    with jt.flag_scope(use_cuda=use_cuda):
        value = probe(jt)
        jt.sync_all(True)
    return np.asarray(value)


def _compare(actual, reference):
    """``(ok, detail)`` for one probe's numbers against the CPU oracle."""
    if actual.shape != reference.shape:
        return False, "shape %r != oracle %r" % (actual.shape, reference.shape)
    if actual.dtype.kind not in "fc" or reference.dtype.kind not in "fc":
        if not np.array_equal(actual, reference):
            return False, "exact comparison failed: %r != %r" % (actual, reference)
        return True, ""
    net = net_scaled_max_err(actual, reference)
    if not net < VALUE_TOL:
        return False, "net-scaled error %.3e exceeds %.3e" % (net, VALUE_TOL)
    element = per_element_max_rel_err(actual, reference, atol=PER_ELEMENT_ATOL)
    if not element < PER_ELEMENT_TOL:
        return False, ("per-element relative error %.3e exceeds %.3e -- a "
                       "sub-peak coordinate is wrong" % (element, PER_ELEMENT_TOL))
    return True, ""


class Snapshot:
    """The completed matrix, plus what it took to build it.

    Built once for the whole module so that every assertion below reads the
    *same* matrix, and so the report can name every unverified cell no matter
    which subset of tests ``-k`` selected.
    """

    def __init__(self):
        self.libraries = contract.load_optional_libraries()
        self.rows = contract.backend_rows()
        self.ops = contract.declared_ops(self.rows)
        self.oracles = {}
        self.oracle_failures = {}
        self.statuses = contract.build_matrix(self.rows, self.ops, self._evaluate)

    @property
    def accelerator_rows(self):
        return [row for row in self.rows if row.name != "cpu"]

    def row(self, name):
        return next(row for row in self.rows if row.name == name)

    def cells_for(self, op):
        return {row.name: self.statuses[(row.name, op)] for row in self.rows}

    def _oracle(self, op):
        """The CPU result for this probe, computed once and reused.

        The oracle is a CPU *execution* of the probe, not a CPU *declaration*
        of the operator. That distinction is what lets one cell compare
        ``cub_argsort`` against CPU's ``argsort``: the two backends give one
        contract two implementation names, and the probe is the contract.
        """
        if op in self.oracles or op in self.oracle_failures:
            return self.oracles.get(op)
        try:
            self.oracles[op] = _run_probe(contract.PROBES[op], use_cuda=0)
        except BaseException as error:           # noqa: BLE001 - becomes a cell
            self.oracle_failures[op] = "%s: %s" % (type(error).__name__,
                                                   str(error)[:400])
            return None
        return self.oracles[op]

    def _evaluate(self, row, op):
        reference = self._oracle(op)
        if reference is None:
            return False, "CPU oracle failed: " + self.oracle_failures[op]
        if row.name == "cpu":
            # The CPU cell is the oracle's own cell. It cannot be compared
            # against itself, so what it asserts is that the probe produced
            # usable numbers: a NaN oracle would make every accelerator cell
            # above it meaningless while still reading as a pass.
            if reference.dtype.kind == "f" and not np.isfinite(reference).all():
                return False, "CPU probe produced non-finite values"
            return True, ""
        try:
            actual = _run_probe(contract.PROBES[op], use_cuda=1)
        except BaseException as error:           # noqa: BLE001 - becomes a cell
            return False, "%s: %s" % (type(error).__name__, str(error)[:400])
        return _compare(actual, reference)


@pytest.fixture(scope="module")
def snapshot():
    return Snapshot()


@pytest.mark.parametrize("op", sorted(contract.PROBES))
def test_operator_contract_agrees_across_every_runnable_backend(snapshot, op):
    """One operator, every backend row, all in one test id.

    Parametrized on the *probe* names rather than on the registry, because
    collection must not import a built jittor to decide what exists. The
    registry-driven half of the check -- that no declared operator is missing
    from these parameters -- is the next test.
    """
    if op not in snapshot.ops:
        pytest.skip("no registered backend declares %s in this build" % op)
    failures = [
        "%s/%s %s -- %s" % (backend, op, status, detail)
        for backend, (status, detail) in snapshot.cells_for(op).items()
        if status == contract.FAILED
    ]
    assert not failures, "\n".join(failures)
    verified = [backend for backend, (status, _detail)
                in snapshot.cells_for(op).items() if status == contract.PASSED]
    assert verified, (
        "%s is declared by %s but no backend cell for it ran; a declared "
        "operator with no executed cell is the failure this gate exists to "
        "report" % (op, sorted(row.name for row in snapshot.rows if op in row.ops))
    )


def test_every_declared_operator_is_probed_or_excused(snapshot):
    """The ratchet: the registry decides what needs covering, not this file.

    An implementation that enters ``backend_supported_ops`` arrives here with
    no probe and no reason, and fails. That is the property a written-down
    operator list cannot have, and the reason this matrix is generated.
    """
    unaccounted = sorted(
        op for op in snapshot.ops
        if op not in contract.PROBES and op not in contract.UNPROBED_REASONS
    )
    assert not unaccounted, (
        "these implementations are declared by a registered backend but are "
        "neither probed nor listed in UNPROBED_REASONS with a cause: %s"
        % unaccounted
    )
    empty = sorted(op for op, reason in contract.UNPROBED_REASONS.items()
                   if not reason.strip())
    assert not empty, "an excused operator needs a stated cause: %s" % empty
    both = sorted(set(contract.PROBES) & set(contract.UNPROBED_REASONS))
    assert not both, "an operator cannot be both probed and excused: %s" % both


def test_no_probe_or_excuse_is_dead_weight(snapshot):
    """A probe for an operator no registered backend declares is not coverage.

    This is the other direction of the same rule. ``PROBES`` may carry entries
    for backends this build does not have -- ``mkl_matmul`` on a build without
    MKL, ``cutt_transpose`` before cuTT loads -- and those are legitimate. What
    is not legitimate is an excuse for an operator that no longer exists
    anywhere in the tree, because it silently narrows what the ratchet above
    demands.
    """
    tree_wide = set()
    for row in snapshot.rows:
        tree_wide |= set(row.ops)
    stale = sorted(op for op in contract.UNPROBED_REASONS if op not in tree_wide)
    # Excuses for implementations belonging to a library that did not load are
    # kept: this build's registry cannot see them, which is not the same as
    # their being gone.
    unloaded = tuple(name for name, reason in snapshot.libraries.items() if reason)
    orphaned = [op for op in stale
                if not any(op.startswith(name) for name in unloaded)]
    assert not orphaned, (
        "these excuses name implementations no registered backend declares and "
        "no unloaded library owns, so they excuse nothing: %s" % orphaned
    )


def test_every_known_backend_appears_with_an_explicit_state(snapshot):
    """No backend is silently absent from the matrix.

    ``known_backends()`` is the core's declared universe; every name in it gets
    a row, and a row that cannot run carries the status saying which of the two
    reasons applies -- the build has no descriptor for it, or the machine has
    no device. Neither is a pytest skip, because a skip is indistinguishable
    from a pass in a summary (0.24).
    """
    known = set(jt.core.known_backends())
    assert known, "the core declares no backends"
    rows = {row.name: row for row in snapshot.rows}
    assert known <= set(rows), (
        "known_backends() names backends the matrix has no row for: %s"
        % sorted(known - set(rows))
    )
    for name in sorted(known):
        row = rows[name]
        status = row.unverified_status
        assert row.runnable or status in (contract.NOT_BUILT, contract.NO_DEVICE), (
            "%s is neither runnable nor carrying an unverified reason" % name)
        if not row.runnable:
            cells = {op: snapshot.statuses[(name, op)][0] for op in snapshot.ops}
            assert contract.PASSED not in cells.values(), (
                "%s cannot run here yet has passing cells: %s"
                % (name, sorted(op for op, value in cells.items()
                                if value == contract.PASSED)))


def test_backends_without_hardware_are_marked_unverified_not_skipped(snapshot):
    """The 0.24 rule, as an assertion about this file's own output.

    Ascend/ROCm/Corex have no hardware here. The requirement is not that they
    be skipped quietly but that the matrix say so, with which of the two
    reasons applies, for every operator the core declares.
    """
    unrunnable = [row for row in snapshot.rows if not row.runnable]
    for row in unrunnable:
        statuses = {snapshot.statuses[(row.name, op)][0] for op in snapshot.ops}
        assert statuses <= {contract.NOT_BUILT, contract.NO_DEVICE,
                            contract.NOT_DECLARED}, (row.name, sorted(statuses))
    report = contract.format_matrix(snapshot.rows, snapshot.statuses,
                                    snapshot.libraries)
    for row in unrunnable:
        assert row.name in report, (
            "%s is unverified and the report does not name it" % row.name)


def test_a_cuda_build_actually_executed_its_cuda_column(snapshot):
    """A build with CUDA may not report its CUDA column as unverified.

    This is the assertion that would have caught the long-running claim that
    this hardware had no CUDA. ``jt.has_cuda`` is decided by the build, not by
    this file, so when it is true the matrix has to show an executed
    accelerator column -- and when it is false, the column has to carry a
    reason. Either way the outcome is stated rather than assumed.
    """
    accelerators = [row for row in snapshot.accelerator_rows if row.registered]
    if not jt.has_cuda:
        for row in accelerators:
            assert not row.runnable or row.devices > 0
        pytest.skip("no CUDA in this build; the accelerator column is "
                    "reported unverified in the matrix report")
    runnable = [row for row in accelerators if row.runnable]
    assert runnable, (
        "jt.has_cuda is true but no registered accelerator row is runnable; "
        "registered=%s" % [(row.name, row.devices) for row in accelerators])
    passed = sum(1 for row in runnable for op in snapshot.ops
                 if snapshot.statuses[(row.name, op)][0] == contract.PASSED)
    assert passed >= 20, (
        "a CUDA build executed only %d accelerator cells; the accelerator "
        "column is effectively unverified" % passed)


def test_optional_libraries_are_loaded_before_the_registry_is_read(snapshot):
    """The registry snapshot must not be taken before the lazy loaders fire.

    Optional libraries register their operators on first use, so a matrix built
    at import time under-reports the contract -- MKL's operators are absent
    until something calls into ``nn.functional.matrix``. That is the same shape
    of failure as ``setup_cutt()`` having had no call site, which left six cuTT
    cases permanently skipped while reading green. So the snapshot force-loads
    every library and records the ones that did not load, with a reason.
    """
    assert snapshot.libraries, "no optional libraries were considered"
    unnamed = sorted(name for name, reason in snapshot.libraries.items()
                     if reason is not None and not reason.strip())
    assert not unnamed, "a library that did not load needs a reason: %s" % unnamed
    if jt.has_cuda:
        loaded = {name for name, reason in snapshot.libraries.items()
                  if reason is None}
        # cuTT is in this set deliberately. Its wrapper stopped compiling when
        # the CUDA backend moved (`stream_compat.h` was not on cuTT's own
        # include path, only on the one `setup_cuda_lib` builds), and the
        # symptom was a library that reported itself unavailable -- so
        # tests/backends/cuda/test_cutt*.py skipped, on a machine with cuTT,
        # and read as green. A build error must not be able to hide as
        # absent hardware.
        assert {"cublas", "cudnn", "curand", "cub", "cufft", "cusparse",
                "cutt"} <= loaded, (
            "a CUDA build did not load its own libraries: %s"
            % {name: reason for name, reason in snapshot.libraries.items()
               if reason})


def test_a_registered_backend_without_a_device_is_unverified_not_passed():
    """The no-hardware path, checked on hardware that has some.

    On this machine ACL/ROCm/Corex exercise ``NOT_BUILT``, but nothing
    exercises ``NO_DEVICE`` -- a backend whose descriptor registered and whose
    driver reports zero devices, which is what an Ascend or ROCm box without a
    card looks like. Leaving that branch to be discovered on the machine that
    has the problem is how "unverified" quietly becomes "passed", so it is
    driven here from synthetic rows instead. ``BackendRow`` and ``cell_status``
    are pure, so this needs no device and no build.
    """
    def never(row, op):
        raise AssertionError("a backend with no device must not be probed: %s/%s"
                             % (row.name, op))

    no_device = contract.BackendRow("ascend", registered=True, devices=0,
                                    ops=("binary", "reduce"), capabilities=())
    not_built = contract.BackendRow("elsewhere", registered=False, devices=0,
                                    ops=(), capabilities=())
    assert not no_device.runnable and not not_built.runnable
    assert no_device.unverified_status == contract.NO_DEVICE
    assert not_built.unverified_status == contract.NOT_BUILT

    matrix = contract.build_matrix([no_device, not_built],
                                   ["binary", "reduce", "unary"], never)
    # Declared but unrunnable: unverified with the device as the reason. The
    # important half is that it is not PASSED -- `never` above proves no probe
    # ran, so a pass could only have come from assuming one.
    assert matrix[("ascend", "binary")][0] == contract.NO_DEVICE
    assert matrix[("ascend", "reduce")][0] == contract.NO_DEVICE
    # Not declared by this backend: nothing to verify, and that is not the same
    # claim as "unverified".
    assert matrix[("ascend", "unary")][0] == contract.NOT_DECLARED
    assert matrix[("elsewhere", "binary")][0] == contract.NOT_DECLARED
    assert contract.PASSED not in {status for status, _ in matrix.values()}
    report = contract.format_matrix([no_device, not_built], matrix,
                                    {"mkl": None, "cutt": "compile failed"})
    assert "ascend" in report and contract.NO_DEVICE in report
    assert "elsewhere" in report and contract.NOT_BUILT in report
    assert "cutt -- compile failed" in report


def test_matrix_report_states_the_coverage_it_achieved(snapshot, capsys):
    """Print the matrix, and require it to be a statement rather than a count.

    A gate that reports "N passed" says nothing about what it did not check.
    The report names every backend row with its state, counts the unverified
    cells, and lists the operators that have no probe -- so a green run still
    carries what this machine could not answer for.
    """
    report = contract.format_matrix(snapshot.rows, snapshot.statuses,
                                    snapshot.libraries)
    with capsys.disabled():
        print(report)
    for row in snapshot.rows:
        assert row.name in report
    assert "unverified cells:" in report
    verified_backends = sorted(row.name for row in snapshot.rows if row.runnable)
    assert "cpu" in verified_backends, "the CPU column did not run"
    passed = sum(1 for value in snapshot.statuses.values()
                 if value[0] == contract.PASSED)
    assert passed >= len(verified_backends) * 20, (
        "only %d cells passed across %s; the matrix verified almost nothing"
        % (passed, verified_backends))
