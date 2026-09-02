"""Subprocess launch helpers for MPI tests."""

import os
from pathlib import Path

import jittor as jt

from _helpers.child_process import run_mpi_python


_MIGRATED_TEST_PATHS = {
    "test_mpi": "tests/distributed/test_mpi.py",
    "test_mpi_batchnorm": "tests/distributed/test_mpi_batchnorm.py",
    "test_mpi_dtypes": "tests/distributed/test_mpi_dtypes.py",
    "test_mpi_half_reduce": "tests/distributed/test_mpi_half_reduce.py",
    "test_mpi_op": "tests/distributed/test_mpi_op.py",
    "test_mpi_var_ops": "tests/distributed/test_mpi_var_ops.py",
    "test_nccl_ops": "tests/distributed/test_nccl_ops.py",
    "test_resnet": "tests/models/test_resnet.py",
    "test_single_process_scope": "tests/distributed/test_single_process_scope.py",
}


def run_mpi_test(num_procs, name):
    if jt.compile_extern.inside_mpi():
        return
    relative = _MIGRATED_TEST_PATHS.get(name)
    if relative is None:
        raise ValueError("unknown migrated MPI test: {}".format(name))
    repo_root = Path(__file__).resolve().parents[2]
    # Every rank is started by mpirun, not by this process, so none of them
    # inherits the checkout on sys.path -- the helper pins PYTHONPATH for all.
    completed = run_mpi_python(
        num_procs, ["-m", "pytest", "-q", os.fspath(repo_root / relative)],
        cwd=repo_root, merge_stderr=True)
    print(completed.stdout)
    assert completed.returncode == 0, completed.stdout[-4000:]
