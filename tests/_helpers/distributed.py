"""Subprocess launch helpers for MPI tests."""

import os
from pathlib import Path
import subprocess
import sys

import jittor as jt


_MIGRATED_TEST_PATHS = {
    "test_mpi": "tests/distributed/test_mpi.py",
    "test_mpi_batchnorm": "tests/distributed/test_mpi_batchnorm.py",
    "test_mpi_dtypes": "tests/distributed/test_mpi_dtypes.py",
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
    mpirun_path = jt.compile_extern.mpicc_path.replace("mpicc", "mpirun")
    command = [
        mpirun_path,
        "-np",
        str(num_procs),
        sys.executable,
        "-m",
        "pytest",
        "-q",
        os.fspath(repo_root / relative),
    ]
    print("run cmd:", " ".join(command))
    subprocess.run(command, cwd=os.fspath(repo_root), check=True)
