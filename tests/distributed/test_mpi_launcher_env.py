# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""MPI launcher detection and the single source of rank/world_size (6.B15).

Two failure modes are covered.

1. Only ``OMPI_COMM_WORLD_SIZE`` was recognized, so a job started with MPICH,
   Intel MPI, MVAPICH or srun looked like a plain single-process run and every
   rank trained on its own.

2. The same question was answered independently in C++ (``detect_inside_mpi``)
   and in Python (``inside_mpi``), and rank/world_size were copied from
   ``compile_extern`` into ``jittor`` and again into ``core_api``. Any branch
   that updated one and not the others produced a process where C++ believed it
   was rank 0 and Python believed it was rank 2 -- which does not crash, it just
   exchanges tensors with the wrong peers.
"""
import os
from pathlib import Path
import re
import unittest

import jittor as jt
from jittor import compile_extern

from _helpers.child_process import run_python_child

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WRAPPER_SRC = (_REPO_ROOT / "python" / "jittor" / "extern" / "mpi" / "src"
                / "mpi_wrapper.cc")

_LAUNCHER_ENV_KEYS = (
    "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE", "PMIX_RANK",
    "SLURM_NTASKS", "SLURM_NPROCS", "JT_MPI",
)


def _run_child(env_overrides, code):
    env = dict(os.environ)
    for key in _LAUNCHER_ENV_KEYS:
        env.pop(key, None)
    env.update(env_overrides)
    # inherit=False: this caller *removed* variables, and merging back onto
    # os.environ would put every one of them straight back.
    return run_python_child(["-c", code], env=env, inherit=False,
                            cwd=_REPO_ROOT, merge_stderr=True, timeout=1800)


class TestMpiLauncherDetection(unittest.TestCase):

    def test_python_and_cxx_lists_do_not_drift(self):
        """The C++ and Python launcher lists must name the same variables.

        Python has to answer before the MPI module is even built, so the lists
        genuinely cannot be shared; pinning them to each other here is the
        substitute.
        """
        source = _WRAPPER_SRC.read_text()
        body = source[source.index("static bool detect_inside_mpi()"):]
        body = body[:body.index("\nstruct mpi_initer")]
        cxx_vars = set(re.findall(r'"([A-Z0-9_]+)"', body))
        py_vars = (set(compile_extern._MPI_LAUNCHER_VARS)
                   | set(compile_extern._MPI_LAUNCHER_SIZE_VARS)
                   | {"JT_MPI"})
        self.assertEqual(cxx_vars, py_vars)

    def test_inside_mpi_recognizes_every_launcher(self):
        cases = [
            ({"OMPI_COMM_WORLD_SIZE": "2"}, True),   # Open MPI
            ({"PMI_SIZE": "2"}, True),               # MPICH / Intel MPI
            ({"MV2_COMM_WORLD_SIZE": "2"}, True),    # MVAPICH2
            ({"PMIX_RANK": "1"}, True),              # PMIx launchers
            ({"SLURM_NTASKS": "4"}, True),           # srun, multi-task
            ({"SLURM_NPROCS": "4"}, True),
            # srun is also used for ordinary single-task jobs; those are not
            # MPI jobs and must not trigger MPI_Init.
            ({"SLURM_NTASKS": "1"}, False),
            ({}, False),
            # Explicit declaration wins in both directions.
            ({"JT_MPI": "1"}, True),
            ({"JT_MPI": "0", "OMPI_COMM_WORLD_SIZE": "2"}, False),
        ]
        saved = {k: os.environ.get(k) for k in _LAUNCHER_ENV_KEYS}
        try:
            for overrides, want in cases:
                with self.subTest(env=overrides):
                    for k in _LAUNCHER_ENV_KEYS:
                        os.environ.pop(k, None)
                    os.environ.update(overrides)
                    self.assertEqual(compile_extern.inside_mpi(), want)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_cxx_agrees_with_python_on_a_non_openmpi_launcher(self):
        """PMI_SIZE (MPICH-style) must be seen by the C++ side too.

        Before the fix the C++ side looked only at OMPI_COMM_WORLD_SIZE, so it
        stayed disabled while Python said in_mpi -- caught now by the
        consistency check at the end of compile_extern.
        """
        if not compile_extern.has_mpi:
            self.skipTest("no mpi found")
        result = _run_child(
            {"PMI_SIZE": "2", "PMI_RANK": "0", "nvcc_path": ""},
            "import jittor as jt\n"
            "from jittor import compile_extern as ce\n"
            "print('PY_IN_MPI', ce.in_mpi)\n"
            "print('CXX_ENABLED', bool(ce.mpi.get_state()))\n")
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertIn("PY_IN_MPI True", result.stdout)
        self.assertIn("CXX_ENABLED True", result.stdout)


class TestSingleSourceOfRank(unittest.TestCase):

    def test_jt_rank_reads_through_to_compile_extern(self):
        saved = (compile_extern.rank, compile_extern.world_size,
                 compile_extern.in_mpi)
        try:
            compile_extern.rank = 7
            compile_extern.world_size = 9
            compile_extern.in_mpi = True
            self.assertEqual(jt.rank, 7)
            self.assertEqual(jt.world_size, 9)
            self.assertIs(jt.in_mpi, True)
        finally:
            (compile_extern.rank, compile_extern.world_size,
             compile_extern.in_mpi) = saved

    def test_jittor_module_holds_no_copy(self):
        # A plain module attribute would shadow __getattr__ and go stale again.
        for name in ("rank", "world_size", "in_mpi"):
            self.assertNotIn(name, vars(jt), "jittor." + name + " is a copy again")

    def test_mpi_param_broadcast_follows_the_owner(self):
        """core_api used to read its own `from jittor import *` snapshot.

        With the snapshot, turning distributed on after import (which is what
        the torch NCCL installer does) left mpi_param_broadcast() silently
        doing nothing -- every rank keeps its own random init and the models
        never match.
        """
        code = (
            "import jittor as jt\n"
            "from jittor import compile_extern as ce\n"
            "m = jt.nn.Linear(2, 2)\n"
            "ce.in_mpi = True\n"
            "try:\n"
            "    m.mpi_param_broadcast(0)\n"
            "except AttributeError:\n"
            "    print('ATTEMPTED')\n"   # took the distributed path (no mpi ops here)
            "else:\n"
            "    print('SKIPPED')\n"     # read a stale snapshot and returned
        )
        # use_mpi=0 so Var has no mpi_broadcast: taking the distributed path is
        # then observable as an AttributeError.
        result = _run_child({"use_mpi": "0", "nvcc_path": "",
                             "CUDA_VISIBLE_DEVICES": ""}, code)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertIn("ATTEMPTED", result.stdout)
        self.assertNotIn("SKIPPED", result.stdout)


if __name__ == "__main__":
    unittest.main()
