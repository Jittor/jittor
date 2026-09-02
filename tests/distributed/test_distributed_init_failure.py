# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Distributed init must fail loudly once distributed has been requested (6.B04).

The failure this guards against is the quiet one: a launcher starts N ranks,
the collective backend fails to come up, and every rank carries on as an
independent single-card job. Losses look sensible, nothing errors, and nothing
is ever exchanged between ranks -- an N-card job silently became N one-card
jobs training N different models.

Each case runs in a subprocess because the condition is decided during
``import jittor``.
"""
from pathlib import Path
import unittest

from _helpers.child_process import run_python_child

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_import(env_overrides, code="import jittor"):
    return run_python_child(["-c", code], env=env_overrides, cwd=_REPO_ROOT,
                            merge_stderr=True, timeout=1800)


# No CUDA and no nvcc, so setup_nccl() cannot bring NCCL up. Setting
# JT_NCCL_WORLD_SIZE also forces use_mpi=0, so there is no MPI fallback either:
# the process is told it is rank 1 of 2 and has nothing to communicate through.
_NO_BACKEND = {
    "CUDA_VISIBLE_DEVICES": "",
    "nvcc_path": "",
    "JT_NCCL_WORLD_SIZE": "2",
    "JT_NCCL_RANK": "1",
    "JT_NCCL_LOCAL_RANK": "0",
}


class TestDistributedInitFailure(unittest.TestCase):

    def test_requested_but_no_backend_is_a_hard_error(self):
        result = _run_import(_NO_BACKEND)
        self.assertNotEqual(result.returncode, 0, "import succeeded:\n" + result.stdout[-3000:])
        self.assertIn("no collective backend", result.stdout)

    def test_world_size_one_is_not_a_request(self):
        # A single-rank run must stay silent and importable -- "distributed was
        # not requested" and "distributed failed" are different situations.
        env = dict(_NO_BACKEND)
        env["JT_NCCL_WORLD_SIZE"] = "1"
        env["JT_NCCL_RANK"] = "0"
        result = _run_import(env)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])

    def test_plain_import_is_unaffected(self):
        result = _run_import({"CUDA_VISIBLE_DEVICES": "", "nvcc_path": ""})
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])

    def test_fsdp2_world_size_does_not_swallow(self):
        # _world_size()/_rank() used to return 1/0 on any exception, which is
        # the same silent single-card degradation one layer up.
        code = (
            "import jittor as jt\n"
            "from jittor.compat.fsdp2 import common\n"
            "jt.world_size = 'not-a-number'\n"
            "try:\n"
            "    common._world_size()\n"
            "except Exception as e:\n"
            "    print('RAISED', type(e).__name__)\n"
            "else:\n"
            "    print('SWALLOWED')\n"
        )
        result = _run_import({"CUDA_VISIBLE_DEVICES": "", "nvcc_path": ""}, code)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertIn("RAISED", result.stdout)
        self.assertNotIn("SWALLOWED", result.stdout)


if __name__ == "__main__":
    unittest.main()
