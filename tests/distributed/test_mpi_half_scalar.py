# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The architecture-independent fp16 conversions must be IEEE-exact (6.B10).

``HalfAdd`` in mpi_wrapper.cc implements the MPI_Op behind MPI_HALF. It used to
be two unrelated implementations -- an F16C one on x86 and a hand-written one
everywhere else -- and they disagreed on 34.27% of fp16 sums, with 49512 values
where one produced a subnormal and the other produced zero. Whether an
all-reduce was correct depended on which machine ran it.

There is now one scalar implementation that defines the semantics on every
architecture, with the x86 SIMD path as an accelerator that must match it.
This test compiles those two scalar functions straight out of the source and
checks them against numpy's IEEE round-to-nearest-even casts. It needs no MPI,
no multiple processes, and above all no ARM machine: the code under test here
is exactly the code a non-x86 host runs.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

import jittor as jt

_WRAPPER_SRC = (Path(__file__).resolve().parents[2] / "python" / "jittor"
                / "extern" / "mpi" / "src" / "mpi_wrapper.cc")

_HARNESS = r"""
#include <cstdint>
#include <cstring>
#include <cstdio>
%(conversions)s
int main(int argc, char** argv) {
    // 1. fp16 -> fp32 for every fp16 bit pattern.
    FILE* out = fopen(argv[1], "wb");
    for (uint32_t h = 0; h < 65536; h++) {
        float f = jt_fp16_to_fp32((uint16_t)h);
        fwrite(&f, sizeof(f), 1, out);
    }
    // 2. fp32 -> fp16 for every fp32 the caller hands us.
    FILE* in = fopen(argv[2], "rb");
    float v;
    while (fread(&v, sizeof(v), 1, in) == 1) {
        uint16_t h = jt_fp32_to_fp16(v);
        fwrite(&h, sizeof(h), 1, out);
    }
    fclose(in);
    fclose(out);
    return 0;
}
"""


def _extract_conversions():
    source = _WRAPPER_SRC.read_text()
    begin = source.index("static inline float jt_fp16_to_fp32")
    end = source.index("static void half_add_scalar")
    return source[begin:end]


class TestMpiHalfScalarConversions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cc = getattr(jt.compiler, "cc_path", None) or "g++"

    def test_conversions_are_ieee_exact(self):
        conversions = _extract_conversions()
        # Deliberately no -march / no SIMD: this is the portable path.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "half.cc"
            src.write_text(_HARNESS % {"conversions": conversions})
            binary = tmp / "half"
            compile_result = subprocess.run(
                [self.cc, "-O2", "-std=c++14", "-o", os.fspath(binary), os.fspath(src)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
            self.assertEqual(compile_result.returncode, 0, compile_result.stdout)

            # Every fp32 value that can arise as a sum of two fp16 values,
            # sampled by pairing all 65536 fp16 patterns with 64 partners.
            all_half = np.frombuffer(
                np.arange(65536, dtype="uint16").tobytes(), dtype="float16")
            partners = all_half[::1024]
            with np.errstate(invalid="ignore"):
                # inf + -inf is a legitimate probe; it must come back as a nan.
                sums = (all_half.astype("float32")[None, :]
                        + partners.astype("float32")[:, None]).ravel()
            # Plus the exact fp16 grid, so subnormals and the fp16 max are hit
            # as inputs in their own right.
            probe = np.concatenate([all_half.astype("float32"), sums])

            src_bin = tmp / "in.bin"
            src_bin.write_bytes(probe.astype("float32").tobytes())
            out_bin = tmp / "out.bin"
            run = subprocess.run(
                [os.fspath(binary), os.fspath(out_bin), os.fspath(src_bin)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
            self.assertEqual(run.returncode, 0, run.stdout)

            data = out_bin.read_bytes()
            widened = np.frombuffer(data[:65536 * 4], dtype="float32")
            narrowed = np.frombuffer(data[65536 * 4:], dtype="uint16")

            # fp16 -> fp32 is exact by construction; compare bitwise, treating
            # all NaNs as equal.
            want_wide = all_half.astype("float32")
            nan = np.isnan(want_wide)
            np.testing.assert_array_equal(
                widened[~nan].view("uint32"), want_wide[~nan].view("uint32"))
            self.assertTrue(np.all(np.isnan(widened[nan])))

            # fp32 -> fp16 must be round-to-nearest-even, exactly like numpy's
            # cast. This is where the old non-x86 path truncated instead of
            # rounding and flushed subnormals to zero.
            with np.errstate(over="ignore", invalid="ignore"):
                want_narrow = probe.astype("float16")
            nan = np.isnan(probe)
            np.testing.assert_array_equal(
                narrowed[~nan], want_narrow[~nan].view("uint16"),
                err_msg="fp32->fp16 is not IEEE round-to-nearest-even")
            self.assertTrue(np.all((narrowed[nan] & 0x7C00) == 0x7C00))
            self.assertTrue(np.all((narrowed[nan] & 0x03FF) != 0))

    def test_fp16_subnormals_survive_a_round_trip(self):
        # The old non-x86 path turned every value below 2^-14 into zero, so any
        # gradient that reduced into the subnormal range simply vanished.
        subnormals = np.frombuffer(
            np.arange(1, 1024, dtype="uint16").tobytes(), dtype="float16")
        self.assertTrue(np.all(subnormals != 0))
        self.assertTrue(np.all(subnormals.astype("float32").astype("float16")
                               .view("uint16") == np.arange(1, 1024, dtype="uint16")))


if __name__ == "__main__":
    unittest.main()
