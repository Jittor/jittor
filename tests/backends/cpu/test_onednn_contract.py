# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""What the oneDNN backend declares, and whether the declaration is true.

The audit's entry for MKL matmul reads: "only supports fp32 --
``mkl_matmul_op.cc:28`` ``ASSERT(a->dtype().dsize()==4)``; CPU fp64/fp16/bf16
matmul falls back to the generic meta-operator while CUDA supports all of them,
**and this kind of per-backend capability difference has nowhere to be
declared**". Two things were missing rather than one:

* a place to declare it -- now ``OpCapabilityRegistration::dtypes``, queryable
  from Python as ``jt.core.backend_capability_dtypes(backend, capability)``;
* any runtime evidence that the claim was even accurate. There was none,
  because every one of Jittor's 13 MKL test cases was dead: 5 failing with
  ``AttributeError`` on a ``None`` and 8 skipping, both because
  ``jt.compile_extern.mkl_ops`` is a *query* and not an accessor. See
  ``tests/_helpers/onednn.py``.

So this file asserts the declaration and then checks it against what actually
runs, per dtype. A declaration nobody compares against reality is a comment
with a type; the comparison is what makes it a contract.
"""

import unittest

import numpy as np

import jittor as jt

from _helpers.onednn import requires_onednn


#: The capabilities oneDNN publishes for the CPU backend. Every one of them
#: must carry a dtype declaration: this is the ratchet, so that dropping a
#: declaration -- or adding a capability without one -- fails here rather than
#: quietly restoring the state the audit recorded.
CPU_CAPABILITIES = ("matmul", "conv2d", "conv2d_backward_input",
                    "conv2d_backward_weight")

#: The dtype axis. Only the first is expected to reach oneDNN today; the rest
#: are the widths the audit says CUDA has and CPU does not.
DTYPES = ("float32", "float64", "float16", "bfloat16")


def _matmul_implementation(dtype):
    """``(executed oneDNN implementation or None, result, reference)``.

    The two-dimensional broadcast/multiply/reduce spelling is the only form
    ``MatmulTuner`` recognises, so this is the shape in which a CPU matmul can
    reach oneDNN at all. ``auto_convert_64_to_32=0`` matters: with Jittor's
    default on, ``jt.array`` of a float64 array yields a **float32** Var, so a
    "float64" case would silently measure float32 and agree with any
    declaration at all.
    """
    random = np.random.RandomState(11)
    left = random.randn(16, 24)
    right = random.randn(24, 12)
    with jt.flag_scope(use_cuda=0, enable_tuner=1, auto_convert_64_to_32=0,
                       compile_options={"onednn_dtype_probe_" + dtype: 1}):
        a = jt.array(left).cast(dtype)
        b = jt.array(right).cast(dtype)
        jt.sync([a, b])
        with jt.profile_scope() as report:
            product = (a.broadcast([16, 24, 12], [2]) *
                       b.broadcast([16, 24, 12], [0])).sum(1)
            result = product.numpy()
    executed = [row[0] for row in report[1:] if "mkl_matmul" in row[0]]
    return (executed[0] if executed else None), result, left @ right


class TestOnednnCapabilityDeclaration(unittest.TestCase):
    """oneDNN is the CPU backend, so every case here pins ``use_cuda`` off."""

    def setUp(self):
        requires_onednn()

    def test_every_cpu_capability_declares_its_dtypes(self):
        published = set(jt.core.backend_supported_capabilities("cpu"))
        assert set(CPU_CAPABILITIES) <= published, sorted(published)
        undeclared = [name for name in CPU_CAPABILITIES
                      if not jt.core.backend_capability_dtypes("cpu", name)]
        assert not undeclared, (
            "these CPU capabilities publish no dtype declaration, which is the "
            "state the audit recorded -- the difference from the accelerator "
            "is real and has to be sayable: %s" % undeclared)

    def test_cpu_matmul_declares_float32_only(self):
        assert jt.core.backend_capability_dtypes("cpu", "matmul") == ["float32"]
        # The convolutions are the same story: `dnnl_sgemm` and the f32-only
        # `supports_conv_layout` predicate decide it, not oneDNN's own reach.
        for name in CPU_CAPABILITIES:
            assert jt.core.backend_capability_dtypes("cpu", name) == ["float32"], name

    def test_the_declaration_matches_which_implementation_actually_runs(self):
        """The anti-drift check, and the audit entry's missing evidence.

        For each dtype: oneDNN's matmul runs if and only if the capability
        declared that dtype. A declaration that grew a dtype the kernel cannot
        do would fail here, and so would a kernel that quietly started taking
        one it never declared -- which is the more dangerous direction, since
        ``mkl_matmul_op.cc``'s float-only ``dnnl_sgemm`` would be handed
        non-float storage.
        """
        declared = set(jt.core.backend_capability_dtypes("cpu", "matmul"))
        observed = {}
        for dtype in DTYPES:
            implementation, result, reference = _matmul_implementation(dtype)
            observed[dtype] = implementation
            # Whichever path ran, the numbers have to be right. Tolerance is
            # per dtype because the generic kernel accumulates in the input
            # width, and bf16 has 8 mantissa bits.
            tolerance = {"float32": 1e-4, "float64": 1e-4,
                         "float16": 2e-2, "bfloat16": 3e-1}[dtype]
            error = float(np.abs(np.asarray(result, dtype="float64") - reference).max())
            assert error < tolerance, (dtype, error, tolerance)
        reached = {dtype for dtype, implementation in observed.items()
                   if implementation is not None}
        assert reached == declared, (
            "the capability declares %s but oneDNN's matmul actually ran for "
            "%s (per dtype: %s)" % (sorted(declared), sorted(reached), observed))

    def test_a_capability_with_no_declaration_reads_as_empty(self):
        """Empty means "undeclared", and callers must not read it as "none".

        Stated as a test because the two are easy to confuse and the
        difference decides whether a caller may skip a backend.
        """
        assert jt.core.backend_capability_dtypes("cpu", "random") == []
        assert "random" not in jt.core.backend_supported_capabilities("cpu")


class TestOnednnLibraryLayout(unittest.TestCase):
    """Which shared library name the build accepts.

    The version pin (``manifest.MKL``, oneDNN 2.2.0 from 2021) was not the only
    thing holding the library version down. The 2.2 archive ships
    ``libdnnl.so`` *and* a ``libmkldnn.so`` compatibility alias; oneDNN v3
    dropped the alias. Every place that decides "is oneDNN installed" and
    "what do I link against" named the alias, so a correct v3 tree would have
    been reported as "downloaded but not installed" and linked with a
    ``-lmkldnn`` that does not resolve.

    These cases need no oneDNN at all -- they drive the detection over
    synthetic directory layouts, including the v3 one this machine does not
    have. That is deliberate: oneDNN publishes no prebuilt binary for v3 (the
    releases after v2.x carry no assets), so the v3 layout cannot be checked
    by installing it here, and leaving it unchecked is how the pin stayed.
    """

    def _layout(self, tmp_path, subdirectory, filename):
        directory = tmp_path / subdirectory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_bytes(b"")
        return jt.compile_extern.mkl_library_layout(str(tmp_path))

    def test_a_v3_tree_with_only_libdnnl_is_accepted(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as root:
            layout = self._layout(Path(root), "lib", "libdnnl.so")
            assert layout is not None, "a oneDNN v3 tree was rejected"
            path, linker_name = layout
            assert path.endswith("lib/libdnnl.so")
            assert linker_name == "dnnl", (
                "linking a v3 tree with -l%s does not resolve" % linker_name)

    def test_the_pinned_v2_alias_is_still_accepted(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as root:
            layout = self._layout(Path(root), "lib", "libmkldnn.so")
            assert layout is not None
            assert layout[1] == "mkldnn"

    def test_a_tree_with_no_library_is_not_mistaken_for_an_install(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "lib").mkdir()
            assert jt.compile_extern.mkl_library_layout(root) is None

    def test_the_layout_the_build_is_using_here_resolves(self):
        """Whatever this machine actually has must be one of the accepted ones.

        Otherwise the cases above are checking a detection function that the
        build does not use.
        """
        requires_onednn()
        assert jt.compile_extern.mkl_ops is not None


if __name__ == "__main__":
    unittest.main()
