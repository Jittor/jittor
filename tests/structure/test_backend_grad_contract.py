"""Every backend gradient implementation has a named reference route.

A backend ``grad`` that no test ever compares against a reference is the
quietest kind of defect: it produces numbers, training converges to something,
and nothing reports. This module is the inventory that makes such an
implementation impossible to add unnoticed -- the manifest below must equal,
exactly, the set of gradients in the tree.

Two things this file used to get wrong, both of which made it pass while
covering less than it claimed:

* It scanned ``backends/cuda/kernels`` and ``python/jittor/extern`` only. The
  two ROCm gradients live in ``backends/rocm/libraries``, so they were never
  in the inventory, and the total (26) looked healthy the whole time. Scan
  roots are now enumerated per backend and each one that should hold a
  gradient is asserted non-empty, so a path that stops matching fails loudly
  instead of shrinking the set it compares against.
* It looked for ``::grad(`` in ``.cc`` files only. ACL implements 23 of its
  gradients as ``jt.Function`` subclasses in ``backends/acl/kernels/ops``, and
  the hand-written CUDA kernels add 9 more in ``backends/cuda/kernels/nn``;
  none were counted.

Device kernels and communication gradients have distinct owners under
``backends/``; each owner is scanned independently.

Hardware honesty: this machine has CUDA and nothing else. Entries whose kind
ends in ``_hardware`` cannot be executed here, and they are not pretended to
pass; ``agent/manuals/deferred-hardware.md`` carries the prerequisite, the
command and the pass criterion for each, and a test below checks that every
such kind is actually described there.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Roots that hold at least one backend gradient today. Each is asserted
#: non-empty, individually: a total that still looks plausible is exactly how
#: the missing ROCm pair stayed invisible.
SCAN_ROOTS = (
    "backends/acl",
    # 4.15 moved the oneDNN operator sources here from
    # python/jittor/extern/mkl/ops; without this root the manifest entry for
    # MklBatchedMatmulOp reads as "no longer exists in the tree".
    "backends/cpu",
    "backends/cuda",
    "backends/rocm",
    "backends/comm/mpi",
    "backends/comm/nccl",
    "backends/comm/hccl",
)

#: Scanned too, and expected to yield nothing. Corex is a runtime and a header
#: with no operators of its own. Asserting the directory is real and has
#: sources keeps this from silently becoming a mistyped path that "passes".
EMPTY_SCAN_ROOTS = ("backends/corex",)

#: The core ops. Not backend gradients, but scanned so that a backend moving
#: in here cannot escape the inventory by changing address.
CORE_ROOT = "src"

_CPP_GRAD = re.compile(r"^VarPtr\s+(\w+)::grad\(", re.M)


# (source, symbol, reference nodeid, kind)
BACKEND_GRAD_COVERAGE = (
    # ---- CUDA, C++ ops. Real hardware here; these all run. ----------------
    ("backends/cuda/kernels/cub/cub_arg_reduce_op.cc", "CubArgReduceOp",
     "tests/ops/test_arg_reduce_op.py::TestArgReduceOp::test_backward_cuda",
     "cuda_cpu_formula"),
    ("backends/cuda/kernels/cub/cub_argsort_op.cc", "CubArgsortOp",
     "tests/ops/test_argsort_op.py::TestArgsortOp::test_cub_backward",
     "cuda_cpu_formula"),
    ("backends/cuda/kernels/cub/cub_cumsum_op.cc", "CubCumsumOp",
     "tests/backends/cuda/test_cub_cumsum.py::TestCubCumsumOp::test_1d_backward",
     "cuda_cpu_jittor"),
    ("backends/cuda/kernels/cublas/cublas_matmul_op.cc", "CublasMatmulOp",
     "tests/backends/cuda/test_cublas_matmul_grad.py::TestCublasMatmulGrad::test_all_transpose_combinations",
     "cuda_numpy"),
    ("backends/cuda/kernels/cublas/cublas_batched_matmul_op.cc",
     "CublasBatchedMatmulOp",
     "tests/backends/cuda/test_cublas_matmul_grad.py::TestCublasMatmulGrad::test_linear_3d_random_projection_grad",
     "cuda_numpy"),
    ("backends/cuda/kernels/cudnn/cudnn_conv_op.cc", "CudnnConvOp",
     "tests/backends/cuda/test_cudnn_conv_plan.py::TestCudnnConvPlan::test_plain_fp32",
     "cuda_cpu_jittor"),
    ("backends/cuda/kernels/cudnn/cudnn_conv_backward_x_op.cc",
     "CudnnConvBackwardXOp",
     "tests/backends/cuda/test_cudnn_conv_plan.py::TestCudnnConvPlan::test_plain_fp32",
     "cuda_cpu_jittor"),
    ("backends/cuda/kernels/cudnn/cudnn_conv_backward_w_op.cc",
     "CudnnConvBackwardWOp",
     "tests/backends/cuda/test_cudnn_conv_plan.py::TestCudnnConvPlan::test_plain_fp32",
     "cuda_cpu_jittor"),
    ("backends/cuda/kernels/cudnn/cudnn_conv3d_op.cc", "CudnnConv3dOp",
     "tests/backends/cuda/test_cudnn_conv3d_algo_cache.py::TestCudnnConv3dAlgoCache::test_forward_and_gradients_match_cpu_reference",
     "cuda_cpu_jittor"),
    ("backends/cuda/kernels/cudnn/cudnn_conv3d_backward_x_op.cc",
     "CudnnConv3dBackwardXOp",
     "tests/backends/cuda/test_cudnn_conv3d_algo_cache.py::TestCudnnConv3dAlgoCache::test_forward_and_gradients_match_cpu_reference",
     "cuda_cpu_jittor"),
    ("backends/cuda/kernels/cudnn/cudnn_conv3d_backward_w_op.cc",
     "CudnnConv3dBackwardWOp",
     "tests/backends/cuda/test_cudnn_conv3d_algo_cache.py::TestCudnnConv3dAlgoCache::test_forward_and_gradients_match_cpu_reference",
     "cuda_cpu_jittor"),
    ("backends/cuda/kernels/cufft/cufft_fft_op.cc", "CufftFftOp",
     "tests/ops/test_fft_op.py::TestFFTOp::test_fft_backward",
     "cuda_numpy"),
    ("backends/cuda/kernels/cutt/cutt_transpose_op.cc", "CuttTransposeOp",
     "tests/backends/cuda/test_cutt_transpose_op.py::TestCuttTransposeOp::test_grad",
     "cuda_numpy"),

    # ---- CUDA, hand-written kernels as jt.Function. Real hardware here. ---
    ("backends/cuda/kernels/nn/batch_norm_training_cuda.py", "BatchNormCUDA",
     "tests/nn/test_norm.py::TestBatchNorm::test_cuda_fast_path_forward_and_all_gradients",
     "cuda_cpu_jittor"),
    ("backends/cuda/kernels/nn/batch_norm_training_cuda.py", "BatchNormEvalCUDA",
     "tests/nn/test_norm.py::TestBatchNorm::test_cuda_eval_fast_path_forward_and_all_gradients",
     "cuda_cpu_jittor"),
    ("backends/cuda/kernels/nn/channel_bias_cuda.py", "ChannelBiasCUDA",
     "tests/nn/test_norm.py::TestChannelBias::test_cuda_forward_and_bias_gradient",
     "cuda_cpu_jittor"),
    ("backends/cuda/kernels/nn/full_reduce_cuda.py", "FullSumCUDA",
     "tests/backends/cuda/test_full_reduce.py::TestFullReduce::test_gradients",
     "cuda_numpy"),
    ("backends/cuda/kernels/nn/group_norm_cuda.py", "GroupNormCUDA",
     "tests/nn/test_norm.py::TestGroupNorm::test_cuda_fast_path_forward_and_all_gradients",
     "cuda_cpu_jittor"),
    ("backends/cuda/kernels/nn/layer_norm_training_cuda.py", "LayerNormCUDA",
     "tests/nn/test_norm.py::TestLayerNorm::test_cuda_fast_path_forward_and_all_gradients",
     "cuda_cpu_jittor"),
    ("backends/cuda/kernels/nn/rms_norm_training_cuda.py", "RMSNormTrainingCUDA",
     "tests/nn/test_norm.py::TestRMSNorm::test_cuda_training_forward_and_all_gradients",
     "cuda_cpu_jittor"),
    ("backends/cuda/kernels/nn/softmax_cuda.py", "CodeSoftmax",
     "tests/backends/cuda/test_softmax_cuda_grad.py::TestSoftmaxCudaGrad::test_every_schedule_matches_an_exact_float64_reference",
     "cuda_exact_and_cpu"),
    ("backends/cuda/kernels/nn/softmax_cuda.py", "CodeSoftmaxStreaming",
     "tests/backends/cuda/test_softmax_cuda_grad.py::TestSoftmaxCudaGrad::test_every_schedule_matches_an_exact_float64_reference",
     "cuda_exact_and_cpu"),

    # ---- CPU oneDNN. Runs everywhere. -------------------------------------
    ("backends/cpu/libraries/mkl/mkl_batched_matmul_op.cc",
     "MklBatchedMatmulOp",
     "tests/ops/test_mkl_batched_matmul.py::TestMklBatchedMatmul::test_three_dimensional_batch",
     "cpu_jittor"),

    # ---- ROCm. No AMD device on this machine. -----------------------------
    ("backends/rocm/libraries/hipblas/hipblas_matmul_op.cc", "HipblasMatmulOp",
     "tests/backends/rocm/test_rocm.py::TestBMM::test_bmm_rocm",
     "rocm_hardware"),
    ("backends/rocm/libraries/rocprim/rocprim_cumsum_op.cc", "RocprimCumsumOp",
     "tests/backends/rocm/test_rocm.py::TestROCm::test_rocm_fused_op",
     "rocm_hardware_no_grad_test"),

    # ---- ACL / Ascend. No NPU on this machine. ----------------------------
    ("backends/acl/kernels/ops/arg_reduce_op.py", "ArgReduceACL",
     "tests/backends/npu/test_acl.py::TestACL::test_float_arg_reduce_backward_runs_on_acl",
     "npu_hardware"),
    ("backends/acl/kernels/ops/bmm_op.py", "BmmACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_bmm_grad_a",
     "npu_hardware"),
    ("backends/acl/kernels/ops/clamp_op.py", "ClampACL",
     "tests/backends/npu/test_acl.py::TestACL::test_clamp_scalar_forward_backward_uses_cann",
     "npu_hardware"),
    ("backends/acl/kernels/ops/cumsum_op.py", "CumsumACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_cumsum_grad",
     "npu_hardware"),
    ("backends/acl/kernels/ops/dropout_op.py", "DropoutACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_dropout_grad",
     "npu_hardware"),
    ("backends/acl/kernels/ops/embedding_op.py", "EmbeddingACL",
     "tests/backends/npu/test_acl.py::TestACL::test_training_embedding",
     "npu_hardware"),
    ("backends/acl/kernels/ops/flip_op.py", "FlipACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_flip_grad",
     "npu_hardware"),
    ("backends/acl/kernels/ops/floor_op.py", "FloorIntACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_floor_int",
     "npu_hardware_no_grad_test"),
    ("backends/acl/kernels/ops/gather_scatter_op.py", "GatherACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_gather_grad",
     "npu_hardware"),
    ("backends/acl/kernels/ops/gather_scatter_op.py", "ScatterACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_scatter_grad",
     "npu_hardware"),
    ("backends/acl/kernels/ops/getitem_op.py", "GetItemACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_getitem_grad",
     "npu_hardware"),
    ("backends/acl/kernels/ops/index_op.py", "IndexACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_index",
     "npu_hardware_no_grad_test"),
    ("backends/acl/kernels/ops/norms_op.py", "RmsNormACL",
     "tests/backends/npu/test_acl.py::TestACL::test_training_rms_norm",
     "npu_hardware"),
    ("backends/acl/kernels/ops/pool_op.py", "PoolACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_maxpool_grad",
     "npu_hardware"),
    ("backends/acl/kernels/ops/roll_op.py", "RollACL",
     "tests/backends/npu/test_acl_torch_compat.py::TestACLTorchCompat::test_roll_bfloat16_forward_backward_stays_on_acl",
     "npu_hardware"),
    ("backends/acl/kernels/ops/rope_op.py", "RotaryPositionEmbeddingACL",
     "tests/backends/npu/test_acl.py::TestACL::test_rotary_embedding_composite_gradient",
     "npu_hardware"),
    ("backends/acl/kernels/ops/setitem_op.py", "SetItemACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_setitem_grad",
     "npu_hardware"),
    ("backends/acl/kernels/ops/sigmoid_op.py", "SigmoidACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_sigmoid_grad",
     "npu_hardware"),
    ("backends/acl/kernels/ops/softmax_op.py", "SoftmaxACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_softmax_grad",
     "npu_hardware"),
    ("backends/acl/kernels/ops/stack_op.py", "StackACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_stack",
     "npu_hardware_no_grad_test"),
    ("backends/acl/kernels/ops/triu_op.py", "TriuACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_triu",
     "npu_hardware_no_grad_test"),
    ("backends/acl/kernels/ops/where_op.py", "NonzeroACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_nonzero_1",
     "npu_hardware_no_grad_test"),
    ("backends/acl/kernels/ops/where_op.py", "WhereACL",
     "tests/backends/npu/test_aclop.py::TestACL::test_where_grad",
     "npu_hardware"),

    # ---- ACL / HCCL collectives. Needs >=2 Ascend cards. ------------------
    ("backends/comm/hccl/ops/hccl_all_gather_op.cc",
     "HcclAllGatherOp",
     "tests/distributed/test_hccl_check_macros.py::TestHcclCheckMacros::test_finalizer_does_not_use_the_throwing_macro",
     "unsupported_hardware"),
    ("backends/comm/hccl/ops/hccl_all_reduce_op.cc",
     "HcclAllReduceOp",
     "tests/backends/npu/test_acl.py::TestACL::test_product_reduction_forward_backward",
     "npu_hardware"),
    ("backends/comm/hccl/ops/hccl_broadcast_op.cc",
     "HcclBroadcastOp",
     "tests/backends/npu/test_acl.py::TestACL::test_broadcast",
     "npu_hardware"),
    ("backends/comm/hccl/ops/hccl_reduce_op.cc", "HcclReduceOp",
     "tests/backends/npu/test_acl.py::TestACL::test_all_reduction",
     "npu_hardware"),

    # ---- CUDA / NCCL collectives. Needs a multi-GPU launcher. -------------
    ("backends/comm/nccl/ops/nccl_all_gather_op.cc",
     "NcclAllGatherOp",
     "tests/distributed/test_fsdp2_nccl.py::TestFSDP2Nccl::test_nccl_all_gather_autograd",
     "nccl_hardware"),
    ("backends/comm/nccl/ops/nccl_all_reduce_op.cc",
     "NcclAllReduceOp",
     "tests/distributed/test_nccl_ops.py::TestNcclOps::test_all_reduce",
     "nccl_hardware"),
    ("backends/comm/nccl/ops/nccl_broadcast_op.cc",
     "NcclBroadcastOp",
     "tests/distributed/test_nccl_ops.py::TestNcclOps::test_broadcast",
     "nccl_hardware"),
    ("backends/comm/nccl/ops/nccl_reduce_op.cc", "NcclReduceOp",
     "tests/distributed/test_nccl_ops.py::TestNcclOps::test_reduce",
     "nccl_hardware"),
    ("backends/comm/nccl/ops/nccl_reduce_scatter_op.cc",
     "NcclReduceScatterOp",
     "tests/distributed/test_fsdp2_nccl.py::TestFSDP2Nccl::test_nccl_all_gather_autograd",
     "nccl_hardware"),

    # ---- MPI collectives. Needs an mpirun launcher. -----------------------
    ("backends/comm/mpi/ops/mpi_all_reduce_op.cc", "MpiAllReduceOp",
     "tests/distributed/test_mpi_op.py::TestMpiOps::test_all_reduce",
     "mpi_hardware"),
    ("backends/comm/mpi/ops/mpi_broadcast_op.cc", "MpiBroadcastOp",
     "tests/distributed/test_mpi_op.py::TestMpiOps::test_broadcast",
     "mpi_hardware"),
    ("backends/comm/mpi/ops/mpi_reduce_op.cc", "MpiReduceOp",
     "tests/distributed/test_mpi_op.py::TestMpiOps::test_reduce",
     "mpi_hardware"),
)

#: Kinds that name a reference which actually executes somewhere in CI.
EXECUTABLE_KINDS = {
    "cpu_jittor",          # CPU backend against jittor's generic ops
    "cuda_cpu_jittor",     # CUDA against the same graph run on CPU
    "cuda_cpu_formula",    # CUDA against the closed-form CPU gradient
    "cuda_numpy",          # CUDA against an independent NumPy reference
    "cuda_exact_and_cpu",  # both: float64 NumPy for tightness, CPU for independence
}

#: Kinds that cannot run here. Each must be described in the deferred manual.
DEFERRED_KINDS = {
    "mpi_hardware",
    "nccl_hardware",
    "npu_hardware",
    "npu_hardware_no_grad_test",
    "rocm_hardware",
    "rocm_hardware_no_grad_test",
    "unsupported_hardware",
}

#: The kinds whose *gradient* has no test even on the right hardware. These are
#: the open gaps; the manual lists them one by one.
NO_GRAD_TEST_KINDS = {"npu_hardware_no_grad_test", "rocm_hardware_no_grad_test",
                      "unsupported_hardware"}

REFERENCE_KINDS = EXECUTABLE_KINDS | DEFERRED_KINDS

DEFERRED_MANUAL = ROOT / "agent/manuals/deferred-hardware.md"


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------

def _grads_in_file(path):
    """(symbol, ...) for every gradient defined in one source file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix in (".cc", ".cu"):
        return [match.group(1) for match in _CPP_GRAD.finditer(text)]
    found = []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
               and item.name == "grad" for item in node.body):
            found.append(node.name)
    return found


def scan(root):
    """{(relative path, symbol)} for every gradient under ``root``."""
    directory = ROOT / root
    found = set()
    for path in sorted(directory.rglob("*")):
        if path.suffix not in (".cc", ".cu", ".py") or not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        for symbol in _grads_in_file(path):
            found.add((path.relative_to(ROOT).as_posix(), symbol))
    return found


def _scan_all(roots):
    found = set()
    for root in roots:
        found |= scan(root)
    return found


def _source_files(root):
    return [path for path in (ROOT / root).rglob("*")
            if path.suffix in (".cc", ".cu", ".py") and path.is_file()
            and "__pycache__" not in path.parts]


def compare(manifest, scanned):
    """(in the tree but unlisted, listed but not in the tree)."""
    listed = {(source, symbol) for source, symbol, _, _ in manifest}
    return sorted(scanned - listed), sorted(listed - scanned)


def _nodeid_exists(nodeid):
    parts = nodeid.split("::")
    assert len(parts) in (2, 3), "unsupported nodeid: %s" % nodeid
    path = ROOT / parts[0]
    if not path.is_file():
        return False
    body = ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body
    for index, name in enumerate(parts[1:]):
        kinds = ((ast.ClassDef,) if index == 0 and len(parts) == 3
                 else (ast.FunctionDef, ast.AsyncFunctionDef))
        node = next((item for item in body
                     if isinstance(item, kinds) and item.name == name), None)
        if node is None:
            return False
        body = node.body
    return True


# --------------------------------------------------------------------------
# The inventory
# --------------------------------------------------------------------------

def test_each_scan_root_holds_at_least_one_gradient():
    """Per root, not in total.

    The predecessor asserted a total of 26 and scanned two directories. When
    the backends moved, ROCm's two gradients left the scanned set and the
    total was still 26, because the manifest had been written from the same
    two directories. A per-root floor is what that missed.
    """
    empty = [root for root in SCAN_ROOTS if not scan(root)]
    assert empty == [], (
        "these roots are expected to hold backend gradients and matched "
        "nothing -- the path is wrong, or the sources moved again: %s" % empty)


def test_roots_without_gradients_are_still_really_scanned():
    for root in EMPTY_SCAN_ROOTS:
        assert (ROOT / root).is_dir(), "%s is not a directory" % root
        assert _source_files(root), (
            "%s matched no sources at all; the root is mistyped and its "
            "emptiness below proves nothing" % root)
        assert scan(root) == set(), (
            "%s gained a gradient; add it to BACKEND_GRAD_COVERAGE and move "
            "the root into SCAN_ROOTS" % root)


def test_the_manifest_is_exactly_the_gradients_in_the_tree():
    unlisted, stale = compare(BACKEND_GRAD_COVERAGE, _scan_all(SCAN_ROOTS))
    assert unlisted == [], (
        "these backend gradients have no reference route on record; add one "
        "to BACKEND_GRAD_COVERAGE (and the test it names): %s" % unlisted)
    assert stale == [], (
        "these manifest entries no longer exist in the tree: %s" % stale)


def test_the_manifest_has_no_duplicate_rows():
    keys = [(source, symbol) for source, symbol, _, _ in BACKEND_GRAD_COVERAGE]
    assert len(keys) == len(set(keys)), "duplicate rows in BACKEND_GRAD_COVERAGE"


def test_every_route_names_a_test_that_exists():
    missing = sorted({nodeid for _, _, nodeid, _ in BACKEND_GRAD_COVERAGE
                      if not _nodeid_exists(nodeid)})
    assert missing == [], "coverage nodes are missing: %s" % missing


def test_every_kind_is_declared():
    unknown = sorted({kind for _, _, _, kind in BACKEND_GRAD_COVERAGE
                      if kind not in REFERENCE_KINDS})
    assert unknown == [], "undeclared reference kinds: %s" % unknown


def test_the_core_ops_are_scanned_but_hold_no_backend_gradient():
    """A backend cannot escape the inventory by moving into the core tree.

    ``src`` is where the generic ops live; their gradients are
    not backend gradients and are not listed here. But the directory is
    scanned, and if a symbol whose name marks it as backend-owned turns up,
    this fails rather than letting it through unlisted.
    """
    core = scan(CORE_ROOT)
    assert core, "%s matched nothing; the core scan is broken" % CORE_ROOT
    backendish = sorted(
        entry for entry in core
        if re.search(r"^(Cudnn|Cublas|Cub|Cufft|Cutt|Curand|Cusparse|Hipblas|"
                     r"Rocprim|Hccl|Nccl|Mkl|Mpi|Acl)", entry[1]))
    assert backendish == [], (
        "backend-owned gradients appeared under %s; they belong in the "
        "manifest: %s" % (CORE_ROOT, backendish))


# --------------------------------------------------------------------------
# Hardware honesty
# --------------------------------------------------------------------------

def test_deferred_kinds_are_described_in_the_deferred_manual():
    """No entry may be parked on "hardware" without saying what to run.

    Skipping is not passing. The manual carries the prerequisite, the command
    and the pass criterion; this checks each deferred kind's sources are named
    there so hardware day does not start by re-reading the source tree.
    """
    manual = DEFERRED_MANUAL.read_text(encoding="utf-8")
    deferred = {kind for _, _, _, kind in BACKEND_GRAD_COVERAGE
                if kind in DEFERRED_KINDS}
    undocumented = sorted(kind for kind in deferred if kind not in manual)
    assert undocumented == [], (
        "these kinds defer to hardware but %s never names them: %s"
        % (DEFERRED_MANUAL.name, undocumented))


def test_every_gradient_without_a_gradient_test_is_listed_in_the_manual():
    """The open gaps are named individually, not summarised.

    ``npu_hardware_no_grad_test`` means the hardware test that exists covers
    the forward only, so even on an Ascend card the gradient stays unchecked.
    Those are the rows a reader must be able to find by symbol.
    """
    manual = DEFERRED_MANUAL.read_text(encoding="utf-8")
    gaps = [(source, symbol) for source, symbol, _, kind in BACKEND_GRAD_COVERAGE
            if kind in NO_GRAD_TEST_KINDS]
    assert gaps, "NO_GRAD_TEST_KINDS matched nothing; the filter is wrong"
    missing = sorted(symbol for _, symbol in gaps if symbol not in manual)
    assert missing == [], (
        "these gradients have no gradient test on any hardware and are not "
        "named in %s: %s" % (DEFERRED_MANUAL.name, missing))


def test_no_deferred_entry_claims_to_run_here():
    """CUDA is the only accelerator on this machine.

    A ``cuda_*`` kind means the reference really executes; asserting the
    deferred backends never carry one keeps a future edit from quietly
    relabelling an Ascend route as an executed one.
    """
    for source, symbol, _, kind in BACKEND_GRAD_COVERAGE:
        on_cuda = source.startswith(("backends/cuda", "backends/comm/nccl"))
        if kind in EXECUTABLE_KINDS and kind != "cpu_jittor":
            assert on_cuda, (
                "%s is not a CUDA source but claims an executed CUDA "
                "reference (%s)" % (symbol, kind))


# --------------------------------------------------------------------------
# Counter-examples: the gate has to fail, in both directions
# --------------------------------------------------------------------------

def test_a_new_backend_gradient_without_a_manifest_row_is_caught(tmp_path):
    """Direction (a): add a gradient, write no test.

    Both source languages, because the predecessor only understood C++ and
    that is how 32 Python gradients stayed out of the inventory.
    """
    (tmp_path / "fake_op.cc").write_text(
        "#include <x.h>\n"
        "VarPtr FakeBackendOp::grad(Var* out, Var* dout, Var* v, int i) {\n"
        "    return dout;\n}\n", encoding="utf-8")
    (tmp_path / "fake_op.py").write_text(
        "import jittor as jt\n\n"
        "class FakeBackendACL(jt.Function):\n"
        "    def execute(self, x):\n        return x\n\n"
        "    def grad(self, g):\n        return g\n", encoding="utf-8")

    found = {(path.name, symbol)
             for path in sorted(tmp_path.iterdir())
             for symbol in _grads_in_file(path)}
    assert found == {("fake_op.cc", "FakeBackendOp"),
                     ("fake_op.py", "FakeBackendACL")}, found

    unlisted, stale = compare(BACKEND_GRAD_COVERAGE, found)
    assert sorted(unlisted) == [("fake_op.cc", "FakeBackendOp"),
                                ("fake_op.py", "FakeBackendACL")]
    assert stale, "the real manifest should not match a directory of fakes"


def test_a_manifest_row_whose_source_disappeared_is_caught():
    """Direction (b): delete a listed gradient from the tree.

    The tree it compares against is the manifest's own set minus the victim,
    not the live scan, so this reports its own direction even while direction
    (a) is failing for an unrelated reason.
    """
    victim = ("backends/rocm/libraries/rocprim/rocprim_cumsum_op.cc",
              "RocprimCumsumOp")
    assert victim in _scan_all(SCAN_ROOTS), "pick a victim really in the tree"

    listed = {(source, symbol) for source, symbol, _, _ in BACKEND_GRAD_COVERAGE}
    unlisted, stale = compare(BACKEND_GRAD_COVERAGE, listed - {victim})
    assert stale == [victim], stale
    assert unlisted == []


def test_a_route_pointing_at_a_test_that_does_not_exist_is_caught():
    assert not _nodeid_exists("tests/backends/cuda/test_cudnn_conv_plan.py"
                              "::TestCudnnConvPlan::test_that_was_deleted")
    assert not _nodeid_exists("tests/backends/cuda/test_no_such_file.py"
                              "::TestX::test_y")
    assert _nodeid_exists("tests/backends/cuda/test_cudnn_conv_plan.py"
                          "::TestCudnnConvPlan::test_plain_fp32")


def test_an_empty_scan_root_cannot_pass_the_per_root_floor(tmp_path):
    """The floor itself has teeth.

    ``test_each_scan_root_holds_at_least_one_gradient`` is only worth
    something if ``scan`` really returns nothing for a root that matches
    nothing -- the failure mode being guarded against is a mistyped root
    quietly contributing an empty set to a healthy-looking union.
    """
    (tmp_path / "notes.txt").write_text("no operators here\n", encoding="utf-8")
    (tmp_path / "empty_op.py").write_text(
        "class NoGradient:\n    def execute(self, x):\n        return x\n",
        encoding="utf-8")
    found = {symbol for path in tmp_path.iterdir()
             for symbol in _grads_in_file(path)}
    assert found == set(), found

    missing = ROOT / "backends/does_not_exist"
    assert not missing.is_dir(), "pick a root that really is absent"
    assert scan("backends/does_not_exist") == set(), (
        "a mistyped root must contribute nothing, which is what the per-root "
        "floor then catches")
