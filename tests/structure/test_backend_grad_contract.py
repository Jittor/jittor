"""Every backend gradient implementation has an auditable reference route.

The manifest deliberately distinguishes CPU/Jittor, independent NumPy, and
distributed or hardware-only references.  A backend implementation must not
silently disappear from the maintained test inventory when a new ``grad``
definition is added under ``python/jittor/extern``.
"""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


BACKEND_GRAD_COVERAGE = (
    # ACL/HCCL has no host implementation on this machine.  Keep the source
    # and the NPU/macro route visible rather than pretending a CPU fallback.
    ("python/jittor/extern/acl/hccl/ops/hccl_all_gather_op.cc",
     "tests/distributed/test_hccl_check_macros.py::TestHcclCheckMacros::test_finalizer_does_not_use_the_throwing_macro",
     "unsupported_hardware"),
    ("python/jittor/extern/acl/hccl/ops/hccl_all_reduce_op.cc",
     "tests/backends/npu/test_acl.py::TestACL::test_product_reduction_forward_backward",
     "npu_hardware"),
    ("python/jittor/extern/acl/hccl/ops/hccl_broadcast_op.cc",
     "tests/backends/npu/test_acl.py::TestACL::test_broadcast",
     "npu_hardware"),
    ("python/jittor/extern/acl/hccl/ops/hccl_reduce_op.cc",
     "tests/backends/npu/test_acl.py::TestACL::test_all_reduction",
     "npu_hardware"),

    # CUDA/CUB.
    ("python/jittor/extern/cuda/cub/ops/cub_arg_reduce_op.cc",
     "tests/ops/test_arg_reduce_op.py::TestArgReduceOp::test_backward_cuda",
     "cuda_cpu_formula"),
    ("python/jittor/extern/cuda/cub/ops/cub_argsort_op.cc",
     "tests/ops/test_argsort_op.py::TestArgsortOp::test_cub_backward",
     "cuda_cpu_formula"),
    ("python/jittor/extern/cuda/cub/ops/cub_cumsum_op.cc",
     "tests/backends/cuda/test_cub_cumsum.py::TestCubCumsumOp::test_1d_backward",
     "cuda_cpu_jittor"),
    ("python/jittor/extern/cuda/cublas/ops/cublas_matmul_op.cc",
     "tests/backends/cuda/test_cublas_matmul_grad.py::TestCublasMatmulGrad::test_all_transpose_combinations",
     "cuda_numpy"),
    ("python/jittor/extern/cuda/cublas/ops/cublas_batched_matmul_op.cc",
     "tests/backends/cuda/test_cublas_matmul_grad.py::TestCublasMatmulGrad::test_linear_3d_random_projection_grad",
     "cuda_numpy"),
    ("python/jittor/extern/cuda/cudnn/ops/cudnn_conv_op.cc",
     "tests/backends/cuda/test_cudnn_conv_plan.py::TestCudnnConvPlan::test_plain_fp32",
     "cuda_cpu_jittor"),
    ("python/jittor/extern/cuda/cudnn/ops/cudnn_conv_backward_x_op.cc",
     "tests/backends/cuda/test_cudnn_conv_plan.py::TestCudnnConvPlan::test_plain_fp32",
     "cuda_cpu_jittor"),
    ("python/jittor/extern/cuda/cudnn/ops/cudnn_conv_backward_w_op.cc",
     "tests/backends/cuda/test_cudnn_conv_plan.py::TestCudnnConvPlan::test_plain_fp32",
     "cuda_cpu_jittor"),
    ("python/jittor/extern/cuda/cudnn/ops/cudnn_conv3d_op.cc",
     "tests/backends/cuda/test_cudnn_conv3d_algo_cache.py::TestCudnnConv3dAlgoCache::test_forward_and_gradients_match_cpu_reference",
     "cuda_cpu_jittor"),
    ("python/jittor/extern/cuda/cudnn/ops/cudnn_conv3d_backward_x_op.cc",
     "tests/backends/cuda/test_cudnn_conv3d_algo_cache.py::TestCudnnConv3dAlgoCache::test_forward_and_gradients_match_cpu_reference",
     "cuda_cpu_jittor"),
    ("python/jittor/extern/cuda/cudnn/ops/cudnn_conv3d_backward_w_op.cc",
     "tests/backends/cuda/test_cudnn_conv3d_algo_cache.py::TestCudnnConv3dAlgoCache::test_forward_and_gradients_match_cpu_reference",
     "cuda_cpu_jittor"),
    ("python/jittor/extern/cuda/cufft/ops/cufft_fft_op.cc",
     "tests/ops/test_fft_op.py::TestFFTOp::test_fft_backward",
     "cuda_numpy"),
    ("python/jittor/extern/cuda/cutt/ops/cutt_transpose_op.cc",
     "tests/backends/cuda/test_cutt_transpose_op.py::TestCuttTransposeOp::test_grad",
     "cuda_numpy"),

    # CUDA/NCCL.  The FSDP2 node exercises all_gather's reverse
    # reduce-scatter path as well as the forward collective.
    ("python/jittor/extern/cuda/nccl/ops/nccl_all_gather_op.cc",
     "tests/distributed/test_fsdp2_nccl.py::TestFSDP2Nccl::test_nccl_all_gather_autograd",
     "nccl_hardware"),
    ("python/jittor/extern/cuda/nccl/ops/nccl_all_reduce_op.cc",
     "tests/distributed/test_nccl_ops.py::TestNcclOps::test_all_reduce",
     "nccl_hardware"),
    ("python/jittor/extern/cuda/nccl/ops/nccl_broadcast_op.cc",
     "tests/distributed/test_nccl_ops.py::TestNcclOps::test_broadcast",
     "nccl_hardware"),
    ("python/jittor/extern/cuda/nccl/ops/nccl_reduce_op.cc",
     "tests/distributed/test_nccl_ops.py::TestNcclOps::test_reduce",
     "nccl_hardware"),
    ("python/jittor/extern/cuda/nccl/ops/nccl_reduce_scatter_op.cc",
     "tests/distributed/test_fsdp2_nccl.py::TestFSDP2Nccl::test_nccl_all_gather_autograd",
     "nccl_hardware"),

    # CPU oneDNN and MPI.
    ("python/jittor/extern/mkl/ops/mkl_batched_matmul_op.cc",
     "tests/ops/test_mkl_batched_matmul.py::TestMklBatchedMatmul::test_three_dimensional_batch",
     "cpu_jittor"),
    ("python/jittor/extern/mpi/ops/mpi_all_reduce_op.cc",
     "tests/distributed/test_mpi_op.py::TestMpiOps::test_all_reduce",
     "mpi_hardware"),
    ("python/jittor/extern/mpi/ops/mpi_broadcast_op.cc",
     "tests/distributed/test_mpi_op.py::TestMpiOps::test_broadcast",
     "mpi_hardware"),
    ("python/jittor/extern/mpi/ops/mpi_reduce_op.cc",
     "tests/distributed/test_mpi_op.py::TestMpiOps::test_reduce",
     "mpi_hardware"),
)

REFERENCE_KINDS = {
    "cpu_jittor", "cuda_cpu_jittor", "cuda_cpu_formula", "cuda_numpy",
    "mpi_hardware", "nccl_hardware", "npu_hardware",
    "unsupported_hardware",
}


def _source_grad_definitions():
    extern = ROOT / "python/jittor/extern"
    found = set()
    for path in extern.rglob("*.cc"):
        if "/ops/" not in path.as_posix():
            continue
        text = path.read_text(encoding="utf-8")
        if "::grad(" in text:
            found.add(path.relative_to(ROOT).as_posix())
    return found


def _assert_nodeid_exists(nodeid):
    parts = nodeid.split("::")
    assert len(parts) in (2, 3), "unsupported nodeid: {}".format(nodeid)
    path = ROOT / parts[0]
    assert path.is_file(), "coverage node file is missing: {}".format(nodeid)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = tree.body
    for index, name in enumerate(parts[1:]):
        if index == 0 and len(parts) == 3:
            kinds = (ast.ClassDef,)
        else:
            kinds = (ast.FunctionDef, ast.AsyncFunctionDef)
        node = next((item for item in body
                     if isinstance(item, kinds) and item.name == name), None)
        assert node is not None, "coverage node is missing: {}".format(nodeid)
        body = node.body


def test_every_extern_grad_has_one_reference_route():
    entries = {source: (nodeid, kind)
               for source, nodeid, kind in BACKEND_GRAD_COVERAGE}
    assert len(entries) == len(BACKEND_GRAD_COVERAGE) == 26
    assert set(entries) == _source_grad_definitions()
    for source, (nodeid, kind) in entries.items():
        assert (ROOT / source).is_file(), source
        assert kind in REFERENCE_KINDS, (source, kind)
        _assert_nodeid_exists(nodeid)


def test_hardware_only_routes_are_explicit():
    for source, nodeid, kind in BACKEND_GRAD_COVERAGE:
        if kind.endswith("hardware") or kind == "unsupported_hardware":
            assert nodeid.startswith("tests/"), (source, nodeid)
            assert kind in {"mpi_hardware", "nccl_hardware", "npu_hardware",
                            "unsupported_hardware"}
