from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "python/jittor/extern/acl/acl_op_exec.cc"


def test_acl_disables_cuda_external_ops_by_explicit_registry():
    text = SOURCE.read_text(encoding="utf-8")
    assert "acl_cuda_external_ops" in text
    assert 'startswith(name, "cu")' not in text
    assert "acl_cuda_external_ops.count(name) != 0" in text
    for name in (
        "cublas_matmul",
        "cudnn_conv",
        "cufft_fft",
        "curand_random",
        "cusparse_spmmcsr",
        "cutt_transpose",
    ):
        assert f'"{name}"' in text

