from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "cuda.yml"


def test_cuda_jobs_restore_and_save_a_configuration_partitioned_jittor_cache():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert workflow.count("actions/cache/restore@v4") >= 2
    assert workflow.count("actions/cache/save@v4") >= 2
    assert workflow.count("../jittor-lab/_state/nox/cache") >= 4
    assert "needs.baseline.outputs.cuda_version" in workflow
    assert "cuda_archs-${{ steps.cuda-config.outputs.cuda_archs }}" in workflow
    assert "nvcc_flags-${{ steps.cuda-config.outputs.nvcc_flags_hash }}" in workflow
    assert "python/jittor/src/**" in workflow
    assert "python/jittor/extern/**" in workflow
