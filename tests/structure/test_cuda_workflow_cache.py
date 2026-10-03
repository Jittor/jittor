from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "cuda.yml"


def test_cuda_jobs_restore_and_save_a_configuration_partitioned_jittor_cache():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert workflow.count("actions/cache/restore@v4") >= 2
    assert workflow.count("actions/cache/save@v4") >= 2
    assert workflow.count("path: ${{ env.JITTOR_NOX_CACHE_PATH }}") == 4
    assert workflow.count("realpath -m") == 2
    assert "path: ${{ github.workspace }}/../jittor-lab" not in workflow
    assert "needs.baseline.outputs.cuda_version" in workflow
    assert "cuda_archs-${{ steps.cuda-config.outputs.cuda_archs }}" in workflow
    assert "nvcc_flags-${{ steps.cuda-config.outputs.nvcc_flags_hash }}" in workflow
    assert "src/**" in workflow
    assert "backends/**" in workflow
    assert "python/jittor/extern/**" not in workflow


def test_cpu_jobs_pass_only_normalized_paths_to_actions_cache():
    workflow = (REPO_ROOT / ".github" / "workflows" / "cpu.yml").read_text(
        encoding="utf-8"
    )

    assert workflow.count("path: ${{ env.JITTOR_NOX_CACHE_PATH }}") == 3
    assert workflow.count("realpath -m") == 2
    assert "path: ${{ github.workspace }}/../jittor-lab" not in workflow
