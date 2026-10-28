from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "cuda.yml"


def test_cuda_jobs_restore_and_save_a_configuration_partitioned_jittor_cache():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "../jittor-lab/_state/nox/cache" not in workflow
    jobs = yaml.safe_load(workflow)["jobs"]
    actions = {"actions/cache/restore@v4", "actions/cache/save@v4"}
    partition = (
        "needs.baseline.outputs.cuda_version",
        "cuda_archs-${{ steps.cuda-config.outputs.cuda_archs }}",
        "nvcc_flags-${{ steps.cuda-config.outputs.nvcc_flags_hash }}",
    )
    for name in ("cuda", "benchmark-cuda"):
        steps = jobs[name]["steps"]
        # Both jobs explicitly place JITTOR_LAB_ROOT outside the checkout.
        assert any('JITTOR_LAB_ROOT=${RUNNER_TEMP}/jittor-lab' in step.get("run", "")
                   for step in steps), name
        caches = [step for step in steps if step.get("uses") in actions
                  and step.get("with", {}).get("key", "").startswith("jittor-build-cuda-")]
        assert len(caches) == 2, name
        assert {step["uses"] for step in caches} == actions, name
        for step in caches:
            config = step["with"]
            assert config["path"] == "${{ runner.temp }}/jittor-lab/_state/nox/cache", name
            assert all(field in config["key"] for field in partition), name
            assert "src/**" in config["key"] and "backends/**" in config["key"], name
            assert "python/jittor/extern/**" not in config["key"], name
            if step["uses"] == "actions/cache/restore@v4":
                assert all(field in config["restore-keys"] for field in partition), name
