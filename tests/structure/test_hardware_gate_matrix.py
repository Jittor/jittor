from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = REPO_ROOT / ".github" / "workflows"
TEST_SYSTEM = REPO_ROOT / "docs" / "testing" / "test-system.md"
MANUAL_SESSIONS = ("optional", "rocm", "mpi", "nccl")


def _workflow_text():
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(WORKFLOW_ROOT.glob("*.yml"))
    )


def test_manual_hardware_sessions_are_documented_and_not_scheduled():
    documentation = TEST_SYSTEM.read_text(encoding="utf-8")
    workflows = _workflow_text()

    for session in MANUAL_SESSIONS:
        assert "| `%s` | Manual |" % session in documentation
        assert "nox -s %s" % session not in workflows


def test_cuda_labeled_pull_requests_run_the_real_cuda_session():
    workflow = (WORKFLOW_ROOT / "cuda.yml").read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "types: [labeled, reopened, synchronize]" in workflow
    assert "contains(github.event.pull_request.labels.*.name, 'ci:cuda')" in workflow
    assert '"${JITTOR_CI_PYTHON}" -m nox -s cuda' in workflow
    assert "| `cuda` | Automated |" in TEST_SYSTEM.read_text(encoding="utf-8")
