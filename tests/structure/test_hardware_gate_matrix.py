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


def test_cuda_session_requires_real_device_and_an_executed_accelerator_case():
    source = (REPO_ROOT / "noxfile.py").read_text(encoding="utf-8")
    cuda_start = source.index("def cuda(session):")
    cuda_end = source.index("\n\n@nox.session", cuda_start)
    cuda = source[cuda_start:cuda_end]
    assert 'env["JITTOR_TEST_REQUIRE_CUDA"] = "1"' in cuda
    assert 'env["JITTOR_TEST_ACCELERATOR_MIN_EXECUTED"] = "1"' in cuda

    policy = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "has_cuda is false" in policy
    assert "_ACCELERATOR_EXECUTED < required_accelerator" in policy


def test_npu_session_requires_real_acl_and_an_executed_accelerator_case():
    source = (REPO_ROOT / "noxfile.py").read_text(encoding="utf-8")
    npu_start = source.index("def npu(session):")
    npu_end = source.index("\n\n@nox.session", npu_start)
    npu = source[npu_start:npu_end]
    assert 'env["JITTOR_TEST_REQUIRE_ACL"] = "1"' in npu
    assert 'env["JITTOR_TEST_ACCELERATOR_MIN_EXECUTED"] = "1"' in npu

    policy = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "JITTOR_TEST_REQUIRE_ACL" in policy
    assert "has_acl is false" in policy
    assert "_ACCELERATOR_EXECUTED < required_accelerator" in policy


def test_accelerator_execution_counter_uses_cuda_nodeids(monkeypatch):
    import conftest as policy

    class Report:
        nodeid = "tests/backends/cuda/test_matmul.py::test_forward"

    class CpuReport:
        nodeid = "tests/backends/cpu/test_matmul.py::test_forward"

    assert policy._is_accelerator_case(Report())
    assert not policy._is_accelerator_case(CpuReport())
    monkeypatch.setenv("JITTOR_TEST_ACCELERATOR_MIN_EXECUTED", "2")
    assert policy._required_accelerator_executions() == 2
