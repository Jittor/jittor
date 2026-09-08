from pathlib import Path
from contextlib import contextmanager
import sys
from types import SimpleNamespace
import pytest


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

    policy = (REPO_ROOT / "tests/_helpers/pytest_policy.py").read_text(encoding="utf-8")
    assert "jt.introspection.capabilities.backend(name)" in policy
    assert '("cuda", require_cuda)' in policy
    assert "_ACCELERATOR_EXECUTED < required_accelerator" in policy


def test_npu_session_requires_real_acl_and_an_executed_accelerator_case():
    source = (REPO_ROOT / "noxfile.py").read_text(encoding="utf-8")
    npu_start = source.index("def npu(session):")
    npu_end = source.index("\n\n@nox.session", npu_start)
    npu = source[npu_start:npu_end]
    assert 'env["JITTOR_TEST_REQUIRE_ACL"] = "1"' in npu
    assert 'env["JITTOR_TEST_ACCELERATOR_MIN_EXECUTED"] = "1"' in npu

    policy = (REPO_ROOT / "tests/_helpers/pytest_policy.py").read_text(encoding="utf-8")
    assert "JITTOR_TEST_REQUIRE_ACL" in policy
    assert '("acl", require_acl)' in policy
    assert "capability.state.value, capability.reason" in policy
    assert "_ACCELERATOR_EXECUTED < required_accelerator" in policy


def test_nccl_session_rejects_an_all_skipped_communication_gate():
    source = (REPO_ROOT / "noxfile.py").read_text(encoding="utf-8")
    nccl_start = source.index("def nccl(session):")
    nccl_end = len(source)
    nccl = source[nccl_start:nccl_end]
    assert 'env["JITTOR_TEST_REQUIRE_EXECUTION"] = "1"' in nccl
    assert 'env["JITTOR_TEST_ACCELERATOR_MIN_EXECUTED"] = "1"' in nccl

    policy = (REPO_ROOT / "tests/_helpers/pytest_policy.py").read_text(encoding="utf-8")
    assert '"nccl", "hccl"' in policy


def test_accelerator_execution_counter_uses_cuda_nodeids(monkeypatch):
    from _helpers import pytest_policy as policy

    class Report:
        nodeid = "tests/backends/cuda/test_matmul.py::test_forward"

    class CpuReport:
        nodeid = "tests/backends/cpu/test_matmul.py::test_forward"

    assert policy._is_accelerator_case(Report())
    assert not policy._is_accelerator_case(CpuReport())

    class NcclReport:
        nodeid = "tests/backends/comm/nccl/test_nccl_comm_stream.py::test_all_reduce"

    assert policy._is_accelerator_case(NcclReport())
    monkeypatch.setenv("JITTOR_TEST_ACCELERATOR_MIN_EXECUTED", "2")
    assert policy._required_accelerator_executions() == 2


def _capability(state):
    return SimpleNamespace(enabled=state == "available", failed=state == "failed",
                           unprobed=state == "unprobed", state=SimpleNamespace(value=state),
                           reason="driver/probe evidence")


@pytest.mark.parametrize("name,variable", [("cuda", "JITTOR_TEST_REQUIRE_CUDA"), ("acl", "JITTOR_TEST_REQUIRE_ACL")])
@pytest.mark.parametrize("state", ["available", "disabled", "failed", "unprobed"])
def test_declared_gate_observes_the_requested_backend(monkeypatch, name, variable, state):
    from _helpers import pytest_policy as policy
    calls = []
    def query(backend):
        calls.append(backend)
        return _capability(state)
    owner = SimpleNamespace(introspection=SimpleNamespace(capabilities=SimpleNamespace(backend=query)))
    monkeypatch.setitem(sys.modules, "jittor", owner)
    monkeypatch.delenv("JITTOR_TEST_REQUIRE_CUDA", raising=False)
    monkeypatch.delenv("JITTOR_TEST_REQUIRE_ACL", raising=False)
    monkeypatch.setenv(variable, "1")
    if state == "available":
        policy._require_real_accelerator()
    else:
        with pytest.raises(pytest.UsageError, match=state + ": driver/probe evidence"):
            policy._require_real_accelerator()
    assert calls == [name]


def test_rocm_fixture_scope_covers_yield_and_failure(monkeypatch):
    from _helpers import pytest_policy as policy
    values = {"use_rocm": 0}
    @contextmanager
    def scope(**changes):
        assert changes == {"use_rocm": 1}
        before = values["use_rocm"]
        values.update(changes)
        try:
            yield
        finally:
            values["use_rocm"] = before
    owner = SimpleNamespace(runtime=SimpleNamespace(scope=scope), introspection=SimpleNamespace(
        capabilities=SimpleNamespace(backend=lambda name: _capability("available"))))
    monkeypatch.setitem(sys.modules, "jittor", owner)
    request = SimpleNamespace(node=SimpleNamespace(get_closest_marker=lambda name: object()))
    fixture = policy.rocm_backend.__wrapped__(request)
    next(fixture)
    assert values["use_rocm"] == 1
    with pytest.raises(ValueError, match="body failed"):
        fixture.throw(ValueError("body failed"))
    assert values["use_rocm"] == 0


@pytest.mark.parametrize("state", ["disabled", "failed", "unprobed"])
def test_rocm_fixture_never_skips_a_failed_or_unprobed_backend(monkeypatch, state):
    from _helpers import pytest_policy as policy
    owner = SimpleNamespace(introspection=SimpleNamespace(capabilities=SimpleNamespace(
        backend=lambda name: _capability(state))))
    monkeypatch.setitem(sys.modules, "jittor", owner)
    request = SimpleNamespace(node=SimpleNamespace(get_closest_marker=lambda name: object()))
    expected = pytest.skip.Exception if state == "disabled" else pytest.UsageError
    with pytest.raises(expected, match=state):
        next(policy.rocm_backend.__wrapped__(request))
