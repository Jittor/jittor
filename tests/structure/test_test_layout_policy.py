"""Both test roots share selection policy and retain pre-move input seeds."""
import ast
import runpy
from types import SimpleNamespace

import pytest

from _helpers import gate_scope, pytest_policy
from _helpers.layout_seed import seed_nodeid
from _helpers import layout_seed
from _helpers.paths import REPO_ROOT


def test_shared_policy_and_root_adapters_remain_in_lint_and_format_gates():
    source = ast.parse((REPO_ROOT / "noxfile.py").read_text())
    selected = {"PYTEST_POLICY_FILES", "RATCHET_FILES", "FORMAT_FILES"}
    assignments = [node for node in source.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id in selected
                           for target in node.targets)]
    namespace = {"NN_MIGRATION_FILES": ()}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "noxfile.py", "exec"), namespace)
    expected = {"tests/_helpers/pytest_policy.py", "tests/conftest.py",
                "compat/tests/conftest.py", "adapters/tests/conftest.py"}
    for name in ("RATCHET_FILES", "FORMAT_FILES"):
        assert expected <= set(namespace[name])


def test_both_root_adapters_register_the_same_policy_once():
    registered = {}
    manager = SimpleNamespace(hasplugin=registered.__contains__,
                              register=lambda module, name: registered.setdefault(name, module))
    for root in (REPO_ROOT / "tests", REPO_ROOT / "compat/tests", REPO_ROOT / "adapters/tests"):
        adapter = runpy.run_path(str(root / "conftest.py"))
        adapter["pytest_addoption"](None, manager)
    assert registered == {"_helpers.pytest_policy": pytest_policy}


def test_standalone_adapter_host_tests_do_not_require_monorepo_support(tmp_path):
    root = tmp_path / "distribution" / "tests"
    root.mkdir(parents=True)
    adapter = root / "conftest.py"
    adapter.write_text((REPO_ROOT / "adapters/tests/conftest.py").read_text())
    namespace = runpy.run_path(str(adapter))
    registered = []
    manager = SimpleNamespace(hasplugin=lambda name: False,
                              register=lambda *args: registered.append(args))
    namespace["pytest_addoption"](None, manager)
    assert not registered


def test_moved_backend_paths_keep_device_and_process_contracts():
    assert pytest_policy._backend_markers(None, ("backends", "acl", "test_acl.py")) == ("npu",)
    assert pytest_policy._backend_markers(None, ("backends", "comm", "nccl", "test_nccl.py")) == ("mpi",)
    assert pytest_policy._backend_markers(None, ("fsdp2", "test_fsdp_mesh.py")) == ("mpi",)
    assert pytest_policy._relative_to_test_root(REPO_ROOT / "compat/tests/torch/test_transaction.py") == (
        "torch", "test_transaction.py")
    native = gate_scope.selected_files(REPO_ROOT, gate_scope.native_arguments())
    torch = gate_scope.selected_files(REPO_ROOT, gate_scope.torch_arguments())
    assert "compat/tests/fsdp2/test_fsdp_mesh.py" in native
    assert "compat/tests/torch/test_transaction.py" in torch
    native_mock = "tests/structure/backends/acl/test_acl_dtype_preservation.py"
    assert native_mock in native and native_mock not in torch
    assert not native & torch


def test_seed_mapping_changes_only_existing_test_file_prefix():
    before = "tests/core/test_flags.py::TestFlags::test_get_set[case-a]"
    after = "tests/runtime/test_flags.py::TestFlags::test_get_set[case-a]"
    assert seed_nodeid(after) == before
    new_case = "tests/runtime/test_flags.py::TestFlags::test_new_after_layout"
    assert seed_nodeid(new_case) == new_case
    assert seed_nodeid("tests/new_file.py::test_new") == "tests/new_file.py::test_new"


def test_seed_identity_rejects_new_classes_and_explicit_generator_name_collisions(monkeypatch, tmp_path):
    source = tmp_path / "tests/moved.py"
    source.parent.mkdir()
    source.write_text("class Existing:\n"
                      "    def test_reference(self): pass\n"
                      "    def test_reference_extra(self): pass\n"
                      "class Added:\n"
                      "    def test_reference(self): pass\n")
    monkeypatch.setattr(layout_seed, "_ROOT", tmp_path)
    monkeypatch.setattr(layout_seed, "_MOVED", {"tests/moved.py": {
        "path": "tests/old.py", "cases": ["Existing::test_reference"],
        "families": ["Existing::test_reference"],
        "device_classes": {"ExistingCPU": "Existing"}, "dynamic_cases": [],
    }})
    for suffix in ("Existing::test_reference[p]", "ExistingCPU::test_reference_float32[p]"):
        assert seed_nodeid("tests/moved.py::" + suffix) == "tests/old.py::" + suffix
    for suffix in ("Added::test_reference[p]", "Existing::test_reference_extra[p]",
                   "ExistingCPU::test_reference_extra[p]"):
        nodeid = "tests/moved.py::" + suffix
        assert seed_nodeid(nodeid) == nodeid


def test_collection_traverses_torch_ancestors_for_explicit_native_mock(monkeypatch):
    monkeypatch.setattr(pytest_policy, "_torch_mode_is_active", lambda: False)
    assert pytest_policy.pytest_ignore_collect(REPO_ROOT / "tests/structure", None) is None
    assert pytest_policy.pytest_ignore_collect(REPO_ROOT / "tests/structure/backends/acl", None) is None
    assert pytest_policy.pytest_ignore_collect(REPO_ROOT / "tests/structure/test_gate_scope.py", None) is True


def test_sessionstart_executes_accelerator_and_legacy_selection_guards(monkeypatch):
    calls = []
    monkeypatch.setattr(pytest_policy, "_require_real_accelerator", lambda: calls.append("accelerator"))
    monkeypatch.setattr(pytest_policy, "_torch_mode_is_active", lambda: True)
    for key in pytest_policy._LEGACY_SELECTION:
        monkeypatch.delenv(key, raising=False)
    session = SimpleNamespace(config=SimpleNamespace(args=[]))
    pytest_policy.pytest_sessionstart(session)
    assert calls == ["accelerator"]
    monkeypatch.setenv("test_skip_l", "1")
    with pytest.raises(pytest.UsageError, match="legacy jittor.test selection"):
        pytest_policy.pytest_sessionstart(session)
    assert calls == ["accelerator", "accelerator"]


def test_compat_working_directory_uses_repository_relative_execution_accounting(monkeypatch):
    monkeypatch.setattr(pytest_policy, "_SELECTED_FILES", set())
    monkeypatch.setattr(pytest_policy, "_PYTEST_ROOT", REPO_ROOT / "compat")
    config = SimpleNamespace(args=["tests/structure/test_factory_install_owners.py"],
                             invocation_params=SimpleNamespace(dir=REPO_ROOT / "compat"),
                             option=SimpleNamespace(ignore=[]))
    pytest_policy._snapshot_selected_files(config)
    expected = "compat/tests/structure/test_factory_install_owners.py"
    assert pytest_policy._SELECTED_FILES == {expected}
    assert pytest_policy._relative_to_repo("tests/structure/test_factory_install_owners.py") == expected


def test_sessionfinish_flushes_worker_survey_and_keeps_execution_gate(monkeypatch):
    flushed = []
    monkeypatch.setattr(pytest_policy, "_write_state_leak_report", lambda suffix: flushed.append(suffix))
    monkeypatch.setattr(pytest_policy, "_required_accelerator_executions", lambda: 0)
    monkeypatch.setattr(pytest_policy, "_requires_execution", lambda: True)
    monkeypatch.setattr(pytest_policy, "_execution_exemptions", lambda: {})
    monkeypatch.setattr(pytest_policy, "_files_that_executed_nothing", lambda: [])
    monkeypatch.setattr(pytest_policy, "_files_that_collected_nothing", lambda: ["missing.py"])
    session = SimpleNamespace(config=SimpleNamespace(workerinput={"workerid": "gw2"},
                                                     option=SimpleNamespace(collectonly=False)),
                              exitstatus=0)
    pytest_policy.pytest_sessionfinish(session, 0)
    assert flushed == [".gw2"]
    assert session.exitstatus == 1
