from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs/testing/torch-install-state-boundary.md"
HELPER = ROOT / "tests/_helpers/child_process.py"
DIST_INSTALLER = ROOT / "python/jittor/compat/torch/installers/distributed.py"
RUNTIME = ROOT / "python/jittor/compat/shim/runtime.py"
INTEGRATIONS = ROOT / "python/jittor/compat/integrations.py"
TORCH_INSTALL = ROOT / "python/jittor/compat/torch/__init__.py"


def test_install_boundary_distinguishes_namespace_and_process_isolation():
    text = GUIDE.read_text(encoding="utf-8")
    for token in (
        "torch*` namespace",
        "jt.flags",
        "os.environ",
        "builtins.__import__",
        "sys.meta_path",
        "module_patcher",
        "child_env()",
        "PYTHONPATH",
        "reversible mutation ledger",
        "hard-failure contract",
        "must",
        "not claim full install rollback",
        "flags.__dict__",
        "JT_NCCL_WORLD_SIZE",
        "JT_NCCL_RANK",
        "JT_NCCL_LOCAL_RANK",
        "JT_NCCL_ROOTINFO_FILE",
        "use_nccl",
        "use_mpi",
        "explicit allowlist",
    ):
        assert token in text


def test_distributed_env_writes_are_explicit_and_child_helper_is_pure():
    helper = HELPER.read_text(encoding="utf-8")
    installer = DIST_INSTALLER.read_text(encoding="utf-8")
    assert "def child_env(" in helper
    assert "env = dict(os.environ) if inherit else {}" in helper
    assert "def set_env(name, value):" in installer
    assert "tx.mutate_env(name, value)" in installer
    assert "tx.mutate_flag(jt.flags, \"use_cuda\", 1)" in installer
    for name in (
        "JT_NCCL_WORLD_SIZE", "JT_NCCL_RANK", "JT_NCCL_LOCAL_RANK",
        "JT_NCCL_ROOTINFO_FILE", "use_nccl", "use_mpi",
    ):
        assert 'set_env("%s"' % name in installer
    assert "JITTOR_TORCH_SHIM" in helper


def test_activation_passes_outer_transaction_and_clears_inner_install_state():
    runtime = RUNTIME.read_text(encoding="utf-8")
    integrations = INTEGRATIONS.read_text(encoding="utf-8")
    torch_install = TORCH_INSTALL.read_text(encoding="utf-8")
    assert "def apply_external_runtime_patches(logger=None, transaction=None):" in integrations
    assert "apply_external_runtime_patches(\n        logger=" in runtime
    assert "transaction=_transaction" in runtime
    assert "transaction=transaction" in integrations
    assert "context.state.pop(\"_install_transaction\", None)" in torch_install
