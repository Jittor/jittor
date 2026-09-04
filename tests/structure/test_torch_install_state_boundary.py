from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs/testing/torch-install-state-boundary.md"
HELPER = ROOT / "tests/_helpers/child_process.py"
DIST_INSTALLER = ROOT / "python/jittor/compat/torch/installers/distributed.py"


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
    for name in (
        "JT_NCCL_WORLD_SIZE", "JT_NCCL_RANK", "JT_NCCL_LOCAL_RANK",
        "JT_NCCL_ROOTINFO_FILE", "use_nccl", "use_mpi",
    ):
        assert 'os.environ["%s"]' % name in installer
    assert "JITTOR_TORCH_SHIM" in helper
