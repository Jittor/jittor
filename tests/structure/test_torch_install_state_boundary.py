from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs/testing/torch-install-state-boundary.md"


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
