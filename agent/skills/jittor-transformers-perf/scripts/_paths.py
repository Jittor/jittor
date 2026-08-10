import os
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("JITTOR_REPO_ROOT", SCRIPT_ROOT.parents[3])).resolve()
LAB_ROOT = Path(
    os.environ.get("JITTOR_LAB_ROOT", REPO_ROOT.parent / "jittor-lab")
).resolve()
WORK_ROOT = Path(
    os.environ.get(
        "JITTOR_TRANSFORMERS_PERF_WORKDIR",
        LAB_ROOT / "jittor_transformers_perf",
    )
).resolve()
RUNTIME_ROOT = WORK_ROOT / "runtime"
