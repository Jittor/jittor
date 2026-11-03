"""External state paths shared by manual core profiling tools."""
import os
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("JITTOR_REPO_ROOT", SCRIPT_ROOT.parents[1])).resolve()
LAB_ROOT = Path(os.environ.get("JITTOR_LAB_ROOT", REPO_ROOT.parent / "jittor-lab")).resolve()
WORK_ROOT = LAB_ROOT / "_state" / "profiling" / os.environ.get("cache_name", "manual")
