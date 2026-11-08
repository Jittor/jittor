"""External state paths shared by manual core profiling tools."""
import os
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("JITTOR_REPO_ROOT", SCRIPT_ROOT.parents[1])).resolve()
LAB_ROOT = Path(os.environ.get("JITTOR_LAB_ROOT", REPO_ROOT.parent / "jittor-lab")).resolve()
WORK_ROOT = LAB_ROOT / "_state" / "profiling" / os.environ.get("cache_name", "manual")


def resolve_source_root(value=None):
    if value:
        root = Path(value).expanduser().resolve()
    else:
        # Share the profiling directory's canonical discovery, including the
        # optional JITTOR_REPO_ROOT override, after the tools' relocation.
        root = REPO_ROOT
    if not (root / 'python' / 'jittor').is_dir() or not (root / 'AGENTS.md').is_file():
        raise ValueError('Cannot locate the source checkout; pass --source-root explicitly')
    return root


def lab_output_path(value, source_root):
    """Keep reports, caches prepared here, and images outside the checkout."""
    configured = os.environ.get('JITTOR_LAB_ROOT')
    if not configured:
        raise ValueError('JITTOR_LAB_ROOT must name an external laboratory directory')
    lab = Path(configured).expanduser().resolve()
    target = Path(value).expanduser().resolve()
    try:
        target.relative_to(lab)
    except ValueError:
        raise ValueError('Output paths must be below JITTOR_LAB_ROOT') from None
    try:
        target.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError('Output paths must be outside the source checkout')
    if target == lab:
        raise ValueError('Output must name a file below JITTOR_LAB_ROOT')
    return target
