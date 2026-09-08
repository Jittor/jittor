"""Optional host-only harness for import contracts, without native compilation.

Numerical tests use the real installed backend unless explicitly requested.
"""
import os
import sys
import types
from pathlib import Path


if os.environ.get("JITTOR_VLLM_HOST_ONLY") == "1":
    source = Path(os.environ.get("JITTOR_COMPAT_SOURCE", Path(__file__).resolve().parents[3] / "compat")).resolve()
    if not (source / "transaction.py").is_file():
        raise RuntimeError("JITTOR_COMPAT_SOURCE must identify the real compat source")
    root = types.ModuleType("jittor")
    root.__path__ = []
    compat = types.ModuleType("jittor.compat")
    compat.__path__ = [str(source)]
    root.compat = compat
    sys.modules["jittor"] = root
    sys.modules["jittor.compat"] = compat
