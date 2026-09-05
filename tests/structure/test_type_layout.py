"""Contracts for core scalar/container types staying in the type package."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "python" / "jittor" / "src"


def test_nano_types_have_a_single_canonical_source_location():
    assert (SRC / "type" / "nano_string.h").is_file()
    assert (SRC / "type" / "nano_string.cc").is_file()
    assert (SRC / "type" / "nano_vector.h").is_file()
    assert not (SRC / "misc" / "nano_string.h").exists()
    assert not (SRC / "misc" / "nano_string.cc").exists()
    assert not (SRC / "misc" / "nano_vector.h").exists()


def test_nano_type_references_use_the_canonical_include_path():
    candidates = [
        *(SRC.rglob("*.h")),
        *(SRC.rglob("*.cc")),
        *(SRC.rglob("*.cu")),
        *(ROOT / "python" / "jittor" / "extern").rglob("*.h"),
    ]
    stale = []
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "misc/nano_string" in text or "misc/nano_vector" in text:
            stale.append(str(path.relative_to(ROOT)))
    assert stale == []


def test_core_build_lists_nano_string_under_type():
    compiler = (ROOT / "python" / "jittor" / "compiler.py").read_text(
        encoding="utf-8"
    )
    assert '"src/type/nano_string.cc"' in compiler
    assert '"src/misc/nano_string.cc"' not in compiler
