"""Contracts for vendored C++ sources kept out of core module directories."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "python" / "jittor" / "src"


def test_miniz_is_classified_as_third_party_source():
    third_party = SRC / "third_party"
    misc = SRC / "misc"
    miniz_cc = third_party / "miniz.cc"
    miniz_h = third_party / "miniz.h"

    assert third_party.is_dir()
    assert miniz_cc.is_file()
    assert miniz_h.is_file()
    assert not (misc / "miniz.cc").exists()
    assert not (misc / "miniz.h").exists()
    assert '#include  "third_party/miniz.h"' in miniz_cc.read_text()

    # The core builder discovers sources recursively, so this move must not
    # require a second source manifest or a path-specific compiler exception.
    compiler = (ROOT / "python" / "jittor" / "compiler.py").read_text()
    assert 'glob.glob(jittor_path+"/src/**/*."+ext_args, recursive=True)' in compiler
