from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACLNN_HEADER = ROOT / "python/jittor/extern/acl/aclnn/aclnn.h"


def test_aclnn_header_has_include_guard_before_declarations():
    lines = ACLNN_HEADER.read_text().splitlines()
    first = next(line.strip() for line in lines if line.strip())
    assert first == "#pragma once"
