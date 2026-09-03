"""Static contract for ACL per-operator synchronization diagnostics."""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_OP = REPO_ROOT / "python" / "jittor" / "extern" / "acl" / "aclops" / "base_op_acl.cc"
GUIDE = REPO_ROOT / "docs" / "guides" / "ascend-910b.md"


def _function_body(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for position in range(opening, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:position]
    raise AssertionError("unterminated function: {}".format(signature))


def _without_comments(source):
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//.*", "", source)


def test_acl_sync_run_checks_the_stream_and_documents_the_910b3_probe():
    source = BASE_OP.read_text(encoding="utf-8")
    body = _without_comments(_function_body(source, "void BaseOpRunner::syncRun()"))
    assert "if (!sync_run)" in body
    assert "aclrtSynchronizeStream(aclstream)" in body
    assert "sync_ret != ACL_SUCCESS" in body
    assert "LOGf" in body
    assert "name" in body
    assert "acl_error_to_string(sync_ret)" in body

    guide = GUIDE.read_text(encoding="utf-8")
    for required in (
        "Ascend 910B3",
        "npu-smi info",
        "sync_run=1",
        "sync_run=0",
        "aclrtSynchronizeStream",
        "fallback cpu",
        "return code",
        "operator name",
    ):
        assert required in guide
