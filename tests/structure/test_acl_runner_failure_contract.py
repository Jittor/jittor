"""ACL runner failures must be attributed and stop before execution continues."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ACL_ROOT = REPO_ROOT / "python" / "jittor" / "extern" / "acl"
BASE_OP = ACL_ROOT / "aclops" / "base_op_acl.cc"
ACLOPS = ACL_ROOT / "aclops"
EXEC = ACL_ROOT / "acl_op_exec.cc"
GUIDE = REPO_ROOT / "docs" / "guides" / "ascend-910b.md"


def _block_body(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for position in range(opening, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:position]
    raise AssertionError("unterminated block: {}".format(marker))


def test_acl_workspace_status_and_runner_lookup_fail_loudly():
    source = BASE_OP.read_text(encoding="utf-8")
    check = _block_body(source, "void BaseOpRunner::checkRet(aclnnStatus ret)")
    assert "ret == ACL_SUCCESS" in check
    assert "LOGf" in check
    assert "name" in check
    assert "acl_error_to_string(ret)" in check
    assert "aclGetRecentErrMsg()" in check
    assert "CHECK_RET" not in check

    run = _block_body(source, "void BaseOpRunner::run()")
    assert run.count("aclOpFuncMap.find(name)") == 2
    assert run.count("it == aclOpFuncMap.end()") == 2
    assert "ACL operator has no registered launcher" in run

    callers = sum(path.read_text(encoding="utf-8").count("checkRet(")
                  for path in ACLOPS.glob("*_acl.cc") if path != BASE_OP)
    assert callers == 65, "every maintained executeOp tail must use checkRet"


def test_acl_fused_queue_checks_the_current_op_without_shadowing():
    source = EXEC.read_text(encoding="utf-8")
    loop = _block_body(source, "while (!queue.empty())")
    assert "auto *current_op = queue.front();" in loop
    assert "auto op = queue.front();" not in loop
    assert loop.index("auto *current_op = queue.front();") < loop.index(
        "current_op->inputs()")
    assert loop.count("current_op->inputs()") == 2
    assert "current_op->outputs()" in loop
    assert "current_op->name()" in loop
    assert "current fused operator input is not allocated" in loop


def test_ascend_guide_records_runner_failure_attribution():
    guide = GUIDE.read_text(encoding="utf-8")
    for required in (
        "aclnn workspace-size query failed",
        "ACL operator has no registered launcher",
        "current fused operator input",
        "return code",
        "operator name",
        "fallback cpu",
    ):
        assert required in guide
