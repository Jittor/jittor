"""ACL runner failures must be attributed and stop before execution continues."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ACL_ROOT = REPO_ROOT / "python" / "jittor" / "extern" / "acl"
BASE_OP = ACL_ROOT / "aclops" / "base_op_acl.cc"
ACLOPS = ACL_ROOT / "aclops"
EXEC = ACL_ROOT / "acl_op_exec.cc"
GUIDE = REPO_ROOT / "docs" / "guides" / "ascend-910b.md"

# The aclnn execute call takes exactly these four arguments, so its presence is
# a reliable marker of a runner that drives the launch itself instead of going
# through BaseOpRunner::launch.
EXECUTE_CALL = "workspaceAddr, workspaceSize, executor, aclstream"

# reduce prod is the one maintained owner that keeps a hand-rolled tail: it runs
# a two-step reduction over an intermediate tensor with its own stream
# synchronisation between the steps, which the shared single-launch tail cannot
# express. KVCacheMemcpy is excluded for a different reason -- it is a plain
# aclrtMemcpyAsync path with no aclnn workspace executor at all -- so it never
# shows up here in the first place.
HAND_ROLLED_TAIL_OWNERS = {"reduce_op_acl.cc"}


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


def test_no_family_drives_the_aclnn_execute_call_itself():
    """The tail that logs an execute failure and returns must not come back.

    This used to be pinned as "65 families each call checkRet". That number was
    a count of the copies, so the 8.06 migration invalidated it on its very
    first commit and it stayed red while roughly forty families moved -- the
    assertion could no longer distinguish a migrated tail from a deleted check.
    The invariant that survives the migration is about the execute call, not
    about how many copies of the guard exist: a family that reaches
    BaseOpRunner::launch cannot log-and-continue, because launch raises.
    """
    offenders = {
        path.name
        for path in sorted(ACLOPS.glob("*_acl.cc"))
        if path != BASE_OP and EXECUTE_CALL in path.read_text(encoding="utf-8")
    }
    assert offenders == HAND_ROLLED_TAIL_OWNERS

    # The shared tail is the only other place allowed to make the call, and it
    # raises on a non-zero status rather than logging and falling through.
    launch = _block_body(BASE_OP.read_text(encoding="utf-8"),
                         "void BaseOpRunner::launch(")
    assert EXECUTE_CALL in launch
    assert "execute launcher failed" in launch
    assert "return;" not in launch


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


def test_ascend_guide_states_the_launcher_migration_is_closed():
    """The two remaining hand-rolled tails must stay named and justified.

    Ten board waves recorded the launcher owners as "exhausted" while four
    standard owners still drove the execute call, so the exclusion list is
    pinned here rather than left to prose.
    """
    guide = GUIDE.read_text(encoding="utf-8")
    for required in (
        "Shared launcher migration is closed for the standard owners",
        "SWhere, Sigmoid backward, BatchNorm",
        "reduce prod runs a\ntwo-step reduction",
        "KVCacheMemcpy is a per-token",
        "npu-smi info",
        "CPU fallback detected: this is NOT an NPU validation",
        "must not report hardware validation",
        "acl-host-syntax-check",
        "It is not hardware validation.",
    ):
        assert required in guide
