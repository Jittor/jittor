"""ACL runner failures must be attributed and stop before execution continues."""

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[4]
ACL_ROOT = REPO_ROOT / "backends" / "acl"
BASE_OP = ACL_ROOT / "kernels" / "native" / "base_op_acl.cc"
ACLOPS = ACL_ROOT / "kernels" / "native"
EXEC = ACL_ROOT / "src" / "acl_op_exec.cc"
GUIDE = REPO_ROOT / "docs" / "guides" / "ascend-910b.md"

# The aclnn execute call takes exactly these four arguments, so its presence is
# a reliable marker of a runner that drives the launch itself instead of going
# through BaseOpRunner::launch.
EXECUTE_CALL = "workspaceAddr, workspaceSize, executor, aclstream"

# Nothing is exempt any more. reduce prod was the last holdout: its staged
# multi-axis path was thought to need its own tail for the synchronisation
# between steps, but those steps only need to be asynchronous, which
# launch(ret, f, false) expresses, and the barrier before the intermediates are
# freed is a separate unconditional aclrtSynchronizeStream that stays at the
# call site. KVCacheMemcpy never appeared here -- it is a plain
# aclrtMemcpyAsync path with no aclnn workspace executor at all.
HAND_ROLLED_TAIL_OWNERS = set()


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
    assert "acl_op_registry().find(name)" in run
    assert "it == acl_op_registry().end()" in run
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
    # Allocation checks, contiguous handling and liveness release may each
    # inspect inputs. What matters is their owner, not a frozen access count.
    owners = re.findall(r"\b(\w+)->(?:inputs|outputs)\(", loop)
    assert owners and set(owners) == {"current_op"}
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
        "forbid_backend_fallbacks()",
        "backend_fallback_count()",
        "backend_fallback=error",
    ):
        assert required in guide


def test_ascend_guide_states_the_launcher_migration_is_closed():
    """The claim of zero remaining tails must be written down, with the caveats.

    Ten board waves recorded the launcher owners as "exhausted" while four
    standard owners still drove the execute call, so what is and is not covered
    is pinned here rather than left to prose.
    """
    guide = GUIDE.read_text(encoding="utf-8")
    for required in (
        "Shared launcher migration is closed for the standard owners",
        "SWhere, Sigmoid backward, BatchNorm",
        "all 71 `executeOp` owners are tail-free",
        "AdamW loop synchronises once after",
        "KVCacheMemcpy never had a tail",
        "npu-smi info",
        "Fallback attempts are NOT NPU validation",
        "must not report hardware validation",
        "acl-host-syntax-check",
        "It is not hardware validation.",
    ):
        assert required in guide


def test_direct_owners_and_missing_registry_dispatch_compile_and_execute(tmp_path):
    """Execute the production base header/run and direct-owner constructors."""
    import os
    import shlex
    import subprocess

    header = (ACL_ROOT / 'include/aclops/base_op.h').read_text()
    header = re.sub(r'^#(?:include|pragma).*$', '', header, flags=re.M)
    base_source = BASE_OP.read_text()
    run = 'void BaseOpRunner::run() {' + _block_body(
        base_source, 'void BaseOpRunner::run()') + '}'
    constructors = []
    for filename, marker in (
        ('truth_reduce_op_acl.cc', 'TruthReduceOpRunner::TruthReduceOpRunner('),
        ('flashattention_op_acl.cc', 'KVCacheMemcpyOpRunner::KVCacheMemcpyOpRunner('),
        ('flashattention_op_acl.cc', 'IncreFlashAttentionOpRunner::IncreFlashAttentionOpRunner('),
    ):
        source = (ACLOPS / filename).read_text()
        start = source.index(marker)
        opening = source.index('{', start)
        constructors.append(source[start:opening + 1] + _block_body(source, marker) + '}')
    preamble = r'''
#include <cassert>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using std::string;
using std::vector;
using aclnnStatus = int;
using aclrtStream = void*;
struct aclOpExecutor {};
struct aclTensor {};
struct Var {};
struct AclOpAttr { virtual ~AclOpAttr() = default; };
struct ReduceAttr : AclOpAttr {};
using AclOpRegistry = std::map<string, int>;
AclOpRegistry registry;
const AclOpRegistry& acl_op_registry() { return registry; }
struct Fatal {
    std::ostringstream text;
    template<class T> Fatal& operator<<(const T& value) { text << value; return *this; }
    ~Fatal() noexcept(false) { throw std::runtime_error(text.str()); }
};
#define LOGf Fatal()
vector<string> events;
'''
    owners = r'''
namespace jittor {
void BaseOpRunner::setupInputDesc() { events.push_back("input"); }
void BaseOpRunner::setupOutputDesc() { events.push_back("output"); }
void BaseOpRunner::cleanupDesc() { events.push_back("cleanup"); }
void BaseOpRunner::syncRun() {}
struct TruthReduceOpRunner : BaseOpRunner {
    bool reduce_all;
    ReduceAttr* attr;
    explicit TruthReduceOpRunner(bool);
    void executeOp(AclOpRegistry::const_iterator& it) override {
        assert(it == acl_op_registry().end()); events.push_back(name);
    }
};
struct KVCacheMemcpyOpRunner : BaseOpRunner {
    KVCacheMemcpyOpRunner();
    void executeOp(AclOpRegistry::const_iterator& it) override {
        assert(it == acl_op_registry().end()); events.push_back(name);
    }
};
struct IncreFlashAttentionOpRunner : BaseOpRunner {
    IncreFlashAttentionOpRunner();
    void executeOp(AclOpRegistry::const_iterator& it) override {
        assert(it == acl_op_registry().end()); events.push_back(name);
    }
};
struct GenericRunner : BaseOpRunner {
    GenericRunner(string name, bool grouped) : BaseOpRunner(name) { is_group_op = grouped; }
    void executeOp(AclOpRegistry::const_iterator& it) override {
        assert(it != acl_op_registry().end()); events.push_back(name);
    }
};
'''
    checks = r'''
}
int main() {
    using namespace jittor;
    for (bool all : {false, true}) {
        events.clear(); TruthReduceOpRunner runner(all); runner.run();
        assert(events == vector<string>({"input", "output", all ? "All" : "Any", "cleanup"}));
    }
    events.clear(); KVCacheMemcpyOpRunner copy; copy.run();
    assert(events == vector<string>({"input", "output", "KVCacheMemcpy", "cleanup"}));
    events.clear(); IncreFlashAttentionOpRunner attention; attention.run();
    assert(events == vector<string>({"input", "output", "IncreFlashAttention", "cleanup"}));
    for (bool grouped : {false, true}) {
        events.clear(); GenericRunner missing("Missing", grouped);
        bool caught = false;
        try { missing.run(); }
        catch (const std::runtime_error& error) {
            caught = string(error.what()).find("no registered launcher:Missing") != string::npos;
        }
        assert(caught && events.empty());
        registry["Present"] = 1;
        GenericRunner present("Present", grouped); present.run();
        assert(events == vector<string>({"input", "output", "Present", "cleanup"}));
        registry.clear();
    }
}
'''
    unit = tmp_path / 'runner_dispatch.cc'
    unit.write_text(preamble + header + owners + run + '\n'.join(constructors) + checks)
    executable = tmp_path / 'runner_dispatch'
    result = subprocess.run(
        [*shlex.split(os.environ.get('CXX', 'g++')), '-std=c++14',
         str(unit), '-o', str(executable)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_all_runner_constructors_have_a_valid_registry_or_direct_owner():
    """Audit all production constructors, including names absent from the registry.

    Unary/binary resolve their name dynamically and dereference the iterator;
    direct CANN owners must never depend on a missing registry entry.
    """
    import sys
    tests_root = REPO_ROOT / 'tests'
    if str(tests_root) not in sys.path:
        sys.path.insert(0, str(tests_root))
    from _helpers.acl_launch_tails import execute_op_bodies, strip_comments
    registered = set(re.findall(r'\{"([^"]+)"',
                                (ACL_ROOT / 'src/acl_jittor.cc').read_text()))
    checked, direct, generic = set(), set(), set()
    for path in ACLOPS.glob('*.cc'):
        source = strip_comments(path.read_text())
        bodies = dict(execute_op_bodies(source))
        for constructor in re.finditer(
                r'(\w+)::\1\([^)]*\)\s*:\s*BaseOpRunner\(([^)]*)\)', source):
            owner, arguments = constructor.groups()
            assert owner in bodies, owner
            checked.add(owner)
            uses_registry = bool(re.search(r'\bit\s*->', bodies[owner]))
            if 'Dispatch::Direct' in arguments:
                direct.add(owner)
                assert not uses_registry, owner
                continue
            if uses_registry:
                generic.add(owner)
                continue
            names = re.findall(r'"([^"]+)"', arguments)
            assert set(names) <= registered, (owner, set(names) - registered)
    assert checked and direct
    assert generic == {'UnaryOpRunner', 'BinaryOpRunner'}
