"""Find hand-rolled launch tails in the ACL ``executeOp`` bodies.

``BaseOpRunner::launch`` owns five things: handling a failed
``...GetWorkspaceSize`` result, allocating the workspace, issuing the
``(workspaceAddr, workspaceSize, executor, aclstream)`` execute call, handling a
failed execute result, and the ``syncRun()`` diagnostic policy. Any of the first
four written inside an ``executeOp`` body is leftover boilerplate, and a new
operator that writes one is the regression this backs.

The check is an invariant, not a count. The previous shape of this contract
asserted "65 sites call ``checkRet``", which a legal refactor invalidated; it
then stayed red for about 40 commits without anyone noticing.

``syncRun()`` is reported but not treated as boilerplate: two owners
legitimately place one themselves (the AdamW loop synchronises once after its
last step, the staged product path after freeing its intermediates).

There is no Ascend card on the development host, so this is a source-level
contract. What it cannot see is registered in
``agent/manuals/deferred-hardware.md``.
"""

import re
from pathlib import Path

# Native ACL operators have one physical owner; HCCL moves separately.
ACL_ROOTS = ("backends/acl",)
ACLOPS = "backends/acl/kernels/native"
# The one file allowed to own the tail.
SHARED_TAIL = "base_op_acl.cc"

OWNER = re.compile(r"void\s+(\w+)::executeOp\b")
QUERY = re.compile(r"\b(?:aclnn\w*GetWorkspaceSize|it->second\.getWorkspaceSizeFunc\w*)\s*\(")
WORKSPACE_MALLOC = re.compile(r"\bmallocWorkSpace\s*\(")
DIRECT_EXECUTE = re.compile(r"(\w[\w.>-]*)\s*\(\s*workspaceAddr\s*,\s*workspaceSize\s*,"
                            r"\s*executor\s*,\s*aclstream\s*\)")
# Every spelling of a caller-side handler the tree used before the shared tail.
HANDLER = re.compile(r"checkRet\s*\(\s*ret\s*\)|"
                     r"CHECK_RET\s*\(\s*ret\s*==\s*ACL_SUCCESS|"
                     r"if\s*\(\s*ret\s*!=\s*ACL_SUCCESS\s*\)")
SYNC_RUN = re.compile(r"\bsyncRun\s*\(\s*\)\s*;")
# Handlers guarding something other than a workspace query keep their place;
# they are recognised by the call whose result they check.
NON_QUERY_SUBJECT = re.compile(r"\b(aclrtMemcpyAsync|aclrtMemsetAsync|aclrtMalloc|"
                               r"CreateAclTensor|aclrtSynchronizeStream)\s*\(")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def execute_op_bodies(text):
    """Yield ``(owner, body)`` for every ``X::executeOp`` definition."""
    for match in OWNER.finditer(text):
        start = text.index("{", match.end())
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    yield match.group(1), text[start:idx + 1]
                    break


def query_handlers(body):
    """Handlers sitting between a workspace query and its execute call.

    A handler is attributed to the query when the nearest preceding call whose
    result went into ``ret`` is a workspace query -- that is how the hand-rolled
    tail spelled "the query failed, give up", and it is what made a failed query
    produce an uninitialised output instead of an error.
    """
    found = []
    for match in HANDLER.finditer(body):
        preceding = body[:match.start()]
        subjects = [(m.start(), "query") for m in QUERY.finditer(preceding)]
        subjects += [(m.start(), "other") for m in NON_QUERY_SUBJECT.finditer(preceding)]
        if subjects and max(subjects)[1] == "query":
            found.append(match.group(0).split("(")[0].strip())
    return found


def survey(repo_root):
    """Return ``(owners, tails)`` for the ACL operator sources.

    ``owners`` is every ``executeOp`` found, ``tails`` maps
    ``"<file>:<owner>"`` to the boilerplate found in it.
    """
    owners, tails = [], {}
    for path in sorted((Path(repo_root) / ACLOPS).glob("*.cc")):
        if path.name == SHARED_TAIL:
            continue
        text = strip_comments(path.read_text())
        for owner, body in execute_op_bodies(text):
            key = f"{path.name}:{owner}"
            owners.append(key)
            findings = {}
            if WORKSPACE_MALLOC.search(body):
                findings["workspace_malloc"] = len(WORKSPACE_MALLOC.findall(body))
            direct = DIRECT_EXECUTE.findall(body)
            if direct:
                findings["direct_execute"] = sorted(set(direct))
            handlers = query_handlers(body)
            if handlers:
                findings["query_handler"] = handlers
            if findings:
                tails[key] = findings
    return owners, tails


def caller_side_syncs(repo_root):
    """``{"<file>:<owner>": count}`` for owners that place their own syncRun."""
    result = {}
    for path in sorted((Path(repo_root) / ACLOPS).glob("*.cc")):
        if path.name == SHARED_TAIL:
            continue
        text = strip_comments(path.read_text())
        for owner, body in execute_op_bodies(text):
            count = len(SYNC_RUN.findall(body))
            if count:
                result[f"{path.name}:{owner}"] = count
    return result


def populated_roots(repo_root):
    """``{root: file count}`` so an empty scan cannot look like a clean one."""
    return {root: sum(1 for path in (Path(repo_root) / root).rglob("*")
                      if path.is_file())
            for root in ACL_ROOTS}


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1] if len(sys.argv) > 1 else
                Path(__file__).resolve().parents[2])
    all_owners, found = survey(root)
    for key, findings in sorted(found.items()):
        print(f"{key} {findings}")
    for key, count in sorted(caller_side_syncs(root).items()):
        print(f"note {key} keeps {count} caller-side syncRun()")
    print(f"\n{len(all_owners) - len(found)}/{len(all_owners)} executeOp owners "
          f"carry no hand-rolled launch tail")
    print("acl roots:", populated_roots(root))
    sys.exit(1 if found else 0)
