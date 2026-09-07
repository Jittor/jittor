"""Extract the launch program of every ACL ``executeOp`` owner.

The point is to make "removing the boilerplate tail did not change what gets
launched" checkable without an Ascend card. Each ``executeOp`` body is reduced
to an ordered token stream that keeps only the parts a device would observe:

  ``query <fn>``       a ``...GetWorkspaceSize`` result assigned to ``ret``
  ``queryfail <how>``  how a failed query is handled: fatal / return / throw
  ``exec <fn>``        the aclnn execute entry point that actually runs
  ``execfail <how>``   how a failed execute is handled, ``unchecked`` included
  ``sync <0|1>``       whether the diagnostic ``syncRun()`` policy follows it
  ``hardsync``         an unconditional ``aclrtSynchronizeStream``

Everything else -- descriptor construction, RAII, workspace bookkeeping, the
error message text -- is dropped, because the shared tail owns it identically
for every caller. Two revisions whose token streams are equal launch the same
entry points in the same order under the same failure and sync policy.

``launch(ret, f, flag)`` contributes
``queryfail fatal / exec f / execfail fatal / sync flag`` because that is
literally what ``BaseOpRunner::launch`` does -- unless a caller-side handler
returns or throws first, in which case that one wins.

What this does not compare is the *arguments*: pass the wrong tensor count,
order, or type and the token stream is unchanged. That layer needs the stub-SDK
translation units (``agent/skills/acl-host-syntax-check``) and, for the workspace
query signatures, a real Ascend build. Neither layer is hardware validation.

Usage: ``python acl_launch_program.py <repo> [<repo> ...]``. With two repos it
prints a unified diff of the two programs and exits non-zero when they differ.
The method, and the reverse controls that keep it honest, are written up in
``agent/skills/static-launch-equivalence``.
"""
import difflib
import re
import sys
from pathlib import Path

ACLOPS = "backends/acl/kernels/native"

OWNER = re.compile(r"void\s+(\w+)::executeOp\b")
QUERY = re.compile(r"\b(aclnn\w*GetWorkspaceSize)\s*\(")
REGISTRY_QUERY = re.compile(r"it->second\.(getWorkspaceSizeFunc\w*)\s*\(")
LAUNCH = re.compile(r"\blaunch\(\s*ret\s*,\s*([^,]+?)\s*,\s*(true|false)\s*\)")
RAW_EXEC = re.compile(r"(\w+(?:->second\.\w+)?|it->second\.\w+)\s*\(\s*workspaceAddr\s*,"
                      r"\s*workspaceSize\s*,\s*executor\s*,\s*aclstream\s*\)")
LAUNCHER_VAR = re.compile(r"AclExecuteLauncher\s+(\w+)\s*=\s*([^;]+);")
LAUNCHER_ASSIGN = re.compile(r"^\s*(\w+)\s*=\s*(aclnn\w+)\s*;", re.M)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def bodies(text):
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


def resolve(expr, body):
    """Normalise a launcher expression into a stable ``a|b`` name."""
    expr = expr.strip()
    names = re.findall(r"aclnn\w+", expr)
    if not names:
        # A local of type AclExecuteLauncher: collect every value it can take.
        candidates = []
        for var, value in LAUNCHER_VAR.findall(body):
            if var == expr:
                candidates += re.findall(r"aclnn\w+", value)
        for var, value in LAUNCHER_ASSIGN.findall(body):
            if var == expr:
                candidates.append(value)
        names = candidates or [expr]
    return "|".join(sorted(set(names)))


def failure_policy(body, position):
    """Classify the handler that follows ``position`` in the body."""
    window = body[position:position + 420]
    if re.search(r"throw\s", window):
        return "throw"
    if "CHECK_RET(ret == ACL_SUCCESS" in window and "return" in window:
        return "return"
    if "checkRet(ret)" in window:
        return "fatal"
    return "unchecked"


def program(body):
    """Ordered token stream for one ``executeOp`` body."""
    events = []
    for match in QUERY.finditer(body):
        events.append((match.start(), "query", match.group(1)))
    for match in REGISTRY_QUERY.finditer(body):
        events.append((match.start(), "query", "registry." + match.group(1)))
    for match in LAUNCH.finditer(body):
        events.append((match.start(), "launch",
                       (resolve(match.group(1), body), match.group(2))))
    for match in RAW_EXEC.finditer(body):
        events.append((match.start(), "exec", resolve(match.group(1), body)))
    for match in re.finditer(r"\bsyncRun\(\);", body):
        events.append((match.start(), "syncrun", None))
    for match in re.finditer(r"\baclrtSynchronizeStream\(", body):
        events.append((match.start(), "hardsync", None))

    events.sort()
    events = merge_alternatives(body, events)
    # A trailing syncRun() applies to every execute before it, which is how the
    # hand-rolled switch bodies expressed "synchronise after whichever case ran",
    # and how the AdamW loop expresses "synchronise once after the last step".
    trailing = [pos for pos, kind, _ in events if kind == "syncrun"]

    tokens = []
    pending = []
    for position, kind, payload in events:
        if kind == "query":
            pending.append((payload, failure_policy(body, position)))
            continue
        if kind == "hardsync":
            tokens.append("hardsync")
            continue
        if kind == "syncrun":
            continue
        synced = any(t > position for t in trailing)
        if kind == "launch":
            name, flag = payload
            exec_policy = "fatal"
            query_policy = "fatal"
            synced = synced or flag == "true"
        else:
            name = payload
            exec_policy = failure_policy(body, position)
            query_policy = None
        tokens += dispatch(pending, name, query_policy, exec_policy, synced)
        pending = []
    for name, policy in pending:
        # A query with no execute of its own (an early-out branch).
        tokens += ["query " + name, "queryfail " + policy]
    return tokens


def dispatch(queries, name, query_policy, exec_policy, synced):
    """One operator's launch record: what is queried, run, and synchronised.

    ``queries`` are the alternative workspace queries feeding a single execute
    (the ``if is_max ... else ...`` shape), so they collapse into one
    alternation. Their failure policies collapse too: ``unchecked`` is dropped
    whenever a sibling branch does carry a handler, because the handler that
    follows an if/else chain covers every branch.
    """
    names = "|".join(sorted({q for q, _ in queries})) or "-"
    policies = {p for _, p in queries}
    if query_policy is not None:
        # ``launch`` handles a failed query, but only if control reaches it: a
        # caller-side handler that returns or throws fires first and wins.
        # Without this the tool reports "no change" for exactly the edit that
        # turns "leave the output uninitialised and carry on" into "raise".
        short_circuit = policies & {"return", "throw"}
        policies = short_circuit or {query_policy}
    elif len(policies) > 1:
        policies.discard("unchecked")
    return ["query " + names,
            "queryfail " + "|".join(sorted(policies or {"-"})),
            "exec " + name,
            "execfail " + exec_policy,
            "sync " + ("1" if synced else "0")]


def merge_alternatives(body, events):
    """Fold ``ret = c ? aclnnA(...) : aclnnB(...)`` into one exec event.

    The registry form of the same choice is a single ``launch`` with an
    ``AclExecuteLauncher`` local, so without folding, the two spellings of one
    dispatch would not compare equal.
    """
    merged = []
    for event in events:
        position, kind, payload = event
        if (kind == "exec" and merged and merged[-1][1] == "exec"
                and ";" not in body[merged[-1][0]:position]):
            names = sorted(set(merged[-1][2].split("|") + payload.split("|")))
            merged[-1] = (merged[-1][0], "exec", "|".join(names))
            continue
        merged.append(event)
    return merged


def dump(root):
    lines = []
    for path in sorted((Path(root) / ACLOPS).glob("*_acl.cc")):
        text = strip_comments(path.read_text())
        for owner, body in bodies(text):
            lines.append(f"{path.name} {owner}")
            lines += ["    " + token for token in program(body)]
    return lines


def main():
    roots = sys.argv[1:] or ["."]
    if len(roots) == 1:
        print("\n".join(dump(roots[0])))
        return 0
    before, after = dump(roots[0]), dump(roots[1])
    delta = list(difflib.unified_diff(before, after, "before", "after", lineterm="", n=3))
    print("\n".join(delta) if delta else
          f"identical launch program: {len(before)} lines")
    return 1 if delta else 0


if __name__ == "__main__":
    sys.exit(main())
