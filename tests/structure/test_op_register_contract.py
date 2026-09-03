"""No op constructor may be resolved at load time.

``get_op_info`` asserts the op is registered, and registration is itself a
static initialiser in another translation unit (``gen_ops.cc``'s
``int caller = (initer(), 0)``, ``op_utils.cc``'s ``init()``). C++ does not
order static initialisers across translation units, so a namespace-scope

    static auto make_binary = get_op_info("binary")
        .get_constructor<VarPtr, Var*, Var*, NanoString>();

is a lookup that *may* run before the thing it looks up exists. There were 113
of them. When the order does not hold the ASSERT throws out of a static
initialiser -- before ``main``, with no catch anywhere -- so the process
terminates with no message naming the op, the file, or the reason.

``op_constructor<...>("name")`` stores the name and resolves on first call, by
which time ``main`` is running. This test keeps the old spelling from coming
back, which matters because reintroducing it costs nothing today: the link
order that makes it work is stable until someone adds a file.

Function-local statics are fine and are not flagged -- they are already lazy.
The rule is therefore about *indentation*: a call at column 0 is at namespace
scope, and one inside a function is not.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (
    REPO_ROOT / "python" / "jittor" / "src",
    REPO_ROOT / "python" / "jittor" / "extern",
)

#: The registry's own implementation and its unit test name these on purpose.
ALLOWED = {
    Path("python/jittor/src/ops/op_register.cc"),
    Path("python/jittor/src/ops/op_register.h"),
    Path("python/jittor/src/test/test_op_register.cc"),
}


def _sources():
    for root in SOURCE_ROOTS:
        if not root.is_dir():
            continue
        for suffix in ("*.cc", "*.h"):
            yield from sorted(root.rglob(suffix))


def test_no_op_is_looked_up_at_namespace_scope():
    offenders = []
    for path in _sources():
        relative = path.relative_to(REPO_ROOT)
        if relative in ALLOWED:
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if "get_op_info(" not in line:
                continue
            if line[:1].isspace():
                continue          # inside a function: already lazy
            if line.lstrip().startswith(("//", "*", "/*")):
                continue
            offenders.append("%s:%d: %s" % (relative, number, line.strip()))
    assert not offenders, (
        "these resolve an op at load time, which depends on an unspecified "
        "static-initialisation order; use op_constructor<...>(\"name\") "
        "instead:\n  " + "\n  ".join(offenders))
