"""Two places in the C++ core where an exception has nowhere to go.

**Destructors.** A throw that leaves a destructor during stack unwinding is
``std::terminate`` -- no message, no traceback, and the original error, the one
that started the unwinding, is lost.  Jittor's error macros all throw:
``ASSERT``, ``ASSERTop``, ``CHECK``, ``CHECKop``, ``LOGf``, and the per-backend
wrappers built on them (``CHECK_ACL``, ``HCCLCHECK``, ``checkCudaErrors``).
Teardown code has to report and carry on instead, which is what the ``_PEEK`` /
``peek...Always`` variants next to each of those macros are for.

**Signal handlers.** A handler runs on a thread that may have been interrupted
anywhere, including inside ``malloc``.  Anything that allocates can deadlock
there, and a throw unwinds through a frame the runtime did not create -- there
is no catch on that path, so it reaches ``std::terminate`` and the crash report
that the handler existed to produce never arrives.  ``write`` and ``_exit`` are
the two things it may do.

Neither rule has a runtime test that can fail: both failures are "the process
died and told you nothing".  Whether the code satisfies them is decided by
reading it, so it is read here.

The Python-side counterpart -- the generated ``tp_dealloc``, which must free
the instance even when the destructor throws and must not disturb the
interpreter's exception state -- is in
``tests/core/test_pyjt_compiler_parser.py``.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (
    REPO_ROOT / "python" / "jittor" / "src",
    REPO_ROOT / "python" / "jittor" / "extern",
)

#: Every macro in the tree that reports by throwing.  The `_PEEK` and
#: `peek...Always` spellings deliberately do not appear: they are the answer.
THROWING = re.compile(
    r"\b(LOGf|ASSERTop|ASSERT|CHECKop|CHECK|CHECK_ACL|HCCLCHECK"
    r"|USER_ERROR|USER_CHECKop|USER_CHECK|INTERNAL_ERROR"
    r"|INTERNAL_ASSERTop|INTERNAL_ASSERT|checkCudaErrors|throw)\b")

#: `~name(args) {` -- with or without a `Class::` prefix.  A declaration
#: (`~Foo();`) has no body and is not matched.
DESTRUCTOR = re.compile(
    r"(?:(\w+)\s*::\s*)?~\s*(\w+)\s*\([^)]*\)\s*(?:noexcept\s*)?\{")


def strip_comments_and_strings(text):
    """Blank out comments, keep offsets and line numbers.

    A comment that *names* a forbidden macro to explain why it is not used --
    several of these destructors carry exactly that comment -- must not read as
    a use of it.  String literals are kept: nothing here matches inside one,
    and blanking them would need its own escape handling.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "//":
            end = text.find("\n", i)
            end = n if end < 0 else end
            out.append(" " * (end - i))
            i = end
        elif two == "/*":
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append(re.sub(r"[^\n]", " ", text[i:end]))
            i = end
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def balanced_body(text, open_at):
    depth = 0
    for j in range(open_at, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:j + 1]
    return text[open_at:]


def sources():
    for root in SOURCE_ROOTS:
        if not root.is_dir():
            continue
        for suffix in ("*.cc", "*.h"):
            yield from sorted(root.rglob(suffix))


def test_no_destructor_reports_by_throwing():
    offenders = []
    scanned = 0
    for path in sources():
        raw = path.read_text(encoding="utf8", errors="replace")
        if "~" not in raw:
            continue
        text = strip_comments_and_strings(raw)
        for match in DESTRUCTOR.finditer(text):
            scanned += 1
            body = balanced_body(text, match.end() - 1)
            used = sorted(set(THROWING.findall(body)))
            if used:
                line = raw[:match.start()].count("\n") + 1
                offenders.append("%s:%d: ~%s uses %s" % (
                    path.relative_to(REPO_ROOT), line, match.group(2),
                    ", ".join(used)))
    assert scanned > 50, (
        "the destructor scan matched %d bodies; it has stopped finding them "
        "and would pass on anything" % scanned)
    assert not offenders, (
        "a throw leaving a destructor is std::terminate; report and carry on "
        "instead (CHECK_ACL_PEEK, HCCLCHECK_PEEK, peekCudaErrorsAlways, or a "
        "plain LOGe):\n  " + "\n  ".join(offenders))


#: Anything in this list either allocates or throws, and the handler may have
#: interrupted the allocator.  `sig_write*` and `_exit` are what remains.
UNSAFE_IN_HANDLER = re.compile(
    r"\b(LOGf|LOGe|LOGw|LOGi|LOGv|ASSERT|ASSERTop|CHECK|CHECKop"
    r"|USER_ERROR|USER_CHECKop|USER_CHECK|INTERNAL_ERROR"
    r"|INTERNAL_ASSERTop|INTERNAL_ASSERT"
    r"|std::cerr|std::cout|ostringstream|printf|fprintf|malloc|free"
    r"|dladdr|backtrace_symbols|system)\s*(?:\(|<<)")

#: `exit(` runs atexit handlers and static destructors -- including the ones
#: that free device memory -- while the other threads of a process that just
#: faulted are still running.
EXIT_NOT_UNDERSCORE_EXIT = re.compile(r"(?<![_\w])exit\s*\(")


def test_the_signal_handler_only_writes_and_exits():
    path = REPO_ROOT / "python" / "jittor" / "src" / "utils" / "log.cc"
    text = strip_comments_and_strings(path.read_text(encoding="utf8"))
    marker = "void segfault_sigaction("
    assert marker in text, "the handler was renamed; this test now checks nothing"
    body = balanced_body(text, text.index("{", text.index(marker)))

    unsafe = sorted(set(m.group(1) for m in UNSAFE_IN_HANDLER.finditer(body)))
    assert not unsafe, (
        "segfault_sigaction may only do async-signal-safe work; these can "
        "allocate or throw and will deadlock a process that faulted inside "
        "malloc: " + ", ".join(unsafe))
    assert not EXIT_NOT_UNDERSCORE_EXIT.search(body), (
        "the handler must _exit(), not exit(): exit() runs atexit handlers "
        "and static destructors while the other threads are still running")
