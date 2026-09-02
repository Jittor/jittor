"""Where the compatibility layer records what it decided to continue past.

This layer bridges two frameworks whose surfaces do not line up, so it has real
reasons to keep going when something fails: an optional integration is not
installed, a Var subclass refuses an attribute, a driver entry point is missing
on this build.  What it may not do is keep going *without a trace*.

``except Exception: pass`` was the layer's most common statement -- 144 of them,
plus 269 handlers catching ``Exception`` at large.  Each one threw away the only
evidence that would later explain the symptom: a propagation marker that never
got attached, a dtype that was never restored, an optimizer grad map that was
never rebuilt.  The program continued, the numbers changed, and nothing said why.

The rule this module supports:

* **name the exceptions you expect** -- ``except (AttributeError, TypeError)``,
  never ``except Exception`` -- so a failure you did *not* anticipate still
  propagates instead of being absorbed; and
* **record what you swallowed**, so a run can be asked about it afterwards.

Writing it::

    from jittor.compat.diagnostics import swallowed

    try:
        module.__class__ = fsdp_cls
    except TypeError as exc:
        swallowed("fully_shard: retype %s" % type(module).__name__, exc)

Reading it back::

    import jittor as torch
    torch.compat_swallowed()            # every record, in order
    torch.compat_swallowed("dtype")     # only records whose label matches
    torch.compat_swallowed_counts()     # {(label, exception): times}

``JITTOR_COMPAT_DEBUG=1`` additionally prints each record to stderr with its
traceback as it happens.  That is the switch to turn on when a run finishes
"successfully" and produces the wrong thing.

The recording path is deliberately cheap: no traceback is formatted unless the
debug switch is on, because several of these handlers sit inside per-element
and per-step code.
"""

from __future__ import absolute_import

import collections
import os
import sys
import traceback

__all__ = ["swallowed", "records", "counts", "clear", "debug_enabled",
           "set_debug", "Record", "EXPECTED", "ENV_VAR", "LIMIT"]

#: What a compatibility probe is allowed to absorb when the guarded block is
#: heterogeneous enough that a tighter tuple would be a guess.
#:
#: Every entry is something this layer's probes really do provoke: an attribute
#: a Var subclass refuses (AttributeError), an argument shape the other
#: framework does not accept (TypeError, ValueError), a key or index that only
#: exists in one of the two (KeyError, IndexError), a Jittor core refusal
#: (RuntimeError -- what the C++ side raises), a file or driver that is not
#: there (OSError), an optional package (ImportError), an API this build does
#: not have (NotImplementedError).
#:
#: What it deliberately leaves out is the point of having it. ``except
#: Exception`` also absorbed KeyboardInterrupt's sibling SystemExit, MemoryError
#: and RecursionError -- so a full disk or an out-of-memory looked like "the
#: optional feature is unavailable" -- and it absorbed NameError, AssertionError
#: and StopIteration, which are this layer's own bugs and must never be
#: silenced. Those now propagate.
#:
#: One known leak: RecursionError is a subclass of RuntimeError, which has to be
#: on the list because RuntimeError is what Jittor's C++ core raises. A stack
#: overflow inside a probe is therefore still absorbed -- but it is absorbed
#: *with a record*, which is what separates it from the old behaviour.
#: tests/structure/test_compat_exception_policy.py pins this as a stated
#: property rather than a surprise.
EXPECTED = (AttributeError, TypeError, ValueError, KeyError, IndexError,
            RuntimeError, OSError, ImportError, NotImplementedError)

ENV_VAR = "JITTOR_COMPAT_DEBUG"

#: Bounded so a training loop that swallows something every step cannot grow
#: without limit. The counts below are not bounded, so nothing is lost silently.
LIMIT = 2048

_records = collections.deque(maxlen=LIMIT)
_counts = collections.Counter()
_debug = None


def debug_enabled():
    """True when swallowed failures should also be printed as they happen."""
    if _debug is not None:
        return bool(_debug)
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return False
    return str(raw).strip().lower() not in ("", "0", "false", "no", "off")


def set_debug(value):
    """Force stderr reporting on/off; ``None`` goes back to reading the env var.

    Returns the previous override so a caller can restore it.
    """
    global _debug
    previous = _debug
    _debug = None if value is None else bool(value)
    return previous


class Record(object):
    """One failure the compatibility layer chose to continue past."""

    __slots__ = ("what", "exception", "message", "hint", "stack")

    def __init__(self, what, exception, message, hint, stack):
        self.what = what
        self.exception = exception
        self.message = message
        self.hint = hint
        #: Formatted traceback, but only when JITTOR_COMPAT_DEBUG was on.
        self.stack = stack

    def __repr__(self):
        text = "%s: %s: %s" % (self.what, self.exception, self.message)
        return text + (" (%s)" % self.hint if self.hint else "")


def swallowed(what, exc, hint=None):
    """Record an exception this layer deliberately continued past.

    :param what: what was being attempted, phrased so a reader can tell what is
        now missing -- ``"propagate _torch_leaf onto the reshaped Var"``, not
        ``"setattr failed"``.
    :param exc: the exception instance.
    :param hint: optional sentence naming the consequence, or the way out.
    :returns: the :class:`Record`, so a caller can attach it to a warning.
    """
    what = str(what)
    name = type(exc).__name__
    debug = debug_enabled()
    record = Record(what, name, str(exc),
                    None if hint is None else str(hint),
                    traceback.format_exc() if debug else None)
    _records.append(record)
    _counts[(what, name)] += 1
    if debug:
        sys.stderr.write("[jittor.compat] continued past %r\n%s"
                         % (record, record.stack))
    return record


def records(match=None):
    """Swallowed failures so far, most recent last; filtered by label substring."""
    if match is None:
        return list(_records)
    needle = str(match)
    return [record for record in _records if needle in record.what]


def counts():
    """``{(label, exception name): times}`` over the whole run, never truncated."""
    return dict(_counts)


def clear():
    """Forget everything recorded so far (used by tests)."""
    _records.clear()
    _counts.clear()
