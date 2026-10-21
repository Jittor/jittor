"""Environment variables read on a per-operator path, re-read only on a write.

An attention call asked the environment ten times whether flash attention was
required or the inference cache enabled -- `os.environ.get` encodes the name
and looks it up each time -- which was a tenth of its host cost in an eager
decode. `getenv` answers from a cache that every `os.putenv` and `os.unsetenv`
invalidates, through an audit hook. Where audit hooks are unavailable it reads
the environment every time, as before.
"""

import os
import sys

_PROBE = "jittor.environment.probe"
_STATE = {"epoch": 0, "watching": False}
_CACHE = {}


def _audit(event, args):
    if event == "os.putenv" or event == "os.unsetenv":
        _STATE["epoch"] += 1
    elif event == _PROBE:
        _STATE["watching"] = True


def _watch():
    try:
        sys.addaudithook(_audit)
        sys.audit(_PROBE)
    except (RuntimeError, TypeError):
        # No hook, no cache: `getenv` keeps reading the environment.
        _STATE["watching"] = False


_watch()


def getenv(name, default=None):
    """`os.environ.get(name, default)`, cached until the environment is written."""
    if not _STATE["watching"]:
        return os.environ.get(name, default)
    epoch = _STATE["epoch"]
    hit = _CACHE.get(name)
    if hit is None or hit[0] != epoch:
        hit = (epoch, os.environ.get(name))
        _CACHE[name] = hit
    value = hit[1]
    return default if value is None else value
