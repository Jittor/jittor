"""Drive every public native Jittor callable and record what actually happens.

A name that imports but cannot be called, or that raises something other than a
declared argument error, is the failure class this exists to find: today's
regressions were all "a cleanup removed something load-bearing through
indirection", and that shape stays invisible until the entry point is invoked.
"""
import argparse, inspect, json, sys, traceback, types
import numpy as np
import jittor as jt

SKIP_NAME = (
    "exit","fork","clean","gc","display","profil","download","save","load",
    "dump","print","sync_all","seed_all","install","compile","import","reload",
    "server","daemon","sleep","wait","lock","mkdir","remove","rmtree","system",
    "popen","spawn","kill","signal","exec","eval_","input","help","breakpoint",
    "set_lock","cleanup","flush","abort","terminate","shutdown","fetch","request",
)

def _first_line(exc):
    """A message-less exception is itself a finding, not a reason to crash.

    A bare `assert` produces an empty str(), so `str(exc).splitlines()[0]`
    raises IndexError and the sweep dies on the very entry point it just
    caught. Returning the marker instead turns that case into a recorded row --
    which is how the 11 message-less public entries were found.
    """
    text = str(exc)
    return text.splitlines()[0][:160] if text.strip() else "(no message)"


def skip(name):
    low = name.lower()
    return any(t in low for t in SKIP_NAME)

def candidates():
    """Argument tuples tried in order; first one that runs wins."""
    a = jt.array(np.array([[0.5, 1.5], [2.5, 3.5]], dtype="float32"))
    b = jt.array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"))
    p = jt.array(np.array([[0.2, 0.8], [0.6, 0.4]], dtype="float32"))
    i = jt.array(np.array([0, 1], dtype="int32"))
    v = jt.array(np.array([1.0, 2.0, 3.0], dtype="float32"))
    v4 = jt.array(np.random.RandomState(0).rand(2, 3, 4, 4).astype("float32"))
    return [
        (a,), (a, b), (v,), (p,), (v4,),
        (a, 0), (a, 1), (v, 0),
        (a, b, 0), ((2, 2),), (a, i), (),
    ]

def drive(obj, name):
    for args in candidates():
        try:
            out = obj(*args)
        except TypeError as exc:
            msg = str(exc)
            if "argument" in msg or "positional" in msg or "takes" in msg \
               or "required" in msg or "missing" in msg or "expected" in msg:
                continue                       # wrong shape of call, try next
            return "raised", "TypeError: " + msg[:160]
        except Exception as exc:               # noqa: BLE001 -- classification
            return "raised", type(exc).__name__ + ": " + _first_line(exc)
        try:
            if isinstance(out, jt.Var):
                out.sync()
            elif isinstance(out, (list, tuple)):
                for o in out:
                    if isinstance(o, jt.Var):
                        o.sync()
        except Exception as exc:               # noqa: BLE001
            return "exec-failed", type(exc).__name__ + ": " + _first_line(exc)
        return "ok", "args=%d" % len(args)
    return "no-signature-match", ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    opt = ap.parse_args()
    if opt.device == "cuda":
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0

    mods = {"jt": jt}
    for nm in ("nn","ops","linalg","fft","init","distributions","optim",
               "sparse","autograd","misc","pool","contrib","attention"):
        m = getattr(jt, nm, None)
        if isinstance(m, types.ModuleType):
            mods["jt." + nm] = m

    rows = []
    for mod_name, mod in sorted(mods.items()):
        for name in sorted(n for n in dir(mod) if not n.startswith("_")):
            if skip(name):
                continue
            obj = getattr(mod, name, None)
            if not callable(obj) or isinstance(obj, type):
                continue
            if getattr(obj, "__module__", "") and "jittor" not in str(getattr(obj, "__module__", "")):
                continue                       # re-exported third-party helper
            try:
                status, detail = drive(obj, name)
            except BaseException as exc:       # noqa: BLE001 -- must not abort sweep
                status, detail = "harness-error", type(exc).__name__ + ": " + _first_line(exc)
            rows.append({"module": mod_name, "name": name,
                         "status": status, "detail": detail})
            if len(rows) % 25 == 0:
                with open(opt.out + ".partial", "w") as fh:
                    json.dump(rows, fh)
    with open(opt.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    from collections import Counter
    print(opt.device, dict(Counter(r["status"] for r in rows)))

main()
