"""Reproducible deployment of the jittor *triton* shim into a python env.

Mirrors ``python -m jittor.torch_shim.deploy``. Installs a tiny ``triton``
package into site-packages whose body simply re-exports
``jittor.triton_shim``, so that a **bare** ``import triton`` /
``import triton.language as tl`` resolves to the shim even before
``import jittor`` runs (e.g. when a library does ``import triton`` at the very
top of its module).

    python -m jittor.triton_shim.deploy            # deploy into the active env
    python -m jittor.triton_shim.deploy --check    # verify what's deployed
    python -m jittor.triton_shim.deploy --target /path/to/site-packages
    python -m jittor.triton_shim.deploy --remove   # uninstall the shim

It writes:
  - triton/__init__.py            -> re-exports jittor.triton_shim
  - triton/language.py            -> re-exports jittor.triton_shim.language
  - triton-<ver>.dist-info/METADATA + top_level.txt
        so importlib.metadata.version("triton") resolves.

Idempotent and safe to re-run. It REFUSES to overwrite a directory that looks
like a *real* triton install (no ``__jittor_triton_shim__`` marker) unless
``--force`` is given, so it never silently clobbers a genuine triton.
"""
import os
import sys
import shutil
import sysconfig

_HERE = os.path.dirname(os.path.abspath(__file__))

_VERSION = "3.1.0"

_INIT_BODY = '''\
"""Deployed jittor triton-shim redirect: `import triton` -> jittor.triton_shim.

Written by `python -m jittor.triton_shim.deploy`. Editing the real shim only
requires editing jittor/triton_shim/ (imported live); this file is just the
redirect that makes a bare `import triton` find it.
"""
__jittor_triton_shim__ = True
from jittor.triton_shim import *          # noqa: F401,F403
from jittor.triton_shim import (          # noqa: F401
    __version__, language, install, jit, JITFunction, cdiv, next_power_of_2,
    Config, autotune, heuristics, runtime,
)
# register dotted submodules (triton.language, ...) into sys.modules
install()
'''

_LANG_BODY = '''\
"""Deployed jittor triton-shim redirect: `import triton.language` -> shim."""
__jittor_triton_shim__ = True
from jittor.triton_shim.language import *   # noqa: F401,F403
from jittor.triton_shim import language as _lang
import sys as _sys
_sys.modules[__name__] = _lang
'''

_METADATA = (
    "Metadata-Version: 2.1\n"
    "Name: triton\n"
    "Version: {0}\n"
    "Summary: jittor triton compatibility shim (not the real triton)\n"
).format(_VERSION)


def _default_site_packages():
    sp = sysconfig.get_paths().get("purelib")
    if sp and os.path.isdir(sp):
        return sp
    for p in sys.path:
        if p.endswith("site-packages") and os.path.isdir(p):
            return p
    raise RuntimeError("could not locate site-packages; pass --target")


def _is_real_triton(target):
    """True if <target>/triton looks like a genuine (non-shim) triton install."""
    init = os.path.join(target, "triton", "__init__.py")
    if not os.path.isfile(init):
        return False
    try:
        with open(init, "r") as f:
            head = f.read(4096)
    except Exception:
        return True  # be conservative
    return "__jittor_triton_shim__" not in head


def _plan(target):
    """Return list of (relpath, body) files to write under target."""
    return [
        (os.path.join("triton", "__init__.py"), _INIT_BODY),
        (os.path.join("triton", "language.py"), _LANG_BODY),
        (os.path.join("triton-{0}.dist-info".format(_VERSION), "METADATA"), _METADATA),
        (os.path.join("triton-{0}.dist-info".format(_VERSION), "top_level.txt"), "triton\n"),
    ]


def deploy(target=None, force=False):
    target = target or _default_site_packages()
    if _is_real_triton(target) and not force:
        raise RuntimeError(
            "refusing to overwrite what looks like a real triton install in "
            "{0}/triton (no __jittor_triton_shim__ marker). Pass --force to "
            "override, or `pip uninstall triton` first.".format(target)
        )
    done = []
    for rel, body in _plan(target):
        dst = os.path.join(target, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as f:
            f.write(body)
        done.append(dst)
    return target, done


def check(target=None):
    target = target or _default_site_packages()
    missing = []
    for rel, _body in _plan(target):
        if not os.path.isfile(os.path.join(target, rel)):
            missing.append(os.path.join(target, rel))
    return target, missing


def remove(target=None):
    target = target or _default_site_packages()
    removed = []
    if not _is_real_triton(target):
        d = os.path.join(target, "triton")
        if os.path.isdir(d):
            shutil.rmtree(d)
            removed.append(d)
    di = os.path.join(target, "triton-{0}.dist-info".format(_VERSION))
    if os.path.isdir(di):
        shutil.rmtree(di)
        removed.append(di)
    return target, removed


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    target = None
    force = "--force" in argv
    if "--target" in argv:
        i = argv.index("--target")
        target = argv[i + 1]
    if "--check" in argv:
        t, missing = check(target)
        if missing:
            print("triton-shim NOT fully deployed in {0}; missing {1}:".format(t, len(missing)))
            for m in missing:
                print("  -", m)
            return 1
        print("triton-shim deployed in {0} (all files present)".format(t))
        return 0
    if "--remove" in argv:
        t, removed = remove(target)
        print("triton-shim removed from {0}:".format(t))
        for r in removed:
            print("  -", os.path.relpath(r, t))
        return 0
    t, done = deploy(target, force=force)
    print("triton-shim deployed into {0}:".format(t))
    for d in done:
        print("  +", os.path.relpath(d, t))
    print("verify with:  python -c 'import triton, triton.language as tl; "
          "print(triton.__file__, triton.__version__, triton.cdiv(10,3))'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
