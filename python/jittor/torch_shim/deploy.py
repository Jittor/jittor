"""Reproducible deployment of the jittor torch-shim into a python environment.

Installs the shim so that `import torch` (and torchvision/torchaudio/torchdata)
resolve to jittor, letting torch-targeted libraries (transformers, diffusers,
LlamaFactory, ...) run unmodified. Replaces the error-prone manual `cp` documented
in README.md with a single command:

    python -m jittor.torch_shim.deploy            # deploy into the active env
    python -m jittor.torch_shim.deploy --check    # verify what's deployed
    python -m jittor.torch_shim.deploy --target /path/to/site-packages

It copies:
  - torch__init__.py            -> <site-packages>/torch/__init__.py
  - stubs/<pkg>/__init__.py     -> <site-packages>/<pkg>/__init__.py   (torchvision/audio/data)
  - torch_dist_info/METADATA    -> <site-packages>/torch-<ver>.dist-info/METADATA

Idempotent and safe to re-run after editing torch__init__.py.
"""
import os
import sys
import shutil
import sysconfig

_HERE = os.path.dirname(os.path.abspath(__file__))


def _default_site_packages():
    sp = sysconfig.get_paths().get("purelib")
    if sp and os.path.isdir(sp):
        return sp
    # fall back to the first site-packages on sys.path
    for p in sys.path:
        if p.endswith("site-packages") and os.path.isdir(p):
            return p
    raise RuntimeError("could not locate site-packages; pass --target")


def _version():
    meta = os.path.join(_HERE, "torch_dist_info", "METADATA")
    try:
        for line in open(meta):
            if line.lower().startswith("version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "2.11.0"


def _plan(target):
    """Return list of (src, dst) copy operations."""
    ops = [(os.path.join(_HERE, "torch__init__.py"), os.path.join(target, "torch", "__init__.py"))]
    stubs = os.path.join(_HERE, "stubs")
    if os.path.isdir(stubs):
        for pkg in sorted(os.listdir(stubs)):
            src = os.path.join(stubs, pkg, "__init__.py")
            if os.path.isfile(src):
                ops.append((src, os.path.join(target, pkg, "__init__.py")))
    meta = os.path.join(_HERE, "torch_dist_info", "METADATA")
    if os.path.isfile(meta):
        ops.append((meta, os.path.join(target, f"torch-{_version()}.dist-info", "METADATA")))
    return ops


def deploy(target=None):
    target = target or _default_site_packages()
    done = []
    for src, dst in _plan(target):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        done.append(dst)
    return target, done


def check(target=None):
    target = target or _default_site_packages()
    missing = []
    for _src, dst in _plan(target):
        if not os.path.isfile(dst):
            missing.append(dst)
    return target, missing


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    target = None
    if "--target" in argv:
        i = argv.index("--target")
        target = argv[i + 1]
    if "--check" in argv:
        t, missing = check(target)
        if missing:
            print(f"torch-shim NOT fully deployed in {t}; missing {len(missing)}:")
            for m in missing:
                print("  -", m)
            return 1
        print(f"torch-shim deployed in {t} (all files present)")
        return 0
    t, done = deploy(target)
    print(f"torch-shim deployed into {t}:")
    for d in done:
        print("  +", os.path.relpath(d, t))
    print("verify with:  python -c 'import torch; print(torch.__file__, torch.__version__)'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
