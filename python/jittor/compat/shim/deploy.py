"""Reproducible deployment of the jittor torch-shim into a python environment.

Installs the shim so that `import torch` (and torchvision/torchaudio/torchdata)
resolve to jittor, letting torch-targeted libraries (transformers, diffusers,
LlamaFactory, ...) run unmodified. Replaces the error-prone manual `cp` documented
in README.md with a single command:

    jittor-torch-shim                              # deploy into the active env
    jittor-torch-shim --check                      # verify what's deployed
    jittor-torch-shim --target /path/to/site-packages

It copies:
  - resources/torch_init.py                 -> <site-packages>/torch/__init__.py
  - resources/stubs/<pkg>/**/*.py           -> <site-packages>/<pkg>/**/*.py
  - resources/torch_dist_info/METADATA      -> <site-packages>/torch-<ver>.dist-info/METADATA

Idempotent and safe to re-run after editing the packaged resources.
"""
import hashlib
import os
import sys
import shutil
import sysconfig

try:
    from .preflight import resources_root
except ImportError:  # filesystem-only tests load this leaf without its package
    from pathlib import Path

    def resources_root():
        return Path(__file__).resolve().parent / "resources"


def _default_site_packages():
    sp = sysconfig.get_paths().get("purelib")
    if sp and os.path.isdir(sp):
        return sp
    # fall back to the first site-packages on sys.path
    for p in sys.path:
        if p.endswith("site-packages") and os.path.isdir(p):
            return p
    raise RuntimeError("could not locate site-packages; pass --target")


def _version(resource_root=None):
    root = os.fspath(resource_root or resources_root())
    meta = os.path.join(root, "torch_dist_info", "METADATA")
    _required_source_file(meta, "torch metadata")
    with open(meta, encoding="utf-8") as meta_file:
        for line in meta_file:
            if line.lower().startswith("version:"):
                version = line.split(":", 1)[1].strip()
                if version:
                    return version
    raise RuntimeError("torch metadata has no Version field: %s" % meta)


def _normalise_target(target):
    return os.path.realpath(os.path.abspath(os.path.expanduser(os.fspath(target))))


def _safe_component(component, label):
    component = os.fspath(component)
    if (
        not isinstance(component, str)
        or not component
        or component in (os.curdir, os.pardir)
        or os.path.splitdrive(component)[0]
        or os.path.basename(component) != component
        or os.path.sep in component
        or (os.path.altsep and os.path.altsep in component)
    ):
        raise RuntimeError("unsafe %s path component: %r" % (label, component))
    return component


def _required_source_file(path, label):
    if os.path.islink(path) or not os.path.isfile(path):
        raise RuntimeError("missing or unsafe %s: %s" % (label, path))
    return path


def _destination(target, *parts):
    target = _normalise_target(target)
    safe_parts = [_safe_component(part, "deployment") for part in parts]
    destination = os.path.abspath(os.path.join(target, *safe_parts))
    try:
        inside_target = os.path.commonpath((target, destination)) == target
    except ValueError:
        inside_target = False
    if not inside_target:
        raise RuntimeError("deployment path escapes target: %s" % destination)
    return destination


def _stub_python_files(stubs):
    """Yield stub Python files and their paths relative to ``stubs``."""
    if os.path.islink(stubs) or not os.path.isdir(stubs):
        raise RuntimeError("missing or unsafe stubs directory: %s" % stubs)
    package_count = 0
    for package in sorted(os.listdir(stubs)):
        package_root = os.path.join(stubs, package)
        if os.path.islink(package_root):
            raise RuntimeError("stub source directory is a symlink: %s" % package_root)
        if not os.path.isdir(package_root):
            continue
        if package == "__pycache__":
            continue
        package_init = os.path.join(package_root, "__init__.py")
        if not os.path.isfile(package_init):
            raise RuntimeError(
                "stub package directory is missing __init__.py: %s" % package_root
            )
        _required_source_file(package_init, "stub package initializer")
        package_count += 1
        _safe_component(package, "stub")
        for root, dirs, files in os.walk(
            package_root, topdown=True, followlinks=False
        ):
            dirs.sort()
            files.sort()
            for dirname in dirs:
                path = os.path.join(root, dirname)
                if os.path.islink(path):
                    raise RuntimeError("stub source directory is a symlink: %s" % path)
            for filename in files:
                if not filename.endswith(".py"):
                    continue
                source = os.path.join(root, filename)
                if os.path.islink(source) or not os.path.isfile(source):
                    raise RuntimeError("stub source is not a regular file: %s" % source)
                relative = os.path.relpath(source, stubs)
                parts = relative.split(os.path.sep)
                for part in parts:
                    _safe_component(part, "stub")
                yield source, parts
    if package_count == 0:
        raise RuntimeError("stubs directory contains no Python packages: %s" % stubs)


def _plan(target, resource_root=None):
    """Return list of (src, dst) copy operations."""
    target = _normalise_target(target)
    root = os.fspath(resource_root or resources_root())
    torch_init = _required_source_file(
        os.path.join(root, "torch_init.py"), "torch shim"
    )
    ops = [(
        torch_init,
        _destination(target, "torch", "__init__.py"),
    )]
    stubs = os.path.join(root, "stubs")
    for source, parts in _stub_python_files(stubs):
        ops.append((source, _destination(target, *parts)))
    meta = _required_source_file(
        os.path.join(root, "torch_dist_info", "METADATA"), "torch metadata"
    )
    version_dir = "torch-%s.dist-info" % _safe_component(
        _version(root), "version"
    )
    ops.append((meta, _destination(target, version_dir, "METADATA")))

    destinations = [destination for _source, destination in ops]
    if len(destinations) != len(set(destinations)):
        raise RuntimeError("torch-shim deployment plan has duplicate destinations")
    return ops


def _unsafe_path(target, destination):
    """Return an existing path component that makes a destination unsafe."""
    target = _normalise_target(target)
    relative = os.path.relpath(destination, target)
    if relative == os.pardir or relative.startswith(os.pardir + os.path.sep):
        raise RuntimeError("deployment path escapes target: %s" % destination)

    current = target
    for component in relative.split(os.path.sep)[:-1]:
        current = os.path.join(current, component)
        if os.path.lexists(current):
            if os.path.islink(current):
                return "symlink", current
            if not os.path.isdir(current):
                return "non-directory", current
    if os.path.lexists(destination):
        if os.path.islink(destination):
            return "symlink", destination
        if not os.path.isfile(destination):
            return "non-file", destination
    return None


def _ensure_safe_parent(target, destination):
    """Reject existing path components that make a destination unsafe."""
    unsafe = _unsafe_path(target, destination)
    if unsafe is not None:
        kind, path = unsafe
        raise RuntimeError(
            "unsafe deployment path component (%s): %s" % (kind, path)
        )


def _same_file(source, destination):
    if not os.path.lexists(destination):
        return False
    try:
        return os.path.samefile(source, destination)
    except OSError:
        return False


def _ensure_distinct_files(source, destination):
    if _same_file(source, destination):
        raise RuntimeError(
            "unsafe deployment path component (same-file): %s" % destination
        )


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_details(target=None, resource_root=None):
    """Return ``(target, [(kind, path), ...])`` for deployment differences."""
    target = _normalise_target(target or _default_site_packages())
    plan = _plan(target, resource_root=resource_root)
    expected = {destination: source for source, destination in plan}
    problems = []

    for destination, source in expected.items():
        if _unsafe_path(target, destination) is not None:
            problems.append(("unsafe", destination))
        elif _same_file(source, destination):
            problems.append(("unsafe", destination))
        elif not os.path.isfile(destination):
            problems.append(("missing", destination))
        elif _sha256(source) != _sha256(destination):
            problems.append(("modified", destination))
    return target, problems


def _reported_path(requested_target, normalised_target, path):
    relative = os.path.relpath(path, normalised_target)
    return os.path.join(os.fspath(requested_target), relative)


def deploy(target=None, resource_root=None):
    requested_target = target or _default_site_packages()
    normalised_target = _normalise_target(requested_target)
    done = []
    plan = _plan(normalised_target, resource_root=resource_root)
    for src, dst in plan:
        _ensure_safe_parent(normalised_target, dst)
        _ensure_distinct_files(src, dst)
    os.makedirs(normalised_target, exist_ok=True)
    for src, dst in plan:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        done.append(_reported_path(requested_target, normalised_target, dst))
    return requested_target, done


def check(target=None, resource_root=None):
    """Return the historical ``(target, problem_paths)`` result shape."""
    requested_target = target or _default_site_packages()
    normalised_target, problems = check_details(
        requested_target, resource_root=resource_root
    )
    return requested_target, [
        _reported_path(requested_target, normalised_target, path)
        for _kind, path in problems
    ]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    target = None
    if "--target" in argv:
        i = argv.index("--target")
        if i + 1 >= len(argv) or argv[i + 1] in ("--check", "--target"):
            print("torch-shim deploy: --target requires a path")
            return 2
        target = argv[i + 1]
    if "--check" in argv:
        t, problems = check_details(target)
        if problems:
            print(f"torch-shim NOT fully deployed in {t}; found {len(problems)} problem(s):")
            for kind, path in problems:
                print("  - %s: %s" % (kind, os.path.relpath(path, t)))
            return 1
        print(f"torch-shim deployed in {t} (all files present and match source)")
        return 0
    t, done = deploy(target)
    print(f"torch-shim deployed into {t}:")
    for d in done:
        print("  +", os.path.relpath(d, t))
    print(
        "verify with:  python -c 'import torch; "
        "print(torch.__file__, torch.__torch_version__, "
        "torch.version.__version__)'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
