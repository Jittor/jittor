# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""One pass over everything a Jittor build needs, reported all at once.

`import jittor` is a build system, and its preconditions used to be checked by
the order the module-level code happened to run in: find a compiler, stop; find
Python headers, stop; find OpenMP, stop. A user missing three of them learns
about them one `pip install` at a time, and each round trip costs another cold
build. Worse, several of the conditions were not checked at all and surfaced
much later as something else -- a full disk as scattered compile failures, an
unreachable mirror as a truncated archive.

So: check everything, report everything, and say for each item whether Jittor
can fix it itself or the user has to.

    python -m jittor_utils.preflight          # human-readable report
    python -m jittor_utils.preflight --json   # same thing for scripts

Nothing here imports jittor or compiles anything; it is safe to run on a
machine where `import jittor` does not work, which is the case it exists for.
"""

from __future__ import print_function

import collections
import ctypes.util
import json
import os
import platform
import shutil
import sys

#: One finding. ``status`` is "ok", "warn" or "fail"; only "fail" stops a
#: build. ``fixable`` says whether Jittor can resolve it without the user
#: doing anything -- that is the difference between "we will download this for
#: you" and "install a compiler".
Result = collections.namedtuple("Result", "name status detail remedy fixable")


def _ok(name, detail):
    return Result(name, "ok", detail, "", False)


def _fail(name, detail, remedy, fixable=False):
    return Result(name, "fail", detail, remedy, fixable)


def _warn(name, detail, remedy, fixable=False):
    return Result(name, "warn", detail, remedy, fixable)


def check_compiler(cc_path=None):
    """A C++ compiler Jittor can drive."""
    if cc_path is None:
        cc_path = os.environ.get("cc_path") or ""
    if not cc_path:
        for candidate in ("g++", "clang++", "clang", "icc"):
            found = shutil.which(candidate)
            if found:
                cc_path = found
                break
    if not cc_path:
        return _fail(
            "c++ compiler", "no g++, clang++ or icc on PATH",
            "install a C++ compiler (Debian/Ubuntu: `apt install g++`; "
            "RHEL/Fedora: `dnf install gcc-c++`; macOS: `xcode-select "
            "--install`), or set cc_path to one that is installed")
    if not os.path.isfile(cc_path) and not shutil.which(cc_path):
        return _fail(
            "c++ compiler", "cc_path=%s does not exist" % cc_path,
            "point cc_path at a compiler that exists, or unset it to let "
            "Jittor search PATH")
    return _ok("c++ compiler", cc_path)


def check_python_headers():
    """``Python.h``, which every compiled Jittor object includes."""
    include = None
    try:
        import sysconfig
        include = sysconfig.get_paths().get("include")
    except Exception:
        pass
    candidates = [path for path in (include,) if path]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "Python.h")):
            return _ok("python headers", os.path.join(path, "Python.h"))
    version = "%d.%d" % (sys.version_info[0], sys.version_info[1])
    return _fail(
        "python headers",
        "Python.h not found in %s" % (candidates or "any known include dir"),
        "install the development headers for this interpreter "
        "(Debian/Ubuntu: `apt install python%s-dev`; conda: the headers ship "
        "with the interpreter, so a missing one means a broken environment). "
        "This must be the headers for %s, not for the system python."
        % (version, sys.executable))


def check_openmp(cc_type=None):
    """The OpenMP runtime every CPU kernel links against."""
    if platform.system() != "Linux":
        return _ok("openmp", "not required on " + platform.system())
    if cc_type is None:
        cc_path = os.environ.get("cc_path") or shutil.which("g++") or ""
        cc_type = "clang" if "clang" in os.path.basename(cc_path) else "g++"
    library = {"clang": "omp", "icc": "iomp5", "g++": "gomp"}.get(cc_type, "gomp")
    found = ctypes.util.find_library(library)
    if not found:
        return _fail(
            "openmp runtime", "lib%s not found" % library,
            "install it (Debian/Ubuntu: `apt install libgomp1` for g++, "
            "`apt install libomp-dev` for clang), or set cc_path to a "
            "compiler whose runtime is installed")
    return _ok("openmp runtime", found)


def check_disk_space(path=None, minimum_mb=None):
    """Room for a build. A full disk does not announce itself.

    It surfaces as a truncated object file that the cache records as complete,
    and then as scattered compile failures and segfaults in operators that have
    nothing to do with anything the user changed.
    """
    if path is None:
        from jittor_utils import cache_root
        path = cache_root()
    if minimum_mb is None:
        try:
            minimum_mb = float(os.environ.get("JT_MIN_FREE_SPACE_MB", 512))
        except ValueError:
            minimum_mb = 512.0
    probe = path
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        stat = os.statvfs(probe)
    except (OSError, AttributeError, ValueError):
        return _warn("disk space", "cannot be measured on this filesystem", "")
    free_mb = stat.f_bavail * stat.f_frsize / float(1 << 20)
    detail = "%.0f MB free at %s" % (free_mb, probe)
    remedy = ("free some space, point JITTOR_HOME at a larger filesystem, or "
              "run `python -m jittor_utils.clean_cache core`")
    if free_mb < minimum_mb:
        return _fail("disk space", detail, remedy, fixable=True)
    if free_mb < 2048:
        return _warn("disk space", detail + "; a cold build writes on the "
                     "order of 1 GB", remedy, fixable=True)
    return _ok("disk space", detail)


def _asset_present(asset, cache, mirror):
    from jittor_utils import misc
    _, digest = _digest_of(asset)
    for directory in (cache, mirror):
        if not directory:
            continue
        # The cache keeps each archive in a directory named after the asset
        # (`<cache>/mkl/dnnl_....tgz`); an offline mirror is flat.
        for path in (os.path.join(directory, asset.key, asset.filename),
                     os.path.join(directory, asset.filename)):
            if digest and misc.check_file_exist(path, digest):
                return True
            if not digest and os.path.isfile(path):
                return True
    return False


def _digest_of(asset):
    from jittor_utils import manifest
    return manifest.digest_of(asset)


def check_third_party(cache=None, mirror=None):
    """Which downloads a cold build still needs, if any."""
    from jittor_utils import manifest
    if cache is None:
        from jittor_utils import cache_root
        cache = cache_root()
    if mirror is None:
        mirror = os.environ.get("JITTOR_OFFLINE_PATH") or None
    system = platform.system()
    machine = platform.machine()
    wanted = []
    try:
        wanted.append(manifest.mkl_asset(system, machine))
    except Exception:
        pass
    for asset in (manifest.CUB, manifest.CUTT):
        wanted.append(asset)
    missing = [asset.filename for asset in wanted
               if not _asset_present(asset, cache, mirror)]
    if not missing:
        return _ok("third-party archives", "all present, no network needed")
    return _warn(
        "third-party archives", "%d still to download: %s"
        % (len(missing), ", ".join(missing)),
        "they are fetched automatically; to install without a network, run "
        "`nox -s prefetch` on a connected machine and point "
        "JITTOR_OFFLINE_PATH at the directory it fills",
        fixable=True)


def check_network(needed=True, host=None, timeout=5.0):
    """Whether the archive host answers. Skipped when nothing is missing."""
    if not needed:
        return _ok("network", "not needed, every archive is already on disk")
    if host is None:
        from jittor_utils import manifest
        try:
            from urllib.parse import urlparse
        except ImportError:
            from urlparse import urlparse
        host = urlparse(manifest.ASSET_BASE).hostname
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.create_connection((host, 443), timeout=timeout).close()
    except Exception as error:
        return _fail(
            "network", "cannot reach %s: %s" % (host, error),
            "Jittor needs it once, to fetch MKL and cub. On a machine with no "
            "network, run `nox -s prefetch` elsewhere and point "
            "JITTOR_OFFLINE_PATH at the directory it fills",
            fixable=True)
    return _ok("network", "%s reachable" % host)


def check_cuda(nvcc_path=None):
    """nvcc and, if the pip CUDA stack is installed, whether it is coherent."""
    if nvcc_path is None:
        nvcc_path = os.environ.get("nvcc_path")
    if nvcc_path == "":
        return _ok("cuda", "nvcc_path is empty, this is a CPU-only build")
    resolved = nvcc_path or shutil.which("nvcc")
    has_driver = os.path.exists("/proc/driver/nvidia/version")
    if not resolved:
        if not has_driver:
            return _ok("cuda", "no GPU driver and no nvcc, CPU-only build")
        return _warn(
            "cuda", "this machine has an NVIDIA driver but no nvcc on PATH",
            "install a CUDA toolkit and put nvcc on PATH, or set "
            "nvcc_path=\"\" to make the CPU-only build explicit. Jittor no "
            "longer downloads a toolkit by itself",
            fixable=False)
    results = [_ok("cuda", resolved)]
    try:
        from jittor_utils import cuda_wheel
        from jittor_utils import get_version
        version = get_version(resolved)[1:-1]
        report = cuda_wheel.inspect_cuda_wheel_stack(version)
        if report.stack is None and report.present:
            status = _fail if report.broken else _warn
            results.append(status(
                "cuda pip wheels", report.reason,
                "the build falls back to the CUDA on this system; if that one "
                "ships cuDNN 9, the failure you would get later comes from "
                "here. `pip install jittor[cuda12]` reinstalls the pinned set",
                fixable=True))
    except Exception as error:
        results.append(_warn("cuda pip wheels",
                             "could not be inspected: %s" % error, ""))
    return results


def check_git_affects_cache():
    """Whether the cache directory depends on `git`, which is invisible today."""
    from jittor_utils import _git_head_file
    import jittor_utils
    source = os.path.dirname(jittor_utils.__file__)
    if _git_head_file(source) is None:
        return _ok("git", "this install is not in a git checkout")
    if not shutil.which("git"):
        return _warn(
            "git", "this install is inside a git checkout but git is not on "
            "PATH, and the cache directory is named after the branch",
            "install git, or set cache_name to a fixed value so the cache "
            "directory does not depend on it")
    return _ok("git", "checkout detected; the cache directory is named after "
               "the branch (set cache_name to override)")


def run_all():
    """Every check, in the order a build would hit them."""
    results = []
    results.append(check_compiler())
    results.append(check_python_headers())
    results.append(check_openmp())
    results.append(check_disk_space())
    third_party = check_third_party()
    results.append(third_party)
    results.append(check_network(needed=third_party.status != "ok"))
    cuda = check_cuda()
    results.extend(cuda if isinstance(cuda, list) else [cuda])
    results.append(check_git_affects_cache())
    return results


def failures(results):
    return [result for result in results if result.status == "fail"]


def format_report(results, only_problems=False):
    """One message describing everything that is wrong, not just the first."""
    lines = []
    marks = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}
    for result in results:
        if only_problems and result.status == "ok":
            continue
        lines.append("%s %-20s %s" % (marks[result.status], result.name,
                                      result.detail))
        if result.remedy:
            who = "Jittor can do this for you" if result.fixable \
                else "needs you"
            lines.append("       -> %s (%s)" % (result.remedy, who))
    if not lines:
        return "all build preconditions satisfied"
    return "\n".join(lines)


def assert_ready():
    """Raise once, naming every unmet precondition.

    The point is the *once*. Checking these in the order the module-level code
    happens to run means a user missing three of them pays three cold builds to
    find out.
    """
    results = run_all()
    bad = failures(results)
    if bad:
        raise RuntimeError(
            "Jittor cannot build here. %d precondition(s) are not met:\n%s\n"
            "Run `%s -m jittor_utils.preflight` to see the full report."
            % (len(bad), format_report(results, only_problems=True),
               sys.executable))
    return results


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    results = run_all()
    if "--json" in argv:
        print(json.dumps([result._asdict() for result in results], indent=1))
    else:
        print(format_report(results))
    return 1 if failures(results) else 0


if __name__ == "__main__":
    sys.exit(main())
