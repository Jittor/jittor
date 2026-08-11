#!/usr/bin/env python3
"""Reject incomplete or polluted Jittor source distributions."""

from __future__ import print_function

import argparse
from collections import Counter
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile


REQUIRED_SOURCE_PATHS = (
    "docs/conf.py",
    "docs/index.md",
    "docs/locales/zh_CN/LC_MESSAGES/index.po",
    "examples/README.md",
    "examples/gan/simple_cgan.py",
    "examples/notebooks/basics.md",
    "requirements/examples.txt",
    "requirements/docs.in",
    "requirements/docs.txt",
    "tools/README.md",
    "tools/build/build_aarch64_mkl.sh",
    "tools/release/pack_offline.py",
)

FORBIDDEN_DIRECTORY_NAMES = frozenset(
    (
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
        ".git",
        ".nox",
        "dist",
    )
)

FORBIDDEN_TOP_LEVEL_NAMES = frozenset(("doc", "jittor_fsdp2"))

FORBIDDEN_SOURCE_PREFIXES = (
    "python/jittor/_misc/",
    "python/jittor/_nn/",
    "python/jittor/_pool/",
    "python/jittor/_torch_compat/",
    "python/jittor/_torch_fsdp2/",
    "python/jittor/torch_fsdp2_compat/",
    "python/jittor/torch_shim/",
    "python/jittor/triton_shim/",
)

FORBIDDEN_EXACT_SOURCE_PATHS = frozenset(
    (
        "README.cn",
        "README.cn.md",
        "python/jittor/depthwise_conv.py",
        "python/jittor/misc.py",
        "python/jittor/monkeypatch_ops.py",
        "python/jittor/nn.py",
        "python/jittor/pool.py",
        "python/jittor/torch_compat.py",
        "python/jittor/torch_fsdp2_compat.py",
        "python/jittor_utils/translator.py",
    )
)

FORBIDDEN_SOURCE_SUFFIXES = (".ipynb", ".src.md")

CANONICAL_EGG_INFO_DIRECTORY = "python/jittor.egg-info"
CANONICAL_EGG_INFO_MEMBERS = frozenset(
    (
        "python/jittor.egg-info/PKG-INFO",
        "python/jittor.egg-info/SOURCES.txt",
        "python/jittor.egg-info/dependency_links.txt",
        "python/jittor.egg-info/entry_points.txt",
        "python/jittor.egg-info/requires.txt",
        "python/jittor.egg-info/top_level.txt",
    )
)
ALLOWED_GENERATED_SDIST_PATHS = frozenset(
    (CANONICAL_EGG_INFO_DIRECTORY,)
) | CANONICAL_EGG_INFO_MEMBERS


class SourceDistributionError(Exception):
    """Raised when a source archive cannot be audited."""


def _expected_source_paths(repo_root):
    command = (
        "git",
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "docs",
        "examples",
        "tools",
        "requirements/docs.in",
        "requirements/docs.txt",
        "requirements/examples.txt",
        "python",
    )
    try:
        result = subprocess.run(
            command,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise SourceDistributionError("cannot inventory source checkout: {}".format(error))
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise SourceDistributionError("cannot inventory source checkout: {}".format(detail))
    paths = frozenset(
        item
        for item in result.stdout.decode("utf-8").split("\0")
        if item
        and _generated_cache_reason(item) is None
        and not _is_egg_info_path(item)
    )
    missing_sentinels = sorted(set(REQUIRED_SOURCE_PATHS) - paths)
    if missing_sentinels:
        raise SourceDistributionError(
            "source checkout is missing required paths: {}".format(", ".join(missing_sentinels))
        )
    return paths


def _member_issues(name, source):
    issues = []
    path = PurePosixPath(name)
    if not name:
        return ["{} contains an empty member name".format(source)]
    if name.startswith("/"):
        issues.append("{} contains an absolute member: {!r}".format(source, name))
    if "\\" in name:
        issues.append("{} contains a non-POSIX member: {!r}".format(source, name))
    if any(ord(character) < 32 for character in name):
        issues.append("{} contains a control character in member: {!r}".format(source, name))
    if ".." in path.parts:
        issues.append("{} contains a parent traversal member: {!r}".format(source, name))
    return issues


def _generated_cache_reason(relative):
    parts = PurePosixPath(relative).parts
    directories = parts if relative.endswith("/") else parts[:-1]
    for part in directories:
        if part in FORBIDDEN_DIRECTORY_NAMES:
            return "generated directory {!r}".format(part)
    if parts and parts[0] == "build":
        return "generated top-level build directory"
    if relative.endswith((".pyc", ".pyo")):
        return "generated Python bytecode"
    return None


def _is_egg_info_path(relative):
    return any(
        part.endswith(".egg-info") for part in PurePosixPath(relative).parts
    )


def _pollution_reason(relative):
    if relative in ALLOWED_GENERATED_SDIST_PATHS:
        return None
    if _is_egg_info_path(relative):
        return "unapproved generated .egg-info metadata"
    parts = PurePosixPath(relative).parts
    if parts and parts[0] in FORBIDDEN_TOP_LEVEL_NAMES:
        return "forbidden legacy top-level path"
    if relative in FORBIDDEN_EXACT_SOURCE_PATHS:
        return "forbidden legacy source path"
    if any(
        relative == prefix.rstrip("/") or relative.startswith(prefix)
        for prefix in FORBIDDEN_SOURCE_PREFIXES
    ):
        return "forbidden legacy source subtree"
    generated_reason = _generated_cache_reason(relative)
    if generated_reason is not None:
        return generated_reason
    if relative.endswith(FORBIDDEN_SOURCE_SUFFIXES):
        return "forbidden notebook source/product suffix"
    return None


def audit_sdist(path, expected_paths):
    path = Path(path)
    if not path.is_file():
        raise SourceDistributionError("source distribution does not exist: {}".format(path))
    try:
        with tarfile.open(str(path), "r:gz") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError) as error:
        raise SourceDistributionError("cannot read source distribution {}: {}".format(path, error))

    issues = []
    names = [member.name for member in members]
    for name in names:
        issues.extend(_member_issues(name, path))
    for name, count in sorted(Counter(names).items()):
        if count > 1:
            issues.append("{} contains duplicate member: {}".format(path, name))

    roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
    if len(roots) != 1:
        issues.append(
            "{} must contain exactly one archive root, found {}".format(path, sorted(roots))
        )
        return issues, frozenset()

    root = next(iter(roots))
    relative_names = set()
    member_by_relative = {}
    for member in members:
        parts = PurePosixPath(member.name).parts
        if not parts or parts[0] != root:
            continue
        relative = PurePosixPath(*parts[1:]).as_posix()
        if not relative or relative == ".":
            continue
        relative_names.add(relative)
        member_by_relative[relative] = member
        reason = _pollution_reason(relative)
        if reason:
            issues.append("polluting sdist member ({}): {}".format(reason, relative))

    missing = sorted(set(expected_paths) - relative_names)
    for relative in missing:
        issues.append("missing required source-distribution member: {}".format(relative))

    governed_members = {
        relative
        for relative, member in member_by_relative.items()
        if not member.isdir()
        and (
            relative.startswith("examples/")
            or relative.startswith("docs/")
            or relative.startswith("python/")
            or relative.startswith("tools/")
            or relative
            in (
                "requirements/docs.in",
                "requirements/docs.txt",
                "requirements/examples.txt",
            )
        )
    }
    unexpected = sorted(
        governed_members - set(expected_paths) - CANONICAL_EGG_INFO_MEMBERS
    )
    for relative in unexpected:
        issues.append("unexpected source-distribution member: {}".format(relative))

    for relative in expected_paths:
        member = member_by_relative.get(relative)
        if member is not None and not member.isfile():
            issues.append("required source-distribution member is not a file: {}".format(relative))

    egg_info_directory = member_by_relative.get(CANONICAL_EGG_INFO_DIRECTORY)
    if egg_info_directory is not None and not egg_info_directory.isdir():
        issues.append(
            "canonical generated .egg-info path is not a directory: {}".format(
                CANONICAL_EGG_INFO_DIRECTORY
            )
        )
    for relative in CANONICAL_EGG_INFO_MEMBERS:
        member = member_by_relative.get(relative)
        if member is not None and not member.isfile():
            issues.append(
                "canonical generated .egg-info member is not a file: {}".format(
                    relative
                )
            )

    return issues, frozenset(relative_names)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args(argv)
    try:
        repo_root = Path(__file__).resolve().parents[2]
        expected_paths = _expected_source_paths(repo_root)
        issues, members = audit_sdist(args.sdist, expected_paths)
    except SourceDistributionError as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 2
    if issues:
        for issue in issues:
            print("ERROR: {}".format(issue), file=sys.stderr)
        return 1
    print("source distribution OK: {} ({} members)".format(args.sdist, len(members)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
