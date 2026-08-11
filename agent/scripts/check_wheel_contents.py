#!/usr/bin/env python3
"""Audit a wheel against an older wheel or an approved hash manifest.

Every reference member is pinned by its SHA-256 digest. Additions, removals, and
content changes fail unless the exact transition is explicitly allowlisted.
Candidate wheels are also rejected when required runtime resources are absent or
common cache, build, and experiment artifacts are present.

Examples:

    python agent/scripts/check_wheel_contents.py compare dist/jittor.whl
    python agent/scripts/check_wheel_contents.py compare new.whl --old-wheel old.whl
    python agent/scripts/check_wheel_contents.py compare new.whl \
        --old-wheel old.whl --addition-allowlist approved-additions.txt \
        --content-change-allowlist approved-content-changes.txt
    python agent/scripts/check_wheel_contents.py manifest old.whl --output baseline.txt
"""

from __future__ import print_function

import argparse
from collections import Counter
import hashlib
from pathlib import Path, PurePosixPath
import sys
import zipfile


AGENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = AGENT_ROOT / "baselines" / "wheel-contents-stage5.txt"
DEFAULT_ADDITION_ALLOWLIST = (
    AGENT_ROOT / "baselines" / "wheel-additions-stage6.txt"
)
DEFAULT_CONTENT_CHANGE_ALLOWLIST = (
    AGENT_ROOT / "baselines" / "wheel-content-changes-stage6.txt"
)

REQUIRED_MEMBERS = (
    "jittor/compat/shim/cpp_extension/include/ATen/cuda/detail/UnpackRaw.cuh",
    "jittor/compat/shim/resources/stubs/flash_attn/flash_attn_interface.py",
    "jittor/compat/shim/resources/torch_init.py",
)

FORBIDDEN_DIRECTORY_NAMES = frozenset((
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".git",
    ".claude",
    ".codex",
    ".agents",
    "build",
    "dist",
    "__data__",
    "worktrees",
    "_state",
))

FORBIDDEN_TOP_LEVEL_NAMES = frozenset((
    "agent",
    "jittor-lab",
    "jittor_fsdp2",
))

FORBIDDEN_MEMBER_PREFIXES = (
    "jittor/_misc/",
    "jittor/_nn/",
    "jittor/_pool/",
    "jittor/_torch_compat/",
    "jittor/_torch_fsdp2/",
    "jittor/demo/",
    "jittor/notebook/",
    "jittor/script/",
    "jittor/test/",
    "jittor/torch_fsdp2_compat/",
    "jittor/torch_shim/",
    "jittor/triton_shim/",
    "jittor/vcompiler/",
)

FORBIDDEN_EXACT_MEMBERS = frozenset((
    "jittor/depthwise_conv.py",
    "jittor/misc.py",
    "jittor/monkeypatch_ops.py",
    "jittor/nn.py",
    "jittor/pool.py",
    "jittor/torch_compat.py",
    "jittor/torch_fsdp2_compat.py",
    "jittor/utils/polish.py",
    "jittor/utils/polish_centos.py",
    "jittor/version",
    "jittor_utils/pack_offline.py",
    "jittor_utils/translator.py",
))

FORBIDDEN_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".log",
    ".swp",
    ".ipynb",
    ".src.md",
)


class WheelContentsError(Exception):
    """Raised for an unreadable or structurally invalid audit input."""


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _validate_member_name(name, source):
    issues = []
    if not name:
        return ["{} contains an empty archive member name".format(source)]
    if name.startswith("/"):
        issues.append("{} contains an absolute member: {!r}".format(source, name))
    if "\\" in name:
        issues.append("{} contains a non-POSIX member: {!r}".format(source, name))
    if any(ord(character) < 32 for character in name):
        issues.append("{} contains a control character in member: {!r}".format(
            source, name))
    if ".." in PurePosixPath(name).parts:
        issues.append("{} contains a parent traversal member: {!r}".format(
            source, name))
    return issues


def _read_wheel(path):
    path = Path(path)
    if not path.is_file():
        raise WheelContentsError("wheel does not exist: {}".format(path))
    try:
        with zipfile.ZipFile(str(path), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            bad_member = archive.testzip()
            members = {}
            for info in infos:
                with archive.open(info, "r") as stream:
                    members[info.filename] = _stream_sha256(stream)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise WheelContentsError("cannot read wheel {}: {}".format(path, error))

    issues = []
    if bad_member is not None:
        issues.append("{} has a corrupt member: {}".format(path, bad_member))
    duplicates = sorted(
        name for name, count in Counter(names).items() if count > 1)
    for name in duplicates:
        issues.append("{} contains duplicate member: {}".format(path, name))
    for name in names:
        issues.extend(_validate_member_name(name, path))
    return members, issues


def _parse_sha256(value, label):
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise WheelContentsError(
            "{} has an invalid lowercase SHA-256 digest: {!r}".format(label, value)
        )
    return value


def _read_hashed_path_list(path, label):
    path = Path(path)
    if not path.is_file():
        raise WheelContentsError("{} does not exist: {}".format(label, path))
    entries = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise WheelContentsError("cannot read {} {}: {}".format(label, path, error))
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line != raw_line:
            raise WheelContentsError(
                "{} {}:{} has leading or trailing whitespace".format(
                    label, path, line_number
                )
            )
        digest, separator, name = line.partition("  ")
        location = "{} {}:{}".format(label, path, line_number)
        if not separator or not name:
            raise WheelContentsError(
                "{} must use '<sha256>  <wheel-path>'".format(location)
            )
        _parse_sha256(digest, location)
        issues = _validate_member_name(name, path)
        if issues:
            raise WheelContentsError(issues[0])
        if name in entries:
            raise WheelContentsError(
                "{} {} contains duplicate path: {}".format(label, path, name)
            )
        entries[name] = digest
    return entries


def _read_path_list(path, label):
    path = Path(path)
    if not path.is_file():
        raise WheelContentsError("{} does not exist: {}".format(label, path))
    entries = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise WheelContentsError("cannot read {} {}: {}".format(label, path, error))
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line != raw_line:
            raise WheelContentsError(
                "{} {}:{} has leading or trailing whitespace".format(
                    label, path, line_number))
        entries.append(line)

    duplicates = sorted(
        name for name, count in Counter(entries).items() if count > 1)
    if duplicates:
        raise WheelContentsError(
            "{} {} contains duplicate entries: {}".format(
                label, path, ", ".join(duplicates)))
    for name in entries:
        issues = _validate_member_name(name, path)
        if issues:
            raise WheelContentsError(issues[0])
    return frozenset(entries)


def _pollution_reason(name):
    parts = PurePosixPath(name).parts
    if not parts:
        return "empty path"
    if parts[0] in FORBIDDEN_TOP_LEVEL_NAMES:
        return "forbidden top-level experiment path"
    if name in FORBIDDEN_EXACT_MEMBERS:
        return "retired runtime-wheel member"
    if any(
        name == prefix.rstrip("/") or name.startswith(prefix)
        for prefix in FORBIDDEN_MEMBER_PREFIXES
    ):
        return "retired runtime-wheel subtree"
    if len(parts) >= 2 and parts[:2] == ("jittor", "projects"):
        return "repository-local experiment package"
    directory_parts = parts if name.endswith("/") else parts[:-1]
    for part in directory_parts:
        if part in FORBIDDEN_DIRECTORY_NAMES or part.endswith(".egg-info"):
            return "forbidden cache/build directory {!r}".format(part)
    if name.endswith(FORBIDDEN_SUFFIXES):
        return "forbidden generated file suffix"
    if len(parts) == 1 and parts[0] in ("core", "perf.data", "perf.data.old"):
        return "forbidden runtime artifact"
    return None


def _candidate_issues(members, extra_required):
    issues = []
    names = frozenset(members)
    required = set(REQUIRED_MEMBERS)
    required.update(extra_required)
    for name in sorted(required - names):
        issues.append("required wheel member is missing: {}".format(name))
    for name in sorted(names):
        reason = _pollution_reason(name)
        if reason is not None:
            issues.append("polluting wheel member ({}): {}".format(reason, name))
    return issues


def _manifest_text(wheel_path, members):
    header = [
        "# Jittor wheel contents baseline (SHA-256 and exact archive member paths)",
        "# source-wheel: {}".format(wheel_path.name),
        "# source-wheel-sha256: {}".format(_sha256(wheel_path)),
        "# entries: {}".format(len(members)),
        "# Regenerate only after reviewing every path and content change.",
    ]
    entries = [
        "{}  {}".format(members[name], name) for name in sorted(members)
    ]
    return "\n".join(header + entries) + "\n"


def _write_manifest(args):
    wheel_path = Path(args.wheel)
    members, wheel_issues = _read_wheel(wheel_path)
    issues = wheel_issues + _candidate_issues(members, args.require)
    if issues:
        for issue in issues:
            print("ERROR: {}".format(issue), file=sys.stderr)
        return 1

    content = _manifest_text(wheel_path, members)
    if args.output == "-":
        sys.stdout.write(content)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        print("wrote {} entries to {}".format(len(members), output))
    return 0


def _load_reference(args):
    if args.old_wheel is not None:
        path = Path(args.old_wheel)
        members, issues = _read_wheel(path)
        if issues:
            raise WheelContentsError("; ".join(issues))
        return members, "wheel {}".format(path)
    path = Path(args.baseline or DEFAULT_BASELINE)
    return _read_hashed_path_list(path, "baseline"), "baseline {}".format(path)


def _load_removal_allowlist(args, reference):
    allowed = set(args.allow_removal)
    if args.removal_allowlist is not None:
        allowed.update(_read_path_list(
            args.removal_allowlist, "removal allowlist"))
    unknown = sorted(allowed - set(reference))
    if unknown:
        raise WheelContentsError(
            "removal allowlist contains path(s) absent from the reference: {}".format(
                ", ".join(unknown)))
    return allowed


def _uses_default_policy(args):
    if args.old_wheel is not None:
        return False
    if args.baseline is None:
        return True
    return Path(args.baseline).resolve() == DEFAULT_BASELINE.resolve()


def _merge_hashed_entries(target, incoming, label):
    for name, digest in incoming.items():
        previous = target.get(name)
        if previous is not None and previous != digest:
            raise WheelContentsError(
                "{} contains conflicting approved hashes for {}: {} and {}".format(
                    label, name, previous, digest
                )
            )
        target[name] = digest


def _inline_hashed_entries(values, label):
    entries = {}
    for digest, name in values:
        _parse_sha256(digest, label)
        issues = _validate_member_name(name, label)
        if issues:
            raise WheelContentsError(issues[0])
        _merge_hashed_entries(entries, {name: digest}, label)
    return entries


def _load_hashed_allowlist(args, kind, default_path):
    defaults = {}
    if _uses_default_policy(args):
        _merge_hashed_entries(
            defaults,
            _read_hashed_path_list(default_path, "default {} allowlist".format(kind)),
            "{} allowlist".format(kind),
        )
    explicit = {}
    for path in getattr(args, "{}_allowlist".format(kind)):
        _merge_hashed_entries(
            explicit,
            _read_hashed_path_list(path, "{} allowlist".format(kind)),
            "{} allowlist".format(kind),
        )
    _merge_hashed_entries(
        explicit,
        _inline_hashed_entries(
            getattr(args, "allow_{}".format(kind)),
            "inline {} allowance".format(kind),
        ),
        "{} allowlist".format(kind),
    )
    # A later migration stage may deliberately change a member that an earlier
    # default stage already changed.  Explicit policy describes the candidate
    # being audited, while conflicts within explicit policy remain errors.
    allowed = defaults.copy()
    allowed.update(explicit)
    return allowed


def _load_change_allowlists(args, reference):
    additions = _load_hashed_allowlist(
        args, "addition", DEFAULT_ADDITION_ALLOWLIST
    )
    content_changes = _load_hashed_allowlist(
        args, "content_change", DEFAULT_CONTENT_CHANGE_ALLOWLIST
    )
    invalid_additions = sorted(set(additions) & set(reference))
    if invalid_additions:
        raise WheelContentsError(
            "addition allowlist contains reference path(s): {}".format(
                ", ".join(invalid_additions)
            )
        )
    invalid_content_changes = sorted(set(content_changes) - set(reference))
    if invalid_content_changes:
        raise WheelContentsError(
            "content-change allowlist contains path(s) absent from the reference: {}".format(
                ", ".join(invalid_content_changes)
            )
        )
    no_op_content_changes = sorted(
        name
        for name, digest in content_changes.items()
        if reference[name] == digest
    )
    if no_op_content_changes:
        raise WheelContentsError(
            "content-change allowlist does not describe a change for: {}".format(
                ", ".join(no_op_content_changes)
            )
        )
    return additions, content_changes


def _compare(args):
    candidate_path = Path(args.wheel)
    candidate, candidate_archive_issues = _read_wheel(candidate_path)
    reference, reference_label = _load_reference(args)
    allowed_removals = _load_removal_allowlist(args, reference)
    allowed_additions, allowed_content_changes = _load_change_allowlists(
        args, reference
    )

    reference_names = set(reference)
    candidate_names = set(candidate)
    removed = reference_names - candidate_names
    added = candidate_names - reference_names
    changed = {
        name
        for name in reference_names & candidate_names
        if reference[name] != candidate[name]
    }

    approved_removals = removed & allowed_removals
    unexpected_removals = removed - allowed_removals
    unconsumed_removal_allowances = allowed_removals - removed
    approved_additions = {
        name
        for name in added
        if allowed_additions.get(name) == candidate[name]
    }
    unexpected_additions = added - approved_additions
    approved_content_changes = {
        name
        for name in changed
        if allowed_content_changes.get(name) == candidate[name]
    }
    unexpected_content_changes = changed - approved_content_changes
    unconsumed_addition_allowances = set(allowed_additions) - added
    unconsumed_content_change_allowances = set(allowed_content_changes) - changed

    issues = list(candidate_archive_issues)
    issues.extend(_candidate_issues(candidate, args.require))
    for name in sorted(unexpected_removals):
        issues.append("wheel member removed without approval: {}".format(name))
    for name in sorted(unconsumed_removal_allowances):
        issues.append(
            "approved wheel removal is not present: {}".format(name)
        )
    for name in sorted(unexpected_additions):
        expected = allowed_additions.get(name)
        if expected is None:
            issues.append("wheel member added without approval: {}".format(name))
        else:
            issues.append(
                "wheel addition hash does not match approval: {} "
                "(expected {}, found {})".format(name, expected, candidate[name])
            )
    for name in sorted(unexpected_content_changes):
        expected = allowed_content_changes.get(name)
        if expected is None:
            issues.append(
                "wheel member content changed without approval: {}".format(name)
            )
        else:
            issues.append(
                "wheel content-change hash does not match approval: {} "
                "(expected {}, found {})".format(name, expected, candidate[name])
            )
    for name in sorted(unconsumed_addition_allowances):
        issues.append(
            "approved wheel addition is absent from the candidate: {}".format(name)
        )
    for name in sorted(unconsumed_content_change_allowances):
        found = candidate.get(name)
        if found is None:
            detail = "member is absent"
        else:
            detail = "found {}".format(found)
        issues.append(
            "approved wheel content change is not present: {} "
            "(expected {}, {})".format(
                name, allowed_content_changes[name], detail
            )
        )

    print("reference: {} ({} paths)".format(reference_label, len(reference)))
    print("candidate: {} ({} paths)".format(candidate_path, len(candidate)))
    print(
        "added: {} (approved: {}, unexpected: {}); "
        "content changed: {} (approved: {}, unexpected: {}); "
        "removed: {} (approved: {}, unexpected: {})".format(
            len(added),
            len(approved_additions),
            len(unexpected_additions),
            len(changed),
            len(approved_content_changes),
            len(unexpected_content_changes),
            len(removed),
            len(approved_removals),
            len(unexpected_removals),
        )
    )
    for name in sorted(approved_additions):
        print("  + {} [approved {}]".format(name, candidate[name]))
    for name in sorted(approved_content_changes):
        print("  ~ {} [approved {}]".format(name, candidate[name]))
    for name in sorted(approved_removals):
        print("  - {} [approved]".format(name))

    if issues:
        for issue in issues:
            print("ERROR: {}".format(issue), file=sys.stderr)
        return 1
    print("wheel contents OK")
    return 0


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    manifest = subparsers.add_parser(
        "manifest", help="emit a sorted, auditable path manifest for a wheel")
    manifest.add_argument("wheel", help="wheel used as the approved source")
    manifest.add_argument(
        "--output", "-o", default="-",
        help="manifest output path; default: stdout")
    manifest.add_argument(
        "--require", action="append", default=[], metavar="WHEEL_PATH",
        help="require an additional exact member path")
    manifest.set_defaults(run=_write_manifest)

    compare = subparsers.add_parser(
        "compare", help="compare a candidate wheel with an approved reference")
    compare.add_argument("wheel", help="candidate wheel to audit")
    reference = compare.add_mutually_exclusive_group()
    reference.add_argument(
        "--baseline", help="approved one-path-per-line manifest; default: {}".format(
            DEFAULT_BASELINE))
    reference.add_argument(
        "--old-wheel", help="approved older wheel used as the path reference")
    compare.add_argument(
        "--removal-allowlist", metavar="FILE",
        help="one exact approved removal path per line; comments start with #")
    compare.add_argument(
        "--allow-removal", action="append", default=[], metavar="WHEEL_PATH",
        help="approve one exact removal; may be repeated")
    compare.add_argument(
        "--addition-allowlist", action="append", default=[], metavar="FILE",
        help=(
            "approved '<candidate-sha256>  <wheel-path>' additions; may be "
            "repeated"
        ),
    )
    compare.add_argument(
        "--allow-addition", action="append", nargs=2, default=[],
        metavar=("SHA256", "WHEEL_PATH"),
        help="approve one exact addition and candidate hash; may be repeated",
    )
    compare.add_argument(
        "--content-change-allowlist", action="append", default=[], metavar="FILE",
        help=(
            "approved '<candidate-sha256>  <wheel-path>' content changes; "
            "may be repeated"
        ),
    )
    compare.add_argument(
        "--allow-content-change", action="append", nargs=2, default=[],
        metavar=("SHA256", "WHEEL_PATH"),
        help="approve one exact changed member and candidate hash; may be repeated",
    )
    compare.add_argument(
        "--require", action="append", default=[], metavar="WHEEL_PATH",
        help="require an additional exact member path")
    compare.set_defaults(run=_compare)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        return args.run(args)
    except WheelContentsError as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
