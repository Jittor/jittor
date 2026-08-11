#!/usr/bin/env python3
"""Check internal links in tracked Markdown without executing documentation code."""

from __future__ import print_function

import argparse
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)")
HTML_LINK = re.compile(r"\b(?:href|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)
DOC_ROLE = re.compile(r"\{(?:doc|download)\}`([^`]+)`")
SCHEMES = frozenset(("data", "ftp", "http", "https", "mailto", "tel"))
SKIPPED_AGENT_TREES = ("agent/baselines/", "agent/results/")


def _tracked_markdown(repo_root):
    result = subprocess.run(
        ("git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"),
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip())
    selected = []
    for relative in result.stdout.decode("utf-8").split("\0"):
        if not relative.endswith(".md"):
            continue
        if any(relative.startswith(prefix) for prefix in SKIPPED_AGENT_TREES):
            continue
        parts = Path(relative).parts
        if len(parts) == 1 or parts[0] in ("agent", "docs", "examples"):
            selected.append(repo_root / relative)
    return sorted(selected)


def _strip_inline_code(line):
    output = []
    index = 0
    while index < len(line):
        if line[index] != "`":
            output.append(line[index])
            index += 1
            continue
        end = index
        while end < len(line) and line[end] == "`":
            end += 1
        marker = line[index:end]
        close = line.find(marker, end)
        if close < 0:
            output.append(line[index])
            index += 1
        else:
            output.append(" " * (close + len(marker) - index))
            index = close + len(marker)
    return "".join(output)


def _markdown_targets(path):
    in_fence = False
    fence_marker = ""
    toctree = False
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw_line.lstrip()
        fence = re.match(r"(`{3,}|~{3,}|:{3,})(.*)$", stripped)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker[0]
                toctree = fence.group(2).strip() == "{toctree}"
            elif marker[0] == fence_marker:
                in_fence = False
                toctree = False
            continue
        if in_fence:
            if toctree:
                candidate = stripped.strip()
                if candidate and not candidate.startswith(":"):
                    match = re.search(r"<([^>]+)>$", candidate)
                    yield number, match.group(1) if match else candidate, "toctree"
            continue

        line = _strip_inline_code(raw_line)
        for match in INLINE_LINK.finditer(line):
            target = match.group(1).strip()
            if target.startswith("<") and ">" in target:
                target = target[1 : target.index(">")]
            else:
                target = target.split(None, 1)[0]
            yield number, target, "markdown"
        match = REFERENCE_LINK.match(line)
        if match:
            yield number, match.group(1).strip("<>"), "reference"
        for match in HTML_LINK.finditer(line):
            yield number, match.group(1), "html"
        for match in DOC_ROLE.finditer(line):
            value = match.group(1)
            label = re.search(r"<([^>]+)>$", value)
            yield number, label.group(1) if label else value, "doc"


def _resolve(repo_root, source, raw_target, kind):
    target = raw_target.strip()
    if not target or target.startswith("#") or "{{" in target or "${" in target:
        return None
    split = urlsplit(target)
    if split.scheme.lower() in SCHEMES or split.netloc:
        return None
    clean = unquote(split.path).replace("\\", "/")
    if not clean or any(character in clean for character in "*?["):
        return None
    if kind in ("doc", "toctree"):
        if clean.startswith("/"):
            candidate = (repo_root / "docs" / clean.lstrip("/")).resolve()
        else:
            candidate = (source.parent / clean).resolve()
        if not clean.endswith(".md"):
            candidate = Path(str(candidate) + ".md")
    elif clean.startswith("/"):
        candidate = (repo_root / clean.lstrip("/")).resolve()
    else:
        candidate = (source.parent / clean).resolve()

    try:
        candidate.relative_to(repo_root)
    except ValueError:
        return "target escapes repository: {}".format(raw_target)
    if candidate.is_dir():
        candidate = candidate / "README.md"
    if not candidate.exists():
        return "missing target: {}".format(candidate.relative_to(repo_root))
    return None


def check(repo_root, paths=None):
    paths = list(paths) if paths is not None else _tracked_markdown(repo_root)
    issues = []
    for path in paths:
        for line, target, kind in _markdown_targets(path):
            problem = _resolve(repo_root, path, target, kind)
            if problem:
                issues.append(
                    "{}:{}: {} ({})".format(
                        path.relative_to(repo_root).as_posix(), line, problem, kind
                    )
                )
    return issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    paths = [path.resolve() for path in args.paths] if args.paths else None
    issues = check(repo_root, paths)
    if issues:
        print("\n".join(issues), file=sys.stderr)
        return 1
    count = len(paths) if paths is not None else len(_tracked_markdown(repo_root))
    print("internal Markdown links OK: {} files".format(count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
