#!/usr/bin/env python3
"""Check active Markdown links and the repository documentation contract."""

from __future__ import print_function

import html
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[2]

# These trees are historical evidence, not authoritative active documentation.
EXCLUDED_PREFIXES = (
    "agent/baselines/",
    "agent/results/",
)

REQUIRED_DOCUMENTS = (
    "README.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
    "agent/manuals/project-context.md",
    "agent/manuals/environment.md",
    "agent/manuals/known-issues.md",
    "docs/architecture/repository-layout.md",
    "docs/architecture/source-architecture.md",
    "docs/architecture/torch-compatibility-principles.md",
    "docs/architecture/complex-dtype.md",
    "docs/testing/test-system.md",
    "docs/development/known-issues/parallel-compiler-segfault.md",
    "docs/performance/benchmarking.md",
    "docs/research/agentic-optimization.md",
    "docs/releases/2.0.md",
)

INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+[^)]*)?\)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(?:<([^>]+)>|(\S+))")
HTML_LINK = re.compile(r"\b(?:href|src)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _relative(path):
    return path.relative_to(REPO_ROOT).as_posix()


def _is_excluded(relative):
    return any(relative.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def active_documents():
    documents = []
    for path in sorted(REPO_ROOT.rglob("*.md")):
        relative = _relative(path)
        if any(part in {".git", ".nox", ".pytest_cache", "__pycache__"} for part in path.parts):
            continue
        if not _is_excluded(relative):
            documents.append(path)
    return documents


def prose_lines(path):
    fence_marker = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = FENCE.match(line)
        if match:
            marker = match.group(1)[0]
            if fence_marker is None:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = None
            continue
        if fence_marker is None:
            yield number, line


def link_targets(line):
    for match in INLINE_LINK.finditer(line):
        yield match.group(1) or match.group(2)
    match = REFERENCE_LINK.match(line)
    if match:
        yield match.group(1) or match.group(2)
    for match in HTML_LINK.finditer(line):
        yield match.group(1)


def local_target(raw_target):
    target = html.unescape(raw_target.strip())
    if not target or target.startswith("#") or target.startswith("//"):
        return None, None
    if SCHEME.match(target):
        return None, None

    parsed = urlsplit(target)
    path_text = unquote(parsed.path).replace("\\ ", " ")
    if not path_text:
        return None, None
    if path_text.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", path_text):
        return None, "absolute repository link is not portable: {0}".format(raw_target)
    return path_text, None


def check_links(documents):
    errors = []
    for document in documents:
        for line_number, line in prose_lines(document):
            for raw_target in link_targets(line):
                target, error = local_target(raw_target)
                if error:
                    errors.append("{0}:{1}: {2}".format(_relative(document), line_number, error))
                    continue
                if target is None:
                    continue
                resolved = (document.parent / target).resolve()
                try:
                    resolved.relative_to(REPO_ROOT)
                except ValueError:
                    errors.append(
                        "{0}:{1}: link escapes repository: {2}".format(
                            _relative(document), line_number, raw_target
                        )
                    )
                    continue
                if not resolved.exists():
                    errors.append(
                        "{0}:{1}: missing relative-link target: {2}".format(
                            _relative(document), line_number, raw_target
                        )
                    )
    return errors


def check_contract(documents):
    errors = []
    for relative in REQUIRED_DOCUMENTS:
        if not (REPO_ROOT / relative).is_file():
            errors.append("missing required document: {0}".format(relative))

    root_readmes = sorted(path.name for path in REPO_ROOT.glob("README*.md"))
    if root_readmes != ["README.md"]:
        errors.append("root must contain only README.md; found {0}".format(root_readmes))

    if (REPO_ROOT / "agent" / "manuals" / "design").exists():
        errors.append("retired documentation path still exists: agent/manuals/design")

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for token in (
        "Quickstart / 快速开始",
        "Install / 安装",
        "Tutorials / 教程",
        "Contributing / 贡献",
        "python -m jittor.selftest",
    ):
        if token not in readme:
            errors.append("README.md is missing bilingual/current contract: {0}".format(token))

    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    for token in (
        "pyproject.toml",
        "noxfile.py",
        "tests/",
        "python -m pytest",
        "python -m nox -s structure",
        "python -m nox -s docs",
        "python -m nox -s docs_zh",
        "python -m nox -s docs_links",
        "python -m nox -s tutorials",
    ):
        if token not in contributing:
            errors.append("CONTRIBUTING.md is missing current workflow: {0}".format(token))
    for token in ("doc/source/", "build_doc.sh", "jittor.test.test_example"):
        if token in contributing:
            errors.append("CONTRIBUTING.md references retired workflow: {0}".format(token))

    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for token in ("cc-connect", "/home/"):
        if token in agents:
            errors.append("AGENTS.md contains host/tool-specific rule: {0}".format(token))

    context_path = REPO_ROOT / "agent" / "manuals" / "project-context.md"
    context = context_path.read_text(encoding="utf-8")
    context_lines = len(context.splitlines())
    if context_lines > 180:
        errors.append(
            "project-context.md must remain an index (<=180 lines); found {0}".format(context_lines)
        )
    if "/home/" in context or re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", context):
        errors.append("project-context.md contains machine-specific environment details")
    for token in ("Last reviewed:", "Baseline reviewed:", "Owner:", "Freshness expires:"):
        if token not in context:
            errors.append("project-context.md is missing freshness metadata: {0}".format(token))

    stale_references = (
        "README.cn.md",
        "README.src.md",
        "agent/manuals/design/",
        "manuals/design/",
        "doc/source/",
        "build_doc.sh",
        "jittor.test.test_example",
    )
    for document in documents:
        source = document.read_text(encoding="utf-8")
        for stale in stale_references:
            if stale in source:
                errors.append(
                    "{0}: references retired documentation path: {1}".format(
                        _relative(document), stale
                    )
                )
    return errors


def main():
    documents = active_documents()
    errors = check_contract(documents)
    errors.extend(check_links(documents))
    if errors:
        print("documentation governance failures:", file=sys.stderr)
        for error in errors:
            print("- " + error, file=sys.stderr)
        return 1
    print("documentation governance OK: {0} active Markdown files".format(len(documents)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
