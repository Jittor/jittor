#!/usr/bin/env python3
"""Validate bilingual Sphinx output and the curated API inventory."""

from __future__ import print_function

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys


def _docnames(root):
    return {path.relative_to(root).with_suffix("").as_posix() for path in root.rglob("*.html")}


def _check_api(html_root, inventory_path):
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    issues = []
    checked = 0
    for page in inventory["pages"]:
        html_path = html_root / (page["docname"] + ".html")
        if not html_path.is_file():
            issues.append("missing API page: {}".format(html_path))
            continue
        source = html_path.read_text(encoding="utf-8")
        identifier_list = re.findall(r'\bid=["\']([^"\']+)["\']', source)
        duplicates = sorted(
            identifier for identifier, count in Counter(identifier_list).items() if count > 1
        )
        if duplicates:
            issues.append(
                "{} has duplicate HTML anchors: {}".format(page["docname"], ", ".join(duplicates))
            )
        identifiers = set(identifier_list)
        for name in page["objects"]:
            checked += 1
            if name not in identifiers:
                issues.append("{} has no HTML anchor in {}".format(name, page["docname"]))
    return issues, checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en", type=Path, required=True)
    parser.add_argument("--zh-cn", type=Path)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "api" / "inventory.json",
    )
    args = parser.parse_args(argv)
    issues = []
    en_root = args.en.resolve()
    en_index = en_root / "index.html"
    if not en_index.is_file() or en_index.stat().st_size == 0:
        issues.append("English index.html is missing or empty")
    logo = en_root / "_static" / "logo.png"
    if not logo.is_file() or logo.stat().st_size == 0:
        issues.append("rendered logo is missing or empty")
    api_issues, checked = _check_api(en_root, args.inventory.resolve())
    issues.extend(api_issues)

    if args.zh_cn:
        zh_root = args.zh_cn.resolve()
        zh_index = zh_root / "index.html"
        if not zh_index.is_file() or zh_index.stat().st_size == 0:
            issues.append("Chinese index.html is missing or empty")
        elif "计图文档" not in zh_index.read_text(encoding="utf-8"):
            issues.append("Chinese index does not contain the translated site title")
        if _docnames(en_root) != _docnames(zh_root):
            issues.append("English and Chinese HTML docname sets differ")

    if issues:
        print("\n".join("ERROR: " + issue for issue in issues), file=sys.stderr)
        return 1
    print("documentation build OK: {} API objects".format(checked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
