#!/usr/bin/env python3
"""Reject stale, fuzzy, obsolete, or compiled gettext catalog state."""

from __future__ import print_function

import argparse
from pathlib import Path
import sys

from babel.messages.pofile import read_po


FULL_TRANSLATION_CATALOGS = {
    "guides/cpp-console.po",
    "guides/debugging.po",
    "guides/distributed-mpi.po",
    "guides/memory-optimization.po",
    "guides/performance-comparison.po",
    "index.po",
}


def _catalogs(root):
    catalogs = {}
    for path in root.rglob("*.po"):
        with path.open(encoding="utf-8") as stream:
            catalogs[path.relative_to(root).as_posix()] = read_po(stream)
    return catalogs


def _messages(catalog):
    return {(message.id, message.context): message for message in catalog if message.id}


def _translated(message):
    if isinstance(message.string, (list, tuple)):
        return bool(message.string) and all(message.string)
    return bool(message.string)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args(argv)
    expected = _catalogs(args.expected.resolve())
    actual = _catalogs(args.actual.resolve())
    issues = []
    missing = sorted(set(expected) - set(actual))
    added = sorted(set(actual) - set(expected))
    changed = []
    for name in sorted(set(expected) & set(actual)):
        expected_messages = _messages(expected[name])
        actual_messages = _messages(actual[name])
        if set(expected_messages) != set(actual_messages):
            changed.append(name)
            continue
        for key, message in expected_messages.items():
            if message.string != actual_messages[key].string:
                changed.append(name)
                break
    if missing or added or changed:
        issues.append(
            "catalogs are stale (missing={}, added={}, changed={})".format(missing, added, changed)
        )
    for name, catalog in sorted(expected.items()):
        messages = _messages(catalog).values()
        if catalog.fuzzy or any("fuzzy" in message.flags for message in messages):
            issues.append("fuzzy translation: {}".format(name))
        if catalog.obsolete:
            issues.append("obsolete translation: {}".format(name))
        translated = [message for message in messages if _translated(message)]
        if not translated:
            issues.append("empty translation catalog: {}".format(name))
        if name in FULL_TRANSLATION_CATALOGS and len(translated) != len(messages):
            issues.append(
                "legacy catalog is not fully translated: {} ({}/{})".format(
                    name, len(translated), len(messages)
                )
            )
    generated = list(args.expected.resolve().rglob("*.mo"))
    if generated:
        issues.append("compiled catalogs are tracked beside source: {}".format(generated))
    if issues:
        print("\n".join("ERROR: " + issue for issue in issues), file=sys.stderr)
        return 1
    print("gettext catalogs are fresh: {} files".format(len(expected)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
