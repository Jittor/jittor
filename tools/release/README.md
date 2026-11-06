# Release Tools

Release utilities are repository commands, not installed APIs.

`pack_offline.py` downloads optional runtime assets and builds the
`jittor_offline` source distribution below an explicit output directory or
`$JITTOR_LAB_ROOT/_state/release/offline-package`:

```bash
python tools/release/pack_offline.py --dry-run
python tools/release/pack_offline.py --output-dir /path/to/state
```

Jittor releases use the root release workflow and `python -m build`. The old
source-polish/precompiled-data release pipeline is retired.

The contents-checker unit tests live in `tests/structure/release/`, so the
regular structure gate collects them without a separate tools-only selection.
They can also run with the standard library alone:

```bash
python -m unittest discover -s tests/structure/release -v
```

`baselines/` contains inputs consumed by the wheel audit and shim resource
contract. Historical wheel inventories without a current consumer remain in
Git history rather than the release tools directory.
