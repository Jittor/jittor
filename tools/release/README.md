# Release Tools

Release utilities are repository commands, not installed APIs.

`pack_offline.py` downloads optional runtime assets and builds the
`jittor_offline` source distribution below an explicit output directory or
`$JITTOR_LAB_ROOT/_state/release/offline-package`:

```bash
python tools/release/pack_offline.py --dry-run
python tools/release/pack_offline.py --output-dir /path/to/state
```

The `legacy/` directory retains the previous source-polish workflows. Canonical
Jittor releases use the root release workflow and `python -m build`.
