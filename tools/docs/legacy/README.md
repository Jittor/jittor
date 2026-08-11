# Legacy Documentation Tool

`make_doc.py` reproduces the old Doxygen download and build. It is import-safe,
supports `--dry-run`, and stores downloads and output under
`$JITTOR_LAB_ROOT/_state/tools/doxygen` by default.

```bash
python tools/docs/legacy/make_doc.py --dry-run
```

The modern Sphinx/MyST documentation pipeline supersedes this tool in Stage 8.
