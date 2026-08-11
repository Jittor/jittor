# Legacy Release Polish

These commands preserve historical precompiled-data workflows. They no longer
write a package-local version marker, import the old translator, or place
release artifacts inside the runtime package. Output defaults to
`$JITTOR_LAB_ROOT/_state/release`.

Run a no-write preview before using either workflow:

```bash
python tools/release/legacy/polish.py --dry-run
python tools/release/legacy/polish_centos.py --dry-run
```
