# Modern wheel baseline candidate

This is a review artifact, not the active release baseline. It was generated
from a fresh source wheel built with:

```text
python setup.py bdist_wheel --dist-dir <temporary-directory>
python tools/release/check_wheel_contents.py manifest <wheel> \
  --output docs/results/baselines/wheel-contents-modern-candidate.txt
```

The candidate contains **1,033** members and passes the current required-member
and pollution checks. Compared with the historical
`wheel-contents-final.txt`, it has **820 additions** and **648 removals**.
The complete path-and-digest lists are in:

- `wheel-additions-modern-candidate.txt`
- `wheel-removals-modern-candidate.txt`

The historical baseline is intentionally unchanged. Before adopting this
candidate, review the complete additions/removals and run the release compare
gate against the target wheel produced by CI. `jittor/build/onednn.py` is
explicitly listed as a runtime build helper because `compile_extern.py` imports
it from the wheel.

Validation performed on 2026-09-09:

- `setup.py bdist_wheel` in a temporary output directory: passed;
- `check_wheel_contents.py manifest`: passed, 1,033 members;
- required members: complete;
- forbidden pollution members: zero.
