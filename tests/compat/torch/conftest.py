"""Audit hook for the torch-compat permissive import finder (task 7.10).

The permissive finder answers a *known list* of import-time references under
``torch._inductor``/``torch._dynamo``/... and refuses the rest.  Widening a
list must be evidence-based, so run the workload once in audit mode and read
back what it actually needed::

    JITTOR_TORCH_PERMISSIVE_AUDIT=1 \\
    JITTOR_PERMISSIVE_AUDIT_OUT=/tmp/permissive.json \\
    python -m pytest tests/compat/torch/

The file lists every module that was fabricated and every one that was refused.
Without ``JITTOR_PERMISSIVE_AUDIT_OUT`` this hook does nothing.
"""


def pytest_sessionfinish(session, exitstatus):
    import json
    import os

    out = os.environ.get("JITTOR_PERMISSIVE_AUDIT_OUT")
    if not out:
        return
    try:
        from jittor.compat import permissive
    except Exception:
        return
    with open(out, "w") as handle:
        json.dump({"fabricated": sorted(permissive.fabricated_modules()),
                   "refused": sorted(permissive.refused_modules())},
                  handle, indent=1)
