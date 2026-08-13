"""Version-independent entry-point discovery shared by compatibility hooks."""

from __future__ import annotations

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - exercised on Python 3.7 CI
    import importlib_metadata  # type: ignore[no-redef]


def entry_points(group: str):
    discovered = importlib_metadata.entry_points()
    select = getattr(discovered, "select", None)
    if callable(select):
        return list(select(group=group))
    return list(discovered.get(group, ()))


__all__ = ["entry_points"]
