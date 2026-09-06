"""Backend implementation packages, with source-checkout package mapping."""

from pathlib import Path as _Path

_checkout_root = _Path(__file__).resolve().parents[3]
_checkout_backends = _checkout_root / "backends"
if (_checkout_root / "pyproject.toml").is_file() and _checkout_backends.is_dir():
    __path__.insert(0, str(_checkout_backends))
