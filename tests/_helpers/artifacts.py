"""Temporary or explicitly retained artifacts for maintained protocol gates.

This owns directories only. The caller must validate its final measurements
before reporting success or writing a completion record. Persistent failures
keep their raw files; an existing output is never reused or overwritten.
"""

from contextlib import contextmanager
from pathlib import Path
import tempfile


def _outside_checkout(path, checkout):
    resolved = path.resolve()
    if resolved == checkout or checkout in resolved.parents:
        raise ValueError('Artifact output must be outside the source checkout')
    for ancestor in (resolved, *resolved.parents):
        metadata = ancestor / '.git'
        if metadata.exists() or metadata.is_symlink():
            raise ValueError('Artifact output must be outside every Git checkout')
    return resolved


@contextmanager
def artifact_directory(checkout, *, output=None, prefix='jittor-artifacts-'):
    """Yield a fresh directory, retaining it only for an explicit output path.

    ``output`` must be absolute and outside all Git checkouts, including through
    symlinks. Its leaf is created exclusively; parents may already exist.
    Without it, preserve TemporaryDirectory cleanup on success and failure.
    No successful-execution claim or completion marker is generated here.
    """
    checkout = Path(checkout).resolve()
    if output is None:
        with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
            yield _outside_checkout(Path(temporary), checkout)
        return
    requested = Path(output).expanduser()
    if not requested.is_absolute():
        raise ValueError('Persistent artifact output must be absolute')
    if requested.exists() or requested.is_symlink():
        raise FileExistsError('Artifact output already exists: ' + str(requested))
    directory = _outside_checkout(requested, checkout)
    directory.mkdir(parents=True, exist_ok=False)
    yield directory
