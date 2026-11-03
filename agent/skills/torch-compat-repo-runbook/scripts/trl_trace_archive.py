"""Preserve completed oracle bulk arrays in verified, lossless Zstandard streams.

Dry-run by default. ``--archive`` removes only the redundant uncompressed copy
AFTER durable recovery metadata and a byte-identical archive have been verified.
Existing receipts/archives are never overwritten. A failed operation deliberately
leaves its journal and all compressed evidence for manual recovery; it cannot be
mistaken for a completed archive. Uses the optional external ``zstd`` executable.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from trl_rl_probe import array_hash
from trl_sft_probe import digest

JOURNAL = 'lossless-bulk-archive.jsonl'
COMPLETE = 'lossless-bulk-archive-complete.json'


def confined(directory, relative, suffix='.npy'):
    directory = Path(directory).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or relative.suffix != suffix:
        raise AssertionError('Invalid confined artifact path')
    path = directory / relative
    if directory not in path.resolve().parents:
        raise AssertionError('Artifact path escapes trace directory')
    if any(part.is_symlink() for part in [path, *path.parents] if part != directory and directory in part.parents):
        raise AssertionError('Symlink artifact paths are not accepted')
    return path


def checked_metadata(array, info):
    import numpy as np
    if list(array.shape) != info['shape'] or str(array.dtype) != info['dtype']:
        raise AssertionError('Array artifact shape/dtype mismatch')
    if array_hash(array) != info['sha256']:
        raise AssertionError('Array artifact hash mismatch')
    if not np.isfinite(array).all():
        raise AssertionError('Non-finite artifact')
    return array


def bulk_inventory(directory):
    directory = Path(directory).resolve()
    manifest = directory / 'manifest.json'
    if manifest.is_symlink():
        raise AssertionError('Symlink manifest is not accepted')
    trace = json.loads(manifest.read_text())
    if trace.get('status') != 'complete' or trace.get('runtime') != 'torch' or trace.get('fallback_count') != 0:
        raise AssertionError('Only completed zero-fallback Torch oracle traces can be archived')
    inventory = {key: info for key, info in trace['arrays'].items()
                 if any(part in ('final_gradient', 'final_parameter') for part in key.split('/')[:-1])}
    if not inventory:
        raise AssertionError('No eligible final_gradient/final_parameter arrays')
    paths = [str(confined(directory, info['path'])) for info in inventory.values()]
    if len(paths) != len(set(paths)):
        raise AssertionError('Duplicate bulk artifact paths')
    bulk_paths = set(paths)
    for key, info in trace['arrays'].items():
        if key not in inventory and info.get('path') is not None:
            if str(confined(directory, info['path'])) in bulk_paths:
                raise AssertionError('Bulk artifact path is shared with a nonbulk observation')
    return manifest, inventory


def receipt_paths(directory):
    directory = Path(directory).resolve()
    paths = directory.parent / JOURNAL, directory.parent / COMPLETE
    if any(path.is_symlink() for path in paths):
        raise AssertionError('Symlink archive receipts are not accepted')
    return paths


def archive_records(directory):
    """Validate completed receipt, original manifest binding, and entire inventory.

    Also accepts the original v1 archival receipt, whose journal does not contain
    a schema field. Both formats bind the same original bytes and array metadata.
    """
    manifest, inventory = bulk_inventory(directory)
    journal, complete = receipt_paths(directory)
    receipt = json.loads(complete.read_text())
    if digest(journal) != receipt['receipt_sha256']:
        raise AssertionError('Archive journal hash mismatch')
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    if not rows or rows[0].get('kind') != 'header' or rows[0].get('original_manifest_sha256') != digest(manifest):
        raise AssertionError('Archive original manifest binding mismatch')
    records = {}
    for row in rows[1:]:
        key = row.get('key')
        if row.get('kind') != 'verified' or key not in inventory or key in records:
            raise AssertionError('Invalid/duplicate archive inventory record')
        info = inventory[key]
        original = confined(directory, row['original_path'])
        archive = confined(directory, row['archive_path'], '.zst')
        if row['array_metadata'] != info or row['original_path'] != info['path'] or archive != original.with_suffix('.npy.zst'):
            raise AssertionError('Archive array metadata/path mismatch')
        if row['original_bytes'] <= 0 or row['archive_bytes'] <= 0:
            raise AssertionError('Invalid archive byte count')
        records[key] = row
    if set(records) != set(inventory) or receipt['files'] != len(records):
        raise AssertionError('Incomplete archive inventory')
    for name in ('original_bytes', 'archive_bytes'):
        if receipt[name] != sum(row[name] for row in records.values()):
            raise AssertionError('Archive receipt byte total mismatch')
    return records


def decode_array(archive, info, raw_sha256, original_bytes, archive_bytes):
    """Bound decompression to the recorded byte count; verify file AND array bytes."""
    import numpy as np
    if archive.stat().st_size != archive_bytes:
        raise AssertionError('Compressed archive byte count mismatch')
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(['zstd', '-q', '-d', '-c', '--', str(archive)], stdout=subprocess.PIPE, stderr=errors)
        try:
            raw = process.stdout.read(original_bytes + 1)
            if len(raw) > original_bytes:
                process.kill()
                raise AssertionError('Decompressed archive exceeds recorded byte count')
            code = process.wait()
            if code:
                raise AssertionError('Zstandard decompression failed')
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
    if len(raw) != original_bytes or hashlib.sha256(raw).hexdigest() != raw_sha256:
        raise AssertionError('Lossless reconstruction hash/length mismatch')
    return checked_metadata(np.load(io.BytesIO(raw), allow_pickle=False), info)


def read_archived_array(directory, info):
    """Read original NPY or a completed, bound sibling archive with full validation."""
    import numpy as np
    directory = Path(directory).resolve()
    source = confined(directory, info['path'])
    if source.exists():
        return checked_metadata(np.load(source, mmap_mode='r', allow_pickle=False), info)
    records = archive_records(directory)
    matches = [row for row in records.values() if row['array_metadata'] == info]
    if len(matches) != 1:
        raise AssertionError('Array is not uniquely bound to completed archive')
    row = matches[0]
    return decode_array(confined(directory, row['archive_path'], '.zst'), info,
                        row['raw_sha256'], row['original_bytes'], row['archive_bytes'])


class ArchiveReader:
    """Scoped bulk reader: validate metadata once, recheck its binding on exit.

    Payload hashes are always checked separately on every read. File identity,
    size and nanosecond modification/change times are checked per call; full
    metadata hashes are checked on normal context exit. Never use outside a
    ``with`` block or retain results after a failed context exit.
    """
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.records = None
        self.active = False
        self.bindings = {}

    def __enter__(self):
        if self.active:
            raise RuntimeError('ArchiveReader context already active')
        self.active = True
        return self

    @staticmethod
    def signature(path):
        value = path.stat()
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def __call__(self, info):
        import numpy as np
        if not self.active:
            raise RuntimeError('Use ArchiveReader inside its context')
        for path, (signature, _) in self.bindings.items():
            if self.signature(path) != signature:
                raise AssertionError('Archive metadata changed during read')
        source = confined(self.directory, info['path'])
        if source.exists():
            return checked_metadata(np.load(source, mmap_mode='r', allow_pickle=False), info)
        if self.records is None:
            paths = (self.directory / 'manifest.json', *receipt_paths(self.directory))
            before = {path: (self.signature(path), digest(path)) for path in paths}
            records = archive_records(self.directory)
            if any(self.signature(path) != signature or digest(path) != sha for path, (signature, sha) in before.items()):
                raise AssertionError('Archive metadata changed during validation')
            self.bindings = before
            self.records = {row['original_path']: row for row in records.values()}
        row = self.records.get(info['path'])
        if row is None or row['array_metadata'] != info:
            raise AssertionError('Array is not bound to completed archive')
        return decode_array(confined(self.directory, row['archive_path'], '.zst'), info,
                            row['raw_sha256'], row['original_bytes'], row['archive_bytes'])

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False
        if exc_type is None:
            for path, (signature, sha) in self.bindings.items():
                if self.signature(path) != signature or digest(path) != sha:
                    raise AssertionError('Archive metadata changed during read')
        self.records = None
        self.bindings = {}


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_row(stream, value):
    stream.write(json.dumps(value, sort_keys=True) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def archive_trace(directory, archive=False):
    """Plan or execute once. Never resume/replace an existing evidence journal."""
    import numpy as np
    directory = Path(directory).resolve()
    manifest, inventory = bulk_inventory(directory)
    journal, complete = receipt_paths(directory)
    if journal.exists() or complete.exists():
        raise FileExistsError('Archive journal/completion already exists')
    planned = []
    for key, info in inventory.items():
        source = confined(directory, info['path'])
        target = source.with_suffix('.npy.zst')
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        if not source.is_file():
            raise FileNotFoundError(source)
        planned.append((key, info, source, target))
    plan = dict(operation='archive' if archive else 'dry-run', files=len(planned),
                original_bytes=sum(source.stat().st_size for _, _, source, _ in planned),
                original_manifest_sha256=digest(manifest))
    if not archive:
        return plan
    rows = []
    # Exclusive creation is also a per-trace writer lock. Never truncate/reuse it.
    with journal.open('x') as stream:
        durable_row(stream, dict(kind='header', schema=2, original_manifest_sha256=plan['original_manifest_sha256'],
                                tool_sha256=digest(Path(__file__)), operation='Verified lossless archival'))
        fsync_directory(journal.parent)
        for key, info, source, target in planned:
            checked_metadata(np.load(source, mmap_mode='r', allow_pickle=False), info)
            original_bytes, raw_sha256 = source.stat().st_size, digest(source)
            with target.open('xb') as compressed:
                subprocess.run(['zstd', '-q', '-1', '-T2', '-c', '--', str(source)], stdout=compressed, check=True)
                compressed.flush()
                os.fsync(compressed.fileno())
            fsync_directory(target.parent)
            archive_bytes = target.stat().st_size
            decode_array(target, info, raw_sha256, original_bytes, archive_bytes)
            row = dict(kind='verified', key=key, original_path=info['path'], archive_path=str(target.relative_to(directory)),
                       raw_sha256=raw_sha256, original_bytes=original_bytes, archive_bytes=archive_bytes, array_metadata=info)
            durable_row(stream, row)
            fsync_directory(journal.parent)
            # Fail closed if the supposedly immutable trace/source changed while compressing.
            if digest(manifest) != plan['original_manifest_sha256'] or source.stat().st_size != original_bytes or digest(source) != raw_sha256:
                raise AssertionError('Source or manifest changed during archival')
            source.unlink()
            fsync_directory(source.parent)
            rows.append(row)
    # Seal the journal read-only; completion is exclusively created, never replaced.
    os.chmod(journal, 0o444)
    with journal.open('rb') as stream:
        os.fsync(stream.fileno())
    result = dict(files=len(rows), original_bytes=sum(row['original_bytes'] for row in rows),
                  archive_bytes=sum(row['archive_bytes'] for row in rows), receipt_sha256=digest(journal), schema=2)
    with complete.open('x') as stream:
        durable_row(stream, result)
    os.chmod(complete, 0o444)
    with complete.open('rb') as stream:
        os.fsync(stream.fileno())
    fsync_directory(complete.parent)
    return result


def restore_trace(directory, destination):
    """Restore a byte-verified raw trace elsewhere without modifying its source.

    This also supports producers which still require ordinary ``np.load``.
    A partial destination is deliberately retained after failure; only a
    completed recovery receipt makes it a reusable reference.
    """
    import numpy as np
    directory = Path(directory).resolve()
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if directory == destination.resolve() or directory in destination.resolve().parents \
            or destination.resolve() in directory.parents:
        raise AssertionError('Recovery destination must be separate from source')
    records = archive_records(directory)
    manifest_path = directory / 'manifest.json'
    original_manifest = digest(manifest_path)
    bindings = {p: digest(p) for p in (manifest_path, *receipt_paths(directory))}
    manifest = json.loads(manifest_path.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    journal = destination.parent / 'restore-journal.jsonl'
    complete = destination.parent / 'restore-complete.json'
    if journal.exists() or complete.exists():
        raise FileExistsError('Recovery receipts already exist')
    destination.mkdir()
    copied = restored = total = 0
    with journal.open('x') as stream:
        durable_row(stream, dict(kind='header', source=str(directory),
                                destination=str(destination), manifest_sha256=original_manifest))
        fsync_directory(journal.parent)
        for source in sorted(directory.rglob('*')):
            if source.is_symlink():
                raise AssertionError('Symlink trace entries are not accepted')
            if not source.is_file() or source.name.endswith('.npy.zst'):
                continue
            target = destination / source.relative_to(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum = digest(source)
            shutil.copy2(source, target)
            with target.open('rb') as saved:
                os.fsync(saved.fileno())
            if digest(target) != checksum or digest(source) != checksum:
                raise AssertionError('Source/copy changed during recovery')
            copied += 1
            total += target.stat().st_size
            durable_row(stream, dict(kind='copied', path=str(target.relative_to(destination)), sha256=checksum))
        for row in records.values():
            target = confined(destination, row['original_path'])
            if target.exists():
                if target.stat().st_size != row['original_bytes'] or digest(target) != row['raw_sha256']:
                    raise AssertionError('Existing raw archive payload changed')
                continue
            source = confined(directory, row['archive_path'], '.zst')
            if source.stat().st_size != row['archive_bytes']:
                raise AssertionError('Archive byte count changed')
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum, count = hashlib.sha256(), 0
            with tempfile.TemporaryFile() as errors, target.open('xb') as saved:
                process = subprocess.Popen(['zstd', '-q', '-d', '-c', '--', str(source)],
                                           stdout=subprocess.PIPE, stderr=errors)
                try:
                    while True:
                        data = process.stdout.read(1024 * 1024)
                        if not data:
                            break
                        count += len(data)
                        if count > row['original_bytes']:
                            raise AssertionError('Restored archive exceeds recorded length')
                        checksum.update(data)
                        saved.write(data)
                    if process.wait() or count != row['original_bytes'] \
                            or checksum.hexdigest() != row['raw_sha256']:
                        raise AssertionError('Restored archive hash/length mismatch')
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    process.stdout.close()
                saved.flush()
                os.fsync(saved.fileno())
            checked_metadata(np.load(target, mmap_mode='r', allow_pickle=False), row['array_metadata'])
            restored += 1
            total += count
            durable_row(stream, dict(kind='restored', path=row['original_path'], sha256=row['raw_sha256']))
        for info in manifest['arrays'].values():
            if info.get('path'):
                checked_metadata(np.load(confined(destination, info['path']), mmap_mode='r', allow_pickle=False), info)
        if any(digest(path) != checksum for path, checksum in bindings.items()):
            raise AssertionError('Source metadata changed during recovery')
        for child in sorted((p for p in destination.rglob('*') if p.is_dir()),
                            key=lambda p: len(p.parts), reverse=True):
            fsync_directory(child)
        fsync_directory(destination)
    result = dict(source=str(directory), destination=str(destination), manifest_sha256=original_manifest,
                  copied_files=copied, restored_files=restored, bytes=total,
                  journal_sha256=digest(journal), source_unchanged=True)
    with complete.open('x') as stream:
        durable_row(stream, result)
    fsync_directory(complete.parent)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--archive', action='store_true', help='Explicitly compress and remove verified redundant NPY copies')
    operation.add_argument('--restore-to', type=Path, help='Restore verified raw arrays into a new separate trace directory')
    args = parser.parse_args()
    result = restore_trace(args.trace, args.restore_to) if args.restore_to else archive_trace(args.trace, archive=args.archive)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
