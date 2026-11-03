"""Lossless retention, crash safety, provenance, and hostile-path controls."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest


@pytest.fixture
def archive(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('trace_archive', scripts / 'trl_trace_archive.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace(archive, tmp_path, runtime='torch', status='complete'):
    directory = tmp_path / 'run' / 'trace'
    directory.mkdir(parents=True)
    arrays = {}
    for key, name in [('step/10/final_parameter/w', 'parameter'),
                      ('step/10/final_gradient/w', 'gradient'), ('step/10/loss', 'loss')]:
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        np.save(directory / (name + '.npy'), data)
        arrays[key] = dict(path=name + '.npy', shape=[3, 4], dtype='float32', sha256=archive.array_hash(data))
    manifest = dict(runtime=runtime, status=status, fallback_count=0, arrays=arrays)
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    return directory, manifest


def require_zstd():
    if not shutil.which('zstd'):
        pytest.skip('optional zstd executable unavailable')


def test_default_dry_run_changes_nothing(archive, tmp_path):
    directory, _ = trace(archive, tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert archive.archive_trace(directory)['files'] == 2
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_lossless_roundtrip_bound_receipt_and_exclusive_outputs(archive, tmp_path):
    require_zstd()
    directory, manifest = trace(archive, tmp_path)
    raw = {p.name: p.read_bytes() for p in directory.glob('*.npy')}
    original_manifest = (directory / 'manifest.json').read_bytes()
    result = archive.archive_trace(directory, archive=True)
    assert result['files'] == 2
    assert (directory / 'loss.npy').read_bytes() == raw['loss.npy']
    assert not (directory / 'gradient.npy').exists()
    assert (directory / 'manifest.json').read_bytes() == original_manifest
    for info in manifest['arrays'].values():
        np.testing.assert_array_equal(archive.read_archived_array(directory, info), np.arange(12).reshape(3, 4))
    assert (directory.parent / archive.JOURNAL).stat().st_mode & 0o222 == 0
    assert (directory.parent / archive.COMPLETE).stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        archive.archive_trace(directory, archive=True)


def test_restore_raw_trace_is_exact_and_does_not_change_source(archive, tmp_path):
    require_zstd()
    directory, manifest = trace(archive, tmp_path)
    initial = directory / 'initial'
    initial.mkdir()
    np.savez(initial / 'extras.npz', bias=np.ones(3, dtype=np.float32))
    raw = {str(p.relative_to(directory)): p.read_bytes()
           for p in directory.rglob('*') if p.is_file()}
    archive.archive_trace(directory, archive=True)
    archived = {str(p.relative_to(tmp_path)): p.read_bytes()
                for p in directory.parent.rglob('*') if p.is_file()}
    destination = tmp_path / 'restored' / 'trace'
    result = archive.restore_trace(directory, destination)
    assert result['restored_files'] == 2
    assert (destination.parent / 'restore-complete.json').is_file()
    assert raw == {str(p.relative_to(destination)): p.read_bytes()
                   for p in destination.rglob('*') if p.is_file()}
    assert archived == {str(p.relative_to(tmp_path)): p.read_bytes()
                        for p in directory.parent.rglob('*') if p.is_file()}
    for info in manifest['arrays'].values():
        archive.checked_metadata(np.load(destination / info['path']), info)
    with pytest.raises(FileExistsError):
        archive.restore_trace(directory, destination)


@pytest.mark.parametrize('relative', ['run/trace/nested', 'run'])
def test_restore_rejects_overlapping_directories(archive, tmp_path, relative):
    require_zstd()
    directory, _ = trace(archive, tmp_path)
    archive.archive_trace(directory, archive=True)
    with pytest.raises((AssertionError, FileExistsError)):
        archive.restore_trace(directory, tmp_path / relative)


def test_restore_corrupt_archive_never_marks_partial_output_complete(archive, tmp_path):
    require_zstd()
    directory, _ = trace(archive, tmp_path)
    archive.archive_trace(directory, archive=True)
    (directory / 'gradient.npy.zst').write_bytes(b'broken archive')
    destination = tmp_path / 'restored' / 'trace'
    with pytest.raises(AssertionError, match='byte count|hash/length'):
        archive.restore_trace(directory, destination)
    assert (directory.parent / archive.COMPLETE).is_file()
    assert not (destination.parent / 'restore-complete.json').exists()


def test_restore_flushes_nested_directories_before_completion(archive, tmp_path, monkeypatch):
    require_zstd()
    directory, _ = trace(archive, tmp_path)
    (directory / 'initial').mkdir()
    (directory / 'initial/extra.json').write_text('{}')
    archive.archive_trace(directory, archive=True)
    destination = tmp_path / 'restored/trace'
    flushed = set()
    real_fsync = archive.fsync_directory
    real_row = archive.durable_row

    def fsync(path):
        flushed.add(Path(path))
        real_fsync(path)

    def record(stream, value):
        if 'restored_files' in value:
            assert destination / 'initial' in flushed
            assert destination in flushed
        real_row(stream, value)

    monkeypatch.setattr(archive, 'fsync_directory', fsync)
    monkeypatch.setattr(archive, 'durable_row', record)
    archive.restore_trace(directory, destination)


@pytest.mark.parametrize('runtime,status', [('jittor', 'complete'), ('torch', 'running')])
def test_reject_nonoracle_or_incomplete(archive, tmp_path, runtime, status):
    directory, _ = trace(archive, tmp_path, runtime, status)
    with pytest.raises(AssertionError, match='completed'):
        archive.archive_trace(directory, archive=True)


@pytest.mark.parametrize('relative', ['../outside.npy', '/tmp/escape.npy', 'array.txt'])
def test_reject_escape_and_wrong_extension(archive, tmp_path, relative):
    with pytest.raises(AssertionError, match='confined'):
        archive.read_archived_array(tmp_path, dict(path=relative))


def test_reject_symlink_even_within_trace(archive, tmp_path):
    directory, manifest = trace(archive, tmp_path)
    (directory / 'alias.npy').symlink_to(directory / 'gradient.npy')
    info = dict(next(iter(manifest['arrays'].values())), path='alias.npy')
    with pytest.raises(AssertionError, match='Symlink'):
        archive.read_archived_array(directory, info)


def test_failed_recovery_journal_does_not_unlink_source(archive, tmp_path, monkeypatch):
    require_zstd()
    directory, _ = trace(archive, tmp_path)
    original = archive.durable_row
    def fail_verified(stream, row):
        if row['kind'] == 'verified':
            raise OSError('simulated fsync failure')
        return original(stream, row)
    monkeypatch.setattr(archive, 'durable_row', fail_verified)
    with pytest.raises(OSError, match='fsync'):
        archive.archive_trace(directory, archive=True)
    assert (directory / 'parameter.npy').exists()
    assert (directory / 'parameter.npy.zst').exists()
    assert not (directory.parent / archive.COMPLETE).exists()
    with pytest.raises(FileExistsError):
        archive.archive_trace(directory, archive=True)


def test_payload_and_journal_durable_before_unlink(archive, tmp_path, monkeypatch):
    require_zstd()
    directory, _ = trace(archive, tmp_path)
    real_unlink = Path.unlink
    def checked_unlink(path, *args, **kwargs):
        if path.suffix == '.npy':
            records = [json.loads(line) for line in (directory.parent / archive.JOURNAL).read_text().splitlines()]
            assert records[-1]['original_path'] == path.name
            assert path.with_suffix('.npy.zst').is_file()
        return real_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', checked_unlink)
    archive.archive_trace(directory, archive=True)


@pytest.mark.parametrize('damage', ['archive', 'manifest', 'journal', 'completion', 'metadata'])
def test_reject_corrupt_or_unbound_archives(archive, tmp_path, damage):
    require_zstd()
    directory, manifest = trace(archive, tmp_path)
    archive.archive_trace(directory, archive=True)
    info = dict(manifest['arrays']['step/10/final_parameter/w'])
    if damage == 'archive':
        path = directory / 'parameter.npy.zst'
        value = bytearray(path.read_bytes())
        value[len(value)//2] ^= 0xff
        path.write_bytes(value)
    elif damage == 'manifest':
        (directory / 'manifest.json').write_text(json.dumps(dict(manifest, device='changed')))
    elif damage == 'journal':
        with (directory.parent / archive.JOURNAL).open('a') as stream:
            stream.write('{}\n')
    elif damage == 'completion':
        (directory.parent / archive.COMPLETE).unlink()
    else:
        info['sha256'] = '0' * 64
    with pytest.raises((AssertionError, FileNotFoundError)):
        archive.read_archived_array(directory, info)


def test_raw_file_hash_checked_independently_of_array_hash(archive, tmp_path):
    require_zstd()
    directory, manifest = trace(archive, tmp_path)
    archive.archive_trace(directory, archive=True)
    info = manifest['arrays']['step/10/final_parameter/w']
    row = archive.archive_records(directory)['step/10/final_parameter/w']
    for raw_hash, length in [('0'*64, row['original_bytes']), (row['raw_sha256'], row['original_bytes'] - 1)]:
        with pytest.raises(AssertionError, match='hash/length|byte count'):
            archive.decode_array(directory / row['archive_path'], info, raw_hash, length, row['archive_bytes'])


def test_shape_dtype_finite_checks(archive):
    array = np.array([np.nan], dtype=np.float32)
    info = dict(shape=[1], dtype='float32', sha256=archive.array_hash(array))
    with pytest.raises(AssertionError, match='Non-finite'):
        archive.checked_metadata(array, info)
    with pytest.raises(AssertionError, match='shape/dtype'):
        archive.checked_metadata(np.ones(2, dtype=np.float32), info)


def test_scoped_reader_caches_validated_inventory_and_rechecks_binding(archive, tmp_path, monkeypatch):
    require_zstd()
    directory, manifest = trace(archive, tmp_path)
    archive.archive_trace(directory, archive=True)
    real = archive.archive_records
    calls = []
    def record(path):
        calls.append(path)
        return real(path)
    monkeypatch.setattr(archive, 'archive_records', record)
    with archive.ArchiveReader(directory) as reader:
        for info in manifest['arrays'].values():
            assert reader(info).shape == (3, 4)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match='context'):
        reader(next(iter(manifest['arrays'].values())))
    with pytest.raises(AssertionError, match='changed'):
        with archive.ArchiveReader(directory) as reader:
            reader(next(iter(manifest['arrays'].values())))
            with (directory.parent / archive.JOURNAL).open('a') as stream:
                stream.write('{}\n')


def test_reject_bulk_path_shared_with_nonbulk_observation(archive, tmp_path):
    directory, manifest = trace(archive, tmp_path)
    manifest['arrays']['step/10/loss'] = dict(manifest['arrays']['step/10/final_parameter/w'])
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    for execute in (False, True):
        with pytest.raises(AssertionError, match='shared with a nonbulk'):
            archive.archive_trace(directory, archive=execute)
    assert before == {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
