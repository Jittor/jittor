"""Directory ownership contracts; no framework imports or numerical claims."""

from pathlib import Path
import tempfile

import pytest

from _helpers.artifacts import artifact_directory


@pytest.fixture
def checkout(tmp_path):
    path = tmp_path / 'checkout'
    path.mkdir()
    return path


def test_default_is_unique_temporary_and_cleans_on_success(checkout, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    with artifact_directory(checkout) as first, artifact_directory(checkout) as second:
        assert first != second and first.is_dir() and second.is_dir()
        (first / 'raw.json').write_text('{}')
    assert not first.exists() and not second.exists()


def test_default_cleans_on_failure(checkout, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    with pytest.raises(RuntimeError, match='real comparison failed'):
        with artifact_directory(checkout) as output:
            (output / 'raw.json').write_text('{}')
            raise RuntimeError('real comparison failed')
    assert not output.exists()


def test_explicit_output_keeps_completed_files_without_claiming_passed(checkout, tmp_path):
    requested = tmp_path / 'evidence/cpu/model'
    with artifact_directory(checkout, output=requested) as directory:
        assert directory == requested.resolve()
        (directory / 'raw.json').write_text('{"actual": true}')
    assert (directory / 'raw.json').read_text() == '{"actual": true}'
    assert {path.name for path in directory.iterdir()} == {'raw.json'}


def test_failure_preserves_raw_files_for_diagnosis(checkout, tmp_path):
    requested = tmp_path / 'failed-run'
    with pytest.raises(RuntimeError, match='missing gradient'):
        with artifact_directory(checkout, output=requested) as directory:
            (directory / 'candidate.npz').write_bytes(b'failed raw evidence')
            raise RuntimeError('missing gradient')
    assert (requested / 'candidate.npz').read_bytes() == b'failed raw evidence'
    assert not (requested / 'complete-comparison.json').exists()


@pytest.mark.parametrize('kind', ('directory', 'file', 'symlink', 'dangling'))
def test_existing_output_never_overwritten(checkout, tmp_path, kind):
    requested = tmp_path / 'existing'
    target = tmp_path / 'original'
    target.write_bytes(b'unchanged')
    if kind == 'directory':
        requested.mkdir()
    elif kind == 'file':
        requested.write_bytes(b'unchanged')
    else:
        requested.symlink_to(target if kind == 'symlink' else tmp_path / 'missing')
    with pytest.raises(FileExistsError), artifact_directory(checkout, output=requested):
        pytest.fail('An existing output must never be yielded')
    assert target.read_bytes() == b'unchanged'
    if kind == 'file':
        assert requested.read_bytes() == b'unchanged'
    if kind in ('symlink', 'dangling'):
        assert requested.is_symlink()


def test_same_live_directory_cannot_be_reused(checkout, tmp_path):
    requested = tmp_path / 'once'
    with artifact_directory(checkout, output=requested):
        with pytest.raises(FileExistsError), artifact_directory(checkout, output=requested):
            pytest.fail('Two live protocols must not own the same directory')


@pytest.mark.parametrize('kind', ('relative', 'source', 'other_git', 'git_file', 'symlink_parent'))
def test_rejects_repository_and_relative_outputs(checkout, tmp_path, kind):
    other = tmp_path / 'other-checkout'
    other.mkdir()
    if kind == 'relative':
        requested = Path('relative-output')
    elif kind == 'source':
        requested = checkout / 'raw/nested'
    elif kind in ('other_git', 'git_file'):
        if kind == 'other_git':
            (other / '.git').mkdir()
        else:
            (other / '.git').write_text('gitdir: elsewhere')
        requested = other / 'raw/nested'
    else:
        alias = tmp_path / 'source-alias'
        alias.symlink_to(checkout, target_is_directory=True)
        requested = alias / 'raw/nested'
    with pytest.raises(ValueError), artifact_directory(checkout, output=requested):
        pytest.fail('Unsafe output must never be yielded')
    if kind != 'relative':
        assert not requested.exists()


def test_temporary_directory_also_refuses_source_location(checkout, monkeypatch):
    monkeypatch.setattr(tempfile, 'tempdir', str(checkout))
    with pytest.raises(ValueError), artifact_directory(checkout):
        pytest.fail('An implicit TMPDIR cannot put artifacts inside the source')
    assert not list(checkout.iterdir())
