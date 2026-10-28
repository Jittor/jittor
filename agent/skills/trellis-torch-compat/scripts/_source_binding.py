"""Prove the real TRELLIS source-checkout import, without inventing a release.

This upstream pin has no root packaging declaration or package version. Its
documented checkout import is an installation; source@SHA identifies its bytes,
not a PyPI distribution or a wheel. No tensor framework is imported here.
"""

import hashlib
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_installation(source, origin):
    from _installed_source_proof import package_path, source_rows

    require(source.get('repo') == 'trellis', 'Expected the actual TRELLIS inventory')
    version = source['upstream']['version']
    require(version == 'source@' + source['upstream']['commit'],
            'TRELLIS checkout identity must retain the source commit')
    origin = Path(origin).resolve()
    require(origin.name == '__init__.py' and origin.parent.name == 'trellis2',
            'Unexpected actual TRELLIS import origin')
    package = origin.parent
    required = source_rows(source.get('package_sources'))
    known = source_rows(source.get('package_files'))
    require(required.items() <= known.items(), 'Incomplete TRELLIS tracked package inventory')
    expected = {}
    for name, digest in known.items():
        prefix, relative = package_path(name)
        require(prefix == 'trellis2', 'Unexpected TRELLIS package source prefix')
        expected[relative] = digest
    actual = {}
    for path in sorted(package.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
            continue
        require(package in path.resolve().parents, 'TRELLIS package file escapes its import directory')
        actual[path.relative_to(package).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    require(actual == expected, 'Actual TRELLIS installation differs from the complete clean source payload')
    return {'repo': 'trellis', 'version': version, 'installation_kind': 'source-checkout',
            'package': str(package), 'origin': str(origin),
            'source_sha256': required, 'installed_files_sha256': actual}


def validate_phase(value, source):
    """Independently validate each runtime/perturbation against the common proof."""
    from _installed_source_proof import installation_proofs, validate_installations

    binding = value.get('downstream_distribution')
    require(isinstance(binding, dict) and binding.get('installation_kind') == 'source-checkout',
            'Missing actual TRELLIS source-checkout installation')
    require(binding.get('version') == source['upstream']['version'], 'TRELLIS installation version mismatch')
    require(binding.get('origin') == value.get('trellis_origin'), 'TRELLIS actual import origin mismatch')
    require(Path(binding.get('package', '')).name == 'trellis2'
            and str(Path(binding['package']) / '__init__.py') == binding.get('origin'),
            'Invalid TRELLIS installation package/origin binding')
    proof = installation_proofs({'repo': 'trellis'}, {'trellis': source}, [])
    validate_installations(value, proof)
    expected = {name[len('trellis2/'):]: digest for name, digest in proof['packages'][0]['known'].items()}
    require(binding.get('installed_files_sha256') == expected,
            'Incomplete or unexpected TRELLIS installed payload')
