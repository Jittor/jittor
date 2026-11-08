"""Recheck captured package bytes, independently of release version labels."""

import base64
import csv
from email.parser import BytesParser
from importlib import metadata
from importlib.util import source_from_cache
import io
import hashlib
from pathlib import Path, PurePosixPath
import re
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_path(value):
    require(isinstance(value, str) and value and "\\" not in value,
            "Invalid installed source path")
    path = PurePosixPath(value)
    require(bool(path.parts) and not path.is_absolute() and ".." not in path.parts and str(path) == value,
            "Invalid installed source path")
    return path


def hashes(value):
    require(isinstance(value, dict) and value, "Missing installed source/file hashes")
    for name, digest in value.items():
        relative_path(name)
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
                "Invalid installed source/file hash")
    return value


def source_rows(rows):
    require(isinstance(rows, list) and rows, "Missing regenerated full-package source inventory")
    require(all(isinstance(row, dict) and set(row) >= {"path", "sha256"} for row in rows),
            "Invalid regenerated package source inventory")
    require(all(isinstance(row["path"], str) for row in rows), "Invalid package source path")
    result = {row["path"]: row["sha256"] for row in rows}
    require(len(result) == len(rows), "Duplicate regenerated package source path")
    return hashes(result)


def package_path(name):
    parts = relative_path(name).parts
    parts = parts[1:] if parts[0] == "src" else parts
    require(len(parts) >= 2, "Invalid package source prefix")
    return parts[0], PurePosixPath(*parts[1:]).as_posix()


def digest_member(wheel, name):
    digest = hashlib.sha256()
    with wheel.open(name) as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_payload(artifact, repo, version, prefix, known, *, distribution_names=None, identity=None):
    def normalize(value):
        return re.sub(r"[-_.]+", "-", value).lower()
    try:
        with zipfile.ZipFile(artifact["path"]) as wheel:
            names = wheel.namelist()
            require(len(names) == len(set(names)), "Duplicate upstream wheel member")
            for name in names:
                relative_path(name.rstrip("/"))
            metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
            require(len(metadata) == 1, "Expected exactly one upstream wheel METADATA")
            header = BytesParser().parsebytes(wheel.read(metadata[0]))
            names_declared = distribution_names if distribution_names is not None else [repo]
            require(len(header.get_all("Name", [])) == 1 and header['Name'] and
                    normalize(header["Name"]) in {normalize(name) for name in names_declared},
                    "Upstream wheel distribution name mismatch")
            if identity is not None:
                identity['distribution_name'] = header['Name']
            versions = header.get_all("Version", [])
            require(len(versions) == 1 and versions[0] and not versions[0].startswith("source@") and
                    (version.startswith("source@") or versions == [version] or
                     equivalent_release_versions(version, versions[0])),
                    "Upstream wheel version mismatch")
            payload = {name[len(prefix) + 1:]: digest_member(wheel, name)
                       for name in names if name.startswith(prefix + "/") and not name.endswith("/")}
            require(payload, "Empty upstream wheel package payload")
            for name, expected in known.items():
                _, relative = package_path(name)
                if relative in payload:
                    require(payload[relative] == expected, "Upstream wheel/source bytes mismatch: " + name)
            return versions[0], payload
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("Invalid upstream wheel artifact") from error


def equivalent_release_versions(module_version, distribution_version):
    """Recognize PEP 440 spelling normalization, preserving both original values.

    Import packaging only for this explicit, nonliteral wheel proof. Source-only
    inventory/audit and existing literal or source@ identities do not need it.
    """
    from packaging.version import Version

    return Version(module_version) == Version(distribution_version)


def installation_proofs(record, inventories, artifacts, *, skip_own=False):
    repos = {record["repo"]} | {ref["repo"] for ref in record.get("registry_refs", [])}
    if skip_own:
        repos.discard(record["repo"])
    result = []
    for repo in sorted(repos):
        source = inventories[repo]
        required = source_rows(source.get("package_sources"))
        known = source_rows(source.get("package_files", source["package_sources"]))
        require(required.items() <= known.items(), "Package source inventory disagrees with full package files")
        prefixes = {package_path(name)[0] for name in known}
        require(len(prefixes) == 1, "Ambiguous package source prefix")
        prefix = prefixes.pop()
        roles = {repo + "-official-wheel"}
        if repo == record["repo"]:
            roles.add("upstream-wheel")
        wheels = [item for item in artifacts if item.get("role") in roles]
        require(len(wheels) <= 1, "Ambiguous upstream wheel proof")
        strict_component = repo in ('torchmetrics', 'tensordict')
        strict_diffusers = repo == 'diffusers'
        if strict_component:
            from _metric_td_installation import wheel_pin

            require(len(wheels) == 1, 'Component requires its fixed official wheel: ' + repo)
            wheel_pin(source, wheels[0]['path'])
        if strict_diffusers:
            from _diffusers_installation import wheel_pin

            require(len(wheels) == 1, 'Diffusers requires its fixed official wheel')
            wheel_pin(source, wheels[0]['path'])
        version, payload = source["upstream"]["version"], None
        normalized_distribution_version = None
        declared_names = None
        if 'distribution_names' in source:
            from _distribution_names import validate_name_declaration
            declared_names = validate_name_declaration(source['distribution_names'], source['sources'])
        identity = {}
        if wheels:
            distribution_version, payload = wheel_payload(wheels[0], repo, version, prefix, known,
                                                         distribution_names=declared_names, identity=identity)
            if version.startswith("source@"):
                version = distribution_version
            elif version != distribution_version:
                normalized_distribution_version = distribution_version
            require(all(package_path(name)[1] in payload for name in required),
                    "Upstream wheel omits required package source")
        package = {"repo": repo, "version": version, "required": required, "known": known,
                   "payload": payload, "wheel_sha256": wheels[0]["sha256"] if wheels else None}
        if strict_component:
            package['strict_distribution'] = {'source': source, 'wheel': wheels[0]['path'],
                                              'baseline': record.get('jittor_commit'),
                                              'platform': record.get('platform')}
        if strict_diffusers:
            package['strict_diffusers'] = {'source': source, 'wheel': wheels[0]['path'],
                                           'baseline': record.get('jittor_commit'),
                                           'platform': record.get('platform')}
        if normalized_distribution_version is not None:
            package["distribution_version"] = normalized_distribution_version
        if declared_names is not None:
            require(wheels, 'Source-declared distribution variants require an actual upstream wheel')
            package['distribution_name'] = identity['distribution_name']
            package['distribution_version'] = distribution_version
        result.append(package)
    return {"proof_kind": "installed-source", "packages": result}


def validate_installations(value, proof):
    for package in proof["packages"]:
        repo = package["repo"]
        plural = value.get("downstream_distributions", {})
        require(isinstance(plural, dict), "Malformed installed distribution/source bindings")
        singular = value.get("downstream_distribution")
        binding = plural.get(repo)
        if isinstance(singular, dict) and singular.get("repo") == repo:
            require(binding is None or binding == singular, "Ambiguous installed distribution/source binding")
            binding = singular
        require(isinstance(binding, dict) and binding.get("repo") == repo and
                binding.get("version") == package["version"], "Missing installed distribution/source binding: " + repo)
        if "distribution_version" in package:
            require(binding.get("distribution_version") == package["distribution_version"],
                    "Missing or mismatched installed distribution version: " + repo)
        if 'distribution_name' in package:
            require(binding.get('distribution_name') == package['distribution_name'],
                    'Missing or mismatched installed distribution name: ' + repo)
        actual = hashes(binding.get("source_sha256"))
        require(package["required"].items() <= actual.items() <= package["known"].items(),
                "Incomplete or mismatched installed distribution/source binding: " + repo)
        files = hashes(binding.get("installed_files_sha256"))
        for name, digest in actual.items():
            require(files.get(package_path(name)[1]) == digest,
                    "Installed file/source hash mismatch: " + name)
        if package["payload"] is not None:
            require(binding.get("wheel_sha256") == package["wheel_sha256"] and files == package["payload"],
                    "Incomplete or mismatched installed wheel payload: " + repo)
        else:
            known_python = {package_path(name)[1]: digest for name, digest in package["known"].items()
                            if name.endswith(".py")}
            require(all(known_python.get(name) == digest for name, digest in files.items()
                        if name.endswith(".py")), "Unbound installed Python source: " + repo)
        if 'strict_distribution' in package:
            from _metric_td_installation import validate_captured, validate_runtime_metadata

            strict = package['strict_distribution']
            validate_captured(binding, strict['source'], strict['wheel'])
            if strict['baseline'] is not None:
                validate_runtime_metadata(value, strict['baseline'], strict['platform'])
        if 'strict_diffusers' in package:
            from _diffusers_installation import validate_captured, validate_runtime_metadata

            strict = package['strict_diffusers']
            validate_captured(binding, strict['source'], strict['wheel'])
            if strict['baseline'] is not None:
                validate_runtime_metadata(value, strict['baseline'], strict['platform'])


def wheel_distribution_binding(source, wheel, binding, distribution, *, expected_wheel_sha256):
    """Recheck a selected on-disk distribution, retaining pip's real RECORD."""
    wheel, dist = Path(wheel), distribution
    repo = source['repo']
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    require(wheel_sha256 == expected_wheel_sha256, 'Unverified official wheel: ' + repo)
    package = Path(binding['package']).resolve()
    require(Path(dist.locate_file(repo + '/__init__.py')).resolve() == package / '__init__.py',
            'Package distribution metadata belongs to another installation')
    require(dist.metadata['Name'] == repo and dist.version == binding['version'] == source['upstream']['version'],
            'Actual package distribution identity mismatch')
    metadata_file = Path(dist._path) / 'METADATA'
    record_file = Path(dist._path) / 'RECORD'
    actual_record = record_file.read_bytes()
    rows = list(csv.reader(io.StringIO(actual_record.decode('utf-8'))))
    require(all(len(row) == 3 for row in rows), 'Malformed installed package RECORD')
    recorded = {row[0]: row[1:] for row in rows}
    require(len(recorded) == len(rows), 'Duplicate installed package RECORD path')
    with zipfile.ZipFile(wheel) as archive:
        entries = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        require(len(entries) == 1, 'Ambiguous upstream wheel METADATA')
        require(metadata_file.read_bytes() == archive.read(entries[0]),
                'Installed package METADATA differs from official wheel')
        package_entries = {n: archive.getinfo(n) for n in archive.namelist()
                           if n.startswith(repo + '/') and not n.endswith('/')}
        expected_payload = {name[len(repo) + 1:]: digest_member(archive, name)
                            for name in package_entries}
        require(binding.get('installed_files_sha256') == expected_payload,
                'Actual complete package payload differs from official wheel: ' + repo)
        installed_names = {n for n in recorded if n.startswith(repo + '/')}
        require(package_entries and set(package_entries) <= installed_names,
                'Incomplete installed package RECORD package scope')
        # Pip records generated bytecode with empty hashes/sizes. As in the
        # shared installed-source protocol, generated caches are not payload;
        # only cache entries mapping to an actual wheel Python source qualify.
        for name in installed_names - set(package_entries):
            try:
                original = source_from_cache(name)
            except ValueError as error:
                raise ValueError('Unbound installed package RECORD package entry: ' + name) from error
            require(name.endswith('.pyc') and original in package_entries
                    and original.endswith('.py') and recorded[name] == ['', ''],
                    'Unbound installed package RECORD bytecode entry: ' + name)
        for name, info in package_entries.items():
            relative = name[len(repo) + 1:]
            digest = binding['installed_files_sha256'].get(relative)
            require(isinstance(digest, str) and len(digest) == 64,
                    'Missing actual package hash: ' + name)
            encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('=')
            require(recorded[name] == ['sha256=' + encoded, str(info.file_size)],
                    'Installed package RECORD hash/size mismatch: ' + name)
    result = dict(binding, wheel_sha256=wheel_sha256,
                  distribution_name=dist.metadata['Name'], distribution_version=dist.version,
                  metadata=str(metadata_file), metadata_sha256=hashlib.sha256(metadata_file.read_bytes()).hexdigest(),
                  record=str(record_file), record_sha256=hashlib.sha256(actual_record).hexdigest())
    return result


def bind_official_wheel(source, module, wheel, binding, *, expected_wheel_sha256):
    """Verify original METADATA and installed RECORD's complete package entries.

    Pip legitimately rewrites RECORD when installing scripts and local metadata.
    The wheel and installed RECORD remain distinct artifacts; every wheel package
    member must retain its original hash and size in the actual installed RECORD.
    The common proof additionally compares the complete on-disk package payload.
    """
    wheel = Path(wheel)
    repo = source['repo']
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    require(wheel_sha256 == expected_wheel_sha256, 'Unverified official wheel: ' + repo)
    package = Path(module.__file__).resolve().parent
    require({Path(p).resolve() for p in module.__path__} == {package},
            'Additional package search root')
    require(Path(binding['package']).resolve() == package, 'Changed package root')
    dist = metadata.distribution(repo)
    require(Path(dist.locate_file(repo + '/__init__.py')).resolve() == Path(module.__file__).resolve(),
            'Package distribution metadata belongs to another installation')
    require(dist.metadata['Name'] == repo and dist.version == str(module.__version__)
            == source['upstream']['version'], 'Actual package distribution identity mismatch')
    result = wheel_distribution_binding(source, wheel, binding, dist,
                                        expected_wheel_sha256=expected_wheel_sha256)
    artifact = {'role': 'upstream-wheel', 'path': str(wheel), 'sha256': result['wheel_sha256']}
    proof = installation_proofs({'repo': repo}, {repo: source}, [artifact])
    validate_installations({'downstream_distribution': result}, proof)
    return result
