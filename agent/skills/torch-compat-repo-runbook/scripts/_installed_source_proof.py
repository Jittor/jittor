"""Recheck captured package bytes, independently of release version labels."""

from email.parser import BytesParser
import hashlib
from pathlib import PurePosixPath
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


def wheel_payload(artifact, repo, version, prefix, known):
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
            require(len(header.get_all("Name", [])) == 1 and
                    normalize(header["Name"]) == normalize(repo), "Upstream wheel distribution name mismatch")
            versions = header.get_all("Version", [])
            require(len(versions) == 1 and versions[0] and not versions[0].startswith("source@") and
                    (version.startswith("source@") or versions == [version]),
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
        version, payload = source["upstream"]["version"], None
        if wheels:
            version, payload = wheel_payload(wheels[0], repo, version, prefix, known)
            require(all(package_path(name)[1] in payload for name in required),
                    "Upstream wheel omits required package source")
        result.append({"repo": repo, "version": version, "required": required, "known": known,
                       "payload": payload, "wheel_sha256": wheels[0]["sha256"] if wheels else None})
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
