"""Bind the real pinned Omni source-built wheel and its actual installation."""

import base64
import csv
from email.parser import BytesParser
from importlib import metadata
from importlib.util import source_from_cache
import io
from pathlib import Path
import subprocess
import zipfile

from _installed_source_proof import installation_proofs, require, validate_installations
from _vllm_installation import digest_file, framework_snapshot
from component_measure import package_files


COMMIT = "446c2b5474dabffccf9a0034836eeab77c62b4b8"
VERSION = "0.29.0rc2.dev58+g446c2b547"
PACKAGE = "vllm_omni"


def validate(binding, source, wheel):
    """Rehash each actual prefix independently; never compare prefixes as identity."""
    require(source.get("repo") == "vllm-omni"
            and source["upstream"]["commit"] == COMMIT,
            "Unexpected Omni source pin")
    require(isinstance(binding, dict) and binding.get("version") == VERSION
            and binding.get("installation_kind") == "source-built-wheel",
            "Unexpected actual Omni module version")
    package = Path(binding.get("package", ""))
    origin = Path(binding.get("origin", ""))
    require(package.is_absolute() and package.name == PACKAGE
            and origin == package / "__init__.py" and origin.is_file(),
            "Invalid actual Omni import root")
    for path in package.rglob("*"):
        if path.is_file():
            require(package in path.resolve().parents, "Omni package file escapes its import root")
    require(package_files(package) == binding.get("installed_files_sha256"),
            "Changed actual Omni installed payload")
    # This SCM wheel adds only setuptools-scm's generated version module.
    # Matching an arbitrary replacement wheel is insufficient source provenance.
    known = {row["path"].split("/", 1)[1] for row in source["package_files"]}
    require(set(binding["installed_files_sha256"]) == known | {"_version.py"},
            "Unexpected Omni source-built payload closure")
    wheel = Path(wheel)
    require(digest_file(wheel) == binding.get("wheel_sha256"), "Changed Omni source-built wheel")
    artifact = {"role": "upstream-wheel", "path": str(wheel), "sha256": binding["wheel_sha256"]}
    proof = installation_proofs({"repo": "vllm-omni"}, {"vllm-omni": source}, [artifact])
    validate_installations({"downstream_distribution": binding}, proof)
    paths = {}
    for key, filename in (("metadata", "METADATA"), ("record", "RECORD")):
        path = Path(binding.get(key, ""))
        require(path.is_file() and path.name == filename
                and path.parent.parent == package.parent
                and digest_file(path) == binding.get(key + "_sha256"),
                "Changed or foreign actual Omni " + key)
        paths[key] = path
    require(paths["metadata"].parent == paths["record"].parent,
            "Mixed actual Omni distribution metadata roots")
    header = BytesParser().parsebytes(paths["metadata"].read_bytes())
    require(header.get_all("Name") == [binding.get("distribution_name")]
            and header.get_all("Version") == [VERSION]
            and binding.get("distribution_version") == VERSION,
            "Wrong actual Omni distribution identity")
    rows = list(csv.reader(io.StringIO(paths["record"].read_bytes().decode("utf-8"))))
    require(all(len(row) == 3 for row in rows), "Malformed actual Omni RECORD")
    recorded = {row[0]: row[1:] for row in rows}
    require(len(rows) == len(recorded), "Duplicate actual Omni RECORD path")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        require(len(metadata_names) == 1
                and archive.read(metadata_names[0]) == paths["metadata"].read_bytes(),
                "Actual Omni METADATA differs from source-built wheel")
        entries = {name: archive.getinfo(name) for name in names
                   if name.startswith(PACKAGE + "/") and not name.endswith("/")}
        installed = {name for name in recorded if name.startswith(PACKAGE + "/")}
        require(entries and set(entries) <= installed, "Incomplete actual Omni RECORD package scope")
        for name in installed - set(entries):
            try:
                original = source_from_cache(name)
            except ValueError as error:
                raise ValueError("Unbound actual Omni RECORD entry: " + name) from error
            require(name.endswith(".pyc") and original in entries
                    and original.endswith(".py") and recorded[name] == ["", ""],
                    "Unbound actual Omni RECORD bytecode")
        for name, info in entries.items():
            digest = binding["installed_files_sha256"][name[len(PACKAGE) + 1:]]
            encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")
            require(recorded[name] == ["sha256=" + encoded, str(info.file_size)],
                    "Actual Omni RECORD package hash/size mismatch: " + name)


def capture(source, module, wheel):
    """Use the loaded module and genuine installed distribution, without rewriting versions."""
    package = Path(module.__file__).resolve().parent
    require({Path(path).resolve() for path in module.__path__} == {package},
            "Additional actual Omni package search root")
    dist = metadata.distribution("vllm-omni")
    require(Path(dist.locate_file(PACKAGE + "/__init__.py")).resolve()
            == Path(module.__file__).resolve(), "Foreign actual Omni distribution prefix")
    actual = package_files(package)
    sources = {}
    for row in source["package_sources"]:
        name = row["path"]
        prefix, relative = name.split("/", 1)
        require(prefix == PACKAGE and actual.get(relative) == row["sha256"],
                "Actual Omni source differs from clean checkout: " + name)
        sources[name] = actual[relative]
    paths = {key: Path(dist._path) / filename
             for key, filename in (("metadata", "METADATA"), ("record", "RECORD"))}
    binding = {"repo": "vllm-omni", "version": str(module.__version__),
               "package": str(package), "origin": str(Path(module.__file__).resolve()),
               "distribution_name": dist.metadata["Name"], "distribution_version": dist.version,
               "source_sha256": sources, "installed_files_sha256": actual,
               "wheel_sha256": digest_file(wheel),
               "installation_kind": "source-built-wheel"}
    binding.update({key: str(path) for key, path in paths.items()})
    binding.update({key + "_sha256": digest_file(path) for key, path in paths.items()})
    validate(binding, source, wheel)
    return binding


def framework_identity(root):
    root = Path(root)
    require(not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"]),
            "H3 diagnostics require a clean source checkout")
    captured = framework_snapshot(root)
    for name in ("_omni_installation.py", "probe_h3.py"):
        relative = "agent/skills/vllm-omni-torch-compat/scripts/" + name
        captured[relative] = digest_file(root / relative)
    return {"commit": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "clean": True, "source_sha256": captured}
