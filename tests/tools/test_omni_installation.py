"""Actual Omni payload, SCM version and legitimate pip RECORD must agree."""

import base64
import csv
import hashlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent/skills/torch-compat-repo-runbook/scripts"))
sys.path.insert(0, str(ROOT / "agent/skills/vllm-omni-torch-compat/scripts"))
import _omni_installation as installation  # noqa: E402


@pytest.fixture
def installed(tmp_path, monkeypatch):
    site = tmp_path / "site"
    package = site / "vllm_omni"
    package.mkdir(parents=True)
    payload = {"__init__.py": b"from ._version import __version__\n",
               "_version.py": ("__version__ = %r\n" % installation.VERSION).encode(),
               "schema.json": b'{"runtime": "real-source-fixture"}\n'}
    for name, data in payload.items():
        (package / name).write_bytes(data)
    metadata_dir = site / ("vllm_omni-" + installation.VERSION + ".dist-info")
    metadata_dir.mkdir()
    md = ("Metadata-Version: 2.1\nName: vllm_omni\nVersion: "
          + installation.VERSION + "\n").encode()
    (metadata_dir / "METADATA").write_bytes(md)
    rows = []
    for name, data in payload.items():
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        rows.append(["vllm_omni/" + name, "sha256=" + encoded, str(len(data))])
    rows.append([metadata_dir.name + "/RECORD", "", ""])
    out = io.StringIO(newline="")
    csv.writer(out).writerows(rows)
    (metadata_dir / "RECORD").write_bytes(out.getvalue().encode())
    wheel = tmp_path / "omni-source-built.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in payload.items():
            archive.writestr("vllm_omni/" + name, data)
        archive.writestr(metadata_dir.name + "/METADATA", md)
        archive.writestr(metadata_dir.name + "/RECORD", out.getvalue())
    monkeypatch.setattr(installation, "LOCAL_OMNI_WHEEL_SHA256", installation.digest_file(wheel))
    sources = [{"path": "vllm_omni/__init__.py",
                "sha256": hashlib.sha256(payload["__init__.py"]).hexdigest()}]
    known = sources + [{"path": "vllm_omni/schema.json",
                        "sha256": hashlib.sha256(payload["schema.json"]).hexdigest()}]
    source = {"repo": "vllm-omni", "upstream": {"commit": installation.COMMIT,
              "version": "source@" + installation.COMMIT},
              "package_sources": sources, "package_files": known}
    module = SimpleNamespace(__file__=str(package / "__init__.py"),
                             __path__=[str(package)], __version__=installation.VERSION)
    dist = SimpleNamespace(_path=metadata_dir, version=installation.VERSION,
                           metadata={"Name": "vllm_omni"}, locate_file=lambda path: site / path)
    monkeypatch.setattr(installation.metadata, "distribution", lambda _name: dist)
    return source, module, wheel, metadata_dir, payload


def test_complete_actual_source_built_installation_and_generated_version(installed):
    source, module, wheel, md, payload = installed
    binding = installation.capture(source, module, wheel)
    assert binding["version"] == installation.VERSION
    assert source["upstream"]["version"].startswith("source@")
    assert set(binding["installed_files_sha256"]) == set(payload)
    assert binding["record_sha256"] == hashlib.sha256((md / "RECORD").read_bytes()).hexdigest()
    installation.validate(binding, source, wheel)


@pytest.mark.parametrize("change", ("missing_payload", "changed_payload", "extra_payload",
                                    "distribution_version", "coherent_forged_version", "record_hash", "record_scope",
                                    "wheel", "extra_origin", "foreign_metadata"))
def test_actual_prefix_and_wheel_tampering_is_rejected(installed, change, tmp_path):
    source, module, wheel, md, _payload = installed
    binding = installation.capture(source, module, wheel)
    package = Path(module.__file__).parent
    if change == "missing_payload":
        (package / "schema.json").unlink()
    elif change == "changed_payload":
        (package / "schema.json").write_bytes(b"changed")
    elif change == "extra_payload":
        (package / "untracked.bin").write_bytes(b"not in source or wheel")
    elif change in ("distribution_version", "coherent_forged_version"):
        (md / "METADATA").write_bytes((md / "METADATA").read_bytes().replace(
            installation.VERSION.encode(), b"9.9.9"))
        binding["metadata_sha256"] = installation.digest_file(md / "METADATA")
        binding["distribution_version"] = "9.9.9"
        if change == "coherent_forged_version":
            binding["version"] = "9.9.9"
    elif change in ("record_hash", "record_scope"):
        data = (md / "RECORD").read_bytes()
        if change == "record_hash":
            data = data.replace(b"sha256=", b"sha256=wrong", 1)
        else:
            data += b"vllm_omni/foreign.bin,,\r\n"
        (md / "RECORD").write_bytes(data)
        binding["record_sha256"] = installation.digest_file(md / "RECORD")
    elif change == "wheel":
        wheel.write_bytes(b"changed source wheel")
    elif change == "extra_origin":
        module.__path__.append(str(tmp_path / "foreign"))
        with pytest.raises(ValueError, match="search root"):
            installation.capture(source, module, wheel)
        return
    elif change == "foreign_metadata":
        foreign = tmp_path / "foreign.dist-info"
        foreign.mkdir()
        (foreign / "METADATA").write_bytes((md / "METADATA").read_bytes())
        binding["metadata"] = str(foreign / "METADATA")
    if change in ("missing_payload", "changed_payload", "extra_payload"):
        # Honest rehashing must not turn a changed, equally reported payload
        # into source/wheel identity. This also catches non-Python omissions.
        binding["installed_files_sha256"] = installation.package_files(package)
    with pytest.raises(ValueError):
        installation.validate(binding, source, wheel)


def test_legitimate_unhashed_generated_bytecode_record_is_preserved(installed):
    source, module, wheel, md, _payload = installed
    with (md / "RECORD").open("ab") as stream:
        stream.write(b"vllm_omni/__pycache__/_version.cpython-311.pyc,,\r\n")
    binding = installation.capture(source, module, wheel)
    assert binding["record_sha256"] == installation.digest_file(md / "RECORD")


def test_generated_version_payload_cannot_be_replaced_by_a_coherent_wheel(installed):
    source, module, wheel, md, _payload = installed
    with zipfile.ZipFile(wheel) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    name = "vllm_omni/_version.py"
    data = members[name] + b"UNBOUND_GENERATED_PAYLOAD = 1\n"
    members[name] = data
    (Path(module.__file__).parent / "_version.py").write_bytes(data)
    rows = list(csv.reader(io.StringIO((md / "RECORD").read_bytes().decode())))
    for row in rows:
        if row[0] == name:
            row[1] = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            row[2] = str(len(data))
    out = io.StringIO(newline="")
    csv.writer(out).writerows(rows)
    members[md.name + "/RECORD"] = out.getvalue().encode()
    (md / "RECORD").write_bytes(out.getvalue().encode())
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, payload in members.items():
            archive.writestr(path, payload)
    # The same version and complete, coherently rehashed payload cannot prove
    # that a replacement generated module came from the reviewed source build.
    with pytest.raises(ValueError, match="fixed artifact"):
        installation.capture(source, module, wheel)


@pytest.mark.parametrize("change", ("unbound.py", "unbound.json", "missing_schema"))
def test_coherently_rebuilt_wheel_cannot_change_source_payload(installed, change):
    source, module, wheel, md, _payload = installed
    package = Path(module.__file__).parent
    if change == "missing_schema":
        (package / "schema.json").unlink()
        with zipfile.ZipFile(wheel) as archive:
            members = {name: archive.read(name) for name in archive.namelist()
                       if name != "vllm_omni/schema.json"}
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        raw = (md / "RECORD").read_bytes()
        (md / "RECORD").write_bytes(b"\r\n".join(
            line for line in raw.split(b"\r\n") if not line.startswith(b"vllm_omni/schema.json,")))
    else:
        data = b"UNBOUND_PAYLOAD = 1\n"
        (package / change).write_bytes(data)
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr("vllm_omni/" + change, data)
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        with (md / "RECORD").open("ab") as stream:
            stream.write(("vllm_omni/" + change + ",sha256=" + encoded + ","
                          + str(len(data)) + "\r\n").encode())
    # Recompute all actual installation/wheel hashes; no deliberately stale binding.
    with pytest.raises(ValueError, match="source-built payload closure"):
        installation.capture(source, module, wheel)
