"""Installed release payload contracts, with real miniature wheel bytes."""

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       "agent/skills/torch-compat-repo-runbook/scripts"))
from _installed_source_proof import installation_proofs, validate_installations  # noqa: E402


def digest(value):
    return hashlib.sha256(value).hexdigest()


class InstalledSourceProofTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.record = {"repo": "peft", "registry_refs": [{"repo": "transformers"}]}
        self.inventories = {}
        self.artifacts = []
        self.bindings = {}
        for repo in ("peft", "transformers"):
            payload = {"__init__.py": b"# source\n", "kernels/data.cu": b"// CUDA source\n"}
            source = {"src/" + repo + "/" + name: digest(data) for name, data in payload.items()}
            self.inventories[repo] = {
                "upstream": {"version": "1.0"},
                "package_sources": [{"path": name, "sha256": sha} for name, sha in source.items()
                                    if name.endswith(".py")],
                "package_files": [{"path": name, "sha256": sha} for name, sha in source.items()],
            }
            path = Path(self.temp.name) / (repo + ".whl")
            with zipfile.ZipFile(path, "w") as wheel:
                for name, data in payload.items():
                    wheel.writestr(repo + "/" + name, data)
                wheel.writestr(repo + "-1.0.dist-info/METADATA", "Name: " + repo + "\nVersion: 1.0\n")
            wheel_sha = digest(path.read_bytes())
            self.artifacts.append({"role": repo + "-official-wheel", "path": str(path), "sha256": wheel_sha})
            self.bindings[repo] = {"repo": repo, "version": "1.0", "source_sha256": source,
                                   "installed_files_sha256": {name: digest(data) for name, data in payload.items()},
                                   "wheel_sha256": wheel_sha}

    def proof(self):
        return installation_proofs(self.record, self.inventories, self.artifacts)

    def test_full_non_python_source_superset_and_both_dependencies_pass(self):
        validate_installations({"downstream_distributions": self.bindings}, self.proof())

    def test_each_referenced_library_requires_its_own_installation(self):
        for repo in self.bindings:
            bindings = copy.deepcopy(self.bindings)
            bindings.pop(repo)
            with self.subTest(repo=repo), self.assertRaisesRegex(ValueError, "installed distribution/source"):
                validate_installations({"downstream_distributions": bindings}, self.proof())

    def test_complete_python_source_does_not_prove_full_wheel_payload(self):
        for repo in self.bindings:
            for field in ("installed_files_sha256", "wheel_sha256"):
                bindings = copy.deepcopy(self.bindings)
                if field == "installed_files_sha256":
                    bindings[repo][field].pop("kernels/data.cu")
                else:
                    bindings[repo][field] = "f" * 64
                with self.subTest(repo=repo, field=field), self.assertRaises(ValueError):
                    validate_installations({"downstream_distributions": bindings}, self.proof())

    def test_conflicting_singular_and_plural_binding_rejected(self):
        value = {"downstream_distributions": self.bindings,
                 "downstream_distribution": dict(self.bindings["peft"], version="2.0")}
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            validate_installations(value, self.proof())

    def test_untracked_source_claim_rejected(self):
        self.bindings["peft"]["source_sha256"]["src/peft/unknown.py"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "installed distribution/source"):
            validate_installations({"downstream_distributions": self.bindings}, self.proof())

    def test_without_wheel_unknown_python_requires_byte_binding(self):
        self.artifacts = []
        validate_installations({"downstream_distributions": self.bindings}, self.proof())
        self.bindings["peft"]["installed_files_sha256"]["unknown.py"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "Unbound installed Python"):
            validate_installations({"downstream_distributions": self.bindings}, self.proof())

    def test_wheel_name_version_and_bytes_independently_verified(self):
        path = Path(self.artifacts[0]["path"])
        original = path.read_bytes()
        for name, version, source in (("other", "1.0", b"# source\n"),
                                      ("peft", "2.0", b"# source\n"),
                                      ("peft", "1.0", b"# changed\n")):
            with zipfile.ZipFile(path, "w") as wheel:
                wheel.writestr("peft/__init__.py", source)
                wheel.writestr("peft-1.0.dist-info/METADATA", "Name: " + name + "\nVersion: " + version + "\n")
            with self.subTest(name=name, version=version), self.assertRaises(ValueError):
                self.proof()
            path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
