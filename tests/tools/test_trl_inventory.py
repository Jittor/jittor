"""Offline source-denominator checks; never import TRL or tensor runtimes."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def module():
    path = ROOT / "agent/skills/torch-compat-repo-runbook/scripts/trl_inventory.py"
    spec = importlib.util.spec_from_file_location("trl_inventory_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def commit(root):
    git(root, "add", "trl", "setup.cfg")
    git(root, "-c", "user.name=Offline Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def upstream(tmp_path):
    root = tmp_path / "upstream"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "remote", "add", "origin", "https://github.com/huggingface/trl.git")
    (root / "trl/trainer").mkdir(parents=True)
    (root / "trl/__init__.py").write_text('''
raise RuntimeError("Inventory must never execute upstream code")
__version__ = "0.21.0"
_import_structure = {"trainer": ["AlphaTrainer", "AlphaConfig"]}
try:
    if not is_diffusers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["trainer"].extend(["DDPOTrainer"])
if TYPE_CHECKING:
    _import_structure["trainer"].extend(["TypingOnlyTrainer"])
''')
    (root / "trl/trainer/__init__.py").write_text('''
_import_structure = {"alpha": ["AlphaTrainer", "AlphaConfig"]}
try:
    if not is_diffusers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["ddpo"] = ["DDPOTrainer"]
''')
    (root / "trl/trainer/alpha.py").write_text("class AlphaTrainer: pass\n")
    (root / "trl/trainer/ddpo.py").write_text("class DDPOTrainer: pass\n")
    (root / "setup.cfg").write_text('''
[metadata]
name = trl
version = 0.21.0
[options]
python_requires = >=3.9
install_requires =
    transformers>=4.55.0
    datasets>=3.0.0
[options.extras_require]
diffusers =
    diffusers>=0.18.0
''')
    return root, commit(root)


def test_inventory_includes_conditional_exports_without_importing(module, upstream):
    root, revision = upstream
    result = module.inventory(root, revision, selected={"AlphaTrainer"})
    assert result["denominator"]["members"] == ["AlphaTrainer", "DDPOTrainer"]
    assert result["denominator"]["count"] == 2
    assert result["evaluation_subset"] ["count"] == 1
    assert result["coverage_claim"] is None
    ddpo = next(row for row in result["trainers"] if row["name"] == "DDPOTrainer")
    assert ddpo["conditional"] and not ddpo["selected_for_evaluation"]
    assert "is_diffusers_available" in ddpo["top_level_export"]["conditions"][0]
    assert ddpo["class_source_sha256"] == module.digest(root / ddpo["class_source"])
    assert result["dependencies"]["install_requires"] == ["transformers>=4.55.0", "datasets>=3.0.0"]
    assert result["dependencies"]["extras_require"] == {"diffusers": ["diffusers>=0.18.0"]}


def test_unified_registry_uses_the_complete_trl_source_denominator(monkeypatch, upstream):
    monkeypatch.syspath_prepend(str(ROOT / "agent/skills/torch-compat-repo-runbook/scripts"))
    import registry_inventory
    root, revision = upstream
    result = registry_inventory.inventory("trl", root)
    assert result["upstream"]["commit"] == revision
    assert result["upstream"]["version"] == "0.21.0"
    assert result["axes"][0]["axis"] == "trainer"
    assert result["axes"][0]["members"] == ["AlphaTrainer", "DDPOTrainer"]
    assert result["axes"][0]["count"] == 2
    assert "no member is verified" in result["scope"]
    assert any(item["path"] == "trl/trainer/ddpo.py" for item in result["package_sources"])
    assert any(item["path"] == "setup.cfg" for item in result["sources"])


def test_new_source_export_automatically_changes_full_denominator(module, upstream):
    root, _ = upstream
    with (root / "trl/__init__.py").open("a") as stream:
        stream.write('\n_import_structure["trainer"].append("NewTrainer")\n')
    with (root / "trl/trainer/__init__.py").open("a") as stream:
        stream.write('\n_import_structure["new"] = ["NewTrainer"]\n')
    (root / "trl/trainer/new.py").write_text("class NewTrainer: pass\n")
    result = module.inventory(root, commit(root), selected={"AlphaTrainer"})
    assert result["denominator"]["count"] == 3
    assert "NewTrainer" in result["denominator"]["members"]
    assert result["evaluation_subset"]["count"] == 1


@pytest.mark.parametrize("source", [
    '_import_structure = make_registry()',
    '_import_structure = {}; _import_structure.update({"trainer": ["XTrainer"]})',
    '_import_structure = {}; _import_structure["trainer"] += ["XTrainer"]',
    '_import_structure = {"trainer": ["XTrainer"]}; _import_structure["trainer"].append("XTrainer")',
])
def test_dynamic_or_ambiguous_registry_fails_loudly(module, source):
    with pytest.raises(ValueError):
        module.import_structure(source)


def test_wrong_revision_is_rejected(module, upstream):
    root, _ = upstream
    with pytest.raises(ValueError, match="expected full SHA"):
        module.inventory(root, "0" * 40, selected={"AlphaTrainer"})


def test_wrong_remote_is_rejected(module, upstream):
    root, revision = upstream
    git(root, "remote", "set-url", "origin", "https://example.invalid/trl.git")
    with pytest.raises(ValueError, match="official"):
        module.inventory(root, revision, selected={"AlphaTrainer"})


def test_dirty_source_is_rejected(module, upstream):
    root, revision = upstream
    (root / "trl/trainer/alpha.py").write_text("class AlphaTrainer: changed = True\n")
    with pytest.raises(ValueError, match="clean"):
        module.inventory(root, revision, selected={"AlphaTrainer"})


def test_export_without_actual_class_is_rejected(module, upstream):
    root, _ = upstream
    (root / "trl/trainer/alpha.py").write_text("AlphaTrainer = object()\n")
    with pytest.raises(ValueError, match="declared upstream class"):
        module.inventory(root, commit(root), selected={"AlphaTrainer"})


def test_source_packaging_version_mismatch_is_rejected(module, upstream):
    root, _ = upstream
    target = root / "setup.cfg"
    target.write_text(target.read_text().replace("version = 0.21.0", "version = 0.22.0"))
    with pytest.raises(ValueError, match="versions disagree"):
        module.inventory(root, commit(root), selected={"AlphaTrainer"})
