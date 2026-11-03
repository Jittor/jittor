"""Statically inventory public TRL Trainers from a clean pinned upstream tree.

This reads Python AST and packaging metadata without importing TRL, Torch, or
optional dependencies. Conditional exports count in the complete denominator.
The selected evaluation subset is an explicit scope policy, not a substitute
for the full inventory and not a claim that any Trainer has passed validation.
"""

import argparse
import ast
import configparser
import hashlib
import json
from pathlib import Path
import re
import subprocess


PINNED_REVISION = "46d09bd2408f17605409fb3ee8ba12705add7faa"
MAINSTREAM = frozenset({"SFTTrainer", "DPOTrainer", "GRPOTrainer", "KTOTrainer",
                       "ORPOTrainer", "CPOTrainer", "RewardTrainer", "PPOTrainer",
                       "RLOOTrainer", "GKDTrainer", "OnlineDPOTrainer"})
EXCLUSIONS = {
    "AlignPropTrainer": "Diffusion/image training; upstream class also explicitly deprecated in this pin.",
    "DDPOTrainer": "Conditional diffusion/image trainer; upstream class also explicitly deprecated in this pin.",
    "BCOTrainer": "Alternative binary-classifier preference objective outside this initial evaluation subset; not declared deprecated here.",
    "IterativeSFTTrainer": "Older iterative SFT interface outside the selected SFTTrainer path; not declared deprecated here.",
    "NashMDTrainer": "Additional online preference objective outside this initial evaluation subset; not declared deprecated here.",
    "PRMTrainer": "Process-level reward modeling differs from the selected sequence RewardTrainer task; not declared deprecated here.",
    "XPOTrainer": "Additional exploration-based online preference objective outside this initial evaluation subset; not declared deprecated here.",
}


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _literal(value, expected_type):
    try:
        result = ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValueError("Dynamic import structure is not supported") from error
    if not isinstance(result, expected_type):
        raise ValueError("Unexpected literal in import structure")
    return result


def import_structure(source):
    """Enumerate literal registry assignments/extends, retaining control guards.

    Both sides of runtime conditions are inspected; TYPE_CHECKING-only imports
    are excluded. Unsupported mutations fail loudly rather than shrink coverage.
    """
    tree = ast.parse(source)
    exports = []
    initialized = False

    def registry_key(node):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "_import_structure":
            return _literal(node.slice, str)
        return None

    def add(module, values, line, guards):
        if not all(isinstance(value, str) for value in values):
            raise ValueError("Registry exports must be literal names")
        exports.extend({"module": module, "name": value, "line": line,
                        "conditions": list(guards)} for value in values)

    def walk(statements, guards=()):
        nonlocal initialized
        for statement in statements:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name) and target.id == "_import_structure":
                        if initialized:
                            raise ValueError("Multiple registry replacements need an explicit parser update")
                        structure = _literal(statement.value, dict)
                        for module, values in structure.items():
                            if not isinstance(module, str) or not isinstance(values, list):
                                raise ValueError("Unexpected import structure entry")
                            add(module, values, statement.lineno, guards)
                        initialized = True
                    else:
                        module = registry_key(target)
                        if module is not None:
                            add(module, _literal(statement.value, list), statement.lineno, guards)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                if isinstance(call.func, ast.Attribute):
                    if isinstance(call.func.value, ast.Name) and call.func.value.id == "_import_structure":
                        raise ValueError("Unsupported registry mutation")
                    module = registry_key(call.func.value)
                    if module is not None:
                        if call.keywords or len(call.args) != 1 or call.func.attr not in ("extend", "append"):
                            raise ValueError("Unsupported registry mutation")
                        values = (_literal(call.args[0], list) if call.func.attr == "extend"
                                  else [_literal(call.args[0], str)])
                        add(module, values, statement.lineno, guards)
            elif isinstance(statement, ast.If):
                if isinstance(statement.test, ast.Name) and statement.test.id == "TYPE_CHECKING":
                    walk(statement.orelse, guards)
                    continue
                condition = ast.unparse(statement.test)
                walk(statement.body, (*guards, "if " + condition))
                walk(statement.orelse, (*guards, "else of " + condition))
            elif isinstance(statement, ast.Try):
                body = "; ".join(ast.unparse(node) for node in statement.body)
                walk(statement.body, (*guards, "try body"))
                walk(statement.orelse, (*guards, "try succeeded: " + body))
                for handler in statement.handlers:
                    walk(handler.body, (*guards, "except " + (ast.unparse(handler.type) if handler.type else "all")))
                walk(statement.finalbody, guards)
            elif isinstance(statement, (ast.AugAssign, ast.AnnAssign)):
                target = statement.target
                if registry_key(target) is not None or (isinstance(target, ast.Name) and target.id == "_import_structure"):
                    raise ValueError("Unsupported registry assignment")
    walk(tree.body)
    if not initialized:
        raise ValueError("No static _import_structure found")
    identities = [(entry["module"], entry["name"]) for entry in exports]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate exports require disambiguation")
    return exports


def inventory(root, expected_revision=PINNED_REVISION, selected=MAINSTREAM):
    root = Path(root).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_revision):
        raise ValueError("Expected revision must be a full SHA")
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("Upstream path must be the repository root")
    revision, remote = git(root, "rev-parse", "HEAD"), git(root, "remote", "get-url", "origin")
    if revision != expected_revision:
        raise ValueError("TRL HEAD does not match the expected full SHA")
    if remote not in ("https://github.com/huggingface/trl", "https://github.com/huggingface/trl.git",
                      "git@github.com:huggingface/trl.git"):
        raise ValueError("Upstream origin must be official huggingface/trl")
    if git(root, "status", "--porcelain"):
        raise ValueError("Upstream checkout must be clean")
    top_path, trainer_path, metadata_path = (root / name for name in
        ("trl/__init__.py", "trl/trainer/__init__.py", "setup.cfg"))
    top_source = top_path.read_text()
    public = [entry for entry in import_structure(top_source) if entry["name"].endswith("Trainer")]
    definitions = import_structure(trainer_path.read_text())
    trainers = []
    source_hashes = {str(path.relative_to(root)): digest(path)
                     for path in (top_path, trainer_path, metadata_path)}
    for entry in public:
        if entry["module"] != "trainer":
            raise ValueError("Public Trainer export moved outside trainer registry")
        matches = [definition for definition in definitions if definition["name"] == entry["name"]]
        if len(matches) != 1:
            raise ValueError("Cannot uniquely resolve public Trainer: " + entry["name"])
        definition = matches[0]
        path = root / "trl/trainer" / (definition["module"].replace(".", "/") + ".py")
        tree = ast.parse(path.read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == entry["name"]]
        if len(classes) != 1:
            raise ValueError("Export is not a declared upstream class: " + entry["name"])
        relative = str(path.relative_to(root))
        source_hashes[relative] = digest(path)
        trainers.append({"name": entry["name"], "module": "trl.trainer." + definition["module"],
                         "class_source": relative, "class_line": classes[0].lineno,
                         "class_source_sha256": source_hashes[relative],
                         "top_level_export": entry, "trainer_package_export": definition,
                         "conditional": bool(entry["conditions"] or definition["conditions"]),
                         "selected_for_evaluation": entry["name"] in selected,
                         "exclusion_reason": None if entry["name"] in selected else EXCLUSIONS.get(
                             entry["name"], "Public export outside the explicitly selected evaluation subset."),
                         "validation_status": "not_assessed_by_inventory"})
    names = {row["name"] for row in trainers}
    if len(names) != len(trainers) or not trainers:
        raise ValueError("Empty or duplicate public Trainer inventory")
    if set(selected) - names:
        raise ValueError("Selected Trainers missing from pinned source: " + repr(sorted(set(selected) - names)))
    metadata = configparser.ConfigParser()
    metadata.read(metadata_path)
    versions = [ast.literal_eval(node.value) for node in ast.parse(top_source).body
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__version__"
                                                       for target in node.targets)]
    if versions != [metadata["metadata"]["version"]]:
        raise ValueError("Source and packaging versions disagree")
    def lines(text):
        return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    return {"schema_version": 1, "upstream": {"remote": remote, "revision": revision, "version": versions[0]},
            "denominator": {"axis": "top-level public Trainer exports, including conditional exports",
                            "count": len(trainers), "members": sorted(names),
                            "derivation": "AST of trl/__init__.py and trl/trainer/__init__.py; no imports"},
            "evaluation_subset": {"count": len(selected), "members": sorted(selected),
                                  "policy": "Explicit pinned-version scope; not the full denominator or an upstream mainstream designation"},
            "trainers": sorted(trainers, key=lambda row: row["name"]),
            "source_files_sha256": source_hashes,
            "dependencies": {"python_requires": metadata["options"]["python_requires"],
                             "install_requires": lines(metadata["options"]["install_requires"]),
                             "extras_require": {name: lines(value) for name, value in metadata["options.extras_require"].items()}},
            "coverage_claim": None, "warning": "Inventory and scope selection do not establish runtime compatibility or numerical acceptance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[4]
    output = args.output.expanduser().resolve()
    if output == repository or repository in output.parents:
        raise ValueError("Raw inventory must be outside the Jittor source checkout")
    result = inventory(args.upstream, args.revision)
    result["inventory_script_sha256"] = digest(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "public_trainers": result["denominator"]["count"],
                      "selected": result["evaluation_subset"]["count"], "coverage_claim": None}))


if __name__ == "__main__":
    main()
