"""Derive downstream coverage denominators from clean, pinned upstream Git sources.

No downstream or tensor package is imported. A new upstream registry layout must
be taught explicitly; an empty or unresolvable registry is an error, not zero
coverage. Registry members describe registered APIs, never verified support.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


REPOS = ("transformers", "peft", "diffusers", "ms-swift", "verl")
ROOT = Path(__file__).resolve().parents[4]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root)] + list(args), text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Raw inventories and scorecards must be outside the Jittor checkout")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class SourceRegistry:
    def __init__(self, root, paths):
        self.root = root
        self.trees = {}
        self.constants = {}
        self.members = {}
        for relative in sorted(paths):
            path = root / relative
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            self.trees[relative] = tree
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    try:
                                        value = ast.literal_eval(item.value)
                                    except (ValueError, TypeError):
                                        continue
                                    if isinstance(value, str):
                                        self.constants[node.name + "." + target.id] = value
                                        self.members.setdefault(target.id, set()).add(value)

    def value(self, node):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                key = node.value.id + "." + node.attr
                if key in self.constants:
                    return self.constants[key]
            # ModelType aggregates constants through multiple inheritance.
            # Resolve inherited enum members only when the value is unambiguous.
            values = self.members.get(node.attr, set())
            if len(values) == 1:
                return next(iter(values))
            raise ValueError("Unresolved registry member: " + node.attr)
        return ast.literal_eval(node)

    def mapping_keys(self, node):
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise ValueError("Registry dictionary unpacking needs an explicit extractor")
            return [self.value(key) for key in node.keys]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("OrderedDict", "dict") and len(node.args) == 1
                and isinstance(node.args[0], (ast.List, ast.Tuple))):
            return [self.value(pair.elts[0]) for pair in node.args[0].elts]
        raise ValueError("Unsupported registry construction at line %s" % getattr(node, "lineno", "?"))

    def assignments(self, pattern):
        found = []
        for relative, tree in self.trees.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and re.fullmatch(pattern, target.id):
                            found.append((relative, target.id, node.value))
        return found


def inventory(repo, checkout):
    if repo not in REPOS:
        raise ValueError("Unknown downstream repo: " + repo)
    checkout = Path(checkout).expanduser().resolve()
    commit = git(checkout, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected a full upstream commit SHA")
    if git(checkout, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Upstream checkout is dirty; denominator must match its commit")
    remote = git(checkout, "remote", "get-url", "origin")
    files = git(checkout, "ls-files", "--", "*.py").splitlines()
    prefix = {"transformers": "src/transformers/", "peft": "src/peft/",
              "diffusers": "src/diffusers/", "ms-swift": "swift/", "verl": "verl/"}[repo]
    files = [path for path in files if path.startswith(prefix)]
    if repo == "transformers":
        selected = [path for path in files if path.endswith("models/auto/modeling_auto.py")]
    elif repo == "peft":
        selected = [path for path in files if path.endswith("utils/peft_types.py")]
    elif repo == "diffusers":
        selected = [path for path in files if path.endswith(("pipelines/auto_pipeline.py", "pipelines/__init__.py"))]
    elif repo == "ms-swift":
        selected = [path for path in files if "/model/" in path or "/tuners/" in path
                    or path.endswith("model_type.py")]
    else:
        selected = [path for path in files if path.endswith("trainer/ppo/core_algos.py")
                    or ("/models/" in path and "registry" in path)]
    source = SourceRegistry(checkout, selected)
    axes = []
    used = set()

    def add(axis, name, members, sources):
        members = sorted(set(members))
        if not members or not all(isinstance(member, str) and member for member in members):
            raise ValueError("Empty or invalid %s registry %s" % (repo, name))
        axes.append({"axis": axis, "registry": name, "count": len(members), "members": members,
                     "sources": sorted(set(sources))})
        used.update(sources)

    if repo == "transformers":
        mappings = source.assignments(r"MODEL_.*MAPPING_NAMES|MODEL_MAPPING_NAMES")
        base = [(path, name, node) for path, name, node in mappings if name == "MODEL_MAPPING_NAMES"]
        if len(base) != 1:
            raise ValueError("Expected exactly one Transformers MODEL_MAPPING_NAMES")
        path, name, node = base[0]
        add("model_family", name, source.mapping_keys(node), [path])
        add("task", "MODEL_*_MAPPING_NAMES", [name for _, name, _ in mappings], [p for p, _, _ in mappings])
    elif repo == "peft":
        for cls, axis in (("PeftType", "method"), ("TaskType", "task")):
            values, paths = [], []
            for path, tree in source.trees.items():
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == cls:
                        values.extend(source.constants[node.name + "." + item.targets[0].id]
                                      for item in node.body if isinstance(item, ast.Assign)
                                      and isinstance(item.targets[0], ast.Name)
                                      and node.name + "." + item.targets[0].id in source.constants)
                        paths.append(path)
            add(axis, cls, values, paths)
    elif repo == "diffusers":
        mappings = source.assignments(r"AUTO_.*PIPELINES_MAPPING")
        families, pipeline_classes, pipeline_paths = [], [], []
        for path, tree in source.trees.items():
            if not path.endswith("pipelines/__init__.py"):
                continue
            for node in ast.walk(tree):
                target, values = None, None
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target, values = node.targets[0], node.value
                elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "extend" and len(node.args) == 1):
                    target, values = node.func.value, node.args[0]
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id == "_import_structure"):
                    key = target.slice
                    if hasattr(ast, "Index") and isinstance(key, ast.Index):
                        key = key.value
                    names = ast.literal_eval(values)
                    pipelines = [name for name in names if "Pipeline" in name
                                 and not name.endswith(("Output", "Mixin"))]
                    if pipelines:
                        families.append(ast.literal_eval(key))
                        pipeline_classes.extend(pipelines)
                        pipeline_paths.append(path)
        add("model_family", "pipelines._import_structure pipeline modules (all optional branches)", families, pipeline_paths)
        add("pipeline_class", "pipelines._import_structure pipeline exports", pipeline_classes, pipeline_paths)
        add("auto_family", "AUTO_*_PIPELINES_MAPPING.keys", [member for _, _, node in mappings
                                                               for member in source.mapping_keys(node)],
            [path for path, _, _ in mappings])
        add("task", "AUTO_*_PIPELINES_MAPPING", [name for _, name, _ in mappings], [p for p, _, _ in mappings])
    elif repo == "ms-swift":
        models, model_paths = [], []
        for path, tree in source.trees.items():
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "register_model" and node.args):
                    meta = node.args[0]
                    if not isinstance(meta, ast.Call):
                        raise ValueError("Dynamic ms-swift model registration in " + path)
                    model = meta.args[0] if meta.args else next(
                        (item.value for item in meta.keywords if item.arg == "model_type"), None)
                    if model is None:
                        raise ValueError("Missing model_type in " + path)
                    models.append(source.value(model))
                    model_paths.append(path)
        add("model_family", "register_model(ModelMeta)", models, model_paths)
        mappings = source.assignments(r"SWIFT_MAPPING")
        add("method", "SWIFT_MAPPING", [member for _, _, node in mappings for member in source.mapping_keys(node)],
            [path for path, _, _ in mappings])
    else:
        for function, axis in (("register_adv_est", "advantage_estimator"), ("register_policy_loss", "policy_loss")):
            members, paths = [], []
            for path, tree in source.trees.items():
                for node in ast.walk(tree):
                    if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
                        for decorator in node.decorator_list:
                            if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)
                                    and decorator.func.id == function and decorator.args):
                                members.append(source.value(decorator.args[0]))
                                paths.append(path)
            add(axis, function, members, paths)
    version = None
    for relative in files:
        if relative.endswith(("/__init__.py", "/version.py", "/_version.py")):
            tree = ast.parse((checkout / relative).read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(value, str):
                        version = value
                        used.add(relative)
                        break
            if version:
                break
    if repo == "verl" and (checkout / "verl/version/version").is_file():
        version = (checkout / "verl/version/version").read_text(encoding="utf-8").strip()
        used.add("verl/version/version")
    return {"schema_version": 1, "repo": repo, "upstream": {"commit": commit, "remote": remote, "version": version},
            "scope": "Source-declared registry members; no member is verified by enumeration.", "axes": axes,
            "sources": [{"path": path, "sha256": digest(checkout / path)} for path in sorted(used)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, choices=REPOS)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output")
    options = parser.parse_args()
    result = inventory(options.repo, options.checkout)
    if options.output:
        write_json(options.output, result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
