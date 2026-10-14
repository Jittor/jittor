"""Audit runbooks and build fail-closed, registry-denominated downstream scorecards.

The audit is source-only. A skill, registered case, or historical Markdown report
is discoverable evidence, never a passed execution. Scorecards count only fresh,
validated records with pinned source inventories and external raw-artifact hashes.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re

from registry_inventory import ROOT, git, inventory, write_json


PLATFORM_FIELDS = ("os", "cpu", "accelerator_model", "card_count", "card_indices", "driver",
                   "cuda", "cann", "nvcc", "gcc", "nvidia_wheels", "jittor_flags", "oracle")
FLAGS = ("use_cuda", "use_acl", "graph_replay", "tf32", "hf32")


def _assignments(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)}


def ecosystem_cases(root):
    result = {}
    for path in (root / "compat/tests/torch/_ecosystem_cases.py", root / "compat/tests/torch/_ecosystem_speed.py"):
        mapping = _assignments(path).get("CASES")
        if not isinstance(mapping, ast.Dict):
            raise ValueError("Expected literal ecosystem CASES in " + str(path))
        for key, value in zip(mapping.keys, mapping.values):
            name = ast.literal_eval(key)
            requirements = ast.literal_eval(value.elts[1])
            result[name] = {"requires": list(requirements), "source": path.relative_to(root).as_posix()}
    return result


def audit(root=ROOT):
    root = Path(root).resolve()
    cases = ecosystem_cases(root)
    gate_files = ("compat/tests/torch/test_ecosystem_parity.py", "compat/tests/torch/test_ecosystem_speed.py")
    methods = {}
    for relative in gate_files:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                methods.setdefault(node.name[5:], []).append(relative + "::" + node.name)
    rows = []
    for skill in sorted((root / "agent/skills").glob("*-torch-compat/SKILL.md")):
        repo = skill.parent.name[:-len("-torch-compat")]
        prefix = repo.replace("-", "_")
        direct = [name for name in cases if name.startswith((prefix + "_", "large_" + prefix + "_"))]
        source = skill.read_text(encoding="utf-8")
        references = sorted(set(re.findall(r"docs/results/([\w.-]+\.md)", source)) |
                            set(re.findall(r"20\d{2}-\d{2}-\d{2}-[A-Za-z0-9._-]+\.md", source)))
        reports = ["docs/results/" + name for name in references if (root / "docs/results" / name).is_file()]
        rows.append({
            "repo": repo, "skill": skill.relative_to(root).as_posix(),
            "registered_cases": direct,
            "gate_cases": [{"case": name, "tests": methods[name]} for name in direct if name in methods],
            "dependency_cases": [name for name, data in cases.items()
                                 if ("swift" if repo == "ms-swift" else repo) in data["requires"]],
            "reports": reports,
            "report_mentions": [str(path.relative_to(root)) for path in sorted((root / "docs/results").glob("*.md"))
                                if path.name != "index.md" and re.search(r"\b" + re.escape(repo) + r"\b", path.read_text(encoding="utf-8"), re.I)],
            "missing_reports": [name for name in references if "docs/results/" + name not in reports],
            "mentioned_levels": sorted(set(re.findall(r"\bL[1-5]\b", source))),
            "verified_level": None,
            "note": "Registration/report discovery does not establish execution; validate a current scorecard.",
        })
    return {"schema_version": 1, "baseline": git(root, "rev-parse", "HEAD"),
            "scope": "Shared ecosystem gate admission; separate adapter/unit gates are not claimed absent.",
            "repos": rows}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label, positive=False):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and (value > 0 if positive else value >= 0),
             "Invalid finite " + label)
    return value


def validate_record(record, inventories, baseline, root=ROOT):
    """Validate one actual oracle comparison. Unknown/missing metadata never passes."""
    _require(record.get("status") == "passed", "Only passed executions can claim a verified level")
    repo = record.get("repo")
    _require(repo in inventories, "Missing regenerated upstream inventory for " + str(repo))
    _require(record.get("jittor_commit") == baseline, "Stale Jittor baseline")
    _require(record.get("upstream") == inventories[repo]["upstream"], "Upstream SHA/version/remote mismatch")
    version = inventories[repo]["upstream"].get("version")
    versions = record.get("downstream_versions", {})
    _require(version and versions.get("candidate") == version and versions.get("oracle") == version,
             "Installed downstream versions do not match pinned upstream")
    level = record.get("level")
    _require(type(level) is int and 1 <= level <= 5, "Level must be L1..L5")
    _require(record.get("tier") in ("core", "broad", "tail"), "Missing sampling tier")
    references = record.get("registry_refs", [])
    _require(references, "Missing registered denominator member")
    for ref in references:
        _require(ref.get("repo") in inventories, "Unknown registry reference repo")
        axes = inventories[ref["repo"]]["axes"]
        _require(any(axis["axis"] == ref.get("axis") and ref.get("member") in axis["members"] for axis in axes),
                 "Unregistered model/method member")
    _require(any(ref.get("repo") == repo for ref in references), "Record has no member from its own repo")
    device = record.get("device")
    _require(device in ("cpu", "cuda", "npu", "rocm", "corex"), "Unknown requested device")
    _require(record.get("observed_device") == device, "Requested/observed device mismatch")
    platform = record.get("platform", {})
    _require(all(key in platform and platform[key] is not None for key in PLATFORM_FIELDS), "Incomplete platform block")
    flags = platform["jittor_flags"]
    _require(all(key in flags for key in FLAGS), "Incomplete Jittor flags")
    _require(flags["use_cuda"] == int(device != "cpu"), "CPU/CUDA flag mismatch")
    _require(flags["use_acl"] == int(device == "npu"), "CPU/NPU flag mismatch")
    if device == "cpu":
        _require(platform["card_count"] == 0 and platform["card_indices"] == [], "CPU record claims an accelerator")
    else:
        _require(type(platform["card_count"]) is int and platform["card_count"] > 0
                 and len(platform["card_indices"]) == platform["card_count"], "Missing allocated device mapping")
    oracle = platform["oracle"]
    _require(oracle.get("is_shim") is False and oracle.get("torch_version")
             and str(oracle.get("binary_extension", "")).endswith((".so", ".pyd", ".dylib"))
             and "torch_npu_version" in oracle,
             "Oracle must prove independent binary PyTorch identity")
    inputs = record.get("inputs", {})
    _require(type(inputs.get("seed")) is int and inputs.get("hf_hub_offline") is True
             and inputs.get("shape_kind") in ("tiny", "real"), "Unfixed input metadata")
    evidence = record.get("evidence", {})
    _require(evidence.get("command"), "Missing reproducible command")
    report = evidence.get("report", "")
    report_path = (root / report).resolve()
    _require(report.startswith("docs/results/") and root.resolve() in report_path.parents and report_path.is_file(),
             "Missing versioned results report")
    gate = evidence.get("gate_nodeid", "")
    gate_path = gate.split("::")[0]
    gate_file = (root / gate_path).resolve()
    _require(gate_path.endswith(".py") and root.resolve() in gate_file.parents and gate_file.is_file() and "::" in gate,
             "Missing concrete admitted gate node")
    names = {node.name for node in ast.walk(ast.parse(gate_file.read_text(encoding="utf-8")))
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    _require(all(part.split("[")[0] in names for part in gate.split("::")[1:]), "Unknown gate node selector")
    artifacts = evidence.get("artifacts", [])
    _require({artifact.get("role") for artifact in artifacts} >= {"reference", "candidate", "comparison", "platform"},
             "Missing raw reference/candidate/comparison/platform artifacts")
    for artifact in artifacts:
        path = Path(artifact.get("path", "")).expanduser().resolve()
        _require(path.is_file() and root.resolve() not in path.parents,
                 "Raw artifact missing or inside checkout")
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == artifact.get("sha256"), "Artifact SHA256 mismatch")
    if device != "cpu":
        fallback = record.get("fallback", {})
        _require(fallback.get("policy") == "error" and fallback.get("count") == 0,
                 "Accelerator comparisons require a zero-fallback guard")
    if level >= 2:
        numerical = record.get("numerics", {})
        _require(numerical.get("passed") is True and numerical.get("finite") is True
                 and numerical.get("checked_tensors", 0) > 0
                 and numerical.get("missing_tensors") == []
                 and numerical.get("input_perturbation_verified") is True,
                 "No complete numerical/non-no-op evidence")
        for label in ("max_abs", "max_scaled_relative", "atol", "rtol"):
            _number(numerical.get(label), label)
        _require(numerical["max_abs"] <= numerical["atol"]
                 or numerical["max_scaled_relative"] <= numerical["rtol"], "Numerical agreement failed")
        for axis in ("parameter_gradients", "input_gradients"):
            values = numerical.get(axis, {})
            _require(type(values.get("requested")) is int and values["requested"] >= 0
                     and values.get("checked") == values["requested"], "Missing " + axis)
    if level >= 3:
        training = record.get("training", {})
        _require(training.get("steps", 0) >= 2 and training.get("loss_trajectory_passed") is True
                 and training.get("parameter_trajectory_passed") is True, "L3 requires multi-step loss and parameter trajectories")
    if level >= 4:
        speed = record.get("speed", {})
        _require(inputs["shape_kind"] == "real", "Tiny configurations cannot establish L4")
        for key in ("model", "batch", "sequence", "dtype", "precision", "phase"):
            _require(key in speed, "Incomplete workload specification: " + key)
        for key in ("jittor_seconds", "oracle_seconds", "agreement", "agreement_tolerance"):
            _number(speed.get(key), key, positive=key.endswith("seconds"))
        _require(speed["agreement"] <= speed["agreement_tolerance"], "Speed measurement lacks numerical agreement")
        _require(speed.get("warmup", 0) >= 1 and speed.get("iterations", 0) >= 10
                 and speed.get("graph_regime") in ("eager", "default")
                 and speed.get("no_concurrent_work") is True
                 and speed.get("statistic") in ("minimum", "median"), "Unfixed measurement regime")
        memory = speed.get("memory", {})
        _require(memory.get("source") == {"cpu": "process_rss", "npu": "npu_smi_process"}.get(device, "nvml_process"),
                 "L4 needs whole-process peak memory on both runtimes")
        _number(memory.get("jittor_peak_bytes"), "jittor process peak", positive=True)
        _number(memory.get("oracle_peak_bytes"), "oracle process peak", positive=True)
        ratio = speed["jittor_seconds"] / speed["oracle_seconds"]
        _number(speed.get("ratio_limit"), "performance acceptance limit", positive=True)
        _require(ratio <= speed["ratio_limit"], "L4 performance acceptance limit exceeded")
        _require(speed["graph_regime"] == flags["graph_replay"], "Platform/measurement graph regime mismatch")
        _require(speed.get("ratio") is None or math.isclose(speed["ratio"], ratio, rel_tol=1e-9), "Speed ratio mismatch")
    if level >= 5:
        fallback = record.get("fallback", {})
        _require(device != "cpu" and fallback.get("policy") == "error" and fallback.get("count") == 0,
                 "L5 requires actual accelerator execution with zero fallback")
    return level


def scorecard(inventories, records, baseline, root=ROOT):
    accepted = []
    rejected = []
    for index, record in enumerate(records):
        try:
            level = validate_record(record, inventories, baseline, root)
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({"index": index, "case": record.get("case"), "reason": str(error)})
            continue
        accepted.append(dict(record, verified_level=level))
    def scenario(row):
        return (row["repo"], row.get("case"), json.dumps(row["registry_refs"], sort_keys=True))
    cpu = {scenario(row) for row in accepted if row["device"] == "cpu" and row["verified_level"] >= 2}
    coverage = []
    for repo, source in inventories.items():
        for axis in source["axes"]:
            for device in ("cpu", "cuda", "npu", "rocm", "corex"):
                verified = set()
                for row in accepted:
                    if row["device"] != device or row["verified_level"] < 2:
                        continue
                    if device != "cpu" and scenario(row) not in cpu:
                        continue
                    for ref in row["registry_refs"]:
                        if ref["repo"] == repo and ref["axis"] == axis["axis"]:
                            verified.add(ref["member"])
                coverage.append({"repo": repo, "axis": axis["axis"], "device": device,
                                 "verified_members": sorted(verified), "numerator": len(verified),
                                 "denominator": axis["count"], "upstream_commit": source["upstream"]["commit"]})
    return {"schema_version": 1, "baseline": baseline, "inventories": inventories,
            "coverage": coverage, "accepted": accepted, "rejected": rejected,
            "tiers": {tier: sum(row["tier"] == tier for row in accepted) for tier in ("core", "broad", "tail")},
            "note": "L1 is not counted. Accelerator breadth requires the matching independent CPU oracle comparison. A representative case never proves all checkpoints in its family."}


def render_audit(value):
    rows = ["| Repo | Ecosystem gate cases | Existing reports | Level established by this audit |",
            "| --- | --- | --- | --- |"]
    for row in value["repos"]:
        rows.append("| %s | %s | %d | Not established |" %
                    (row["repo"], ", ".join(item["case"] for item in row["gate_cases"]) or "0", len(row["reports"])))
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    discovery = commands.add_parser("audit")
    discovery.add_argument("--json", action="store_true")
    discovery.add_argument("--root", default=str(ROOT))
    scores = commands.add_parser("scorecard")
    scores.add_argument("--upstream", action="append", default=[], metavar="REPO=CHECKOUT", required=True)
    scores.add_argument("--evidence", action="append", default=[])
    scores.add_argument("--output", required=True)
    options = parser.parse_args()
    if options.command == "audit":
        value = audit(Path(options.root))
        print(json.dumps(value, indent=2, ensure_ascii=False) if options.json else render_audit(value))
    elif options.command == "scorecard":
        sources = {}
        for item in options.upstream:
            repo, path = item.split("=", 1)
            if repo in sources:
                raise ValueError("Duplicate upstream source: " + repo)
            sources[repo] = inventory(repo, path)
        records = []
        for path in options.evidence:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
            _require(document.get("schema_version") == 1, "Unknown evidence schema")
            records.extend(document["records"])
        value = scorecard(sources, records, git(ROOT, "rev-parse", "HEAD"))
        write_json(options.output, value)
        for row in value["coverage"]:
            if row["device"] == "cpu":
                print("%s %s: %d / %d @ %s" % (row["repo"], row["axis"], row["numerator"], row["denominator"], row["upstream_commit"]))
        if value["rejected"]:
            raise SystemExit("Rejected %d scorecard record(s); see output" % len(value["rejected"]))
    else:
        parser.error("choose audit or scorecard")


if __name__ == "__main__":
    main()
