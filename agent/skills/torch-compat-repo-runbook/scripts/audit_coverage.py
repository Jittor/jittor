"""Audit runbooks and build fail-closed, registry-denominated downstream scorecards.

The audit is source-only. A skill, registered case, or historical Markdown report
is discoverable evidence, never a passed execution. Scorecards count only fresh,
validated records with pinned source inventories and external raw-artifact hashes.
"""

import argparse
import ast
from email.parser import BytesParser
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

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


def dedicated_gates(root):
    """Discover maintained non-model protocols without counting ordinary CASES."""
    declarations = {
        "deepspeed": ("deepspeed-engine", "ecosystem_deepspeed",
                      "compat/tests/torch/test_ecosystem_deepspeed.py",
                      "test_real_deepspeed_engine_training_and_resume"),
        "trellis": ("trellis-models", "ecosystem_trellis",
                    "compat/tests/torch/test_ecosystem_trellis.py",
                    "test_real_trellis_models_training"),
        "flash-attention": ("flash-attention-training", "ecosystem_flash_attention",
                            "compat/tests/torch/test_ecosystem_flash_attention.py",
                            "test_real_flash_attention_training"),
    }
    session_tree = ast.parse((root / "noxfile.py").read_text(encoding="utf-8"))
    sessions = {node.name for node in session_tree.body if isinstance(node, ast.FunctionDef)}
    result = {}
    for repo, (protocol, session, relative, method) in declarations.items():
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        methods = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        if session not in sessions or method not in methods:
            raise ValueError("Maintained dedicated gate entry is missing: " + repo)
        result[repo] = [{"protocol": protocol, "session": session,
                         "tests": [relative + "::" + method]}]
    return result


def audit(root=ROOT):
    root = Path(root).resolve()
    cases = ecosystem_cases(root)
    dedicated = dedicated_gates(root)
    gate_files = ("compat/tests/torch/test_ecosystem_parity.py",
                  "compat/tests/torch/test_ecosystem_speed.py",
                  "compat/tests/torch/test_ecosystem_verl.py",
                  "compat/tests/torch/test_ecosystem_longformer.py",
                  "compat/tests/torch/test_ecosystem_openmmlab.py",
                  "compat/tests/torch/test_ecosystem_vllm.py")
    methods = {}

    def collect_methods(body, relative, parents=()):
        for node in body:
            if isinstance(node, ast.ClassDef):
                collect_methods(node.body, relative, parents + (node.name,))
            elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                selector = "::".join((relative,) + parents + (node.name,))
                methods.setdefault(node.name[5:], []).append(selector)

    for relative in gate_files:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        collect_methods(tree.body, relative)
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
            "dedicated_gates": dedicated.get(repo, []),
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
            "scope": "Shared model cases and dedicated protocol gate admission; separate adapter/unit gates are not claimed absent.",
            "repos": rows}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label, positive=False):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and (value > 0 if positive else value >= 0),
             "Invalid finite " + label)
    return value


def _validate_source_wheel(record, source, artifacts):
    """Prove an SCM distribution matches source without renaming its version."""
    wheels = [item for item in artifacts if item.get("role") == "upstream-wheel"]
    _require(len(wheels) == 1, "Source-only installed versions require one upstream-wheel artifact")
    artifact = wheels[0]
    path = Path(artifact["path"]).expanduser()
    _require(path.suffix == ".whl", "Upstream distribution artifact must be a wheel")
    sources = source.get("package_sources", [])
    _require(sources and len({item["path"] for item in sources}) == len(sources),
             "Source-only wheel proof requires unique regenerated full-package source hashes")
    generated = {}
    try:
        with zipfile.ZipFile(path) as wheel:
            names = wheel.namelist()
            _require(len(names) == len(set(names)), "Duplicate upstream wheel members")
            metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
            _require(len(metadata) == 1, "Expected exactly one upstream wheel METADATA")
            package = BytesParser().parsebytes(wheel.read(metadata[0]))
            def normalize(value):
                return re.sub(r"[-_.]+", "-", value).lower()
            _require(package.get("Name") and normalize(package["Name"]) == normalize(record["repo"]),
                     "Upstream wheel distribution name mismatch")
            version = record["downstream_versions"]["candidate"]
            _require(package.get_all("Name") == [package["Name"]]
                     and package.get_all("Version") == [version], "Upstream wheel version mismatch")
            for item in sources:
                relative = item["path"]
                member = relative[4:] if relative.startswith("src/") else relative
                _require(member in names and hashlib.sha256(wheel.read(member)).hexdigest() == item["sha256"],
                         "Upstream wheel source mismatch: " + relative)
            tracked = {item["path"][4:] if item["path"].startswith("src/") else item["path"]
                       for item in sources}
            generated = {name: hashlib.sha256(wheel.read(name)).hexdigest()
                         for name in names if name.endswith(".py") and name not in tracked}
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("Invalid upstream wheel artifact") from error
    return {"repo": record["repo"], "version": version, "wheel_sha256": artifact["sha256"],
            "source_sha256": {item["path"]: item["sha256"] for item in sources},
            "generated_source_sha256": generated}


def _validate_measurements(record, level, artifacts, installation=None):
    """Bind the platform to hashed, actual measurement JSON, not a copied label."""
    by_role = {}
    for artifact in artifacts:
        role = artifact["role"]
        _require(role not in by_role, "Duplicate artifact role: " + role)
        by_role[role] = artifact

    def document(role):
        _require(role in by_role, "Missing measurement artifact: " + role)
        try:
            value = json.loads(Path(by_role[role]["path"]).expanduser().read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError("Invalid measurement JSON for " + role) from error
        _require(isinstance(value, dict), "Measurement JSON must be an object: " + role)
        return value

    platform = record["platform"]
    _require(document("platform") == platform, "Platform artifact does not match record")
    expected = platform.get("precision")
    _require(isinstance(expected, dict), "Missing platform measurement precision")
    _require(all(type(expected.get(key)) is bool for key in ("matmul", "cudnn", "cudnn_benchmark"))
             and expected.get("matmul_precision") in ("highest", "high", "medium"),
             "Incomplete platform measurement precision")
    _require(platform["jittor_flags"]["tf32"] == expected["matmul"],
             "Platform flag/measurement TF32 precision mismatch")

    def pair(reference, candidate):
        for value in (reference, candidate):
            _require(isinstance(value, dict) and value.get("case") == record["case"],
                     "Measurement case mismatch")
            _require(value.get("device") == record["device"], "Measurement actual device mismatch")
            _require(value.get("tf32") == expected, "Platform/measurement precision mismatch")
            _require(value.get("runtime_conditions", {}).get("precision") == expected,
                     "Measurement runtime precision mismatch")
            if installation is not None:
                _require(value.get("downstream_distribution") == installation,
                         "Measurement installed distribution/source binding mismatch")
        _require(reference["runtime_conditions"] == candidate["runtime_conditions"],
                 "Oracle/candidate runtime conditions mismatch")
        backend = candidate.get("backend", {})
        _require(backend.get("use_cuda") == bool(platform["jittor_flags"]["use_cuda"])
                 and backend.get("use_acl") == bool(platform["jittor_flags"]["use_acl"]),
                 "Measurement backend flags mismatch")
        _require(candidate.get("fallback_policy") == "error" and candidate.get("fallback_count") == 0,
                 "Measurement zero-fallback guard missing")

    base = document("base-comparison")
    _require(base.get("baseline") == record["jittor_commit"], "Measurement baseline mismatch")
    cases = [value for value in base.get("cases", []) if value.get("case") == record["case"]]
    _require(len(cases) == 1 and cases[0].get("status") == "passed", "Missing passed base measurement")
    pair(cases[0].get("oracle_result"), cases[0].get("jittor_result"))
    perturb = document("perturb-comparison")
    _require(perturb.get("case") == record["case"] and perturb.get("status") == "passed"
             and perturb.get("input_perturbation_verified") is True, "Missing passed perturbation measurement")
    pair(perturb.get("reference_result"), perturb.get("candidate_result"))
    if level >= 3:
        reference = document("training-reference-metadata")
        candidate = document("training-candidate-metadata")
        pair(reference, candidate)
        _require(reference.get("runtime") == "torch" and candidate.get("runtime") == "jittor",
                 "Training runtime identity mismatch")
        training = record["training"]
        for value in (reference, candidate):
            _require(value.get("jittor_commit") == record["jittor_commit"], "Training measurement baseline mismatch")
            _require(value.get("steps") == training["steps"], "Training measurement step mismatch")
            for measured, declared, count in (("trainable_parameters", "trainable_parameters", "trainable"),
                                               ("frozen_parameters", "frozen_parameters", "frozen")):
                names = value.get(measured)
                _require(isinstance(names, list) and all(isinstance(name, str) for name in names)
                         and len(names) == len(set(names)), "Missing actual training parameter scope")
                _require(training.get(declared) == names and training.get(count) == len(names),
                         "Declared/measurement training parameter scope mismatch")


def validate_record(record, inventories, baseline, root=ROOT):
    """Validate one actual oracle comparison. Unknown/missing metadata never passes."""
    _require(record.get("status") == "passed", "Only passed executions can claim a verified level")
    repo = record.get("repo")
    _require(repo in inventories, "Missing regenerated upstream inventory for " + str(repo))
    _require(record.get("jittor_commit") == baseline, "Stale Jittor baseline")
    _require(record.get("upstream") == inventories[repo]["upstream"], "Upstream SHA/version/remote mismatch")
    version = inventories[repo]["upstream"].get("version")
    versions = record.get("downstream_versions", {})
    source_distribution = (version == "source@" + inventories[repo]["upstream"]["commit"]
                           and isinstance(versions.get("candidate"), str)
                           and bool(versions["candidate"]) and not versions["candidate"].startswith("source@")
                           and versions.get("oracle") == versions["candidate"])
    _require(version and ((versions.get("candidate") == version and versions.get("oracle") == version)
                          or source_distribution),
             "Installed downstream versions do not match pinned upstream")
    level = record.get("level")
    _require(type(level) is int and 1 <= level <= 5, "Level must be L1..L5")
    _require(record.get("tier") in ("core", "broad", "tail"), "Missing sampling tier")
    kind = record.get("record_kind", "registry")
    _require(kind in ("registry", "protocol"), "Unknown scorecard record_kind")
    if kind == "protocol":
        from _protocol_scorecards import validate_spec
        validate_spec(record, inventories[repo])
        _require(level in (2, 3), "Protocol scorecards admit only independent L2/L3 evidence")
    references = record.get("registry_refs", [])
    _require(kind == "protocol" or references, "Missing registered denominator member")
    for ref in references:
        _require(ref.get("repo") in inventories, "Unknown registry reference repo")
        axes = inventories[ref["repo"]]["axes"]
        _require(any(axis["axis"] == ref.get("axis") and ref.get("member") in axis["members"] for axis in axes),
                 "Unregistered model/method member")
    _require(kind == "protocol" or any(ref.get("repo") == repo for ref in references), "Record has no member from its own repo")
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
    installation = (_validate_source_wheel(record, inventories[repo], artifacts)
                    if source_distribution else None)
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
    if level >= 2:
        if kind == "protocol":
            from _protocol_scorecards import validate_measurements
            validate_measurements(record, artifacts, inventories[repo])
        else:
            _validate_measurements(record, level, artifacts, installation)
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
    protocol_rows = [row for row in accepted if row.get("record_kind") == "protocol"]
    blocked = []
    if protocol_rows:
        from _protocol_scorecards import pairing_key as protocol_scenario
        protocol_cpu = {protocol_scenario(row) for row in protocol_rows
                        if row["device"] == "cpu" and row["verified_level"] >= 2}
        for row in protocol_rows:
            if row["device"] != "cpu" and protocol_scenario(row) not in protocol_cpu:
                blocked.append(dict(row, reported_status=row["status"], status="blocked", level=None,
                                    verified_level=None, independent_verified_level=row["verified_level"],
                                    reason="Missing verified CPU counterpart for this exact protocol"))
        accepted = [row for row in accepted if row.get("record_kind") != "protocol"
                    or row["device"] == "cpu" or protocol_scenario(row) in protocol_cpu]
    registry_accepted = [row for row in accepted if row.get("record_kind") != "protocol"]
    def scenario(row):
        return (row["repo"], row.get("case"), json.dumps(row["registry_refs"], sort_keys=True))
    cpu = {scenario(row) for row in registry_accepted if row["device"] == "cpu" and row["verified_level"] >= 2}
    coverage = []
    for repo, source in inventories.items():
        for axis in source["axes"]:
            for device in ("cpu", "cuda", "npu", "rocm", "corex"):
                verified = set()
                scenarios = []
                for row in registry_accepted:
                    if row["device"] != device or row["verified_level"] < 2:
                        continue
                    if device != "cpu" and scenario(row) not in cpu:
                        continue
                    matching = [ref for ref in row["registry_refs"]
                                if ref["repo"] == repo and ref["axis"] == axis["axis"]]
                    verified.update(ref["member"] for ref in matching)
                    if matching:
                        training = row.get("training", {})
                        scenarios.append({"case": row["case"], "repo": row["repo"], "level": row["verified_level"],
                                          "registry_refs": row["registry_refs"], "shape_kind": row["inputs"]["shape_kind"],
                                          "trainable_scope": {
                                              "trainable": training.get("trainable"), "frozen": training.get("frozen"),
                                              "trainable_parameters": training.get("trainable_parameters"),
                                              "frozen_parameters": training.get("frozen_parameters")}})
                coverage.append({"repo": repo, "axis": axis["axis"], "device": device,
                                 "verified_members": sorted(verified), "verified_scenarios": scenarios, "numerator": len(verified),
                                 "denominator": axis["count"], "upstream_commit": source["upstream"]["commit"]})
    value = {"schema_version": 2, "baseline": baseline, "inventories": inventories,
            "coverage": coverage, "accepted": accepted, "rejected": rejected,
            "tiers": {tier: sum(row["tier"] == tier for row in accepted) for tier in ("core", "broad", "tail")},
            "note": "L1 is not counted. Accelerator breadth requires the matching independent CPU oracle comparison. A representative case never proves all checkpoints in its family."}
    if any(row.get("record_kind") == "protocol" for row in records):
        from _protocol_scorecards import pairing_key as protocol_scenario
        value["blocked"] = blocked
        value["protocols"] = [{"repo": row["repo"], "case": row["case"], "device": row["device"],
                               "protocol": row["protocol"],
                               "level": row["verified_level"] if protocol_scenario(row) in protocol_cpu else None,
                               "independent_verified_level": row["verified_level"],
                               "status": "passed" if protocol_scenario(row) in protocol_cpu else "blocked",
                               "reason": None if protocol_scenario(row) in protocol_cpu else "Missing verified CPU counterpart for this exact protocol",
                               "matching_cpu_verified": protocol_scenario(row) in protocol_cpu,
                               "registry_coverage": "not-applicable"}
                              for row in protocol_rows]
        for failure in rejected:
            row = records[failure["index"]]
            if row.get("record_kind") == "protocol":
                value["protocols"].append({"repo": row.get("repo"), "case": row.get("case"), "device": row.get("device"),
                                           "protocol": row.get("protocol"), "level": None, "status": "rejected",
                                           "reported_status": row.get("status"), "reason": failure["reason"],
                                           "reported_reason": row.get("reason"),
                                           "matching_cpu_verified": False, "registry_coverage": "not-applicable"})
    return value


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
    scores.add_argument("--baseline", help="Full historical Jittor SHA; explicitly audits that commit, not current HEAD")
    options = parser.parse_args()
    if options.command == "audit":
        value = audit(Path(options.root))
        print(json.dumps(value, indent=2, ensure_ascii=False) if options.json else render_audit(value))
    elif options.command == "scorecard":
        baseline = git(ROOT, "rev-parse", "HEAD")
        if options.baseline:
            _require(re.fullmatch(r"[0-9a-f]{40}", options.baseline) is not None, "Historical baseline requires a full SHA")
            _require(git(ROOT, "rev-parse", options.baseline + "^{commit}") == options.baseline,
                     "Unknown historical Jittor baseline")
            baseline = options.baseline
        sources = {}
        for item in options.upstream:
            repo, path = item.split("=", 1)
            if repo in sources:
                raise ValueError("Duplicate upstream source: " + repo)
            sources[repo] = inventory(repo, path)
        records = []
        for path in options.evidence:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
            _require(document.get("schema_version") == 2, "Unknown evidence schema; regenerate hashed measurement bindings for v2")
            records.extend(document["records"])
        value = scorecard(sources, records, baseline)
        value["auditor_commit"] = git(ROOT, "rev-parse", "HEAD")
        value["baseline_policy"] = "explicit-historical" if options.baseline else "current-head"
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
