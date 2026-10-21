#!/usr/bin/env python3
"""List performance suites and source-declared workloads without importing frameworks.

This is a discovery tool, not a benchmark runner or a device validation report.
Suite descriptions belong here; workload inventories belong to their source suites.
"""

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
SCOPE_NOTE = "Devices describe source-declared scope, not hardware validation. CI selects only part of ASV."

# Paths are checked on every listing. Keep descriptions in sync when a suite's
# owner changes its entrypoints or execution protocol; names/parameters below
# are discovered from sources, never copied into this registry.
SUITES = (
    {
        "id": "asv",
        "purpose": "Track operator, model, optimizer and transfer regressions across commits.",
        "entrypoints": ["benchmarks/asv/asv.conf.json"],
        "references": ["noxfile.py", ".github/workflows/cpu.yml", ".github/workflows/cuda.yml"],
        "role": "Selected CPU CI and scheduled/manual CUDA CI; other rows are manual.",
        "declared_devices": ["cpu", "cuda"],
        "runtimes": ["jittor", "optional binary PyTorch", "isolated Jittor Torch for Accelerate"],
        "notes": "ASV setup may skip unavailable optional dependencies/devices; inventory is not a run result.",
    },
    {
        "id": "torch-compat",
        "purpose": "Compare identical real-model workloads in independent Jittor Torch and PyTorch processes.",
        "entrypoints": ["benchmarks/torch_compat/run.py"],
        "references": ["benchmarks/torch_compat/README.md", "benchmarks/torch_compat/workloads.py"],
        "role": "Manual, with per-run results and reports.",
        "runtimes": ["jittor Torch frontend", "binary PyTorch (+ torch_npu for NPU)"],
    },
    {
        "id": "acl",
        "purpose": "Compare native Ascend models, replay training numerics and probe CANN launch costs.",
        "entrypoints": ["benchmarks/acl/bench_jittor.py", "benchmarks/acl/bench_torch.py"],
        "references": ["benchmarks/acl/README.md", "benchmarks/acl/models.md", "benchmarks/acl/xval", "benchmarks/acl/probes"],
        "role": "Manual; real Ascend and CANN required.",
        "declared_devices": ["npu"],
        "runtimes": ["native Jittor ACL", "binary PyTorch + torch_npu", "direct CANN probes"],
    },
    {
        "id": "ecosystem-speed",
        "purpose": "Measure realistic-size ecosystem forward/backward cases against an independent oracle.",
        "entrypoints": ["compat/tests/torch/test_ecosystem_speed.py"],
        "references": ["compat/tests/torch/_ecosystem_speed.py"],
        "role": "Opt-in pytest measurement; ratio gate only when JITTOR_ECOSYSTEM_SPEED_RATIO is set.",
        "declared_devices": ["cpu", "cuda"],
        "runtimes": ["jittor Torch frontend", "binary PyTorch"],
        "notes": "Owned by compat tests; requires REAL_TORCH_PYTHON and JITTOR_ECOSYSTEM_LARGE=1.",
    },
    {
        "id": "graph-build",
        "purpose": "Separate host graph construction phases from completed-step execution.",
        "entrypoints": ["tools/benchmarks/graph_build_phases.py"],
        "references": [],
        "role": "Manual diagnostic; phase counters require a profiling build.",
        "runtimes": ["jittor Torch frontend"],
    },
    {
        "id": "profiling",
        "purpose": "Find operator, training, kernel bandwidth and cold-build bottlenecks.",
        "entrypoints": [
            "agent/skills/jittor-transformers-perf/SKILL.md",
            "agent/skills/cuda-elementwise-bandwidth-roofline/SKILL.md",
            "agent/skills/jittor-single-kernel-cold-profile/SKILL.md",
        ],
        "references": [],
        "role": "Manual, individually scoped diagnostic tools; follow each skill's environment protocol.",
        "declared_devices": ["cpu", "cuda", "npu"],
        "runtimes": ["tool-specific Jittor/native or Torch frontend", "optional binary PyTorch"],
        "notes": "Legacy tools have distinct frontend assumptions; they are not an interchangeable validated suite.",
    },
)


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assignments(body):
    return {
        node.targets[0].id: node.value
        for node in body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }


def _literal(node, constants):
    """Resolve literal axes and named literal constants without executing source."""
    if isinstance(node, ast.Name):
        if node.id not in constants:
            raise ValueError("catalog cannot resolve ASV parameter constant: " + node.id)
        return _literal(constants[node.id], constants)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(value, constants) for value in node.elts]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("list", "tuple") and len(node.args) == 1
            and not node.keywords):
        return list(_literal(node.args[0], constants))
    return ast.literal_eval(node)


def asv_workloads(root):
    rows = []
    for path in sorted((root / "benchmarks/asv").glob("*.py")):
        tree = _tree(path)
        constants = _assignments(tree.body)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            methods = [item.name for item in node.body if isinstance(item, ast.FunctionDef)
                       and item.name.startswith(("time_", "track_", "peakmem_", "mem_"))]
            if not methods:
                continue
            fields = dict(constants, **_assignments(node.body))
            names = _literal(fields["param_names"], fields) if "param_names" in fields else []
            axes = _literal(fields["params"], fields) if "params" in fields else []
            if len(names) != len(axes):
                raise ValueError("ASV parameter names/axes mismatch: " + str(path))
            parameters = dict(zip(names, axes))
            for method in methods:
                rows.append({
                    "name": "%s.%s.%s" % (path.stem, node.name, method),
                    "source": path.relative_to(root).as_posix(),
                    "parameters": parameters,
                })
    if not rows:
        raise ValueError("No ASV workloads found")
    return rows


def _argument_value(path, option, keyword):
    for node in ast.walk(_tree(path)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and ast.literal_eval(node.args[0]) == option):
            for value in node.keywords:
                if value.arg == keyword:
                    return ast.literal_eval(value.value)
    raise ValueError("No literal %s for %s in %s" % (keyword, option, path))


def catalog(root=REPO, selected=None):
    suites = []
    for template in SUITES:
        if selected is not None and template["id"] != selected:
            continue
        suite = dict(template)
        for path in suite["entrypoints"] + suite["references"]:
            if not (root / path).exists():
                raise FileNotFoundError("Stale benchmark catalog path: " + path)
        if suite["id"] == "asv":
            suite["workloads"] = asv_workloads(root)
            suite["declared_devices"] = sorted({
                device for row in suite["workloads"]
                for device in row["parameters"].get("device", [])
            })
        elif suite["id"] == "torch-compat":
            entry = root / suite["entrypoints"][0]
            # Reuse the suite's own host-only inventory, in a process that
            # excludes site packages. No framework/JIT/model initialization.
            result = subprocess.run([sys.executable, "-S", str(entry), "--list", "--json"],
                                    check=True, stdout=subprocess.PIPE, text=True)
            suite["workloads"] = json.loads(result.stdout)
            suite["declared_devices"] = list(_argument_value(entry, "--device", "choices"))
        elif suite["id"] == "acl":
            inventories = [_argument_value(root / path, "--models", "default").split(",")
                           for path in suite["entrypoints"]]
            if inventories[0] != inventories[1]:
                raise ValueError("ACL default model inventories differ between runtimes")
            suite["workloads"] = [{"name": name} for name in inventories[0]]
        elif suite["id"] == "ecosystem-speed":
            fields = _assignments(_tree(root / suite["entrypoints"][0]).body)
            suite["workloads"] = [{"name": name} for name in ast.literal_eval(fields["SPEED_CASES"])]
        elif suite["id"] == "graph-build":
            entry = root / suite["entrypoints"][0]
            cases = _assignments(_tree(entry).body)["CASES"]
            suite["workloads"] = [{"name": ast.literal_eval(key)} for key in cases.keys]
            suite["declared_devices"] = list(_argument_value(entry, "--device", "choices"))
        else:
            scripts = []
            for entry in suite["entrypoints"]:
                scripts.extend((root / entry).parent.rglob("*.py"))
            suite["workloads"] = [
                {"name": path.stem, "source": path.relative_to(root).as_posix()}
                for path in sorted(scripts) if not path.name.startswith("_")
            ]
        suites.append(suite)
    return {"schema_version": 1, "scope_note": SCOPE_NOTE, "suites": suites}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=[suite["id"] for suite in SUITES])
    parser.add_argument("--json", action="store_true", help="emit a machine-readable catalog")
    options = parser.parse_args()
    result = catalog(selected=options.suite)
    if options.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print(result["scope_note"])
    for suite in result["suites"]:
        print("\n%s [%s]" % (suite["id"], ", ".join(suite["declared_devices"])))
        print("  " + suite["purpose"])
        print("  " + suite["role"])
        print("  entry: " + ", ".join(suite["entrypoints"]))
        print("  runtime: " + ", ".join(suite["runtimes"]))
        for row in suite["workloads"]:
            print("    " + row["name"])
        if suite.get("notes"):
            print("  note: " + suite["notes"])


if __name__ == "__main__":
    main()
