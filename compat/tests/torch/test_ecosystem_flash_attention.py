"""Independent upstream FlashAttention training and controlled input perturbations."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from _helpers.child_process import child_env, run_python_child
from _helpers.artifacts import artifact_directory


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "agent/skills/flash-attention-torch-compat/scripts"
APIS = (
    "flash_attn_func", "flash_attn_kvpacked_func", "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func", "flash_attn_varlen_kvpacked_func", "flash_attn_varlen_qkvpacked_func",
)


@pytest.fixture(scope="module")
def comparison():
    spec = importlib.util.spec_from_file_location("flash_training_comparison", SCRIPTS / "compare_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configured():
    device = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
    require_flash = os.environ.get("JITTOR_REQUIRE_FLASH_ATTENTION", "").lower() in ("1", "true", "yes", "on")
    if device == "cpu":
        if require_flash:
            pytest.fail("Required FlashAttention training needs actual CUDA")
        pytest.skip("Upstream FlashAttention CUDA training has no CPU execution")
    assert device == "cuda", "FlashAttention needs exactly JITTOR_TEST_DEVICES=cuda"
    names = ("REAL_TORCH_PYTHON", "JITTOR_FLASH_ATTN_JITTOR_SRC",
             "JITTOR_FLASH_ATTN_ORACLE_EXTENSION", "JITTOR_FLASH_ATTN_ORACLE_CUDA_HOME")
    values = {name: os.environ.get(name, "").strip() for name in names}
    for name in ('JITTOR_FLASH_ATTN_SOURCE_PROOF', 'JITTOR_FLASH_ATTN_ORACLE_STATE'):
        if os.environ.get(name):
            values[name] = os.environ[name]
    missing = [name for name, value in values.items() if not value or not Path(value).exists()]
    if missing:
        required = any(os.environ.get(name, "").lower() in ("1", "true", "yes", "on")
                       for name in ("JITTOR_REQUIRE_REAL_TORCH", "JITTOR_REQUIRE_FLASH_ATTENTION"))
        if required:
            pytest.fail("FlashAttention requires configured " + ", ".join(missing))
        pytest.skip("Independent Torch CUDA FlashAttention environment is not configured")
    return values


def _run(runtime, values, api, dtype, perturbation, output):
    environment = dict(os.environ)
    arguments = [str(SCRIPTS / "flash_training_probe.py"), "--runtime", runtime,
                 "--upstream", values["JITTOR_FLASH_ATTN_JITTOR_SRC"], "--api", api,
                 "--dtype", dtype, "--perturbation", str(perturbation), "--output", str(output)]
    if values.get('JITTOR_FLASH_ATTN_SOURCE_PROOF'):
        arguments.extend(('--source-proof', values['JITTOR_FLASH_ATTN_SOURCE_PROOF']))
    if runtime == "jittor":
        environment["JITTOR_FLASH_ATTN_JITTOR_REQUIRED"] = "1"
        completed = run_python_child(arguments, env=environment, inherit=False,
                                     merge_stderr=True, timeout=1800)
    else:
        environment = {key: value for key, value in environment.items()
                       if not key.startswith(("JT_", "JITTOR_"))}
        environment.pop("LD_LIBRARY_PATH", None)
        extension = Path(values["JITTOR_FLASH_ATTN_ORACLE_EXTENSION"]).resolve()
        environment["PYTHONPATH"] = ('' if values.get('JITTOR_FLASH_ATTN_SOURCE_PROOF') else
                                     os.pathsep.join((str(extension.parent), values["JITTOR_FLASH_ATTN_JITTOR_SRC"])))
        if values.get('JITTOR_FLASH_ATTN_ORACLE_STATE'):
            state = Path(values['JITTOR_FLASH_ATTN_ORACLE_STATE'])
            for name, leaf in (('HOME', 'home'), ('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg-cache')):
                path = state / leaf
                path.mkdir(parents=True, exist_ok=True)
                environment[name] = str(path)
        environment["CUDA_HOME"] = values["JITTOR_FLASH_ATTN_ORACLE_CUDA_HOME"]
        completed = subprocess.run(
            [values["REAL_TORCH_PYTHON"], *arguments],
            env=child_env(environment, inherit=False, repo_paths=False, without_torch_mode=True),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
    output.with_suffix(".log").write_text(completed.stdout, encoding="utf-8")
    assert completed.returncode == 0, completed.stdout[-12000:]
    assert output.is_file() and output.with_suffix(".json").is_file()


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_real_flash_attention_training(api, dtype, tmp_path, comparison):
    values = _configured()
    evidence = os.environ.get("JITTOR_FLASH_ATTN_EVIDENCE_DIR", "").strip()
    output = Path(evidence) / (api + '-' + dtype) if evidence else None
    with artifact_directory(ROOT, output=output, prefix='flash-training-') as directory:
        _exercise_training(api, dtype, values, directory, comparison)


def _exercise_training(api, dtype, values, directory, comparison):
    metadata = {}
    for label, perturbation in (("base", 0.0), ("perturbed", 0.125)):
        paths = {}
        for runtime in ("torch", "jittor"):
            paths[runtime] = directory / (runtime + "-" + label + ".npz")
            _run(runtime, values, api, dtype, perturbation, paths[runtime])
            metadata[runtime, label] = json.loads(paths[runtime].with_suffix(".json").read_text())
        result = comparison.compare(paths["torch"], paths["jittor"])
        result["baseline"] = metadata["jittor", label]["baseline"]
        result["cases"] = [{"case": metadata["jittor", label]["measurement"]["case"],
                            "status": result["status"],
                            "oracle_result": metadata["torch", label]["measurement"],
                            "jittor_result": metadata["jittor", label]["measurement"]}]
        (directory / (label + "-comparison.json")).write_text(json.dumps(result, indent=2) + "\n")
        assert result["status"] == "passed", result["tensors"]
    for runtime in ("torch", "jittor"):
        base, changed = metadata[runtime, "base"], metadata[runtime, "perturbed"]
        for name in ("q", "k", "v", "target"):
            assert base["input_hashes"][name] == changed["input_hashes"][name], name
        assert base["input_hashes"]["noise"] != changed["input_hashes"]["noise"]
        with np.load(directory / (runtime + "-base.npz")) as before, np.load(
                directory / (runtime + "-perturbed.npz")) as after:
            assert any(not np.array_equal(before[f"step{step}_output"], after[f"step{step}_output"])
                       for step in range(3)), "input perturbation did not affect attention output"
    (directory / "perturbation.json").write_text(json.dumps({
        "api": api, "dtype": dtype, "input_perturbation_verified": True,
        "changed_input": "noise", "fixed_inputs": ["q", "k", "v", "target"],
    }, indent=2) + "\n")
    for runtime, label in (("torch", "reference"), ("jittor", "candidate")):
        (directory / ("training-" + label + "-metadata.json")).write_text(
            json.dumps(metadata[runtime, "base"]["measurement"], indent=2) + "\n")
    (directory / "perturb-comparison.json").write_text(json.dumps({
        "case": metadata["jittor", "base"]["measurement"]["case"], "status": "passed",
        "input_perturbation_verified": True,
        "reference_result": metadata["torch", "perturbed"]["measurement"],
        "candidate_result": metadata["jittor", "perturbed"]["measurement"],
    }, indent=2) + "\n")
