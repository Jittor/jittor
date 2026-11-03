"""Offline acceptance guards for matched official Trainer throughput rows."""

import ast
import copy
import importlib.util
from pathlib import Path
import types

import pytest


@pytest.fixture
def probe(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "agent/skills/torch-compat-repo-runbook/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "throughput_test_owner", scripts / "trl_throughput_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(probe, precision="fp32", runtime="torch", repeat=0):
    hardware = {
        "uuid": "device1",
        "name": "GPU",
        "driver_version": "driver",
        "clocks.max.sm": "1500",
        "clocks.applications.graphics": "N/A",
        "power.limit": "250",
        "clocks.current.sm": "1400",
    }
    return dict(
        status="complete",
        mode="measure",
        precision=precision,
        runtime=runtime,
        repeat=repeat,
        fallback_count=0,
        optimizer_calls=25,
        global_step=25,
        sequence_lengths=probe.LENGTHS,
        timing=dict(
            warmup_optimizer_calls=5,
            timed_optimizer_calls=20,
            seconds=2.0 if runtime == "torch" else 4.0,
            jit_changes={},
            counts=probe.token_counts(probe.LENGTHS),
        ),
        verification=dict(
            passed=True, all112_parameter_states_finite=True, all112_parameter_counters=25
        ),
        hardware_before=hardware,
        hardware_after=copy.deepcopy(hardware),
        protocol={"precision": precision},
        versions={},
        initial={},
        initial_adapters_sha256="adapters",
        initial_buffers_sha256="buffers",
        dataset_sha256="data",
        source_files={},
        cpu_affinity=[0, 1, 2, 3],
        model_config_sha256="config",
        trl_source={},
        runtime_sources={"owner": "stable"},
        compiled_core={"path": runtime + "/core.so", "sha256": runtime + "-core"},
    )


def test_real_unequal_lengths_have_exact_timed_token_counts(probe):
    assert probe.token_counts(probe.LENGTHS) == dict(
        sequences=40, nonpadding_tokens=3352, causal_target_tokens=3312
    )
    with pytest.raises(ValueError):
        probe.token_counts(probe.LENGTHS, updates=19)


def test_repetitions_alternate_runtime_order_without_missing_rows(probe):
    planned = probe.schedule()
    assert len(planned) == 18 and len(set(planned)) == 18
    for precision in ("fp32", "fp16", "bf16"):
        groups = [
            [runtime for p, r, runtime in planned if p == precision and r == repeat]
            for repeat in range(3)
        ]
        assert groups[0] == list(reversed(groups[1])) == groups[2]


@pytest.mark.parametrize(
    "bad",
    [
        "skip",
        "fake_global_step",
        "missing_real_warmup",
        "cold_jit",
        "wrong_tokens",
        "wrong_lengths",
        "fallback",
        "prime_only",
        "nonfinite",
        "missing_state",
        "wrong_counter",
        "clock_change",
    ],
)
def test_invalid_throughput_measurements_are_rejected(probe, bad):
    value = row(probe)
    if bad == "skip":
        value["optimizer_calls"] = 24
    elif bad == "fake_global_step":
        value["global_step"] = 24
    elif bad == "missing_real_warmup":
        value["timing"]["warmup_optimizer_calls"] = 4
    elif bad == "cold_jit":
        value["timing"]["jit_changes"] = {"new.so": [1, 2]}
    elif bad == "wrong_tokens":
        value["timing"]["counts"]["nonpadding_tokens"] = 3340
    elif bad == "wrong_lengths":
        value["sequence_lengths"] = [128] * 10
    elif bad == "fallback":
        value["fallback_count"] = 1
    elif bad == "prime_only":
        value["mode"] = "prime"
    elif bad == "nonfinite":
        value["verification"]["passed"] = False
    elif bad == "missing_state":
        value["verification"]["all112_parameter_states_finite"] = False
    elif bad == "wrong_counter":
        value["verification"]["all112_parameter_counters"] = 24
    else:
        value["hardware_after"]["power.limit"] = "200"
    with pytest.raises(AssertionError):
        probe.validate_row(value)


def test_summary_reports_medians_ranges_and_same_condition_ratio(probe):
    rows = [row(probe, p, runtime, r) for p, r, runtime in probe.schedule()]
    summary = probe.summarize(rows)
    for profile in summary["profiles"].values():
        assert profile["jittor_over_torch_throughput"] == 0.5
        assert profile["runtimes"]["torch"]["nonpadding_tokens_per_second"]["median"] == 1676.0
    rows[0]["initial_adapters_sha256"] = "different"
    with pytest.raises(AssertionError):
        probe.summarize(rows)


def test_missing_or_duplicate_repetitions_do_not_produce_summary(probe):
    rows = [row(probe, p, runtime, r) for p, r, runtime in probe.schedule()]
    with pytest.raises(AssertionError):
        probe.summarize(rows[:-1])
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(AssertionError):
        probe.summarize(rows)


def test_observer_callbacks_do_not_read_or_copy_tensor_state(probe):
    tree = ast.parse(Path(probe.__file__).read_text())
    callbacks = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "Timing"
    )
    forbidden = {"numpy", "cpu", "state_dict", "item", "grad", "state", "digest", "array_hash"}
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in forbidden for node in ast.walk(callbacks)
    )
    counted = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "counted"
    )
    calls = [node for node in ast.walk(counted) if isinstance(node, ast.Call)]
    assert (
        len(calls) == 1 and isinstance(calls[0].func, ast.Name) and calls[0].func.id == "original"
    )


def test_explicit_compiler_symlink_keeps_existing_cache_identity(probe, tmp_path):
    compiler = tmp_path / 'nvcc-real'
    compiler.write_text('#!/bin/sh\nexit 0\n')
    compiler.chmod(0o755)
    selected = tmp_path / 'nvcc'
    selected.symlink_to(compiler)
    args = types.SimpleNamespace(nvcc=selected)
    probe.normalize_paths(args)
    assert args.nvcc == selected
    assert args.nvcc != compiler


@pytest.mark.parametrize('seconds', [float('nan'), float('inf'), float('-inf'), 0., -1.])
def test_nonfinite_or_nonpositive_time_is_never_accepted(probe, seconds):
    value = row(probe)
    value['timing']['seconds'] = seconds
    with pytest.raises(AssertionError):
        probe.validate_row(value)


@pytest.mark.parametrize('field', ['runtime_sources', 'compiled_path', 'compiled_sha'])
def test_changed_runtime_provenance_is_rejected(probe, field):
    rows = [row(probe, p, runtime, r) for p, r, runtime in probe.schedule()]
    if field == 'runtime_sources':
        rows[-1]['runtime_sources'] = {'owner': 'different'}
    else:
        rows[-1]['compiled_core']['path' if field == 'compiled_path' else 'sha256'] = 'different'
    with pytest.raises(AssertionError):
        probe.summarize(rows)


def test_runtime_sources_bind_untracked_native_owners(probe, tmp_path):
    import subprocess
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    tracked = checkout / "python/jittor/owner.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("value = 1\n")
    subprocess.run(["git", "-C", str(checkout), "add", "python"], check=True)
    original = probe.runtime_sources(checkout)
    helper = checkout / "backends/cuda/kernels/nn/new_precision.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("value = 2\n")
    with_helper = probe.runtime_sources(checkout)
    key = str(helper.relative_to(checkout))
    assert key in with_helper and with_helper != original
    helper.write_text("value = 3\n")
    assert probe.runtime_sources(checkout)[key] != with_helper[key]
    helper.unlink()
    assert probe.runtime_sources(checkout) == original
