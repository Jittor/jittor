"""Exercise real breadth-probe preflight decisions without importing either runtime."""

from contextlib import nullcontext
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "agent/skills/torch-compat-repo-runbook/scripts"


class ReachedBuilder(Exception):
    """Stop after preflight, before any model construction or training."""


@pytest.fixture
def probe_environment(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    probe = importlib.import_module("trl_mainstream_probe")
    case = importlib.import_module("trl_offline_cases")
    reached = []

    def build(*args):
        reached.append(True)
        raise ReachedBuilder

    monkeypatch.setattr(case, "build", build)
    monkeypatch.setattr(probe, "tokenizer_for_probe", lambda: object())
    monkeypatch.setattr(probe, "bind_trl_source", lambda *args: {"source": "fixed-offline-fixture"})
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *args, **kwargs: "fixture-checkout\n")
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: probe.PINS[name])
    fake_torch = SimpleNamespace(__file__=__file__, __version__="fixture",
                                 _C=SimpleNamespace(__file__=__file__),
                                 set_num_threads=lambda *args: None, manual_seed=lambda *args: None)
    runner = SimpleNamespace(__file__=__file__, _import_torch=lambda runtime: fake_torch,
                             _select_device=lambda *args, **kwargs: None,
                             _configure_tf32=lambda *args: {"matmul": False})
    monkeypatch.setitem(sys.modules, "_ecosystem_runner", runner)
    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers",
                        SimpleNamespace(Qwen3Config=object, Qwen3ForCausalLM=object,
                                        Qwen3ForSequenceClassification=object))
    monkeypatch.setitem(sys.modules, "jittor",
                        SimpleNamespace(__file__=str(ROOT / "python/jittor/__init__.py"),
                                        runtime=SimpleNamespace(scope=lambda **kwargs: nullcontext()),
                                        core=SimpleNamespace(backend_fallback_count=lambda: 0)))
    monkeypatch.setitem(sys.modules, "jittor._runtime.fallback",
                        SimpleNamespace(forbid_backend_fallbacks=nullcontext))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def arguments(runtime, output, reference=None):
        return SimpleNamespace(root=ROOT, upstream=tmp_path / "upstream", output=output,
                               runtime=runtime, reference=reference, device="cpu", algorithm="sft",
                               threads=4, steps=10, learning_rate=1e-4)

    oracle = tmp_path / "oracle"
    with pytest.raises(ReachedBuilder):
        probe.run(arguments("torch", oracle))
    manifest = json.loads((oracle / "manifest.json").read_text())
    # This is an explicitly synthetic preflight fixture, not completed-run evidence.
    manifest.update(status="complete", fallback_count=0)
    (oracle / "manifest.json").write_text(json.dumps(manifest))
    reached.clear()
    return SimpleNamespace(probe=probe, args=arguments("jittor", tmp_path / "candidate", oracle),
                           manifest=manifest, path=oracle / "manifest.json", reached=reached)


def test_matching_preflight_reaches_builder(probe_environment):
    env = probe_environment
    with pytest.raises(ReachedBuilder):
        env.probe.run(env.args)
    assert env.reached == [True]


@pytest.mark.parametrize("field,value", [
    ("status", "running"), ("runtime", "jittor"), ("fallback_count", 1),
])
def test_incomplete_wrong_runtime_or_fallback_reference_is_rejected(probe_environment, field, value):
    env = probe_environment
    env.manifest[field] = value
    env.path.write_text(json.dumps(env.manifest))
    with pytest.raises(AssertionError, match="complete zero-fallback genuine Torch oracle"):
        env.probe.run(env.args)
    assert not env.reached


@pytest.mark.parametrize("field", ["protocol", "versions", "trl_source", "tool_sources", "device"])
def test_reference_provenance_mismatch_is_rejected_before_builder(probe_environment, field):
    env = probe_environment
    if field == "device":
        env.manifest[field] = "cuda"
    else:
        env.manifest[field] = {**env.manifest[field], "unexpected_drift": True}
    env.path.write_text(json.dumps(env.manifest))
    with pytest.raises(AssertionError, match="Oracle configuration/source mismatch: " + field):
        env.probe.run(env.args)
    assert not env.reached


def test_network_enabled_environment_is_rejected(probe_environment, monkeypatch):
    env = probe_environment
    monkeypatch.delenv("HF_HUB_OFFLINE")
    with pytest.raises(ValueError, match="Require offline environments"):
        env.probe.run(env.args)
    assert not env.reached


@pytest.mark.parametrize('fail', [False, True])
def test_online_loss_observer_preserves_backward_and_restores(monkeypatch, fail):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from trl_mainstream_online import observe_backward_loss
    events = []
    loss, argument, result = object(), object(), object()

    def backward(value, extra, *, retain_graph):
        events.append(('backward', value, extra, retain_graph))
        if fail:
            raise RuntimeError('original backward failure')
        return result

    accelerator = SimpleNamespace(backward=backward)
    def record(name, value):
        events.append((name, value))
    if fail:
        with pytest.raises(RuntimeError, match='original backward failure'):
            with observe_backward_loss(accelerator, record):
                accelerator.backward(loss, argument, retain_graph=True)
    else:
        with observe_backward_loss(accelerator, record):
            assert accelerator.backward(loss, argument, retain_graph=True) is result
    assert accelerator.backward is backward
    assert events == [('loss', loss), ('backward', loss, argument, True)]
