"""No-model controls for offline TRL experiment data and observation contracts."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = (Path(__file__).resolve().parents[2]
           / "agent/skills/torch-compat-repo-runbook/scripts")
CASES = None


@pytest.fixture(autouse=True)
def load_cases(monkeypatch):
    spec = importlib.util.spec_from_file_location("offline_cases_under_test", SCRIPTS / "trl_offline_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(globals(), "CASES", module)


@pytest.mark.parametrize("algorithm", ("sft", "dpo", "kto", "orpo", "cpo", "reward"))
def test_ten_distinct_examples_use_only_real_fixture_tokens(algorithm):
    rows = CASES.records(algorithm)
    assert len(rows) == 10
    assert len({repr(row) for row in rows}) == 10
    vocabulary = {"t%d" % number for number in range(252)}
    for row in rows:
        for value in row.values():
            if isinstance(value, str):
                assert value.split() and set(value.split()) <= vocabulary
        if "rejected" in row:
            assert row["chosen"] != row["rejected"]


def test_kto_each_actual_microbatch_contains_both_labels():
    rows = CASES.records("kto")
    for start in range(0, 10, 2):
        assert {row["label"] for row in rows[start:start + 2]} == {True, False}


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown offline method"):
        CASES.records("invented_method")


@pytest.fixture
def dependencies(monkeypatch):
    """Stub external constructors, keeping the tested builder's real control flow."""
    calls, observations = [], []
    result = (object(), {"metric": 1.25})

    class Trainer:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def compute_loss(self, *args, **kwargs):
            calls.append((args, kwargs))
            return result

        def get_batch_loss_metrics(self, *args, **kwargs):
            calls.append((args, kwargs))
            return result

    trl = SimpleNamespace()
    for name in ("SFT", "DPO", "KTO", "ORPO", "CPO", "Reward"):
        setattr(trl, name + "Trainer", Trainer)
        setattr(trl, name + "Config", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setitem(__import__("sys").modules, "trl", trl)
    monkeypatch.setitem(__import__("sys").modules, "datasets",
                        SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows)))
    torch = SimpleNamespace(is_tensor=lambda value: False,
                            utils=SimpleNamespace(data=SimpleNamespace(SequentialSampler=iter)))

    class Tokenizer:
        unk_token_id = 3
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            return {"input_ids": [int(token[1:]) + 4 for token in text.split()]}

    class Parameter:
        requires_grad = True

        def requires_grad_(self, value):
            self.requires_grad = value

    roles = {}

    def models(role, kind="causal", copy_from=None):
        parameter = Parameter()
        model = SimpleNamespace(config=SimpleNamespace(), training=None,
                                parameters=lambda: [parameter])
        model.train = lambda: setattr(model, "training", True)
        model.eval = lambda: setattr(model, "training", False)
        roles[role] = (model, kind, copy_from)
        return model

    def record(name, value, exact=False):
        observations.append((name, value, exact))

    return SimpleNamespace(torch=torch, tokenizer=Tokenizer(), models=models,
                           record=record, calls=calls, result=result,
                           observations=observations, roles=roles)


def build(dependencies, tmp_path, algorithm):
    args = SimpleNamespace(algorithm=algorithm, output=tmp_path, device="cpu", steps=10,
                           learning_rate=1e-4, training_kwargs={})
    return CASES.build(args, dependencies.torch, dependencies.tokenizer,
                       dependencies.models, dependencies.record)


@pytest.mark.parametrize("algorithm", ("sft", "dpo", "kto", "orpo", "cpo", "reward"))
def test_training_scope_and_reference_contract(dependencies, tmp_path, algorithm):
    trainer, metadata = build(dependencies, tmp_path, algorithm)
    batch = 2 if algorithm == "kto" else 1
    assert metadata["training_examples_seen"] == 10 * batch
    assert metadata["equivalent_dataset_passes"] == batch
    assert trainer.args.max_steps == 10
    assert trainer.args.per_device_train_batch_size == batch
    assert not trainer.args.fp16 and not trainer.args.bf16
    assert trainer.args.optim == "sgd" and trainer.args.gradient_accumulation_steps == 1
    assert list(trainer._get_train_sampler()) == CASES.records(algorithm)
    policy, kind, _ = dependencies.roles["policy"]
    assert kind == ("sequence_classification" if algorithm == "reward" else "causal")
    assert policy.training and not policy.config.use_cache
    if algorithm in ("dpo", "kto"):
        reference, _, copied_from = dependencies.roles["reference"]
        assert copied_from == "policy" and reference is not policy
        assert not reference.training and all(not p.requires_grad for p in reference.parameters())
    else:
        assert set(dependencies.roles) == {"policy"}


def test_observer_preserves_upstream_return_and_host_labels(dependencies, tmp_path):
    trainer, _ = build(dependencies, tmp_path, "kto")
    labels = [True, False]
    inputs = {"label": labels}
    sentinel_model = object()
    actual = trainer.compute_loss(sentinel_model, inputs, return_outputs=True, num_items_in_batch=2)
    assert actual is dependencies.result
    assert dependencies.calls[-1] == ((sentinel_model, inputs),
                                      {"return_outputs": True, "num_items_in_batch": 2})
    observed_labels = next(row for row in dependencies.observations if row[0] == "input::label")
    assert observed_labels[1] is labels and observed_labels[2] is True
    assert trainer.get_batch_loss_metrics(sentinel_model, inputs) is not None
    assert dependencies.calls[-1] == ((sentinel_model, inputs), {})


def test_preprocessing_must_not_silently_remove_examples(dependencies, tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys.modules["datasets"].Dataset, "from_list", lambda rows: rows[:-1])
    with pytest.raises(AssertionError, match="changed dataset size"):
        build(dependencies, tmp_path, "reward")
