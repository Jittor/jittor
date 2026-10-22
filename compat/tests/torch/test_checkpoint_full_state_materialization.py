import jittor as jt
import jittor.compat.torch.installers.distributed as distributed


def test_full_state_cpu_offload_materializes_distributed_gather_before_drop(monkeypatch):
    events = []

    class Provider:
        def _get_full_state_dict(self, model):
            events.append("gather")
            return {"weight": object()}

    class Model:
        def named_modules(self):
            return [("", self)]

    monkeypatch.setattr(distributed, "_checkpoint_model", lambda model: model)
    monkeypatch.setattr(
        distributed, "_checkpoint_fsdp_provider", lambda model, options: Provider())
    monkeypatch.setattr(distributed, "_distributed_world_size", lambda: 2)
    monkeypatch.setattr(jt, "sync_all", lambda *args: events.append("materialize"))
    monkeypatch.setattr(
        distributed, "_checkpoint_offload",
        lambda value, options: events.append("offload") or value)

    options = distributed.StateDictOptions(full_state_dict=True, cpu_offload=True)
    state = distributed._get_model_state_dict(Model(), options=options)

    assert list(state) == ["weight"]
    assert events == ["gather", "materialize", "offload"]

def test_full_optimizer_cpu_offload_materializes_state_gather_before_drop(monkeypatch):
    events = []

    class Provider:
        def refresh_optimizer_fsdp_params(self, optimizer):
            events.append("refresh")

        def _gather_optimizer_state_dict(self, model, state, ranks):
            events.append("optimizer_gather")
            return state

    class Model:
        pass

    monkeypatch.setattr(distributed, "_checkpoint_model", lambda model: model)
    monkeypatch.setattr(
        distributed, "_checkpoint_fsdp_provider", lambda model, options: Provider())
    monkeypatch.setattr(distributed, "_checkpoint_parameter_names", lambda model: ({}, {}))
    monkeypatch.setattr(distributed, "_checkpoint_optimizers", lambda optimizers: [object()])
    monkeypatch.setattr(
        distributed, "_checkpoint_initialize_optimizer",
        lambda optimizer: events.append("initialize"))
    monkeypatch.setattr(
        distributed, "_checkpoint_optimizer_layout",
        lambda optimizer, names: ({"state": {}, "param_groups": []}, {}))
    monkeypatch.setattr(
        distributed, "_native_all_gather_object",
        lambda object_list, obj: events.append("metadata_gather"))
    monkeypatch.setattr(distributed, "_distributed_world_size", lambda: 2)
    monkeypatch.setattr(jt, "sync_all", lambda *args: events.append("materialize"))
    monkeypatch.setattr(
        distributed, "_checkpoint_offload",
        lambda value, options: events.append("offload") or value)

    options = distributed.StateDictOptions(full_state_dict=True, cpu_offload=True)
    state = distributed._get_optimizer_state_dict(Model(), [object()], options=options)

    assert state == {"state": {}, "param_groups": []}
    assert events == [
        "refresh", "initialize", "metadata_gather", "optimizer_gather",
        "materialize", "offload",
    ]
