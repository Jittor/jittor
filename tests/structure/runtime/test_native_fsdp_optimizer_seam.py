"""Native FSDP optimizer wiring contracts without a framework runtime.

The source/AST checks protect dispatch to existing owners. Small protocol spies
exercise only canonical Python lifecycle functions; tensor arithmetic, real
CPU execution and NCCL/NPU qualification remain separate gates.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]


def source(relative):
    return ast.parse((ROOT / relative).read_text())


def owner_function(relative, name, namespace):
    node = next(item for item in source(relative).body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), namespace)
    return namespace[name]


def methods():
    owner = next(item for item in source("python/jittor/optim/base.py").body
                 if isinstance(item, ast.ClassDef) and item.name == "Optimizer")
    return {item.name: item for item in owner.body if isinstance(item, ast.FunctionDef)}


def calls(node, name):
    return [item for item in ast.walk(node)
            if isinstance(item, ast.Call) and ast.unparse(item.func) == name]


def canonical_imports(node):
    return {alias.name for item in ast.walk(node) if isinstance(item, ast.ImportFrom)
            and item.module == "jittor.distributed.fsdp.optimizer"
            for alias in item.names}


def test_backward_prepares_full_targets_through_canonical_owner():
    backward = methods()["backward"]
    assert {"_prepare_backward", "_finish_backward"} <= canonical_imports(backward)
    prepare, = calls(backward, "_prepare_backward")
    assert [ast.unparse(arg) for arg in prepare.args] == [
        "self", "params", "params_has_grad", "retain_graph"]
    synchronized = calls(backward, "jt.sync")
    assert synchronized and ast.unparse(synchronized[0].args[0]) == "grad_targets"
    assert prepare.lineno < synchronized[0].lineno


def test_only_legacy_path_finishes_full_gradients_before_local_holders():
    backward = methods()["backward"]
    ready = next(item for item in ast.walk(backward) if isinstance(item, ast.If)
                 and ast.unparse(item.test) == "readiness_plan is not None")
    fast = ast.Module(body=ready.body, type_ignores=[])
    assert isinstance(ready.body[-1], ast.Return), "DDP path must not enter FSDP finish"
    legacy = ast.Module(
        body=[item for item in backward.body if item.lineno > ready.end_lineno],
        type_ignores=[])
    assert not calls(fast, "_finish_backward")
    callback, = calls(fast, "jt.core.grad_with_callback")
    assert ast.unparse(callback.args[1]) == "params_has_grad"
    gradient, = calls(legacy, "jt.grad")
    assert [ast.unparse(arg) for arg in gradient.args] == [
        "loss", "grad_targets", "retain_graph"]
    finish, = calls(legacy, "_finish_backward")
    assert [ast.unparse(arg) for arg in finish.args] == ["params", "grad_targets", "grads"]
    updates = calls(legacy, "pg_grads[i].update")
    assert updates and gradient.lineno < finish.lineno < updates[0].lineno


def test_fsdp_extra_optimizer_inputs_remain_explicitly_unsupported():
    backward = methods()["backward"]
    guard = next(item for item in backward.body if isinstance(item, ast.If)
                 and ast.unparse(item.test) == "fsdp_parameter_ids and self.__input_params")
    error, = [item for item in guard.body if isinstance(item, ast.Raise)]
    assert isinstance(error.exc, ast.Call)
    assert ast.unparse(error.exc.func) == "NotImplementedError"
    assert "extra optimizer input parameters" in error.exc.args[0].value
    assert guard.lineno < calls(backward, "jt.sync")[0].lineno


def test_fsdp_ids_are_excluded_from_both_legacy_collective_loops():
    backward = methods()["backward"]
    assignments = [item for item in ast.walk(backward) if isinstance(item, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "managed_parameter_ids"
                           for target in item.targets)]
    assert len(assignments) == 1
    assert ast.unparse(assignments[0].value) == "ddp_parameter_ids | tp_parameter_ids | fsdp_parameter_ids"
    skipped = [item for item in ast.walk(backward) if isinstance(item, ast.If)
               and ast.unparse(item.test) == "id(parameter) in managed_parameter_ids"]
    assert len(skipped) == 2
    assert all(isinstance(item.body[0], ast.Continue) for item in skipped)


def test_step_readiness_uses_existing_owner_before_update_order_flag():
    pre_step = methods()["pre_step"]
    assert "_assert_step_ready" in canonical_imports(pre_step)
    guard, = calls(pre_step, "_assert_step_ready")
    assert [ast.unparse(arg) for arg in guard.args] == ["self"]
    ddp, = calls(pre_step, "_assert_ddp_step_ready")
    assignment = next(item for item in pre_step.body if isinstance(item, ast.Assign)
                      and ast.unparse(item.targets[0]) == "jt.flags.node_order")
    assert ddp.lineno < guard.lineno < assignment.lineno


def test_ordinary_optimizer_targets_and_gradients_preserve_identity():
    targets = [object(), object()]
    gradients = [object(), object()]
    owner = "python/jittor/distributed/fsdp/optimizer.py"
    prepare = owner_function(owner, "_prepare_backward", {"_states": lambda values: []})
    finish = owner_function(owner, "_finish_backward", {"_states": lambda values: []})
    prepared, managed = prepare(object(), [], targets, True)
    assert prepared is targets and managed == set()
    assert finish([], targets, gradients) is gradients


@pytest.mark.parametrize("failure", [None, "reduce", "sync"])
def test_canonical_finish_releases_only_after_successful_reduction_and_sync(failure):
    full, shard, local = object(), object(), object()
    entry = SimpleNamespace(full_param=full)
    state = SimpleNamespace(true_fsdp_params=[entry], native_forward_pending=True)
    events = []

    def reduce(selected, gradients, api):
        events.append("reduce")
        assert selected is state and gradients == [full_gradient]
        if failure == "reduce":
            raise RuntimeError("reduction failed")
        return [local]

    def sync(roots, device_sync, weak_sync):
        events.append("sync")
        assert roots == [local] and device_sync is True and weak_sync is False
        if failure == "sync":
            raise RuntimeError("sync failed")

    def release(selected):
        events.append("release")
        assert selected is state and state.native_forward_pending
        entry.full_param = None

    full_gradient = object()
    namespace = {"_states": lambda values: [state],
                 "gradients": SimpleNamespace(_sync_sharded_grads_from_full_grads=reduce),
                 "shard": SimpleNamespace(_release_full_params=release),
                 "jt": SimpleNamespace(sync=sync)}
    finish = owner_function("python/jittor/distributed/fsdp/optimizer.py",
                            "_finish_backward", namespace)
    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            finish([shard], [full], [full_gradient])
        assert state.native_forward_pending and entry.full_param is full
        assert "release" not in events
    else:
        assert finish([shard], [full], [full_gradient]) == [local]
        assert events == ["reduce", "sync", "release"]
        assert not state.native_forward_pending and entry.full_param is None


@pytest.mark.parametrize("pending", [False, True])
def test_checkpoint_completed_boundary_refusal_is_preserved(pending):
    state = SimpleNamespace(native_forward_pending=pending)
    entry = object()
    parameter = SimpleNamespace(_jittor_native_fsdp_state=state,
                                _jittor_fsdp2_entry=entry)
    model = SimpleNamespace(named_parameters=lambda: [("weight", parameter)])
    entries = owner_function("python/jittor/distributed/fsdp/checkpoint.py", "_entries",
                             {"_validate_parameter": lambda value: None})
    if pending:
        with pytest.raises(RuntimeError, match="completed forward/backward"):
            entries(model)
    else:
        assert entries(model) == {"weight": (state, entry)}


# These checks execute only this package's tiny export module with import spies.
# The canonical class is a sentinel; no Jittor, Torch or accelerator is imported.
def host_parallel_exports(monkeypatch, owner_error=None):
    import builtins
    import sys
    from types import ModuleType

    package_name = "_host_native_parallel_exports"
    module = ModuleType(package_name)
    module.__package__ = package_name
    module.__path__ = []
    module.__file__ = str(ROOT / "python/jittor/nn/parallel/__init__.py")
    ddp = ModuleType(package_name + ".distributed_data_parallel")
    ddp.DistributedDataParallel = type("DistributedDataParallel", (), {})
    canonical = ModuleType("jittor.distributed.fsdp")
    canonical.FullyShardedDataParallel = type(
        "FullyShardedDataParallel", (), {"__module__": "jittor.distributed.fsdp.api"})
    monkeypatch.setitem(sys.modules, package_name, module)
    monkeypatch.setitem(sys.modules, ddp.__name__, ddp)
    original_import = builtins.__import__
    owner_imports = []

    def import_spy(name, globals=None, locals=None, fromlist=(), level=0):
        assert name.split(".", 1)[0] not in {"torch", "torch_npu"}
        assert not name.startswith("jittor.compat")
        if name == "jittor.distributed.fsdp":
            assert level == 0
            assert tuple(fromlist) == ("FullyShardedDataParallel",)
            owner_imports.append(name)
            if owner_error is not None:
                raise owner_error
            return canonical
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_spy)
    exec(compile(Path(module.__file__).read_text(), module.__file__, "exec"),
         module.__dict__)
    return module, ddp, canonical, owner_imports


def test_lazy_parallel_export_declares_both_public_native_classes():
    tree = source("python/jittor/nn/parallel/__init__.py")
    exports = next(item.value for item in tree.body if isinstance(item, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "__all__"
                           for target in item.targets))
    assert ast.literal_eval(exports) == [
        "DistributedDataParallel", "ColumnParallelLinear", "RowParallelLinear",
        "FullyShardedDataParallel"]
    eager_owners = [item for item in tree.body if isinstance(item, ast.ImportFrom)
                    and item.module == "jittor.distributed.fsdp"]
    assert not eager_owners
    lazy = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                and item.name == "__getattr__")
    owners = [item for item in ast.walk(lazy) if isinstance(item, ast.ImportFrom)
              and item.module == "jittor.distributed.fsdp"]
    assert len(owners) == 1
    assert [alias.name for alias in owners[0].names] == ["FullyShardedDataParallel"]


def test_lazy_parallel_package_execution_preserves_ddp_without_loading_fsdp(monkeypatch):
    module, ddp, _, owner_imports = host_parallel_exports(monkeypatch)
    assert module.DistributedDataParallel is ddp.DistributedDataParallel
    assert "FullyShardedDataParallel" not in module.__dict__
    assert owner_imports == []


def test_lazy_parallel_from_import_returns_the_canonical_class(monkeypatch):
    module, _, canonical, owner_imports = host_parallel_exports(monkeypatch)
    namespace = {}
    exec("from _host_native_parallel_exports import FullyShardedDataParallel as exported",
         namespace)
    assert namespace["exported"] is canonical.FullyShardedDataParallel
    assert module.FullyShardedDataParallel is canonical.FullyShardedDataParallel
    assert namespace["exported"].__module__ == "jittor.distributed.fsdp.api"
    assert owner_imports


@pytest.mark.parametrize("name", ["missing", "fully_shard", "_private"])
def test_lazy_parallel_unknown_attribute_does_not_load_or_fallback(monkeypatch, name):
    module, _, _, owner_imports = host_parallel_exports(monkeypatch)
    with pytest.raises(AttributeError, match=name):
        getattr(module, name)
    assert owner_imports == []


def test_lazy_parallel_canonical_import_error_is_not_hidden(monkeypatch):
    failure = ImportError("canonical native FSDP owner unavailable")
    module, _, _, owner_imports = host_parallel_exports(monkeypatch, owner_error=failure)
    with pytest.raises(ImportError) as caught:
        getattr(module, "FullyShardedDataParallel")
    assert caught.value is failure
    assert owner_imports == ["jittor.distributed.fsdp"]
