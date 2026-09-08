"""Distributed owner identity and simulated group/collective routing."""
import ast
import inspect
import pickle
import textwrap
from types import MappingProxyType

import numpy as np
import pytest
import jittor as jt
from jittor.compat.torch.context import get_install_context
from jittor.compat.torch.tensor_state import compatibility_owner
from jittor.compat.torch.installers import distributed as owner


class Group:
    def __init__(self, ranks=(0, 1, 2, 3), name="world", rank=2):
        self.ranks = tuple(ranks)
        self.name = name
        self.global_rank = rank
        self.created = 0
        self.reductions = []

    def size(self):
        return len(self.ranks)

    def rank(self):
        return self.ranks.index(self.global_rank) if self.global_rank in self.ranks else -100

    def _get_backend_name(self):
        return "nccl"

    def _create_backend_communicator(self):
        self.created += 1

    def _all_reduce(self, tensor, operation):
        self.reductions.append((tensor, operation))
        return Tensor(tensor.values + 100)


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.roots = []

    def update(self, other):
        self.values = other.values.copy()
        self.shape = self.values.shape

    def reshape(self, shape):
        return Tensor(self.values.reshape(shape))

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def mpi_broadcast(self, root):
        self.roots.append(root)
        return Tensor(self.values + root)


@pytest.fixture
def distributed_state(monkeypatch):
    context = get_install_context(jt)
    world = Group()
    state = {"initialized": False, "store": None}
    groups = {world: ("nccl",)}
    monkeypatch.setitem(context.state, "distributed_api", MappingProxyType({
        "state": state, "world_group": world, "pg_map": groups,
        "dist": compatibility_owner(jt).distributed,
    }))
    monkeypatch.setattr(owner, "_distributed_rank", lambda: 2)
    monkeypatch.setattr(owner, "_distributed_world_size", lambda: 4)
    monkeypatch.setattr(owner, "_native_distributed_active", lambda: False)
    return state, world, groups


def test_installer_and_bootstrap_bind_stable_implementations():
    for function in (owner._install_distributed, owner._bootstrap_native_distributed):
        node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        assert not [child for child in ast.walk(node) if child is not node and
                    isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.Lambda))]
    dist = compatibility_owner(jt).distributed
    for name, implementation in (
        ("all_reduce", owner._all_reduce), ("all_gather", owner._all_gather),
        ("broadcast", owner._broadcast), ("get_rank", owner._get_rank),
        ("new_group", owner._new_group), ("init_process_group", owner._init_process_group),
        ("P2POp", owner.P2POp), ("Backend", owner.Backend),
    ):
        assert getattr(dist, name) is implementation
        assert pickle.loads(pickle.dumps(implementation)) is implementation
    assert dist.ProcessGroup is owner._JittorProcessGroup
    assert dist.distributed_c10d.Work is owner._JittorWork
    assert dist.distributed_c10d._get_default_group() is dist.group.WORLD


def test_group_creation_and_rank_maps(distributed_state, monkeypatch):
    _, _, groups = distributed_state
    monkeypatch.setattr(owner, "_JittorProcessGroup", Group)
    group = owner._new_group([0, 2])
    assert group.ranks == (0, 2) and group.created == 1
    assert groups[group] == ("nccl",)
    assert owner._get_rank(group) == 1
    assert owner._get_world_size(group) == 2
    assert owner._api_dist_get_global_rank(group, 1) == 2
    assert owner._api_dist_get_process_group_ranks(group) == [0, 2]
    with pytest.raises(ValueError, match="unique"):
        owner._new_group([2, 2])
    with pytest.raises(ValueError, match="outside WORLD"):
        owner._new_group([4])


def test_collective_group_identity_reduction_and_work(distributed_state):
    _, group, _ = distributed_state
    tensor = Tensor([1, 2])
    work = owner._all_reduce(tensor, owner._ReduceOp.MAX, group, async_op=True)
    assert group.reductions == [(tensor, "max")]
    assert isinstance(work, owner._JittorWork)
    assert work.wait() is tensor
    np.testing.assert_array_equal(tensor.values, [101, 102])
    absent = Group((0, 1), rank=3)
    owner._all_reduce(tensor, group=absent)
    assert absent.reductions == []
    assert owner._autograd_all_reduce(tensor, group=absent) is tensor


def test_gather_and_broadcast_argument_routes(distributed_state, monkeypatch):
    _, group, _ = distributed_state
    source = Tensor([7, 8])
    calls = []
    def gather(tensor):
        calls.append(tensor)
        return Tensor(np.arange(8))
    monkeypatch.setattr(owner, "_native_all_gather_flat", gather)
    output = [Tensor([0, 0]) for _ in range(4)]
    owner._all_gather(output, source, group)
    for index, value in enumerate(output):
        np.testing.assert_array_equal(value.values, [2 * index, 2 * index + 1])
    combined = Tensor(np.zeros((4, 2)))
    owner._all_gather_into_tensor(combined, source, group)
    np.testing.assert_array_equal(combined.values, np.arange(8).reshape(4, 2))
    assert calls == [source, source]
    owner._broadcast(source, src=3, group=group, group_src=1)
    assert source.roots == [1]
    with pytest.raises(ValueError, match="shorter"):
        owner._all_gather([], source, group)


def test_init_destroy_store_and_backend_validation(distributed_state, monkeypatch):
    state, _, _ = distributed_state
    class Store:
        closed = 0
        def close(self):
            self.closed += 1
    store = Store()
    owner._init_process_group(backend="gloo", rank=0, world_size=1, store=store)
    assert state == {"initialized": True, "store": store}
    assert owner._api_c10d_get_default_store() is store
    owner._destroy_process_group()
    assert store.closed == 1 and state == {"initialized": False, "store": None}
    monkeypatch.setattr(owner, "_native_distributed_active", lambda: True)
    with pytest.raises(RuntimeError, match="does not match"):
        owner._init_process_group(backend="hccl", rank=2, world_size=4)
    owner._init_process_group(backend="cpu:gloo,cuda:nccl", rank=2, world_size=4)
    assert state["initialized"]


def test_checkpoint_stubs_keep_policy_and_importable_owner(monkeypatch):
    from jittor.compat import stub_policy
    dist = compatibility_owner(jt).distributed
    monkeypatch.setattr(stub_policy, "allow_stub", lambda: False)
    for name in ("load", "save", "load_state_dict", "save_state_dict"):
        function = getattr(dist.checkpoint, name)
        assert "<locals>" not in function.__qualname__
        assert pickle.loads(pickle.dumps(function)) is function
        assert function._jittor_unimplemented.endswith("." + name)
        with pytest.raises(NotImplementedError, match="checkpoint"):
            function({})
