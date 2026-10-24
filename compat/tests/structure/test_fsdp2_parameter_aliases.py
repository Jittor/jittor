"""Execute FSDP ownership/lifecycle functions without importing Jittor or JIT.

These are metadata contracts, not distributed numerical or device evidence.
"""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref


ROOT = Path(__file__).resolve().parents[2] / "fsdp2"


def load_functions(filename, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for node in functions:
        node.decorator_list = []
    exec(compile(ast.Module(body=functions, type_ignores=[]), filename, "exec"), namespace)


class Var:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)
        self.dtype = "float32"
        self.requires_grad = True

    def numel(self):
        return len(self.values)

    def reshape(self, shape):
        return self

    def stop_grad(self):
        self.requires_grad = False
        return self

    def start_grad(self):
        self.requires_grad = True
        return self

    def is_stop_grad(self):
        return not self.requires_grad

    def sync(self):
        return self

    def copy(self):
        return Var(self.values)

    def update(self, other):
        self.values = list(other.values)

    assign = update
    _update = update

    def mpi_broadcast(self, rank):
        return self

    def __getitem__(self, item):
        return Var(self.values[item])

    def __mul__(self, scalar):
        return Var([v * scalar for v in self.values])

    def __sub__(self, other):
        return Var([a - b for a, b in zip(self.values, other.values)])


class Module:
    def named_children(self):
        return [(key, value) for key, value in vars(self).items() if isinstance(value, Module)]

    def named_modules(self, prefix=""):
        yield prefix, self
        for name, child in self.named_children():
            yield from child.named_modules(prefix + "." + name if prefix else name)

    def modules(self):
        return [module for _, module in self.named_modules()]


class State(SimpleNamespace):
    pass


def environment(flat):
    common = SimpleNamespace(
        StateRecord=State, _in_true_distributed=lambda: True,
        _world_size=lambda: 2, _rank=lambda: 0,
        _param_numel=lambda p: p.numel(), _ceil_div=lambda n, d: (n + d - 1) // d,
        _fsdp2_flat_enabled=lambda *a: flat,
        _flatten_var=lambda p: p, _pad_flat=lambda p, n: Var(p.values + [0] * (n - p.numel())),
        _slice_flat=lambda p, start, length: p[start:start + length],
        _all_gather_shards=lambda p, group: Var(p.values * 2),
        _frontend_scope=lambda state: contextlib.nullcontext(),
    )
    ns = dict(common=common, weakref=weakref,
              jt=SimpleNamespace(Var=Var, concat=lambda values, dim=0: Var([v for p in values for v in p.values])),
              nn=SimpleNamespace(Module=Module, ParameterList=type("ParameterList", (Module,), {})),
              _jittor_dtype_name=lambda dtype: dtype,
              EXPECTED=(AttributeError, TypeError), swallowed=lambda *a: None)
    load_functions("shard.py", ns)
    def mark(var, state, entry, role):
        var._jittor_fsdp2_state = state
        var._jittor_fsdp2_entry = entry
        return var
    ns["_mark_fsdp_param_var"] = mark
    return ns


def tied_module():
    model = Module()
    model.embed, model.head = Module(), Module()
    model.embed.weight = model.head.weight = Var([1, 2, 3, 4])
    return model


class TestFSDPParameterAliases(unittest.TestCase):
    def initialize(self, flat):
        ns = environment(flat)
        model = tied_module()
        model._is_fsdp_module = True
        model._fsdp_state = State()
        ns["_init_true_fsdp_state_impl"](model, model._fsdp_state)
        return ns, model, model._fsdp_state

    def assert_shared(self, model, tensor):
        self.assertIs(model.embed.weight, tensor)
        self.assertIs(model.head.weight, tensor)

    def test_init_shards_once_and_rebinds_all_owners(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                ns, model, state = self.initialize(flat)
                self.assertEqual(len(state.true_fsdp_params), 1)
                self.assert_shared(model, state.true_fsdp_params[0].shard)
                self.assertEqual(model.embed.weight.values, [1, 2])

    def test_single_rank_initializes_identity_shards_without_transport(self):
        ns = environment(False)
        ns["common"]._in_true_distributed = lambda: False
        ns["common"]._world_size = lambda: 1
        model = tied_module()
        original = model.embed.weight
        state = State()

        ns["_init_true_fsdp_state_impl"](model, state)

        self.assertTrue(state.true_fsdp_initialized)
        self.assertEqual((state.true_fsdp_rank, state.true_fsdp_world_size), (0, 1))
        self.assertFalse(state.true_fsdp_flat)
        self.assertIs(state.true_fsdp_params[0].shard, original)
        self.assert_shared(model, original)
        self.assertEqual(state.true_fsdp_params[0].shard_numel, 4)

    def test_acl_flat_entry_views_use_one_native_split(self):
        ns = environment(False)
        calls = []
        ns["jt"].compiler = SimpleNamespace(has_acl=True)
        ns["jt"].core = SimpleNamespace(dispatch_context=lambda values: ("acl", 0))
        ns["jt"].flags = SimpleNamespace(no_grad=0)
        ns["jt"].flag_scope = lambda **kwargs: contextlib.nullcontext()

        def split(value, sections, dim=0):
            calls.append((list(sections), dim))
            result, offset = [], 0
            for section in sections:
                result.append(Var(value.values[offset:offset + section]))
                offset += section
            return result

        ns["jt"].split = split
        ns["common"]._zeros_like_shape = lambda value, shape: Var([0] * shape[0])
        state = State(
            true_fsdp_rank=0,
            true_fsdp_flat_shard_numel=4,
            true_fsdp_params=[
                State(flat_offset=0, numel=1),
                State(flat_offset=1, numel=2),
                State(flat_offset=7, numel=1),
            ],
        )
        pieces = ns["_flat_entry_slices"](state, Var([10, 20, 30, 0]))
        self.assertEqual(calls, [([1, 2, 1], 0)])
        self.assertEqual([piece.values for piece in pieces], [[10], [20, 30], []])

    def test_single_rank_parent_skips_a_managed_child_identity_shard(self):
        ns = environment(False)
        ns["common"]._in_true_distributed = lambda: False
        ns["common"]._world_size = lambda: 1
        model = Module()
        model.child = Module()
        model.child.weight = Var([1, 2])
        model.weight = Var([3, 4])
        child_state, root_state = State(), State()

        ns["_init_true_fsdp_state_impl"](model.child, child_state)
        ns["_init_true_fsdp_state_impl"](model, root_state)

        self.assertTrue(child_state.true_fsdp_initialized)
        self.assertTrue(root_state.true_fsdp_initialized)
        self.assertEqual([entry.name for entry in child_state.true_fsdp_params], ["weight"])
        self.assertEqual([entry.name for entry in root_state.true_fsdp_params], ["weight"])
        self.assertIs(model.child.weight, child_state.true_fsdp_params[0].shard)
        self.assertIs(model.weight, root_state.true_fsdp_params[0].shard)

    def test_unshard_reshard_preserve_aliases(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                ns, model, state = self.initialize(flat)
                ns["_unshard_module_params_impl"](model)
                self.assert_shared(model, state.true_fsdp_params[0].full_param)
                ns["_reshard_module_params"](model)
                self.assert_shared(model, state.true_fsdp_params[0].shard)

    def test_nested_cross_owner_rejected(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                ns = environment(flat)
                model = tied_module()
                model.embed._fsdp_state = State()
                ns["_init_true_fsdp_state_impl"](model.embed, model.embed._fsdp_state)
                with self.assertRaisesRegex(NotImplementedError, "shared.*FSDP"):
                    ns["_init_true_fsdp_state_impl"](model, State())

    def test_child_wrapped_after_parent_rejected(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                ns, model, state = self.initialize(flat)
                with self.assertRaisesRegex(NotImplementedError, "shared.*FSDP"):
                    ns["_init_true_fsdp_state_impl"](model.head, State())

    def test_same_module_two_attributes_and_new_cross_unit_alias(self):
        ns = environment(False)
        model = Module()
        model.weight = model.alias = Var([1, 2, 3, 4])
        state = State()
        ns["_init_true_fsdp_state_impl"](model, state)
        self.assertEqual(len(state.true_fsdp_params), 1)
        self.assertIs(model.weight, model.alias)
        sibling = Module()
        sibling.weight = model.weight
        with self.assertRaisesRegex(NotImplementedError, "shared.*FSDP"):
            ns["_init_true_fsdp_state_impl"](sibling, State())

    def test_independent_nested_unit_still_accepted(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                ns = environment(flat)
                model = tied_module()
                model.head.weight = Var([5, 6, 7, 8])
                child_state = model.embed._fsdp_state = State()
                ns["_init_true_fsdp_state_impl"](model.embed, child_state)
                parent_state = State()
                ns["_init_true_fsdp_state_impl"](model, parent_state)
                self.assertEqual([e.name for e in parent_state.true_fsdp_params], ["head.weight"])
                self.assertIs(model.embed.weight, child_state.true_fsdp_params[0].shard)

    def test_regular_optimizer_updates_shared_parameter_once(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                ns, model, state = self.initialize(flat)
                calls = []
                ns["jt"].flags = SimpleNamespace()
                opt = dict(ns, shard=SimpleNamespace(**ns),
                           _SHARDED_KINDS=("sgd",),
                           _realign_state_buffers=lambda groups: None,
                           get_tensor_state=lambda jt: SimpleNamespace(active_optimizers=[]),
                           _optimizer_param_steps=lambda pg: pg.setdefault("steps", [0]),
                           grad_sync=SimpleNamespace(
                               _fsdp_states_from_optimizers=lambda opts: [state],
                               _sync_visible_full_grads_to_optimizer=lambda opt: None))
                load_functions("optimizer.py", opt)
                opt["_optimizer_kind"] = lambda optimizer: "sgd"
                def update(optimizer, pg, state, entry, param, grad, value):
                    calls.append(entry)
                    return param - grad * 0.1, value
                opt["_sgd_update_for_param"] = update
                optimizer = SimpleNamespace(param_groups=[{
                    "params": [state.true_fsdp_params[0].shard], "grads": [Var([1, 2])]}],
                    _build_grad_map=lambda: None)
                self.assertTrue(opt["optimizer_step"](optimizer))
                self.assertEqual(len(calls), 1)
                self.assertEqual(optimizer.param_groups[0]["steps"], [1])
                self.assert_shared(model, state.true_fsdp_params[0].shard)
                self.assertIs(optimizer.param_groups[0]["params"][0], model.embed.weight)
                self.assertEqual(model.embed.weight.values, [0.9, 1.8])

    def test_all_entry_publication_sites_use_one_helper(self):
        for name in ("shard.py", "optimizer.py", "_state_dict.py"):
            tree = ast.parse((ROOT / name).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != "__setattr__" or not node.args:
                    continue
                owner = node.args[0]
                self.assertFalse(isinstance(owner, ast.Attribute) and owner.attr == "owner",
                                 "%s still publishes only the first entry owner" % name)

    def test_optimizer_sgd_rebinds_all_owners(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                ns, model, state = self.initialize(flat)
                state.true_fsdp_last_flat_grad = Var([1, 2])
                opt = dict(ns, shard=SimpleNamespace(**ns), grad_sync=SimpleNamespace(
                    sync_sharded_grads=lambda *a, **k: [Var([1, 2])]))
                load_functions("optimizer.py", opt)
                opt["sharded_sgd_step"](model, None, lr=0.1)
                self.assert_shared(model, state.true_fsdp_params[0].shard)
                self.assertEqual(model.embed.weight.values, [0.9, 1.8])

    def test_state_load_rebinds_all_owners(self):
        # Full-state loading now also validates WORLD mesh/device/strict alias
        # keys. Reuse its two-rank host transport rather than weakening those
        # production checks for this older ownership-only fixture.
        path = Path(__file__).with_name("test_fsdp2_full_state_loading.py")
        spec = importlib.util.spec_from_file_location("_alias_full_state_contract", path)
        contracts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contracts)
        for flat in (False, True):
            with self.subTest(flat=flat):
                models, results, errors, hub = contracts.run_pair(flat=flat)
                self.assertEqual(errors, [None, None])
                for rank, model in enumerate(models):
                    self.assert_shared(model, model._fsdp_state.true_fsdp_params[0].shard)
                    self.assertEqual(model.embed.weight.values, [7, 8] if rank == 0 else [9, 10])


if __name__ == "__main__":
    unittest.main()
