"""Full-state host contracts executing real loader bodies with two simulated ranks.

These test control flow and metadata, not real-device/distributed numerics.
"""
import ast
from collections import namedtuple
import contextlib
import copy
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
fixture = None
Tensor = None


def facade(provider):
    path = ROOT / "torch/installers/distributed.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_set_model_state_dict")
    ns = {"_fsdp_hooks": SimpleNamespace(provider=lambda: provider)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[node.name]


def options(**kwargs):
    values = dict(full_state_dict=True, cpu_offload=False, ignore_frozen_params=False,
                  keep_submodule_prefixes=True, strict=True,
                  broadcast_from_rank0=True, flatten_optimizer_state_dict=False)
    values.update(kwargs)
    return SimpleNamespace(**values)


def _initialize_fixture():
    global fixture, Tensor
    if fixture is not None:
        return
    spec = importlib.util.spec_from_file_location(
        "_fsdp_alias_fixture", Path(__file__).with_name("test_fsdp2_parameter_aliases.py"))
    loaded_fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded_fixture)

    class LoaderTensor(loaded_fixture.Var):
        def __init__(self, values, device="cpu", dtype="float32"):
            super().__init__(values)
            self.device_tag, self.dtype = device, dtype
            self.device_id = int(device.split(":")[-1]) if device != "cpu" else 0

        def to(self, *, device, dtype):
            return Tensor(self.values, device, dtype)

        def copy(self):
            return Tensor(self.values, self.device_tag, self.dtype)

        clone = copy

        def detach(self):
            return self

        def location(self):
            return "cpu" if self.device_tag == "cpu" else "device"

        def update(self, other):
            super().update(other)
            self.device_tag, self.dtype, self.device_id = other.device_tag, other.dtype, other.device_id

        assign = update
        _update = update

        def __getitem__(self, item):
            return Tensor(self.values[item], self.device_tag, self.dtype)

        def mpi_broadcast(self, rank):
            raise AssertionError("loader bypassed the public device collective")

    fixture, Tensor = loaded_fixture, LoaderTensor


class Hub:
    """Test-only rendezvous; every call must be entered by both ranks."""
    def __init__(self):
        self.barrier = threading.Barrier(2, timeout=3)
        self.values = {}
        self.indices = [0, 0]
        self.payloads = []

    def exchange(self, rank, kind, value):
        index = self.indices[rank]
        self.indices[rank] += 1
        self.values.setdefault(index, {})[rank] = (kind, value)
        self.barrier.wait()
        pair = self.values[index]
        if pair[0][0] != pair[1][0]:
            raise AssertionError("ranks entered different collectives")
        result = [pair[r][1] for r in range(2)]
        self.barrier.wait()
        return result

    def endpoint(self, rank):
        def gather(out, obj):
            out[:] = self.exchange(rank, "objects", copy.deepcopy(obj))
        def objects(out, src=0):
            out[:] = self.exchange(rank, "broadcast_objects", copy.deepcopy(out))[src]
        def tensor(value, src=0):
            assert value.location() == "device", "CPU payload entered device transport"
            self.payloads.append((rank, value.device_tag))
            received = self.exchange(rank, "broadcast_tensor", value.copy())[src]
            value.update(received.to(device=value.device_tag, dtype=value.dtype))
        return SimpleNamespace(all_gather_object=gather, broadcast_object_list=objects, broadcast=tensor)


def checkpoint(values=(7, 8, 9, 10)):
    _initialize_fixture()
    tensor = Tensor(values)
    return {"embed.weight": tensor, "head.weight": tensor}


def target(rank, hub, flat=False):
    ns = fixture.environment(flat)
    ns["common"]._rank = lambda: rank
    ns["jt"].Var = Tensor
    ns["jt"].flags = SimpleNamespace(device_id=rank, no_grad=0)
    ns["jt"].core = SimpleNamespace(dispatch_context=lambda values: (
        "acl" if values[0].location() == "device" else "cpu", values[0].device_id))
    def concat(values, dim=0):
        return Tensor([v for t in values for v in t.values], values[0].device_tag, values[0].dtype)
    ns["jt"].concat = concat
    ns["jt"].flag_scope = lambda **kwargs: contextlib.nullcontext()
    def split(value, sections, dim=0):
        self = []
        offset = 0
        for section in sections:
            self.append(Tensor(value.values[offset:offset + section], value.device_tag, value.dtype))
            offset += section
        return self
    ns["jt"].split = split
    ns["common"]._pad_flat = lambda t, n: Tensor(t.values + [0] * (n - t.numel()), t.device_tag, t.dtype)
    model = fixture.tied_module()
    model.embed.weight = model.head.weight = Tensor([1, 2, 3, 4])
    model._is_fsdp_module = True
    mesh = SimpleNamespace(ndim=1, device_type="npu",
                           mesh=SimpleNamespace(reshape=lambda shape: [0, 1]))
    group = SimpleNamespace(size=lambda: 2, rank=lambda: rank, ranks=None)
    model._fsdp_state = fixture.State(mesh=mesh, shard_group=group)
    for child in model.modules():
        child.named_buffers = lambda recurse=False, remove_duplicate=False, owner=child: [
            (key, value) for key, value in vars(owner).items()
            if isinstance(value, Tensor) and getattr(value, "is_buffer", False)]
    ns["_init_true_fsdp_state_impl"](model, model._fsdp_state)
    torch = SimpleNamespace(distributed=hub.endpoint(rank),
                            cuda=SimpleNamespace(current_device=lambda: rank),
                            npu=SimpleNamespace(current_device=lambda: rank),
                            empty=lambda shape, dtype, device:
                            Tensor([0] * shape[0], device, dtype))
    loader = dict(ns, shard=SimpleNamespace(**ns), dtype_name=lambda dtype: dtype,
                  get_install_context=lambda jt: SimpleNamespace(target_namespace=torch),
                  _VALIDATION_ERRORS=(RuntimeError, ValueError, TypeError, KeyError, NotImplementedError),
                  _LoadResult=namedtuple("_IncompatibleKeys", "missing_keys unexpected_keys"))
    fixture.load_functions("_state_dict.py", loader)
    return model, facade(SimpleNamespace(_load_full_state_dict=loader["_load_full_state_dict"]))


def run_pair(states=None, configs=None, flat=False, setup=None):
    _initialize_fixture()
    hub = Hub()
    targets = [target(rank, hub, flat) for rank in range(2)]
    if setup is not None:
        for rank, (model, load) in enumerate(targets):
            setup(rank, model)
    states = [checkpoint(), {}] if states is None else states
    configs = [options(), options()] if configs is None else configs
    results, errors = [None, None], [None, None]
    def run(rank):
        try:
            results[rank] = targets[rank][1](targets[rank][0], states[rank], options=configs[rank])
        except BaseException as error:
            errors[rank] = error
    threads = [threading.Thread(target=run, args=(rank,), daemon=True) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(8)
        if thread.is_alive():
            raise AssertionError("host collective deadlocked")
    return [item[0] for item in targets], results, errors, hub


class TestFSDPFullStateLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _initialize_fixture()

    def assert_ok(self, result, values=(7, 8, 9, 10)):
        models, results, errors, hub = result
        self.assertEqual(errors, [None, None])
        for rank, model in enumerate(models):
            self.assertEqual(model.embed.weight.values, list(values[rank * 2:rank * 2 + 2]))
            self.assertIs(model.embed.weight, model.head.weight)
            self.assertEqual(model.embed.weight.device_tag, "npu:%s" % rank)
            self.assertEqual(results[rank].missing_keys, [])
            self.assertEqual(results[rank].unexpected_keys, [])
        return models

    def assert_error(self, result, text, kind=RuntimeError):
        models, results, errors, hub = result
        self.assertTrue(all(isinstance(error, kind) for error in errors), errors)
        self.assertEqual(str(errors[0]), str(errors[1]))
        self.assertIn(text, str(errors[0]))
        self.assertFalse(hub.payloads)
        for rank, model in enumerate(models):
            self.assertEqual(model.embed.weight.values, [1, 2] if rank == 0 else [3, 4])

    def test_public_options_reach_owner_and_return_result(self):
        calls, result = [], object()
        def load(*args, **kwargs):
            calls.append((args, kwargs))
            return result
        config = options()
        actual = facade(SimpleNamespace(_load_full_state_dict=load))(
            SimpleNamespace(_is_fsdp_module=True), {}, options=config)
        self.assertIs(calls[0][1].get("options"), config)
        self.assertIs(actual, result)

    def test_load_plan_uses_public_mesh_device_ordinal_not_ambient_flag(self):
        path = ROOT / "fsdp2/_state_dict.py"
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_load_plan")
        for kind, current, raw_flag, expected in (
                ("cuda", 0, -1, "cuda:0"),
                ("cuda", 2, 7, "cuda:2"),
                ("npu", 1, -1, "npu:1"),
                ("cuda", -1, 2, None),
                ("cpu", None, -1, "cpu")):
            with self.subTest(kind=kind, current=current, raw_flag=raw_flag):
                calls = []
                mesh = SimpleNamespace(ndim=1, device_type=kind,
                                       mesh=SimpleNamespace(reshape=lambda shape: [0]))
                state = SimpleNamespace(true_fsdp_initialized=True, true_fsdp_unsharded=False,
                                        true_fsdp_params=[], mesh=mesh, shard_group=SimpleNamespace(
                                            size=lambda: 1, ranks=None),
                                        true_fsdp_rank=0, true_fsdp_world_size=1)
                model = SimpleNamespace(_fsdp_state=state, named_buffers=lambda **kwargs: [])
                model.named_modules = lambda: [("", model)]
                def query_device(name):
                    calls.append(name)
                    self.assertEqual(name, kind)
                    return current
                torch = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: query_device("cuda")),
                                        npu=SimpleNamespace(current_device=lambda: query_device("npu")))
                flags = SimpleNamespace(device_id=raw_flag)
                namespace = {"jt": SimpleNamespace(flags=flags),
                             "common": SimpleNamespace(_world_size=lambda: 1, _rank=lambda: 0),
                             "shard": SimpleNamespace(_iter_fsdp_modules=lambda module, recurse: [module]),
                             "get_install_context": lambda jt: SimpleNamespace(target_namespace=torch)}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
                if expected is None:
                    with self.assertRaisesRegex(RuntimeError, "no current cuda device"):
                        namespace["_load_plan"](model)
                else:
                    targets, states, device = namespace["_load_plan"](model)
                    self.assertEqual(device, expected)
                    self.assertEqual(targets, [])
                    self.assertEqual(states, [state])
                self.assertEqual(calls, [] if kind == "cpu" else [kind])
                self.assertEqual(flags.device_id, raw_flag, "planning must not mutate ambient flags")

    def test_rank0_only_cpu_dictionary_loads_both_shards_and_flat_storage(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                result = run_pair(flat=flat)
                models = self.assert_ok(result)
                self.assertEqual(len(result[3].payloads), 2)
                if flat:
                    for rank, model in enumerate(models):
                        self.assertEqual(model._fsdp_state.true_fsdp_flat_shard.device_tag, "npu:%s" % rank)

    def test_without_broadcast_uses_each_local_dictionary(self):
        result = run_pair([checkpoint(), checkpoint((11, 12, 13, 14))],
                          [options(broadcast_from_rank0=False)] * 2)
        self.assertEqual(result[2], [None, None])
        self.assertEqual(result[0][0].embed.weight.values, [7, 8])
        self.assertEqual(result[0][1].embed.weight.values, [13, 14])
        self.assertFalse(result[3].payloads)

    def test_load_retains_existing_parameter_holders_and_frozen_state(self):
        for flat in (False, True):
            held = []
            def setup(rank, model):
                held.append(model.embed.weight)
                model.embed.weight.stop_grad()
            with self.subTest(flat=flat):
                result = run_pair(flat=flat, setup=setup)
                models = self.assert_ok(result)
                for rank, model in enumerate(models):
                    self.assertIs(held[rank], model.embed.weight,
                                  "an optimizer created before loading must keep its parameter")
                    self.assertTrue(model.embed.weight.is_stop_grad())

    def _assert_live_state_rejected_before_payload(self, kind):
        for flat in (False, True):
            for live_rank in (0, 1):
                with self.subTest(flat=flat, live_rank=live_rank, state=kind):
                    held, copies = {}, []
                    class ObservedSource(Tensor):
                        def to(self, **kwargs):
                            copies.append(kwargs)
                            return super().to(**kwargs)
                    def setup(rank, model):
                        state = model._fsdp_state
                        entry = state.true_fsdp_params[0]
                        if rank == live_rank:
                            if kind == "full_param":
                                entry.full_param = Tensor([31, 32, 33, 34], "npu:%s" % rank)
                            elif kind == "unsharded":
                                state.true_fsdp_unsharded = True
                            else:
                                entry.shard._torch_grad = Tensor([41, 42], "npu:%s" % rank)
                        held[rank] = (entry.shard, getattr(entry, "full_param", None),
                                      getattr(entry.shard, "_torch_grad", None),
                                      state.true_fsdp_unsharded, entry.shard.device_tag,
                                      list(state.true_fsdp_flat_shard.values) if flat else None)
                    state_dict = checkpoint()
                    state_dict["head.weight"] = ObservedSource([7, 8, 9, 10])
                    result = run_pair([state_dict, {}], flat=flat, setup=setup)
                    self.assert_error(result, "quiescent", NotImplementedError)
                    self.assertEqual(result[1], [None, None])
                    self.assertIn("rank %s" % live_rank, str(result[2][0]))
                    self.assertEqual(result[3].indices, [1, 1],
                                     "reject collectively in initial validation, before manifest or payload")
                    self.assertEqual(copies, [], "do not even stage a checkpoint device copy")
                    for rank, model in enumerate(result[0]):
                        state = model._fsdp_state
                        entry = state.true_fsdp_params[0]
                        parameter, full, grad, unsharded, device, flat_values = held[rank]
                        self.assertIs(model.embed.weight, parameter)
                        self.assertIs(model.head.weight, parameter)
                        self.assertIs(entry.full_param, full)
                        self.assertIs(getattr(parameter, "_torch_grad", None), grad)
                        self.assertEqual(state.true_fsdp_unsharded, unsharded)
                        self.assertEqual(parameter.device_tag, device)
                        if flat:
                            self.assertEqual(state.true_fsdp_flat_shard.values, flat_values)

    def test_live_forward_parameters_reject_collectively_without_mutation(self):
        self._assert_live_state_rejected_before_payload("full_param")

    def test_unsharded_state_rejects_collectively_without_mutation(self):
        self._assert_live_state_rejected_before_payload("unsharded")

    def test_pending_gradient_rejects_collectively_without_mutation(self):
        self._assert_live_state_rejected_before_payload("gradient")

    def test_alias_last_module_path_wins_and_source_is_not_modified(self):
        state = checkpoint()
        state["head.weight"] = Tensor([21, 22, 23, 24], dtype="float64")
        result = run_pair([state, {}])
        models = self.assert_ok(result, (21, 22, 23, 24))
        self.assertEqual(models[0].embed.weight.dtype, "float32")
        self.assertEqual(state["head.weight"].device_tag, "cpu")
        self.assertEqual(state["head.weight"].dtype, "float64")

    def test_strict_missing_alias_extra_and_shape_errors_are_collective(self):
        for kind in ("missing", "extra", "shape"):
            state = checkpoint()
            if kind == "missing":
                del state["head.weight"]
            elif kind == "extra":
                state["extra"] = Tensor([0])
            else:
                state["head.weight"] = Tensor([1])
            with self.subTest(kind=kind):
                self.assert_error(run_pair([state, {}]), "shape mismatch" if kind == "shape" else "keys")

    def test_nonroot_validation_failure_reaches_both_ranks(self):
        result = run_pair([checkpoint(), {}], [options(broadcast_from_rank0=False)] * 2)
        self.assert_error(result, "rank 1")

    def test_persistent_buffer_and_nonpersistent_exclusion(self):
        def setup(rank, model):
            for name in ("running", "rope"):
                value = Tensor([0])
                value.is_buffer = True
                value.persistent = name == "running"
                setattr(model, name, value)
        state = checkpoint()
        state["running"] = Tensor([11])
        result = run_pair([state, {}], setup=setup)
        self.assert_ok(result)
        for rank, model in enumerate(result[0]):
            self.assertEqual(model.running.values, [11])
            self.assertEqual(model.running.device_tag, "npu:%s" % rank)
            self.assertEqual(model.rope.values, [0])

    def test_unsupported_options_fail_closed_on_all_ranks(self):
        for key, value in (("full_state_dict", False), ("strict", False),
                           ("cpu_offload", True), ("ignore_frozen_params", True),
                           ("flatten_optimizer_state_dict", True), ("keep_submodule_prefixes", False)):
            with self.subTest(option=key):
                self.assert_error(run_pair(configs=[options(), options(**{key: value})]),
                                  key, NotImplementedError)

    def test_group_mesh_and_schema_mismatch_fail_before_payload(self):
        def subgroup(rank, model):
            if rank == 1:
                model._fsdp_state.shard_group.ranks = (1,)
        self.assert_error(run_pair(setup=subgroup), "WORLD", NotImplementedError)
        def mesh(rank, model):
            if rank == 1:
                model._fsdp_state.mesh.ndim = 2
        self.assert_error(run_pair(setup=mesh), "one-dimensional", NotImplementedError)
        def schema(rank, model):
            if rank == 1:
                model._fsdp_state.true_fsdp_params[0].dtype = "float64"
        self.assert_error(run_pair(setup=schema), "schemas")

    def test_failed_actual_device_copy_is_collective_error(self):
        class BadCopy(Tensor):
            def to(self, **kwargs):
                return Tensor(self.values)
        state = checkpoint()
        state["head.weight"] = BadCopy([7, 8, 9, 10])
        self.assert_error(run_pair([state, {}]), "requested device")

    def test_default_internal_options_keep_full_rank0_contract(self):
        self.assert_ok(run_pair(configs=[None, None]))

    def test_failed_dtype_conversion_is_rejected_before_payload(self):
        class BadDtype(Tensor):
            def to(self, *, device, dtype):
                return Tensor(self.values, device, "float64")
        state = checkpoint()
        state["head.weight"] = BadDtype([7, 8, 9, 10])
        self.assert_error(run_pair([state, {}]), "destination parameter dtype")

    def test_mutation_failure_is_not_reported_as_success_by_other_rank(self):
        def setup(rank, model):
            if rank == 1:
                def fail(value):
                    raise RuntimeError("simulated allocation failure")
                model._fsdp_state.true_fsdp_params[0].shard.update = fail
        result = run_pair(setup=setup)
        errors = result[2]
        self.assertTrue(all(isinstance(error, RuntimeError) for error in errors), errors)
        self.assertEqual(str(errors[0]), str(errors[1]))
        self.assertIn("simulated allocation failure", str(errors[0]))


class TestFSDPFullStateShapePublication(unittest.TestCase):
    """Real loader/init bodies, shape-aware fake storage; no Jittor import."""

    @classmethod
    def setUpClass(cls):
        _initialize_fixture()

    def setup_shapes(self, world_size=1, flat=False, frozen=False, public_frozen=False):
        class ShapeTensor(Tensor):
            def __init__(self, values, shape=None, device="cpu", dtype="float32"):
                super().__init__(values, device, dtype)
                self.native_stopped = False
                self.shape = tuple(shape) if shape is not None else (len(self.values),)
                count = 1
                for size in self.shape:
                    count *= size
                if count != len(self.values):
                    raise ValueError("fixture shape does not match values")

            def reshape(self, shape):
                shape = (len(self.values),) if tuple(shape) == (-1,) else tuple(shape)
                return ShapeTensor(self.values, shape, self.device_tag, self.dtype)

            def copy(self):
                return ShapeTensor(self.values, self.shape, self.device_tag, self.dtype)

            clone = copy

            def __getitem__(self, item):
                return ShapeTensor(self.values[item], device=self.device_tag, dtype=self.dtype)

            def update(self, other):
                super().update(other)
                self.shape = tuple(other.shape)

            _update = update

            def stop_grad(self):
                self.native_stopped = True
                self.requires_grad = False
                return self

            def start_grad(self):
                self.native_stopped = False
                self.requires_grad = True
                return self

            def is_stop_grad(self):
                return self.native_stopped

        ns = fixture.environment(flat)
        ns["jt"].Var = ShapeTensor
        ns["jt"].core = SimpleNamespace(dispatch_context=lambda values: ("cpu", 0))
        ns["jt"].concat = lambda values, dim=0: ShapeTensor(
            [item for value in values for item in value.values],
            device=values[0].device_tag, dtype=values[0].dtype)
        ns["common"]._world_size = lambda: world_size
        ns["common"]._flatten_var = lambda value: value.reshape((-1,))
        ns["common"]._pad_flat = lambda value, size: ShapeTensor(
            value.values + [0] * (size - value.numel()),
            device=value.device_tag, dtype=value.dtype)
        model = fixture.Module()
        model.embed, model.head = fixture.Module(), fixture.Module()
        model.embed.weight = model.head.weight = ShapeTensor(range(6), (2, 3))
        model.bias = ShapeTensor([0], ())
        model.vector = ShapeTensor([0, 0], (2,))
        if frozen:
            for value in (model.embed.weight, model.bias, model.vector):
                value.stop_grad()
        if public_frozen:
            for value in (model.embed.weight, model.bias, model.vector):
                # Public reversible disabling must not imply native stop.
                value.requires_grad = False
        model._is_fsdp_module = True
        model._fsdp_state = fixture.State()
        ns["_init_true_fsdp_state_impl"](model, model._fsdp_state)
        scope = dict(ns, shard=SimpleNamespace(**ns))
        fixture.load_functions("_state_dict.py", scope)
        state = model._fsdp_state
        targets, full_values, originals = [], {}, {}
        for entry in state.true_fsdp_params:
            target = dict(state=state, entry=entry)
            targets.append(target)
            source = ShapeTensor([10 + i for i in range(entry.numel)], entry.shape)
            full_values[id(target)] = source
            originals[entry.name] = (entry.shard, source, tuple(entry.shard.shape))
        return model, state, scope, targets, full_values, originals

    def test_singleton_load_preserves_matrix_scalar_vector_shape_and_identity(self):
        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                model, state, scope, targets, values, held = self.setup_shapes(frozen=frozen)
                self.assertFalse(state.true_fsdp_flat)
                self.assertEqual(state.true_fsdp_world_size, 1)
                self.assertEqual(model.embed.weight.shape, (2, 3))
                self.assertEqual(model.bias.shape, ())
                optimizer_parameters = [entry.shard for entry in state.true_fsdp_params]
                # Repeated loads must neither flatten nor replace a holder.
                for _ in range(2):
                    scope["_apply_full_values"](targets, [state], values, "cpu")
                    for index, entry in enumerate(state.true_fsdp_params):
                        with self.subTest(parameter=entry.name):
                            original, source, original_shape = held[entry.name]
                            self.assertIs(entry.shard, original)
                            self.assertIs(entry.shard, optimizer_parameters[index])
                            self.assertEqual(tuple(entry.shard.shape), tuple(entry.shape))
                            self.assertEqual(tuple(entry.shard.shape), original_shape)
                            self.assertEqual(entry.shard.values, source.values)
                            self.assertEqual(entry.shard.device_tag, "cpu")
                            self.assertEqual(entry.shard.dtype, "float32")
                            self.assertEqual(entry.shard.is_stop_grad(), frozen)
                            self.assertEqual(source.shape, entry.shape)
                            self.assertEqual(source.device_tag, "cpu")
                            self.assertIsNone(entry.full_param)
                    self.assertIs(model.embed.weight, model.head.weight)
                    self.assertFalse(state.true_fsdp_unsharded)

    def test_multirank_nonflat_and_flat_local_layouts_are_not_reshaped_global(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                model, state, scope, targets, values, held = self.setup_shapes(
                    world_size=2, flat=flat)
                local_shapes = [tuple(entry.shard.shape) for entry in state.true_fsdp_params]
                scope["_apply_full_values"](targets, [state], values, "cpu")
                for entry, local_shape in zip(state.true_fsdp_params, local_shapes):
                    self.assertIs(entry.shard, held[entry.name][0])
                    self.assertEqual(tuple(entry.shard.shape), local_shape)
                    self.assertEqual(len(local_shape), 1)
                    self.assertEqual(entry.shard.device_tag, "cpu")
                    self.assertFalse(entry.shard.is_stop_grad())
                self.assertIs(model.embed.weight, model.head.weight)

    def test_singleton_load_preserves_public_freeze_when_native_stop_is_false(self):
        model, state, scope, targets, values, held = self.setup_shapes(public_frozen=True)
        for entry in state.true_fsdp_params:
            self.assertFalse(entry.shard.requires_grad)
            self.assertFalse(entry.shard.is_stop_grad(), "fixture must keep the two flags independent")
            self.assertFalse(entry.requires_grad)
        for iteration in range(2):
            scope["_apply_full_values"](targets, [state], values, "cpu")
            for entry in state.true_fsdp_params:
                with self.subTest(iteration=iteration, parameter=entry.name):
                    original, source, shape = held[entry.name]
                    self.assertIs(entry.shard, original)
                    self.assertFalse(entry.shard.requires_grad)
                    self.assertFalse(entry.requires_grad)
                    self.assertEqual(entry.shard.shape, shape)
                    self.assertEqual(entry.shard.values, source.values)
                    self.assertEqual(entry.shard.device_tag, "cpu")
            self.assertIs(model.embed.weight, model.head.weight)


if __name__ == "__main__":
    unittest.main()
