"""HOST-only AST routing/owner contracts; no framework or native RNG execution.

Proposed replacement for compat/tests/structure/test_cpu_rng_state_routing.py.
Fake bytes below exercise Python routing only, never C++ blob validity.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]


class HostArray:
    def __init__(self, values):
        self.values = list(values)

    def copy(self):
        return HostArray(self.values)

    def tobytes(self):
        return bytes(self.values)


class Tensor:
    def __init__(self, data, dtype="uint8", device="cpu", shape=None):
        self.data = data.copy() if isinstance(data, HostArray) else HostArray(data)
        self.dtype = dtype
        self.shape = (len(self.data.values),) if shape is None else shape
        self.device = SimpleNamespace(type=device)

    def numpy(self):
        return self.data.copy()


def cpp_body(source, signature):
    """Find balanced function braces, not the first nested closing brace."""
    start = source.index(signature + " {") + len(signature) + 2
    depth = 1
    for i in range(start, len(source)):
        if source[i] == "{": depth += 1
        elif source[i] == "}": depth -= 1
        if depth == 0: return source[start:i]
    raise AssertionError("unterminated C++ function")


class TestCpuRngStateRouting(unittest.TestCase):
    def setUp(self):
        path = ROOT / "compat/torch/installers/core.py"
        names = {"_normalize_seed", "_manual_seed", "_get_rng_state", "_set_rng_state", "initial_seed"}
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in nodes}, names)
        self.calls = []
        self.blob = "JITTOR_RNG_1 cpu 0 18446744073709551615 0 1 2147483646 17"
        self.native_seed = (1 << 64) - 1
        self.generator = object()
        def tensor(data, *, dtype, device):
            self.calls.append(("tensor", dtype, device))
            return Tensor(data, dtype, device)
        def snapshot(backend):
            self.calls.append(("rng_state", backend)); return self.blob
        def restore(backend, payload):
            self.calls.append(("set_rng_state", backend, payload))
            if payload != self.blob: raise ValueError("native fixture rejected full blob")
        def initial(backend):
            self.calls.append(("rng_initial_seed", backend)); return self.native_seed
        def seed(value):
            self.calls.append(("_set_seed64", value)); self.native_seed = value
        self.native = SimpleNamespace(rng_state=snapshot, set_rng_state=restore,
                                      rng_initial_seed=initial, _set_seed64=seed)
        self.context = SimpleNamespace(state={"core_misc": {"seed": 7}},
            target_namespace=SimpleNamespace(tensor=tensor, uint8="uint8", default_generator=self.generator))
        env = {"jt": SimpleNamespace(core=self.native, Var=Tensor),
               "_misc_context": lambda: self.context, "_jittor_dtype_name": str,
               "_np": SimpleNamespace(uint8="uint8", frombuffer=lambda raw, dtype: HostArray(raw))}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
        self.api = env

    def test_manual_seed_full_uint64_and_signed_remap_returns_public_generator(self):
        for value, expected in ((0,0), (71,71), (-1,(1 << 64)-1),
                                (-(1 << 63),1 << 63), ((1 << 64)-1,(1 << 64)-1)):
            with self.subTest(value=value):
                self.calls.clear()
                self.assertIs(self.api["_manual_seed"](value), self.generator)
                self.assertEqual(self.calls, [("_set_seed64", expected)])
                self.assertEqual(self.context.state["core_misc"]["seed"], expected)

    def test_seed_overflow_rejected_before_owner_or_metadata_mutation(self):
        for value in (-(1 << 63)-1, 1 << 64, 1 << 80):
            with self.subTest(value=value):
                self.calls.clear(); before = dict(self.context.state["core_misc"])
                with self.assertRaises(ValueError): self.api["_manual_seed"](value)
                self.assertEqual(self.calls, []); self.assertEqual(self.context.state["core_misc"], before)

    def test_normalize_integer_contract_types_and_boundaries(self):
        fn = self.api["_normalize_seed"]
        for value in (True, False, 1.5, "1", object()):
            with self.subTest(kind=type(value).__name__):
                with self.assertRaises((TypeError, RuntimeError)): fn(value, integer=True)
        self.assertEqual(fn(-1, integer=True), (1 << 64)-1)
        self.assertEqual(self.calls, [])

    def test_get_copies_entire_blob_via_explicit_public_cpu_uint8(self):
        result = self.api["_get_rng_state"]()
        self.assertEqual(result.numpy().tobytes(), self.blob.encode("ascii"))
        self.assertEqual((result.dtype, result.device.type, result.shape), ("uint8", "cpu", (len(self.blob),)))
        self.assertEqual(self.calls, [("rng_state", "cpu"), ("tensor", "uint8", "cpu")])
        result.data.values[0] = 0
        self.assertEqual(self.api["_get_rng_state"]().numpy().tobytes(), self.blob.encode("ascii"))

    def test_restore_full_blob_and_read_back_seed_without_reseed(self):
        self.assertIsNone(self.api["_set_rng_state"](Tensor(self.blob.encode("ascii"))))
        self.assertEqual(self.calls, [("set_rng_state", "cpu", self.blob), ("rng_initial_seed", "cpu")])
        self.assertEqual(self.context.state["core_misc"]["seed"], (1 << 64)-1)

    def test_initial_seed_reads_canonical_cpu_owner_not_compat_cache(self):
        self.assertEqual(self.api["initial_seed"](), (1 << 64)-1)
        self.assertEqual(self.calls, [("rng_initial_seed", "cpu")])

    def test_invalid_api_values_never_reach_native_restore(self):
        cases = (7, [7], Tensor([7], "int64"), Tensor([7], shape=()),
                 Tensor([7], shape=(1,1)), Tensor([7], device="cuda"),
                 Tensor([7], device="npu"), Tensor([255]))
        for i, value in enumerate(cases):
            with self.subTest(case=i):
                self.calls.clear(); before = dict(self.context.state["core_misc"])
                with self.assertRaises((TypeError, ValueError)): self.api["_set_rng_state"](value)
                self.assertEqual(self.calls, []); self.assertEqual(self.context.state["core_misc"], before)

    def test_empty_and_foreign_blobs_are_validated_by_native_owner_not_seed_fallback(self):
        for raw in (b"", b"71", b"foreign state", b"JITTOR_CPU_RNG_V1\nlegacy"):
            with self.subTest(length=len(raw)):
                self.calls.clear(); before = dict(self.context.state["core_misc"])
                with self.assertRaisesRegex(ValueError, "native fixture"):
                    self.api["_set_rng_state"](Tensor(raw))
                self.assertEqual(self.calls, [("set_rng_state", "cpu", raw.decode("ascii"))])
                self.assertEqual(self.context.state["core_misc"], before)

    def test_missing_canonical_owner_is_not_replaced_by_legacy_api(self):
        for public, native, args in (("_get_rng_state","rng_state",()),
                                    ("initial_seed","rng_initial_seed",()),
                                    ("_manual_seed","_set_seed64",(1,))):
            with self.subTest(public=public):
                method = getattr(self.native,native); delattr(self.native,native)
                try:
                    with self.assertRaises(AttributeError): self.api[public](*args)
                finally: setattr(self.native,native,method)

    def test_cpp_cpu_restore_parses_before_sync_before_host_only_apply(self):
        src = (ROOT / "src/runtime/rng_state.cc").read_text()
        body = cpp_body(src, "void set_rng_state(const string& backend, const string& state, int device)")
        ordered = ("async_exec_wait()", "ExecutorEntryScope lock", "stream_device(backend, device)",
                   "parse_state(state, backend, device)", "sync_all(true)", "apply_state(backend, device, parsed)")
        self.assertEqual([body.index(x) for x in ordered], sorted(body.index(x) for x in ordered))
        apply = cpp_body(src, "void apply_state(const string& backend, int device, const ParsedState& state)")
        cpu = apply.split('if (backend == "cpu") {',1)[1].split("} else",1)[0]
        self.assertIn("owner.host_seed = state.seed;",cpu)
        self.assertIn("owner.host_engine = state.engine;",cpu)
        for forbidden in ("global_seed", "acl_stream", "offset", "callbacks", "set_seed", "seed_stream"):
            self.assertNotIn(forbidden,cpu)
        for forbidden in ("apply_global_seed", "callbacks", "set_seed(", "seed_stream("):
            self.assertNotIn(forbidden,body+apply)

    def test_cpp_snapshot_format_and_rejection_guards_are_canonical(self):
        src = (ROOT / "src/runtime/rng_state.cc").read_text()
        parse = cpp_body(src, "ParsedState parse_state(const string& payload, const string& backend, int device)")
        for guard in ('payload.empty()', 'payload.size() > 65536', 'version != "JITTOR_RNG_1"',
                      'saved_backend != backend', 'saved_device != device', 'result.offset < 0',
                      'result.offset != 0', 'minimum != std::default_random_engine::min()',
                      'maximum != std::default_random_engine::max()', '!input.eof()'):
            self.assertIn(guard,parse)
        for token in ("input >> minimum >> maximum >> std::ws", "input.tellg()",
                      "input >> raw_state", "raw_state < minimum", "raw_state > maximum",
                      "input.clear()", "input.seekg(engine_position)", "input >> result.engine"):
            self.assertIn(token, parse)
        self.assertLess(parse.index("raw_state > maximum"), parse.index("input >> result.engine"))
        encode = cpp_body(src,"string encode_state(const string& backend, int device)")
        for token in ('"JITTOR_RNG_1 "','owner.host_seed','owner.host_engine'):
            self.assertIn(token,encode)
        for name in ("rng_state","set_rng_state","rng_initial_seed"):
            self.assertIn("// @pyjt("+name+")",(ROOT / "src/runtime/rng_state.h").read_text())

    def test_legacy_v1_is_separate_native_api_not_torch_routing(self):
        src = (ROOT / "src/runtime/init.cc").read_text()
        self.assertIn("JITTOR_CPU_RNG_V1",src)
        self.assertIn("__GLIBCXX__",src); self.assertIn("_LIBCPP_VERSION",src)
        legacy = cpp_body(src,"void set_cpu_rng_state(const string& state)")
        for token in ("encode_cpu_rng_state(restored, seed) != state",
                      "raw < std::default_random_engine::min()",
                      "raw > std::default_random_engine::max()",
                      "rng.host_engine = restored;", "rng.host_seed = static_cast<uint64>(seed);"):
            self.assertIn(token,legacy)
        self.assertLess(legacy.index("encode_cpu_rng_state(restored, seed) != state"),legacy.index("async_exec_wait()"))
        self.assertLess(legacy.index("sync_all(true)"),legacy.index("rng.host_engine = restored;"))
        for forbidden in ("callbacks", "apply_global_seed", "rng.acl_streams", "rng.global_seed"):
            self.assertNotIn(forbidden,legacy)
        # This asserts only legacy-source continuity. Public routing above uses
        # rng_state("cpu")/set_rng_state("cpu", blob), never these V1 methods.

    def test_cpp_seed64_and_engine_access_use_single_runtime_owner(self):
        src = (ROOT / "src/runtime/init.cc").read_text()
        body = cpp_body(src,"void set_seed64(uint64 seed)")
        order = ("async_exec_wait()","ExecutorEntryScope lock","sync_all(true)","apply_global_seed(seed)")
        self.assertEqual([body.index(x) for x in order],sorted(body.index(x) for x in order))
        self.assertNotIn("static_cast<int>",body)
        apply = cpp_body(src,"static void apply_global_seed(uint64 seed)")
        for token in ("runtime_rng_state()","rng.global_seed = seed;","rng.host_seed = seed;",
                      "rng.host_engine.seed(seed);","rng.acl_streams.clear();","for (auto cb : callbacks)"):
            self.assertIn(token,apply)
        access = cpp_body(src,"std::default_random_engine* get_random_engine()")
        self.assertIn("return &runtime_rng_state().host_engine;",access)
        self.assertNotIn("eng.reset",access+body+apply)
        self.assertIn("@pyjt(_set_seed64)",(ROOT / "src/runtime/init.h").read_text())


if __name__ == "__main__": unittest.main()
