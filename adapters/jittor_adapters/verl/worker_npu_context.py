"""Initialize ACL context in the actual verl worker execution thread.

Ray may start its Python worker before assigning accelerator visibility, and
ACL's current-device context is thread-local.  A process-level bootstrap is
therefore insufficient for verl's colocated ``WorkerDict`` construction.  The
pinned Worker initializer is the first downstream-owned point after Ray has
installed the worker environment and before PEFT/model modules allocate a
tensor, so establish the logical NPU there.
"""
from functools import wraps
import hashlib
import inspect
import numpy as np
import os
from pathlib import Path
import sys
import threading
import types

from .._common import require_version, required_patch, replace, UnsupportedAdapterVersion


SUPPORTED_VERSIONS = frozenset(("0.9.1",))
MODULE = "verl.single_controller.base.worker"
SOURCE_SHA256 = "385af797001667d4b812db85dac709b7150dd21243ec8493c5f92210f6d795df"
INIT_SHA256 = "47614b42dc4015eeee0d90ca0859ef20352c43dc35faf66273987a6cf6ac6d7e"
POLICY = "initialize_acl_after_ray_worker_environment_v2"
RAY_NOSET_ASCEND = "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"
AGENT_MODULE = "verl.experimental.agent_loop.agent_loop"
AGENT_SOURCE_SHA256 = "eb2196253a0b5392432b509fb4e4e21da8af4842e6f0a950f52bbd130d5834de"
AGENT_INIT_SHA256 = "8228262721996925f9cb7ce82026bdb419db3edd89310a59adb1c8358854cc74"
AGENT_POSTPROCESS_SHA256 = "d55eabeea2dc52a69666bfccaeee1c2c381d732ee8cd183e57cd6744733f12bf"
AGENT_POLICY = "initialize_acl_then_keep_agent_loop_tensor_factories_on_cpu_v2"
AGENT_REWARD_SCORE_POLICY = "replace_lazy_indexed_rm_scores_with_host_tensor_v1"
AGENT_POSTPROCESS_ORIGINAL = "_jittor_verl_original_agent_loop_postprocess_v091"
TOKENIZER_MODULE = "verl.utils.tokenizer.continuous_token"
TOKENIZER_SOURCE_SHA256 = "29aefacb6a17877b0f5955ab3838e0196d6a440f04472541b61cfeedb968dbfd"
TOKENIZER_INIT_SHA256 = "7fa86da170bdad109ed6354ae938be26ed87627faa1e95f2336c369df7d216c8"
TOKENIZER_POLICY = "initialize_acl_in_vl_tokenizer_executor_thread_v1"
REWARD_MODULE = "verl.experimental.reward_loop.reward_loop"
REWARD_SOURCE_SHA256 = "f30932e2445ab01ca7fa4fbcec13910c85bfaf2cd8a4f49d040f15d7fe2ed973"
REWARD_INIT_SHA256 = "89d97a377ad90c089faf1707e2a734f41de182dd1754e74ad7320974f4de9de5"
REWARD_POLICY = "initialize_acl_then_keep_reward_loop_tensor_factories_on_cpu_v1"
REWARD_DECODE_POLICY = "initialize_acl_in_reward_tokenizer_executor_thread_v1"
RAY_UTILS_MODULE = "verl.utils.ray_utils"
RAY_UTILS_SOURCE_SHA256 = "1f70a9ac05b4a6f92a3ad2486bee17ac9721a9568e7e72b17e76172a60a7751b"
RAY_PARALLEL_PUT_SHA256 = "a9ffa1c0a5e7fd182e44d965c2fc9e03a96d9c5c05ceb0b331aa6a205733c2d4"
RAY_PARALLEL_PUT_POLICY = "initialize_acl_in_parallel_ray_put_thread_v1"
PROTOCOL_MODULE = "verl.protocol"
PROTOCOL_SOURCE_SHA256 = "8d69430131e9e894a6c7aea673d489e19e44117b1bc573085a6db521b38c6792"
PROTOCOL_METHOD_SHA256 = {
    "__getstate__": "0325671e24f858bb8abed62e5d04dab0aa941ea6c5b3c8851b82470cabfdd7a2",
    "__setstate__": "cf3177eea4f0f56b7247b5701481e9887a269b4463ae891ce8464917c423c986",
}
PROTOCOL_SERIALIZER_SHA256 = "1d3acc35f611f2a66ef257d8bafe75a8fc90e14e4f973766b1de2cf5c11d60e0"
PROTOCOL_BATCH_CHUNK_SHA256 = "64e9f8d67c3b88c1f8f9a4a070457ca15eaecf6e7ced7066ae498d5608c286ec"
PROTOCOL_POLICY = "initialize_acl_at_dataproto_serialization_boundary_v1"
PROTOCOL_SHAPE_POLICY = "normalize_dataproto_serialized_shape_to_tuple_v1"
PROTOCOL_HOST_BYTES_POLICY = "materialize_dataproto_bytes_after_numpy_boundary_v1"
PROTOCOL_NESTED_CHUNK_POLICY = "skip_tensordict_consolidate_for_public_nested_type_v1"
_DATAPROTO_IO_LOCK = threading.RLock()


def _trace(message):
    if os.environ.get("JITTOR_VERL_FATAL_TRACE") == "1":
        print("JITTOR_VERL_TRACE " + message, file=sys.stderr, flush=True)


def _check_source(module, initializer):
    try:
        filename = inspect.getsourcefile(module)
        source = Path(filename).read_bytes() if filename else b""
        function_source = inspect.getsource(initializer).encode()
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("verl Worker source is required") from error
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise UnsupportedAdapterVersion("verl Worker module source changed")
    if hashlib.sha256(function_source).hexdigest() != INIT_SHA256:
        raise UnsupportedAdapterVersion("verl Worker.__init__ source changed")


def _require_frontend():
    from .native_npu import enabled

    if not enabled():
        raise UnsupportedAdapterVersion("worker NPU context policy requires explicit native-NPU opt-in")
    require_version("verl", SUPPORTED_VERSIONS)
    from jittor.compat.torch import namespace_owner

    frontend = sys.modules.get("torch")
    if frontend is None or namespace_owner(frontend) is None:
        raise UnsupportedAdapterVersion("worker NPU context policy requires the Jittor Torch namespace")
    return frontend


def _require_runtime():
    frontend = _require_frontend()
    npu = getattr(frontend, "npu", None)
    if not callable(getattr(npu, "set_device", None)):
        raise UnsupportedAdapterVersion("Jittor Torch namespace has no NPU device owner")
    return npu


def _check_agent_source(module, initializer, postprocess):
    try:
        filename = inspect.getsourcefile(module)
        source = Path(filename).read_bytes() if filename else b""
        function_source = inspect.getsource(initializer).encode()
        postprocess_source = inspect.getsource(postprocess).encode()
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("verl AgentLoopWorker source is required") from error
    if hashlib.sha256(source).hexdigest() != AGENT_SOURCE_SHA256:
        raise UnsupportedAdapterVersion("verl AgentLoopWorker module source changed")
    if hashlib.sha256(function_source).hexdigest() != AGENT_INIT_SHA256:
        raise UnsupportedAdapterVersion("verl AgentLoopWorker.__init__ source changed")
    if hashlib.sha256(postprocess_source).hexdigest() != AGENT_POSTPROCESS_SHA256:
        raise UnsupportedAdapterVersion("verl AgentLoopWorker._postprocess source changed")


def _check_tokenizer_source(module, initializer):
    try:
        filename = inspect.getsourcefile(module)
        source = Path(filename).read_bytes() if filename else b""
        function_source = inspect.getsource(initializer).encode()
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("verl VL tokenizer source is required") from error
    if hashlib.sha256(source).hexdigest() != TOKENIZER_SOURCE_SHA256:
        raise UnsupportedAdapterVersion("verl continuous-token module source changed")
    if hashlib.sha256(function_source).hexdigest() != TOKENIZER_INIT_SHA256:
        raise UnsupportedAdapterVersion("verl VL build_initial_tokens source changed")


def _check_reward_source(module, initializer):
    try:
        filename = inspect.getsourcefile(module)
        source = Path(filename).read_bytes() if filename else b""
        function_source = inspect.getsource(initializer).encode()
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("verl RewardLoopWorker source is required") from error
    if hashlib.sha256(source).hexdigest() != REWARD_SOURCE_SHA256:
        raise UnsupportedAdapterVersion("verl reward-loop module source changed")
    if hashlib.sha256(function_source).hexdigest() != REWARD_INIT_SHA256:
        raise UnsupportedAdapterVersion("verl RewardLoopWorker.__init__ source changed")


def _check_ray_utils_source(module, parallel_put):
    try:
        filename = inspect.getsourcefile(module)
        source = Path(filename).read_bytes() if filename else b""
        function_source = inspect.getsource(parallel_put).encode()
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("verl ray_utils source is required") from error
    if hashlib.sha256(source).hexdigest() != RAY_UTILS_SOURCE_SHA256:
        raise UnsupportedAdapterVersion("verl ray_utils module source changed")
    if hashlib.sha256(function_source).hexdigest() != RAY_PARALLEL_PUT_SHA256:
        raise UnsupportedAdapterVersion("verl parallel_put source changed")


def _check_protocol_source(module, methods, serializer, batch_chunk):
    try:
        filename = inspect.getsourcefile(module)
        source = Path(filename).read_bytes() if filename else b""
        method_sources = {name: inspect.getsource(method).encode() for name, method in methods.items()}
        serializer_source = inspect.getsource(serializer).encode()
        batch_chunk_source = inspect.getsource(batch_chunk).encode()
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("verl DataProto source is required") from error
    if hashlib.sha256(source).hexdigest() != PROTOCOL_SOURCE_SHA256:
        raise UnsupportedAdapterVersion("verl protocol module source changed")
    for name, method_source in method_sources.items():
        if hashlib.sha256(method_source).hexdigest() != PROTOCOL_METHOD_SHA256[name]:
            raise UnsupportedAdapterVersion("verl DataProto source changed: " + name)
    if hashlib.sha256(serializer_source).hexdigest() != PROTOCOL_SERIALIZER_SHA256:
        raise UnsupportedAdapterVersion("verl serialize_single_tensor source changed")
    if hashlib.sha256(batch_chunk_source).hexdigest() != PROTOCOL_BATCH_CHUNK_SHA256:
        raise UnsupportedAdapterVersion("verl BatchData.chunk source changed")


def _logical_device():
    visible = [item.strip() for item in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
               if item.strip()]
    # A normal Ray worker receives one physical card and therefore addresses it
    # as logical zero.  Jittor's Ray integration deliberately retains the full
    # parent mask, however, because Ray applies its one-card mask only after the
    # worker executable has already imported CANN.  In that mode derive the
    # logical device from the rank metadata that verl injects in runtime_env.
    if len(visible) <= 1:
        return 0 if visible else int(os.environ.get("LOCAL_RANK", "0"))
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = int(os.environ.get("RANK", "0"))
        local_world_size = int(os.environ.get("RAY_LOCAL_WORLD_SIZE", str(len(visible))))
        local_rank = rank % local_world_size
    if not 0 <= local_rank < len(visible):
        raise RuntimeError(
            "logical NPU rank {} is outside ASCEND_RT_VISIBLE_DEVICES={}".format(
                local_rank, ",".join(visible)
            )
        )
    return local_rank


def _make_initializer(original):
    @wraps(original)
    def initialize_worker_and_acl(self, *args, **kwargs):
        logical_device = _logical_device()
        # With Ray's NOSET switch enabled, verl asks Ray for the assigned
        # accelerator id and passes that *physical* id straight to
        # torch.npu.set_device().  Jittor, like PyTorch after applying a
        # visibility mask, accepts the logical index instead.  Skip only that
        # upstream branch while Worker.__init__ runs, provide the correct
        # logical LOCAL_RANK to its store, then restore NOSET so later Ray code
        # continues to preserve the full mask.
        noset_value = os.environ.pop(RAY_NOSET_ASCEND, None)
        os.environ["LOCAL_RANK"] = str(logical_device)
        try:
            result = original(self, *args, **kwargs)
        finally:
            if noset_value is not None:
                os.environ[RAY_NOSET_ASCEND] = noset_value
        _require_runtime().set_device(logical_device)
        return result

    initialize_worker_and_acl._jittor_verl_worker_npu_context = POLICY
    initialize_worker_and_acl._jittor_verl_original = original
    return initialize_worker_and_acl


def _make_agent_initializer(original):
    @wraps(original)
    def initialize_agent_on_cpu(self, *args, **kwargs):
        # AgentLoopWorker owns tokenization and batch assembly only; Ray gives
        # it CPU resources and no accelerator assignment.  The shared worker
        # executable enables Jittor's accelerator globally for training actors,
        # so explicitly restore PyTorch's CPU-default factory semantics here.
        # Otherwise processor-created torch.tensor(...) queries ACL device 0
        # despite this actor having no ACL context or NPU reservation.
        # Changing Jittor's global accelerator flag queries the current ACL
        # device while preserving/restoring the index.  Establish that context
        # first; Ray inherits ASCEND_RT_VISIBLE_DEVICES from this single-card
        # job, so logical device 0 is the same card used by the colocated
        # training/rollout workers.
        _require_runtime().set_device(_logical_device())
        if os.environ.get("JITTOR_VERL_FATAL_TRACE") == "1":
            import faulthandler
            faulthandler.enable(all_threads=True)
        frontend = _require_frontend()
        setter = getattr(frontend, "set_default_device", None)
        if not callable(setter):
            raise UnsupportedAdapterVersion("Jittor Torch namespace has no default-device owner")
        setter("cpu")
        return original(self, *args, **kwargs)

    initialize_agent_on_cpu._jittor_verl_agent_loop_cpu_context = AGENT_POLICY
    initialize_agent_on_cpu._jittor_verl_original = original
    return initialize_agent_on_cpu


def _make_agent_postprocess(original):
    def postprocess_with_host_rm_scores(self, inputs, *args, **kwargs):
        module = __import__(
            "verl.experimental.agent_loop.agent_loop",
            fromlist=("_jittor_verl_original_agent_loop_postprocess_v091",),
        )
        upstream = getattr(module, "_jittor_verl_original_agent_loop_postprocess_v091")
        numpy = __import__("numpy")
        result = upstream(self, inputs, *args, **kwargs)
        scores = [item.reward_score for item in inputs]
        if not scores or not all(score is not None for score in scores):
            return result

        # Upstream builds rm_scores with a lazy zeros_like followed by advanced
        # indexed assignment.  In the async CPU-only AgentLoop actor that graph
        # SIGBUSes only when it is later materialized for Ray transport.  Keep
        # the exact placement rule, but evaluate it in host NumPy and import an
        # owned CPU tensor.  All other upstream post-processing remains intact.
        batch = result.batch
        attention_mask = numpy.asarray(batch["attention_mask"].numpy())
        prompt_length = int(batch["prompts"].shape[1])
        response_length = attention_mask[:, prompt_length:].sum(axis=1).astype(numpy.int64) - 1
        shape = tuple(batch["response_mask"].shape)
        rm_scores = numpy.zeros(shape, dtype=numpy.float32)
        rm_scores[numpy.arange(shape[0]), response_length] = numpy.asarray(scores, dtype=numpy.float32)
        # Resolve the frontend by name inside the Ray actor. Capturing the
        # adapter's TorchNamespace here makes Ray cloudpickle traverse an
        # intentionally non-picklable runtime owner while serializing the
        # dynamically modified actor class.
        batch["rm_scores"] = __import__("torch").from_numpy(rm_scores)
        return result

    postprocess_with_host_rm_scores.__name__ = original.__name__
    postprocess_with_host_rm_scores.__qualname__ = original.__qualname__
    postprocess_with_host_rm_scores.__module__ = original.__module__
    postprocess_with_host_rm_scores.__doc__ = original.__doc__
    postprocess_with_host_rm_scores._jittor_verl_agent_loop_rm_scores = AGENT_REWARD_SCORE_POLICY
    return postprocess_with_host_rm_scores


def _make_tokenizer_thread_initializer(original):
    @wraps(original)
    def build_tokens_with_acl_context(self, *args, **kwargs):
        # AgentLoopBase dispatches this method through run_in_executor(). ACL's
        # current-device state is thread-local, so the actor constructor cannot
        # initialize the executor thread on its behalf.
        _require_runtime().set_device(_logical_device())
        return original(self, *args, **kwargs)

    build_tokens_with_acl_context._jittor_verl_tokenizer_thread_context = TOKENIZER_POLICY
    build_tokens_with_acl_context._jittor_verl_original = original
    return build_tokens_with_acl_context


def _make_reward_initializer(original):
    @wraps(original)
    def initialize_reward_on_cpu(self, *args, **kwargs):
        # RewardLoopWorker is a CPU-only Ray actor, but DataProto arguments are
        # reconstructed through torch.load before compute_score* is invoked.
        # Jittor's tensor rebuild queries ACL even for the subsequent CPU move,
        # so establish the actor task thread's device context in its constructor.
        # Keep ordinary reward/tokenizer factories on CPU afterwards.
        _require_runtime().set_device(_logical_device())
        frontend = _require_frontend()
        setter = getattr(frontend, "set_default_device", None)
        if not callable(setter):
            raise UnsupportedAdapterVersion("Jittor Torch namespace has no default-device owner")
        setter("cpu")
        result = original(self, *args, **kwargs)

        # NaiveRewardManager sends tokenizer.decode to run_in_executor().  The
        # new thread does not inherit ACL's current-device state from this
        # constructor, yet decode calls Tensor.tolist() on the response ids.
        # Bind the context to this worker's tokenizer instance so only the
        # downstream-owned decode entry point is affected.
        tokenizer = getattr(self, "input_tokenizer", None)
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            raise UnsupportedAdapterVersion("verl RewardLoopWorker has no input tokenizer decode owner")

        @wraps(decode)
        def decode_with_acl_context(*decode_args, **decode_kwargs):
            _trace("reward_decode_start")
            _require_runtime().set_device(_logical_device())
            result = decode(*decode_args, **decode_kwargs)
            _trace("reward_decode_end")
            return result

        decode_with_acl_context._jittor_verl_reward_decode_context = REWARD_DECODE_POLICY
        decode_with_acl_context._jittor_verl_original = decode
        tokenizer.decode = decode_with_acl_context
        return result

    initialize_reward_on_cpu._jittor_verl_reward_loop_cpu_context = REWARD_POLICY
    initialize_reward_on_cpu._jittor_verl_original = original
    return initialize_reward_on_cpu


def _make_parallel_put(original):
    @wraps(original)
    def parallel_put_with_acl_context(data_list, max_workers=None):
        assert len(data_list) > 0, "data_list must not be empty"
        concurrent_futures = original.__globals__["concurrent"].futures
        ray = original.__globals__["ray"]

        def put_data(index, data):
            # ACL current-device state is thread-local. NestedTensor pickling
            # materializes its Jittor leaves inside this executor thread, before
            # any worker method can establish a device on its behalf.
            _require_runtime().set_device(_logical_device())
            return index, ray.put(data)

        if max_workers is None:
            max_workers = min(len(data_list), 16)
        with concurrent_futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(put_data, index, data) for index, data in enumerate(data_list)]
            results = [future.result() for future in concurrent_futures.as_completed(futures)]
        output = [None for _ in range(len(data_list))]
        for index, data_ref in results:
            output[index] = data_ref
        return output

    parallel_put_with_acl_context._jittor_verl_parallel_put_context = RAY_PARALLEL_PUT_POLICY
    parallel_put_with_acl_context._jittor_verl_original = original
    return parallel_put_with_acl_context


def _make_dataproto_context_method(original, method_name):
    @wraps(original)
    def call_with_acl_context(self, *args, **kwargs):
        # Ray may pickle/unpickle DataProto in an async actor or executor
        # thread. TensorDict consolidation and torch.load both touch Jittor
        # tensors before downstream worker code can establish a device.
        # Async Ray actors may pickle multiple task arguments concurrently.
        # Jittor's host materialization and ACL runtime are not re-entrant
        # across those Python executor threads, so keep only the transport
        # conversion critical section process-serial.
        _trace("dataproto_%s_wait" % method_name)
        with _DATAPROTO_IO_LOCK:
            _trace("dataproto_%s_start" % method_name)
            _require_runtime().set_device(_logical_device())
            result = original(self, *args, **kwargs)
            _trace("dataproto_%s_end" % method_name)
            return result

    call_with_acl_context._jittor_verl_dataproto_context = (PROTOCOL_POLICY, method_name)
    call_with_acl_context._jittor_verl_original = original
    return call_with_acl_context


def _make_single_tensor_serializer(original):
    @wraps(original)
    def serialize_with_tuple_shape(obj):
        _trace("serialize_tensor_start dtype=%s shape=%s" % (obj.dtype, tuple(obj.shape)))
        # verl's NumPy transport normally reinterprets the tensor as uint8
        # before crossing the NumPy boundary.  For a lazy Jittor graph produced
        # by AgentLoopWorker's final torch.cat(), that dtype-view can SIGBUS on
        # aarch64/Ascend while the tensor's direct host materialization is
        # valid.  Materialize once using the public NumPy boundary, then do the
        # byte reinterpretation in NumPy.  This preserves verl's wire format
        # without adding a Jittor dtype-view node to the lazy graph.
        host = np.ascontiguousarray(obj.numpy())
        dtype = str(obj.dtype).removeprefix("torch.")
        shape = tuple(obj.shape)
        # Do not return a NumPy view backed by Jittor-owned host storage.  Ray
        # retains every encoded field until the whole TensorDict has been
        # pickled; retaining several such exports made a later lazy tensor
        # materialization crash in the AgentLoop actor.  ``bytes`` is accepted
        # by verl's bytearray-based decoder and owns an immutable copy.
        data = host.tobytes(order="C")
        # torch.Tensor.shape is torch.Size (a tuple subclass); Jittor's native
        # NanoVector carries the same dimensions but cannot cross a Ray pickle
        # boundary. Preserve the values using the public tuple contract.
        result = dtype, shape, data
        _trace("serialize_tensor_end dtype=%s shape=%s" % (dtype, result[1]))
        return result

    serialize_with_tuple_shape._jittor_verl_serialized_shape = PROTOCOL_SHAPE_POLICY
    serialize_with_tuple_shape._jittor_verl_host_bytes = PROTOCOL_HOST_BYTES_POLICY
    serialize_with_tuple_shape._jittor_verl_original = original
    return serialize_with_tuple_shape


def _make_batchdata_chunk(original, tensor_dict_type, nested_type):
    @wraps(original)
    def chunk_without_private_nested_consolidation(self, chunks):
        data = self._data
        if isinstance(data, tensor_dict_type):
            from verl.utils.tensordict_utils import chunk_tensordict, contiguous

            raw_chunks = chunk_tensordict(data, chunks)
            output = []
            for value in raw_chunks:
                prepared = contiguous(value)
                if any(isinstance(leaf, nested_type) for leaf in prepared.values()):
                    # TensorDict's nested consolidation imports PyTorch private
                    # nested internals. Jittor's public NestedTensor is already
                    # contiguous here; retain the valid split without applying
                    # a storage-layout optimization that cannot be represented.
                    output.append(prepared)
                else:
                    output.append(prepared.consolidate())
            return tuple(output)
        return original(self, chunks)

    chunk_without_private_nested_consolidation._jittor_verl_nested_chunk = PROTOCOL_NESTED_CHUNK_POLICY
    chunk_without_private_nested_consolidation._jittor_verl_original = original
    return chunk_without_private_nested_consolidation


@required_patch
def patch_worker(module):
    _require_runtime()
    if module.__name__ != MODULE:
        raise UnsupportedAdapterVersion("unexpected verl Worker module")
    worker = vars(module).get("Worker")
    if not isinstance(worker, type) or worker.__module__ != MODULE:
        raise UnsupportedAdapterVersion("verl Worker ownership changed")
    current = vars(worker).get("__init__")
    if (getattr(current, "_jittor_verl_worker_npu_context", None) == POLICY
            and getattr(current, "__module__", None) == __name__):
        _check_source(module, current._jittor_verl_original)
        return False
    if not isinstance(current, types.FunctionType) or current.__module__ != MODULE:
        raise UnsupportedAdapterVersion("verl Worker.__init__ ownership changed")
    _check_source(module, current)
    replacement = _make_initializer(current)
    replace(worker, "__init__", replacement, current)
    return True


@required_patch
def patch_agent_loop_worker(module):
    _require_frontend()
    if module.__name__ != AGENT_MODULE:
        raise UnsupportedAdapterVersion("unexpected verl AgentLoopWorker module")
    worker = vars(module).get("AgentLoopWorker")
    if not isinstance(worker, type) or worker.__module__ != AGENT_MODULE:
        raise UnsupportedAdapterVersion("verl AgentLoopWorker ownership changed")
    current = vars(worker).get("__init__")
    current_postprocess = vars(worker).get("_postprocess")
    init_patched = getattr(current, "_jittor_verl_agent_loop_cpu_context", None) == AGENT_POLICY
    postprocess_patched = (
        getattr(current_postprocess, "_jittor_verl_agent_loop_rm_scores", None)
        == AGENT_REWARD_SCORE_POLICY
    )
    original_init = current._jittor_verl_original if init_patched else current
    original_postprocess = (
        vars(module).get(AGENT_POSTPROCESS_ORIGINAL) if postprocess_patched else current_postprocess
    )
    if not isinstance(original_init, types.FunctionType) or original_init.__module__ != AGENT_MODULE:
        raise UnsupportedAdapterVersion("verl AgentLoopWorker.__init__ ownership changed")
    if (not isinstance(original_postprocess, types.FunctionType)
            or original_postprocess.__module__ != AGENT_MODULE):
        raise UnsupportedAdapterVersion("verl AgentLoopWorker._postprocess ownership changed")
    _check_agent_source(module, original_init, original_postprocess)
    changed = False
    if not init_patched:
        replace(worker, "__init__", _make_agent_initializer(current), current)
        changed = True
    if not postprocess_patched:
        from jittor.compat.module_patcher import patch_method

        alias = vars(module).get(AGENT_POSTPROCESS_ORIGINAL)
        if alias is not None and alias is not current_postprocess:
            raise UnsupportedAdapterVersion("verl AgentLoopWorker._postprocess alias changed")
        if alias is None:
            patch_method(module, AGENT_POSTPROCESS_ORIGINAL, current_postprocess)
        replace(
            worker,
            "_postprocess",
            _make_agent_postprocess(current_postprocess),
            current_postprocess,
        )
        changed = True
    return changed


@required_patch
def patch_tokenizer_thread(module):
    _require_runtime()
    if module.__name__ != TOKENIZER_MODULE:
        raise UnsupportedAdapterVersion("unexpected verl continuous-token module")
    mixin = vars(module).get("VLContinuousTokenMixin")
    if not isinstance(mixin, type) or mixin.__module__ != TOKENIZER_MODULE:
        raise UnsupportedAdapterVersion("verl VLContinuousTokenMixin ownership changed")
    current = vars(mixin).get("build_initial_tokens")
    if (getattr(current, "_jittor_verl_tokenizer_thread_context", None) == TOKENIZER_POLICY
            and getattr(current, "__module__", None) == __name__):
        _check_tokenizer_source(module, current._jittor_verl_original)
        return False
    if not isinstance(current, types.FunctionType) or current.__module__ != TOKENIZER_MODULE:
        raise UnsupportedAdapterVersion("verl VL build_initial_tokens ownership changed")
    _check_tokenizer_source(module, current)
    replacement = _make_tokenizer_thread_initializer(current)
    replace(mixin, "build_initial_tokens", replacement, current)
    return True


@required_patch
def patch_reward_loop_worker(module):
    _require_frontend()
    if module.__name__ != REWARD_MODULE:
        raise UnsupportedAdapterVersion("unexpected verl RewardLoopWorker module")
    worker = vars(module).get("RewardLoopWorker")
    if not isinstance(worker, type) or worker.__module__ != REWARD_MODULE:
        raise UnsupportedAdapterVersion("verl RewardLoopWorker ownership changed")
    current = vars(worker).get("__init__")
    if (getattr(current, "_jittor_verl_reward_loop_cpu_context", None) == REWARD_POLICY
            and getattr(current, "__module__", None) == __name__):
        _check_reward_source(module, current._jittor_verl_original)
        return False
    if not isinstance(current, types.FunctionType) or current.__module__ != REWARD_MODULE:
        raise UnsupportedAdapterVersion("verl RewardLoopWorker.__init__ ownership changed")
    _check_reward_source(module, current)
    replacement = _make_reward_initializer(current)
    replace(worker, "__init__", replacement, current)
    return True


@required_patch
def patch_ray_utils(module):
    _require_runtime()
    if module.__name__ != RAY_UTILS_MODULE:
        raise UnsupportedAdapterVersion("unexpected verl ray_utils module")
    current = vars(module).get("parallel_put")
    if getattr(current, "_jittor_verl_parallel_put_context", None) == RAY_PARALLEL_PUT_POLICY:
        _check_ray_utils_source(module, current._jittor_verl_original)
        return False
    if not isinstance(current, types.FunctionType) or current.__module__ != RAY_UTILS_MODULE:
        raise UnsupportedAdapterVersion("verl parallel_put ownership changed")
    _check_ray_utils_source(module, current)
    replace(module, "parallel_put", _make_parallel_put(current), current)
    return True


@required_patch
def patch_dataproto_serialization(module):
    _require_runtime()
    if module.__name__ != PROTOCOL_MODULE:
        raise UnsupportedAdapterVersion("unexpected verl DataProto module")
    data_proto = vars(module).get("DataProto")
    if not isinstance(data_proto, type) or data_proto.__module__ != PROTOCOL_MODULE:
        raise UnsupportedAdapterVersion("verl DataProto ownership changed")
    batch_data = vars(module).get("BatchData")
    if not isinstance(batch_data, type) or batch_data.__module__ != PROTOCOL_MODULE:
        raise UnsupportedAdapterVersion("verl BatchData ownership changed")

    chunk = vars(batch_data).get("chunk")
    chunk_patched = getattr(chunk, "_jittor_verl_nested_chunk", None) == PROTOCOL_NESTED_CHUNK_POLICY
    chunk_original = chunk._jittor_verl_original if chunk_patched else chunk
    if not isinstance(chunk_original, types.FunctionType) or chunk_original.__module__ != PROTOCOL_MODULE:
        raise UnsupportedAdapterVersion("verl BatchData.chunk ownership changed")

    serializer = vars(module).get("serialize_single_tensor")
    if getattr(serializer, "_jittor_verl_serialized_shape", None) == PROTOCOL_SHAPE_POLICY:
        serializer_original = serializer._jittor_verl_original
        serializer_prepared = None
    else:
        if not isinstance(serializer, types.FunctionType) or serializer.__module__ != PROTOCOL_MODULE:
            raise UnsupportedAdapterVersion("verl serialize_single_tensor ownership changed")
        serializer_original = serializer
        serializer_prepared = _make_single_tensor_serializer(serializer)

    originals = {}
    prepared = []
    for name in PROTOCOL_METHOD_SHA256:
        current = vars(data_proto).get(name)
        marker = getattr(current, "_jittor_verl_dataproto_context", None)
        if marker == (PROTOCOL_POLICY, name):
            originals[name] = current._jittor_verl_original
            continue
        if not isinstance(current, types.FunctionType) or current.__module__ != PROTOCOL_MODULE:
            raise UnsupportedAdapterVersion("verl DataProto ownership changed: " + name)
        originals[name] = current
        prepared.append((name, current, _make_dataproto_context_method(current, name)))

    _check_protocol_source(module, originals, serializer_original, chunk_original)
    if serializer_prepared is not None:
        replace(module, "serialize_single_tensor", serializer_prepared, serializer)
    for name, current, replacement in prepared:
        replace(data_proto, name, replacement, current)
    if not chunk_patched:
        from jittor.compat.torch import nested_tensor_type

        tensor_dict_type = vars(module).get("TensorDict")
        nested_type = nested_tensor_type()
        if not isinstance(tensor_dict_type, type) or not isinstance(nested_type, type):
            raise UnsupportedAdapterVersion("verl BatchData nested chunk types changed")
        replace(
            batch_data,
            "chunk",
            _make_batchdata_chunk(chunk, tensor_dict_type, nested_type),
            chunk,
        )
    return serializer_prepared is not None or bool(prepared) or not chunk_patched


def register(register_module_patch):
    from .native_npu import enabled

    if enabled():
        register_module_patch(MODULE, patch_worker)
        register_module_patch(AGENT_MODULE, patch_agent_loop_worker)
        register_module_patch(TOKENIZER_MODULE, patch_tokenizer_thread)
        register_module_patch(REWARD_MODULE, patch_reward_loop_worker)
        register_module_patch(RAY_UTILS_MODULE, patch_ray_utils)
