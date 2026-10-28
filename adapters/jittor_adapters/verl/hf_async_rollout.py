"""Version-pinned Hugging Face async rollout for verl 0.9.1.

This backend is intentionally small and correctness-oriented.  It lets the
real async PPO trainer use the checkpoint-native Transformers generation model
when a compatible vLLM/SGLang build is unavailable.  Generation,
actor/ref/critic work and PPO updates remain owned by verl; this module only
supplies the two rollout plugin interfaces that 0.9.1 exposes in separate
registries.

The backend is opt-in and currently supports TP=PP=1.  Data-parallel trainers
get one independently updated rollout replica per actor rank.  Model weights
are copied to the server one tensor at a time so a complete model is never put
in Ray's object store as one giant object.
"""
from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np

from .._common import require_version, required_patch, replace, UnsupportedAdapterVersion


OPT_IN = "JITTOR_VERL_HF_ASYNC_ROLLOUT"
RAY_NPU_CONTEXT_OPT_IN = "JITTOR_VERL_RAY_NPU_CONTEXT"
NAME = "hf_async"
SUPPORTED_VERSIONS = frozenset(("0.9.1",))
BASE_SOURCE_SHA256 = "aaf4d73ea5afd5af62e607c4655c52ecabdd85ec3dfb303c771e02238b48bc1c"
REPLICA_SOURCE_SHA256 = "f54fa3326b80eaa83c0bc1cfadaa69e7a4717362eafd0f9915a3c1b99e0b5de0"


def enabled() -> bool:
    value = os.environ.get(OPT_IN)
    if value not in (None, "", "0", "1"):
        raise UnsupportedAdapterVersion(OPT_IN + " must be absent, 0 or 1")
    return value == "1"


def _check_source(module, expected: str) -> None:
    try:
        filename = inspect.getsourcefile(module)
        source = Path(filename).read_bytes() if filename else b""
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("verl rollout source is required") from error
    if hashlib.sha256(source).hexdigest() != expected:
        raise UnsupportedAdapterVersion("verl rollout source changed: " + module.__name__)


def _filtered_distribution(logits: np.ndarray, temperature: float, top_k: int, top_p: float) -> np.ndarray:
    """Return a finite CPU probability vector using common rollout semantics."""
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("HF async rollout received empty or non-finite logits")
    temperature = float(temperature)
    if temperature <= 0:
        result = np.zeros_like(values)
        result[int(values.argmax())] = 1.0
        return result
    values = values / temperature
    keep = np.ones(values.size, dtype=bool)
    top_k = int(top_k)
    if 0 < top_k < values.size:
        selected = np.argpartition(values, -top_k)[-top_k:]
        keep[:] = False
        keep[selected] = True
    masked = np.where(keep, values, -np.inf)
    masked -= np.max(masked)
    probs = np.exp(masked)
    probs /= probs.sum()
    top_p = float(top_p)
    if 0 < top_p < 1:
        order = np.argsort(-probs)
        cumulative = np.cumsum(probs[order])
        remove = cumulative > top_p
        if remove.any():
            remove[1:] = remove[:-1]
            remove[0] = False
            probs[order[remove]] = 0
            probs /= probs.sum()
    return probs


_CLASSES_READY = False
_CLASSES_BUILDING = False


def _ray_worker_location(worker):
    """Run inside the colocated Ray worker without capturing driver modules."""
    import ray
    from verl.utils.device import get_resource_name

    return (
        ray.get_runtime_context().get_node_id(),
        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
    )


def _rollout_actor_runtime_env(visible_device, backend="npu"):
    """Bind the accelerator before deserializing the rollout actor payload."""
    visible = str(visible_device)
    if not visible.isdecimal():
        raise ValueError("hf_async requires exactly one physical device per replica")
    if backend == "npu":
        env_vars = {RAY_NPU_CONTEXT_OPT_IN: "1", "JITTOR_VERL_RAY_CUDA_CONTEXT": "0",
                    "ASCEND_RT_VISIBLE_DEVICES": visible}
    elif backend == "cuda":
        env_vars = {"JITTOR_VERL_RAY_CUDA_CONTEXT": "1", RAY_NPU_CONTEXT_OPT_IN: "0",
                    "CUDA_VISIBLE_DEVICES": visible}
    else:
        raise ValueError("hf_async supports only cuda or npu")
    # Optional driver-owned output policy must reach this independent actor env.
    # This forwards metadata only; actual profile/ownership checks are caller-owned.
    for name in ("RAY_LOG_TO_STDERR", "VERL_SOURCE_EVIDENCE_ROOT"):
        if name in os.environ:
            value = os.environ[name]
            if type(value) is not str or not value.strip():
                raise ValueError("hf_async privacy environment requires nonempty strings")
            env_vars[name] = value
    return {"env_vars": env_vars}


def _server_init_payload(config, model_config, rollout_mode, replica_rank: int, visible_device: str, backend="npu"):
    """Return a stdlib-only actor payload that cannot import verl on unpickle.

    Ray deserializes actor arguments before invoking ``__init__``.  Passing
    verl dataclass/OmegaConf instances therefore imports ``verl`` before the
    actor can establish its ACL context; verl 0.9.1 transitively imports
    transfer_queue, which creates a module-level tensor.  Keep that boundary
    deliberately primitive instead of relying on runtime-env application
    order inside Ray.
    """
    return {
        "config": {
            "tensor_model_parallel_size": int(config.tensor_model_parallel_size),
            "pipeline_model_parallel_size": int(config.pipeline_model_parallel_size),
            "dtype": str(getattr(config, "dtype", "bfloat16")),
            "response_length": int(config.response_length),
        },
        "model": {
            "local_path": str(getattr(model_config, "local_path", "") or ""),
            "path": str(getattr(model_config, "path", "") or ""),
            "trust_remote_code": bool(getattr(model_config, "trust_remote_code", False)),
        },
        "rollout_mode": str(rollout_mode),
        "replica_rank": int(replica_rank),
        "visible_device": str(visible_device),
        "backend": str(backend),
    }


def _weight_target(parameters, buffers, name: str, tied_output_weight=None):
    """Resolve verl's actor-state name against a Transformers rollout model.

    Jittor's compatibility state stream preserves aliases, while PyTorch's
    ``named_parameters()`` removes duplicate objects by default.  Qwen3.5 ties
    ``lm_head.weight`` to the token embedding, so the former legitimately
    appears only in the actor stream.  Accept that one semantic alias without
    weakening rejection of genuinely unknown tensors.
    """
    target = parameters.get(name)
    if target is None:
        target = buffers.get(name)
    if target is None and name.startswith("model."):
        target = parameters.get(name[6:])
        if target is None:
            target = buffers.get(name[6:])
    if target is None and name == "lm_head.weight":
        target = tied_output_weight
    return target


def _rollout_dtype(torch_api, spelling):
    """Resolve the rollout loading dtype through the supplied runtime."""
    if not isinstance(spelling, str) or spelling not in ("bfloat16", "float16", "float32"):
        raise ValueError("hf_async dtype must be bfloat16, float16 or float32; got " + repr(spelling))
    return getattr(torch_api, spelling)


def _missing_weight_names(parameters, buffers, persistent_names, updated_names,
                          tied_output_weight=None):
    """Require the complete parameter/persistent-buffer stream, accounting for aliases.

    A tied output embedding and its input-embedding spelling are the same
    update target; nonpersistent rotary caches are not checkpoint weights.
    Object identity is used, never tensor equality or a host-value read.
    """
    required = dict(parameters)
    required.update({name: target for name, target in buffers.items() if name in persistent_names})
    updated_targets = set()
    for name in updated_names:
        target = _weight_target(parameters, buffers, name, tied_output_weight)
        if target is not None:
            updated_targets.add(id(target))
    return sorted(name for name, target in required.items() if id(target) not in updated_targets)


def _move_model_storage(model, target, torch_api, backend="npu") -> None:
    """Move a rollout model and return released blocks to the NPU allocator.

    verl colocates rollout and training workers and calls ``sleep_replicas``
    immediately after generation.  A backend that leaves its model resident at
    that boundary raises the critic/actor backward peak by one complete model
    per data-parallel rank.  Keep the movement in a small helper so the lifecycle
    can be tested without importing torch-npu in the host test process.
    """
    model.to(target)
    accelerator = getattr(torch_api, backend, None)
    if accelerator is not None:
        accelerator.synchronize()
        accelerator.empty_cache()


def _ensure_classes() -> None:
    global _CLASSES_READY, _CLASSES_BUILDING, HFAsyncServerAdapter, HFAsyncReplica, HFAsyncServer
    if _CLASSES_READY:
        return
    if _CLASSES_BUILDING:
        return
    _CLASSES_BUILDING = True

    import ray
    from torch.distributed.device_mesh import DeviceMesh
    from verl.workers.config import HFModelConfig, RolloutConfig
    from verl.workers.rollout.base import BaseRollout
    from verl.workers.rollout.replica import RolloutReplica

    class _HFAsyncServer:
        def __init__(
            self,
            payload,
        ):
            import asyncio

            config = payload["config"]
            model_config = payload["model"]
            visible_device = payload["visible_device"]
            backend = payload.get("backend", "npu")
            if int(config["tensor_model_parallel_size"]) != 1 or int(config["pipeline_model_parallel_size"]) != 1:
                raise NotImplementedError("hf_async requires tensor_model_parallel_size=1 and pipeline_model_parallel_size=1")
            if backend not in ("cuda", "npu"):
                raise ValueError("hf_async supports only cuda or npu")
            mask = "CUDA_VISIBLE_DEVICES" if backend == "cuda" else "ASCEND_RT_VISIBLE_DEVICES"
            if os.environ.get(mask) != str(visible_device):
                raise RuntimeError("HF async accelerator mask must be bound before the actor starts")
            import torch

            accelerator = getattr(torch, backend)
            if not accelerator.is_available():
                raise RuntimeError("hf_async requested unavailable " + backend)
            accelerator.set_device(0)
            from transformers import AutoModelForImageTextToText

            self.config = config
            self.model_config = model_config
            self.rollout_mode = payload["rollout_mode"]
            self.replica_rank = int(payload["replica_rank"])
            self.global_steps = None
            self._rejecting = False
            self._lock = asyncio.Lock()
            self._model_resident = True
            self._updated_names: set[str] = set()
            model_path = model_config["local_path"] or model_config["path"]
            if not model_path:
                raise RuntimeError("HF async rollout requires model_config.local_path or path")
            dtype = _rollout_dtype(torch, config["dtype"])
            self.backend = backend
            self.device = torch.device(backend)
            # Qwen3.5's top-level checkpoint is a VLM even when a batch is
            # text-only.  verl builds Qwen3_5ForConditionalGeneration and its
            # state stream therefore includes the visual tower; the causal-LM
            # auto class would silently build only Qwen3_5ForCausalLM and make
            # actor/rollout weight synchronization incomplete.
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=dtype,
                attn_implementation="eager",
                trust_remote_code=model_config["trust_remote_code"],
            ).to(self.device)
            self.model.eval()
            self._parameters = dict(self.model.named_parameters())
            self._buffers = dict(self.model.named_buffers())
            self._persistent_weight_names = set(self.model.state_dict())
            output_embeddings = self.model.get_output_embeddings()
            self._tied_output_weight = None
            if bool(getattr(self.model.config, "tie_word_embeddings", False)) and output_embeddings is not None:
                self._tied_output_weight = output_embeddings.weight
            eos = getattr(self.model.config, "eos_token_id", None)
            self._eos_ids = {int(value) for value in (eos if isinstance(eos, (list, tuple)) else [eos]) if value is not None}

        async def generate(
            self,
            request_id: str,
            prompt_ids: list[int],
            sampling_params: dict[str, Any],
            **kwargs,
        ):
            import torch
            from verl.workers.rollout.replica import TokenOutput

            del kwargs
            if self._rejecting:
                raise RuntimeError("HF async rollout is paused")
            async with self._lock:
                if not self._model_resident:
                    raise RuntimeError("HF async rollout generate called while weights are sleeping")
                ids = [int(value) for value in prompt_ids]
                generated: list[int] = []
                log_probs: list[float] = []
                seed_bytes = hashlib.sha256((str(request_id) + ":" + str(self.replica_rank)).encode()).digest()[:8]
                rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
                max_tokens = int(sampling_params.get("max_tokens", sampling_params.get("max_new_tokens", self.config["response_length"])))
                temperature = float(sampling_params.get("temperature", 1.0))
                top_k = int(sampling_params.get("top_k", -1))
                top_p = float(sampling_params.get("top_p", 1.0))
                for _ in range(max_tokens):
                    input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        output = self.model(input_ids=input_ids, use_cache=False)
                    logits = output.logits[0, -1].float().detach().cpu().numpy()
                    probs = _filtered_distribution(logits, temperature, top_k, top_p)
                    token = int(rng.choice(probs.size, p=probs))
                    generated.append(token)
                    log_probs.append(float(np.log(max(float(probs[token]), np.finfo(np.float64).tiny))))
                    ids.append(token)
                    if token in self._eos_ids:
                        break
                print(
                    "JITTOR_VERL_HF_ASYNC_GENERATE_OK "
                    f"request={request_id} prompt_tokens={len(prompt_ids)} response_tokens={len(generated)}",
                    flush=True,
                )
                return TokenOutput(
                    token_ids=generated,
                    log_probs=log_probs if sampling_params.get("logprobs", False) else None,
                    stop_reason="completed",
                    num_preempted=0,
                    extra_fields={"global_steps": self.global_steps},
                )

        async def begin_weight_update(self):
            self._updated_names.clear()

        async def update_weight(self, name: str, value: np.ndarray) -> bool:
            import torch

            if not self._model_resident:
                raise RuntimeError("HF async rollout weight update requires resumed weights")

            target = _weight_target(
                self._parameters,
                self._buffers,
                name,
                tied_output_weight=self._tied_output_weight,
            )
            if target is None:
                return False
            source = torch.from_numpy(np.asarray(value)).to(device=self.device, dtype=target.dtype)
            with torch.no_grad():
                target.copy_(source)
            self._updated_names.add(name)
            return True

        async def finish_weight_update(self, global_steps: int | None = None):
            if not self._updated_names:
                raise RuntimeError("HF async rollout received no actor weights")
            missing = _missing_weight_names(
                self._parameters, self._buffers, self._persistent_weight_names,
                self._updated_names, self._tied_output_weight,
            )
            if missing:
                raise RuntimeError("HF async rollout received incomplete actor weights: " + ", ".join(missing[:8]))
            # The ACK crosses a Ray process boundary. Drain this server's own
            # device copies before publishing the step; a driver-side wait
            # cannot complete work submitted in this process.
            import torch

            getattr(torch, self.backend).synchronize()
            self.global_steps = global_steps
            print(
                "JITTOR_VERL_HF_ASYNC_WEIGHT_SYNC_OK "
                f"global_steps={global_steps} tensors={len(self._updated_names)}",
                flush=True,
            )
            return len(self._updated_names)

        async def wake_up(self, tags=None):
            import torch

            requested = set(tags or ("weights", "kv_cache"))
            if "weights" not in requested:
                return
            async with self._lock:
                if self._model_resident:
                    return
                _move_model_storage(self.model, self.device, torch, self.backend)
                self._model_resident = True
                print("JITTOR_VERL_HF_ASYNC_WAKE_OK weights=" + self.backend, flush=True)

        async def sleep(self):
            import torch

            async with self._lock:
                if not self._model_resident:
                    return
                _move_model_storage(self.model, torch.device("cpu"), torch, self.backend)
                self._model_resident = False
                print("JITTOR_VERL_HF_ASYNC_SLEEP_OK weights=cpu", flush=True)

        async def clear_kv_cache(self):
            return None

        async def release_kv_cache(self):
            return None

        async def resume_kv_cache(self):
            return None

        async def start_profile(self, **kwargs):
            del kwargs

        async def stop_profile(self):
            return None

        async def abort_all_requests(self, reject_request: bool = False):
            self._rejecting = bool(reject_request)
            return {"aborted_count": 0, "request_ids": []}

        async def resume_generation(self):
            self._rejecting = False

    class _HFAsyncReplica(RolloutReplica):
        def __init__(self, *args, **kwargs):
            import ray

            super().__init__(*args, **kwargs)
            if self.world_size != 1:
                raise NotImplementedError("hf_async supports TP=DP-within-replica=PP=1; use trainer data parallelism for multi-card")
            self.server_class = ray.remote(globals()["HFAsyncServer"])

        @property
        def max_concurrency(self) -> int:
            return 1

        async def launch_servers(self):
            import ray
            from verl.utils.device import get_device_name

            if len(self.workers) != 1:
                raise RuntimeError(f"hf_async replica expected one worker, got {len(self.workers)}")
            node_id, visible_device = await self.workers[0].__ray_call__.remote(
                _ray_worker_location
            )
            name = f"hf_async_server_{self.replica_rank}_0"
            backend = get_device_name()
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                name=name,
                max_concurrency=self.max_concurrency,
                runtime_env=_rollout_actor_runtime_env(visible_device, backend),
            ).remote(payload=_server_init_payload(
                self.config,
                self.model_config,
                self.rollout_mode,
                self.replica_rank,
                str(visible_device),
                backend,
            ))
            self.servers = [server]
            self._server_handle = server
            self._server_address = name

    class _HFAsyncServerAdapter(BaseRollout):
        def __init__(self, config: RolloutConfig, model_config: HFModelConfig, device_mesh: DeviceMesh, replica_rank: int = -1):
            super().__init__(config, model_config, device_mesh)
            rank = int(os.environ["RANK"])
            rollout_world_size = int(self.config.tensor_model_parallel_size) * int(self.config.data_parallel_size) * int(self.config.pipeline_model_parallel_size)
            self.replica_rank = rank // rollout_world_size if replica_rank == -1 else int(replica_rank)
            self.rollout_rank = rank % rollout_world_size
            self.is_leader_rank = self.rollout_rank == 0
            self.server_handle = None

        def _server(self):
            import ray

            if not self.is_leader_rank:
                return None
            if self.server_handle is None:
                self.server_handle = ray.get_actor(f"hf_async_server_{self.replica_rank}_0")
            return self.server_handle

        async def resume(self, tags: list[str]):
            server = self._server()
            if server is not None:
                await server.wake_up.remote(tags=tags)

        async def release(self):
            server = self._server()
            if server is not None:
                await server.sleep.remote()

        async def update_weights(self, weights, global_steps: int = None, wire_format: str = "named_tensors", **kwargs):
            del kwargs
            if wire_format != "named_tensors":
                raise NotImplementedError("hf_async only supports named_tensors weight updates")
            server = self._server()
            if server is None:
                for _ in weights:
                    pass
                return
            await server.begin_weight_update.remote()
            unknown = []
            for name, weight in weights:
                cpu = weight.detach().to("cpu")
                if str(cpu.dtype) == "torch.bfloat16":
                    cpu = cpu.float()
                accepted = await server.update_weight.remote(str(name), cpu.numpy())
                if not accepted:
                    unknown.append(str(name))
            if unknown:
                raise RuntimeError("HF async rollout rejected actor weights: " + ", ".join(unknown[:8]))
            await server.finish_weight_update.remote(global_steps=global_steps)

    HFAsyncServer = _HFAsyncServer
    HFAsyncReplica = _HFAsyncReplica
    HFAsyncServerAdapter = _HFAsyncServerAdapter
    _CLASSES_READY = True
    _CLASSES_BUILDING = False


@required_patch
def patch_rollout_base(module):
    if not enabled():
        return False
    require_version("verl", SUPPORTED_VERSIONS)
    _check_source(module, BASE_SOURCE_SHA256)
    _ensure_classes()
    current = module._ROLLOUT_REGISTRY
    key = (NAME, "async")
    fqdn = __name__ + ".HFAsyncServerAdapter"
    if current.get(key) == fqdn:
        return False
    if key in current:
        raise UnsupportedAdapterVersion("verl already owns rollout registry key " + repr(key))
    updated = dict(current)
    updated[key] = fqdn
    replace(module, "_ROLLOUT_REGISTRY", updated, current)
    return True


@required_patch
def patch_rollout_replica(module):
    if not enabled():
        return False
    require_version("verl", SUPPORTED_VERSIONS)
    _check_source(module, REPLICA_SOURCE_SHA256)
    if not _CLASSES_BUILDING:
        _ensure_classes()
    registry = module.RolloutReplicaRegistry
    existing = registry._registry.get(NAME)
    if existing is not None:
        if getattr(existing, "_jittor_hf_async_loader", False):
            return False
        raise UnsupportedAdapterVersion("verl already owns rollout replica registry key " + NAME)

    def load_hf_async():
        return HFAsyncReplica

    load_hf_async._jittor_hf_async_loader = True
    registry.register(NAME, load_hf_async)
    return True


def register(register_module_patch):
    if enabled():
        register_module_patch("verl.workers.rollout.base", patch_rollout_base)
        register_module_patch("verl.workers.rollout.replica", patch_rollout_replica)
