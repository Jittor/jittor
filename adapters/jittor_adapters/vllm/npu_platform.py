"""Minimal vLLM platform bridge for Jittor's ACL-backed Torch shim.

This is deliberately opt-in. On CUDA hosts the entry point returns ``None``
and the normal vLLM platform selection is unchanged. ACL is exposed as the
shim's ``cuda`` device *API*, not as a claim that Ascend implements CUDA.
The first supported execution mode is eager, non-quantized text generation.
"""

import os
import sys


def platform_plugin():
    if os.environ.get("JITTOR_VLLM_NPU_PLATFORM") == "1":
        # Jittor's automatic module replay can capture a decode call while
        # vLLM is updating its external KV cache. On ACL this produced wrong
        # tokens after several steps unless every attention layer performed a
        # host readback. The eager ACL path needs neither capture nor fence.
        os.environ.setdefault("JT_AUTO_GRAPH_REPLAY", "0")
        jittor = sys.modules.get("jittor")
        if jittor is not None and os.environ["JT_AUTO_GRAPH_REPLAY"] == "0":
            jittor.flags.auto_graph_replay = 0
        return "jittor_adapters.vllm.npu_platform.JittorNpuPlatform"
    return None


from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum


class JittorNpuPlatform(Platform):
    # vLLM's GPU runner checks `is_cuda()`; the Torch shim maps its CUDA API
    # to ACL. Never use this enum to infer that CUDA kernels run on Ascend.
    _enum = PlatformEnum.CUDA
    device_name = "ascend-jittor"
    device_type = "cuda"
    dispatch_key = "CUDA"
    device_control_env_var = "ASCEND_RT_VISIBLE_DEVICES"
    dist_backend = "hccl"
    simple_compile_backend = "eager"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config,
                             num_heads=None):
        if attn_selector_config.use_mla:
            raise NotImplementedError("Jittor ACL vLLM does not support MLA")
        return "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"

    @classmethod
    def get_device_capability(cls, device_id=0):
        # This is only the compatibility level advertised to vLLM. It is not
        # an Ascend hardware capability or permission to launch CUDA kernels.
        return DeviceCapability(8, 0)

    @classmethod
    def get_device_name(cls, device_id=0):
        return "Ascend 910B (Jittor ACL)"

    @classmethod
    def get_device_uuid(cls, device_id=0):
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        ids = [part.strip() for part in visible.split(",") if part.strip()]
        physical = ids[device_id] if device_id < len(ids) else str(device_id)
        return f"jittor-acl-{physical}"

    @classmethod
    def get_device_total_memory(cls, device_id=0):
        import torch
        return int(torch.cuda.get_device_properties(device_id).total_memory)

    @classmethod
    def get_current_memory_usage(cls, device=None):
        import torch
        # The shim reads Jittor's per-device live allocation accounting.
        # Avoid CUDA-specific peak-reset or allocator operations on ACL.
        return float(torch.cuda.memory_allocated(device))

    @classmethod
    def device_count(cls):
        import torch
        return int(torch.cuda.device_count())

    @classmethod
    def num_compute_units(cls, device_id=0):
        # The Torch shim's CUDA properties contain a CUDA-oriented fallback
        # SM count, not an Ascend cube-core count. vLLM uses this for launch
        # sizing only; 24 is the conservative 910B fallback used by Ascend.
        return 24

    @classmethod
    def check_if_supports_dtype(cls, dtype):
        import torch
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"Jittor ACL vLLM does not support dtype {dtype}")

    @classmethod
    def set_device(cls, device):
        import torch
        torch.cuda.set_device(device)

    @classmethod
    def manual_seed_all(cls, seed):
        import torch
        torch.cuda.manual_seed_all(seed)

    @classmethod
    def get_device_communicator_cls(cls):
        return (
            "vllm.distributed.device_communicators."
            "base_device_communicator.DeviceCommunicatorBase"
        )

    @classmethod
    def check_and_update_config(cls, vllm_config):
        from vllm.config.compilation import CUDAGraphMode, CompilationMode

        parallel = vllm_config.parallel_config
        if parallel.worker_cls == "auto":
            parallel.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
        compile_config = vllm_config.compilation_config
        compile_config.mode = CompilationMode.NONE
        compile_config.cudagraph_mode = CUDAGraphMode.NONE
        # vLLM's async-output helper calls set_device from a fresh host
        # thread. ACL requires an initialized context in that thread and
        # currently raises aclrtGetDevice(107002), so keep first-stage
        # inference on the synchronous scheduler path.
        vllm_config.scheduler_config.async_scheduling = False
