# MiniMax-H3 on Jittor, through the torch compat layer

Text-to-video-and-audio serving with **unmodified vLLM-Omni** and **unmodified
MiniMax-H3 weights**: Jittor's `compat` layer stands in for PyTorch, and
vLLM-Omni is imported from its own checkout as-is.

Everything below was run end to end on one machine (8x H100-96GB, CUDA 12.9,
driver 535, Python 3.12). Numbers are measured, not projected, unless a line
says otherwise.

## What is and is not patched

| piece | state |
| --- | --- |
| vLLM-Omni | upstream source, imported from its checkout |
| MiniMax-H3 weights and their `trust_remote_code` modules | as downloaded |
| ComfyUI and `ComfyUI-vLLM-Omni` | upstream |
| Jittor | needs the cross-thread compute-stream fix, below |

**Jittor must be new enough to contain the compute-stream handoff**
(`backend_compute_stream_acquire`/`_release`, `src/runtime/backend_streams.cc`).
Without it this stack decodes noise: the VAE decode is built on one Python
thread and fetched on another, and before the fix that produced all-NaN or
garbage output in roughly two thirds of requests. The failure is silent -- the
server returns 200, the job reports `error=None`, and the mp4 is noise. See
`docs/results/2026-09-14-vllm-omni-h3-enablement.md` section 49.

## Environment

These are configuration, not source changes. Each one is here because of a
specific failure.

```bash
export use_cuda=1
export CUDA_VISIBLE_DEVICES=4,5,6,7          # the GPUs to serve on
export PYTHONPATH=/path/to/vllm-omni${PYTHONPATH:+:$PYTHONPATH}

# Import the vLLM-Omni checkout instead of building its CUDA extension.
export JITTOR_TORCH_SKIP_EXT_BUILD=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# flash-attention through the jittor bridge
export JITTOR_FLASH_ATTN_JITTOR_SRC=/path/to/flash-attention
export JITTOR_FLASH_ATTN_JITTOR_REQUIRED=1
export JITTOR_FLASH_ATTN_HEAD_DIMS=64,128
export JITTOR_FLASH_ATTN_DTYPES=bf16,fp16
export JITTOR_FLASH_ATTN_CAST_FLOAT32=bf16
```

Multi-GPU needs three more, none of which the launcher arranges:

```bash
export JITTOR_TORCH_DISTRIBUTED_AUTO_INIT=1   # the shim's dynamic NCCL bootstrap
export JT_BUILD_NCCL_INCLUDE_PATH=/usr/include
export JT_BUILD_NCCL_LIB_PATH=/lib64          # setup_nccl only finds a system NCCL when told
export JITTOR_TORCH_KEEP_TMPDIR=1             # see below
rm -f /tmp/jittor-nccl-*.bin /tmp/jittor-nccl-*.bin.*   # before every run, see below
```

* **`JITTOR_TORCH_KEEP_TMPDIR`** -- the shim overrides `TMPDIR` with
  `<runtime>/tmp`. If that path is long, appending vLLM-Omni's `ipc://` socket
  name exceeds `sockaddr_un.sun_path`'s 107 bytes and the orchestrator dies at
  startup with no useful message.
* **The rendezvous files** are named after `MASTER_ADDR-MASTER_PORT` alone, so a
  rerun on the same port inherits the previous run's unique ids and hangs inside
  distributed init rather than failing.

## Serving

```bash
vllm-omni serve /path/to/MiniMax-H3 \
  --omni --host 0.0.0.0 --port 18091 \
  --trust-remote-code --task-type fl2va \
  --num-gpus 4 --tensor-parallel-size 4 \
  --usp 1 --ring 1 --cfg-parallel-size 1 \
  --text-encoder-tp-size 1 \
  --vae-patch-parallel-size 1 --vae-parallel-mode tile --vae-use-tiling \
  --max-model-len 8192 \
  --diffusion-attention-backend FLASH_ATTN \
  --enforce-eager \
  --diffusion-offload-config '{"mode":"layer","components":["text_encoder"]}'
```

Single GPU is the same with `--num-gpus 1 --tensor-parallel-size 1` and
`CUDA_VISIBLE_DEVICES` naming one device. Ready in 260-330 s from cold.

**The text-encoder offload is not optional on one GPU.** Without it a TP1 run
OOMs: 51.5 GB of encoder plus the DiT plus a float32 VAE against 95 GB.

## Calling it

The response carries metadata and a `file_name`; the bytes come from a separate
endpoint. A client that only knows the inline-base64 shape writes nothing and
looks like a failure while the job is in fact completing.

```python
import json, requests, time

form = {
    "model": "/path/to/MiniMax-H3",
    "prompt": open("prompt.txt").read(),
    "width": "512", "height": "512", "fps": "24", "seconds": "5",
    "num_inference_steps": "50", "seed": "0",
    "extra_params": json.dumps({
        "task": "t2va", "duration": 5.0, "aspect_ratio": "1:1",
        "flow_shift": 12.0, "audio_flow_shift": 3.0,
    }),
}
base = "http://127.0.0.1:18091/v1"
job = requests.post(base + "/videos", data=form, timeout=180).json()["id"]

while True:
    state = requests.get("%s/videos/%s" % (base, job), timeout=30).json()
    if state["status"] in ("completed", "failed"):
        break
    time.sleep(15)
assert state["status"] == "completed", state.get("error")

mp4 = requests.get("%s/videos/%s/content" % (base, job), timeout=300).content
open("out.mp4", "wb").write(mp4)
```

## Measured

Same request, same prompt, same seed, one machine, GPUs shared with an
unrelated tenant on both sides so the comparison is like for like.

| request | TP1 | TP4 |
| --- | --- | --- |
| 512x512, 5 s, 50 steps | 255.6 s | 156 s |
| 1024x768, 10 s, 50 steps | 67.65 s/step | 18.7 s/step |

TP4 is 3.6x on the diffusion steps and 1.6x end to end on the smaller request:
the VAE decode and the text encoder do not scale with TP, so the bigger the
request the more TP4 is worth. TP2 is *slower* than TP1 on a 2-step 256x256
request (55.1 s against 41.9 s) -- the per-layer NCCL traffic outweighs the
per-step compute until the request is large enough.

Against real PyTorch on the same script and request (512x512, 124 frames, 6
steps, one GPU each, warm):

| phase | jittor | torch | ratio |
| --- | --- | --- | --- |
| DiT | 25.92 | 23.35 | 1.11x |
| text encoder | 3.77 | 10.18 | **0.37x** |
| video VAE | 15.69 | 6.79 | 2.31x |
| audio VAE | 5.64 | 0.28 | 20.1x |
| **end to end** | **92.34** | **75.94** | **1.22x** |

The shim is at parity on the DiT, faster on the text encoder, and behind on the
VAEs. The first run of any shape pays JIT compilation (~2.7x) and is not a
speed number.

## Gotchas

* **The clip is not proof the run was executed.** ComfyUI caches node outputs by
  graph hash, so resubmitting an identical workflow returns success and writes
  nothing. Change the seed to force a real run.
* **A successful status is not a correct picture.** Before the Jittor fix,
  corrupted runs reported `status=completed, error=None`. Score the pixels: an
  adjacent-pixel mean delta around 10x the healthy value, with the standard
  deviation halved, is the corruption signature.
* **Killing the server is not enough to free the GPUs.** The worker ranks are
  `multiprocessing.spawn` children whose command lines do not match
  `vllm-omni serve`; they outlive a `pkill` on that pattern and keep tens of GB
  allocated. Wait for the memory to actually drop before starting the next run.
* **The port going quiet is not the port being free.** The server binds
  `0.0.0.0` with `reuse_port=False`, and the lingering socket sits in
  `/proc/net/tcp6`, not `/proc/net/tcp`.
* **If flash-attention raises `TORCH_CHECK(x.is_cuda())`**, the top-level
  `flash_attn_interface` import resolved to the unbridged PyTorch extension.
  Put a module of that name on `PYTHONPATH` re-exporting from
  `flash_attn.flash_attn_interface`. This was needed historically; a run on
  2026-09-22 resolved `FLASH_ATTN` and produced correct output without it.
