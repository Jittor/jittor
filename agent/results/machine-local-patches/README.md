# 重装前从本机抢救出来的补丁

这些改动只存在于一台即将重装的机器上，从未提交或推送到任何地方。
原样导出，**都没有经过本仓库的验证流程**——没有复现记录、没有牙齿检查、
没有逐 nodeid 归因。当作待验证的起点，不是结论。

## `vllm-jittor-shim.patch`

`/home/zy/projects/vllm`（detached at `v0.8.5`）里三个文件的未提交改动，
是让 vLLM 跑在 Jittor 的 torch shim 上的适配。改动自带解释，都指向同一类
问题——**shim 上的切片是脱离的副本，写进去不会回到原张量**：

- `attention/backends/flash_attn.py`：`kv_cache[0]`/`[1]` 在 jittor 上是
  detach 的切片副本，`reshape_and_cache_flash` 写进去不生效；而且
  `torch.ops._C_cache_ops` 在 shim 上是空壳。改成调
  `vllm_jittor_ops.ops.reshape_and_cache_kv` 直接写 `kv_cache`。
  另外 vLLM 往 `prefill_output`/`decode_output` 这两个切片里填结果，同样
  不回写。
- `model_executor/layers/vocab_parallel_embedding.py`、
  `model_executor/model_loader/loader.py`：另外两处。

这份补丁依赖 `vllm_jittor_ops` 这个包，需要单独确认它在哪。

## `jittor-2.0-local.patch`

`/home/zy/projects/jittor`（分支 `2.0`）里两个文件的未提交改动：
`agent/manuals/README.md` 一行、`tests/core/test_setitem.py` 一行。
看不出上下文，基线是 8 月的 `dccca5b2`，比现在落后一个月。存着以防有用。

## `*.ms-swift-*.patch`

见 `agent/results/ms-swift-local/`，同一批抢救，上游是
`modelscope/ms-swift`，所以只存补丁不推外部历史。
