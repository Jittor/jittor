# Qwen3-8B：Ascend 310P3 FP16 推理对比

- 状态：两档真实 NPU 稳态性能与跨框架对拍通过
- 日期：2026-10-13
- Jittor 基线：`5b0d2fd31f810ef892b05b29120a01ef0217dd3c`
- 基线组成：此前 310P 优化 `aad488adc`，整合远端 `3495d443d`；仅补充 SoC 查询的延迟导入以满足新的结构门禁
- 维护者：Jittor core maintainers
- 范围：单卡、FP16、batch=1、动态 KV cache、greedy、固定生成 32 token
- 复查条件：模型快照、框架/CANN/驱动版本、设备、精度或生成配置变化

## 结论

Qwen3-8B 在两侧均能完整运行。502-token 输入时 Jittor 解码比原生 Torch 慢
**1.8%**，完整生成慢约 **0.7%**，本轮接近持平；1998-token 输入时 Jittor
解码快 **20.1%**，完整生成耗时减少 **25.4%**。8B 短输入没有复现此前
0.6B 的速度领先，不能把小模型结论外推到所有模型。

| 实际输入 token | 指标 | Jittor | 原生 torch_npu |
| --- | --- | ---: | ---: |
| 502 | generate(1)，秒 | 0.37607 | 0.41166 |
| 502 | decode，token/s | 9.20 | 9.37 |
| 502 | generate(32)，秒 | 3.74550 | 3.72041 |
| 1998 | generate(1)，秒 | 1.44814 | 2.43054 |
| 1998 | decode，token/s | 8.81 | 7.34 |
| 1998 | generate(32)，秒 | 4.96680 | 6.65605 |

每档完整预热 5 次、计时 7 次，表内为中位数，每次样本前后同步设备。
decode 定义为 `31 / (median(generate(32)) - median(generate(1)))`。
`generate(1)` 是包含生成流程的首 token 延迟；工具另存的直接 `model(...)`
前向计算全部位置 logits，不与此混用。模型加载、权重转换和首次 JIT 不计入稳态。
这是一轮顺序测试；小幅差异不视为跨环境的稳定优势。

## 环境与基线选择

| 项目 | 配置 |
| --- | --- |
| NPU | 单张 Ascend 310P3，设备报告总内存 44213 MB |
| 驱动 / CANN | 26.1.0.b050 / 9.1.0 |
| Python / NumPy | 3.12.3 / 1.26.4 |
| Jittor | 2.0.0，独立 Torch frontend，源码提交见上 |
| 原生参考 | PyTorch 2.12.0+cpu / torch_npu 2.12.0，独立解释器 |
| Transformers | 两侧均 4.56.2 |
| 模型 | Qwen/Qwen3-8B，快照 `b968826d9c46dd6066d109eabc6255188de91218` |
| 结构 | 36 层，hidden=4096，intermediate=12288，H=32，KV=8，head_dim=128 |
| 原始参数量 | 8,190,735,360；两侧相同本地 checkpoint，以 FP16 加载 |
| 生成 | 禁用 thinking，greedy，强制 32 个新 token |
| FP32 RoPE 矩阵乘 | 两侧显式允许 HF32 |

Jittor 使用 SDPA 和 `--freeze-linear`，原生使用 eager、
`--torch-jit-compile off --torch-pack-linear`。两侧都预转换 253 个 Linear，
动态缓存、显式 `compile=none`；Jittor 保持默认 `auto_graph_replay=1`。
原生的 eager、SDPA 分别实测，eager 两档都更快，因此作为主表基线。
原生 SDPA 同口径结果如下：

| 输入 token | decode，token/s | generate(32)，秒 |
| --- | ---: | ---: |
| 502 | 8.38 | 4.22386 |
| 1998 | 4.77 | 10.16345 |

两侧独立进程顺序运行，计时期间没有本任务的其他 NPU 测试或 CPU 编译。
设备保留一个已有服务进程，测试前其设备利用率为 0；没有停止该进程。
本次不涉及训练、量化、并发服务或其他推理引擎。

## 正确性与启动成本

两档主提示词的所有预热、计时及验证 token 序列跨框架逐位一致；另有算术和
法国首都两个提示词，各自重复生成与动态缓存参考一致，并与原生参考逐位对拍。
末位置 logits 全部 finite、argmax 一致，相对 L2 均低于预设 1% 阈值。
Jittor 在 `backend_fallback=error` 和计数守卫下运行，最终计数为 **0**。
原生 Torch 没有对应回退计数器，JSON 记录 `fallback_count_available=false`。

| 输入 token | logits 相对 L2 | 最大绝对误差 | argmax |
| --- | ---: | ---: | --- |
| 502 | 0.004847 | 0.08203125 | 一致 |
| 1998 | 0.003125 | 0.06445312 | 一致 |

Jittor 权重打包耗时约 1.01/1.01 秒；
原生 eager 约 1.18/1.00 秒。Jittor 转换后的 Parameter 元素数为 622,638,080，
其余线性权重作为 buffer 保存，未裁剪或量化模型；原生仍用 Parameter 保存 NZ 权重。
8B 的 embedding 与 lm_head 本来不共享权重。

启动成本另计：短输入进程中，Jittor 模型加载/迁移/转换合计
33.90 秒，原生为 5.53 秒；Jittor 此前还在独立空缓存编译核心。
本轮没有做严格峰值显存采样，不能把稳态速度结论当作启动或内存优势。

## 复现

按 [310P 指南](../guides/ascend-310p.md) 准备两个独立解释器、相同 Transformers
与模型快照，加载 CANN 环境。下面的运行目录和解释器由环境提供；两个框架顺序运行。

```bash
perf_script=agent/skills/jittor-transformers-perf/scripts/benchmark_qwen3_ascend.py
JITTOR_TORCH_SHIM=1 PYTHONPATH=python backend_fallback=error \
  "$JITTOR_PYTHON" "$perf_script" --backend jittor --model "$QWEN_MODEL" \
  --dtype float16 --allow-hf32 --context-tokens 512 --new-tokens 32 \
  --fixed-length --warmups 5 --samples 7 --attn-implementation sdpa \
  --freeze-linear --validate-prompts \
  --logits-output "$RUN/jittor-sdpa-512-logits.npy" \
  --output "$RUN/jittor-sdpa-512.json"

"$TORCH_PYTHON" "$perf_script" --backend torch --model "$QWEN_MODEL" \
  --dtype float16 --allow-hf32 --context-tokens 512 --new-tokens 32 \
  --fixed-length --warmups 5 --samples 7 --attn-implementation eager \
  --torch-jit-compile off --torch-pack-linear --validate-prompts \
  --logits-output "$RUN/torch-eager-512-logits.npy" \
  --output "$RUN/torch-eager-512.json"

"$NUMPY_PYTHON" agent/skills/jittor-transformers-perf/scripts/compare_qwen3_ascend.py \
  "$RUN/jittor-sdpa-512.json" "$RUN/torch-eager-512.json" \
  --output "$RUN/parity-512.json"
```

把 `512` 换成 `2048` 测长输入；原生改 `sdpa` 并使用不同输出名即可复测另一配置。
工具不会强制 prompt 恰好为请求长度，本快照实际为 502/1998 token。

原始数据未版本化，位于 `$JITTOR_LAB_ROOT/_state/qwen3-8b-310p/`：
`bench/` 中的六份 benchmark JSON、两份 `parity-*` JSON 和 logits；
`logs/` 保存原始日志；`checkpoint_manifest.json` 记录同一快照的文件大小、缓存
blob 身份和配置/tokenizer 哈希。运行入口位于 lab 的 `qwen3-8b-310p/`。
原生测试时工作树尚在整合，所用 benchmark 脚本与上述基线一致；Jittor 测试时
工作树已提交且干净。

整合基线的 CPU core 为 322 passed / 42 skipped / 1 xfailed；完整结构门禁
1429 passed / 4 skipped，原生 dtype recorder 5 passed。首次结构失败来自
SoC 查询模块的导入循环，延迟导入后复验通过，未放宽门禁阈值。
