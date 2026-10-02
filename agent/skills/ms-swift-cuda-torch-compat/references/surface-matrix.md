# ms-swift surface manifest（从 checkout 生成）

这不是“所有测试都必须无条件运行”的承诺，而是防止漏测的索引。每次任务开始重新
运行清单命令并把版本、可执行性和结果写入报告。

```bash
MS_SWIFT=/path/to/ms-swift
git -C "$MS_SWIFT" rev-parse HEAD
find "$MS_SWIFT/swift" -mindepth 1 -maxdepth 2 -type d | sort
find "$MS_SWIFT/tests" -maxdepth 2 -type f -name 'test_*.py' | sort
find "$MS_SWIFT/examples" -maxdepth 3 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' \) | sort
find "$MS_SWIFT/requirements" -maxdepth 1 -type f -print | sort
```

当前 checkout 中已确认的功能组和入口如下；新增目录必须加入 manifest：

| 功能组 | 代码入口 | 代表测试/示例方向 |
| --- | --- | --- |
| 模型与模板 | `swift/model`, `swift/template`, `swift/arguments` | `tests/models`, `tests/general/test_template*`, custom model/template |
| 数据 | `swift/dataset`, `swift/dataloader` | `tests/general/test_dataset*`, streaming/packing/persistent workers |
| Trainer | `swift/trainers`, `swift/loss`, `swift/optimizers` | `tests/train/test_sft`, `test_cls`, `test_embedding`, `test_resume*`, optimizer/packing |
| Tuner | `swift/tuners`, `swift/tuner_plugin` | LoRA/QLoRA, freeze, new tokens, plugin tuner |
| RLHF | `swift/rlhf_trainers`, `swift/rewards`, `swift/rl_core`, `swift/rollout` | `tests/train/test_{dpo,kto,grpo,ppo,rlhf,gkd}`, rollout and reward tests |
| 多模态 | template/model processor 和 media utilities | `tests/models/test_mllm`, `tests/test_align/test_mm_processor_align`, image/video/audio/omni examples |
| Megatron/并行 | `swift/megatron`, `swift/sequence_parallel`, `swift/ray*` | `tests/megatron`, FSDP/DeepSpeed/Ray and multi-node examples |
| 推理 | `swift/infer_engine`, `swift/pipelines`, `swift/cli` | `tests/infer`, Transformers/vLLM/sglang/lmdeploy examples |
| 服务/应用 | `swift/ui`, deploy/app scripts | `tests/app`, `tests/deploy`, server/client examples |
| 评测/采样 | eval, sampler, metrics, callbacks | `tests/eval`, `tests/sample`, metrics/reward/teacher tests |
| 导出/量化 | export utilities and CLI | `tests/export`, merge/quant/Ollama/Hub examples |

## 选择用例

对每组登记三种条目：

1. **unit/API smoke**：不下载大模型，验证 import、注册、参数解析和小张量路径；
2. **paired numerical**：native 与 Jtorch 共享完整参数/缓冲区、输入、seed，比较
   forward、所有 trainable gradients 和适用的 input gradients；
3. **public path**：真实 `swift` CLI/API/launcher 或服务入口，带 checkpoint、worker
   日志和 fallback 观测。

没有可离线运行的模型、媒体、数据、服务或可选依赖时，保留条目并写出
`blocked/not-applicable` 的理由，不删除它。
