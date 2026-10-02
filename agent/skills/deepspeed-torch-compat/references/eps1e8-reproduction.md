# Qwen3 eps1e-8 验证工具

维护者：赵佳祥。复查日期：2026-09-27。基线：97b9aab4455b145ca1fb431af3504d705ac1edf3 加已记录的适配改动。运行时、探针、依赖或证据结构变化后重新核验。

以下四个脚本是已验仓外探针的逐字节副本；保留其 SHA，让历史结果可以追溯。旧 eps1e-6 与 Stage0 工具保留历史用途，不能替代这些入口。

| 脚本（scripts/ 下） | 用途 | SHA256 |
| --- | --- | --- |
| qwen3_training_stage2_eps1e8.py | Stage1/2 三步训练，按 DS_ZERO_STAGE 选择 | d66a260ce98114446ac8636b15103ee22bb4f53643d14bcf233087f8994f9dff |
| qwen3_training_stage3_eps1e8.py | Stage3 三步训练，公开完整梯度/参数访问 | 1808c6ab6e84e6b0071b6c9ae2d0e7ac84a033f613a3d22deeb2f56fb3951a3a |
| qwen3_l3_roundtrip_eps1e8.py | Stage1/2/3 已训练模型权重保存、加载、任务及 tokenizer | 2960393d80038115eb7edffa34a83e005e28f926a566a1723e67a5454aae8ebd |
| qwen3_optimizer_resume_eps1e8.py | eps1e-8 优化器恢复与后续一步轨迹 | 1c39277e7e3ebb158cd858e915efb6ee5938063b0860cc3ac2b30535ceee6a8c |

这些探针仍读取环境变量配置；训练必须使用已分配的真实 NPU、两进程和独立 JIT 缓存。现有完整配对启动器及严格 L0-L4 汇总工具尚未统一沉淀；上表不代表已经具备一键从零复现实验的入口。

## 对已有 L3 产物进行独立核验

新入口 [qwen3_l3_source_compare.py](../scripts/qwen3_l3_source_compare.py) 参数化训练与保存产物路径，不依赖某台机器的个人目录。使用装有 NumPy、safetensors 的独立真 PyTorch 解释器，禁用 torch_npu 自动加载后在 CPU 读取文件：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$ORACLE" \
  "$REPO/agent/skills/deepspeed-torch-compat/scripts/qwen3_l3_source_compare.py" \
  --source "$TRAINING_DIR" --roundtrip "$ROUNDTRIP_DIR" \
  --zero-stage 2 --out "$RUN_ROOT/l3-recheck-stage2.json"
```

输出必须是新文件。Stage3 使用其自身的训练和保存目录，并传 --zero-stage 3。不能把 Stage1 的结果当作 Stage2 证据。

输入目录结构：

- 训练目录：comparison.json，以及 oracle/rank0、oracle/rank1、shim/rank0、shim/rank1 下的 report.json 与 step2/updated/*.npy。
- 保存目录：同样四个 rank 目录下的 report.json、roundtrip-diagnostic.json 和 saved-model/*.safetensors。

工具在读取大权重前检查四份报告的 stage、runtime、rank、world_size、NPU、FP32、HCCL、真 oracle 身份、版本一致、零 fallback、eps1e-8、固定真实 checkpoint、source report SHA、固定探针 SHA、任务误差和原诊断容差、tokenizer 输出。随后逐个读取每份 310 个保存参数，要求 shape/dtype 和所属训练最后一步权重完全相等，总计 1240 个张量；不会修改或放宽原容差。

这是对已有真实 NPU 产物的 CPU 重放，**不是重新运行两卡训练**。它不代替训练对拍，不测速度，不证明新硬件或新代码通过。Stage2/3 既有产物分别通过 1240 次保存张量校验；错误 source SHA、错误 stage、缺失 task 字段均被拒绝，且不生成成功报告。日志与原始 JSON 放在仓外实验目录，未版本化。

任务 round-trip 的数值误差由原固定 SHA 探针及其诊断记录提供；本工具复核它们及权重文件，不重新执行模型前向。NPU 真实性仍取决于原实验的设备断言、独立解释器与执行记录。不要将此 CPU 重放描述为新一次真机验收。

## 构造清单的独立比较

[构造比较器](../scripts/qwen3_construct_compare.py) 接受：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$ORACLE" \
  "$REPO/agent/skills/deepspeed-torch-compat/scripts/qwen3_construct_compare.py" \
  --root "$CONSTRUCT_DIR" --zero-stage 2 --out "$RUN_ROOT/construct-check.json"
```

它检查 oracle/shim 各两 rank 的 310 个参数、1 个 buffer、形状、FP32、NPU、固定模型配置与探针 SHA。requested_zero_stage 只是构造实验标签，**不代表 DeepSpeed Engine 按该 stage 初始化成功**；输出明确 engine_construction_verified=false。

旧清单未记录依赖版本及显式 oracle/fallback 字段。其固定 SHA 探针内存在身份、device、fallback 断言；比较器会分别报告 manifest_comparison=passed、status=partial、dependency_versions_not_recorded，并返回退出码 2。不能补造版本，也不能把这次记录审计自动当作原整条 L4 结论的升级或否定。错误 stage、缺 buffer、矛盾 oracle 字段返回非零且不生成成功记录。

新 [增强构造探针](../scripts/qwen3_construct_manifest_evidence.py) 保留原加载和构造算法，增加依赖版本、解释器来源、torch/torch_npu 版本、oracle 身份、同步后的 fallback 计数；shim 逐个同步全部 311 个参数与 buffer，并检查实际 location、placement_backend 和 device_id。oracle 在生成记录前执行 NPU 同步。比较器对增强 SHA 清单强制检查这些字段。增强入口需要获分配的 NPU 重新采集；新增脚本及语法检查本身不证明真机通过。torch 与 torch_npu 的实现版本只用于溯源，不要求与 shim 字面相同。

本次工具整理的旧 Stage1/2/3 清单只读 CPU 核验各通过构造字段比较，且如实返回上述版本缺口。增强采集的真机结果应以另行生成的日志与清单为准。
