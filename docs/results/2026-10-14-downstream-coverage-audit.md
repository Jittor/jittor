# 下游注册表、门禁与 scorecard 覆盖审计

- 状态：源代码覆盖审计已执行；独立 oracle 数值重跑受环境阻塞。本报告不授予新的 L2–L5。
- 日期：2026-10-14（Asia/Shanghai）。
- Baseline commit：`cded3102037edc3663b8f2054a6ebc06da6ebfd2`，加本报告所在提交。
- 范围：全部逐库 runbook、共享 ecosystem CASES/测试节点、被引用报告，以及五个上游的自动分母。
- Owner：Torch 兼容层与下游生态维护者。
- Review when：上游 registry/版本、共享生态测试、runbook、scorecard 协议或验证环境变化。

## 源码入口事实

下表区分注册与执行。报告数仅统计 runbook 引用且当前仍存在的 `docs/results/` 文件；
缺席（包括明确退役）的报告提及也被机器审计记录；全量 results 中对库名的提及另列为弱关联，不能算专属验证。独立 adapter/专属单测不属于这里的共享生态门禁，0 不表示
“没有任何其他测试”。skill 提及的 L 级属于文本事实，本次没有把它当作通过证据。

| Repo / skill | 注册 / gate case 数 | 现存 / 缺席报告提及 | skill 中提及的 L 级（非已验） | 本次可授予级别 |
| --- | ---: | ---: | --- | --- |
| diffusers | 3 / 3 | 1 / 3 | 未显式分级 | 未建立 |
| flash-attention | 0 / 0 | 1 / 0 | 未显式分级 | 未建立 |
| mmcv | 1 / 1 | 0 / 2 | 未显式分级 | 未建立 |
| mmengine | 1 / 1 | 0 / 2 | 未显式分级 | 未建立 |
| ms-swift | 1 / 1 | 0 / 1 | 未显式分级 | 未建立 |
| peft | 1 / 1 | 0 / 1 | 未显式分级 | 未建立 |
| tensordict | 0 / 0 | 0 / 1 | 未显式分级 | 未建立 |
| torchmetrics | 0 / 0 | 0 / 1 | 未显式分级 | 未建立 |
| transformers | 11 / 11 | 1 / 0 | 未显式分级 | 未建立 |
| trellis | 0 / 0 | 0 / 1 | L2 | 未建立 |
| verl | 0 / 0 | 1 / 2 | 未显式分级 | 未建立 |
| vllm-omni | 0 / 0 | 3 / 0 | 未显式分级 | 未建立 |
| vllm | 0 / 0 | 1 / 2 | 未显式分级 | 未建立 |

Transformers 的 11 条含 6 个 tiny case 和 5 个 realistic speed case；Diffusers 的 3 条含
UNet/DiT 模块和 realistic UNet，不等于完整 pipeline 验收。ms-swift 只有
`ms_swift_lora_llama`，不能外推全量/分布式训练；verl 有 runbook 和历史报告入口，
但没有进入共享生态 gate 的 case。本次 core/broad/tail 新接受的计算证据数均为 **0**。

## 自动派生的完整分母

从各 upstream 的干净 Git checkout 枚举；没有导入下游库，没有因为缺依赖缩小分母。
以下 X=0 表示本次没有满足新 scorecard 协议的证据，**不否认历史上运行过的模型**。
CPU、CUDA、NPU、ROCm、Corex 分设备输出；本次各设备新接受分子均为 0。

| Upstream | 完整 commit / version | 自动广度轴与分母 |
| --- | --- | --- |
| transformers | `1ad1ad2f2691ce4f867590d2319bcd12fc2a02cc` / `5.19.0.dev0` | model_family **0 / 564**; task **0 / 53** |
| peft | `20f82b52cb690b249cb90cbab43af49d005ab510` / `0.21.3.dev0` | method **0 / 45**; task **0 / 6** |
| diffusers | `ea981b8b7d80f10deaff369bcda8cbf2c90428dd` / `0.41.0.dev0` | model_family **0 / 93**; pipeline_class **0 / 350**; auto_family **0 / 66**; task **0 / 8** |
| ms-swift | `4dc1e2add536088a3dee85e7568b4022330a2049` / `4.7.0.dev0` | model_family **0 / 237**; method **0 / 11** |
| verl | `e5a627cf55c94d85406556854c2c1a22db4ff5e7` / `0.10.0.dev` | advantage_estimator **0 / 14**; policy_loss **0 / 12** |

上游 remote 分别为 `huggingface/transformers`、`huggingface/peft`、`huggingface/diffusers`、
`modelscope/ms-swift`、`volcengine/verl` 的 GitHub SSH 仓库，实际 URL 保存在原始快照。
Diffusers 全部条件分支得到 93 个 pipeline 模块族与 350 个 pipeline 导出 API；AutoPipeline
66 家族只是子矩阵，不用它代替全库分母。verl 的 14/12 分别是 advantage-estimator/
policy-loss registry，不冒充完整训练算法或所有模型的支持清单。PEFT 的模型族需要另绑
Transformers registry reference，不能把 45 种方法当成 45 个模型。

## 实际重跑与环境边界

CPU ①显式设置 `CUDA_VISIBLE_DEVICES` 为空、`nvcc_path`/`JT_BUILD_NVCC_PATH` 为空、
`use_cuda=0`，并设置独立 JIT 状态、固定种子、离线和线程配置。选取
`transformers_gpt2` 运行独立 PyTorch oracle，启动即失败：
`ModuleNotFoundError: No module named typing_extensions`。原始输出与失败状态已保存，
命令返回非零。两个独立外置 venv 的准备尝试又遇到基础解释器缺少
`_sysconfigdata__linux_x86_64-linux-gnu`，因此没有修改共享环境或继续发布数值/速度比。

当前可发现的下游安装版本为 Transformers 4.56.2、Diffusers 0.35.1，PEFT、ms-swift、verl
缺席，与上面源代码分母版本也不同。不能用旧安装版本的数值给最新 upstream SHA 计覆盖。
CUDA③与 NPU③尚未重跑；ROCm/Corex 未验证；没有平台齐全的新 L4 或 L5 结论。

`verify_repo.py` 已收紧：非零退出（包括先打印成功 marker）、非有限/形状错误张量、
缺张量、设备/依赖版本/精度/线程配置不一致都失败；失败不生成可比 ratio，完整双方
元数据和日志保留。该单步诊断固定为 `scorecard_eligible=false`，不冒充多步训练、
输入扰动或整进程显存采样。

## 当前复验矩阵

| 仓库 / 具体场景 | CPU① | CUDA③ | NPU③ | 当前等级 / speed ratio |
| --- | --- | --- | --- | --- |
| Transformers / tiny GPT-2 | 已尝试，oracle 环境阻塞 | 未跑 | 未跑 | 未建立 / 不发布 |
| Diffusers / tiny UNet2D | 已尝试，同一 oracle 环境阻塞 | 未跑 | 未跑 | 未建立 / 不发布 |
| PEFT / tiny Llama LoRA | 已尝试，同一 oracle 环境阻塞；PEFT 也缺席 | 未跑 | 未跑 | 未建立 / 不发布 |
| ms-swift / tiny Llama LoRA | 已尝试，同一 oracle 环境阻塞；swift/PEFT 也缺席 | 未跑 | 未跑 | 未建立 / 不发布 |
| verl / 算法 registry | 没有共享生态 case，未接入该 gate | 未跑 | 未跑 | 未建立 / 不发布 |

四个已注册库各自命令都返回非零并保存 `verify-report.json`/双方阶段日志。未借用历史
报告补空格，也未以库缺席缩小 registry 分母。结构门禁实际尝试启动时，当前解释器又报
`No module named pytest.__main__`，因此不能声称结构门禁通过。主机侧 unittest、Ruff、
布局及新增活动文档的链接检查已通过。

## 协议与可运行验证

统一平台/速度块、JSON schema 和报告模板见
[下游 scorecard 协议](../development/downstream-scorecards.md)。18 个主机侧定向与负向
测试已通过：覆盖上游新增注册后的分母增长、脏源码/空/动态 registry、L1/skip/xfail 不计分、
过期 SHA/版本、shim oracle、错误设备、缺梯度/扰动、NaN、假 gate、产物哈希篡改、
缺 CPU/零回退、单步冒充 L3，以及 tiny/错 regime/缺 agreement/错内存口径冒充 L4。

```bash
python -m unittest discover -s tests/tools -p test_downstream_coverage.py -v
python -S agent/skills/torch-compat-repo-runbook/scripts/audit_coverage.py audit --json
```

上游分母与 scorecard 的完整复现命令见协议；五个 `--upstream <repo>=<checkout>` 均可
重复派生。空证据清单不会自动通过计算，只会输出可审计的 0/N。带非法 evidence 的
命令保留 rejected 原因并返回非零。

原始快照、scorecard JSON、审计 JSON、失败日志与缓存在
`$JITTOR_LAB_ROOT/_state/downstream-evidence/`，上游 checkout 在
`$JITTOR_LAB_ROOT/downstream-coverage/upstream-ssh/`，均**未版本化**。
本报告只保留可重建的维护者结论，不提交 raw registry 数据或环境路径。
