# vLLM 所需通用运行时能力的主仓整合验证

[English](2026-11-01-vllm-runtime-integration.en.md)

- 状态：本次运行时与安装入口的定向验收完成；广域 smoke 未全绿，剩余基线 AMP 和延迟导入限制见下文。
- 日期：2026-11-01 至 2026-11-02。
- 远端基线：`origin/2.0-refactor`，`88b48a1f18b2964f05090f65561e07ab64343715`。
- 整合代码：四项修复已重放至 `5bd26b7859c9589a6d641bbe78ee46ef987593fc`；没有覆盖远端后续实现。
- 范围：cuFFT/CUB 编译兼容、accelerator 显存查询、dtype-view 存储与写回语义、Triton constexpr 参数和缓存身份。
- 维护者：Jittor 核心与 Torch 兼容层维护者。
- 复查条件：相关核心/前端接口、CUDA 工具链、Triton、上游 vLLM 或已声明设备变化。

## 修复归属与基线

这些能力属于框架或通用 Torch 接口，应随主仓源码构建安装。vLLM 专属协议仍由独立
adapter 负责；此报告不把核心修复放进 adapter，也不将旧版本号相同的 wheel 视为已具备修复。

| 能力 | 主仓整合提交 | 行为 |
|---|---|---|
| cuFFT/CUB | `0467f3d991a9ae6b16fb3faf81041be4ab97212e` | 兼容 cuFFT 12 删除的枚举与 CUB 3 的 reduction 接口，保留旧版本分支 |
| 显存查询 | `379db21f22f6b69b9e5c730d0fcc9134180cd49d` | `torch.accelerator.get_memory_info` 复用已有 CUDA 查询 |
| dtype-view | `267888bded3f2f68ea38ba6c42e3d108b8503b70` | 合法非连续布局保持字节 stride、offset、共享存储与双向写回；dtype-view 不参与求导 |
| Triton 参数 | `5bd26b7859c9589a6d641bbe78ee46ef987593fc` | `None` 按 constexpr 特化；完整签名区分缓存身份，运行时参数打包不包含 constexpr |

复核另外将现代 Triton 的 native-specialization helper 检查与通用 ASTSource 缓存检查拆分：
旧 Triton 没有该 helper 时只跳过前者，不跳过缓存身份断言。

## 历史证据：2026-10-30，代码 `40d4414f4`

以下结果来自独立开发基线 `40d4414f45b89b7cb7f5efdc06ca622b674d772a`，
不是本次与远端整合后的测试结果。环境为 Python 3.11、RTX 4090、CUDA 13.0.88 工具链、
cuFFT 12/CUB 3、Triton 3.7.1。

| 范围 | 历史结果 |
|---|---|
| CPU core tier | 327 通过、45 环境/opt-in 跳过、1 预期失败 |
| dtype-view CPU 与邻接契约 | 27 通过；另有 9 个共享语义 case 通过独立 PyTorch 检查 |
| dtype-view CUDA | 11 通过 |
| accelerator 显存查询 CUDA | 3 通过 |
| cuFFT CUDA | 4 通过 |
| softmax CUDA | 5 通过，包含调度、梯度与独立数值参考 |
| Triton CUDA 后端文件 | 23 通过、3 个多设备用例因仅分配单卡跳过 |

历史原始日志和构建产物位于 `$JITTOR_LAB_ROOT/_state/vllm-product/20261030/`，未版本化。
历史 vLLM 固定版本为 0.29.0，提交 `98dff2a81d747d1dba01a47f939f48c3526d4206`；
这些框架回归不等于 vLLM 全库支持声明。

## 当前整合基线的验收

源码行为回归基线为 `9391a815dd6cc67b2d2fbe61fd9b4a1fad1d345a`。
随后 `ee116312ed3448a51c77ab1c9e2abeb45677f7f7` 只更新核心与前端发行身份为
`2.0.1`、配套依赖及版本契约，没有改变上述计算实现；聚合 adapter 载体版本仍为 `2.0.0`。
最终结构、shim 别名和安装后验证针对 `ee116312`；不把两个源码基线混写成同一次运行。

环境为 Linux x86_64、Python 3.11、单卡 RTX 4090、完整 CUDA 13.0.88 工具链、
cuFFT 12/CUB 3、Triton 3.7.1。CPU 与 CUDA 使用独立缓存，首次 JIT 串行执行。
源码门禁显式检查 core/compat 导入路径；最终版本门禁另建环境和缓存，确认核心 `2.0.1`。

| 验证范围 | 基线 | 当前结果 |
|---|---|---|
| CPU core tier | `9391a815` | 327 通过、45 跳过、1 预期失败 |
| CPU dtype-view / 显存查询契约 | `9391a815` | 13 通过、1 跳过 |
| CPU native views | `9391a815` | 56 通过、4 跳过 |
| CUDA cuFFT / softmax | `9391a815` | 9 通过 |
| CUDA dtype-view / 显存查询 | `9391a815` | 14 通过 |
| CUDA Triton backend | `9391a815` | 24 通过、3 个多设备用例跳过 |
| 完整 `tests/structure` | `ee116312` | 1577 通过、4 跳过，267.84 秒 |
| 门禁整理后的最终完整结构检查 | `0121c8a91` | 1577 通过、4 跳过，213.73 秒 |
| `test_torch_shim_aliases.py` | `ee116312` | 11 通过，15.49 秒 |
| CPU smoke 原生模式 | `9391a815` | 4353 通过、2561 跳过、332 未选择、5 预期失败、14 失败；定向清理见下文 |
| CPU smoke Torch 模式 | `9391a815` | 5207 通过、1882 跳过、1593 未选择、30 失败，2702.54 秒；不计为整套通过 |

首次结构检查有 6 项失败：CPU 环境错误继承 `JITTOR_CUDA_WHEEL_DISABLE=1`，
阻止了相关环境契约的预期行为。修正 CPU 配置为 `0` 后，定向复验 22 项通过；
最终 `ee116312` 的完整结构检查通过。未通过修改测试或隐藏失败获得结论。

原生 smoke 失败在不修改运行中源码树的前提下逐项定位：9 项缺少 SciPy，2 项 MKL
模拟值被规范环境变量覆盖，1 项预训练权重下载使用了失效的 Python CA 路径。
使用仓库指定的 NumPy 1.26.4 / SciPy 1.13.1 和系统 CA 后，31 项通过、22 项设备用例跳过，
包含实际 Res2Net 推理；移除冲突环境变量后，两项 MKL 用例通过。

另两项通过最小复现后修正测试门禁：显式发现 BLAS，并直接将两个乘法实现与 NumPy 对照
（矩阵 39 通过、17 跳过，相关结构检查 73 通过）；argmax 检查自身操作前后的存活计数差，
不再假定整个进程初始计数为零（原失败顺序的两项通过，完整 misc 文件 8 通过、6 跳过）。
前置 scalar-mask 操作的保留现象仍记录为
[KI-MEM-004](../../agent/manuals/known-issues.md#ki-mem-004-scalar-boolean-assignment-retains-one-live-operation)。
这些定向复验清理了原生模式失败，但不把首次 smoke 调用重新标记为全绿；
门禁整理及已知限制记录于 `0121c8a91d070dcee15d1cca7188bcc8267874ba`，没有更改核心/前端计算实现。

22 项 host/device 元数据测试原先误入 CPU 检查，且构造参数仍针对旧版 vLLM。
现已明确要求 CUDA、补充 0.29 的块容量参数，并用 NumPy dtype 比较 host proxy；
adapter 显式保留上游空请求的 ValueError。真实 CUDA 上 22 项全部通过，CPU 模式记录
22 项有解释的设备跳过。这些测试及空输入边界修正不改变上述核心/前端归档身份。

独立 bf16 训练用例在实际远端基线 `88b48a1` 和候选 `ee116312` 上完全同样失败：
最终 loss 为 0.158203125，float32 为 0.1223832667，相对差 0.2926859144，
超过未修改的 0.25 阈值。两个工作树/缓存均核对了真实源码身份。CPU autocast 会将
显式 float32 的 loss 运算重新降精度，而此前 LayerNorm dtype 修复仍生效；该既有问题
记录为 KI-COMPAT-025。六项延迟导入/顺序问题记录为 KI-COMPAT-026。
本次不声称解决这些限制或广域 smoke 全绿；验收范围是四项运行时前置、正常安装升级
以及声明的单设备推理配方。

源码复验入口为 `python tools/run_test_suite.py --tier core` / `--tier smoke`；
Torch 模式设置 `JITTOR_TORCH_SHIM=1`、`PYTHONPATH=python:tests` 后运行
`python -m pytest -q tests/structure`、`compat/tests/torch/test_torch_shim_aliases.py`、
`compat/tests/torch/test_dtype_view_strides.py`、`compat/tests/torch/test_accelerator_memory_info.py`
和 `compat/tests/triton/test_triton_backend.py`（最后一项在真实 CUDA 上执行）。
原生 CUDA 模式另行运行 `tests/backends/cuda/test_cufft_errors.py` 与
`tests/backends/cuda/test_softmax_cuda_grad.py`，不与 Torch 模式混用同一进程。

### Torch smoke 的命名空间与环境失败归因

整套 Torch smoke 不能计为通过。针对其中 7 项激活/别名/FSDP 结构失败，
在原 `9391a815` 上独立重跑 4 个完整文件，先得到 44 通过、1 失败。
剩余失败来自验证环境的 `000-frontend-stubs.pth` 自动加入 `shim/resources/stubs`，
违反环境变量独立激活测试的无资源路径前提；将该文件完整移至仓外状态目录备份后，
4 文件共 45 项通过（18.76 秒），见 `logs/shim-clean-env-files.log`。

在收集完成后仅导入 `torch.distributed.run` 与
`jittor.compat.fsdp2.activation_checkpoint`，即可复现另外 6 项失败
（6 失败、35 通过，`logs/shim-order-repro.log`）：前者增加未登记的 Torch 模块键，
严格所有权检查拒绝再次安装；后者是合法惰性子模块导入，增加 FSDP 包属性，
与初始包反射快照不同。相关测试、所有权检查及惰性导入实现与远端 `88b48a1` 相同，
没有被本次四项修复修改。此处保留严格所有权检查，未放宽断言或修改这些测试；
后续需单独处理测试隔离与运行时惰性模块所有权边界。独立文件通过不替代整套 smoke 通过。

## 正常安装与官方入口复验

在独立安装环境先安装旧 `2.0.0` fixture，再正常 pip 安装 adapter 候选及其依赖，
确认依赖解析自动升级到核心/前端 `2.0.1`，没有手动补丁或 `--no-deps` 绕过。
两个直接依赖使用官方仓库 `ee116312` 的不可变源码归档，前端指定
`#subdirectory=compat`；这不表示已向 PyPI 发布 `2.0.1`。
PEP 610 安装来源和 1348 个已安装文件的逐文件比对共同确认源码身份。
安装后导入来自 site-packages，不继承源码工作树路径。

vLLM `0.29.0`、adapter `2.0.0.dev3` 下，官方 Python API 复验 1 项通过、
官方 `vllm serve` HTTP 回归 7 项通过。Qwen3-0.6B 在单卡 CUDA、float32、eager、
TP1、TRITON_ATTN 配置运行；生成前后 596049920 个参数均归属 Jittor CUDA，
fallback 为 0。此结论是当前安装身份的实际执行证据。
历史 tiny 模型独立 oracle 与 token 对照仍保留原基线；此次没有重跑完整独立 oracle，
不能将历史数值误差写成当前源码的新测量。

当前原始日志、环境与产物位于
`$JITTOR_LAB_ROOT/_state/vllm-core-merge/20261101/`，未版本化。
`logs/` 保存源码及安装验收，`version-gate/` 保存最终源码身份、结构与别名日志，
`merged-provenance.json` 保存安装来源与文件比对，`merged-runtime-provenance.json`
保存实际导入与 CUDA 身份。集成由 [PR #18059](https://github.com/Jittor/jittor/pull/18059) 跟踪；不声称整棵测试树全绿。

## 未覆盖与剩余限制

不新增 ROCm、Corex、NPU 或多卡验收声明。Storage façade 指针、一般带 gap 的
`as_strided` 和历史部分 split Python 写回路径仍有独立限制，见
[KI-COMPAT-024](../../agent/manuals/known-issues.md#ki-compat-024-storage-façades-and-some-constructed-views-retain-alias-gaps)。
验证本次 dtype-view 的物理共享应使用 Tensor.data_ptr，并检查 stride、offset 与写穿；
正确读值或外部 kernel 写入不能替代一般 Python alias 语义验收。
