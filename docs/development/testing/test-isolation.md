# 测试隔离、跳过与完成判据

[本栏目目录](../index.md) · [English](test-isolation.en.md)

前提：准备源码 checkout，并了解[测试体系](test-system.md)中的进程模式。本页说明如何隔离测试，以及如何判断跳过、崩溃和会话完成。

## 测试分类与 marker

Marker 在 [`pyproject.toml`](https://github.com/Jittor/jittor/blob/2.0-refactor/pyproject.toml)
中注册：

| Marker | 契约 |
| --- | --- |
| `structure` | 不执行设备代码；布局、打包与静态检查 |
| `cpu` | 维护中的 CPU 行为 |
| `cuda` | 需要 NVIDIA CUDA 环境 |
| `rocm` | 需要 AMD ROCm 环境 |
| `npu` | 需要昇腾 CANN 环境 |
| `mpi` | 需要 MPI 启动器或多进程 |
| `slow` | 排除在快速 PR 门禁之外 |
| `network` | 需要外部网络访问 |
| `manual` | 显式选择；**绝不进入自动的默认运行** |

**使用适用范围最窄的 marker。** 硬件测试要探测真实运算，**不得在 CPU 上静默通过**。
network 与 manual 测试要在模块 docstring 里说明其外部要求。

## 可选 fixture 与手动脚本

有些覆盖需要机器上恰好有的东西——一份数据集、一份 checkpoint、一个独立的 PyTorch。
这些入口**一律显式**：不设就按具体前置条件 skip，**绝不悄悄换成一个弱一点的检查**。

| 变量 | 谁读它 | 不设时 |
| --- | --- | --- |
| `JITTOR_TEST_IMAGENET_TRAIN` | ImageNet 布局的训练目录（每类一个子目录）。`tests/data/test_dataset.py` 用它验证多 worker `Dataset` 与 `num_workers=0` 的逐 batch 一致；`tests/data/test_image_folder.py` 再加上可选的 `torch`/`torchvision`，把 Jittor 的 `ImageFolder` 与 `torchvision` 的对齐 | 整个用例类 skip，reason 指名该变量 |
| `REAL_TORCH_SITE` | 进程内的独立 PyTorch oracle（见[进程模式隔离](test-system.md#进程模式隔离)） | 跳过可选的 PyTorch 对比；**已部署的 shim 不顶替它** |

原生 GPT 前向/梯度探针移至 `tools/probes/acl/`，需要手动配置真实设备。Qwen3 专属探针移至 [jittor-transformers 测试文档](https://github.com/JittorEcosystem/jittor-transformers/blob/main/docs/testing.md)，按其固定版本与显式模型参数运行。设备配置见 [昇腾 910B 指南](../../guides/hardware/ascend-910b.md)。迁移不构成新的硬件验收。

## 维护者探针

[`tools/probes/`](https://github.com/Jittor/jittor/blob/2.0-refactor/tools/README.md) 下是**扫描**，
不是测试：它们不断言一条性质然后变红，而是把两个本应相等的东西放在一起、把不一致全部
打出来，供人判断。输出是问题总账条目引用的证据，因此**不进门禁**，由维护者按需运行。

| 探针 | 拿什么和什么比 |
| --- | --- |
| `adversarial_device_sweep.py` | 整个 OpInfo 数据库在 NaN、±inf、±0、ties、次正规数上的 CPU 与 CUDA 结果 |
| `equivalent_form_sweep.py` | 同一个量的两种写法 |
| `fusion_consistency_sweep.py` | 同一个表达式开融合与 `no_fuse` 两次求值 |
| `roundtrip_consistency_sweep.py` | 保存-重载-重算的往返，以及 `no_grad` 与普通前向 |
| `semantic_divergence_probe.py` | 两种合理实现真会分歧的那些 dtype 与边界语义 |
| `side_effect_probe.py` | 只被要求"读"的操作改了什么（`a.numpy()` 迁走了 `a` 本身） |
| `opinfo_dtype_gaps.py` | 哪些 (算子, dtype) 跑得通却没声明（见[覆盖证据](test-evidence.md)） |
| `error_message_probe.py` | 用户真会犯的错当前收到的**原文**；两次运行的 diff 就是该次改动的验收记录 |

这些一致性扫描不依赖外部参考值，因此也能用于缺少独立参考的表达式。
它们不能替代 OpInfo 的独立前向参考或数值梯度检查。

## Skip 与已知失败

- **skip** 表示前置条件不可用，或某个契约有意不被支持；它的 reason 要指明**确切的
  前置条件或限制**。
- **预期失败** 表示一个已复现的框架缺陷。pytest 使用严格 xfail，因此修复会产生 XPASS
  并**强制清理总账**。
- **不要**在测试体外面捕获任意异常再转成 skip。要在执行前**窄范围地**探测可选环境。
- 每个持久的预期失败都要列进
  [问题总账](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/manuals/known-issues.md)，
  带 owner 和退出条件。

## 先检查执行量和会话完成，再解释失败

通过汇总只有在所选用例确实执行、进程完整结束时才有意义。分别检查：

| 观察 | 应如何解释 |
| --- | --- |
| 没有用例执行 | 说明缺少的前提，或在 `gate_scope.EXECUTES_NOTHING` 登记有理由的例外 |
| 设备数量不足 | 计为 `insufficient-devices`，与没有加速器分开 |
| 进程结束但没有完成哨兵 | 视为未完成；不能根据日志没有 `FAILED` 推断成功 |

`JITTOR_TEST_REQUIRE_EXECUTION=1` 让无法解释的未执行项使门禁失败。
`tests/_helpers/pytest_policy.py` 在会话结束时打印：

```text
========== files this session proved nothing about ==========
tests/.../test_x.py  collected 0 tests
tests/.../test_y.py  12 skipped, 0 executed
```

`collected 0 tests` 表示没有收集到用例，应检查收集错误和过滤规则；
`N skipped, 0 executed` 表示已收集但全部跳过。`expected here` 注记对应
`EXECUTES_NOTHING` 的显式例外；`explained` 表示已识别的环境原因。
先检查没有注记的条目，再读失败列表。

`tests/_helpers/session_completion.py` 从 `pytest_sessionfinish` 写入完成哨兵，
同时记录 collected/executed 数量；原生进程死亡无法写入该哨兵。
`tools/check_session_completed.py` 按哨兵 > 日志标记 > pytest 汇总行的优先级读取证据。
日志须保留哨兵和返回码。长运行用 `-v` 留下逐条结果，以便超时后仍能追踪进度；
只有安静输出不能证明完成。

新增门禁必须做负向验证：构造它要检测的具体缺陷，确认拒绝，再恢复。
扫描陈旧路径或空目录不能算覆盖。

## 子进程固定源码身份与设备

Python 子进程使用 `_helpers.child_process`。父进程 pytest 把 checkout 加进
`sys.path`，并不自动向子进程导出对应 `PYTHONPATH`；直接启动解释器可能加载另一份
可编辑安装，得到与当前代码无关的通过结果。AST 门禁
`tests/structure/test_child_process_contract.py` 检查测试启动点，以及
`tools/run_test_suite.py`、`tools/gate_conclusion_diff.py`。

安装来源测试若故意排除仓库路径，使用 `child_env(..., repo_paths=False)`。
在子进程中检查实际导入源码、进程模式变量和请求的设备；不能根据 flag 或机器有 GPU
推断张量驻留：

```python
where = x.location()
expected = "device" if device == "cuda" else "cpu"
assert where == expected, "child asked for %s, tensor is on %s" % (device, where)
```

这个例子检查 CPU/CUDA；其它后端须使用对应的真实设备断言和零回退检查。
子进程必须执行并同步被验证的计算。

## 隔离崩溃与有状态初始化

已知会以信号终止的复现通过 `run_child_script(..., crash_isolated=True)`
放进独立进程。否则原生崩溃会终止 pytest，后续用例全部丢失；设备 trap 还可能破坏
CUDA context。

可参考 `tests/backends/cuda/test_auto_flush_graph_split.py` 与
`tests/ops/test_index_bounds.py`。前者同时比较 loss 和梯度：不再崩溃不等于反向正确。
初始命名空间和激活状态的断言，若会受之前的延迟导入影响，也应放在干净进程中。

## 驻留对照先固定自动调优策略

对比 `auto_flush_ops` 等调度或驻留开关时，先固定两侧的 cuDNN 算法选择：

```python
import jittor as jt
jt.cudnn.set_benchmark(0)
```

自动调优会根据可用显存选算法；不固定它，就同时改变了被测开关和卷积算法。
前向与梯度都要比较，逐元素相对误差以对应元素作分母，不能用整份梯度的 RMS 替代。
详见[数值契约](../../notes/numerics/numerics-contract.md)；历史动机在
[问题总账](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/manuals/known-issues.md)的
KI-EXEC-001/003 与 KI-TEST-004 中。本协议不构成新的硬件验收。
