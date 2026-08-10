# 2026-08-11 源码架构重构第二批：`nn.py`

> 后续进展：normalization functional 与 CUDA fast path 已在
> [第三批报告](2026-08-11-source-architecture-normalization-refactor.md)完成并验证。

## 结论

完成源码现代化第二批：`python/jittor/nn.py` 从 5,220 行降到 4,675 行，29 个低耦合
纯函数迁入 `jittor._nn` 私有实现包，按 activation、loss、softmax 和 vector 四个
职责模块组织。公开 `jittor.nn` 仍是唯一稳定 facade；类、别名、`Var` 方法绑定、
optimizer/pool 重导出和后端集成点没有移动。

本批净减少 facade 545 行（10.4%），私有包共 6 个文件、679 行，最大实现模块 270
行。没有新增依赖，也没有改变公开 import。源码现代化仍未结束，后续继续处理
normalization、RNN、convolution、`misc.py`、torch shim 和根启动流程。

## 实现结构

```text
python/jittor/
  nn.py                         # 稳定公开 facade、类、别名和后端组合
  _nn/
    __init__.py                 # 私有包说明
    runtime.py                  # composition root 绑定的最小运行时代理
    activations.py              # 12 个激活纯函数
    losses.py                   # 7 个损失纯函数
    softmax.py                  # softmax / log-softmax / logsumexp
    vector.py                   # GLU、normalize、cosine、distance、softsign
```

`nn.py` 先绑定运行时代理，再导入四个实现模块。私有模块在模块加载域不反向导入
`jittor`，避免给根包初始化增加环；函数的 `__module__` 仍恢复为 `jittor.nn`，因此
反射与 pickle 路径保持兼容。

## 兼容边界

迁移没有把 `Module` 子类一起搬走。原因是这些类同时参与参数注册、历史别名、
`jt.make_module()`、Torch functional 扫描和 ACL/CUDA 后端重绑，先保留在 facade
能把本批风险限制在纯函数实现。

结构契约锁定：

- `jt.nn`、`import jittor.nn` 和 facade 是同一模块对象。
- facade 与 29 个私有实现函数保持对象身份。
- moved callable 的 `__module__ == "jittor.nn"`，并可按原路径 pickle。
- `Var.prelu/hardswish/hardsigmoid/rrelu` 仍绑定公开函数。
- optimizer、pool、CTCLoss、DepthwiseConv、norm/conv/module aliases 保持历史身份。
- 私有模块不在模块加载域绝对导入根包，也不使用 `from ..`。
- `nn.py` 不超过 4,800 行，单个私有实现模块不超过 350 行。
- `setup.py` 必须显式声明 `jittor._nn`。

ACL 会合法替换 `relu`、`leaky_relu`、`softmax`、`Pool`、`Conv` 和 `ReLU` 等公开
对象。结构测试只在替换对象确实来自 ACL 编译器时允许身份不同，避免把任意重绑
误判为后端行为。

审查时发现 `log_softmax` 若直接解析私有模块内的 `softmax`，会绕过 ACL 对公开
`nn.softmax` 的运行时替换。最终实现通过 facade 动态分派，并用 monkeypatch 合约
测试锁定。29 个函数最初机械搬迁 AST 全等；清理原文件行尾空格后，最终 25 个保持
严格 AST 全等，`silu` 与 `smooth_l1_loss` 仅文档字符串尾部空白不同，另两个只为
保留上述公开动态分派而调整名称解析。去除文档字符串后，27 个计算定义 AST 全等。

## 新增测试

`test_nn_structure.py` 覆盖包身份、facade 重导出、反射/pickle、`Var` 绑定、关键
历史别名、导入方向、运行时绑定顺序、源码行数预算和 wheel package 声明。

`test_nn_functional_split.py` 补上原测试矩阵缺失的两个函数：

- `rrelu`：独立检查 eval 固定斜率、training 随机斜率区间和正半轴不变。
- `pairwise_distance`：以 NumPy 独立公式覆盖 `p=1/2/3/inf` 和 `keepdim`。

## 验证

| 验证 | 结果 |
| --- | --- |
| 最终 AST 对照 | 25 个严格全等，2 个仅文档空白；softmax/log-softmax 仅保留 facade 动态分派差异 |
| 新增源码测试（CPU） | 16 项：14 通过、2 项 CUDA 条件跳过 |
| 激活/损失兼容（CPU） | 58 项：57 通过、1 项 CUDA 条件跳过 |
| OpInfo NumPy/梯度（CPU） | 84 项：71 通过、13 项按 gradgrad 能力声明跳过 |
| CUDA 结构/兼容/新增测试 | RTX 4090 / JTCUDA 12.2：74/74 通过 |
| CUDA 设备一致性 | 28/28 前向与反向通过 |
| wheel 内容 | 1,012 个条目；6 个 `_nn` 文件齐全；0 个禁止项 |
| wheel 隔离安装 | 16 项：13 通过、3 项环境/源码条件跳过 |

wheel 检查拒绝 `.pyc`、`__pycache__`、`jittor/projects` 和 `jittor_fsdp2`，并确认
两份新增测试随包发布。隔离测试打印的 `jittor.__file__` 位于独立安装目录，不是
源码 checkout。

所有构建、JIT 缓存、wheel 和隔离安装均位于
`/home/zy/projects/jittor-lab/_state/verify/source-refactor`，主仓库未产生构建产物。

## 已知边界与后续

本机没有可用 Ascend/NPU 环境，本批对 ACL 做了源码路径审查与 wrapper 来源契约，
但不宣称真实 NPU 执行验证。下一批建议按以下顺序继续：

1. normalization：先分离普通实现，再单独保留/迁移 CUDA LayerNorm 快路径。
2. RNN：连同递推、参数布局和 cuDNN 路径整体迁移。
3. convolution：最后处理 backend、depthwise、transpose 和别名网络。
4. 随后拆 `misc.py` 和 `torch_shim/torch__init__.py`。
