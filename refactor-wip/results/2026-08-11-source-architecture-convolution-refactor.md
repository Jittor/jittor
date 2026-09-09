# 2026-08-11 源码架构重构第五批：convolution 第一阶段

## 结论

完成源码现代化第五批第一阶段：从 `python/jittor/nn.py` 迁出 6 个 convolution
functional、2 个 cuDNN `Function` 类、3 个 backend helper 和一个 3D 半精度常量，
按普通卷积、转置卷积、cuDNN backend 拆成 3 个私有模块。`nn.py` 从 3,747 行降到
3,255 行，净减少 492 行（13.1%）；相对第二批开始前的 5,220 行累计减少 1,965 行
（37.6%）。

本批没有修改卷积数值算法、shape 公式、groups 布局、bias 广播、cuDNN 条件、反向
参数或半精度 fallback，也没有新增依赖。公开 `Conv*` 类、DepthwiseConv 和 pooling
留待第二阶段，避免把 ACL 类 wrapper 和 Torch shim 快照拆成不完整迁移。

## 实现结构

```text
python/jittor/
  nn.py                         # 稳定公开 facade、Conv 类和历史别名
  _nn/
    convolution.py              # conv1d / conv2d / conv3d
    convolution_cudnn.py        # 2D cuDNN Function/helper、3D 半精度保护
    convolution_transpose.py    # 1D/2D/3D transpose convolution
    runtime.py                  # 运行时代理与公开元数据恢复
```

三个新模块分别为 233、110、194 行；`_nn` 现在共 14 个文件、2,221 行，最大模块
285 行。`nn.py` 的结构预算从 3,800 行收紧到 3,300 行，私有实现模块继续不超过
350 行。

## 兼容边界

- 11 个迁移定义由 facade 直接重导出实现对象，不建立 wrapper 或兼容子类；函数、
  cuDNN 类及其 `execute/grad` 仍显示来自 `jittor.nn`，pickle 路径保持稳定。
- `conv is conv2d` 与 `conv_transpose2d is conv_transpose` 的历史别名语义保持不变；
  `nn.functional` 的 1D/2D/3D 与 transpose 映射继续引用公开 facade。
- ACL 仍在根包完成导入后替换 `nn.conv2d/Conv2d/Conv`。`conv1d` 经
  `jt.nn.conv2d` 动态命中 wrapper，没有静态捕获私有实现。
- `_pair/_triple`、2D/3D cuDNN helper、`conv2d` 和 `conv_transpose` 均经
  `jt.nn` 动态解析，保留后端重绑和公开 monkeypatch 行为。
- DepthwiseConv 仍由 facade 类动态调用公开 `nn.conv2d`；Torch shim 的类快照没有
  改动。公开 `Conv*` 类对象、参数名、state_dict 和实例 pickle 不在本批迁移。

11 个迁移定义与未拆分提交 `391893c5` 完成 AST 对照。还原计划内 `jt.nn.*` 动态
名称并归一化 docstring 行尾空格后 11/11 全等；`_CUDNN_3D_HALF_DTYPES` 的 AST 也
全等。独立只读审计未发现实现级问题，并补充锁定 `functional.conv_transpose2d`。

## 验证

| 验证 | 结果 |
| --- | --- |
| CPU/CUDA 结构契约 | 各 18/18 通过 |
| CPU/CUDA conv-pool 主矩阵 | 各 33/33 通过 |
| CPU OpInfo conv | 12/12 通过 |
| cuDNN fp16/bf16/3D/transpose3D | 4/4 通过 |
| CUDA device parity | conv2d、conv3d、conv_transpose2d 通过 |
| CUDA Torch shim 总入口 | 172/172 通过 |
| wheel 内容 | 1,020 个条目；14 个 `_nn` 文件；0 个禁止项 |
| wheel 隔离安装 | 51 项：50 通过、1 项源码条件跳过 |

wheel 中 `nn.py`、三个新模块和结构测试与构建快照逐字节一致。隔离测试的
`jittor.__file__` 指向 wheel 安装目录；构建、JIT 缓存、日志和隔离环境均位于
`/home/zy/projects/jittor-lab/_state/verify/source-refactor`。仓库布局检查和
`git diff --check` 通过，源码树没有 `.pyc` 或 `__pycache__`。

## 基线同样存在的旧测试问题

- `test_device_parity -k conv` 中 `conv1d` 的 CPU/CUDA 前向 net-scaled error 为
  `4.764730812149781e-4`，超过测试阈值 `2e-4`；未拆分 `391893c5` 得到完全相同
  数值。其余 `conv2d/conv3d/conv_transpose2d` 均通过。
- `test_conv_transpose.py` 的两个旧 PyTorch 对比用例在当前树和基线都因
  `m2.weight.grad is None` 报错。
- `test_reindex_op.test_conv_transpose_grad` 使用测试文件自己的 transpose helper，
  当前树和基线都在相同有限差分断言失败，不经过本批迁移的 `nn` 实现。

这些结果登记为既有测试/精度问题，不在纯结构提交中调整算子语义或测试阈值。

## ACL 与后续

本机没有真实 Ascend/NPU 环境。本批完成 ACL `conv2d/Conv2d/Conv` 后置 wrapper、
`conv1d` 动态分派和 functional 构建顺序的源码/结构审计，但不宣称 NPU 执行验证。

下一阶段迁移公开 convolution 类前，必须整体解决 DepthwiseConv 循环依赖、ACL 类
wrapper、Torch shim 类快照、模块实例 pickle/state_dict 和 pooling 邻接边界。
