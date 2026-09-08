# 2026-08-11 源码架构重构第六批：convolution 类阶段

## 结论

完成源码现代化第六批：从 `python/jittor/nn.py` 迁出 `Conv`、`Conv1d`、
`Conv3d`、`ConvTranspose` 和 `ConvTranspose3d` 五个公开类实现，按 1D/2D、3D、
转置卷积拆成 3 个私有模块。`nn.py` 从 3,255 行降到 2,776 行，净减少 479 行
（14.7%）；相对第二批开始前的 5,220 行累计减少 2,444 行（46.8%）。

本批只迁移类定义，没有重写数值算法、shape 公式、参数初始化、groups 布局、
Depthwise/cuDNN 条件或转置卷积实现。`Conv1d_sp`、`DepthwiseConv`、pooling 和
Torch shim 后置创建的 `ConvTranspose1d` 保持原归属。

## 实现结构

```text
python/jittor/
  nn.py                                  # 稳定公开 facade 与历史别名
  _nn/
    convolution_layers.py                # Conv / Conv1d
    convolution_3d_layers.py             # Conv3d
    convolution_transpose_layers.py      # ConvTranspose / ConvTranspose3d
```

三个模块分别为 279、81、153 行；`_nn` 现在共 17 个文件、2,734 行，最大模块
285 行。`nn.py` 的结构预算从 3,300 行收紧到 2,800 行，私有实现模块继续不超过
350 行。

## 兼容边界

- facade 直接重导出原类对象，没有 wrapper 或兼容子类；类、方法和 pickle 仍显示
  为 `jittor.nn`。
- 非 ACL 下 `Conv2d is Conv`、`ConvTranspose2d is ConvTranspose`；
  `Conv1d_sp` 继续直接继承公开 `Linear`。
- `Conv1d` 经 `jt.nn.Conv` 动态构造内部 2D 卷积，ACL 后置类 wrapper 不会被
  私有静态引用绕过。
- `_pair/_triple`、`init`、`DepthwiseConv`、cuDNN helper 及公开 convolution
  functional 均经 `jt.nn` 动态解析，保留后端重绑和公开 monkeypatch 行为。
- `DepthwiseConv` 仍由 `jittor.depthwise_conv` 拥有，类身份与 pickle 路径不变；
  Torch shim 在 ACL 后扫描并缓存的 `nn.modules.Conv*` 身份保持不变。

五个类与未拆分提交 `b61f08cc` 完成 AST 对照。归一化计划内的
`Module -> jt.Module` 与 `jt.nn.*` 动态名称后 5/5 全等。

## 验证

| 验证 | 结果 |
| --- | --- |
| CPU/CUDA 结构契约 | 各 18/18 通过 |
| CPU/CUDA conv-pool 主矩阵 | 各 33/33 通过 |
| CPU OpInfo conv | 12/12 通过 |
| CPU serialization | 23/23 通过 |
| Conv2d 错误契约 | 3/3 通过 |
| cuDNN fp16/bf16/3D/transpose3D | 4/4 通过 |
| 五类与 Depthwise CPU/CUDA 前反向烟测 | 12/12 通过 |
| CUDA Torch shim 总入口 | 172/172 通过 |
| wheel 内容 | 1,023 个条目；17 个 `_nn` 文件；0 个禁止项 |
| wheel 隔离安装 | 51 项：50 通过、1 项源码条件跳过 |

wheel 位于：

```text
/home/zy/projects/jittor-lab/_state/verify/source-refactor/
  wheel-dist-convolution-classes-final/jittor-1.3.11.0-py3-none-any.whl
```

wheel 中 `nn.py`、三个新模块和结构测试与源码逐字节一致。隔离测试的
`jittor.__file__` 指向 wheel 安装目录；构建、JIT 缓存和隔离环境均位于仓库外。
仓库布局检查和 `git diff --check` 通过，源码树不保留 `.pyc` 或 `__pycache__`。

## 基线问题与未覆盖环境

`test_torch_compat_errors.test_unsupported_op_on_complex` 仍要求 complex
`exp/log/sin` 抛错，但当前实现已支持 `exp(complex)`。未拆分类的 `b61f08cc`
wheel 精确复现同一失败；本批没有触及 complex 实现或该测试，故只登记为既有测试
契约过期。本批涉及的三条 Conv2d 错误测试全部通过。

本机没有真实 Ascend/NPU 环境。ACL 的 `Conv/Conv2d` sibling wrapper、`Conv1d`
动态命中公开 `Conv`、后置 Torch shim 扫描顺序已经完成源码与结构审计，但不宣称
NPU 执行验证。

下一阶段优先迁移低耦合 padding 区域；pooling 应单独处理，`Linear` 与
`Conv1d_sp` 应作为同一继承边界迁移。后续均不应改变 `DepthwiseConv` 的公开归属
和 pickle 路径。
