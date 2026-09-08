# 2026-08-11 源码架构重构第七批：padding

## 结论

完成源码现代化第七批：从 `python/jittor/nn.py` 迁出 `pad` 以及
`ReflectionPad2d`、`ZeroPad2d`、`ConstantPad1d/2d/3d`、`ReplicationPad2d`
六个公开类，并按领域集中到 `_nn/padding.py`。`nn.py` 从 2,776 行降到 2,581 行，
facade 减少 195 行（7.0%）；相对第二批开始前的 5,220 行累计减少 2,639 行
（50.6%）。

本批只迁移现有定义，没有重写 padding 公式、shape 计算、错误条件或设备实现。
新增测试补齐此前缺少的 `ConstantPad1d/3d`、`pad=` 关键字别名和 bool fill 覆盖。

## 实现结构

```text
python/jittor/
  nn.py                    # 稳定公开 facade
  _nn/
    padding.py             # pad functional 与 6 个 Padding 类
```

`padding.py` 为 215 行；`_nn` 现在共 18 个 Python 文件、2,949 行，最大模块仍为
285 行。`nn.py` 的结构预算从 2,800 行收紧到 2,600 行，私有实现模块继续不超过
350 行。`nn.py + _nn` 的总代码量从 5,510 行变为 5,530 行，因此这里描述的是
facade 职责迁出，不把私有模块边界与兼容元数据开销误称为总代码净减。

## 兼容边界

- facade 直接重导出原对象，没有 wrapper 或兼容子类；函数、类、方法与 pickle
  仍显示为 `jittor.nn`。
- `nn.functional.pad is nn.pad`，`jittor.attention.pad is nn.pad`；公开签名继续为
  `(x, padding=None, mode='constant', value=0, pad=None)`。
- 六个类的直接基类、实例字段及字段顺序、空 `state_dict`、类与实例 pickle 均
  保持不变。
- ACL 当前不包装或重绑 padding；functional 和模块形式均继续通过 `Var.reindex`
  在当前 device 执行。Torch functional 在 ACL 初始化后创建，仍获取公开对象。
- `attention.py` 当前静态导入 `pad`。本批用身份契约锁定现状；未来若增加 padding
  backend wrapper，必须同步审计该静态引用和 Torch functional 创建顺序。

七个定义与未拆分提交 `0ec1dddc` 完成 AST 对照。仅归一化计划内的
`Module -> jt.Module` 后 7/7 全等。

## 验证

| 验证 | 结果 |
| --- | --- |
| CPU/CUDA 结构契约 | 各 19/19 通过 |
| CPU/CUDA 独立 NumPy padding 主矩阵 | 各 23/23 通过 |
| CPU OpInfo padding forward/grad/gradgrad | 12/12 通过 |
| CPU legacy attention padding 分支 | 3/3 通过 |
| CPU 旧 conv-pool padding 兼容 | 5/5 通过 |
| CUDA Torch shim 总入口 | 172/172 通过 |
| wheel 内容 | 1,024 个条目；18 个 `_nn` 文件；0 个禁止项 |
| wheel 隔离安装 | 42 项：41 通过、1 项源码条件跳过 |

legacy attention 定向烟测分别启用 `add_bias_kv`、`add_zero_attn` 和二者组合，同时
传入 bool attention/key-padding mask；三组输出和权重均有限、shape 正确且权重归一。

wheel 位于：

```text
/home/zy/projects/jittor-lab/_state/verify/source-refactor/
  wheel-dist-padding-final/jittor-1.3.11.0-py3-none-any.whl
```

wheel 从暂存树快照构建。`nn.py`、`padding.py` 和两份相关测试与暂存内容逐字节
一致；用户工作区中另两个未暂存 EOF 改动对应文件与 `HEAD` 一致，未进入 wheel。
wheel 中没有 `.pyc`、`__pycache__`、`jittor/projects` 或 `jittor_fsdp2`。隔离测试
的 `jittor.__file__` 指向 wheel 安装目录，构建、JIT 缓存和安装目录均在仓库外。

仓库布局、语法编译和 `git diff --check` 均通过。源码树测试产生的忽略缓存会在
提交前清理，不纳入版本控制。

## 基线问题与未覆盖环境

`test_kernel_traps.TestKernelTraps.test_constant_pad_fractional_fill_cpu_asmtuner` 保留
1 条 expected failure：asm tuner 生成的十六进制浮点常量存在既有编译问题。该测试
按预期通过，不是本批结构迁移引入的新失败。

本机没有真实 Ascend/NPU 环境。ACL 未重绑 padding，导入顺序、device 实现和
Torch shim 边界已完成源码审计，但不宣称 NPU 执行验证。

下一阶段优先单独处理 pooling；`Linear` 与 `Conv1d_sp` 应作为同一继承边界迁移。
`DepthwiseConv` 继续保持 `jittor.depthwise_conv` 的公开归属和 pickle 路径。
