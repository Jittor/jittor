# 2026-08-11 源码架构重构第八批：pooling 覆盖层

## 结论

完成源码现代化第八批：从 `python/jittor/nn.py` 迁出 Torch-compatible
`adaptive_avg_pool2d`、`avg_pool2d`、`AdaptiveAvgPool2d` 和 `AvgPool2d`，集中到
`_nn/pooling.py`；同时用 19 个显式重导出替代 `from jittor.pool import *`。
`nn.py` 从 2,581 行降到 2,442 行，facade 减少 139 行（5.4%）；相对第二批开始前
的 5,220 行累计减少 2,778 行（53.2%）。

本批没有重写 pooling 数值公式、shape 计算、梯度或设备实现。`jittor.pool` 中
语义不同的 legacy average pooling，以及 1D/3D、max pooling、unpool 和 ACL 开关
均保持原归属。该 777 行模块仍需在下一批按领域拆分，本批不把局部完成描述成整个
pooling 子系统已经现代化。

## 实现结构

```text
python/jittor/
  nn.py                    # 稳定公开 facade 与显式 pool 重导出
  pool.py                  # legacy pooling 实现，下一批拆分
  _nn/
    pooling.py             # corrected 2D average pooling 覆盖层
```

`pooling.py` 为 146 行；`_nn` 现在共 19 个 Python 文件、3,095 行，最大模块仍为
285 行。`nn.py` 的结构预算从 2,600 行收紧到 2,500 行，私有实现模块继续不超过
350 行。`nn.py + _nn` 从 5,530 行变为 5,537 行，因此 139 行是 facade 职责迁出，
不能误写成整个领域的总代码净减。

## 兼容边界

- facade 直接重导出四个迁移对象；函数、类、方法的 `jittor.nn` 反射来源和 pickle
  路径保持不变。
- `nn.functional.avg_pool2d/adaptive_avg_pool2d` 与公开函数保持同一对象，
  `nn.modules.AvgPool2d/AdaptiveAvgPool2d` 与公开类保持同一对象。
- 两个公开类的直接基类、构造签名、实例字段顺序和值、空 `state_dict`、类与实例
  pickle 均保持不变。
- 两个 functional 经 `jt.nn` 动态构造公开类，`AvgPool2d.execute` 经
  `jt.nn._pair` 动态解析，保留公开 monkeypatch 行为。
- corrected `nn.AvgPool2d/AdaptiveAvgPool2d/avg_pool2d` 继续与 `jittor.pool` 的
  legacy 同名对象分离；其余 19 个 pooling 名称保持显式重导出关系。
- legacy `nn.pool` 和 `nn.max_pool2d` 的模块全局仍指向 `jittor.pool.Pool` 与
  `jittor.pool.MaxPool2d`。ACL 仍只包装直接的 `nn.Pool`，没有暗改 NPU 路由。

四个定义与未拆分提交 `f96bd7b5` 完成 AST 对照。仅归一化计划内的
`Module -> jt.Module`、`_pair -> jt.nn._pair` 和两个公开类动态查找后 4/4 全等；
`nn.py` 其余 141 个顶层 AST 节点与基线顺序和内容完全一致。

## 验证

| 验证 | 结果 |
| --- | --- |
| CPU/CUDA 结构、conv-pool、pool parity 主矩阵 | 各 69/69 通过 |
| CPU pooling OpInfo | 18 通过、3 项 max-pool 二阶能力跳过 |
| legacy Pool/indices/unpool/3D CPU | 11/11 通过 |
| legacy Pool/indices/unpool/3D CUDA | 5/5 通过 |
| 部署 Torch shim pooling 虚拟基类 | 5/5 通过 |
| CUDA Torch shim 总入口 | 172/172 通过 |
| wheel 内容 | 1,025 个条目；19 个 `_nn` 文件；0 个禁止项 |
| wheel 隔离安装 | 69 项：68 通过、1 项源码条件跳过 |

wheel 位于：

```text
/home/zy/projects/jittor-lab/_state/verify/source-refactor/
  wheel-dist-pooling-final/jittor-1.3.11.0-py3-none-any.whl
```

wheel SHA-256 为
`03a31c0e3ca18341a36d23777a0c73826537baf96c4ddd5f1f95ce94fe2710f3`。
wheel 从暂存源码快照构建；`nn.py`、`pooling.py` 和结构测试与暂存内容逐字节一致，
用户工作区中 `python/jittor/src/var_holder.cc` 与
`python/jittor/test/test_setitem.py` 的两个未暂存 EOF 改动没有进入 wheel，也不纳入
本批提交。wheel 中没有 `.pyc`、`__pycache__`、`jittor/projects` 或
`jittor_fsdp2`，构建、JIT 缓存和隔离安装均位于仓库外。

## 未覆盖环境与后续

本机没有真实 Ascend/NPU 环境。ACL 的 post-process 开关、`nn.Pool` wrapper、
Torch functional/modules 快照和 legacy 路由已完成源码与结构审计，但不宣称 NPU
执行验证。

审计同时确认三个既有边界，本批没有顺带修改：`PoolACL` 的 ceil 输出 shape 仍使用
floor 公式；ACL 下 `MaxPool2d(return_indices=True)` fallback 可能只返回单 tensor；
corrected adaptive pooling 计算最大 bin 时有两次 `.data` 设备同步。这些需要分别做
后端正确性或性能专项，不能混入纯结构提交。

下一批优先把 `pool.py` 按 2D core、3D core、adaptive、wrapper/unpool 拆成私有实现，
并保留 `pool_use_code_op` 的动态后端开关；之后再处理 `Linear/Conv1d_sp`。
