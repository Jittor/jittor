# 2026-08-11 源码架构重构第九批：legacy pooling

## 结论

完成源码现代化第九批：将 `python/jittor/pool.py` 中 1D/2D/3D core、adaptive、
wrapper 和 unpool 实现拆入 `jittor._pool` 私有包，原模块保留为稳定公开 facade。
`pool.py` 从 777 行降到 38 行，facade 减少 739 行（95.1%）；22 个函数和类均由
私有模块唯一拥有，最大实现模块为 214 行。

本批是结构迁移，不修改 pooling 数值公式、shape 计算、梯度、indices 或设备实现。
私有包共 874 行，facade 与私有实现合计 912 行，比旧文件增加 135 行；新增部分来自
显式重导出、运行时绑定、职责边界和兼容元数据，不能把 facade 的减少误写成整个
领域的代码净减。

## 实现结构

```text
python/jittor/
  pool.py                   # 稳定公开 facade、后端开关与显式重导出
  _pool/
    runtime.py              # composition root 绑定与公开反射元数据恢复
    core_2d.py              # Pool
    core_3d.py              # Pool3d 与 _triple
    pooling_1d.py           # 1D adaptive/max/average pooling
    adaptive.py             # 2D/3D adaptive average/max pooling
    layers.py               # functional 和 2D/3D wrapper
    unpool.py               # MaxUnpool2d/3d
```

`_pool/__init__.py` 只标记内部包，不承担隐式重导出。各实现模块只依赖
`_pool.runtime`；运行时代理在 facade 导入私有实现前完成绑定，避免私有模块在初始化
阶段反向导入未完成的根包。

## 兼容边界

- `jittor.pool` 继续暴露原有 27 个非下划线名称，且不新增 `__all__`；历史 wildcard
  import 的可见面保持不变，`pool2d is pool` 也保持不变。
- 7 个函数、15 个类的签名、直接基类、默认实例字段、嵌套 wrapper 状态和空
  `state_dict` 均由结构测试锁定。
- 函数、类和方法继续显示来自 `jittor.pool`；类、函数、方法及实例 pickle 都经
  公开路径解析，没有暴露私有模块路径。
- 跨定义调用统一经 `jt.pool` 动态解析，保留公开 monkeypatch、wrapper 和后端重绑
  语义；`Pool/Pool3d.execute` 在每次执行时读取公开 `pool_use_code_op`。
- `nn.modules` 的 15 个 pooling 类与 `nn` 保持同一身份；`nn.functional` 的 5 个
  legacy 函数继续指向 `jittor.pool`，另 2 个 corrected 函数继续指向 `nn` 覆盖层。
  私有实现不依赖 `jt.nn`，因此不会绕过 ACL 只包装直接 `nn.Pool` 的既有边界。
- `setup.py` 显式声明 `jittor._pool`，发布包不依赖 namespace 自动发现。

22 个迁移定义与未拆分提交 `ca982448` 的 `pool.py` 完成严格 AST 对照。仅归一化
计划内的 `Module -> jt.Module` 与跨定义全局名 `name -> jt.pool.name` 后，22/22
全等；生成代码字符串和历史 docstring 也保持原内容。

## 验证

| 验证 | 结果 |
| --- | --- |
| 未拆分 CPU 主矩阵基线 | 88 项：83 通过、5 项按环境跳过 |
| 未拆分 CUDA 主矩阵基线 | 88 项：87 通过、1 项按条件跳过 |
| 拆分后 CPU 主矩阵 | 102 项：97 通过、5 项按环境跳过 |
| 拆分后 CUDA 主矩阵 | 102 项：101 通过、1 项按条件跳过 |
| 新增 facade 结构契约 | CPU/CUDA 各 11/11 通过 |
| 新增独立 NumPy legacy 数值回归 | CPU/CUDA 各 3/3 通过 |
| pooling OpInfo | CPU 18 通过、3 跳过；CUDA 25 通过、3 跳过 |
| 部署 Torch shim pooling 虚拟基类 | 5/5 通过 |
| CUDA Torch shim 总入口 | 172/172 通过 |
| wheel 内容 | 1,035 个条目；8 个 `_pool`、19 个 `_nn` 文件；0 个禁止项 |
| wheel 隔离安装 | 102 项：95 通过、7 项按环境或源码元数据条件跳过 |

CUDA 总入口使用冷缓存时包含较长的 fused-op 编译，但最终无失败。CPU 与 CUDA
OpInfo 中的 3 项均为已有 max-pool 二阶能力声明跳过，不是本批新增失败。

## 发布包

wheel 从源码与测试完成暂存时的 Git 索引树
`af6b1738d56c3786b55d57fa22d1e412d1373086` 在仓库外构建，位于：

```text
/home/zy/projects/jittor-lab/_state/verify/source-refactor/
  wheel-dist-legacy-pool-final/jittor-1.3.11.0-py3-none-any.whl
```

SHA-256 为
`4b9d20a40ba1c640623ada7cc24eb6cca4dd1fe64e87f40268b727e31aa97283`。
候选源码和 wheel 内的迁移文件均与索引逐字节一致；隔离导入确认 `jittor`、
`jittor.pool`、`jittor._pool` 和 JIT 源路径都来自安装目录。wheel 中没有
`jittor_fsdp2`、构建目录、缓存目录或 `.pyc`。用户工作区中
`python/jittor/src/var_holder.cc` 与 `python/jittor/test/test_setitem.py` 的两个
未暂存 EOF 改动取自 `HEAD`，没有进入本批 wheel 或提交。

## 既有风险与未覆盖环境

本机没有真实 Ascend/NPU 环境。ACL 对 `jittor.pool.pool_use_code_op` 的 post-process
开关、直接 `nn.Pool` wrapper 和 fallback 路由已完成源码、结构与动态分派审计，但
不宣称 NPU 执行验证。

审计记录了以下既有行为，本批没有在结构提交中顺带修复：

- ACL/reindex fallback 在带 padding 的全负 max window 上可能暴露填充值零。
- `ceil_mode` 的末尾空 window 可能产生浮点最小值 sentinel。
- `Pool3d(return_indices=True)` 生成代码的内层循环条件沿用了错误的循环变量。
- `Pool3d` 的相等比较与部分 padding 边界仍不完整。
- adaptive pooling 用于上采样尺寸时存在除零风险。
- 部分 1D/adaptive `return_indices` 契约未完整实现。
- overlapping unpool 与默认输出 shape 的历史语义仍需独立对齐。

这些问题需要分别建立最小复现、CPU/CUDA/NPU 对拍和语义修复提交；当前提交只保证
拆分前后行为一致。

## 后续

下一批优先从 3,066 行的 `misc.py` 抽取一个可独立回归的 shape/indexing 领域，继续
采用稳定 facade、私有实现、严格结构契约和隔离 wheel 验证。Torch shim 注册逻辑、
`Linear/Conv1d_sp/DepthwiseConv` 与根启动流程随后按独立领域处理。
