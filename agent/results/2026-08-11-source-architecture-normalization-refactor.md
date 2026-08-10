# 2026-08-11 源码架构重构第三批：normalization

## 结论

完成源码现代化第三批：从 `python/jittor/nn.py` 迁出 6 个 normalization 函数/
helper，其中 5 个进入 `_nn/normalization.py`，完整 274 行 CUDA LayerNorm no-grad
kernel 进入 `_nn/layer_norm_cuda.py`。`nn.py` 从 4,675 行降到 4,306 行，净减少
369 行（7.9%）；相对第二批开始前的 5,220 行已累计减少 914 行（17.5%）。

`BatchNorm`、`InstanceNorm`、`LayerNorm`、`GroupNorm` 四个类及其别名仍保留在公开
facade。本批没有改 normalization 算法、类身份、参数/缓冲区注册、MPI 状态或设备
选择，也没有新增依赖。

## 实现结构

```text
python/jittor/
  nn.py                         # 公开类、别名、AMP decorator、组合入口
  _nn/
    normalization.py            # functional API 与稳定反向 helper
    layer_norm_cuda.py          # CUDA no-grad inference kernel
    runtime.py                  # 运行时代理与公开元数据恢复
```

`_nn` 现在共 8 个文件、1,085 行，最大模块 285 行；`nn.py` 的结构预算同步收紧为
4,400 行，单个私有实现模块仍不超过 350 行。

## 为什么类没有一起搬

ACL 在加载时捕获并原位替换 `jt.nn.LayerNorm.execute`。Torch shim 还缓存
`BatchNorm`、`_BatchNorm`、SyncBatchNorm 和 normalization 子模块的类对象；MPI
BatchNorm 依赖同一类的状态更新。若把类搬入私有模块，方法内的全局名称也会从
`jittor.nn` 变成私有模块，现有对 `nn._ln_normalize` 的 monkeypatch 不再生效。

因此本批只迁函数和 kernel。公开类继续通过 facade 全局名动态解析
`_ln_normalize` 与 `_layer_norm_no_grad_cuda`，ACL、Torch shim、测试 hook 和下游
代码看到的类对象均不变。

## 兼容处理

- 6 个迁移对象仍与 `jittor.nn` 重导出保持同一身份，`__module__` 和 pickle 路径
  保持 `jittor.nn`。
- `_ln_function_cls` 是 `lru_cache` wrapper；元数据恢复现在递归覆盖 `__wrapped__`。
- 动态生成的 `_LN` 类及其 `execute/grad` 也恢复为 `jittor.nn` 来源。
- `instance_norm`、`group_norm` 和 `_ln_normalize` 经 facade 动态解析共享 helper，
  保留公开 monkeypatch 行为。
- `F.batch_norm/instance_norm/layer_norm/group_norm` 继续与 `nn.*` 是同一函数对象。
- `BatchNorm*`、`InstanceNorm*`、`LayerNorm*` 三组 1d/2d/3d 别名仍是同一类对象。
- CUDA kernel 保持原函数 AST 全等，继续排除 ACL，仅在 no-grad CUDA 下启用；
  fp16/fp32、bf16 私有 opt-in、标量 affine 开关和 non-finite double fallback 不变。

6 个迁移定义中，`batch_norm` 与 CUDA helper 严格 AST 全等；另外 4 个只增加 facade
动态解析或元数据恢复，不改变数值计算。

## 验证

| 验证 | 结果 |
| --- | --- |
| 结构契约 | 14/14 通过 |
| CPU norm + compat + serialize | 56 项：51 通过、5 项 CUDA 条件跳过 |
| CPU OpInfo | 6 项：4 通过、2 项按二阶能力声明跳过 |
| CUDA 结构/稳定反向/compat/fast path | 48/48 通过 |
| CUDA 设备前向与反向一致性 | LayerNorm、GroupNorm 2/2 通过 |
| wheel 内容 | 1,014 个条目；8 个 `_nn` 文件齐全；0 个禁止项 |
| wheel 隔离安装 | 51 项：43 通过、8 项环境/源码条件跳过 |

CPU 稳定性覆盖 LayerNorm、GroupNorm、InstanceNorm、BatchNorm 的小方差 float32
反向和 BatchNorm running variance 的 Bessel 修正。CUDA 覆盖同四类稳定反向、完整
normalization 前向、LayerNorm/GroupNorm weight/bias 反向，以及 LayerNorm fast path
的 3D、动态 rows、fp16/fp32、bf16 opt-in、标量 affine 和混合 affine dtype。

旧 `test_batchnorm` 的 InstanceNorm 比较在本批和未拆分基线 `80c97804` 上同位置
失败：Jittor 默认 `affine=True`，外部 PyTorch 默认 `affine=False`，测试没有统一
构造参数。独立 NumPy 的 InstanceNorm 测试均通过，因此登记为既有测试问题，不在
结构提交中修改语义。

wheel 检查确认没有 `.pyc`、`__pycache__`、`jittor/projects` 或 `jittor_fsdp2`。
隔离测试的 `jittor.__file__` 指向 wheel 安装目录。所有构建、JIT 缓存和临时
worktree 均位于 `/home/zy/projects/jittor-lab/_state/verify/source-refactor`；临时
worktree 已删除。

## 已知边界与后续

本机没有 Ascend/NPU 环境。ACL 路径完成源码审计和结构契约，但本批不宣称真实 NPU
执行验证。下一批优先迁移 RNN，再处理耦合更重的 convolution；二者都应保持类、
设备实现和参数协议整体迁移，不能只移动 Python 类壳。
