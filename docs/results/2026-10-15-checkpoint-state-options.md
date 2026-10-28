# Checkpoint StateDictOptions 与子模块契约

- Status: 单进程 CPU 与 CUDA 已验证；分布式 provider 路径保留。
- Reviewed: 2026-10-15。
- Baseline: `9d9310e3c9a4af737702e12d2d6d47062fa015cd`，从
  `origin/2.0-refactor` 重新拉取；实现与证据一并提交。
- Owner: Torch 兼容与 checkpoint 维护者。
- Recheck when: optimizer 归属、参数遍历、FSDP checkpoint hooks、
  支持的 Torch 版本或设备拷贝语义变化。

最初的 checkpoint 续训报告建立了普通的 FQN 导出与恢复，但明确排除了非空子模块选择。
一项新的最小复现确认了该排除、被忽略的 optimizer `strict` 选项，以及对真正共享参数
使用另一 FQN 时被拒绝。独立的二进制 PyTorch 2.11.0+cu128 在修改前确立了受支持的
行为。没有新增下游库或 adapter。

实现仍位于 `jittor.compat.torch.checkpoint_state`。native 参数遍历提供规范名与别名；
model 加载、参数拷贝、optimizer 状态安装与 optimizer 数学仍由各自既有的 owner 负责。
分布式 installer 只发布 checkpoint 函数与选项。没有更改 Module、inplace、native
optimizer 或 FSDP provider 的实现。

## 支持的选项与组合

| Surface | 已验证行为 |
| --- | --- |
| `submodules` | 模型导出按完整的点分前缀选择所请求的子模块，包括持久 buffer；选择 `a` 不会选中 `ab`。多个子模块保留各自不同的前缀。 |
| `keep_submodule_prefixes` | `True` 保留 model FQN；`False` 产生所选子模块的本地键，可加载进该子模块。 |
| 联合 `get_state_dict` | 消费与独立模型导出相同的子模块与 offload 选项。optimizer 状态保持完整，并使用根模型 FQN，与二进制 Torch oracle 一致。 |
| 以 Module 为键的模型加载 | 以子 Module 为键的字典会把本地状态键展开为根 FQN，并委托给普通模型加载；`strict=False` 报告未被触碰的模型键。 |
| `ignore_frozen_params` | 过滤冻结的模型参数，保留 buffer。它不过滤既有 optimizer 状态或参数组，与 Torch 一致。 |
| `strict` optimizer 加载 | 拒绝在其余部分已填充的 optimizer checkpoint 中缺少可训练参数状态；允许缺失冻结状态。`strict=False` 恢复已提供状态，并通过 optimizer owner 重置被省略参数的状态。 |
| FQN 与参数组 | 依据参数身份恢复，尽管活跃参数顺序与序列化的 FQN 顺序都已改变。保存的学习率与 betas 会被消费。`strict=False` 下的部分组归属保留活跃参数列表并消费保存的选项。 |
| 共享参数 | 对普通与展平 checkpoint 接受有效的替代参数 FQN，保留该别名，并再次导出规范 FQN。同一保存参数出现重复别名会被拒绝。 |
| `cpu_offload` | 导出 CPU 张量；恢复时把 moments 移回参数所在设备。下一步优化更新、每个参数梯度与输入梯度都与保存的轨迹一致。 |
| `full_state_dict` | 普通单进程 FULL 加 CPU offload 的往返独立于展平。FSDP FULL gather/shard 仍走既有 provider hook。 |
| `flatten_optimizer_state_dict` | 展平加 CPU offload 的往返独立于 FULL，包括共享参数别名恢复。 |
| 空选择与空载荷 | 空的 optimizer 可迭代对象使联合 get 对 optimizer 状态返回 `{}`。联合 set 也接受空迭代器。空的模型字典允许仅 optimizer 的联合加载。提供 optimizer 但 optimizer 字典为空时，在联合模型加载之前报错。 |

非分片的 optimizer 恢复在调用任何 optimizer 的 loader 之前，先校验所有缺失状态的
加载计划。一项回归提供两个 optimizer，修改第一个已保存的学习率，并移除第二个所需
的一个状态；严格模式的失败使第一个 optimizer 保持不变。这一额外的处务边界并不
构成对分布式集合通信或 native loader 失败进行原子回滚的主张。

## 明确的限制

- 同时以 `full_state_dict=True` 与
  `flatten_optimizer_state_dict=True` 加载会抛出 `NotImplementedError`，包括在
  model 写入之前的联合加载。独立的 Torch oracle 同样无法加载该组合；三种可移植的
  续训组合是普通/offload、FULL/offload 与 flatten/offload。
- `broadcast_from_rank0=True` 仍要求 `full_state_dict=True`。它委托给既有的分布式
  broadcast；本报告不新增任何多 rank broadcast 认证。也不主张新的多 rank FSDP 或
  分片子模块验收。
- 前缀移除针对一个选中的子模块得到验证。调用方必须避免多个子模块去掉前缀后的 FQN
  相互冲突；二进制 Torch 同样允许此类冲突覆盖某个条目。根模块与外部模块的选择不在
  本验收矩阵内。
- 严格加载遵循 Torch 的缺失状态契约，而不是对每个多余状态键施加严格相等：无关的
  状态条目被忽略，无状态 SGD 的空 `state` 映射仍可加载。空 optimizer 字典 `{}`
  与含有 `state` 和 `param_groups` 的有效字典不同。
- 冻结模型过滤针对非共享的子参数验证。optimizer 别名单独验证；冻结模型别名过滤
  的组合不在本矩阵内。
- 更新的 Torch 专用 `dsd_fqn_modifiers` 扩展未加入。既有的七选项构造与已测试的
  native optimizer 布局仍是受支持的表面。公开 fidelity 登记仍为 `APPROXIMATE`。
- ROCm、ACL、其他 CUDA 架构与多 rank 选项组合未在此执行。可选的 Accelerate
  端到端测试被跳过，因为该依赖未安装在隔离环境中；checkpoint 测试不需要任何下游包。

## 设备与 oracle 证据

测试使用 Python 3.11.16、GCC 11.5.0、nvcc 12.2.140、既有的 pip-cu122 Jittor 库，
以及一块可见的 RTX 4090（SM89），驱动 580.178.04。参考解释器加载独立的
PyTorch 2.11.0+cu128 及其二进制 `torch._C` 扩展，并移除了 Jittor 源码路径与 CUDA
库覆盖。CUDA 可用性与张量驻留在实际 GPU 上检查。参考侧使用 CUDA 12.8 库；这是在
同一设备上跨两个 runtime 的数值比较，而不是对共享工具链栈的认证。

固定的输入与权重数组使轨迹无需 RNG 或下游依赖即可复现。公开的 `torch.clone` 创建
不可变快照。同一测试文件在 shim 与二进制 Torch 进程中运行。CUDA shim 测试禁止后端
fallback、退出前同步、验证 model 参数与恢复的 moment 张量驻留在 CUDA，并断言
fallback 计数增量为零。CPU 测试在其 runtime 作用域内禁用 CUDA 执行。

捕获的下一步轨迹比较 24 个数值字段：loss、输入与全部参数梯度、每个 model 参数与
buffer，以及 optimizer moments 与 step 值。相对二进制 Torch 的最大绝对误差在 CPU 上
为 `7.45e-9`，在 CUDA 上为 `1.49e-8`，处于 `atol=2e-6, rtol=2e-5` 之内；两个设备上的
fallback 增量均为零。参数组选项的消耗有单独的断言。

按 `agent/manuals/environment.md` 使用隔离的源码、home、临时与 JIT 目录复现：

```bash
JITTOR_TORCH_SHIM=1 CHECKPOINT_DEVICE=cpu PYTHONPATH=python \
  python -m pytest -q compat/tests/torch/test_checkpoint_state_options.py \
  compat/tests/torch/test_optimizer_checkpoint_state.py
JITTOR_TORCH_SHIM=1 CHECKPOINT_DEVICE=cuda PYTHONPATH=python \
  python -m pytest -q compat/tests/torch/test_checkpoint_state_options.py \
  compat/tests/torch/test_optimizer_checkpoint_state.py

# Independent binary Torch: use its interpreter, with no deployed shim path.
env -u PYTHONPATH -u LD_LIBRARY_PATH CHECKPOINT_DEVICE=cpu \
  python compat/tests/torch/test_checkpoint_state_options.py -v
env -u PYTHONPATH -u LD_LIBRARY_PATH CHECKPOINT_DEVICE=cuda \
  python compat/tests/torch/test_checkpoint_state_options.py -v
```

把 `CHECKPOINT_OPTIONS_REPORT` 设为外部 JSON 目标即可捕获轨迹。原始日志、JSON 报告、
私有环境覆盖与 JIT 缓存未版本化，位于 `$JITTOR_LAB_ROOT/_state/refactor-goal/checkpoint-options/`。

## 验证结果

| 选择 | 结果 |
| --- | --- |
| 新选项测试，二进制 Torch CPU | 22 passed，1 项故意跳过的仅 shim 显式拒绝 |
| 新选项测试，二进制 Torch CUDA | 22 passed，1 项故意跳过的仅 shim 显式拒绝 |
| 新增与既有 checkpoint 测试，CUDA shim | 29 passed，1 项可选 Accelerate skip |
| CPU checkpoint、planner、FSDP2 配置/生命周期/参数角色与 optimizer owner 回归 | 95 passed，1 项可选 Accelerate skip |
| CPU `core`，native 模式 | 98 passed, 41 skipped, 1 existing expected failure |
| CPU `core`，Torch 模式 | 224 passed, 1 skipped |
| CPU `core`，两种模式合并 | 322 passed, 42 skipped, 1 existing expected failure |
| 完整 `tests/structure` | 1432 passed, 4 skipped, 2 documentation-integration failures, 628.44 s |
| 仓库布局与空白 | 通过 |
| 改动 Python 文件以 Python 3.7 语法解析 | 通过；这是语法检查，不是 Python 3.7 运行时主张 |

最终的 CUDA 选择会同步并检查输入、输出、loss、梯度、model 参数与 buffer、恢复的
moment 张量的物理 `location()`、native dispatch 后端与公开设备。CPU 在禁用 CUDA 的
情况下使用同样的 native dispatch 检查。另一项外部互操作探针把二进制 Torch 实际产出
的 checkpoint 加载进 shim，使用反转的活跃参数槽位，然后在 CPU 与 CUDA 上以零 fallback
复现参考侧的下一步。该探针还消费了真实 Torch 额外的参数组元数据，而没有改变 native
AdamW 的数学。

直接的 CPU 与 CUDA checkpoint 测试在任务已暖机的 CUDA 可用缓存中顺序运行，处于显式
的 CPU 或 CUDA runtime 作用域下。CPU core runner 构建了独立的仅 CPU native 与 Torch
缓存。首次核心与外部库构建串行执行；没有 benchmark 使用这些缓存。OneDNN 3.9.1 在
任务状态中构建，并作为只读外部库供 CPU core 会话复用。

两项完整结构失败是
`test_docs_structure.py::TestDocsStructure::test_every_page_is_reachable_and_every_toctree_entry_exists`
与
`test_packaging_structure.py::TestPackagingStructure::test_manifest_covers_runtime_trees_without_cache_payloads`。
它们的 trace 只指向这一新报告：它需要在 results toctree 中有一个条目，并在根
`MANIFEST.in` 中有一行生成的 include。兼容性 manifest 已与其生成器匹配。共享索引与
manifest 的更新保留给统一整合；这一孤立结果不声称完整结构门禁通过。四项 skip 中，
两项是已声明的，两项是孤立解释器无法执行的已安装包解析检查。

统一整合后的主树验证见
[整合回归与 NCCL 缺陷收口](2026-10-15-integration-nccl-coverage.md)：
报告索引与 MANIFEST 已登记，完整结构门禁 1441 passed、6 skipped、0 failed。
CPU checkpoint 30 项、CUDA checkpoint/Module 状态 37 项全部通过，包含
真实 Accelerate 保存与续训；NCCL 2.18.3 双卡 bf16 FULL 恢复及续训
每 rank 1 passed、0 skipped，fallback 为 0。这补充了该选定多 rank
路径的整合证据，仍不扩大为全部 StateDictOptions 多 rank 组合验收。
