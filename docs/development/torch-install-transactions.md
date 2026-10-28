# Torch 安装事务

- 状态：已实现
- 复查日期：2026-10-05（基线 `e3c369acb`）
- Owner：Torch 兼容维护者
- 复查触发：某个安装器开始写入新的进程状态、新增延迟钩子归属者，或
  `compat/transaction.py` 中的回滚助手发生变化时

安装 Torch 前端会写入进程级全局状态：命名空间绑定、Jittor flag、环境变量、导入机制与类属性。
本页说明其中哪些写入是可逆的、谁拥有它们，以及测试对回滚能主张什么。安装器组合本身（步骤、
标记、`InstallContext`）见 [Torch lowering](../compatibility/torch-lowering.md) 第 2 节，以及
[源码架构](source-architecture.md) 的兼容 API 部分。

## 契约：一份可逆的变更账本

`compat/transaction.py` 实现的是一份**可逆的变更账本**，而不是在第一次不可逆写入之前就中止的
硬失败契约。经由它处理的每一次写入都连同其先前值一起记录，安装失败时按相反顺序撤销。若回滚
发现某个槽位已被另一个参与者替换，它不会覆盖，而是继续恢复相互独立的条目，然后抛出
`TransactionConflict` 并点名每一个它无法恢复的槽位，同时把事务留在可查询的 `failed` 状态。重试
会构造一份新事务并重新运行安装步骤，而不是相信那些已被回滚的完成标记。

两种账本共享同一把可重入的进程锁（模块补丁锁与源解析锁总是在它之后获取）：

- `InstallTransaction` 记录一次安装，直到提交。
- `RuntimeHook` 记录一次延迟激活，并在钩子活跃期间一直保留它的撤销账本。

## 记录了哪些状态

| 状态 | 经由什么写入 | 是否回滚 |
| --- | --- | --- |
| `torch*` 命名空间（绑定、删除、删除掩码）、安装标记与已发布的注册表 | publication、命名空间快照撤销、`record_object_diffs`/`record_mapping_diffs` | 是 |
| `jt.flags` | `set_flag(flags, name, value)` | 是，按每个具名 flag |
| `os.environ` | `set_env(key, value)` | 是 |
| 对象与类属性 | `set_attr`、`module_patcher.patch_method`、`RuntimeHook.mutate_attr` | 是；类描述符从本地 `__dict__` 读取，因此 staticmethod 与继承属性得以保留 |
| `sys.modules` | `publish_module`、`RuntimeHook.replace_module` | 是；槽位中已有的外部模块是冲突，绝不覆盖 |
| `sys.path` | `mutate_path` | 只移除自己拥有的那次插入 |
| `sys.meta_path` finder | `module_patcher` 中的 `record_undo` 与 permissive-package finder | 是 |
| `module_patcher` 注册表 | `record_undo(restore_registry)` | 是 |
| `builtins.__import__` | 没有安装器写入它 | —— 将来若有人开始写，必须走账本 |
| 第三方模块补丁回调的副作用 | 任意代码 | **否** |

`jt.flags` 无法做快照：它的原生对象暴露动态 flag 属性，因此 `flags.__dict__` 不是权威枚举。
账本因此只记录安装器通过 `set_flag` 写入的那些 flag——这是代码里的一份显式白名单：
`compat/torch/installers/core.py` 的安装期 flag（`_set_install_flag`）以及
`installers/factories.py` 与 `installers/distributed.py` 中的 `use_cuda`。

分布式安装器通过 `set_env` 写入 `JT_NCCL_WORLD_SIZE`、`JT_NCCL_RANK`、`JT_NCCL_LOCAL_RANK`、
`JT_NCCL_ROOTINFO_FILE`、`use_nccl` 与 `use_mpi`，然后通过 `set_flag` 写入 `use_cuda`。由于
回滚按相反顺序执行，flag 会在它所依赖的环境之前恢复。

## 写入助手只有一个归属者

`compat/transaction.py` 拥有查找与三种写入：`active_transaction()`、`set_flag()`、`set_env()` 与
`set_attr()`。安装器调用它们；它们中没有一个会再从 `jt._torch_compat_install_context` 推导当前
事务。

- `active_transaction()` 只在状态为 `open` 时返回账本。`InstallTransaction.record()` 拒绝已提交
  或已回滚的账本，而同样的助手会在安装很久之后于运行时被调用（`torch.zeros(device="cuda")` 经由
  factory 归属者抵达 `set_flag`），因此已关闭的账本必须把调用变成直接写入，而不是
  `RuntimeError: transaction is rolled_back`。
- `set_env()` 在记录路径与直接路径上都施加 `str()`。一条路径存原始整数、另一条路径存它的文本，
  曾让 rank 变量在回滚期间通不过自己的归属者检查。

有一处 `use_cuda` 写入被刻意放在账本之外：`installers/nn/module_methods.py` 中
`Module.to(device=...)` 表达的是用户在安装之后提出的请求，随安装回滚它等于撤销调用方要求的事。
`torch.backends` 的 TF32 开关完全不再写 Jittor flag：它们在 `installers/cuda/api.py` 中设置前端
拥有的精度档位（见 [float32 精度](../notes/float32-precision-policy.md)）。

## 延迟钩子

`runtime_hook(owner)` 与 `owned_runtime_hook(owner)` 装饰器在不导入 Jittor 的前提下建立当前的记录
作用域；`current_transaction()` 同样不依赖 bootstrap。经由 `set_attr` 或 `patch_method` 的属性
写入，以及经由 `RuntimeHook.replace_module` 的模块发布，都属于该作用域。清理时检查当前对象的
标识，因此外部替换得以保留，并作为 `TransactionConflict` 报告。

- `release_runtime_hooks(owner)` 按相反顺序释放某个归属者的效果；一次冲突不会阻止更早的独立条目
  被恢复。
- 延迟模块补丁回调的归属者形如 `("module_patch", module_name, callback)`，由
  `release_module_patch_hooks` 释放。回调作者必须用变更助手修改类或注册表：任意的第三方副作用
  无法从模块字典快照中推断或撤销。
- `compat.torch.install(torch, parent_transaction=activation)` 在子账本提交之前用
  `activation.adopt(child)` 转移一份成功的子账本，因此之后外层失败仍会恢复类、模块、完成标记与
  缓存的张量状态。保真度登记经由同一作用域记录单条 API 记录。
- `torch.library` 的注册表与元数据写入加入当前的 `InstallTransaction` 或 `RuntimeHook`；
  `Library` 声明 `_transactional_registry = True`，adapter 不得为同一批槽位再加一份快照撤销。
- vLLM adapter 的激活拥有它的扩展替身、API 版本选择、已注册算子与安装标记；flash-attention 的
  发布有一个嵌套的、可独立释放的归属者。已有的外部模块槽位或算子名不会被覆盖。
- C++ 扩展 facade 在执行模块之前发布，使递归导入看到同一个对象；加载器失败会恢复先前的已拥有
  槽位，外部替换则是硬失败。构建产物不会因模块回滚而被删除。
- 源根解析使用带标识标签的路径插入和一个临时的、当前线程的加载器跟踪器。回滚只触碰已记录的
  发布；并发的路径变化与无关导入保持原样。自定义源码加载器通过 `publish_source_module` 发布；
  源根下未被跟踪的发布会被保留并作为硬冲突报告。公开的解析器入口是 `load()`；修改导入状态的
  助手需要它的活跃作用域。

## 测试可以主张什么

子进程测试由 `_helpers.child_process.child_env()` 隔离，并且必须显式校验子进程的 `PYTHONPATH` 与
模式变量。一个干净的子进程环境并不是父进程安装失败后恢复了状态的证据。

测试可以断言上表中已记录归属者的回滚，以及子进程隔离。它们不得为该表之外的状态主张完整的安装
回滚——尤其是模块补丁回调的第三方副作用。

- `compat/tests/torch/test_transaction.py`、`test_install_context.py` 与
  `test_runtime_hook_ownership.py` 针对受控的状态归属者运行真实的账本、钩子、扩展发布与源加载器
  代码；它们既不需要 Jittor 编译，也不需要加速器硬件。
- `compat/tests/structure/test_torch_install_state_boundary.py` 钉住本页，以及分布式安装器显式的
  `set_env`/`set_flag` 写入。
- 集成的激活/上下文测试仍然是完整前端的对照。
