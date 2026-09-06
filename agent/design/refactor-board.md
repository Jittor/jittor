# 整改看板

当前进度以任务表为准。2026-09-06：2.13 完成原生状态与配置分层，2.14 清空原生 misc，4.03/4.04 接通原生设备与算子执行链，4.05 完成 Python 真实分派迁移；
4.06 实际回退策略与 4.07 BuildConfig 已接通，4.10 统一 CUDA 内核目录，4.11 接通 ACL 注册执行并删除 Python 替换与全局编译钩子；下一步移除 legacy SDK 源转换。下方旧波次中的 Python 字段视图不等于原生存储迁移。

> 第217波：`98c8ee94` 迁移 cuda_allow_tf32 Runtime owner（结构 43 passed）；`b8398291` 修正 Native provider teardown 统一 lifecycle events（结构 12 passed）；`9d49c70c` ACL device_size Python/C++ 对齐（ACL 14 passed）；`d44782d4` Torch bootstrap 非字符串/非法 __all__ fail-closed。未声称 CUDA/NPU 实机。


> 第216波：`68b4d443` 迁移 use_tensorcore Runtime owner（结构通过）；`32f1ea99` 统一 Native provider lifecycle value events（结构 12 passed）；`56231caa` ACL DescriptorCache generation-safe release（host-only 25 passed）；`645649b3` Torch publication graph 拒绝未声明额外模块（namespace 41 passed）。未声称 CUDA/NPU 实机。


> 第215波：`2999cce2` 迁移 use_cuda_host_allocator Runtime owner（结构 41 passed）；`97224dbb` Native provider lifecycle event ABI contract（结构 12/JIT通过）；`e4393dc3` ACL descriptor device_size lifecycle count（ACL 10 passed）；`db84f2d0` Torch bootstrap __all__ canonical contract（namespace 40 passed）。未声称 CUDA/NPU 实机。


> 第214波：`22f0889c`/`a09ca82f` 迁移 trace_depth Runtime owner/pyi（结构 40 passed）；`fbbc69d7` Native generation-checked consumer lease（结构 11/JIT 2 passed）；`faca96ac` ACL consumer view 生命周期约束（ACL 20 passed）；`c7393df6` Torch distribution graph 输入 fail-closed（namespace 39 passed）。未声称 CUDA/NPU 实机。


> 第213波：`1f67af34` 迁移 trace_py_var Runtime owner（结构 39 passed）；`8c38c342` Native provider scopes move-only RAII（结构 10/JIT通过）；`a8e4b743` ACL DescriptorCache malformed canonical key fail-closed（host-only 22 passed）；`5f97723f` Torch distribution 显式 aliases fail-closed（namespace 38 passed）。未声称 CUDA/NPU 实机。


> 第212波：`3d73ab52` 迁移四项 allocator policy Runtime owner（结构 38 passed）；`058a6572` 增加 Native OpId consumer dispatch/index（结构/JIT 合同通过）；`91147566` ACL DescriptorCache acquire/lease/tombstone/generation 生命周期（host-only 21 passed）；`234d7dc8` Torch distribution metadata canonical identity（namespace 36 passed）。未声称 CUDA/NPU 实机。


> 第211波：`85d2279e` 迁移 cpu/device memory limits Runtime owner（结构/CPU 通过）；`7b5bda2e` provider generation-safe unbind（结构 10 passed）；`e2fbafd6` canonicalize Nano types 到 `src/type`（结构 14 passed，2.14 子切片）；`69709bdc` standalone torch distribution metadata（namespace 36 passed）。未声称 CUDA/NPU 实机。


> 第210波：`1b2ec5a9` 迁移 rewrite_op Runtime owner（结构 36 passed）；`1c042033` 统一 NativeProvider ABI admission（结构 10/JIT 1 passed）；`01374b45` 完成 5.25 tracer/NVTX/MANIFEST 脱离 utils drawer（打包/结构 17 passed，尾项已闭环）；`afc58e5f` 收紧 Torch distribution manifest canonical order/type（namespace 34 passed）。未声称 CUDA/NPU 实机。


> 第209波：Runtime `use_parallel_op_compiler` owner（结构 35 passed）；Native registration scope + 5.25 dumpdef/build 布局迁移组合提交 `38862b20`（provider 合同 9、布局 11 passed）；Torch synthetic module import metadata `a27e95e4`（namespace 33 passed）。组合提交已记录，后续恢复单任务单提交；未声称 CUDA/NPU 实机。


> 第208波：跨硬件/布局前置继续：`b70afbce` 迁移 float32_matmul_precision Runtime owner（结构 34 passed）；`c04941d9` 增加 Native fail-closed consumer dispatch probe（结构 8 passed）；`b37ec5c1`/`ad41509a`/`9c7a457a` descriptor 前置已在本波完成链；`7eb5ea90` 收紧 Torch distribution alias canonical endpoint（namespace 33 passed）；`416a7fe4` 完成 dlink compiler 进入 build 包并修路径合同（layout 11+1 passed）。未声称 CUDA/NPU 实机。


> 第207波：异机前置继续：`bb0e9897` 迁移 enable_tuner Runtime owner（runtime/root结构合同通过）；`280762dd` 增加 Native observer scope RAII identity restore（结构 8/JIT 2 passed）；`9c7a457a`/`2c2339a5` 增加 ACL descriptor per-device generation/stale-handle guard（host-only 19 passed）；`3079d012` 收紧 standalone torch alias schema validator（namespace 35 passed）。未声称 CUDA/NPU 实机。


> 第206波：异机前置继续：`5b2c7d08` 迁移 node_order Runtime owner（结构 32 passed）；`8ab9591f` 原子发布 Native consumer metadata+dispatch snapshot（结构 7 passed）；`ad41509a` ACL DescriptorCache 增加 erase_device 生命周期失效（C++/Python 19 passed）；`23d2f648` 增加 standalone torch bootstrap surface validator（namespace/structure 37 passed）。未声称 CUDA/NPU 实机。


> 第205波：异机前置继续：`e4399f28` 迁移 log_silent/log_sync/log_v Runtime owner（结构 31 passed）；`838ad234` 增加 NativeProviderConsumerContract ABI/provider一致性（结构 6 passed）；`b37ec5c1` ACL descriptor cache 单键失效（ACL 26 passed）；`3e2e9633` 统一 standalone torch distribution facade（namespace 30 passed）。未声称 CUDA/NPU 实机。


> 第204波：异机前置继续：`5656f81a` 迁移 trace_var_data Runtime owner（结构 28 passed）；`6c073efa` 增加 NativeProviderMetadata value snapshot（结构 5 passed）；`d328bb5a` 建立 ACL descriptor key/cache host-only boundary（ACL 26 passed）；`165f9a0e` 增加 standalone torch distribution manifest validator（namespace 27 passed，既有 install-context 计数失败未归因本切片）。未声称 CUDA/NPU 实机。


> 第203波：异机前置继续：`32ad37ef` 迁移 try_use_32bit_index Runtime owner（结构 27 passed）；`a2f44691` 增加 Native observer teardown identity guard（C++/结构 4 passed）；`782c7663` 固定 ACL cache key 使用 classic locale（host-only 13 passed）；`3ac67208` 集中 Torch independent publication 原子边界（namespace 25 passed）。未声称 CUDA/NPU 实机。


> 第202波：异机前置继续：`97d5ce9a` 迁移 disable_lock Runtime owner（结构 26 passed）；`3a617079` 增加 Native provider stale dispatch key freshness guard（结构 4 passed）；`d3f95d4a`/`4fc52287` 收紧 ACL invalid schema type 与 consumer registration fail-closed（ACL 17 passed）；`53e92aba` 增加 standalone torch publication graph validator（namespace 24 passed）。未声称 CUDA/NPU 实机。


> 第201波：异机分发/ABI 前置继续：`b3034f92`/`97d5ce9a` 迁移 reuse_array Runtime owner（结构 25 passed）；`27ab42a0` 增加 NativeProviderLifecycleObserver lifecycle sink（C++/JIT contract 通过）；`34ab1d99` 增加 ACL AclAttrRunnerContract host-only binding/consume（ACL 13 passed）；`046d20c4` 完善 standalone torch distribution boundary（namespace 22 + structure 27 passed）。未声称 CUDA/NPU 硬件。


> 第200波：跨硬件组织前置继续：`9d8e2a5a` 迁移 auto_convert_64_to_32 Runtime owner（结构 24 + CPU 1 passed）；`46b1fca7` 增加 NativeProviderRegistration ABI/struct-size/generation contract（相关合同 93 passed/2 skipped）；`62ac7f59` 增加 ACL AclDataView/consume host-only consumer seam（ACL 18 passed）；`94637aa5` 建立 standalone torch distribution import manifest（manifest/bootstrap/alias 合同通过）。未声称 CUDA/NPU 实机。


> 第200波 7.05 子项四个 family：`4ecfb14f` 修 `install()` 失败路径——`rollback()` / `release()` / 弹出 ledger 句柄三步无 `finally`，而 `rollback()` 正是那里最可能抛的一步，于是类级 RLock 永久被占、死 ledger 留在 `context.state`；`rollback()` 另改为冲突后继续回滚其余条目并置 `failed`。`dcbbedf3` 把六份 inline 的事务查找收归 `transaction.active_transaction()`/`set_flag`/`set_env`/`set_attr`（六份里只有一份检查了 `state == "open"`，失败安装留下的 ledger 会让此后的 `torch.zeros(device="cuda")` 抱 RuntimeError），并消去第134波引入的 `_set_use_cuda` 重复（`test_cleanup_structure` 红转绿）。`0e336b58` 把 vLLM arming finder 的 undo 改成 owner-aware，并发外部替换不再静默通过。`c2fa74d8` 新增 `test_compat_write_entry_points.py`：AST 扫出 compat 全树 50 个进程全局写入口并分类闭集（ledger 17 / pre-ledger 12 / runtime 11 / pending 8 / deployed-payload 5），两个方向均造过反例。三套门禁：`tests/structure` 15 failed/881 passed 与 `803e3785` 基线（15 failed/872 passed）同集合；CPU torch 模式 `tests/compat/torch` 33 failed/1265 passed 同集合；CUDA torch 模式把本波 10 个源文件回满后定向归因，before/after 的 FAILED 集合逐行相同（16 failed/109 passed）。**看板原记的「`tests/structure` 2 条既存失败」过时，实测 15 条**；剩余四处写入口已钉在 `PENDING`，7.05 保持待领。
>
> 第199波：跨硬件组织前置继续：`2276d09f` 迁移 missing_grad_error Runtime owner（结构 23 passed）；`d8edc265` 为 native provider dispatch 增加 generation/provider_id（registry 29 + C++ 3 + JIT 2 passed）；`e00054b0` 增加 ACL AclDataOwner host-only seam（5 passed）；`195544aa` 将 Torch publication aliases 统一走 publication boundary（namespace 18、alias 1、bootstrap 1 passed）。均未声称硬件实测或聚合任务完成。


> 第198波：异机测试代码组织前置完成：`2d772ef4`/`8d70ae4a` 迁移 AMP 与 profiler metadata owner（结构 22/20 passed）；`c1a67c91`/`b4b7c990` 集中并 provider-aware 化 native OpRegistry（C++14 syntax/contract 通过）；`e2731c0f`/`2dcc3448`/`7a03fa48` 建立 ACL host-only data schema + C++ decoder boundary（host-only 15 passed）；`8fcdc20f` 拆 Torch publication，`09c7665c` 完成 miniz third-party 布局。均未声称 CUDA/NPU 硬件验证。


> 第197波迁移前置：`8d70ae4a` 增加 Runtime profiler metadata owner（结构 20 passed）；`c1a67c91` 集中 native C++ OpRegistry ownership（C++14 syntax + contract 3 passed）；`e2731c0f`/`2dcc3448` 建立无 CANN ACL data schema/normalizer（host-only 11 passed）；`8fcdc20f` 拆出独立 Torch publication boundary（namespace 17 passed，既有 unknown-submodule timing 失败未归因本切片）。这些都是 CUDA/NPU 异机测试前置，整项仍待领。


> 第196波增量：`8aae611d` 将 `profiler_record_peek` 纳入 RuntimeContext owner（结构 18 passed）；`3add5deb` 收紧 RegistrySnapshot provider query（合同 29 passed）；`8699c440` 修复删除 TorchNamespace 元数据后的 owner 泄漏（namespace 17 passed）；`5d79118f` 统一 standalone/nox 线程池环境策略（结构 44 passed）。聚合任务仍保持「待领」。


> 第195波增量：`a13b9cf6` 将 `profiler_rerun` 纳入 RuntimeContext owner（结构 17 passed）；`148ea645` 增加 RegistrySnapshot provider 查询与 teardown 隔离（合同 28 passed）；`2831a09e` 对称化 TorchNamespace `__delattr__` ownership（namespace 16 passed）；`4dc47dae` 统一 standalone runner 六类线程池 policy（结构 8 passed）。聚合任务仍保持「待领」。


> 第194波增量：`e59adcdf` 将 `profiler_enable` 纳入 RuntimeContext owner（结构 16 passed）；`f46fa787` 收紧 RegistrySnapshot ownership 输入（合同 27 passed）；`3eca5513` 隔离 TorchNamespace 模块元数据（namespace 14 passed）；`f7770109` 对齐 standalone runner worker policy（结构 15 passed）。聚合任务仍保持「待领」。


> 第193波增量：`a2f29c02` 将 `check_graph` 纳入 RuntimeContext owner（结构 15 passed）；`6db5a349` 增加 RegistrySnapshot 所有权不变量（合同 25 passed）；`b77217e8` 为独立 TorchNamespace 根补标准 ModuleSpec（namespace 13 passed）；`863f013f` 统一 standalone runner 与 nox 执行策略（结构/环境 26 passed）。聚合任务仍保持「待领」。


> 第192波增量：`e8608976` 将 `no_fuse` 纳入 RuntimeContext owner（结构 14 passed）；`537968fa` 隔离 RegistrySnapshot/provider state 输入与生命周期（合同 23 passed）；`e8cf42d6` 修复 TorchNamespace alias 预检回滚（namespace 12 passed）；`91ca525f` 对齐 budget report 默认配置与 nox runtime workers（结构/环境 32 passed）。聚合任务仍保持「待领」。


> 第191波增量：`ce1b9276` 将 `profiler_warmup` 纳入 RuntimeContext owner（结构 13 passed）；`18ddbd72` 增加 RegistrySnapshot/snapshot_state 一致性（合同 21 passed）；`c6409827` 隔离独立 TorchNamespace 的 root alias（namespace 11 passed）；`f3ef7a70` 统一 nox/预算报告/CLI runtime worker policy（结构 18 passed）。聚合任务仍保持「待领」。


> 第190波增量：`7352b82d` 将 `profile_memory_enable` 纳入 RuntimeContext owner（结构 12 passed）；`6f1a9f35` 深冻结 capability snapshot 并增加 registry snapshot（合同 20 passed）；`49906c5b` 预检 TorchNamespace 完整父级闭包（namespace 10 passed）；`895f69a9` 在 pytest 前输出 effective smoke budget（结构/环境 28 passed）。聚合任务仍保持「待领」。


> 第189波增量：`61ae6160` 将 `use_threading` 纳入 RuntimeContext owner（结构 11 passed）；`df1743ff` 增加 capability 原子撤销（registry 合同 18 passed）；`fda7501c` 对不完整 TorchNamespace 子模块发布 fail-closed（namespace 9 passed）；`bee0263e` 对齐 configured/runtime workers 并校验预算参数（结构/环境 27 passed）。聚合任务仍保持「待领」。


> 第188波增量：`8eb8b7b2` 将 `exec_called` 纳入 RuntimeContext owner（结构 10 passed）；`a1f5e649` 原子发布/撤销 backend capability（registry 合同 17 passed）；`20cea34c` 修复 TorchNamespace 根条目绑定回滚（namespace 8 passed）；`1ecb35ff` 区分配置 worker 与 quota 截断后的实际 worker（结构/环境 26 passed）。聚合任务仍保持「待领」。


> 第187波增量：`04aae3a5` 将 `gopt_disable` 纳入 RuntimeContext owner（结构 9 passed）；`6ac1fe8c` 增加 capability dispatch/fail-closed（registry 合同 15 passed）；`040b39cf` 修复 TorchNamespace registry root 事务回滚（定向 19 passed）；`de78f9ae` 暴露 smoke runtime worker/quota/线程诊断（结构 25 passed）。聚合任务仍保持「待领」。


> 第186波增量：`78fec631` 将 `no_grad` 纳入 RuntimeContext owner（结构 8 passed）；`b83e6889` 支持 provider replacement 并清理旧 backend kernels（结构/行为 13 passed）；`c7bfe24b` 在 pytest 前按预算 fail-closed（结构/环境 24 passed）；`b84498c4` 将 TorchNamespace 子模块绑定纳入事务回滚（定向 9 passed）。聚合任务仍保持「待领」。


> 第185波增量：`8b1c2707` 将 `auto_flush_ops` 纳入 RuntimeContext owner（结构 7 passed）；`ba2c88e5` 增加 backend kernel 原子 teardown（结构/行为 11 passed）；`d52f02ac` 锁定 Torch activation namespace 模式（bootstrap/namespace 11 passed）；`cab14f53` 按 cgroup quota 裁剪 gate workers（结构 24 passed）。聚合任务仍保持「待领」。


> 第184波增量：`ac641485` 将 `lazy_execution` 纳入 RuntimeContext owner（结构 6 passed）；`86b9c1cd` 支持 `cuda`/`cuda:<id>` registry location 并对未知 backend fail-closed（结构 10 passed）；`5c6876cd` 将 cgroup CPU quota 纳入 gate worker budget（22 passed）；`330d0a4c`/`9bbf87aa` 让独立 TorchNamespace 可由显式 activation 发布并接通事务替换（namespace 2 passed、bootstrap 1 passed，py_compile/diff-check 通过）。聚合任务仍保持「待领」。


> 第183波增量：`1c57b2a0` 将 `use_cuda` 纳入 RuntimeContext owner（结构 5 passed）；`32e8517b` 增加 OpRegistry 生命周期/注销错误路径（结构 9 passed）；`41eb41c2` 建立独立 TorchNamespace seam（定向 2 passed）；`f239e2ed` 增加 smoke budget 可执行瓶颈报告（结构 13 passed，预测 445.75/480s）。聚合任务仍保持「待领」。


> 第182波增量：`f37da269` 在 RuntimeContext owner 中增加只读 `device_id`（结构 4 passed），并将 CPU `flatten` 接入 Backend/OpRegistry（registry 合同 8 passed，shape/value 回归通过）。2.13 与 4.03/4.04 仍是聚合任务，继续待领；0.15 的 loadgroup 已实测为安全调度上限。


> 第181波增量：`409de4ea` 将 `jt.runtime.sync_run` 纳入 RuntimeContext owner（结构 3 passed）；`756a0fb6` 让 Torch stop_grad 清理 requires_grad owner（Torch state/autograd 定向 37 passed）；`db0f2a27` 补齐真实 CPU location 与 outer/clamp registry 回归（7 passed）；`faad4898` 将 smoke 独立组改用 loadgroup，11 个 nodeid 结论逐条 IDENTICAL。2.13/7.12/4.03/4.04/0.15 均仍是聚合任务，保持「待领」。


> 第180波增量：`696e5088` 将 7.12 requires_grad 状态收归 `TorchTensorState` owner（生命周期 6 passed）；`6e5c2d5c` 将 CPU `clamp` 接入 BackendRegistry/OpRegistry 真实分派（registry 合同 5 passed、CPU 数值通过）；`adf96b02` 完成 2.19 `reindex_reduce` 负 shape 用户边界。三项均为聚合任务的真实前置，完整任务继续保持「待领」。

> 第180波增量：2.19 `reindex_reduce` shape-value 用户边界完成迁移。`reindex_reduce_op.cc` 将负 shape 检查从 `CHECKop` 改为可跨 pyjt 捕获的 `USER_CHECKop`；现有 `[-1]` 负向用例与结构分类门禁通过（`tests/structure/test_error_categories.py`、`tests/ops/test_reindex_reduce_op.py` 共通过）。2.19 聚合任务仍保持「待领」，其余调用点与 CUDA 多卡/不可达 cuTT 证据未被此切片覆盖。

> 第102波增量：`cb0a4e77` 完成 cuBLAS batched matmul rank 用户错误 cohort（结构 3 passed）；`dbe72f0f` 完成 torch.polar numerical owner cohort（CPU 2 passed）。2.19/7.03 聚合任务仍按剩余范围保持「待领」；`6.B16` 仍待 Ascend 910B3 实机。
>
> 第103波增量：`e38cce97` 完成 cuBLAS acc matmul rank 用户错误 cohort（结构 3 passed）；`6fdb6120` 完成 complex accessor numerical owner cohort（CPU 2 passed）；`448aa10a` 删除 ACL 无消费者 `op_idx_map`（静态合同 68 passed）。2.19/7.03 聚合任务仍待完整审计，ACL 仍待硬件验证。
>
> 第104波增量：`0e275f14` 完成 cuBLAS batched matmul inner-dim 用户错误 cohort（结构 4 passed）；`84f29d9b` 完成 `hann_window`/`stft` signal numerical owner cohort（CPU 3 passed）；ACL `aclnn.h` 头文件契约由 `1e8e90c6` 已完成。聚合任务和硬件验收状态不变。
>
> 第105波增量：`7d366087` 完成 cuBLAS batched matmul batch-shape 用户错误 cohort（结构 5 passed）；`f848a6a7` 完成 `torch.equal` numerical owner cohort（CPU 3 passed）。ACL `AclOpFunctions` 类型擦除需跨注册表/构造器/owner 协同，未做不安全半改。
>
> 第106波增量：`cc9ed2bc` 完成 cuSPARSE CSR shape 用户错误 cohort（结构 3 passed）；`258be343` 完成 `kron`/`logsumexp` numerical owner cohort（CPU 2 passed）；`15c86886` 记录 ACL 类型擦除/属性 data/descriptor cache 的原子迁移边界（静态合同 1 passed）。2.19/7.03/8.06 聚合任务仍按剩余范围保持「待领」。
>
> 第107波增量：`a831fbd6` 完成 cuSPARSE COO shape 用户错误 cohort（结构 3 passed）；`f20bb4ff` 完成 `all`/`any` reduction numerical owner cohort（CPU 2 passed）；`7a35e8e7` 记录 ACL 属性 data 通道候选的协同边界（静态合同 1 passed）。聚合任务和硬件验收状态不变。
>
> 第108波增量：`55a1f481` 完成 cuBLAS matmul 输入 b rank 用户错误 cohort（结构 4 passed）；`9954b4ed` 完成 `tensor_split`/`take` numerical owner cohort（CPU 2 passed）；`296e0837` 补充 ACL 属性 data/descriptor/type-erasure 迁移顺序与 910B3 验收合同（静态合同 1 passed）。聚合任务和硬件验收状态不变。
>
> 第109波增量：`354ec6f1` 完成 cuTT transpose 标量输入用户错误 cohort（结构 3 passed）；`aa3ca532` 完成 `index_copy` numerical owner cohort（CPU 2 passed）。ACL `softmax.dim`/`triu.diagonal` data-channel 迁移仍需统一 schema、C++ 解码和 cache-key 协同，未做半改。
>
> 第110波增量：`26ee6bee` 完成 cuDNN RNN descriptor mode 用户错误 cohort（结构 3 passed）；`d3e33b0e` 完成 `index_put` numerical owner cohort（CPU 2 passed）。ACL data-channel C++ 解码入口仍缺统一实现，未做半改；0.15/10.19 验收状态不变。
>
> 第111波增量：`038819e6` 完成 cuBLAS acc matmul inner-dim 用户错误 cohort（结构 4 passed）；`64f4ecfa` 建立 `torch.autocast` numerical owner（CPU 2 passed）；`3cb0da4e` 形成 ACL data-channel schema 草案与 910B3 验收边界（静态合同 1 passed）。聚合任务和硬件验收状态不变。
>
> 第112波增量：`d2532ac2` 完成 cuDNN convolution format 用户错误 cohort（结构 3 passed）；`dbdfb6d7` 完成 `index_copy_` 原地 numerical owner（CPU 2 passed）；`a768cc9b` 锁定 ACL data schema 静态合同（2 passed）。聚合任务和硬件验收状态不变。
>
> 第113波增量：`de5188ab` 完成 cuDNN backward-x format 用户错误 cohort（结构 3 passed）；`11e9b456` 完成 `index_put_` 原地 numerical owner（CPU 2 passed）；`4b22f6d9` 增加 ACL data schema Python validator（静态合同 6 passed）。聚合任务和硬件验收状态不变。
>
> 第114波关闭 `10.02`：`151c5856` 将 `cpu` 加入 `nox.options.sessions` 默认列表，新增 AST 合同确认默认数值门禁；定向 1 passed。看板已合并 200、待领 74。
>
> 第115波关闭 `10.01`：`5501d0b6` 增加稳定 `nox -s full` 完整 CPU/nightly 入口并切换 CPU workflow；调度 AST 合同 2 passed。看板已合并 201、待领 73。
>
> 第116波整卡审计：未发现可在当前环境完整关闭的新任务。`0.15` 仍约 390 s，`2.19`/`7.03`/`8.06` 仍有聚合剩余，ACL/HCCL 仍待 910B3 实机；本波无代码提交，待领保持 73。
>
> 第117波增量：`c329a822` 完成 cuDNN backward-w format 用户错误 cohort（结构 3 passed）；`24637c96` 完成 vmap numerical owner wrapper（CPU 2 passed）；`25c5ffed` 明确 ACL C++ decoder 接口、错误映射和 canonical cache key 合同（静态合同 2 passed）。聚合任务和硬件验收状态不变。
>
> 第118波整卡审计：8.12、10.17、10.18、10.20、10.21 均缺完整实现/门禁；0.20、0.22、9.01、9.07、9.19 受布局、CUDA、导入或前置依赖阻塞。本波无代码提交，待领保持 73。
>
> 第119波增量：`bb5f72f1` 完成 tensor installer `corrcoef` numerical owner（CPU 2 passed）；`80e0f1b4` 形成 10.17 异步错误诊断契约和静态合同（1 passed），但低开销 ring/runtime 与 CUDA 实机仍待。待领保持 73。
>
> 第120波增量：`a0d3be31` 完成 `broadcast_shapes` numerical owner（CPU 2 passed）；10.17 仍缺 per-thread ring/stream 关联和 CUDA 异步注入，8.12 仍缺 cuDNN POD key 全套。待领保持 73。
>
> 第121波整卡审计：8.12 的 cuDNN 2D/3D forward/backward 六条 legacy cache 路径仍需共享 `LegacyConvAlgoKey`、per-device cache 与 CUDA 验收，当前无可安全半改；10.17 同样缺 runtime ring/stream 真实链路。本波无代码提交，待领保持 73。
>
> 第122波整卡审计：2.22/4.06/8.12/9.07 仍缺统一配置、fallback 三态、cuDNN POD key 或 import 环境隔离；7.05/7.07/10.18/10.20/10.21 仍缺事务锁、patch 边界、核心属性测试、内省 API 或 import lint；布局/多机/ACL 任务也有明确前置。本波无代码提交，待领保持 73。
>
> 第123波整卡审计：7.05 现有 namespace 事务只覆盖部分模块树，仍缺全局安装锁、失败可重试/可查询和 os.environ/flags 全量回滚；本波无代码提交，待领保持 73。
>
> 第124波整卡审计：7.05 复核确认上述缺口仍未改变，不能用现有 namespace snapshot 代替安装锁、失败状态 API 或环境/flags 回滚；本波无代码提交，待领保持 73。
>
> 第125波 7.05 前置：`3c8b46f3` 使 optional install 失败 warn-once、可查询、可重试；`44272e89` 新增 reversible `InstallTransaction`（RLock、逆序 rollback/commit/retry）及 module/env/flags/meta_path 合成测试 2 passed。尚未接入全部 installers，7.05 仍待领。
>
> 第126波 7.05 前置：`fa2027e4` 将 InstallTransaction 锁和 namespace undo 接入 `compat.torch.install`；`4f66701e` 明确 flags/env/import/meta_path/module-patcher 尚未纳入回滚，并补状态边界合同 1 passed。7.05 仍待领。
>
> 第127波 7.05 安全修正：有风险的整表 global snapshot 提交 `ec720cd8` 已由 `9a674001` 撤回；`1aa640cf` 补充 flags/env 写入清单与显式 allowlist/owner-aware restore 要求。7.05 仍待领。
>
> 第128波 7.05 子项：`9949fa10` 修复 completed-install namespace 冲突异常路径的 RLock 泄漏，新增回归测试；`test_install_context.py -k 'completed_install_conflict or optional_failure'` 2 passed。完整 installer mutation ledger 仍待接入，7.05 保持待领。
>
> 第129波 7.05 子项：`c8b993b4` 让 transaction rollback 做 owner-aware 值校验，外部改写时抛 `TransactionConflict`；事务合成测试 3 passed。仍需将所有 installer 写入口纳入 ledger，7.05 保持待领。
>
> 第130波 7.05 子项：`75793c04` 将 distributed installer 的 JT_NCCL_*、`use_nccl/use_mpi` 和 `jt.flags.use_cuda` 写入纳入 mutation ledger；`783699cd` 补 child-env 隔离合同 2 passed。仍有其他 installer 写入口未迁移，7.05 保持待领。
>
> 第131波 7.05 测试修正：`f8f838b2` 将 distributed 环境结构合同改为断言 `tx.mutate_env/mutate_flag`，避免旧 direct-write 断言误报；结构 2 passed、事务相关 5 passed。7.05 仍待领。
>
> 第132波 7.05 修正：`1eb7ec07` 让 `mutate_env` 先规范化字符串再记录 owner 值，修复整数环境变量回滚误报 `TransactionConflict`；事务定向 6 passed。7.05 仍需覆盖全部 installer 写入口。
>
> 第133波 7.05 测试修正：`c49efb80` 删除两个已被显式 activation API 取代的 `wrap_flags` 旧 patch，完整 `test_install_context.py` 从 2 个夹具错误恢复为 20 passed；事务/状态结构合同保持通过。7.05 仍待全量 mutation 接入。
>
> 第134波 7.05 子项：`60197b81` 将 factories/tensor installer 的 `jt.flags.use_cuda` 写入接入 transaction helper；`d9d063a5` 将 core installer 的安装期 `use_cuda` 写入接入 ledger 并补失败回滚测试。nn 的 `.to()` 写入确认属于运行时用户语义，不纳入安装事务；7.05 仍待全部 installer 写入口。
>
> 第135波 7.05 子项：`87ca0a82` 新增 owner-aware `mutate_attr`，将 transformers runtime guard 与 torchmetrics fastpath 的 `builtins.__import__` hook 接入 ledger；`sys.meta_path` permissive finder 因 allowlist/身份耦合暂不迁移。7.05 仍待领。
>
> 第135波 companion：`7af13605` 新增 utilities import-hook 回滚/外部替换冲突测试，定向 6 passed；不改变 7.05 整卡待领状态。
>
> 第136波 7.05 子项：`88795374` 让 permissive finder 新增和既有 allowlist 增量进入 transaction ledger，compiler installer 三处调用已接线；尚未覆盖 module_patcher finder 和并发外部替换，7.05 仍待领。
>
> 第137波测试修正：`9c6a7e92` 将 compile refusal 测试从过时的 `never checked` 文案更新为当前 `unchecked` 合同；transaction/permissive 定向共 27 passed。7.05 仍待完整 mutation 接入。
>
> 第138波 7.05 子项：`1a37b895` 将 module_patcher finder、registry、entry-point 状态接入 transaction undo，并由 integrations 传递 active transaction；外部 finder/allowlist owner 冲突测试仍待，7.05 保持待领。
>
> 第139波 7.05 安全修正：`ee1317c2` 让 permissive finder 新增/索引和既有 allowlist 增量回滚做 owner-aware 校验；外部 allowlist/finder 重排测试纳入事务合同，9 passed。module_patcher/其他 installer 全流程仍待，7.05 保持待领。
>
> 第140波 7.05 子项：`02b1733b` 将已加载 module 属性差异纳入 transaction ledger，`6e0f838a` 增加 module-patcher finder/registry 冲突保护；`test_compat_mechanisms.py` 20 passed，事务测试 9 passed。可变对象内部 mutation 和全部 installer 汇总仍待，7.05 保持待领。
>
> 第141波 7.05 子项：`9f154035` 将 vLLM `_ArmOnFirstImport` finder 插入和 active transaction 传递接入 ledger；external backend registry、vLLM callbacks/extension modules 和 shim runtime 全局写入仍未覆盖，7.05 保持待领。
>
> 第142波 7.05 子项：`367716a7` 将 external backend 的 `_BACKENDS`、`_BACKEND_HINTS`、`_ENTRY_POINTS_LOADED` 注册状态接入 transaction，并由 integrations 传递 active tx；source import 的 sys.path/sys.modules 仍需 owner-aware 或子进程隔离，7.05 保持待领。
>
> 第143波整卡边界：shim runtime activation 的 sys.path、sys.modules、flags、递归安装和扩展构建跨越现有 InstallTransaction 生命周期，不能半接入；需先建立独立 ActivationTransaction（path/module/flag owner token、冲突硬失败、child/retry 测试）。7.05 保持待领。
>
> 第144波 7.05 前置：`c204f4e9` 扩展 InstallTransaction 为 ActivationTransaction，新增 owner-aware `mutate_path`/`publish_module` 和 activation path/module 回滚合同；事务定向 11 passed。尚未接入 shim.runtime.activate，7.05 保持待领。
>
> 第145波 7.05 子项：`4d5b8e61` 将 shim runtime 的 `_composition=True` 分支接入 ActivationTransaction，包住 torch install 与 module publish；普通 activation 的 path/build/no_grad 仍未覆盖，7.05 保持待领。
>
> 第146波 7.05 子项：`e0885bf7` 将普通 shim activation 的父进程 sys.path 增量接入 `ActivationTransaction.mutate_path`，异常回滚、成功提交；child `PYTHONPATH` 保持独立。sys.modules/no_grad/build 和 child/path 专项测试仍待，7.05 保持待领。
>
> 第147波 7.05 修正：`065b71f9` 将 outer ActivationTransaction 显式传给 integrations，并在 inner torch install commit/rollback 后清理 state，修复 nested committed-tx 被误用的问题；`45a1283c` 补结构合同，安装上下文/事务定向 31 passed。7.05 仍待完整 activation 验收。
>
> 第148波回归：`45a1283c` 的 activation transaction handoff 合同独立验证 outer transaction 传递和 inner state 清理，避免后续 refactor 回退；不改变 7.05 待领状态。
>
> 第149波测试修正：`a08d5c15` 更新 activation bootstrap 的两个旧夹具以匹配 `publish_module`/`mutate_path` 新协议；完整 `test_torch_bootstrap.py` 现为 42 passed。7.05 仍待扩展构建和全量运行态验收。
>
> 第150波 7.05 回归：`e05064b9` 新增普通 activation 失败注入测试，验证 owner path/module 回滚、failed 状态和锁释放；完整 `test_torch_bootstrap.py` 43 passed。扩展构建副作用与 child/build 专项仍待，7.05 保持待领。
>
> 第151波轻量复核：9.20/9.22/9.23 已分别由 `1919b035`、`c4bbdd72`、`17e43c9a` 合入；原子产物聚焦 3 passed，孙进程 timeout 聚焦 1 passed。无新代码提交，未宣称未具备机器上的平台验证。
>
> 第153波审计：8.12 的六条 cuDNN legacy cache 仍需一次性 POD key/per-device 迁移；4.13 缺统一跨后端矩阵 owner 与 runner；5.24 未形成安全独立切片。本波无代码提交，三项保持待领，未新增进行中。
>
> 第154波审计：7.03 未找到可独立闭环的 tensor API；2.19 剩余 pyjt ASSERT 属内部不变量；8.06 三个 ACL 剩余面需协同迁移并依赖 910B3/CANN。本波无代码提交，未新增进行中。
>
> 第155波环境核验：`b13b4fe6` 记录 `nvidia-smi`/`probe.json` 确认开发机有 8 张 RTX 4090、CUDA 12.2.140、sm_89。历史“本机无 CUDA”只适用于曾设置 `nvcc_path=""` 的 CPU 进程，不能继续作为跳过真实 CUDA 负向验收的理由；未自动改任务完成状态。
>
> 第156波 CUDA 抽验：2.19 的 cuBLAS 3、cuSPARSE 10、CUB cumsum 1、curand 6、cuFFT 4 个负向/错误节点真实通过；cuTT 5 个因未启用 cutt 跳过。cuDNN RNN 两个 invalid 节点通过，但 4 个 dtype 正向节点进程 abort，不能计为通过，2.19 继续待领。
>
> 第157波 cuDNN 诊断：单独的 `test_float32_matches_reference` 仍 SIGABRT（退出 134），abort 发生在 `_cudnn_forward` 执行阶段；冷缓存还暴露过未预创建 TMPDIR 导致 nvcc 临时文件失败。根因未归因，2.19 继续待领。
>
> 第158波增量（8.06）：**看板此前十波「标准 ACL launcher owner 已穷尽」的记录是错的**——SWhere、SigmoidBackward、BatchNorm 前反向四个标准 owner 一直自己驱动 aclnn execute，且保留着审计列为「关键」的 `LOG_PRINT` 后 `return` 静默失败模式。本波迁完这四个，标准 owner 的 launcher 迁移**真正收口**，剩余两处 hand-rolled 尾部（reduce prod 两步归约、KVCacheMemcpy 无 workspace executor）是有意保留并已钉进合同。同时修掉 `test_acl_runner_failure_contract.py` 里那条 `checkRet == 65` 的计数断言——它在 8.06 第一个提交（`5be5fa15`）就被自己迁移作废，此后**红着穿过了约四十个提交没人发现**，现改为「除两处豁免外没有任何 family 自己发 aclnn execute 调用」的闭集不变量。新增 `agent/skills/acl-host-syntax-check`：无 CANN 主机上的桩 SDK + `g++ -fsyntax-only` 真实 TU 检查，全树 43 个源文件加 68 个 launcher ABI 断言全过，并做过两次反向对照。ACL 静态合同 82 项（81 passed + 1 failed）→ 88 passed。**本机无 CANN/NPU，仍待 910B3 实机**：机型/CANN/`npu-smi` 前置、精确 pytest nodeid 与禁止 CPU fallback 检查见 `docs/guides/ascend-910b.md` 新增小节。data-channel C++ 解码入口、胖 `AclOpFunctions` 类型擦除、属性 data 通道、描述符缓存仍未做。

> 第162波：f5461d6f 完成 7.03 quantile/nanquantile Var owner cohort，定向 7 passed；45d9ed15 完成 2.19 cuBLAS rank/inner-dimension 真实 CUDA 负向 3 passed，异常后计算继续通过；27bbca46 将 ACL 前缀删除逻辑改为显式 external-op registry，结构合同 2 passed。三项聚合任务仍保持待领。
>
> 第163波：8b44ca6d 补 cuBLAS batched/acc 非法输入后继续计算的真实 CUDA 回归，新增场景 1 passed、整文件 10 passed；8c5ef55b 补 ACL 属性 data-channel 原子迁移与 910B3 上机合同 6 passed；7.03 审计确认简单 numerical owner 已基本收完，vmap 与 tensor reduction 不能伪拆。本波未关闭聚合任务。
>
> 第164波审计：cuDNN conv3d rank 边界已有完整覆盖，独立单卡真实运行 1 passed；8.06 ACL data-channel/type-erasure/descriptor cache 仍需协同迁移，相关静态合同 9 passed；0.22 仍受全量 CUDA 时长阻塞；9.01 只读 HOME 合同额外复现 PermissionError（7 passed/1 failed）。本波无代码提交。
>
> 第165波：`01090519` 修复默认只读 HOME 的缓存 fallback，相关测试 6 passed；`0100a475` 对齐 torch 显式策略下 detach 的 requires_grad，定向 2 passed、策略/对拍 10 passed。7.12/9.01 整卡仍有独立包、冷启动等剩余验收，未误标整卡完成。
>
> 第166波审计：7.12 独立 torch 包仍需 torch 身份、TorchTensorState、`_torch_leaf_params` 全链路和多个前置联动；9.01 冷启动仍在 import 阶段编译核心（约 37.6s），无法窄改；8.15–8.18 缺多机 launcher，现有 Store/launch 聚焦 5 passed/1 failed。本波无代码提交。
>
> 第167波复核：远端无新增代码；重新确认 7.12 的 process-global leaf registry、9.01 的冷启动编译、8.15–8.18 的多机参数缺口仍未改变。本波无代码提交，未新增进行中。
>
> 第168波：`d141d8c2` 完成 9.07 独立切片，移除 import 阶段对 `os.environ["cc_path"]` 的反向写入，定向回归 1 passed；4.06 backend_fallback 三态和 10.21 import lint 仍需 BackendRegistry/分层设计，本波未伪造完成。
>
> 第169波：`f63856f8` 修复 smoke fail-open；`41dfb254` 修复 NPU gate fail-open；`515ebf71` 修复 build stamp 漏记生成器依赖；`e9b966e7` 修复 FSDP 重复梯度不累加；`b468afcf` 和 `5e2667f6` 完成两个 5.24 facade 切片；`c02b1481` 让 NCCL 全 skip 时 gate fail-closed。gate/hardware 结构合同 17 passed。 |
>
> 第170波：`6c86fb20` 将 CuPy 从 import jittor 热路径懒加载，导入/CPU回归 2 passed；`7eedcbd8` 统一 SDPA flash stats diagnostics facade，定向 2 passed；`b6de9642` 增加 3.18 单 kernel 冷编译分段 profiling skill，clang 全流程实测通过。聚合任务仍待完整验收。
>
> 第171波：`83c26d42` 完成 2.24 FusedOp 显式 `op_index/var_index` 映射，移除 `Node::custom_data` 最后用户；结构合同 4 passed，fused 聚焦 2 passed。2.24 正式关闭。
>
> 第172波：`5248870d` 删除不可达 EventQueue `run_sync`/Worker dead设施，清理 executor 注释并将 NCCL 两处调用改为直接 CUDA 检查；结构合同 2 passed，event_queue C++ syntax check 通过。3.19 正式关闭。
>
> 第173波审计：2.13 的 flags/Runtime 状态跨约30个 C++ 文件和309个 Python消费者；8.12 六条 cuDNN legacy cache 必须整体迁移；9.01 冷启动核心编译仍在 import 路径；4.06 缺 BackendRegistry/OpRegistry。均无安全窄切片，本波无代码提交。
>
> 第174波复核：近期修复没有解锁新的安全单任务；5.24、9.01、10.20、10.21、8.12、11.01 仍需整卡迁移，2.19 仍有 cuTT/多卡验收缺口。本波无代码提交。
>
> 第175波分布式复核：8.15 的 TCPStore 失败在当前环境卡于 import/编译锁，绕过 Jittor import 的双 rank 直接脚本多次 `[0,0]`，未复现 ConnectionReset；没有足够证据改生命周期代码，8.15 继续待领。
>
> 第176波 8.12 整卡审计：六路 cuDNN legacy cache 仍分别使用字符串 JK key、共享 process-global map，缺 dtype/layout/strides/workspace/device 维度；完整迁移需要共享 POD header、六处 EXTERN_LIB ABI、per-device 生命周期和真实 CUDA 回归。本波无代码提交。
>
> 第177波架构前置：`fcce48e3` 建立 Python BackendRegistry/OpRegistry 最小合同（6 passed），但尚未接入现有 C++/flags 路由，4.03/4.04 整卡继续待领；2.13/7.12/8.12/9.01 仍无安全窄切片。本波无整卡关闭。
>
> 用户范围调整：ACL/HCCL/NPU 与多机实机任务已按用户授权标为「并入 硬件验收」，不计作当前代码待领；8.06/10.19 等混合代码任务仍保留待领。
>
> 第159波增量（7.03，六个 cohort）：`16333333` amax/amin/count_nonzero 收回原生 owner；`9cba7d68` cumsum/cumprod；`50876abf` sort/argsort/topk/median；`d94c5cbd` sign/trunc/frac/exp2/log10 归一到单一 owner；`a7dcae1c` nan_to_num/logaddexp；`d1535282` outer/tensordot/repeat_interleave 改为再导出原生 owner。**这一波补上了 7.03 一直缺的 CUDA 那一层**：此前约三十个 cohort 的证据全是「CPU N passed」，本波两个 cohort 用 `instantiate_device_type_tests` 在 CPU 与 CUDA 各跑一遍（各 15 passed / 13 passed），并跑出两处真实差异——4096 元素 float32 `cumsum` 两侧相对差 1.4e-06（并行前缀和比顺序扫描更准），512 元素重复键 `argsort` 的 indices 两侧不同而 values 逐位相同；两者都判为后端固有并登记进 fidelity，测试改为钉有界不一致与整数路径逐位相等。**同时修掉两处「一个 API 两个对象」**：`torch.sign(int32)` 返回 float32 而 `Tensor.sign()` 返回 int32（真 PyTorch 2.12 两者都是 int32，属静默错 dtype，修前失败/修后通过的用例已随提交落地）；`repeat_interleave` 的转发 wrapper 让`torch.repeat_interleave is jittor.repeat_interleave` 不成立，两条结构门禁因此长期红，现已转绿（`tests/structure` 由本波开始时的 15 failed 降到 4 failed，其中 2 条是本波修的、其余为别的分区）。新增 skill `agent/skills/torch-api-cohort-promotion/`。7.03 仍按剩余范围保持「待领」。
>
> 第178波增量（7.03，改成「清空整个 installer」而不是「再挑几个 cohort」）：`796b8e43c` 清空 `_install_reductions`（内嵌 def/class 14→0、lambda 13→0），`d5740ee7d` 清空 `_install_module_methods`（40→0、lambda 6→0），两个 installer 现在只做绑定并各自单独记为已闭合。**这是 7.03 经过约 66 波之后第一次有「某个 installer 已经彻底空了」这件可验证的事实**，此前每波都是又一批 cohort。计数口径与审计同为「内嵌 def/class」，随手可复跑 `python agent/skills/torch-api-cohort-promotion/count_installer_closures.py`（`CLEARED` 标记＝nested 与 lambda 同时为 0），并各加一条 AST 测试防止下一波往空 installer 里塞新闭包。**归约/排序面都过了 CUDA 那一层**：`_install_module_methods` 没有归约，但 `to`/`cuda`/`cpu` 就是 residency 迁移面，用 `instantiate_device_type_tests` 在 CPU 与 CUDA 各跑一遍，真实 CUDA 105 passed。**修掉一处静默错**：`zero_grad(set_to_none=False)` 一律置 None，真 PyTorch 2.12.1 留的是同 shape 同 dtype 全零张量——因为裁剪与累积都写成 `if p.grad is not None`，参数被静静跳过（修前 1 failed / 修后 7 passed）；同提交把 bridged optimizer 的 `zero_grad()` 移到参数循环之前，否则它会把刚写好的零张量又抹掉。**另查明一处跨域差异，按「先核最终 owner、不跨域抢改」记给 owner**：只要进程里还活着任何一个 optimizer 对象，`backward()` 就把梯度路由进它、`p.grad` 保持 None（开关是 `jt._active_optimizers`，清 `_current_optimizer` 无效），真 torch 无论有无 optimizer 都填 `.grad`；后果是任何「先 backward 再读 `.grad`」的用例都会因为**另一个文件**建过 optimizer 而红、且红的位置与原因无关。三套门禁均**逐条同集合**、零回归：`tests/structure` 15 failed / 887 passed（HEAD 基线同为 15 failed，交接文档记的 14 已漂移）；CPU torch 模式 `tests/compat/torch` 33 failed / 1390 passed vs HEAD 基线 33 failed / 1295 passed（失败集合 diff 为空，+95 全来自新增文件）；原生 CPU `tests/nn` 3 failed / 204 passed 与基线逐条相同、原生 `tests/core` 21 failed / 647 passed。skill 补 §7（清空整个 installer 的口径与两类搬迁手法）与 §8（backward 测试要先隔离进程级 grad 路由，且 `instantiate_device_type_tests` 生成的是 unittest 类、注入不进 pytest fixture）。7.03 整条按剩余面保持「待领」。
>
> 第160波增量（9.01 import 耗时）：`cf3835ee` 把热缓存 `import jittor` 归因到具体一步（核心编译在无事可做时占 CPU-only 配置的 68%），`51d0439f` 把核心编译收进 `compiler.build_core()` 并加构建戳。热缓存 import CPU-only 1.332→0.413 s **达标**、CUDA 2.457→1.545 s **未达标**；冷缓存与换配置仍全量编译核心，9.01 保持待领。另核实三条**既有**阻塞（基线 `534d375d`，均非 9.01 引入、改前改后位置一致）：`tests/core/test_device_methods.py` 与 `tests/backends/cuda/test_device_methods.py` 同名，pytest 收集期报 import file mismatch 并整体中止 native 门禁，需 `--continue-on-collection-errors` 才跑得完；native 门禁在 `test_complex64_linalg::test_svdvals` 进程 abort；torch 门禁在 `test_torch_compat_autograd::test_a_second_call_does_not_steal_the_first_calls_context` 进程 abort。

一行一个任务，与 [refactor-plan.md](refactor-plan.md) 的编号对应。领任务把状态改成「进行中」并写名字，
完成改成「已合并」并填提交号；推送冲突说明别人先领了。状态只有四种：待领 / 进行中 / 已合并 / 并入 X。

> **2026-09-03**：上一轮中断留下的五个 `wip/*` 与工作树残留已经验证、合入或明确退回待领；任务表已无
> 「进行中」或「部分完成」。下一波从 [交接说明](refactor-handoff.md) 第 6 节开始。

## 起点已知失败清单（归责之前先减掉这些）

**任何失败在算成回归之前，先确认它在分支起点是否也失败。** 分支起点是
`9eb696d9`（`origin/2.0`，即 `merge-base origin/2.0 origin/2.0-refactor`）。
没有这份清单，会把继承来的失败当成新回归、把责任安到无辜的提交上。

### A. 分支起点就存在的失败——不是任何 agent 引入的

| 用例 | 症状 |
| --- | --- |
| `tests/compat/torch/test_torch_compat.py` | `RandomOp` 子进程段错误 |
| `tests/data/test_dataset.py::TestDatasetSeed::test_children_died` | **worker 被杀之后 Dataset 不快速退出。** 子脚本 `dataset.workers[0].p.kill()` 之后，父进程应当收到 SIGCHLD 并 quick exit（用例断言 stderr 里有 `SIGCHLD` 与 `quick exit`），实测父进程一直阻塞到子进程超时。单独跑、空闲机器上稳定复现，起点 `9eb696d9` 同样失败（前任两步回退确认）。现已 `xfail(strict=True)`，仍然每轮跑、仍然可见，但不再让门禁整体变红；修好的那天 strict 会把它变红提示删标记。**2026-09-03 补充（bindings，做 507a0f1f 时查清）**：这一条与「worker 抛异常后父进程挂死」是**同一个洞的两半**。抛异常那半已修（worker 写好错误后 `buffer.stop()` 再推 id，父进程两处阻塞都能被叫醒）。这一条是 `p.kill()`：SIGKILL 的 worker 跑不到任何 Python 收尾代码，不写共享槽、不 stop、不推队列，而父进程阻塞在 `RingBuffer::pop` 里**握着 GIL**（`py_ring_buffer.cc` 全文没有 `Py_BEGIN_ALLOW_THREADS`），所以 Python 看门狗线程也跑不了。真正的修法是让这两个阻塞调用释放 GIL 并带超时（核心 C++ 改动，需单独立项）；**不要**退回旧的「SIGCHLD → 父进程 quick exit」，那是 6.C31 拆掉的无声消失 |
| `tests/core/test_array.py::TestArray::test_memcopy_overlap` | **墙钟阈值型 flake，非回归。** 断言是 `t2-t1 < 0.010`——「重叠版比纯计算版慢不超过 10 毫秒」，一条**绝对**墙钟阈值。机器常驻十几个 agent、负载 24 时它必然超。两个分区各自独立确认：内存分区在**未打补丁的树**上跑两次失败，绑定分区独立得出同一结论。**归责方向和真回归相反**：真回归查代码，这条查负载 |
| `tests/compiler/test_atomic_tuner.py::TestAtomicTunerClass::test_atomic_tuner` | 第 4 项 `x.sum()+x.sqr().mean()` 期望两条 `atomictuner: move atomicAdd to loop -1`，实得 0 条。根因是 `032ecfe1`（2026-08-28，起点前 202 个提交）把 CUDA 全量归约改走 `nn/backends/full_reduce_cuda.py` 的 cub 两级折叠 code op，整条全归约不再进融合算子 JIT，AtomicTunerPass 根本看不到 atomic 语句。前三项 add/max/min（reindex_reduce）在起点与起点父提交上都通过 |

（这份表正在用一棵钉在 `9eb696d9` 的只读 worktree 实测补全，跑完会把失败 nodeid 逐条列全。）

**`test_atomic_tuner` 已定论，`9eb696d9` 洗清。** 两棵只读 worktree、两份独立 `JITTOR_HOME`、同一条用例、**串行**跑（并行会串号，见 skill）：`9eb696d9^`(`a88ae02a`) 与 `9eb696d9` 的失败**逐字一致**——同为第 69 行第 4 项 `AssertionError: (0, 2)`。WarpReducePass 挂在 `pass_manager.cc` 的 `AtomicTunerPass` **之后**，原子调优早已打完日志才轮到它改写，它不可能吃掉这些日志。真正的原因见上表那一行：`032ecfe1` 的全归约快路径绕开了整个 JIT。**该用例现在断言的是一条已经不存在的代码路径，属于过期断言，不是回归。**

### B. 已归责，按表中状态处理——不要重复归因

| 用例 | 引入提交 | 责任 |
| --- | --- | --- |
| `tests/compat/torch/test_torch_compat_interpolate.py::TestInterpolateBicubic::test_bicubic_constant_stays_constant` | `13ac1d14` [6.C05] | coreops，正改成变长编码（原属 3.02） |
| `tests/compat/torch/test_torch_compat_autograd.py::TestCustomFunctionCompatibility::test_torch_style_function_keeps_context_and_broadcast_grad`、`test_torch_compat_autograd_semantics.py::TestSavedTensorVersions` 两条 | `5c4e624b` [5.07] | autograd。**已实测确认**：把工作区还原到 `origin/2.0-refactor` 干净树（`git diff > .patch` + `git checkout --`，不用 stash）后三条同样失败，与其他分区改动无关。一次性上下文使 `execute` 里写在 `self` 上的属性（`self.seen_needs_input_grad`）不再留在用户持有的实例上 |
| `tests/structure/test_pytest_contract.py::test_test_modules_avoid_collection_time_backend_side_effects` | `93b48a8e` [4.02 3/3] | 4.02。`tests/backends/cuda/test_device_copy.py:44`、`tests/backends/cuda/test_multi_device.py:34,35`、`tests/compat/torch/test_multi_device.py:37,38` 在**模块体**里调本地 helper `_device_count()`，收集期就去问设备数。改法：挪进 fixture，或用可调用形式的 `skipIf` |

（`test_runtime_composition_structure.py::test_moved_scope_state_stays_synchronized_with_the_root`
已由 `90fc2f0b` 修复并移出本表：2026-09-03 复测 `JITTOR_TORCH_SHIM=1` 下该文件 13 passed。）
| 并发编译读到写了一半的 `.s`：原生 `tests/data/test_dataset.py::TestDataset2::test_dataset_use_jittor`（`.so` 里 `undefined symbol: SetitemOp::jit_run`）+ CPU torch 536 条散布失败（`op.s` 报 `unknown pseudo-op .lasf10` / `invalid operands (*UND* and *UND*) for -` / `junk at end of line`） | `70d97137..a12d81c0` 之间的构建改动 | **已由 `1919b035` 修复并复验**：原子替换专用回归与原四 worker Dataset 用例均通过 |
| `tests/structure/test_nn_structure.py::TestModuleBoundaries::test_first_import_paths_are_cycle_free_in_fresh_processes` | `46dbe946` [0.21] | gates，**已修**：收编子进程调用时把调用点原来的 `env.pop` 丢了。旧写法自己 pop 掉四个 `JITTOR_TORCH_*` / `REAL_TORCH_SITE`；改走 helper 后 `env=` 是**叠加**在 `os.environ` 上的，叠加无法删除，于是四个变量原样回来了——而 `tests/structure` 自己就在 `TORCH_MODE_PATHS` 里，父进程带着 `JITTOR_TORCH_SHIM=1`。子进程先塞了个假 `torch` 模块再 import jittor，于是报 `cannot install Jittor Torch compatibility over an existing Torch module graph`。修法不止是加 `inherit=False`：helper 现在**拒绝**「完整环境 + inherit=True」这种有歧义的调用（认 `PATH` 在不在里面），并提供 `without_torch_mode=True` 把这四个变量的清理写进 helper，不再让每个调用方自己记得 |
| `tests/structure/test_stage2_delivery.py::TestStage2Delivery::test_nox_keeps_fast_structure_and_packaging_separate` | `6adbf488` [0.04] | gates，**已修**：断言 noxfile **源码文本**里出现 `tests/optim/test_optimizer.py` 等具体路径，而清单已搬进 `gate_scope.py`。这两条改成从 `gate_scope` 求门禁选择集再判断；仍是清单的那六条 oracle 路径继续按文本断言（"要跟真 PyTorch 对拍"是测试的属性，不是树的属性） |

### C. 先看这条：把 `tests/compat/torch` 和原生目录写在同一条 pytest 命令里，会让原生用例整片变红

这是**跑法造成的假失败**，不是回归，也不在上面 A 表里。`tests/conftest.py`
的 `_select_torch_mode_for_test_process()` 按 `sys.argv` 选模式：

- 选择「宽」（只写 `tests` 或 `.`）→ 原生模式；
- 选择「窄」且其中**任何一个**路径命中 `TORCH_MODE_PATHS`（含 `tests/compat/torch`）
  → 给**整个进程**设上 `JITTOR_TORCH_SHIM=1`。

所以

```bash
pytest tests/core tests/nn tests/optim tests/structure tests/distributed tests/compat/torch
```

会把 `tests/core`、`tests/nn`、`tests/optim` 全部拖进 torch shim 模式跑。conftest 自己的注释
写得很清楚：torch 模式是进程全局的，会改掉惰性求值、归约默认值和梯度语义，
**「switching the whole tree into it made ordinary native tests fail」**。

**正确跑法是拆成两条命令**，和门禁的划分一致：

```bash
JITTOR_TORCH_SHIM=0 pytest tests/core tests/nn tests/optim tests/distributed   # 原生
JITTOR_TORCH_SHIM=1 pytest tests/structure tests/compat/torch                  # shim
```

拿着一把 F 又不确定来源时，**先按这条拆开重跑**，再来对 A 表。

## 门禁 agent 的最新结果

三套门禁的判据不是 `failed == 0`，而是：`failed` 不超过上面 A 表，且 `passed` **不下降**。
整改期间每个 agent 都在加测试，所以 passed 一路上涨是正常的，起点那行的数是下界不是等号。

| 提交 | 原生 | CPU torch | CUDA | 失败用例 / 责任任务 |
| --- | --- | --- | --- | --- |
| 9eb696d9（分支起点） | 775 passed / 765 skipped | 1595 passed / 285 skipped | 574 passed / 9 skipped / 0 failed | — |
| `70d97137` | **822 passed / 816 skipped / 0 failed**（50 分） | 未完成（运行中断） | 未开始 | 原生绿；收集总数 1540 → 1638，是有人加了测试 |
| `a12d81c0` | 1039 passed / 866 skipped / **1 failed**（51 分） | 1207 passed / 536 skipped / **536 failed**（59 分） | 未跑 | 两套的失败**同一个根因**：并发编译读到写了一半的 `.s`（详见下方一行）。`passed` 两套都在涨，判据里下降的那一半没触发 |

## 热点文件占有（历史）

第一波（2026-09-02 21:xx 派出）：九个 agent 并行，各自独占一张卡/一段核/一个 worktree
（`/home/zy/jittor-lab/refactor/<分区>`，分支 `wk/<分区>`，推送到 `2.0-refactor`）。

| 分区 | 第一波占有者（历史） | 任务 |
| --- | --- | --- |
| 核心节点 | coreops (2.01/2.02/2.03) | GPU3 c24-35 |
| 执行器 | — |  |
| 代码生成 | — |  |
| 类型与日志 | coreops (6.C01/05/06/07/09/30) | GPU3 c24-35 |
| 内存 | — |  |
| 绑定 | bindings (6.C02/22/23/24/25/27/28/29) | GPU0 c0-11 |
| Python 核心 | — |  |
| Python 算子 | — |  |
| Python 其他 | pyother (5.18、5.19、6.P25) | GPU6 c64-75 |
| 兼容层 | compat (7.01) | CPU c96-103 |
| CUDA 后端 | cudabk (6.B07/08/17、8.01、8.03) | GPU7 c76-87 |
| ACL/ROCm/Corex | — |  |
| 分布式 | dist (6.B01/03/04/06/10/11/15) | CPU c88-95 |
| 构建 | build (0.07–0.11/0.17, 9.02–9.06/9.08/9.09/9.11/9.15/9.17, 9.04 部分) | GPU4 c48-63 |
| 门禁 | gates (0.01–0.04/06/12/13/18/19) | CPU c104-111 |

## 执行中出现的、需要认领的杂项

| 事项 | 现象 | 建议归属 |
| --- | --- | --- |
| 全树只有两条「性能断言当正确性门禁」 | 全树扫了一遍时间断言（`time.time()` / `perf_counter` 的 16 个文件逐个看过）。**上界型**（负载会让它变红）只有两条：`tests/core/test_array.py:81` 的 `t2-t1 < 0.010`（绝对毫秒阈值，就是 A 表那条）、`tests/core/test_nano_string.py:28` 的 `nano_time < builtin_time * 1.25`（**相对**比值 + 5 次取最小，抗噪好得多，但仍是上界）。其余全是**下界型**——`test_nccl_rendezvous_timeout.py:188/249` 的 `assertGreater(elapsed, TIMEOUT)`、`test_torch_compat_unimplemented.py:635` 的 `assertGreater(elapsed, 5.0)`、`test_tracer.py:69` 的 `elapsed < 30`（2 秒超时的 15 倍余量，判的是"有界 vs 无界"）——负载只会让下界更成立，不会误伤。**结论：这类假红的来源只有一条，不是一类。** 修法应是把绝对阈值改成相对比值或改成有界性断言，不是放宽数字 | 门禁 gates，随 0.15 |
| wheel 内容基线过期 | **已处理**：基线本身没错，错的是那条断言写死了条目数——任何人加一个模块都会改变它。用当前源码树真构建一个 wheel 核对，45 个新增全是各分区加的合法源码、1 个删除是 9.17 删的 `flags.cc`，基线整个重新生成（861 条）；条目数那条断言换成「基线头部的 `# entries:` 必须与自身条目数一致」这条规则。注意基线是对**当时的源码内容**取的哈希，发版前需要再刷一次（办法写在提交说明里） | 构建，`f869cab8` |
| 结构测试子进程超时 flaky | 已处理，并已与 0.21 合并成一份实现：超时预算搬进 `tests/_helpers/child_process.DEFAULT_TIMEOUT`（600s，`JITTOR_TEST_CHILD_TIMEOUT` 与旧名 `JITTOR_TEST_SUBPROCESS_TIMEOUT` 都认，仍在门禁 `--timeout=900` 之内），`process_modes.SUBPROCESS_TIMEOUT` 随之删除——同一件事只留一处 | 门禁 gates，`46dbe946` |
| `(void)x;` 的识别曾按「语句含 void 一词」 | 已修（3.09，`66e5a153`）。原判据会把 `memset((void*)p,0,n);` 整条删掉：编译通过、缓冲区没清零、静默算错。今天没被咬到只是因为树里没有算子往融合 kernel 里写带 void 转型的语句 | — |
| `UnrollPass` 与 `ExpandEmptyBlockPass` 同名 | 已修（3.14，`6c899325`）。`exclude_pass` 与 `pass_map` 都按名字索引，`emplace` 不覆盖，后跑的那个根本没进表；`get_pass` 还会把它 C 风格强转成另一个类型。g++ 构建上 UnrollPass 根本不跑，所以运行期校验抓不到，用例改成实例化 30 个 pass 比名字 | — |
| 3.13 前置核实：`range10` 今天真的会出现，且 `loop_id` 在合并后已经与循环变量对不上 | 实测（CPU，8 维加 3 次 split、9 维加 2 次 split 起）生成源码里确实出现 `range10`；我试过的 20 组（元素级/归约/广播 × 7–10 维 × 0–4 次 split）**数值都还是对的**，所以这条今天是潜伏的，不是已经在算错。两条具体机制：(1) `merge_loop_var_pass.cc` 用 `size()==6` 判断「是单个 range」、用逐字符拆分把 `range23` 展开成 `range2*range3`，对基础的 `range10` 会拆成 `range1*range0`；更危险的是合并出的新 id 是 `aid+bid` 字符串拼接，合并 1 和 0 得到的 `range10` 与基础 `range10` **同名**，而代码里 `if (!find_define(new_range)) push_back(定义)` 恰好会跳过定义、直接复用那个基础 range —— 循环上界完全错误且能编译。(2) 合并后 `loop_id` 变成拼接串，但循环的归纳变量仍是 `id{bid}`，而 `restride_pass.cc:50` 用 `"id"+fa->attrs["loop_id"]` 去找它 —— 合并后这两者已经对不上了。所以 3.13 不是改一个函数，确实需要计划里说的「loop id 用整数向量、名字只在输出时生成」 | 代码生成分区，3.13 |
| 3.10 前置核实：改名的默认动作是「白名单之外一律加 `op{i}_` 前缀」 | `op_compiler.cc` 的 `unchanged` 只有 24 项（`for/const/auto/int/float/bool/void/if/true/false/...`），`return`、`else`、`while`、`static`、`unsigned`、`char`、`long`、`double`、`size_t`、`int64`、`uint`、`nullptr` 全都不在里面，会变成 `op0_return` 这种。因为这些写法今天写了就编不过，**现有算子里一个都没有**，所以把它们加进「不改名」集合对现存代码是行为不变的（只会让新写法能用）。建议的形状：完整的 C++ 关键字集合 + 运行期已知类型（`op_type->types` 已有）作为不改名集合，再加一条改名后的合法性校验 —— 两个相邻的被改名标识符（`op0_A op0_B`）几乎只能是「把类型名改了名」，直接报一条指名道姓的错误，而不是把 `op0_size_t 不是类型` 丢给 C++ 编译器 | 代码生成分区，3.10 |
| 3.13 结论：合并 range 的命名冲突今天**不可达**，两条别处的巧合各兜着一半 | 3.13 已合并（`bd5b5a67`）。前置核实那条说的命名冲突（合并 1 和 0 得到 `range10`，撞上基础/split 的 `range10`）**构造不出来**：我用 10 维 + `split9` + `order1=1`/`order{i}=2` 把循环 1、0 排到最内层，合并 id 恰为 "10"、`range10` 也确实存在——但 reorder 之后 1 与 0 在**内存序**上不再相邻，`expr::match` 直接失配，合并根本不发生。也就是说合并 id 永远是按嵌套序递增拼接，而嵌套序就是内存序。另外两条兜底：`NanoVector::push_back_check_overflow` 断言 `s<10`（张量最多 10 维，基础 range 只到 `range9`），以及 **split 出来的循环永远进不了合并**（父循环 `inner` 里多一条 range 定义，过不了 `inner.size()==3`），所以所有合并 id 的每一段都是一位数、逐字符拆分恰好是对的。这三条都是别处代码的巧合，不是这里声明的前提——已改成 `parse_loop_id`/`format_loop_id`，段间用 `_` 隔开 | 代码生成分区，3.13 |
| 3.15 结论：pass pipeline 跑两遍是真的，但只占首次执行的 0.04%–0.22%，不值得改 | 3.15 已合并（`97cac22f`）。加 `LOGvvv` 计时后实测：元素级 CPU parse 72us / passes 627us / to_string 38us，而首次执行 1012ms；matmul CPU 两遍合计 4.6ms，首次执行 2080ms；CUDA 元素级 pipeline 0.95ms、首次执行 3871ms。**其余全是 g++/nvcc**。计划提的「一次解析后 clone IR」只省得到第二次的 parse（~170us），而 clone 与 parse 是同一量级、没有 tuner 自信时那一份 clone 纯属白做——最好情况省 0.008%，代价是给 KernelIR 引入一条 father/scope 指针容易出错的复原路径。**真正做了的两件**：ReorderTuner 的候选从 N!（10 维 + 3 次 split 实测 order0..order12、乘积 **3,628,800,000**）改成按 `jit_search_max_candidates`（默认 1024）截断；`Searcher::timeout` 从「声明了没人读」改成由 `jit_search_timeout` 设置并真正生效 | 代码生成分区，3.15 |
| 3.11 结论：relay kernel 里的字节偏移，主路径上被 `cache_compile` 的内容哈希兜着，但 `rewrite_op=0` 时真可达 | 3.11 已合并（`f32d3c83`）。`get_relay_src` 生成 `GET_VAR_MEMBER(rop_0_0, 120) = vars[2].var;`，120/128/136 是算子结构体的字节偏移，写进 JIT 缓存复用而 jit key 里没有布局信息。主路径上 `cache_compile` 把生成源码的内容哈希算进缓存键，偏移一变就重编；但 `rewrite_op` 是个 flag，设成 0 时已存在的 `.cc` 不重写，旧偏移与自己的哈希一致、不重编——这条路真可达。已改成按名字 `set_var_member("a", ...)`。另外审计说「偏移由 compiler.py 用正则扫头文件得出」**不准确**：正则只扫**名字**，偏移是 `offsetof` 在 C++ 编译期算的；但那个正则确实有一种哑失败（`jittor::Var* x;` / `const Var* x;` 扫不到，成员不进注册表、relay 时永不绑定），已抽成 `compiler.parse_var_members` 并让读不懂的写法**构建失败** | 代码生成分区，3.11 |
| 3.10 结论：扩充「不改名」集合对现存代码逐字节不变 | 3.10 已合并（`864fa52c`）。58 组生成源码（元素级 1..8 维 / 五种 dtype / 一元链 / 六种归约 / broadcast / ternary / index / transpose 融合 / 六组 loop option，CPU 与 CUDA 各一遍）改前改后逐字节相同。新增的合法性校验（两个相邻的被改名标识符只可能是「类型名被改名了」）要放过算子自己 `#define` 出来的宏——`index_t`、`Tx`、`T` 都是宏，`op0_index_t op0_i` 是**正常**生成结果 | 代码生成分区，3.10 |
| 冷缓存下跑整套 `tests/compiler` CUDA：会多出几条只在这种条件下红的用例，而且跑到 `test_probe_cache` 附近**无声退出** | 2026-09-03 实测：`rm -rf $JITTOR_HOME/.cache` 之后跑 `tests/compiler` CUDA 全套，`test_cache_path_precedence::test_the_imported_core_is_the_cuda_one`、`test_console`、`test_custom_op` 两条、`test_fused_identifier_rename::test_reserved_identifiers_are_not_renamed` 共 5 条红；把这 5 条**单独**拿出来在同一棵树、同一张卡上跑是 **7 passed**。同一棵树的热缓存轮此前也只有一条失败（且是用例自身问题，已修）。之后进程在第 302 条（`test_probe_cache`）附近**没有 summary 就退出**（`EXIT=1`，pytest 零输出）——这正是简报 §7 与任务 6.C31 记的那个模式：子进程被信号杀死，jittor 的进程级 SIGCHLD 处理器让父进程无声退出。**判据**：整套红了一片而单独跑全绿、且日志没有 summary，先看是不是冷缓存 + 子进程信号，不要当成代码回归 | 代码生成分区，验证方法 |
| `split{i}` 与 `parallel` 不兼容 | 同时设这两个 loop option，`ParallelPass` 在 `ASSERT(def)` 上失败（`Check failed: def`）。`SplitLoopPass` 给内层循环的 range 是 `::min(range{i}-id{i}, stride{i})`，定义在外层循环里且随它变化，`ParallelPass` 在调用点 `func->find_define` 找不到、也无法在调用点求值。CUDA 恒走 `ParallelPass`，所以 CUDA 上任何 split 候选都必然编译失败。用例已钉住：`tests/compiler/test_reduce_tuner.py::test_a_split_candidate_would_not_compile_under_parallel` | 代码生成分区，1.04 的前置 |
| CUDA 归约需要的是线程分解候选，不是 CPU 那套 | `orderN` 候选实测五种形状全部不优于默认（最差 2.1 倍，破坏访存合并），`split{i}` 被上一条挡着，L1 分块尺寸对 GPU 无意义。真正有用的候选是 `ParallelPass` 里的线程分解，属于新工作 | 代码生成分区，待 1.04 前置解决后 |
| shim 的 `Tensor.backward()` 在真实规模 UNet 的 CUDA 图上直接 abort | `node.h Check failed: value_ > 0  backward liveness release without a matching owner`，进程 `terminate`（退出码 134）。复现就是速度门禁自己的入口：`tests/compat/torch/_ecosystem_runner.py large_diffusers_unet2d out.npz --runtime jittor --device cuda`，必现，与 `zero_grad` 无关，崩在 `loss.backward()` 里而不是之后的 sync。同一张图上 `jt.grad(loss, params)` 提交同一批反向算子**正常跑完**（270 个梯度、loss 与校验和稳定），所以问题在 shim 的 backward 路径而不是反向图本身。后果：`test_ecosystem_speed` 的 CUDA UNet 一项现在根本跑不到时间就崩了 | 兼容层分区（7.x），新任务待派 |
| shim 对「float32 张量 ÷ Python float」故意加宽到 float64，在 sm_89 上代价是每步 0.55 ms | `python/jittor/compat/torch/installers/tensor.py` 的 `_make_truediv`：`use_wide = sd.startswith("float") and ...`，注释写的理由是「为 1-ulp 对齐 PyTorch」。sm_89 的 FP64 吞吐是 FP32 的 1/64，于是 diffusers `ResnetBlock2D` 的 `(input + hidden) / self.output_scale_factor`（默认就是 `1.0`）在整张特征图上跑双精度除法：生成的 kernel 里是 `cast f32→f64`、`f64 / f64`、`cast f64→f32`，实测只有 593 GB/s 与 397 GB/s，而同一步里别的逐元素 kernel 是 1100–1900 GB/s。把 `use_wide` 临时改成 `False` 实测：逐元素类 **3.29 → 2.73 ms（−16.8%）**、整步 22.03 → 21.29 ms，且越过 3.23 的验收线。**但 PyTorch 自己不这么算**：`div_true_kernel_cuda` 对 CPU 标量走 `a * (1/b)` 且 `opmath_type<float>` 就是 float，所以「加宽才对齐」这个理由值得复核。改动由兼容层分区决定——`tests/compat/torch/test_torch_compat_promotion.py` 有 bit-exact 的 `view(np.uint32)` 断言钉着它 | 兼容层分区，新任务待派；3.23 的验收依赖它 |
| ~~归约类今天已经比 PyTorch 快一倍以上，3.22 的验收口径需要复核~~ **口径已对齐，结论反转** | 原记录是「Jittor `reduce` 0.57 ms 对 PyTorch `reduce/norm` 1.20 ms，看起来已达到」。2026-09-06 把两个桶拆开：**它们几乎没有交集**。Jittor 的 0.57 ms 只有代码生成器的归约（0.49 通用求和 + 0.08 六个注意力 GroupNorm 的回退），手写 GroupNorm 1.72 ms、手写卷积偏置梯度 0.26 ms、手写 softmax 1.59 ms 都不在里面；PyTorch 的 1.20 ms 是 0.65 ms 真归约**加** 0.47 ms 被误归类的 GroupNorm 逐元素仿射写回（符号名含 `Norm`），而 PyTorch 的 GroupNorm 统计量归约 0.74 ms 落在 `other`。按语义配对并用 `profile_step_torch.py --attribute` 的 aten 归属核对每步次数（卷积偏置梯度 51:51、其余通用求和 67:67、GroupNorm 41:41，全部对上）后：**Jittor 2.54 ms 对 PyTorch 1.93 ms，慢 32%**。详见 3.22 行 | 代码生成分区，3.22 已处理 |
| UNet 归约差距的 75% 在手写 GroupNorm CUDA，不在代码生成里 | 对齐口径后 Jittor 归约类 2279–2297 us（nsys ~2705）对 PyTorch 1928–1998 us：通用求和只差 79–184 us，**GroupNorm 差 241 us（profiler，1537 对 1296）到 555 us（nsys，1851 对 1296）**，两种量法算出的占比都是 75%。其中 1715 us 是 `backends/cuda/kernels/nn/group_norm_cuda.py` 的三个手写 kernel（35 个 GN，另 6 个注意力 GN 因为输入是 3-D 被 `_supports_group_norm` 拒掉、回退到代码生成器）。可查的成因是多搬字节：`group_norm_forward` 除了 `y` 还物化一份全尺寸的 `xhat` 存给反向，`group_norm_backward_x` 于是要三遍走过 `grad_y` 与 `xhat`；PyTorch 不存 `xhat`，反向从 `X`、`mean`、`rstd` 重算。nsys 逐 kernel：Jittor forward 577 / backward_x 923 / backward_affine 276 us，PyTorch `RowwiseMoments` 214 + `ComputeFusedParams` 64 + apply 181 / `ComputeInternalGradients` 409 + `ComputeBackwardFusedParams` 62 + apply 218+71 + `GammaBetaBackward` 57 us。改法与验收要由手写 kernel 的 owner 定；量法在 `agent/skills/cuda-reduction-strategy-comparison/` | 手写 CUDA kernel 分区，新任务待派 |
| `para_opt_level=4` 的块内共享内存归约比默认慢 1.6–2.0 倍 | 实测四种 UNet 形状：默认（warp shuffle）15.7/14.0/15.0/18.1us，lvl 4（`SharedReducePass`）25.3/31.3/25.3/34.8us，不优化 157/92/159/171us。默认值保持 3。要提升需要「warp shuffle → 每 warp 一个值 → 共享内存 → 每输出一次原子」的混合实现，并且要有生态 harness 的端到端数据；数据与方法在 `agent/skills/cuda-reduction-strategy-comparison/` | 代码生成分区，新任务待派 |
| `tests/structure/test_source_root.py` 3 条 | `AttributeError: module 'conftest' has no attribute 'source_python_dir'`——有人从 `tests/conftest.py` 删/改名了 `source_python_dir`，没同步用例。与并发编译无关，是独立真回归 | 改 conftest 的那个提交（0.13/0.17 一带），门禁已记录 |
| `tests/compat/vllm/test_flash_attn.py::TestTheBundleItPublishes::test_a_submodule_it_does_not_carry_still_imports` | `ModuleNotFoundError: No module named 'vllm.vllm_flash_attn.layers'` | 兼容层分区（7.14 一带） |
| `tests/structure` 在一次 session 里被收集两遍 | `-rf` 摘要里 `test_source_root.py` 的三条各出现两次 | 门禁分区，随 0.04 查收集规则 |
| `tests/core/test_type_system.py` 一套门禁都不跑 | **已修**：0.04 之后 CPU 门禁的 torch 会话就是 `TORCH_MODE_PATHS` 本身，这个文件自然进来了。同一批还有 233 个此前一套 workflow 都碰不到的文件 | 门禁 gates，`6adbf488` |
| `test_atomic_tuner` 抓不到日志 | **已修**：根因确认为 `032ecfe1` 的 `full_reduce_cuda.py` 快路径猴补 `Var.sum`/`Var.mean`，全归约不再进 JIT；第 4 条用例改走 `jt.reduce` | codegen，`72f020b3` |
| `asm_tuner.py` 非原子写 `.s`，并发编译读到截断的汇编 | **已修**：`pass_asm()` 改成写 `<路径>.tmp.<pid>` 再 `os.replace`。判据是 inode——改名换 inode，原地重写不换，也就不会消掉那个窗口；用例 `test_asm_tuner.py::TestAsmTunerWritesAtomically` 钉住。缓存里已经存在的坏 `.s` 不会自动修复，删掉再跑 | 构建，`1919b035` |
| `tests/backends/cuda/test_backend_teardown.py` 过不了 0.21 的静态门禁 | **已修**：gates 在 `a5ce7310` 里已改成 `run_python_child(..., crash_isolated=True)`。cudabk 复核了改后的文件保留全部断言（无 terminate / 退出码 0 / 有 teardown 记录 / 真错误 `cudaErrorIllegalAddress` 仍可见）加那条干净退出的对照，并把 `cuda-backend-choice-proof` 里「子进程 abort 会带走 pytest」那段从只描述现象改成指向 helper 的 `crash_isolated`（`1b117a91`） | 门禁 gates，`a5ce7310` |
| `tests/compat/torch` 的 17 条失败与核心分区无关 | 2026-09-03 在 `77641cc8` 上跑 `test_torch_compat_optim/rnn/unimplemented/linalg` 四个文件：17 failed / 104 passed。逐条看过失败原因，**没有一条落在 flag、节点、遍历或执行器上**：AdamW 八条全是 `fused_adamw is only available through a mapped backend`（`ops/fused_adamw_op.cc` 在 CUDA 上拒绝执行）；RNN 五条全是 `cudnn_rnn_descriptor.cc Check failed: is_type<string>(_slots[7])`；`set_default_device` 两条是设备选择；`test_det_slogdet` 是 cupy 的 `NVRTC_ERROR_COMPILATION`；`test_autocast_actually_lowers_op_dtype` 是 shim 的 autocast 没接上 amp——**这条专门核过**：同一个 amp 探针（level 5 下 `a*b`、`matmul` 的 dtype，以及 AmpGradGuard 把六位 amp 字段读回来后的梯度）在 2.01 前后逐字一致（`float16/float16`、`GRAD float32 128.0`），2.01 的重新编号没有动 amp。同一批失败在 rebase 前的树上也是同一份名单 | 依次为 pyops/compat、cudabk（8.01 一带）、device（4.02 一带）、环境、compat |
| 下一次 rebase 会全量重编一次 | 9.04（`2569fe3b`）同时改了缓存路径与缓存键格式，**这是预期的，不是缓存坏了，不要删自己的 `JITTOR_HOME`**（本机冷构建约 63s）。另外 `cache_name` 的语义从「不设 = 当前 git 分支」变成「不设 = `default`」——靠分支自动分开缓存来隔离并行任务的，改成显式设 `cache_name` 或不同 `JITTOR_HOME`；反过来切分支不再触发全量重编 | 全体，已由协调者广播 |
| 8.03 的前期分析（未实现，交接用） | **已落地**，见 8.03 与 `agent/design/float32-precision-policy.md`（三档映射表、默认值为何不变、两条实质行为变化各自的证据）。 | 已完成 |
| 7.08 的 tf32 映射可以再进一步（8.03 之后） | 9aaedba9 把 high/medium 的细分记在 Python 侧（`_torch_float32_matmul_refinement`），理由是「Jittor 表达不了」。8.03 之后 `jt.flags.float32_matmul_precision` 是真的三档 C++ 状态，`medium` 会真的走 bf16 累加，**表达得了了**。但接上去之前要先决定：Jittor 这个策略是 matmul 与卷积**共用**的，torch 的 `set_float32_matmul_precision` 不动 cuDNN；直接接到共用策略上会让下游一句 `set_float32_matmul_precision("high")` 把卷积也降到 tf32。要么 shim 只写 per-domain 覆盖（现状，medium 仍然只到 tf32），要么核心再分出 matmul-only 一档。cudabk 没有替 7.08 做这个决定 | 7.08 接手人 |
| cuDNN 卷积计划缓存缺一个观测点 | `cudnn_conv_plan.h` 的 plan 缓存没有任何 Python 可见的读数，所以「某个字段确实进了缓存键」只能靠规则测试（源码里不许手写 `req.` 赋值）间接钉，没法直接断言。cuFFT 与 cuTT 都有 `*_plan_cache_size()` / `*_set_plan_cache_size()`（6.C 那批加的），照抄一份 `cudnn_conv_plan_cache_size()` 就能直接断言「同形状二次调用只 +1」「只翻 `cuda_allow_cudnn_tf32` 会多一条」。**数值路线试过、不成立**：允许 tensor-op 只是让那些 engine 可选，cuDNN 仍可能挑 FMA——实测 32×32×24×24 的 fp32 卷积，tf32 关 4.58e-05、开 5.91e-05，差 1.3 倍，做不了判据。注意缓存是 header 里的 inline 函数静态量，而 `jit_run` 编在另一个 .so 里，加访问器前要先确认是同一份实例（`fwd_algo_cache` 那种 EXTERN_LIB + 非 JIT 段定义是现成的写法） | 后端，随 8.07 后续 |
| 8.05 的前期核实（未实现，交接用） | **四条前提今天全部成立，逐条核对过**：(1) **版本钉死**：`jittor_utils/manifest.py:41` 钉 `dnnl_lnx_2.2.0_cpu_gomp.tgz`（2021 年的 oneDNN v2.2，带 sha256/md5）；`compile_extern.py:59` 靠 `lib/libmkldnn.so` 判断解包是否成功，而解出来的 v2.2 目录里 `libdnnl.so` 与 `libmkldnn.so` 两套名字都在——v3 只有 `libdnnl.so`，**换库不先改这一处，表现是「下载了但认为没装上」**。`_asset()` 支持显式 `url=`（CUB 就是指向 github codeload），所以不必往镜像上传资产。(2) **用了 v3 已移除的 API**：六个算子里全是 `convolution_forward::desc` / `convolution_backward_data::desc` / `convolution_backward_weights::desc`，v3 删了 `*::desc`，primitive_desc 直接从参数构造。(3) **每调用重建**：`mkl_conv_op.cc:119-120` 每次 `jit_run` 都 `engine eng(cpu,0)` + `stream s(eng)`，后面 memory desc / primitive desc / reorder 目标 memory 全部重建；三个 conv 与两个 matmul 都是这个形状。(4) **matmul 只支持 fp32**：`mkl_matmul_op.cc:28` 一句 `ASSERT(dsize()==4)`，而 CUDA 侧 fp16/bf16/fp32/fp64 全支持，这种后端能力差异今天无处声明。 | **另有一条计划里没写的**：前向 `mkl_conv_op.cc:153` 用 `prop_kind::forward_inference`，两个反向算子（`mkl_conv_backward_x_op.cc:133`、`mkl_conv_backward_w_op.cc:135`）构造 backward pd 所需的 hint 时却用 `prop_kind::forward`（即 `forward_training`）。**前向与反向 hint 对 prop_kind 的说法不一致**；oneDNN 要求 backward pd 的 hint 来自 forward_training 的 forward pd，真实前向用 inference 时 oneDNN 给前向挑的 layout 可能与反向 hint 假定的不同，代价是多一次 reorder。这就是 8.05 里「训练用 forward_training」那条的落点。**做的顺序建议**：先补测试再动代码——`tests/backends/cpu` 今天只有 **5 条**（conv 4 + test_op 1），撑不住一次换库；先加 MKL 卷积与矩阵乘对 reindex 参考路径的对拍（形状/groups/stride/dilation/dtype 矩阵），再 (2) 迁 v3 与 (1) 放开版本，最后 (3) 缓存与 (4) 能力表——**缓存写在 v2 API 上会被 v3 迁移全部重写**。缓存要注意 memory 对象包着 `x->mem_ptr`，每次调用指针会变，所以缓存的是 pd 与 primitive，执行前用 `set_data_handle()` 重新绑指针。 | 下一位接手 8.05 的人 |
| `jt.flags.nvcc_flags` 的拼法变了 | 9.08 之后架构 flag 是 `--generate-code=arch=...,code=...`，不再是 `-arch=compute_N -code=sm_N`。按后者做字符串匹配的地方要改 | 各分区自查，`2d71f792` |
| `torch.split_with_sizes` + `Var.split` 之后进程退出时 abort | `node.h:264 Check failed: value_ > 0 ... backward liveness release without a matching owner`，整个 pytest 进程被带走（零 summary）。**最小复现**（shim 模式、CPU、显式 PYTHONPATH 指向本 worktree）：`with torch.flag_scope(use_cuda=0): t = torch.array(np.arange(12,dtype='float32').reshape(3,4)); a = torch.split_with_sizes(t,[1,3],dim=1); m = t.split([1,3],dim=1)` 然后在 flag_scope **之外**对两组结果各调 `.numpy()`——打印完 OK 之后在解释器退出时 abort。只做其中一次 split、或不出 scope取 numpy，都复现不了。后果是 `tests/compat/torch/test_torch_numerical_fidelity.py::TestTorchNumericalFidelity::test_split_with_sizes_cpu_shapes_values_and_var_split_match_numpy` 会带走整个文件的运行，7.03 期间只能 `--deselect` 它 | 核心节点分区（`node.h` 的 liveness 计数），新任务待派 |
| opinfo 全量归约的参考电池仍把标量提成 `(1,)` | `tests/ops/test_ops.py` 里 `amax`/`amin`/`count_nonzero` 的 `test_reference_*` 共 9 条红（float32/float64/int8/int16/int32/int64/uint8），报的都是 `shape () != (1,)`。成因是 `tests/opinfo/definitions/reductions_extra.py` 的 `_atleast1d` 注释说「jittor 没有 0-d 标量，全量归约得到 (1,)」，而今天全量归约返回的就是 shape ()。**改前改后同为 9 failed / 2 passed / 4 skipped**，与 7.03 的 owner 迁移无关，是参考电池自身过期 | 测试分区（opinfo 参考），新任务待派 |
| 全树跑时 `test_notebooks.py` 没有被当成 manual 跳过 | **已修**：`pytest_collection_modifyitems` 里 `test_notebooks.py` 的 `pytest.mark.manual` 加在跳过判断**之后**，所以全树跑时它照跑不误——2026-09-03 的全树原生一遍里实测 537 秒，是全树最慢的一项（第二名 289 秒）。现在所有标记先挂完再统一判断，manual 探针改由 `JITTOR_TEST_MANUAL=1` 或 `-m manual` 显式打开。**这是「筛选逻辑的顺序决定筛选结果」的第三例**（另两例：按 `sys.argv` 选 shim 模式、`@onlyCPU` 被设备过滤全部跳过） | 门禁 gates，`5c0f2364`（0.13） |
| backward liveness 在 cuDNN RNN 反向上多释放一次 | CUDA 上 `jt.nn.LSTM` 训练 + `jt.grad`，退出期 `LivenessCounter<backward>::release()` 的 `ASSERT(value_ > 0)` 触发（`node.h`「backward liveness release without a matching owner」）。**2.10 之前这是 `int backward_liveness--`，下溢到 -1 后 `if (!backward_liveness)` 恒假，于是节点永不释放——静默泄漏而不是报错**，所以这是上游一直存在、被 2.10 的断言照出来的。2.10 的验收只跑了 CPU（看板原话「状态逻辑后端无关，未追加 GPU 编译」），而 CPU 上同一段 LSTM 反向不触发。**可疑点**：`node.cc` 的 `release_forward_liveness` 里 b3 那段在循环**外**判一次 `liveness.backward.active()`，循环**内**对每个满足条件的输出各 enqueue 一次对 `this` 的 release——backward 计数是 1 而合格输出有两个时就会多释放。这一条要么修计数、要么说明为什么该多释放，**不要用「放宽断言」了事**。**它不是 CUDA 独有的**：`4b5eaaa9` 之前的树上 `JITTOR_TEST_DEVICES=cpu pytest tests/core` 也在 `test_function.py::TestFunctionWithEagerExecution::test_multi_grads_multi_out_stop_grad_1` **SIGABRT（EXIT=134，跑到 65%，没有汇总行）**——也就是说 2.10 合入之后 `tests/core` 的原生 CPU 门禁一直是「进程死了」而不是「几条红」，而没人看出来。2.19 已让析构不再因此 abort（`~VarHolder` catch + LOGe），于是同一个账不平现在报成 **`lived_vars 2 != 0`**：`test_function.py` 的 `TestFunctionWithEagerExecution::test_zmem_leak{,2,3}` 三条红，泄漏的正是那次失败释放留下的 2 个 var。**这三条红是这个缺陷的正确落点，不是 2.19 引入的**（逐条比对过：修前能跑到的用例，修前修后结果逐条一致；这三条修前根本跑不到）。修好计数之后它们会自己变绿。`test_var_holder_teardown.py` 就是靠它触发的，修好之后要给那条用例换触发点 | 核心 coreops，2.10 接手人 |
| `jt.bfloat16(math.inf)` 让 `code` 算子的代码生成死循环 | `tests/backends/cuda/test_bf16.py::test_safe_clip` 与 `test_fp16.py::test_safe_clip`：0 维输入使 `@for(j, in@i@@_dim-2, -1, -1, ...)` 展开成 `@for(j, -2, -1, -1, ...)`，`op_compiler.cc:589` 的 `Check failed: total_step < 1000  Too much step` 触发。**真正贵的是级联**：这一条失败之后同一进程里 `test_bf16.py` 剩下的 18 条、`test_fp16.py` 的 1 条全部报同一个编译错误，单独跑却全绿——所以 CUDA 目录 23 条红里有 21 条是这一个根因。修的时候两件事：0 维的 `@for` 边界，以及一次编译失败为什么会毒化后续无关算子 | 代码生成分区 |
| ~~`setup_cutt()` 全树没有调用点，cuTT 后端不可达~~ **已修 `4bdc7e797`（bindings/2.19 代 9.01 补）** | `compile_extern.py` 里 `setup_mkl` 由 `nn/functional/matrix.py:178` 惰性调用、`setup_nccl` 由 `compat/collectives.py` 调用，**只有 `setup_cutt` 没有任何调用点**（9.01 把三个 setup 改成惰性时漏了它）。后果：`cutt_ops` 恒为 `None`，`tests/backends/cuda/test_cutt.py` 与 `test_cutt_transpose_op.py` 共 6 条用例（含 3 条 `expect_error` 负向）**从来没跑过一次**，2.19 的 `6375a852`（cutt transpose axes 用户边界）没有任何运行时证据。恒 skip 的条目等于没有条目。**修法**：调用点补在 `core_api.transpose` 的首次调用（与 MKL 同形，import 期仍不装载）；测试改 `setUpClass` 里 `load=True`（模块作用域读惰性库属性恒为 `None`，这正是那条假 skip 理由的来源）。**打开后一次暴露三层坏账**：编译命令缺 `cuda_sdk_flags` 与 `backends/cuda/include`（4.13 已并行修掉，见该行）；`transpose`/`fuse_transpose`/`cutt_transpose` 三处构造函数把「axes 从 0 递增」当恒等置换而不比较秩，`axes=[0]` 作用在二维输入上**静默返回未转置的原张量**，`infer_shape` 的 `USER_CHECK` 不可达；`test_matmul_grad` 定义两次、后者盖掉前者。cuTT 六条现为整文件 9 passed | 构建 build，9.01 接手人 |
| CUDA 异步故障不 sync 就永远不报 | 一个越界写的 kernel 之后 `Var.sync()` 干净返回、进程一路正常跑到退出，`cudaDeviceSynchronize()` 此时返回 700，但 jittor 从头到尾没说过一个字；只有显式 `jt.sync_all(True)` 才会在 `cuda_flags.cc:194` 报出 `cudaErrorIllegalAddress`。也就是说**一次真实的非法访存可以完全无声地走完整个训练**，而进程退出时唯一的输出是库句柄销毁失败的 teardown 噪音。`test_backend_teardown.py` 原来的探针就是踩在这上面（它 `except: pass` 了一个根本没抛的异常，于是「真错误盖过清理噪音」这条断言在真机上一直是假的，2.19 已改用 `sync_all(True)`） | 后端 cudabk |
| ~~`test_source_signature_sees_same_size_edits_and_new_files` 在 HEAD 上就红，且会污染后面的用例~~ **经实测不成立，两个命题都不成立** | 交接里这条写的是「它 mock 掉 `compiler.jittor_path`，mock 退出时的还原被拒绝，于是该属性停在已删除的临时目录上；随机顺序下有时把 `test_plain_import_does_not_call_external_setups` 一起带红」。**逐条实测（`-p no:randomly`，nodeid 顺序钉死）**：(1) 那条用例**不在 `tests/structure` 而在 `tests/compiler`**，且在 HEAD 上**已绿**——`d23f9bba6`（9.01）已把它从 patch `jittor_path` 改成传 `core_source_signature(root=)`，该提交的说明里也写了「顺带修了 2.13 冻结 jittor_path 之后一直红的」这一句；单跑 `TestCoreBuildStamp` 7 passed，整文件 31 passed。(2) **描述的污染机制不可达**：把旧的 mock 写法原样复现出来，`mock.patch.object` 在 **`__enter__`** 就被冻结拒绝，属性从未被改过；受害者随后仍绿。真正发生的是 mock 的 `__enter__` 在 setattr 抛出后调自己的 `__exit__` 回滚，而回滚写的是**原值**、也被拒——**报出来的异常来自清理路径且消息里带原值**，读起来像「还原被禁止」。所以它响亮（用例红）且不留残留，不是静默污染。(3) **本环境根本没装随机顺序插件**：pytest 7.4.4 只有 `pytest-xdist` 与 `pytest-timeout`，`-p randomly` 直接 ImportError（`-p no:randomly` 不报错是因为 `no:` 不需要 import，不能用它判断插件在不在），所以「随机顺序下才复现」在本机只能来自 `-n`。(4) **同形态全树扫过：0 处**——`tests/` 里以 `compiler` 模块为 target 的 patch 共 7 处，全是函数或 `has_acl`，无一是 startup flag；唯一点名冻结 flag 的 `test_runtime_sync_state.py:670` 在 `pytest.raises` 里断言拒绝；`test_preflight.py:24` 改的是 `os.environ` 不是模块。**留下的真缺陷是「没人钉住」**：原有 `test_all_native_flag_instances_reject_late_startup_writes` 只写**已经在那儿的值**，因此对「先赋值再抛」的实现照样通过——打洞实测经 `patch.object` 断言 28 passed 一条不红，改成直接 `setattr` 一个**不同**值后 10 failed。已补两道门禁：`tests/core/test_startup_config.py` 的 refuse-before-mutate，与 `tests/_helpers/state_leaks.py` 把冻结 startup config 纳入每文件快照（后者按 owner 报出「哪个文件改了 `compiler.jittor_path`」，与机制无关，实测能点名） | bindings，2.19 一带 |
| `test_cudnn_op.py::TestCudnnConvOp::test_backward_nhwc` 在本机 cuDNN 上 `Unexpected success` | 标着 `expectedFailure` 但在 cuDNN 8.x + sm_89 上真的通过了。要么这条限制已经不存在（那就去掉标记并说明从哪个版本起成立），要么标记的条件写得太宽 | 后端 cudabk |
| `test_shared_reduce.py::test_shared_reduce_helper_is_two_stage` 按 locale 编码读生成源码 | 整目录跑时 `UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2`，单独跑通过。jittor 生成的 JIT 源码里有非 ASCII（op key 用 U+00AB 分隔），读文件时没给 `encoding="utf-8"`，是否报错取决于当轮生成了哪个算子——**这类按环境编码解码的读法全树要扫一遍** | 代码生成分区 |

## 跨用例状态泄漏清单（0.15 的前置，2026-09-03 全树实测）

`tests/conftest.py` 在**每个测试文件**前后拍一次快照（三个存活计数、六个关键 flag、
**冻结的 startup config**、`sys.modules` 里换了对象的条目），只报告不失败。全树原生一遍的
结论比预期干净得多：

> startup config 这一档是 2.19 后来补的（见上表「`test_source_signature_...` 经实测不成立」那行）。
> 它监视的是 `jittor.compiler` 上 `STARTUP_FLAGS` 的当前值，理由与 flag 那档同形但后果更重：
> 这些值是**目录名**，`jittor_path` 停在一个已删除的临时目录不会让改它的那个文件失败，
> 而是让后面某个走源码树的文件报一个与起因毫无关系的缺文件错误。冻结模块会拒绝这类写入，
> 所以这里一旦有差异就说明「拒绝了」不等于「值还是对的」。清单是从 `STARTUP_FLAGS` 求出来
> 而不是手写的，新增一个冻结 flag 不需要有人回来补。实测能点名到文件（`tests/structure/test_state_leak_helper.py` 钉住）。

| 文件 | 留下什么 | 处置 |
| --- | --- | --- |
| `tests/nn/test_nn_capabilities.py` | `number_of_hold_vars 0 → 7` | **不是测试留下的**，见下 |
| `tests/ops/test_fft_op.py` | `number_of_hold_vars 0 → 26` | **不是测试留下的**，见下 |

**这两条已定论，而且原来的解释（「模块级留着 Var」）是错的。** 实测手法：跑完文件后
依次丢掉测试模块、`_pytest`、`_helpers.common`，每步 `gc.collect()` 再读计数——
**三步之后计数一个都没掉**（26 → 26 → 26）。再用 `jt.dump_all_graphs().hold_vars`
把它们逐个打出来，形状说明了一切：

- `test_fft_op.py` 的 26 个是 **13 对** float32 `[n,n]`（n=1..12），正是
  `python/jittor/fft/__init__.py` 的 `_dft_mat_cache`——按尺寸缓存的 DFT cos/sin 矩阵对，
  `OrderedDict` LRU，`_dft_mat_cache_limit = 16` 对。
- `test_nn_capabilities.py` 的 7 个是 int32 一维小向量（`[2] [3] [3] [3] [4] [3] [3]`），
  正是 `python/jittor/nn/attention.py` 的 `_CU_SEQLENS_CACHE`（cu_seqlens 前缀和），
  同样是 LRU，`_CU_SEQLENS_CACHE_LIMIT = 128`。

**两个都是 jittor 自己的、有上限的进程级 memoization，不是泄漏，测试这边没有东西可改。**
真正的结论是那条一般规律有了具体机理：**`number_of_hold_vars` 有一个下界，取决于
这个进程曾经跑过哪些算子，而不取决于当前这条用例**——所以对它做绝对断言按构造就是错的。
`tests/_helpers/state_leaks.py` 现在把这两个缓存的条目数一起快照，报告会直接写
「26 个里有 26 个是 `jittor.fft._dft_mat_cache` 0 → 13 条（上限 16）」，剩下的差值
才是值得去查的东西。

**六个 flag（`use_cuda`/`no_grad`/`amp_reg`/`use_parallel_op_compiler`/`exclude_pass`/`th_mode`）
在原生这一遍一个都没泄漏**，`sys.modules` 也没有未还原的替换——0.12 那一批修到位了。

### 受害者一侧才是要改的地方

已知的五个"单独跑绿、合跑红"样本里，机理清楚的三个都不是污染源的错，是**受害者对全局
计数器做了绝对断言**：

| 样本 | 状态 |
| --- | --- |
| `test_fused_op.py::TestFusedOp::test_add` | **已修**（`bffe0bf4`）：断言 `(hv,lv,lo) == (0,0,0)` 改成比用例开头的基线增量。它真正想断的是"这张图创建了几个节点、融合后活下来几个"，那是一个差 |
| `tests/ops/test_linalg.py::TestBUG4_2Op` | **已修**：`use_cuda=1` 改 `@jt.flag_scope`（0.12 / 6.P23） |
| `tests/compiler/test_jit_tests.py` 的两条 sfrl | **已标记**：墙钟阈值，改 `@pytest.mark.load_sensitive` |
| `test_torch_compat_fsdp2::test_single_rank_fully_shard_preserves_math_and_state` | 待查（torch 会话） |
| `tests/compat/torch/test_torch_compat.py::test_torch_compat` | 待查（torch 会话；单独跑 549s 通过，整套里失败，且在兼容层那批改动之前的基线上就这样） |
| `tests/data/test_dataset.py::TestDatasetSeed::test_children_died` | **已定论，进 A 表**：单独跑也失败，恒在子进程超时上（300s）。不是泄漏，是真缺陷——worker 被 `p.kill()` 之后父进程不再靠 SIGCHLD 快速退出，而是一直阻塞等那个死掉的 worker 的数据。已改 `xfail(strict=True)` 加 `slow`，子进程超时从 300s 收到 90s（「快速退出」本来就该用更短的界来断言），门禁每轮从 302s 降到 95s。strict 意味着谁修好了它门禁会红，提示删掉这个标记 |

**一般规律**：对进程级全局量（存活计数、墙钟、flag）做**绝对**断言，断的不是这条用例的
性质。能写成增量就写增量，写不成就说明这条断言依赖一个它管不着的前提。

还有一类**绝对上界**留着没改，风险低但同形状：`test_inception.py:125`
（`lived_vars < 50000`）、`test_resnet.py:136/138`（`< 8100` / `< 7000`）。
余量是实测污染量（33）的两个数量级，暂不动；真要动就同样改成比循环前的基线。

> 8.06 补充证据：`ba8e2621` TruthReduce all/any 统一共享 launcher，静态合同 36 passed；本机无 CANN/NPU，仍待实机验证。

### 2026-09-04 第六十五波补充证据

- `8.06`：`3f0b8c7d` 将 GroupNorm forward 接入共享 launcher，保留 group/eps/三输出 query；静态合同 53 passed，本机无 CANN/NPU，仍待实机。
- `8.06`：`016fc62d` 将 GroupNorm backward 接入共享 launcher，保留 output-mask、group 属性、三输出 query 与 cleanup；静态合同 54 passed，本机无 CANN/NPU，仍待实机。

## 任务

| 编号 | 任务 | 状态 | 负责 | 提交 |
| --- | --- | --- | --- | --- |
| 0.01 | `TestGradients` 改用 `only_for=("cpu",)` 显式实例化 | 已合并 | gates | aee8ecaa（+355deb6e） |
| 0.02 | 设备过滤后 bases 为空或方法数为 0 时生成器直接 raise | 已合并 | gates | e5eb0d05 |
| 0.03 | `tests/compiler/test_jit_tests.py` 进 CPU 门禁，并断言 … | 已合并 | gates | a5e7f654 |
| 0.04 | 门禁改为「整个 `tests/` 减显式排除清单」，排除项必须写理由 | 已合并 | gates | 6adbf488、689e206b |
| 0.05 | 生态对拍进 nightly | 已合并 | gates | 97125c6e。`nox -s ecosystem` + `.github/workflows/ecosystem.yml`（每天 02:00）。**fail-open 是这里的真问题**：这些用例在 `REAL_TORCH_PYTHON` 没设时自我 skip（对的，拿 shim 和自己比证明不了什么），于是丢了 oracle 的 nightly 会**为它唯一存在的理由报成功**。三道闸：`JITTOR_REQUIRE_REAL_TORCH=1` 时把「缺 torch」从 0.18 的环境解释里撤走并让这类 skip 退出非零、逐条列出哪些对拍没发生；session 层缺 oracle 直接 abort；起手在 oracle 解释器里断言它的 `torch` **不是** shim（防的是最坏情况：两边其实是同一棵树，对拍全绿而什么都没证明——本机 jt311 的 `torch` 正是 shim）。本机验证（jt311 对 jt312b 的 torch 2.12.1）：12 个 CPU 对拍用例真跑、逐参数与逐输入梯度全过、8 分 23 秒 |
| 0.06 | `make_tensor` 种子改为 `hash(nodeid, shape, dtype)` … | 已合并 | gates | a4d041e6。稳定种子包含 nodeid/shape/dtype 等输入，失败信息报告 seed；6 条契约测试覆盖单跑/全量一致性，固定 seed 值不受进程哈希盐影响 |
| 0.07 | 缓存路径追加构建配置指纹 | 已合并 | 构建 | 82dfce6e、6379b2b5、6fdb3807、b25fcdfa（复验） |
| 0.08 | 锁统一为一种类型、一个 fd | 已合并 | 构建 | 460bead0 |
| 0.09 | 探测结果落盘 `cache_path/probe.json` | 已合并 | 构建 | 240a92a3 |
| 0.10 | 写缓存前检查可用磁盘空间，不足时给明确错误 | 已合并 | 构建 | 73eceeaf |
| 0.11 | 「jit_utils 已更新请重跑」改非零退出码 | 已合并 | 构建 | 7e8c7c74 |
| 0.12 | 14 处在用例里裸赋值 `jt.flags.*` 且无 tearDown 的测试改 `flag_… | 已合并 | gates | 26a20905 |
| 0.13 | conftest 的模式由显式环境变量决定，删除 `sys.argv` 嗅探 | 已合并 | gates | 5c0f2364、a4ebb31a。**日常影响**：手跑 `tests/structure`、`tests/compat/torch`、`tests/ops/test_ops.py` 等 `TORCH_MODE_PATHS` 下的路径要带 `JITTOR_TORCH_SHIM=1`，不带会得到一条指名变量的报错（而不是一次语义不对的绿）。`nox -s structure` 已经自己设了 |
| 0.14 | `_session_env` 不再 `os.environ.copy()` | 已合并 | gates | 6b8fb594。未声明宿主变量显式屏蔽，工具链/下载入口按白名单透传；OMP/MKL/OpenBLAS 等线程池固定并随 worker 缩放，子进程 probe 断言线程数与 CPU affinity。聚焦结构 18 passed，真实 nox probe 在受限 affinity 下通过 |
| 0.15 | 门禁分两层 | 待领 | gates | `faad4898` 将 smoke 独立组切到 xdist loadgroup，`f239e2ed` 增加预算/瓶颈报告，`5c6876cd`/`cab14f53` 纳入 cgroup CPU quota，`c7bfe24b` 在 pytest 前执行预算 fail-closed，`de78f9ae`/`1ecb35ff` 暴露配置与实际 worker/quota/线程诊断，`bee0263e` 校验 CLI/runtime worker 参数，`895f69a9` 在 pytest 前输出 effective budget，`f3ef7a70` 统一 runtime worker policy，`91ca525f` 对齐 budget report 默认值，`863f013f` 统一 standalone runner 执行策略；结构/环境定向 44 passed，完整 smoke 仍待最终验收。d957e4aa、9329c4f9、9f6a80c7、2fd26522 已合入：按实测慢文件拆出 smoke/full、并行度单点声明、PR smoke job 与 JIT cache 已接入。`876ec09c` 修正 RingBuffer worker-death 等待；独立 Dataset 两个 worker 监管 nodeid 在临时缓存下 2 passed/65.68 s，但完整 smoke 仍约 390 s、预算模型约 446 s，尚未达到原验收的 5 分钟；还需减少或降低有效测试工作量，不能靠扩大排除清单假达标。**2026-09-06（`c31439067` 一波）量出时间构成，并据此判定 5 分钟验收不是分层能达到的，保持待领。** 热缓存、门禁口径、`-n 4`、16 核、load 13-18 实测：native 406.3 s + torch 91.8 s = **498.1 s**；两个半边都是 work-bound（native 1592.9/4 = 398.2 对墙钟 406.3；torch 328.2/4 = 82.1 对 91.8），**没有被某一个长文件卡住**，所以「再挑几个慢文件推迟」买不到多少。要到 300 s 得让 native 半边工作量从 1592.9 s 降到约 550 s，砍 65%——那是 10.18 的「更少或更便宜的比较」或更多机器，不是排序。**交给这一波的前提（结构套件吃掉整个 smoke 预算）实测不成立**：`tests/structure` 只占 **9.4%**（约 47 s / 498 s），整个在 torch 半边，占 81.6% 墙钟的 native 半边一条结构测试也没有，全部推迟也只到约 451 s。native 半边构成：`tests/ops` 31.8%、`tests/core` 27.3%（`test_setitem.py` 一个文件 15.0%）、`tests/distributed` 16.2%、`tests/compiler` 12.1%、`tests/nn` 9.5%。**本波没有新增任何推迟项**（不靠扩大排除清单假达标）。副产物：修掉判据工具在 xdist 下不给判据的缺陷（见审计 05-tests「marker 体系」行下的已修条目）；并实测出**结论集合有约 27 个 nodeid 的抖动地板**，所以「两轮逐条相同」暂时不能直接当验收，用法见交接文档第二百一十八波。**第二条验收（一轮能报出全部失败）本波也修了一处真缺陷**：smoke 的 `--dist loadgroup`（0.15 自己的 `faad4898` 引入）把有模块级共享状态的 `test_torch_compat_fsdp2.py` 拆到四个 worker 上，torch 半边两轮之间 3 个 nodeid 结论不同，且逐条比全是 `passed -> failed`——**拆散文件会让本该失败的用例通过，这一层在漏报失败**。加模块级 `xdist_group` 后两轮逐条 0 处不同（`compare` 报 IDENTICAL，2201 对 2201），墙钟 88.5/88.8 s 对改前 91.5/88.7 s，不变慢；防退化断言在 `test_gate_tiers.py`。native 半边剩 22 条抖动未逐条追（一批是冷缓存才失败），留待下一波 |
| 0.16 | `test_device_parity.py` 按算子分片并行，不再在 `setUpClass`… | 已合并 | gates | 120b004b。实测结论与原方案相反：4-worker 只快 6% 且 26 项丢 3 个结论，因此保留单进程；只移除错误的串行编译器强制关闭。后续真正压缩时长另见 0.22 |
| 0.17 | `pyproject.toml` 的 `pythonpath` 改由 conftest 按环境变… | 已合并 | 构建 | b19d098f |
| 0.18 | 门禁每条目断言至少执行 1 个非 skip 用例 | 已合并 | gates | ee29bee3、2f3f1aaf。恒 skip 的判据**从路径清单改成规则**：读测试自己写的 skip 理由，全都在说「这台机器缺某样东西」才算解释得通。清单版在这台机器上会是 73 条、换台机器又是另外 73 条，而且每加一个设备测试都要记得报到。规则一上线就抓出四个说不清自己缺什么的文件（`Not use cub, Skip`、`skip_this_test`），都改成说明缺什么，而不是给它们开豁免 |
| 0.19 | 结构测试从「精确清单」改成「规则」 | 已合并（验收一项未达，见 0.25） | gates | c3bcd277。**核实：主体（清单改规则）做到了，但验收里的「`tests/structure` < 2000 行」没有达到，而且方向相反——当时 22 文件 8071 行，现在 107 文件 19774 行。** 不改判成「未完成」，因为 0.15 这一波量过之后可以确认：**行数在这个目录里不是耗时的代理**（16 个 ACL 静态合同 3292 行只花 13.2 s；序贯最贵的三个文件是 116/68/127 行），所以「< 2000 行」原本用来买的那件事（PR 门禁别被结构测试拖慢）**已经不需要靠它买了**——结构测试只占 smoke 的 9.4%。剩下的问题是维护成本，那是一个独立且仍然成立的问题，拆成 `0.25` 单独判 |
| 0.20 | 布局收尾 | 待领 | | ef31a0d6 已合入 1/N：删除 `tools/services/legacy` 的 converter launcher 与说明，清除 tools 活跃导航和 compat converter 对旧部署脚本的引用；converter 模块保留，HTTP 服务部署由应用负责。不存在结构节点 1 passed，仓库布局通过。`agent/design`/`agent/results` 权威树迁移、`tests/system` 删除及 AWESOME/ASV 归位均未做，保持待领 |
| 0.21 | 测试起的子进程不带 PYTHONPATH，门禁机器上是假绿 | 已合并 | gates | 46dbe946、a5ce7310 |
| 0.22 | 压缩设备对拍时长（保留与单进程相同的 nodeid 集合） | 待领 | gates | `dcc335d6`、`f9c26111`、本次提交。**判据已落地并通过，但原验收（压到可接受时长）没达到，所以保持待领。** 方向 (a) 缓存 CPU 侧参考结果做完了：同一批 26 个 nodeid、两轮都是冷算子缓存，**848.5s → 711.1s（−16.2%）**，`tools/gate_conclusion_diff.py compare` 报 `IDENTICAL: every collected nodeid concluded the same way`（26/26 收集、26/26 有结论，逐条比对，不是数个数）。按 253/26 外推：约 2h18m → 1h55m。**计划里「它是最直接的一半」不成立——实测 CPU 那半只占冷缓存墙钟 18%、热缓存 26%。** 更要紧的是 **0.16 的归因反了**：它记「热缓存 1405s ≈ 冷 1444s，所以不是编译瓶颈」，复测（同一批 nodeid、同一 `JITTOR_HOME`、背靠背）**冷 848.5s、热 23.6s，36 倍**——它就是编译瓶颈。所以下一步不该走 (b) 减样本 或 (c) 多卡多进程，而是 **0.23**：CUDA workflow 没有像 `cpu.yml` 那样 restore/save JIT 缓存，每次都从冷开始 |
| 0.23 | CUDA workflow 持久化 JIT 缓存 | 已合并 | gates | `e4682406`：`cuda` 与 `benchmark-cuda` restore/save JIT cache；key 包含 runner、CUDA 版本、`cuda_archs`、NVCC flags hash 和源码 hash。PyYAML + workflow 结构合同 4 passed。 |
| 0.25 | 改判 0.19 的「tests/structure < 2000 行」验收 | 已合并 | 见备注（判定，无代码改动） | **协调者判定（2026-09-07）：行数预算删除，换成「枚举必须双向报红」。** 计划 0.19 的验收单元格已改。理由分三层。**一，原理由已被实测推翻**：0.15 的执行者量出 ACL 静态合同 3292 行只花 13.2 s、`tests/structure` 只占 smoke 墙钟 9.4%，所以「行数大所以 PR 门禁慢」不成立；行数从 22 文件 8071 行长到 **113 文件 21042 行**（我 2026-09-07 实测，比 0.15 记的 107/19774 又长了），而增长主因是缺硬件的后端用静态合同代替运行时验证，那是真覆盖。**二，行数这个代理量根本分不清好坏**：全树有 **32 个**结构测试文件带 ≥8 项的字符串枚举，最大的两个是 `test_nn_structure.py` 的 114 项与 `test_misc_structure.py` 的 109 项——而前者正是 `test_exports_match_the_recorded_public_api` 用的**公开 API 快照**，恰恰是 0.19 明确要求**保留**的那一类。同样，`test_compat_write_entry_points.py` 把 53 个进程全局写入口钉成分类闭集、`test_deferred_hardware_manifest.py` 同步硬件延迟清单，都是有意的枚举。行数预算会一并惩罚这些。**三，真正的判别性质是方向**：合法枚举（快照、闭集、清单）天然双向——新增未登记项报红，登记项从树里消失也报红；而 0.19 要消灭的「精确清单」是单向的，只在有人新增时报红，于是随重构静默过期。2026-09-06 一天内在七个分区各撞到一次同形失效（见交接文档 §6bis），没有一次与文件大小有关。**剩余面**：那 32 个枚举需逐个判定是否双向，这正是 `10.18`「结构测试预算转向核心」的内容，不另开任务号 |
| 0.24 | 没有任何东西检查「CUDA 门禁真的跑过 CUDA」 | 已合并 | gates | `13d314ec`：CUDA nox session 设置 `JITTOR_TEST_REQUIRE_CUDA=1` 与 accelerator 最低执行数；conftest 对 `has_cuda` 和非 skip accelerator 用例 fail-closed。硬件门禁结构合同 4 passed，compileall 通过。 |
| 1.01 | 把 `utils/data.gz` 解出的 `data.cc` 还原为可读的五个翻译单元 | 已合并 | codegen | ecb6a112（+72f020b3 用例） |
| 1.02 | `op_compiler.cc:30-69` 用正则给 `ParallelPass` 输出打补丁… | 已合并 | codegen | 3eb34e6a |
| 1.03 | 查明 `SharedReducePass` 在约 4900 个归约 kernel 里零命中的触发… | 已合并 | codegen | 3eb34e6a |
| 1.04 | `ReduceTuner::run` 不再对 CUDA 直接返回 | 已合并 | codegen | aebb1d73 |
| 1.05 | 布局收尾 | 待领 | | 5ac222bb 已合入 1/N：`python/jittor/src/test` 的 20 个 C++ 单元测试原样迁至 `src/tests`，构建递归发现、两项结构引用和 Python bridge/JIT_TEST 继续工作；活跃注释、allocator skill 与目标布局同步。结构引用 2 passed，bridge 发现与 `test_expr` 2 passed。其余 39 个根文件、三个 pass 与 `data.gz`/`vdp` 未做，保持待领 |
| 2.01 | Var 与 Op 各持自己的 flag 类型 | 已合并 | coreops | 5b197cae |
| 2.02 | 删除 `Node::custom_data` | 已合并 | coreops | 505e9b37（上半：拓扑排序自带入度，内存分析器的手工备份删掉）、77641cc8（下半：grad/dump 各持局部表，执行器与 fuser 的批下标搬到 `Node::batch_index`+`batch_stamp`，写用 `set_batch_index`、读一律 `batch_index_at(stamp)`并当场校验）。**字段本身仍在**：第六个用法是 FusedOp 的跨阶段映射，见 2.24（排在 3.11 之后）。审计描述的危害「任意两个遍历交错就互相破坏」到此消除 |
| 2.03 | `tflag` 全局计数器加魔数改为 epoch 对象或局部集合 | 已合并 | coreops | 6833f96d。嵌套 TraversalEpoch 恢复外层标记，grad/graph/memory profiler 改局部索引或集合；CPU 33 项、CUDA 3 项及结构聚焦通过 |
| 2.04 | `Var::allocator` 去类型双关 | 已合并 | | 9b3841b7 |
| 2.05 | 真正的 0 维张量 | 已合并 | coreops | 2cfc5a0d。空 shape 保留，Python/NumPy/C++ 标量来源及 reduce/arg-reduce/getitem/reshape/JIT/CUB 全归约统一 0-D；新 CPU 3 passed、GPU2 4 passed，相关 CPU 52 项与 GPU autograd 8 项通过 |
| 2.06 | 边表由 list 加反向迭代器改 SmallVector，按下标 O(1) | 已合并 | mem | ae2a1b70。输入/输出边使用内联容量 2 的 SmallVector 与反向下标，随机访问 O(1)；保序删除同步修正移位边下标，保留 fuser 依赖的消费者创建顺序。C++ 容器/边契约 2 项、CPU 生命周期 1 项、GPU1 CUDA 节点 1 项通过 |
| 2.07 | `hold_vars`/`sync_ptr` 析构里 `std::next(end())` 的 … | 已合并 | coreops | 1101f3f5 |
| 2.08 | `Node` 不再 include `pybind/py_var_tracer.h` | 已合并 | coreops | 6221d4c4。NodeLifecycleObserver 接口由 pybind tracer 注册；无 Python include 的语法编译、CPU/CUDA lifecycle/tracer 聚焦通过 |
| 2.09 | `th_mode` 从 C++ 核心上移为 autograd 策略对象 | 已合并 | coreops | b55f1acb。核心改为通用 `AutogradPolicyState`，Python autograd 层提供不可变 native/explicit-requires-grad 策略与可恢复 scope；核心 Torch 专属状态名归零。新 CPU 4 passed/1 skipped、GPU2 1 passed，相关结构 7 passed，布局与静态检查通过 |
| 2.10 | 三套 liveness 计数 | 已合并 | coreops | 8bd07e51。f/b/p 收进无额外存储的 NodeLiveness；own 防溢出，release 对无匹配 owner 的下溢立即报错，跨零返回值统一传播边界；need_free 与 graph expected-count 由封装提供，release 构建常开。C++ liveness/check_graph 契约与 CPU 生命周期 2 项通过；状态逻辑后端无关，未追加 GPU 编译。**2026-09-06（10.18 的属性测试）为这条封装报的那个下溢定位到确切触发条件，缺陷本身仍未修**：经 `Function` 建的**多输出**算子（即 `Tapes`，`src/ops/tape_op.h`）在**有一些但不是全部**输出被 `stop_grad()` 且整批真的执行时，`LivenessCounter<backward>::release()` 被多调一次，报 `backward liveness release without a matching owner`；该异常被 `var_holder.cc` 的 teardown 路径吞掉，于是**每次泄漏恰好 2 个 Var**，各自 `f=0 b=1`（即 `need_free()==true` 却还在注册表里）。这 2 个就是 `tests/core/test_function.py` 的 `test_zmem_leak{,2,3}` 报的 `2 != 0`——三条点用例只报了这个数，没有指向任何机制。**与 `lazy_execution` 无关**（两个模式都泄漏）。八行差分对照（两个都 stop_grad 干净、都不 stop_grad 干净、单输出干净、两个普通算子干净、两输出 `jt.code` 原生算子干净、只 sync 一个干净、不执行干净）见 `tests/core/test_core_invariant_properties.py` 的 `KNOWN_LEAKING_SHAPES` 注释。**原生多输出算子干净说明问题在 taped 路径而不是多输出本身**；`op.cc:268-270` 的 vnbb 守卫写成 `_outputs.size()==1 && ... is_stop_grad()`，正是源码里那处不对称，但**是否就是修点由本任务 owner 判定，10.18 只负责证明性质被违反**。防退化与强制回访已就位：`test_dropping_a_graph_leaks_nothing_new` 绿（多出一个形状就红），`test_dropping_a_graph_leaks_nothing_at_all` 是 `xfail(strict=True)`，**修好那天它 XPASS 变红，强制连同那三条点用例一起更新** |
| 2.11 | `VarHolder` 不再是执行触发点 | 已合并 | coreops | 0f709cff。VarHolder 构造只登记持有关系；lazy/eager/auto-flush 策略迁入 Executor::submit_pending，Var 完成 Python 对象转换后才提交，显式 core.submit_pending 可无设备同步启动目标子图；删除 flush_suspended 与构造期吞错。构造/边界结构 2 项、CPU 显式提交/错误边界 2 项、GPU1 auto-flush 等价 1 项通过 |
| 2.12 | 打破 `Executor ⇄ VarHolder` include 环 | 已合并 | coreops | 318a688e。依赖 exe.allocator 的 migrate_to_cpu/data/raw_ptr/set_data 四个 inline 实现移到 var_holder.cc，var_holder.h 不再包含 executor.h 或引用全局 exe；executor.cc -> var_holder.h 保持单向，方法签名与行为不变。无 Python include 的独立头语法编译、依赖方向结构节点、CPU submit_pending 节点通过 |
| 2.13 | 执行相关全局状态 | 已合并 | coreops | 计划点名的 hold_vars/exe/sync_ptr/tflag_count/use_cuda/device_id/sync_run 已归 NativeRuntime，旧 cuda_flags 文件消失；遍历/设备迁移见 b6eb6dc0、dfa047b3。本提交完成启动配置 10 项、运行策略 66 项、只读计数 5 项的统一分类，jt.config 深只读、jt.runtime 可写且提供可恢复 scope；原生所有 Flags 实例和 Python compiler 旧入口均拒绝晚写启动配置。Torch 严格数学改为捕获到普通/融合 JIT key 的运行策略，不改启动 nvcc_flags；分类清单纳入构建指纹。CPU/CUDA/结构最终 109 passed，含真实 CUDA 舍入与缓存隔离；Torch bootstrap 47 passed；CPU-only/自定义扩展 23 passed、1 个 CUDA 架构字段按能力跳过。旧扩展需重编；NPU/ROCm 仍需异机实测，本项不声称完成其他 Backend 或独立 torch 任务。 |
| 2.14 | `src/misc/` 拆散 | 已合并 | coreops | Nano 类型与 miniz 前置已就位；本提交将剩余 24 个文件按诊断、运行时、数值类型和通用容器归入 debug/runtime/type/utils，src/misc 不再存在。核心、JIT 生成 include、Python 内嵌 C++、CUDA/ACL/HCCL/MPI 消费者和活跃路径测试均同步，算法不变。转换缓存会把过期原生源归档到编译树外，防止搬迁后同时编译新旧实现；回归修前失败、修后通过。CPU/CUDA/双卡定向 41 passed，收尾 12 passed；CPU-only 4 passed/1 个 CUDA 节点跳过。实际执行 NaN checker、CPU erfinv、RingBuffer 与双卡流/拷贝。ACL 两 TU 主机语法及负向对照通过；ROCm 两 ABI blob/Corex basename 审计通过，未做 NPU/ROCm 实机。 |
| 2.15 | NanoString | 已合并 | bindings | 9d5ed413（索引位宽 7→8、static_assert 把表与字段绑住、`ns_check_registration` 在注册期查索引与名字长度；"dtype 表改运行期注册"那半未做，见提交说明） |
| 2.16 | 类型提升表 | 已合并 | bindings | d821c34a（int_dtype_promote 提升格；标量按 `_is_scalar` 标志认，不再按形状；float 标量把整数张量提到默认 float dtype）、a39a2f1c（补：双标量走提升格，交换左右操作数不再改变 dtype 与结果） |
| 2.17 | 算子身份用注册期整型 id | 已合并 | coreops | 1d792e16。OpInfo 注册分配 OpId，核心/tuner/pass 名字比较归零，fast_strcmp 删除，Tape 用显式 pending flag；CPU 80 项、CUDA 5 项及结构契约通过 |
| 2.18 | 算子注册表惰性初始化 | 已合并 | coreops | bca71d1f。注册表函数内惰性构造，typed polymorphic constructor 取代 `type_info + void*` 手工分派，ACL API/op_types 同步惰性；结构 3、C++ 注册 3、custom-op 2、GPU2 跨 so 1 项通过 |
| 2.19 | 错误分两档 | 待领 | | `b1cef650` 新增 `VarHolder::item` 多元素用户输入边界，`41878a9e` 将 `grad` 的 loss/target dtype 边界改为可捕获 `USER_CHECK`，`953462c7` 将 `code` vary-shape 边界分类为用户错误，`e90f6c5d` 将 `reindex` 空 shape 边界分类为用户错误；item 负向与结构合同 14 passed，grad 定向 11 passed，code 定向 2 passed，reindex 定向 23 passed；聚合任务其余调用点仍待领。ed12fe21 已合入析构半项；c119f3bf 迁 7 处公开维度边界；83754995 迁 code/numpy/reindex 共 10 处 shape/数量边界；7c018c86 迁 transpose/fuse_transpose/reshape 共 9 处视图形状边界；b32cd6df 迁 ternary 两处 shape/dim；32758304 迁 broadcast_to 三处用户 shape 边界；37a626bc 迁 reinterpret_view 六处 dtype/shape 用户边界；8a2aebab 迁 binary 一处 shape 用户边界；8e427d2c 迁 setitem 两处 data dim/shape 用户边界；97cf5e0e 迁 getitem 三处索引/shape 用户边界；c7e5306b 迁 py_converter bool slice 一处用户输入边界；7bab54f6 迁 device_copy 一处非法设备号用户边界；83c46ffc 迁 NumPy object dtype 一处用户输入边界；a6ad6585 迁 fused_adamw 四处 TensorList cardinality 用户边界；be7ef67a 迁 var_slices 一处字符串切片长度用户边界；02795d51 迁 set_data 两处 dtype/size 用户边界；a0dd9c44 迁 reuse_np_array 两处类型/C-contiguous 用户边界；58df26d0 迁 random 一处 type 用户边界；28cba3b9 迁 py_caller 一处返回字符串用户边界；6d816bcc 迁 unary 一处 op 语义用户边界；0e2a8483 迁 CUDA curand 两处 dtype/type 用户边界（静态前置）；3ee7669e 迁 CUDNN RNN descriptor 一处 dtype 用户边界（静态前置）；3d943240 迁 CUDNN RNN x/weight dtype 一处用户边界（静态前置）；6375a852 迁 Cutt transpose axes 两处用户边界（静态前置）；0bee930e 迁 CUBLAS matmul 两处 dtype 用户边界（静态前置）；1a4f0b27 迁 CUBLAS batched matmul 两处 dtype 用户边界（静态前置）；c3c437b5 迁 CUBLAS acc matmul 两处 dtype 用户边界（静态前置）；595dac8d 迁 cuSPARSE CSR 两处 dtype 用户边界（静态前置）；c1176841 迁 cuSPARSE COO 两处 dtype 用户边界（静态前置）；f98e5b80 迁 NCCL reduce-scatter 两处 shape 用户边界（静态前置）；ec2ee53c 迁 CUB cumsum 一处 rank 用户边界（静态前置）；45f77257 迁 CUB argsort/arg_reduce 两处 offsets dtype 用户边界（静态前置）；a3890dd9 迁 CUDNN conv forward 一处 format 用户边界（静态前置）；67e710b7 迁 CUDNN conv backward-x 一处 format 用户边界（静态前置）；0a0e820e 迁 CUDNN conv backward-w 一处 format 用户边界（静态前置）；37004fe0 迁 CUDNN conv3d 输入 rank 一处用户边界（静态前置）；85ae0688 迁 CUDNN conv3d 权重 rank 一处用户边界（静态前置）；9d77a5a7 迁 CUDNN conv3d backward-x 权重 rank 一处用户边界（静态前置）；496dd510 迁 CUDNN conv3d backward-x dy rank 一处用户边界（静态前置）；e81ef514 迁 CUDNN conv3d backward-w 输入 rank 一处用户边界（静态前置）；f07cb966 迁 CUDNN conv3d backward-w dy rank 一处用户边界（静态前置）；1910f343 迁 CUB argsort x/indexes rank 一处用户边界（静态前置）；4fe6f687 迁 CUB argsort indexes 维度 shape 一处用户边界（静态前置）；166010a8 迁 CUB argsort offsets rank 一处用户边界（静态前置）；36502b8e 迁 CUB argsort offsets 长度一处用户边界（静态前置）；193d5171 迁 CUB arg_reduce offsets rank 一处用户边界（静态前置）；4c64067e 迁 CUB arg_reduce offsets 长度一处用户边界（静态前置）；fbc69232 迁 CUDNN RNN LSTM mode 一处用户边界（静态前置）；b1c604af 迁 CUDNN RNN 非 LSTM mode 一处用户边界（静态前置）；6afd44df 迁 CUDNN RNN proj_size 一处用户边界（静态前置）；57c6cd92 迁 CUDNN RNN 第二处 proj_size 一处用户边界（静态前置）；aae2f5bc 迁 CUDNN conv3d 分组通道一处用户边界（静态前置）；935bb1a9 迁 CUDNN RNN backward-x LSTM mode 一处用户边界（静态前置）；b3826005 迁 CUDNN RNN backward-x proj_size 一处用户边界（静态前置）；35664df5 迁 CUDNN RNN backward-x 非 LSTM mode 一处用户边界（静态前置）；76dc9dc3 迁 CUDNN RNN backward-x 第二处 proj_size 一处用户边界（静态前置）；408b4832 迁 CUDNN conv 输入 rank 一处用户边界（静态前置）；ceabd84c 迁 CUDNN conv 权重 rank 一处用户边界（静态前置）；44a80c8a 迁 CUDNN conv 分组通道一处用户边界（静态前置）；92a66390 迁 CUDNN conv backward-x dy rank 一处用户边界（静态前置）；1e3bab6e 迁 CUDNN conv backward-w 输入 rank 一处用户边界（静态前置）；5596563f 迁 CUDNN conv backward-w dy rank 一处用户边界（静态前置）；241ab528 迁 CUDNN RNN 输入 rank 一处用户边界（静态前置）；4858b0a2 迁 CUDNN RNN 输入通道 shape 一处用户边界（静态前置）；7a24ca0b 迁 cuFFT dtype 一处用户边界（静态前置）；040e44a0 迁 CUBLAS matmul 输入 rank 一处用户边界（静态前置），累计 114 处。C++ 具体类型、Python 跨 pyjt 可捕获、析构/信号防绕过和六十七组结构计数均有聚焦证据；fused AdamW 构造期长度负向与其结构/TU 证据已记录；var_slices 本批结构计数、`getitem_op.cc` TU 语法与字符串 slice 负向节点通过；set_data 本批结构计数、`var_holder.cc` TU 语法与两个负向节点通过；reuse_np_array 本批结构计数、`py_array_op.cc` TU 语法与两个负向节点通过；random 本批结构计数、`random_op.cc` TU 语法与无效 type 负向节点通过；py_caller 本批结构计数、`py_caller.cc` TU 语法与非字符串返回负向节点通过；unary 本批结构计数、`unary_op.cc` TU 语法与非法 op 负向节点通过；curand 本批结构计数与现有 dtype 负向静态合同通过，`nvcc -c` TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN RNN dtype 本批结构计数、现有 bfloat16 负向静态合同与 descriptor 头 TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN RNN x/weight dtype 本批结构计数、混合 dtype 负向静态合同与 `cudnn_rnn_op.cc` TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；Cutt transpose axes 本批结构计数、两个 axes 负向静态合同与 nvcc TU 语法通过，但 **cuTT 后端在本树里不可达**（`setup_cutt()` 全树无调用点，`cutt_ops` 恒为 None，`tests/backends/cuda/test_cutt*.py` 六条恒 skip），该处负向至今一次也没跑过；CUBLAS matmul 本批结构计数、两个 dtype 负向静态合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUBLAS batched matmul 本批结构计数、两个 dtype 负向静态合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUBLAS acc matmul 本批结构计数、两个 dtype 负向静态合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；cuSPARSE CSR 本批结构计数、两个 dtype 负向静态合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；cuSPARSE COO 本批结构计数、两个 dtype 负向静态合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；NCCL reduce-scatter 本批结构计数与 shape 静态合同、nvcc TU 语法通过，本分区只分到一张卡，NCCL 负向仍未运行；CUB cumsum 本批结构计数、rank-3 负向静态合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUB argsort/arg_reduce offsets dtype 本批结构计数、两个 int64 offsets 负向静态合同与双 nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv format 本批结构计数、`Not a valid format` 静态合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv backward-x format 本批结构计数、`Not a valid format` 静态合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv backward-w format 本批结构计数、`Not a valid format` 静态合同与 nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv3d input rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv3d weight rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv3d backward-x weight rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv3d backward-x dy rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv3d backward-w input rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv3d backward-w dy rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUB argsort x/indexes rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUB argsort indexes shape 本批结构计数与 shape 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUB argsort offsets rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUB argsort offsets length 本批结构计数与长度静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUB arg_reduce offsets rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUB arg_reduce offsets length 本批结构计数与长度静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN RNN LSTM mode 本批结构计数与 mode 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN RNN non-LSTM mode 本批结构计数与 mode 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN RNN proj_size 本批结构计数与 proj_size 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN RNN 第二处 proj_size 本批结构计数与 proj_size 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv3d 分组通道本批结构计数与通道 shape 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN RNN backward-x LSTM mode 本批结构计数与 mode 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN RNN backward-x proj_size 本批结构计数与 proj_size 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN RNN backward-x non-LSTM mode 本批结构计数与 mode 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN RNN backward-x 第二处 proj_size 本批结构计数与 proj_size 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv input rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv weight rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv 分组通道本批结构计数与通道 shape 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv backward-x weight rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv backward-x dy rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv backward-w input rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN conv backward-w dy rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN RNN input rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUDNN RNN input channel shape 本批结构计数与 shape 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；cuFFT dtype 本批结构计数与不支持 dtype 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录；CUBLAS matmul input rank 本批结构计数与 rank 静态合同、nvcc TU 语法通过，本机 CUDA 可用，负向见 2.19 行末运行记录，未迁调用点分类仍待领；a29e0f81 记录 CUDA 后端内部断言分类文档与结构门禁；b2308f41 扩展计划失败断言清单；91718e98 补充 CUDNN RNN 权重查询断言；be526722 补充 CUDNN RNN bias 查询断言；27cc72f2 补充 CUB CUDA 状态断言门禁；6ef94518 补充 CUBLAS 测试入口状态断言清单；54e8d545 补充 CUDNN 测试入口状态断言清单；8595e479 收束无新增安全用户边界说明；8af5fd8d 精确约束 CUTT 返回码内断言；30bfdb6e 精确约束 CUDNN RNN descriptor 内断言；46ab1e17 精确约束 CUDNN 计划 ASSERT(ok) 计数；9f6afe17 精确约束 CUB 测试入口 ASSERT 计数；23d70b26 追加分类文档，剩余断言不作用户错误迁移。**2026-09-04 运行记录（首次在真实 CUDA 上跑 2.19 的验收，此前全部证据都是静态的）**：本机 nvcc 12.2.140、`cuda_archs=[89]`、`has_cuda=1`，此前 82 处「本机无 CUDA」的说法不成立，成因是读 `has_cuda` 时带了 `nvcc_path=""`（写法与判据见 skill `cuda-negative-path-verification`）。`pytest tests/backends/cuda -v -rs` 全目录：**180 passed / 23 failed / 37 skipped / 1 xfailed**（首轮跑到 36% 就 SIGABRT 退出，见下）。新增 `test_cuda_user_error_boundaries.py` **21 passed**，逐条覆盖此前只有静态证据的边界：cuDNN conv 输入/权重 rank、分组通道、非法 format；conv3d 输入/权重 rank、分组通道；RNN 输入 rank、输入通道 shape、proj_size；CUB argsort offsets dtype/indexes rank/indexes shape/offsets rank/offsets 长度；CUB arg_reduce offsets dtype/rank/长度；curand type 与 dtype；cuFFT dtype——每条都断言异常跨 pyjt 可被 Python 捕获、消息指名操作数，**且抛完之后运行时仍可算**。**发现一处迁移是错的**：`~VarHolder` 让异常逃出析构，因为析构隐式 `noexcept`，`std::terminate` 发生在析构自己的栈帧上，ed12fe21 给生成的 `tp_dealloc` 包的那层 catch 在它下面、永远轮不到；真实后果是整个 `tests/backends/cuda` 跑到 `test_cudnn_rnn_dtype.py` 就 SIGABRT，后面 21 个文件一个没跑而 pytest 连汇总行都没有。结构门禁 `test_destructor_and_handler_contract.py` 抓不到，因为它只扫析构体里**字面出现**的抛出宏，这里是经由 `release_both_liveness` 传递抛出的。已修（见本行提交），并补 `test_var_holder_teardown.py` 2 passed（修前 1 failed/rc=134）。另更正 `test_backend_teardown.py` 的探针：它用 `y.sync()` 后 `except: pass`，而 `Var.sync` 不做 device sync，故障根本没被报出来，「真实故障盖过清理噪音」这条断言在真机上一直是假的——改用 `jt.sync_all(True)` 之后 `cudaErrorIllegalAddress` 真的先于 teardown 错误出现，2 passed。**仍缺**：cuTT 六条恒 skip（后端不可达）；NCCL reduce-scatter 负向要多卡；`test_device_copy.py` 7 条、`test_device_methods.py` 4 条等 27 条要两张卡，本分区只分到一张。**2026-09-06 收口（三项）**：（1）cuTT 六条从「恒 skip」变成真跑，`4bdc7e797` 补上 `setup_cutt` 的调用点（9.01 漏项，见杂项行）——打开后暴露一处**静默算错**：`transpose`/`fuse_transpose`/`cutt_transpose` 的构造函数把「axes[i]==i 全部成立」当成恒等置换而**不比较秩**，`axes=[0]` 或 `[0,1,2]` 作用在二维输入上直接 `forward(x)` 返回未转置的原张量，`infer_shape` 里 `6375a852` 迁的那两条 `USER_CHECK` 根本不可达；修前 3 failed（回退三个 `.cc` 实测）、修后 `tests/ops/test_transpose_op.py` 21 passed 与 cuTT 整文件 9 passed。（2）**验收「析构与信号处理器里 grep LOGf 为 0」已达成**：全树 612 个 C++ 文件、110 个析构体，字面 `LOGf` 0 处、其他抛出宏 0 处，一跳外可达 `LOGf` 的 0 处（唯一命中 `~AsyncQueue → lock()` 是 `std::unique_lock<std::mutex> lock(mutex)` 变量声明的误报）；`segfault_sigaction` 与 Windows 的 `handle_signal` 字面与一跳均为 0。**但原门禁的扫描根只有 `python/jittor/{src,extern}`，后端搬进 `backends/` 之后 CUDA/ACL/ROCm 的析构一条都没扫到**，`4bdc7e797` 补上该根与 `.cu/.cuh`，并把断言从「总数 > 50」改成「每个根目录都非空」。全树 `LOGf` 实际 161 处（含测试与宏定义），不是 8 处；计划正文的「62 处」是旧口径。（3）剩余用户边界（`7faf209f3`）：`py_converter.h` 看着像一大批未迁边界，实际**不是**——`from_py_object` 只在生成的参数解析器用 `is_type<T>` 认可之后才跑，坏输入更早就被解析器以可捕获 `RuntimeError`（指名算子与实参类型）挡掉了，里面的 `CHECK(is_type<...>)` 只在同一份生成代码的两半自相矛盾时才触发，属内部不变量，已写进 `docs/testing/error-categories.md` 并加结构门禁钉住。真正的用户边界是**从绑定里返回来**、没有任何解析器看过的值：`GradCallback` 收到用户 `Function.grad` 的返回值，梯度个数不对或不是 Var 三处已迁 `USER_CHECK`，负向 2 passed。**本波三项里前两项已收口，第三项仍待领**，剩余清单：全树 `ASSERT`／`ASSERTop`／`CHECK`／`CHECKop` 四个宏（词边界匹配，不含 `USER_`／`INTERNAL_` 前缀）仍有 986 处（含 `src/tests/*.cc` 的 ~380 处自测断言，那些天然是内部档），未逐条归类的主要聚集在 `op_compiler.cc` 51、`opt/kernel_ir.cc` 47、`utils/cache_compile.cc` 38、`opt/expr.cc` 22、`ops/op_register.cc` 22、`mem/swap.cc` 12——这些都在 JIT 代码生成与编译缓存路径上，**先验判断是内部不变量，但没有逐条走过公开实参可达性**，需要的正是本波用的方法：构造 Python 侧输入实测哪些能到达。多卡门控仍缺：NCCL reduce-scatter 负向、`test_device_copy.py` 7 条、`test_device_methods.py` 4 条等 27 条要两张卡 |
| 2.20 | 信号处理器只做 `write` 与 `_exit`，符号化交给预建 helper 进程 | 已合并 | bindings | 上半 9b92f38d（去 stdio/LOGf/exit，标志改 volatile sig_atomic_t）；下半 640a4f07（符号化搬进崩溃前 fork 的 helper，经父进程 /proc/<pid>/maps 解析）；d874b01d 修 jit_key 用例（它原先靠信号处理器抛异常） |
| 2.21 | `DEFINE_FLAG_WITH_SETTER` 先赋值再调 setter，签名收新旧两值 | 已合并 | coreops | 14336afd |
| 2.22 | 环境变量统一 `JT_` 前缀 | 已合并 | pyops | `4751ce240` 机制（`jittor_utils/env_config.py` 唯一读取点 + `env_manifest.py` 生成器）、`972adc722` C++ 侧、`66a7727f2` 启动摘要与弃用告警、`1a4c9bebb` 调用点、`7e58c3216` 门禁与反例、`043d19be2` 文档与审计。两个命名空间：`JT_BUILD_<NAME>` 决定编译产物并进缓存指纹，`JT_<NAME>` 是运行期原生 flag，**分界线不新发明**，直接用 `2.13` 的 `flag_policy.STARTUP_FLAGS`（一份名单三个消费者）；C++ 侧因为不能 import Python 必须重抄一份，结构门禁读两边断言集合相同。旧的无前缀小写名**全部仍然生效**（`nvcc_path=`、`use_mpi=`、`log_v=` 等 gate 与文档在用的写法一个都不断），但会被记录并在启动时打成一行摘要加一次 `DeprecationWarning`，替代掉原来「每个 flag 一行 `LOGi`，而默认 `log_v=0` 与 `log_silent` 都把它吞掉」的报告方式——实测这条已在真机生效：带 `nvcc_path=/usr/local/cuda/bin/nvcc` 导入打出 `environment set 1 setting(s): nvcc_path=... [deprecated name]` 与 `nvcc_path -> JT_BUILD_NVCC_PATH`。**验收「shell 里导出的 `name`/`debug` 不再改变框架行为」已达成**，规则是机械的：**名字里没有 `_` 的设置只从带前缀的名字读**，无前缀形式不是弃用而是忽略并告知换成什么。**三处实测修正**（写进 `codebase-audit/04-build-tooling.md`，不删原条目）：（1）**`name` 从来不是 flag**，原条目那份「76 个可设 flag」名单来自把 `log.h` 的 `DEFINE_FLAG(type, name, default, doc)` 宏形参当成了定义，同一个正则也把 `#ifdef TEST_LOG` 里的 `nthread` 算了进去，实测（rebase 后）`dir(jt.flags)` 89 项、其中原生 flag 84 个、**无 `name` 无 `nthread`**；（2）真正能被普通英文单词命中的只有 `debug` 一个；（3）`cc_flags` 的「追加对替换」矛盾里，核心那次「替换」其实是死代码——`compiler.py` 本来就会把这 9 个构建 flag 整体覆写——却看着像活的，现在核心对这 9 个（`compiler_owned_flag_names`）完全不读环境，`JT_BUILD_CC_FLAGS` 只有一个读者、语义只有「追加」，门禁断言这 9 个名字恰好等于 `compiler.py` 里 `flags.X =` 赋的那一组。清单 `python -m jittor_utils.env_manifest` 实测 104 项（84 个原生 flag 从 `DEFINE_FLAG` 现场扫、20 项 Python 侧设置来自 `env_config` 的表；原生那部分随主线变动，本波内就从 82 漂到 84，所以门禁断的是集合相等而不是数量）；扫描两个坑写进代码与 skill：跳过 `log.h` 自身，以及 `#ifdef TEST_LOG` 必须**按嵌套层级**剔除而不能从第一次出现处截断（`log.cc` 在 `log_v`/`log_silent`/`log_sync`/`log_file`/`log_vprefix` 定义**上方** 17 行就有一个，一截断就把框架里最常被设置的五个 flag 从「完整清单」里删掉，而清单看起来一样完整）。**门禁反例两个方向都跑过**：往 `dataset.py` 插一行 `os.environ.get("nvcc_path")` → 1 failed 且失败用例名里带那个文件；把扫描根指到不存在的目录 → 6 failed（含 `test_every_scan_root_matches_the_tree` 与哨兵参数 `[<no python module matched the scan root>]`）。**扫描根非空是本波新加的**，因为这正是清单类门禁最典型的坏法且坏在往「全过」的方向：根一失配 `rglob` 什么都不产出、参数化规则展开成**零个用例**，「什么都没检查」报出和「全都检查过了」一样的绿；断言写成**每个根各自非空、各用一个必须包含的文件**，**故意不写成「总数多于 N」**（一个总数在「两个根里一个空了、另一个大得足以兜住」时照样通过）。三套门禁与基线对拍零回归：torch shim `tests/structure` **15 failed / 1199 passed**（基线 15 failed / 887 passed，15 条 FAILED 逐条相同、全部与环境变量无关，passed 增量含本任务新增 307）；CUDA `tests/backends/cuda` **5 failed / 269 passed / 35 skipped**（与基线 5 failed / 269 passed 相同）；新增 `tests/core/test_env_var_namespaces.py` 与 `tests/compiler/test_lock.py` 共 **20 passed**（子进程实测 `JT_BUILD_DEBUG=1` 改缓存目录、裸 `debug=1` 被忽略且报告为忽略、旧名 `log_v=1` 仍生效且恰好告警一次并写出 `JT_LOG_V`、`JT_CC_FLAGS` 走错前缀无效而 `JT_BUILD_CC_FLAGS` 有效）。CUDA 冒烟：`import jittor` + matmul，`has_cuda=1`、`Found device architectures: [89,]`、nvcc 12.2.140。**未做（不属本项验收，是开放面）**：大写无前缀的一组（`CUTT_PATH`、`DISABLE_MULTIPROCESSING`、`FIX_TORCH_ERROR`、`SKEY`、`JTCUDA*`）与 `JITTOR_*` 前缀的约 40 项只进了清单、没有改名；`import` 期反写环境变量与 `cuda_arch` 死代码属 `9.07` |
| 2.23 | 布局收尾 | 待领 | | 2.13/2.14 已完成 runtime 状态与 misc 支持文件归位，src/misc 已消失；init/profiler/lock 和 pyjt/pybind 合并到 bindings 尚未完成，不能仅凭 misc 消失关闭本项。 |
| 2.24 | `custom_data` 的最后一个用户：FusedOp 跨阶段 var 索引 | 已合并 | coreops | `83c26d42`：FusedOp 建立显式 `Op*`/`Var*` index map，update/load/relay 共用映射；移除 Node::custom_data。结构 4 passed，fused 聚焦 2 passed。 |
| 2.25 | 反向可达叶子查询（`is_leaf`/`grad_fn` 的内核答案） | 已合并 | coreops | `c6e62ba1`（查询与内核用例）、`781d4188`（与真 PyTorch 的逐例对拍）。2026-09-04 由 `7.11`／`7.12` 的共同前置派生。一条查询 `backward_grad_fn(Var*)`（`grad.h`）四种拼写（`Var.is_backward_leaf`、`grad_fn_node_id`、`grad_fn_op_id` 用 2.17 的注册期 id、`grad_fn_name` 仅诊断）；语义是 requires_grad 与「生产者有一条能带梯度的入边」的合取，四条过滤器与 `grad()` 的 `bfs_backward` 同源（两个 requires_grad 标志、生产者自身 stop_grad、控制依赖边、`Op::init` 冻结的 disabled 边）。**O(生产者入度)，不遍历、不缓存、不引入进程级 id 键字典**；查询前后 `tflag_count` 不变的用例把「没有遍历」钉住，另一条在未结束的 `TraversalEpoch` 里查询证明 2.03 的机制没被动。修前 20 failed → 修后 20 passed；定向 CPU 208 passed 对基线 188 passed（同 4 条既有失败，零回归），CUDA 73 passed；与真 PyTorch 2.12.1 的 19 个用例 16 个三元组全等、2 个只差 `requires_grad`、1 个的形状差异由 `requires_grad` 传导（在 `EXPLICIT_REQUIRES_GRAD` 策略下同样全等）。**`7.11` 的接线未做**，`compat/**` 属兼容层分区 |
| 3.01 | `Executor::run_sync` | 已合并 | coreops | **协调者判定（2026-09-06）：结构拆分即交付，计划缓存那一项按实测否决，故判已合并而非待领。** 计划的验收有两句：「`executor.h` 不再有流水线补丁状态」**已满足并有结构断言钉住**；「UNet 每步执行器 CPU 时间由约 16 ms 降到计划命中后的发射成本」的**前提被实测推翻**——那 16 ms 里没有规划，是 1.94 ms 发射加 9.78 ms 等设备，规划总共 0.173 ms 且换一档规模仍是 0.17 ms（是常数不是比例），缓存最多省 0.128 ms（0.9%），代价是结构哈希必须覆盖 `count_fuse` 会分支的每个字段、漏一个就是静默的错误融合。按计划第 2 节对「实测否决」的既有处置（如 0.16、系统 A2）记为不采用。**被它挡的 3.02／3.03／3.04／3.05／3.07／3.20／5.02／10.17／10.18／11.04 十条前置至此解除**；下一步方向是 phase 6 的 per-op 发射常数与 phase 7 那 9.78 ms 的 GIL 释放（即 3.07，本波已量出规模）。以下为执行者原记录。 **结构拆分已落地，性能验收未达，故保持待领。** `8e1ad4bd` executor.h 写成显式契约并给 run_sync 标出七阶段（同时是 11.04 在 executor.h 上的那一份）；`33c10331` 抽出 Planner（`exec_plan.{h,cc}` 的 `build_exec_plan` → 值类型 `ExecPlan`：融合划分、段执行序、段内算子序，除 ops/all_vars 外全是下标）；`fa2d8523` 抽出 Runner（`exec_runner.{h,cc}` 的 `run_exec_plan`：分配、迁移、发射、释放 liveness、等设备），两半各一个翻译单元；`029fa9a4` 把 `last_run_ops`/`flush_active` 移出 `Executor` 到 `runtime/submission_pipeline.h` 的 `SubmissionPipeline`（由 NativeRuntime 持有），**验收里「executor.h 不再有流水线补丁状态」这条已满足**，并加结构断言钉住。run_sync 579 → 39 行，executor.cc 945 → 295 行。**行为逐字不变**，两类证据：(a) 搬过去的两段与 803e37853 原文归一化后逐字相同（脚本判 IDENTICAL；第一轮即抓到 BFS 里漏掉的 `weak_sync` 短路条件，该分支只在 submit_pending 自动冲刷路径走到，CPU 门禁一条都不会红）；(b) 三套门禁与 803e37853 逐条对拍，FAILED 清单**逐字相同**：原生 CPU `tests/core` 串行跑四轮（基线 2 轮、改后 2 轮）四份 FAILED 清单 md5 完全一致，均 21 failed/643 passed/113 skipped/1 xfailed；`tests/ops` 21 failed/257 passed/227 skipped 相同；torch shim `tests/structure` 19 failed 清单相同，passed 866→867（多的那条是本任务新增的结构断言）；CUDA `tests/backends/cuda` 42 failed/220 passed/41 skipped/1 xfailed，42 条 FAILED 逐条相同。**注意**：`test_fuse_memopt`/`test_var_holder`/`test_no_grad` 断言的存活 Var 数受同进程前序影响，**并发跑会飘**（同一份源码两次并发整目录跑实测一次 9 一次 7、`test_fuse_memopt` 一次红一次绿），A/B 必须串行，判据用 FAILED 集合不用断言里的数字。**性能验收未达且判定其前提有误**：验收写「UNet 每步执行器 CPU 时间由约 16 ms 降到计划命中后的发射成本」，而按阶段实测（diffusion UNet，CUDA，batch=16 res=128，每步 7 次 run_sync，15 步中位数）整步 14.21 ms 的构成是**等设备 9.78 ms + 发射 1.94 ms + 规划 0.173 ms**（其中可缓存的融合划分与两次拓扑排序只占 0.128 ms，phase 2 的 BFS 0.045 ms 无论如何都要走）。规划开销与张量大小无关（batch=8 res=64 一档同样 0.17 ms，整步 4.41 ms），是常数不是比例。**所以计划缓存的收益上界是每步 0.128 ms（整步 0.9%）**，代价是一个必须覆盖 count_fuse 会分支的每一个字段（_stop_fuse/_force_fuse/_out_hint/num/dtype/shape/消费者个数/边的下标与兄弟次序）的结构哈希，漏一个就是静默的错误融合，而算哈希本身又是一次 O(算子+边) 的遍历。**判定不做，理由与完整数字见 codebase-audit/07-architecture.md §核心抽象末尾**。重构本身性能中性：同口径 HEAD 4.46/14.20 ms 对改后 4.41/14.30 ms。**CUDA 基线的取法记一笔**：本机冷 CUDA 构建必然失败（`dnnl`/`cub`/`cutt` 三个三方包要联网），唯一能带 CUDA 构建的是预热过的 `$JITTOR_HOME`，所以基线是把工作树临时切回 803e37853 源码、复用同一个 JITTOR_HOME 跑出来的（还原写在 trap 里）。想做 CUDA 对拍的人别再去建新缓存目录。沉淀 skill `pure-code-motion-refactor`，并给 `jittor-core-planning-cost` 加 §4bis（run_sync 阶段分解与临时 flag 计数器的量法）。**剩余**：计划缓存（判定不做，若要重开需先推翻上面的数字）；真正值钱的是 phase 6 的 per-op 发射常数与 phase 7 的 GIL 释放（3.07） |
| 3.02 | jit key 结构化 | 已合并 | coreops | `9cdae20fc`。三个子项：**(1) 2 MB 无检查缓冲与 mprotect 守护页换掉**——`JitKey::buffer` 改堆分配、从 8 KB 按倍数增长，四个真正动内存的写入口（`jk_put_str_with_len`、`operator<<(const string&)`、`operator<<(char)`、`operator<<(const NanoString&)`）各调 `reserve()`，超过新 flag `jit_key_max_size`（默认仍是 2 MB，所以今天装得下的键一个都不会装不下）抛 `UserError`，到 Python 是可捕获 `RuntimeError`；`utils/log.cc` 的 `protected_page` 与信号处理器里认那个页的分支删除。常态内存从每线程 2 MB 降到 8 KB。**(2) 融合边变长编码：核实为已完成，本次未改**——6.C05（`21a4f4fc`）已把 `hex2/hex1/hex2/hex1` 换成 `hex(i).hex(j).hex(k).hex(l),`，`13ac1d14` 的四条临时 `ASSERT(<256)` 在那个提交里就已删除，现在树里没有任何融合规模上限断言。**(3) 以 `_` 开头的 loop option 也入键**——`«choices:` 不再跳过它们；计划另给的「设置时拒绝」单独不够（tuner 直接写 `loop_options_tuned`，不经任何 setter）。**验收两句**：「含 300 个算子的融合段与 200 个的键不同」用 `tests/compiler/test_jit_key_structure.py` 从执行器日志里取真实 key 比对（比的是缓存真正用的东西），**但要注意这条抓不到定长编码**——把 `hex2/hex1` 放回去重编，4 条用例仍全过，因为每个算子都有一条变长的 `«opkey<i>`，算子数不同的两个融合段无论边怎么编码都不同键；所以同文件另加了强形式「同算子数同 var 数、链 vs 平衡树（n=300/512）不得同键」，那才是回绕会碰到的形状，回绕本身的修前失败证据在 6.C05。「溢出可捕获」修前是**杀进程**：在 `f5a0ec5a1` 副本树上跑 `jt.tests.jit_key_guard_page()` 实测 signal 11 + "Accessing protect pages" + exit 1，修后同一条件是 `jit_key_overflow`（本进程 `UserError`，`size` 不半推进，JK 事后可用）与 Python 侧 `pytest.raises(RuntimeError, "jit key too long")`；`CRASHING_TESTS` 因此空掉。下划线那一项有确定性的一对：滤波器放回去 1 failed / 3 passed，换回来 4 passed。**三套门禁与基线 FAILED 集合逐条相同**（core 21f/643p、ops 21f/267p、compiler 18f/376→381p、shim 15f/884p、CUDA 42f/231p），compiler 的 +5 全部是本任务新增。踩到的坑写进 skill `jittor-core-cpp-edit-loop` §7ter：限额是 runtime flag 时，比较阈值必须每次 `clear()` 重读，只在 `grow()` 里重算会让「把 flag 调小」在缓冲区已经变大的进程里完全失效 |
| 3.03 | 三张 kernel 缓存表键改 `string` | 已合并 | coreops | `0ad6c7504`。`utils/string_view_map.h` → `utils/jit_cache_map.h`（类型同名改，一个键是 `std::string` 而名字叫 `string_view_map` 的类型正是要消掉的东西）：键改自有 `string`，每条一个 relaxed atomic 时间戳、满了按最久未用淘汰，容量由新 flag `jit_cache_size`（默认 4096）给。**LRU 不用侵入式链表**是因为「记录一次使用」是写操作，而 `parallel_compiler.cc` 的错误路径经 `Op::get_filename_from_jit_key` 在编译工作线程里无锁查 `jit_key_mapper`。`find` 改返回 `T*`，八个调用点跟着改；**三处 `t[a] = t[b] = v` 全部拆成两句**——表能淘汰之后这个写法是 UB（C++17 起右操作数先求值，内层 `operator[]` 的引用在外层可能已 erase 之后才被读）。`jit_fused_ops` 值改 `shared_ptr<FusedOpContext>`（一份 context 挂在两个键上，淘汰其一不能释放），执行中的 `FusedOp` 另持一份 `context_owner`，所以运行中的 kernel 不会被淘汰掉脚下的 context。`VarRelayManager::fop` 删除，`add_relay_group`/`get_op_relay_info` 改显式传 `FusedOp*`（四个调用点），`execute_fused_prepared` 里那句 `vrm.fop = this` 一并删除——**这一条没有可达后果**：那句重指向让悬垂的 `fop` 在被读之前总是先被覆盖，改的是「正确性依赖一句看不出用意的赋值」本身。**「ASan 无悬垂」跑了真 ASan**：`jit_cache_map.h` 除 `common.h` 与一个 `DECLARE_FLAG` 外无依赖，可单独编译（`-I python/jittor/src -I <python include>`，不需要 cfg 目录），`tests/compiler/test_jit_cache_map_asan.py` 两条——现在这张表 4096 个短键无任何 ASan 报告、`missing=0`；**同文件再编一份 `f5a0ec5a1` 的 `string_view_map` 原文，断言 ASan 确实报它**（`heap-use-after-free`，栈顶 `string_view_map<int>::find`，释放点 `vector<string>::_M_realloc_insert`），这是给第一条装的牙齿。**不用 sanitizer 也是错的**：旧代码 `-O2` 实测 4096 个短键里 2017 个查不回来，所以 `src/tests/test_jit_cache_map.cc` 一条普通 C++ 用例就能独立扛住这条回归。未做的是整个核心带 ASan 重编跑门禁（缺陷完全在一个头文件里，单编那份跑的是同一份代码）。端到端淘汰路径靠 `tests/compiler/test_jit_cache_bound.py`（`jit_cache_size=4` 跑 37 个形状 3 遍，串行/并行编译两条填表路径各一条，另加 matmul 走 relay）。**三套门禁 FAILED 集合与基线逐条相同**（core 21f/647p、ops 21f/267p、shim 15f/885p、CUDA 42f/233p）；`tests/compiler` 在复用缓存里多一条 `test_cache_dependencies::test_no_dependency_is_recorded_twice_with_different_hashes`（报 `op.h` 两个哈希），换空 `JITTOR_HOME` 重跑 **18f/391p，清单与基线逐字相同**，该文件单跑 3 passed/1 skipped——热缓存假失败，成因是本提交改了一个被广泛 include 的头文件而那个缓存目录里同时留着改前改后的条目 |
| 3.04 | 求 jit key 改纯函数 | 待领 | | 前置 3.01 结构拆分已落地。「后端选择结果放执行计划」现在有地方放了：`ExecPlan`（`exec_plan.h`）是个值类型，加一个 per-segment 的后端字段不再需要改 `run_sync` 的控制流。原 `executor.cc:685` 的错误路径二次 do_prepare 现在在 `exec_runner.cc` 的 catch 块 |
| 3.05 | 删除算子构造期回调执行器 | 待领 | | 前置 3.01 结构拆分已落地。嵌套 run_sync 的代价现在写进了 `executor.h` 的契约（不可重入、嵌套批共享 `last_is_cuda`），`ExecPlan` 持有的 epoch 与释放时机也写明了 |
| 3.06 | 并行编译器修到可信 | 已合并 | coreops | e46ba4ec。持久线程池改 per-call future 并完整 get，异常以 exception_ptr 回主线程；任务显式携带原 JIT key，relay 成功后才发布；worker 数受 affinity/cgroup v2 限制，fork 后不再遗留幽灵线程，catch 不二次 prepare。fork 用例修前稳定退出 124；worker 上限、fork、prepare-once、失败归因 4 个 CPU 聚焦节点通过。长期 sanitizer/共享缓存压力仍按 KI-COMPILER-001 复查 |
| 3.07 | 执行器在设备等待段释放 GIL | 待领 | | 前置 3.01 结构拆分已落地：设备等待段现在集中在 `exec_runner.cc` 的 phase 7（`sync_devices`），是一段连续代码而不是散在 579 行里。**3.01 顺带量出这段的规模：UNet 每步 9.78 ms，整步 14.21 ms 里最大的一块，且期间 CPU 纯等**——这是全表最值得做的一条 **协调者勘察（2026-09-07，未动代码）：机制已有，但直接套会开出一个可重入窗口。**(1) 不需要新引入 GIL 机制——`parallel_compiler.cc:50` 已有 `GILReleaseScope`（`PyEval_SaveThread`/`RestoreThread`，`Py_IsInitialized()` 守卫），第 318 行在等编译 future 时就用它，注释里记着为什么必须放：worker 会经 `py_caller()` 用 `PyGILState_Ensure` 取 GIL，主线程不放就死锁。把它提到一个头文件复用即可。**顺带一个缺陷**：它只守 `Py_IsInitialized()`，没守`PyGILState_Check()`——在不持 GIL 的线程上调 `PyEval_SaveThread()` 是 fatal，并行编译那处的调用者恒持 GIL 所以没暴露,提取时应补上这个守卫。(2) **真正的难点**：`executor.h` 的契约写明 `run_sync` **不可重入**（嵌套批共享 `last_is_cuda`）。在 `exec_runner.cc` phase 7 的 `sync_devices` 处放掉 GIL，就允许另一个 Python 线程在此期间进入执行器——而这正是契约禁止的。所以本条必须同时给执行器加入口互斥，否则换来的是一个数据竞争。(3) **加锁有锁序反转**：A 持互斥、放掉 GIL、结束时要回收 GIL；B 持 GIL、等互斥 → B 抱着 GIL 阻塞 → A 拿不回 GIL → 死锁。可行解是让入口互斥**在放掉 GIL 的状态下获取**（一个 RAII：放 GIL → 上锁 → 取回 GIL），这样 B 在等锁时不持 GIL。还要考虑执行器是否可能从不持 GIL 的 C++ 线程进入（fetcher 回调、`event_queue`），那种情形要跳过 GIL 动作但仍然上锁。(4) **`event_queue.flush()` 必须留在释放区之外**——它会跑可能触碰 Python 对象的回调。(5) 验收「另一个 Python 线程在 sync 期间能运行」是功能性的、**不受本机性能测量不可信的限制**（见交接 §6quater）：计数线程 + 长 sync，断言计数在 sync 期间推进；修前应为 0 附近。另需一条并发进入执行器的压力测试证明互斥没把不可重入契约破掉 |
| 3.08 | KernelIR 结构化 | 已合并 | codegen | 属性名 `62cdee84`、节点类型 enum `78410930`、pass 契约 `c3557a1c` |
| 3.09 | 死代码消除不再按「语句含 `void` 一词」删除 | 已合并 | codegen | 66e5a153 |
| 3.10 | 算子内标识符改名走结构化成员表并先做合法性校验，替代三个硬编码白名单与 `op{i}_` 盲目前… | 已合并 | codegen | 864fa52c |
| 3.11 | 生成源码里的结构体字节偏移改显式 setter，成员表用宏声明 | 已合并 | codegen | f32d3c83 |
| 3.12 | `float_atomic_fix_pass.cc:76-80`、`fake_main_pass… | 已合并 | codegen | `1ea90057`；CPU 定向 9 passed，CUDA 定向 3 passed，结构门禁 310 passed / 2 skipped |
| 3.13 | 循环维度身份用整数向量，`range10` 不再被拆成 `range1*range0` | 已合并 | codegen | bd5b5a67，用例修正 e1717fe5 |
| 3.14 | 两个同名 pass | 已合并 | codegen | 6c899325 |
| 3.15 | 一次编译只解析一遍 | 已合并 | codegen | 97cac22f；“只解析一遍”实测不值得，见下方结论行 |
| 3.16 | `token_replace_all` 不再用 CHECK 抛异常做循环终止 | 已合并 | codegen | 9aa683c4。显式终止批量替换，非法模板不再被正常路径异常吞掉；新增 2、C++ TEST 46、CPU op_compiler 5、真实 CUDA GPU3 1 项通过 |
| 3.17 | 只用于代码生成的 JIT 区段与普通 C++ 分离 | 已合并 | codegen | 2aa190af。KernelIR 逃逸 `_Pragma`，字符串/格式串保持原样，生成源码用 `#line` 指回原算子；968ae198 修复重复 `#line` 行号被登记成同名 IR scope 符号的回归，C++ TEST 11、scalar gradient 与 compiler contract 聚焦通过；未改 `jit_compiler.cc` |
| 3.18 | 删掉 `asm_tuner` 链路 | 已合并（验收数字按实测否决） | pyother | 链路删除由 `cb853074`、`acfed956` 完成，本波 `<3.18>` 复测并收口 movnt。**验收「冷编译下降 ≥ 50%」按实测否决**：把同一条真实 kernel 编译命令直接交给 g++、与包成 `python asm_tuner.py --cc_path=…` 各跑 8 对、共 3 轮（先各跑一次不计时预热，两种形态交替先后以抵消负载漂移），直接 0.808–0.853s、包装后 1.013–1.055s，**降幅 18.1–23.1%（中位约 20%）**。审计原文「冷启动 kernel 编译成本 2–3 倍」实测为 **1.25 倍**，≥50% 是从这个错误倍数推导出来的，删链路拿不到。**movnt 一并删除，另有实测依据**：`UseMovntPass` 此前要求 `cc_type=="clang"`，本机默认 g++，`objdump -d` 生成的 `.so` 得 **0 条 movnt**——即默认工具链上它早已完全不生效。写 256 MiB 只写不读实测：普通 store 15.0 GB/s；`_mm256_stream_ps` 整循环流式 25.3 GB/s（+69%，说明这条优化本身有价值）；但**逐元素**形态（pass 唯一能表达的形态）clang `__builtin_nontemporal_store` 14.6 GB/s（打平）、g++ `_mm_stream_si32` 9.1 GB/s（**比普通 store 慢 40%**），两个编译器都不会把逐元素非临时 store 重新向量化。故删 `use_movnt_pass.{cc,h}`、`pass_manager` 注册与 `broadcast_tuner` 的 `use_movnt` 候选；要拿回 +69% 需要循环级流式变换，不是语句级改写 |
| 3.19 | `event_queue` 异步基础设施修好并加测试，或删除 | 已合并 | coreops | `5248870d` 删除不可达 `run_sync`、状态常量、volatile 状态、worker_caller 和无用条件变量；executor/NCCL 调用清理。结构合同 2 passed，C++14 CUDA TU syntax check 通过。 |
| 3.20 | 执行器提供「提交部分图」显式接口，`jt.grad` 与 `Function` 回调降开销，让反… | 待领 | | 前置 3.01 结构拆分已落地。「提交部分图」现在可以表达成「构造一个 `ExecPlan` 再交给 `run_exec_plan`」，不必再往 run_sync 里塞标志；提交时机的状态也已独立成 `SubmissionPipeline`（`029fa9a4`） |
| 3.21 | 每算子建图成本 | 待领 | | **前置 3.02 已解除**（`9cdae20fc`）：jit key 现在是按需增长、有长度检查的缓冲，`JitKey::size_limit()`/`jit_key_max_size` 把上限变成一个可读的量，键构造不再靠「反正有 2 MB」这个假设；量每算子的建图成本时，`clear()`/`reserve()` 是两个可以直接下探针的点 |
| 3.22 | CUDA 归约块内树形归约 | 待领 | | edf70f52 已合入 1/N：level-4 SharedReduce 改为 warp shuffle→每 warp 一个共享值→首 warp shuffle，两次 barrier；修复其对 ParallelPass 子节点位置与 `op` 字符串的旧假设。**`9ab5ea42` 已合入 2/N（量法与用例），本行所在的收口提交给出判定：口径对齐后验收未达成，`edf70f52` 定为 opt-in 不采用为默认。** ① **口径**：3.23 记的「Jittor 0.57 ms 对 PyTorch 1.20 ms、快一倍以上」不成立，两个桶几乎不相交——Jittor 的 0.57 只有代码生成器的归约（0.49 通用求和 + 0.08 六个注意力 GroupNorm 的回退），手写 GroupNorm 1.72、手写卷积偏置梯度 0.26、手写 softmax 1.59 都不在内；PyTorch 的 1.20 是 0.65 真归约 + 0.47 被误归类的 GroupNorm 逐元素仿射写回，而它真正的 GroupNorm 统计量归约 0.74 落在 `other`。用 `profile_step_torch.py --attribute`（新增）把每个 kernel 接回 aten 算子栈后按语义配对，**每步调用次数三行全对**（卷积偏置梯度 51:51、其余通用求和 67:67、GroupNorm 41:41）。② **对齐后**（每边两到三次运行取范围，profiler 的逐算子测量对机器负载敏感，单次会高估 12%）：通用求和 Jittor **744.9–753.2**（nsys ~853）对 PyTorch **652.6–681.9** us，GroupNorm **1532.9–1543.8**（nsys ~1850）对 **1275.5–1315.6** us，**合计 2279–2297（nsys ~2705）对 1928–1998 us，Jittor 慢 15%（profiler 口径）到 36%（nsys 口径）**；同次整步 21.19–21.35（profiler）/ 23.09（nsys）对 21.11–21.95 ms。两种量法在手写 kernel 上系统差一截但方向与归因一致，严格同口径的一对是 nsys 对 CUPTI。**验收「不慢于 PyTorch 的 1.13 ms」未达成**，PyTorch 的实测对应值是 1.93–2.00 ms 而不是 1.13/1.20。**差距 75%（两种量法算出的占比都是 75%）在 GroupNorm**，其中 1.5–1.8 ms 在手写 CUDA，见上方独立发现行。Jittor 手写 attention softmax 1.58（profiler）/ 1.41（nsys）ms 在 PyTorch 侧无独立 kernel（融进 `fmha_cutlassF/B`），单列不进合计。③ **`edf70f52` 的处置**：「四形状慢 1.64%」的判据被推翻——那四个不是 UNet 用的形状。真实 UNet 上 `reduce` 角色 lvl4 三次 527.9 / 517.8 / 514.3（均值 **520.0**）对默认 warp 五次 568.3 / 566.7 / 571.5 / 560.3 / 562.4（均值 **565.8**）us，**快 8.1%**；只看通用求和是 486.0 → 436.4 us（快 10.2%）；逐形状 best-of-30（`reduce_ab.py --shapes unet`）0.968，同向。仍不改默认：约 45 us 只有整步 0.2%、320–740 us 差距的 6%–14%，够不到验收；`--shapes representative` 那四个形状它仍慢到 1.39 倍；`para_opt_level=4` 是同时改 AtomicTunerPass 的粗开关。④ 顺带修 `test_shared_reduce_helper_is_two_stage`：`fd4d8820d` 把 `shared_reduce` 挪进 `backends/cuda/kernels/core/cuda_atomic.h`，用例仍断言 `src/type/cuda_atomic.h`（文件还在但不含该 helper），改成经 `backend_root` 取路径并加一条「helper 必须在这个文件里」的断言。修前 1 failed → 修后 **7 passed**。⑤ 无产品代码改动 |
| 3.23 | 融合逐元素 kernel 带宽效率 | 待领 | | `c0f3420a` 已合入 1/N：可复现的测量口径与实测屋顶线（skill `cuda-elementwise-bandwidth-roofline`，四个脚本，nsys 与 profiler 互校）。**任务描述的前提已失效**：`large_diffusers_unet2d` 一步（RTX 4090、TF32、独占卡）逐元素类实测 **3.37 ms（nsys）/ 3.29 ms（profiler）、1086 GB/s、屋顶线 ratio 0.84**，不是「4.47 ms、475 GB/s、峰值一半」——实测可达 copy 带宽 916.7 GB/s，这一类整体**已经贴着屋顶线**（超出部分由 72 MB L2 承担）。同机 PyTorch 2.12.1 为 elementwise 2.20 + other 0.84 = **3.04 ms**，即验收线的实测对应值；差距 0.30 ms（约 10%），**未达标**。49 种融合 kernel 的正向超出合计仅约 0.59 ms，且三个来源都不在代码生成里：float64 标量除法 0.55 ms（兼容层，见下方独立行）、裸 `transpose` 0.10 ms（`src/ops/transpose_op.cc`）、约 60 次几乎不搬数据的小 kernel 0.23 ms（纯 launch 延迟）。**结论：这条验收靠改代码生成达不到**；把兼容层那条 float64 加宽去掉即可 3.29→2.73 ms 并越过验收线，需先由兼容层分区决策 |
| 3.24 | 布局收尾 | 待领 | | |
| 4.01 | 分配器 id 空间随分配器实例走，不再是进程静态 2M 单例 | 已合并 | device | 4e407447 |
| 4.02 | 合并多卡 | 已合并 | device | `ad9aab3a`（Var 带设备、算子在自己设备上跑、逐设备分配器与库句柄）、`c97b707a`（跨卡拷贝算子）、`93b48a8e`（torch facade）。选了什么、为什么，改写进 `device-placement.md` §5。**一处未达成**：跨卡拷贝的定序在本机不是回归网——8 张卡两两 `cudaDeviceCanAccessPeer` 全 0，驱动把跨卡拷贝经主机中转并自行与源卡串行，把 event 对整对删掉测试仍全过（实测）。测试写好了并会打印当前处于哪种情形，换到能 peer 的机器上才成为守卫。方法沉淀在 `agent/skills/multi-device-verification` |
| 4.03 | `BackendRegistry` | 已合并 | device | 原生 NativeRuntime 持有版本化 BackendOps 表，CPU/CUDA 实现位于 runtime/backends；device/count/set/allocator/copy/sync/stream 实际接入 Var 分配、数组构造、共享迁移、DeviceCopy、fetch 与 swap 复制。注册不探测设备，原始池与 SFRL/Temp/Stat 包装分离；dual/delay-free 报真实设备；跨卡双向流依赖和 fetch 持有保留。旧设备别名发弃用警告。真实回调探针及最终 CPU/CUDA/结构 40 passed，双卡/五库/梯度/共享定向 39 passed，无 GPU 可见 26 passed，CPU-only 31 passed/1 个 CUDA 配置字段跳过。枚举避开 ACL 的 CUDA→ACL token 替换并有主机编译合同。按轻量验收执行，未跑完整模型门禁或 NPU/ROCm 实机；legacy 转换保留，Op/JIT 分派与 Python 原型合并仍归 4.04/4.05，不伪称已完成。 |
| 4.04 | `OpRegistry` | 已合并 | device | 原生 OpDef 按 backend 组合具体 Kernel/Codegen，执行器、并行编译器、tracer 与 fused relay 走注册入口；旧图固定定义，替换和注销重注册隔离 JIT 身份。CUDA 库登记 accelerator 实现，MKL 登记 CPU 实现；核心五处替换及 matmul/conv tuner 改 typed capability，库名移入库自有 TU。真实 CUDA 检查 cuRAND/CUB 执行与 NumPy 对照、cuBLAS/cuDNN 元算子图 relay；CPU-only 自定义扩展通过。修复双源码 CodeOp 缓存错用、fused 首次后端选择及 replacement 缓存身份问题。相关 CPU/CUDA 回归通过，ACL 改动 TU 主机语法通过；按轻量验收未跑完整模型与 NPU/ROCm 实机。Python 分派与 legacy 源转换仍归 4.05/4.11/4.12。 |
| 4.05 | Python 分派表 | 已合并 | device/pyops/build | 统一表按原生运行目标、输入设备、全部 Tensor dtype、shape/grad predicate 与优先级选择真实 callable；矩阵/卷积/RNN、归一化/推理、KV/RoPE、scan/indexing/gamma/FFT/AdamW 已接线。核心提供无同步 dispatch_context，拒绝真实混卡并保留 pending scalar 跟随规则；FFT/attention 缓存区分设备。删除 Python 假 BackendRegistry/bytearray 原型；库与资源单一登记，旧 globals 和 hook 改只读查询，MKL 显式禁用不加载且可恢复。算子域直接读取旧后端 flag 归零，显式设备命令/启动配置保留；legacy FFT mode 限制归表。CPU/CUDA 最终 54 passed，CPU-only 52 passed/8 个 accelerator skip，原 matmul/推理能力及相关合同 61 passed/8 个未加载 MKL skip，额外卷积/RNN 2 passed。完整结构 678 passed/10 failed/2 skipped，其中新增 softmax 签名问题已修复且定向通过，其余九类见交接，不称全门禁通过。保留原 ROCm/Corex 可达注册及 ACL 代码，未做 NPU/ROCm/Corex 实机或完整模型门禁。 |
| 4.06 | `jt.flags.backend_fallback ∈ {error, warn, allow… | 已合并 | device/gates | NativeRuntime 真实持有三态策略和尝试计数，默认 warn；executor 在 CPU 执行与输入迁移前检查，array staging/fetch/显式拷贝不误报。ACL 先完整预检，仅支持缺口请求回退，执行异常清理后原样传播；CPU 回退用 RAII 恢复 mode/flags/fused context。门禁设 error，scope 计数还能抓被吞拒绝，不再 grep 日志。CUDA 修前 error 仍执行 CPU，修后 marker/指针/数值/三态与原生策略 9 passed；严格模式常规 CUDA 与构建组合 28 passed。ACL 82 host/结构、4 TU/5 launcher ABI 与两次反向对照通过；无 NPU 实机，不声称 SDK 资源全量 RAII 完成。完整结构仍有既有失败，见交接。 |
| 4.07 | 后端配置改为返回 `BuildConfig` 值 | 已合并 | build | 三个 provider 返回冻结 BuildConfig，显式 BuildContext 注入服务，compiler 统一发布兼容字段，不再由后端改 globals/追加源文件；entry point 仅加载选中后端，CPU 选择不触碰 CUDA 探测/安装，普通 CPU/CUDA cache 指纹保持。utils 整树反向 import 清零，compile_module 服务注入，序列化实现迁 jittor.serialization，旧路径在 bootstrap 后解析同一对象。CPU/CUDA 构建/扩展依赖/序列化/分派组合 50 passed，strict CUDA 与 CPU选择组合 28 passed，CPU-only 联合 82 passed；离线 provider/SDK 分支 39 passed。完整结构 740 passed/9 failed/2 skipped，旧失败不掩盖。保留 Corex 兼容库与 legacy 源转换；旧 utility pre-bootstrap 使用边界、可选后端初始工具链旧耦合及无 CANN/ROCm/Corex 实机证据写入构建文档。 |
| 4.08 | 流与事件模型 | 已合并 | device | `0dfcb3dd` 每设备 copy/communication stream 与 ready/done event，接入 array H2D、fetch D2H、device_copy、NCCL collective；`78235157` 双 rank NCCL 用 rank 相关输入验证数值且 communication 双向依赖计数精确 +2。GPU 0/2：两 rank 各 1 passed，mixed-device H2D/fetch 2 passed，6.C16 下毒 1 passed，device_copy/multi-device 6 passed，既有 overlap 正确性 1 passed；未用负载敏感绝对墙钟阈值 |
| 4.09 | per-device 库句柄 | 已合并 | device | `13c28084`；4.02 已有五库 per-device 资源，本提交补齐每次执行前 SetStream。GPU 0/2 新增测试实际执行 cuBLAS/cuDNN/cuSPARSE/cuRAND/cuFFT 各两次并断言两卡逐库 bind 计数均 +2，1 passed；各库现有 wrapper 聚焦 5 passed；CPU 聚焦 1 passed |
| 4.10 | CUDA kernel 存放位置统一 | 已合并 | device/build | CUDA 七库、NN/misc/math/pooling/sparse/CCL/loss3d 与核心 GPU 源统一到顶层 backends/cuda，原生注册项选择实际后端源码；共享索引数学保留单 owner，原子前缀与调度属于后端。NN 根目录实现移出并加 exact-entry，ACL KV 移到 backends/acl，旧模块同对象别名保留且不泄漏私有 facade。源码/安装包/legacy 转换资源解析及 sdist/wheel 同步，真实 CUDA 分派与库 13 passed、最终数值/梯度组合 27 passed、原子展开及稀疏/别名 14 passed；隔离 wheel CPU-only 冷编译、索引/scatter/softmax、C++ schedule 与三步训练 selftest 通过。修复迁移中暴露的 candidate i/j 既有错误、原子宏丢失及复合源码行号；完整结构检查的新增目录/接口问题定点修复，既有失败仍见交接。无 NPU/ROCm/Corex 实机；NCCL 通信资源、FlashAttention 与核心总布局仍归 4.15，不伪报完成。 |
| 4.11 | ACL 改为注册表后端 | 已合并 | device/coreops/pyops | 代码阶段按用户授权完成，未声称 NPU 实机：35 个 Python SDK builder 归 backends/acl/kernels/ops，模块级实际实现经确定注册清单接入原生 owner，change_function/warp/公共类替换删除；post_process 仅注册，pinned/并行/归约要求归 BackendOps ABI 2 策略。原生 Kernel.compile 取代全局 hook，注册期 composer 同时处理已有及晚加载 OpDef，旧图 pin 与稳定启动缓存身份保留；显式 backend=acl CodeOp 替代源码注释猜测，反向继承标记，HCCL 明确编译入口。typed 基本 Get/Set 经 aclstream 的设备复制保留整数视图/写回/梯度，地址计划 NumPy 对拍 38 passed，unsupported 变体写入前拒绝；scalar 广播未优化。CPU/CUDA CodeOp/注册 30 passed，索引/低精度梯度/张量入口 13 passed，rotary/安装/策略 24 passed，CPU-only 最终 17 passed/2 个 CUDA skip；两 ACL TU 与坏符号负向检查通过。完整结构及迁移前已有归一化失败证据见交接；4.12 转换器、4.15 SDK 总布局与 NPU 真机/性能仍未完成。 |
| 4.12 | 删除 `process_jittor_source` 与 `process_acl` | 已合并 | cudabk | **代码半已闭合；ROCm 实机确认归「硬件验收」，见 [`deferred-hardware.md`](../manuals/deferred-hardware.md) 的 ROCm 一节。** 前半是今天的 13 个提交（`3b081d582` ACL 配置改原生注册初始化、`79c6d7162` ROCm 编译入口移除源码转换与历史 blob、`06190e2b3` ROCm 改独立 HIP provider、`d861351f0` ACL 自有设备与 pinned 分配器），本次提交收尾。**删除安全性**：`process_acl`/`WTF` 补丁全树 0 处；`process_jittor_source` 只剩定义（`jittor_utils/__init__.py:1289`）、`BuildContext.transform_sources` 字段声明、`compiler.py:61` 赋值三处，另查 `getattr`/字符串反射/`replace(context, ...)` 均无动态取用，确认零消费者后三处全删（连带 `jittor_utils` 里只服务于它的 `backend_root` import）。**`BuildContext` 是 provider 公开契约**（`docs/architecture/backend-build-configuration.md` 原文承诺「supplies explicit compilation, source transformation, ...」），删字段属接口变更，该文档四处＋`source-architecture.md` 一处已同步改写。**防回归合同** `tests/structure/test_core_source_is_not_ported.py`（3 条）钉「核心源码不再是移植的输入」，按形状不按名字：provider 契约不得再提供改写服务、不得有函数「遍历树＋原样拷贝＋写文件＋挑 `.cc/.cu/.cuh/.h`」四signal 齐全、bootstrap 之外不得重新绑定 `jittor_path`。**造 4 个反例全部报红**（复原 `transform_sources` 字段／复原 `process_jittor_source`／换名 `port_tree_for()` 同形状／`evolve(jittor_path=...)`）。旧门禁 `test_backend_conversion_boundary` 漏掉这条正是因为它只匹配名字且只扫 `python/jittor`，而定义在 `jittor_utils`、调用点写成无括号赋值。**三套门禁全部与改前逐条同集合。** 改动前只留了结构门禁基线，CPU/CUDA 两套改用钉在 `HEAD~1` 的只读 worktree 同时刻同负载对跑：结构 15 failed / 880 passed / 2 xfailed，改前改后完全一致、失败名 `diff` 为空（passed 净持平＝删 3 条转换用例加新合同 3 条）；原生 CPU 两侧同为 **88 failed / 1731 passed / 1200 skipped / 3 xfailed**，88 条失败名 `diff` 为空；CPU torch 两侧日志**逐字节相同**（同一处崩溃、每行进度串一致）；CUDA 失败族对跑，改后 42 failed / 89 passed，改前 43 failed / 87 passed，`comm` 出「改后有而改前没有」为空。**既存失败无一由本波引入**，其中两条是分支现状、值得下一位接：(1) 原生门禁被 `14e5920e5 [4.14]` 的同名模块 `tests/core/test_device_methods.py` 与 `tests/backends/cuda/test_device_methods.py`（两个目录都没有 `__init__.py`）**卡在收集期直接 error**，两棵树一样，上面的原生数字是 `--ignore` 掉该文件后取得的；(2) CPU torch 门禁在 55% 处硬崩、无汇总行，两棵树同一处。**CUDA 冒烟通过**（`has_cuda=True`，`registered_backends()==['cpu','cuda']`，64×64 matmul 与 numpy 最大差 7.2e-06）。CUDA 导入的四处断点已由 `803e37853` 修复。**未声称 ROCm 实机验证**。`acl_legacy` 命名不一致未改，归 4.15 |
| 4.13 | 跨后端契约矩阵 | 已合并 | device/gates | 矩阵的**两个轴都从 4.03/4.04 的注册表生成**，不再手写：后端轴 `known_backends()`（新增 `@pyjt` 绑定，核心声明的全集）∪`registered_backends()`，算子轴 `backend_supported_ops(后端)`。`tests/backends/parity/backend_contract_matrix.py`（纯逻辑＋探针）＋`test_backend_contract_matrix.py`（门禁），接进 `noxfile` 的 `cuda` session，`cpu` session 因跑整棵 `tests/` 自动包含。与既有 `test_device_parity.py` 的差别就是本条的价值：那份的两轴都在注册表之外（`op_db`＋一个探测出的 `_ACCEL` 字符串），注册表多一个没人测的实现时它不会红；新矩阵会——声明了却既无探针又无书面理由的实现直接失败。**缺硬件的后端标未验证不跳过**（0.24）：五档状态 `passed`/`failed`/`not-declared`/`unverified:not-built`（本机 acl/rocm/corex）/`unverified:no-device`（CPU-only 配置下的 cuda 列）/`unverified:no-probe`（25 格，逐格书面理由），判定顺序先「后端能不能跑」再「有没有探针」，所以无硬件后端不可能因探针在别处跑通而记成通过；`no-device` 档本机走不到，用合成行＋「一旦被调用就 assert 失败」的探针测。实测 47 个算子名×5 个后端行；CUDA 54 passed（cpu 35＋cuda 38＝73 个已验证格子），CPU-only 原生与 torch-shim 各 42 passed/12 skipped。**两次变异证明有牙**：加速器侧返回值×1.01 → 40 格红；删掉 `"binary"` 探针 → 棘轮单条红。顺带修两条本门禁发现的：`setup_cutt()` 是唯一不走 `setup_cuda_lib()` 的库路径，后端搬顶层后既缺 `-I backends/cuda/include` 也缺 `cuda_sdk_flags`，wrapper 编译不过而症状伪装成「本机没 cuTT」，6 条 cuTT 用例恒 skip 读着是绿的（与看板上「`setup_cutt()` 无调用点」是同一后果的第二个原因，调用点今天在 `compile_extern.py:1305`）；惰性库加载让注册表快照缩水（`backend_supported_ops("cpu")` 强制加载前 35 后 41，少的正是 `mkl_*`），矩阵先 `load_optional_libraries()` 再快照并连原因一起打印。**未覆盖**：矩阵只有 (op, backend) 两轴，交接文档提的 dtype/layout 两轴不在其中（dtype 覆盖仍在 `test_dtype_coverage.py`）；25 个 no-probe 格子仍是未验证；ACL/ROCm/Corex 无硬件，只有 `not-built` 行，**未声称实机验证**。现状记录未改：ACL 描述符注册名是 `acl_legacy`（`backends/acl/src/backend.cc:645`），与 `BackendId::Acl` 规范拼写 `acl` 不一致，层次归属属 4.12/4.15 |
| 4.14 | `Module.cuda(i)`/`npu(i)`/`x.to(...)`/`x.cpu()` … | 已合并 | device | 14e5920e；修前 CPU 2 项、双卡 CUDA 4 项失败；修后新增 CPU 2 项、GPU 0/2 双卡 4 项及 4.02 聚焦回归 4 项通过；无 NPU 硬件，未做真 NPU 验证，无 ACL 时解析设备号后明确报能力错误 |
| 4.15 | 布局收尾 | 待领 | | 前置 `4.12` 已合并（整树文本替换通道删除，`BuildContext` 不再有 `transform_sources`，`jittor_path` 只由 bootstrap 绑定一次——搬 `python/jittor/src/` 到顶层 `src/` 时不必再考虑「转换副本」这条路径）。**顺带接一条**：ACL 后端描述符注册名是 `acl_legacy`（`backends/acl/src/backend.cc:645`），与 `BackendId::Acl` 的规范拼写 `acl` 不一致；改名要动该描述符（C++ 核心重编）外加约 30 处 `register_kernel(..., "acl_legacy")`／`dispatch_context().backend == "acl_legacy"` 调用点与测试，跨 4.12 的改动面，4.12 本波未做，归本条。无 Ascend 硬件，改完需实机复验 |
| 5.01 | 114 个 `foo_` 就地方法改白名单显式声明 | 已合并 | pyops | 9d140c1c。85 个启发式生成别名收敛成显式白名单，错误别名归零，all_/any_ 非就地原语不再伪装；native 聚焦 20 passed/2 skipped |
| 5.02 | 视图与存储模型 | 待领 | | 前置 3.01 结构拆分已落地（`33c10331`、`fa2d8523`）：所有内存分配与主机/设备迁移现在集中在 `exec_runner.cc` 的 phase 6，storage 模型要改的分配点是一处而不是散在 run_sync 里 |
| 5.03 | 转置隐藏标记 | 待领 | | **协调者实测（2026-09-07）：这条静默算错是活的，判据已固化成 strict xfail。**复现：`a=jt.array(arange(12).reshape(3,4)); b=ones(3,5); at=a.transpose(); a.assign(zeros); jt.matmul(at,b)` ——实得 assign **之前**的乘积（首行 0+4+8=12，每列都是 12/15/18/21），而不是 0。参考语义用**真 PyTorch 2.12.1** 核过：`a.t()` 后 `a.zero_()`，`at @ b` 全 0。所以这不是「Jittor 的惰性语义如此」，是转置留下的隐藏标记让 `matmul` 读了源张量的旧内容，调用者静默拿到错值。已加 `tests/core/test_transpose_view_staleness.py`（1 xfailed / 2 passed）：主判据是 **`strict=True`** 的 xfail，`5.02` 的存储模型落地后它一旦开始通过就会以 `XPASS(strict)` **报红**，逼人来摘标记——普通 xfail 会让一个已修好的缺陷永远显示成「预期失败」，那正是本轮那七例的形状（交接 §6bis）。已验过它咬得住：把断言改成当前实际行为，立刻 `1 failed`。另两条是配套：一条钉住「今天返回的确实是 assign 前的乘积」（防止缺陷以别的方式漂移而 xfail 照旧红），一条钉住「不带 assign 的转置乘法本身是对的」（说明要修的是失效通知而不是这条路径） |
| 5.04 | 参数模型 | 已合并 | pyops | 3d40fa9e。`parameters`/`named_parameters`/`state_dict`/`named_buffers`/`_buffers` 共用一份角色遍历；绑定权重按对象身份去重而 state_dict 保留全部别名，BatchNorm buffer 按名字注册，查询不再改写 Var 名称。CPU `tests/nn` 182 passed/145 skipped，CUDA 聚焦 23 passed，Torch-shim 入口 1 passed，独立 PyTorch 2.12.1 语义对拍通过 |
| 5.05 | `eval()`/`train()` 只切 `is_train`，冻结统一由 `requires… | 已合并 | pyother | 4a8c4145 |
| 5.06 | hook 存实例级有序字典，多 hook、prepend/always_call 生效、可移除 … | 已合并 | pyother | 9117b843（含 `Var.register_hook` 返回 handle、`_dispatch_call` 接缝、weight_norm 的单 hook workaround 一并删掉） |
| 5.07 | `jt.Function` 每次调用创建一次性上下文对象，实例无状态 | 已合并 | pyother | 5c4e624b；0f639e5b（收尾：torch 兼容层的 ctx 记账跟着挪到一次性上下文上，`materialize_grads` 原本静默失效） |
| 5.08 | `flag_scope` 的备份改局部栈，`__call__` 每次新建 scope | 已合并 | | 5720e7e8 |
| 5.09 | 29 处融合 kernel 的启用条件由全局 `no_grad` 改为「输出不需要梯度」 | 已合并 | pyops | 11200c4f。native nn/misc 29 处统一按 grad mode 与递归输入 requires_grad 判定，无反向融合输出显式 stop_grad，fp16/bf16 cuDNN backward 放开；CPU/GPU契约 5、GPU norm 5、capability 7、CPU serving 9 项通过 |
| 5.10 | 索引与计数统一 int64 | 已合并 | pyops | 3e4d8a0b（`where_op.h` 的默认 dtype、randperm、topk 的空/非空两条分支、MaxPool2d/3d 与 AdaptiveMaxPool2d/3d 的 return_indices 全部 int64；顺带 `cub_where_op` 的计数与 free 大小、池化索引编码 `p*W+q`、repeat_interleave CUDA 快路径改 64 位并删掉 2^31 断言——该断言在 `misc/tensor_ops.py` 而不是审计写的 `pool/core_2d.py:198`。`jt.argsort`/`argmax`/`arange` 仍是 int32，理由见提交说明） |
| 5.11 | `amp_reg` 位常量命名导出，一律 `\ | 已合并 | pyother | 24a334cf；fc9244c4（收尾：用例的 level 切换改走 flag_scope，撞上 0.15 新加的 flag 泄漏规则） |
| 5.12 | matmul 四条路径共用能力表，dtype 用枚举不用子串 | 已合并 | pyops | 9d987034（`_cublas_can_take` 一个谓词供四处使用，判据是 `a.dtype == b.dtype and a.dtype.is_float()`；`bmm_transpose` 补上 dtype 守卫与 amp_reg。审计两处更正：「`"float" in dtype` 匹配 bfloat16/float64」属实但 cuBLAS 都支持、不是缺陷；「batched 只查 a 的 complex」属实但不可达。真正可达的是 `bmm_transpose` 完全没有守卫，整数/复数操作数在 CUDA 上撞 C++ 断言而同一乘积写成 matmul 就能算） |
| 5.13 | `unique` | 已合并 | pyops | 9c24a433（unique：四条路径合一，CPU 比较器不再把排序键截断成 int；根因不是注释说的「cub 只支持 int32」，而是存索引的输出 var 用了输入的 dtype，外加手工切分的 scratch 对不齐）43985e2c（isnan/isinf/isfinite 不再窄化成 float，float64 的 1e300 在所有后端都不是 inf）c8b4b206 + d6f08532（cumsum 一份实现、一条求导规则、一个 dim 契约，CPU 不再走 numpy 主机回调） |
| 5.14 | `Var.scatter` 改非就地 | 已合并 | | 0b75e187 |
| 5.15 | `.half()`/`.float16()` 删死的 amp 分支 | 已合并 | pyops | bf0317af。四种显式浮点 dtype 转换共用一条路径并覆盖持久/非持久 buffer，整数与 bool buffer 保持原 dtype；删除恒假且会改写整个类 `__call__` 的 amp 分支。新测试修前 5 failed/2 passed、修后 CPU/CUDA 各 7 passed |
| 5.16 | `state_dict(to="torch")` 用 `from_numpy`，不强制 floa… | 已合并 | pyops | b2238e7c。原回归 4 项中修前 2 红（int/bool 被压成 float32、大 int64 值改变）；本波回填复验 CPU 4 passed，真实 CUDA GPU 端 int64/bool dtype、大整数数值与实际计算通过 |
| 5.17 | 同一概念合并 | 已合并 | pyops | 1793f08f（平均池化：删 `pool/layers.py` 旧 AvgPool2d 并转发，2D/3D 同一套 `count_include_pad` 语义）3344cb40（`nn.Conv2d.execute` 委托 `functional.conv2d`，编译选项与输出尺寸校验合一）cd7ce682（BatchNorm/LayerNorm/GroupNorm 模块只做参数管理；`batch_norm(training=True)` 走融合 kernel；BN 的 sync 与非 sync 合并成一套数学——含审计 2026-09-03 补充的第三处：sync 分支的 `E[x²]-E[x]²` 只在 MPI 下跑，均值远大于标准差时相对误差约 7e-2） |
| 5.18 | 同一概念合并 | 已合并 | pyother | 40fa8695（efficientnet 投影层）37ac0ac5（models/_utils）4179c899（loss 的 _reduce）d5892775（分布类）d569f22d（旧式 scheduler）dd1cbe30（init 的 gain 表与 fan）96cb9b1c（linalg helper）f23dc9b8（normalize 合并到 torch 语义） |
| 5.19 | 被静默忽略的参数改为传非默认值时 warn 或 raise | 已合并 | pyops + pyother | 1710aef1（算子参数：relu/leaky_relu/silu/mish 的 inplace、instance_norm 与 InstanceNorm 的 running stats/momentum/is_train/sync、svd 的 compute_uv/driver、inv_ex 的 check_errors、ctc_loss 的 zero_infinity、sort 的 stable；topk 的 sorted 判为无需处理，见提交说明）。共用基础设施 `python/jittor/_arg_policy.py`。4cf6df28（实现 ResNet `zero_init_residual`）；211339c9（vjp/jvp strict、DataLoader pin_memory/persistent_workers、kaiming generator、fftfreq/rfftfreq dtype/device 与未知 kwargs）。统一回归 44 passed |
| 5.20 | import 期副作用删除 | 已合并 | pyother | 505a1155 |
| 5.21 | 六个 monkeypatch 安装器写成显式有序清单并加断言 | 已合并 | pyother | `3cd1a614`：新增 `_install_order.SEQUENCE` 显式声明十步安装顺序与 `record/verify` 运行时校验；`jt.sum`/`Var.sum` 共用 full-reduce 路径。`tests/core/test_install_order.py` 17 项、`tests/structure/test_install_order.py` 6 项在提交中通过 |
| 5.22 | `nn` facade 不导出 39 个下划线名，内部用模块局部名不经 `jt.nn.*` 晚绑… | 已合并 | pyops | 5d67f36b。源码 `jt.nn._*` 使用与 `dir(jt.nn)` 私有导出均为 0，后端私有覆盖迁入 `nn.backends.hooks`；结构 17 passed，CPU 25 passed/8 skipped |
| 5.23 | 根命名空间显式 `__all__` | 已合并 | pyops | d80d0b99。根星号来源归零，414 名运行时 `__all__` 与生成 pyi 顶层声明一致；namespace 13 passed，结构聚焦 4 passed |
| 5.24 | 10 个 `jt._*` 跨模块契约 | 待领 | | 核查（2026-09-06）：`2.13` 已落地，`jt.runtime` 与 `jt.config` 都在。根命名空间剩下的下划线名基本是 import 机制残留（`_publish`、`_limit_openmp`、`_NATIVE_*_EXPORTS`、`_compat_*` 等），**不是本条要收的跨模块运行时契约**。四个真契约（`_torch_leaf_params`、`_active_optimizers`、`_current_optimizer`、`_torch_retained`）现在都由 `compat/torch/tensor_state.py` 拥有，它把历史名作为兼容别名 `setattr` 回 jittor 根模块——**即本条的剩余面全在 `compat/` 内，与 `7.12`（205 个 `_torch_*` 并入一个 `TorchTensorState`、验收 `grep _torch_` 于 compat 外为 0）是同一批改动**。**协调者更正（2026-09-07）：说「全在 compat 内、与 7.12 同一批」只对了一半。** 7.12 的验收是 `grep _torch_` 于 compat 外为 0，而本条的剩余面里有三个名字**不带 `_torch_` 前缀**，那条 grep 抓不到：`_current_optimizer`（`compat/torch/optimizers.py:42` 写 `jt._current_optimizer = self`，消费者在 `installers/nn.py:302,458,1866`）、`_active_optimizers`（`compat/torch/tensor_state.py:88` 把历史名 `setattr` 回根模块）、`_transform_getitem_to_index_depth`（`installers` 里的深度计数器写在根上）。三者的定义与消费者确实都在 compat 内，**但写入目标是 jittor 根模块**，所以本条真正的剩余面是「compat 不得把下划线运行时契约 setattr 到根命名空间」，需要一条独立门禁，不会随 7.12 自动关闭。移除这三个别名属破坏性变更（虽为私有名，仍应记进 `docs/releases/2.0.md`）。建议与 7.12 同一波由 compat 分区做，届时一并补那条门禁 |
| 5.25 | `python/jittor/utils/` 拆散 | 已合并 | compat | `be2935f0`、`fdf3b759`（translator/server 迁入 compat，jtune/nvtx 迁入 jittor.tools，仓库脚本迁入顶层 tools）；`b70afbce`/`416a7fe4`/`02a6b5ee` 将 dlink compiler/dumpdef 迁入 build；本提交将 C++ tracer 迁入 `jittor.tools.tracer` 并改掉兼容层旧 nvtx 引用，utils 抽屉清空。结构/打包合同通过。 |
| 5.26 | 布局收尾 | 待领 | | |
| 6.C01 | `.item()` 对无符号 dtype | 已合并 | | 9b3023b1 |
| 6.C02 | `PySlice_Unpack` 返回值检查，三个变量初始化 | 已合并 | bindings | 78d08344 |
| 6.C03 | 整数提升 | 并入 2.16 | | |
| 6.C04 | 含 `void` 语句被删 | 并入 3.09 | | |
| 6.C05 | 融合边号 ≥256 回绕 | 已合并 | | 21a4f4fc |
| 6.C06 | `grad.cc:65-68` 判空对象改为 `dx` | 已合并 | | 4875a7aa |
| 6.C07 | 缺失梯度默认报错 | 已合并 | | 78c154e4 |
| 6.C08 | `grad.cc:146-261` 两趟遍历合一趟并快照结构，删无边界游标 | 已合并 | coreops | 096804a9。每个 gvar 局部快照 outgoing 与 op 输入输出后立即消费，删除 id_buffer 和无边界游标；动态新增输出回归修前进程终止，修后 CPU/GPU2 各1项及 autograd 各8项通过 |
| 6.C09 | `backward()` 可重复 | 已合并 | | 93b6e813 |
| 6.C10 | CUDA 分配钩子两张 map 用 `find` 加显式错误，释放后 `erase` | 已合并 | mem | `59c7a9b3` |
| 6.C11 | CPU 分配失败抛异常，返回值必须检查 | 已合并 | mem | `a683274e` |
| 6.C12 | `cuda_device_allocator.cc:32-37` 的 managed 回退放到 … | 已合并 | mem | `e48c52c2` |
| 6.C13 | 零字节分配不返回伪指针 `0x10` | 已合并 | mem | `b8b978e1` |
| 6.C14 | SFRL | 已合并 | mem | `a0da8374` 完成 SFRL 映射表清零、free/share_with 校验与五个分配器写回 allocation；`b0d90d44` 将 getitem/setitem 的别名判断改为 `7e223483` 引入的显式 share 环关系，并覆盖共享与非共享对象语义 |
| 6.C15 | `migrate_to_cpu/gpu` 迁移前检查 share_with 关系，整组迁移或拒绝 | 已合并 | mem | `7e223483`。Var 加共享环（`share_prev/share_next`），migrate 看到环就整组搬走并保持相对偏移。顺带两条：`ArrayOp::run()` 绕过 `free_var_mem` 换内存要自己摘环；裸分配器表达不了共享，新增 `Allocator::can_share()`，为假时退回旧行为**并告警**（不再静默断开） |
| 6.C16 | fetch 跨流 | 已合并 | mem | `9095484b`。**修法与任务行不同**：不是「记 event 让默认流等」——副流本来就在等默认流，再让默认流等副流等于取消掉异步重叠（`test_memcopy_overlap` 那条性能断言测的正是它）。改成在源块上多持一份引用直到主机回调之后；event 栅栏降级为 `can_share()` 为假时的兜底 |
| 6.C17 | `TempAllocator` 删遮蔽基类的 `used_memory`/`unused_mem… | 已合并 | mem | `4357c8bb` |
| 6.C18 | CachingBlock 保存底层 allocation 并原样回传，不再传 0 | 已合并 | mem | `74264dd3` |
| 6.C19 | 每个分配器一把锁并覆盖 `gc()` | 已合并 | mem | `6a73832b` |
| 6.C20 | swap | 已合并 | mem | `4b33609d`（文件名用运行期 pid 加随机 token、cudaMemcpy 查错、去静态 8 MB buffer）+ `2940d88d`（后半）。后半的核实结论：`save_mem` **已经**是编译期常量，「未完成特性挂在最热释放路径上」就现状而言不成立；真正坏的是 `export JT_SAVE_MEM=1` 从来没被翻译成 `-DJT_SAVE_MEM=1`，文档教的开法是空操作。已接上，开着时才进构建指纹 |
| 6.C21 | 检查 `NODE_MEMCHECK` 外 `check_graph` 静默空转 | 已合并 | mem | `2bce371e`。 绑定分区已让出（曾误领）。前置核实结论：`do_graph_check()` 前半段（从 hold_vars 反向遍历、重算 f/b/p）在任何构建下都真跑；只有后半段查悬挂节点的那个循环读 `lived_nodes`，而它只在 `-DNODE_MEMCHECK`（`compiler.py:1164`）下填充——所以 `check_graph=1` 在 release 下交付的是它宣称的一半。另：`Node::memcheck_all_exist()` 在出厂 object 里也是空的，它本该断言什么无法从 object 恢复（见 812714d5 还原者说明），不要当成还原时丢的。**做的是完整版而非最小版**（协调者确认）：登记表改成跟着 `check_graph` 走而不是跟着构建类型走，于是 release 下开 `check_graph=1` 两半都真查；`do_graph_check()` 返回悬垂扫描的节点数，扫到 0 时打一条每进程一次的警告说明为什么——不对称本身也可见了。关着时的开销量过：40 万个 Node 的构造从约 955 ns/个变成约 980 ns（+2.6%），在同进程 ±15% 的波动里 |
| 6.C22 | pyjt 关键字参数 | 已合并 | bindings | ed148a56 |
| 6.C23 | `is_type<NanoString>` 收窄 | 已合并 | bindings | f8f9de43 |
| 6.C24 | 带实例 `__dict__` 的类型加 `Py_TPFLAGS_HAVE_GC` 与 trave… | 已合并 | bindings | 4a30c5e4 |
| 6.C25 | 生成绑定补 `catch (...)` | 已合并 | bindings | b58ba756 |
| 6.C26 | `pyjt_compiler.py` 的 C++ 解析 | 已合并 | bindings | 4105d091 |
| 6.C27 | `Var.data` 返回的 numpy 视图 base 指向包裹该次 allocation 的… | 已合并 | bindings | 9504e520 |
| 6.C28 | 生成带「已构造」标志的 `tp_new` 或 `tp_dealloc` 先检查 | 已合并 | bindings | 8bd40d02 |
| 6.C29 | 标量转数组的全局 `tmp_data` 改自带 buffer | 已合并 | bindings | b57c31a1 |
| 6.C30 | `helper_cuda.h` 的 `peek` 去掉进程级闩 `peek_logged` | 已合并 | coreops | bcdf1593 |
| 6.C31 | 失败的 import jittor 在退出期 abort，父进程无声消失 | 已合并 | bindings | 64350894 |
| 6.C32 | `test_complex64_linalg.py::TestComplex64LinalgCPU::test_svdvals` 在 CPU-only 下 abort 且无诊断，带走整个 session | 已合并 | | 由 2.19 的 `4b5eaaa9` 修掉（`~VarHolder` 自己接住，liveness 队列的排空改 RAII，抛异常的步骤不再留下半排空队列）。**原描述的归因不成立**：abort 不是这条用例的问题。含 `4b5eaaa9` 的树上该文件 11 passed / 11 skipped、连跑 6 次零 abort，完整 `tests/core` 也跑到了汇总；不含它的树（`3baa0f4b`）上同一命令 EXIT=134。当时看到"单选也复现"是因为 abort 发生在进程退出期而不是用例里——单选时 pytest 已打完汇总，容易误读成这条用例失败 |
| 6.C33 | 0 维 bfloat16 输入让 `code` 算子的 `@for` 展开成死循环 | 已合并 | `8607ea4c4` | 2026-09-06 由 2.19 与 3.22 的执行者各自独立撞到。**登记的范围两头都不准**：dtype 与它无关（float32/float64/int32 一样失败，CPU 与 CUDA 一样失败），触发条件只有**秩为 0**；低精度只是碰巧——`test_safe_clip` 是全树唯一把 0 维喂给 `jt.code` 的用例。也不是死循环而是**编译失败**：`op_compiler.cc` 的展开器按 `vii!=vir` 停机，步长模板 `@for(i, DIM-2, -1, -1, ...)` 在秩 0 上成为 `@for(i, -2, -1, -1)`，计数器朝反方向走直到撞上 `total_step<1000`。改成按步长方向比较，与 `range()` 一致；步长 0 显式报错。**原有测试把缺陷钉成了契约**（断言 `@for(i,0,-1,@i)` 抛 "Too much step"），只跑现有测试的话红的会是修复本身。「约 40 条级联」协调者已独立复核：`tests/backends/cuda` 42 failed → **5 failed / 269 passed**，剩余 5 条（`test_cublas_test_op` 3 条、`test_cudnn_op::test_backward_nhwc`、`test_shared_reduce_helper_is_two_stage`）与本条无关且修前就在 |
| 6.P01 | 转置标记陈旧 | 并入 5.03 | | |
| 6.P02 | Function 实例复用、no_grad 泄漏、tied weight 参数集合 | 并入 5.07、5.08、5.04 | | |
| 6.P03 | H1 分组 conv3d 的 ww reindex 形状顺序 | 已合并 | pyops | a50c5678 |
| 6.P04 | H2 Pool3d `return_indices` 内核第三层循环变量 | 已合并 | pyops | 2fb2d15d |
| 6.P05 | H3 Pool3d CUDA 反向用 `pout_shape` 作上界 | 已合并 | pyops | 359031f4 |
| 6.P06 | H4 MaxUnpool2d/3d 在 `stride != kernel_size` 时用原始… | 已合并 | pyops | 70d97137（机制与审计描述不同：`xshape3` 本就是重建体宽度，真正错的是默认 `output_size`） |
| 6.P07 | H5 eigh 反向 `dout` 全零时写零 | 已合并 | pyops | aeeca502 |
| 6.P08 | H6 `_autograd_grad` 的 zip 用过滤后的 `new_grad_output… | 已合并 | pyops | e9c704cb |
| 6.P09 | H7 irfft 对实数输入与显式 `n` 的处理走 `:68-73` 的判别函数 | 已合并 | pyops | b59563c1（实数输入在默认 `n` 下原本就与 numpy 一致；错的是显式 `n`） |
| 6.P10 | H8 ReduceLROnPlateau 每轮从初始 lr 计算 | 已合并 | pyother | 634a8e8a |
| 6.P11 | H9 `unique(return_counts=True, return_inverse=Fa… | 已合并 | pyother | 7c854f1d |
| 6.P12 | H10 Adan 的 `clip_grad_norm` 移出 param_group 循环 | 已合并 | pyother | 888947fd |
| 6.P13 | H11 `zero_grad` 清缓冲而非只翻标志 | 已合并 | pyother | b116b545 |
| 6.P14 | H12 Adam 偏差修正用每 param 的步数 | 已合并 | pyother | 0d67526a |
| 6.P15 | H13 worker 异常不再变成给父进程发 SIGINT | 已合并 | pyother | 042bc2c7 |
| 6.P16 | H14 `mp_log_v` 做 int 转换 | 已合并 | pyother | f7162b68 |
| 6.P17 | H15 Pillow 版本用元组比较 | 已合并 | pyother | 03cf502d |
| 6.P18 | H16 `Dataset.__deepcopy__` memo 存对象不存 id | 已合并 | pyother | 9763203a |
| 6.P19 | H17 `LogitRelaxedBernoulli` 返回 logit | 已合并 | pyother | 2a76e252 |
| 6.P20 | H18 `ComplexNumber.__rsub__` 虚部符号、`__imatmul__` … | 已合并 | pyother | 8b36f3c4 |
| 6.P21 | H19 稀疏卷积重复坐标 CPU/CUDA 语义统一 | 已合并 | pyother | b3ebd1b5 |
| 6.P22 | H20 `to_dense` 对 COO 重复索引求和 | 已合并 | pyother | 9d1bf2a1 |
| 6.P23 | eigh 的特征向量梯度在 CUDA 上错约 60% | 已合并 | | d361100e |
| 6.P24 | Pool3d 的 count_include_pad 读原始参数 | 已合并 | | d221dbde |
| 6.P25 | Adan 偏差修正仍用全局 n_step；连带第一步 grad_diff 语义 | 已合并 | pyother | 2d5804a4 |
| 6.P26 | MaxPool3d 的 ceil_mode 输出尺寸比 torch 多一个平面 | 已合并 | pyops | f982a6b8。修前输出 `(4,4,4)` 对 Torch `(4,4,3)`；修后 CPU 18 passed/15 skipped，真实 CUDA GPU4 尺寸、索引往返和前后向 4 passed |
| 6.B01 | MPI 的 int64 改 `MPI_INT64_T` | 已合并 | dist | 03518707 |
| 6.B02 | ACL | 并入 硬件验收 | | `03daccfb`/`5388864c` 已完成 tensor/workspace/checkRet 代码前置与静态合同；按用户授权将 910B3+CANN 正常、失败传播、释放验证并入硬件验收，不计作代码缺口。 |
| 6.B03 | HCCL 宏错误时抛而非 return | 已合并 | dist | c657ab01 |
| 6.B04 | 分布式一旦被请求，初始化失败硬失败 | 已合并 | dist | 8ae65e24 |
| 6.B05 | cuBLAS `use_tensorcore` 三目判断写反 | 已合并 | cudabk | 9f5c3e90 |
| 6.B06 | `var_broadcast` 用传入的 root | 已合并 | dist | 89dd014b |
| 6.B07 | cuDNN RNN（dropout 掩码/work_space/infer_shape 泄漏 + 按实际 dtype） | 已合并 | cudabk | f5540427、da5bcad4 |
| 6.B08 | cuSPARSE | 已合并 | cudabk | 44b8a8a6 |
| 6.B09 | curand 奇数长度用临时 buffer 不越界写 | 已合并 | cudabk | 08a1bd66 |
| 6.B10 | MPI fp16 归约统一标量参考实现加可选 SIMD 与运行期 CPUID 检测 | 已合并 | dist | 734d55a1 |
| 6.B11 | ACL 六个算子静默把输入升到 fp32 | 已合并 | dist | 492e5385 |
| 6.B12 | `cutt_transpose_op.cc:77` 的 `cudaGetLastError()`… | 已合并 | cudabk | 58215816 |
| 6.B13 | cuFFT `cufftCreate` 后被 `cufftPlanMany` 覆盖的句柄泄漏 | 已合并 | cudabk | 11697758 |
| 6.B14 | conv3d 三算子迁到 backend plan 缓存 | 已合并 | cudabk | 8432a181 |
| 6.B15 | MPI 同时识别 PMI_/SLURM_ 环境变量或要求显式声明 | 已合并 | dist | 956c4b23 |
| 6.B16 | `sync_run` 在 ACL 上实现或删 flag | 并入 硬件验收 | | `15bccb92` 已完成 BaseOpRunner 同步尾部和静态合同；按用户授权将 `sync_run=0/1` 在 910B3/CANN 上的真实同步、失败归因并入硬件验收。 |
| 6.B17 | 析构不得抛 | 已合并 | cudabk | 272f00ba |
| 7.01 | 「看起来支持其实空操作」一律改为实现或抛 `NotImplementedError`，需显式 `… | 已合并 | 兼容层分区 | ff395ecc b7c12ddc 0446217e 47012a27 46bc9ea7 49d41acf 9053a7c0 |
| 7.02 | DDP 真实梯度同步 | 已合并 | 兼容层分区 | 4f08f1da |
| 7.03 | 每个 torch API 一个模块级一等对象加保真度标注 | 待领 | | 3f009970 已合入 1/N：新增 exact/approximate/unimplemented 保真度注册与确定性报告；factories 最终拥有的 20 个 API 成为稳定模块级 callable。8589860b 把 compiler family 的 `compile/trace/script` 提升为稳定对象；6d65fdca 把 numerical owner 的 `eye` 提升并登记 approximate fidelity，CPU 3 项通过。52ddeabc 把 `empty_like` 收回 factory owner，登记保守 approximate fidelity，compiler 阶段只绑定稳定对象并删除临时闭包；5dc59d85 将 `vstack`/`row_stack`/`hstack`/`dstack`/`column_stack` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，并以 1-D、2-D/混合 CPU NumPy shape/value对拍、身份与 metadata 共 7 项通过。630a3e44 将 `movedim`/`moveaxis` 提升为 numerical 模块级稳定函数，让 Var 方法共用内部实现避免递归，登记保守 approximate fidelity；身份、metadata、CPU 正负单轴/多轴及 Var 方法共 5 项通过。c38e1453 将 `unflatten`/`swapaxes`/`swapdims`/`ravel` 提升为 numerical 模块级稳定函数，让 Var 方法共用内部实现，登记保守 approximate fidelity；身份、metadata、CPU/Var shape-value 定点 3 项通过。d2c3b8f4 将 `copysign`/`xlogy`/`heaviside`/`signbit` 提升为 numerical 模块级稳定函数，让 Var 方法共用内部实现，登记保守 approximate fidelity；身份、metadata、CPU NumPy 对拍及 Var 方法定点 4 项通过。217e107f 将 `trace`/`diag_embed`/`diagflat` 提升为 numerical 模块级稳定函数，让 Var 方法共用内部实现，登记保守 approximate fidelity；身份、metadata、CPU/Var NumPy shape/value 定点 4 项通过。c0c6d283 将 `float_power` 提升为 numerical 模块级稳定函数，让 Var/root 共用内部实现，登记保守 approximate fidelity；模块身份、metadata、CPU NumPy value/float64 dtype 对拍 3 项通过。0ff1b6d1 将 `isclose`/`allclose` 提升为 numerical 模块级稳定函数，保留 `rtol`/`atol`/`equal_nan` 并让 `allclose` 返回 Python bool，登记保守 approximate fidelity；身份、metadata、CPU NumPy 与 bool 返回定点 3 项通过。af6280d7 将 `cdist`/`bucketize` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity；身份、metadata、CPU `cdist p=1/2` 与 `bucketize right=False/True`/dtype 定点 4 项通过。ac4877d6 将 `nansum`/`nanmean` 提升为 numerical 模块级稳定函数，让 Var/root 共用内部实现，登记保守 approximate fidelity；身份、metadata、CPU 全量/dim/keepdim/NaN-count 与 Var 方法定点 4 项通过。203d19c1 将 `aminmax` 提升为 numerical 模块级稳定函数，让 Var/root 共用内部实现，登记保守 approximate fidelity；身份、metadata、CPU 全量/dim/keepdim 与 Var 方法定点 3 项通过。9ada5bef 将 `pdist` 提升为 numerical 模块级稳定函数，让 Var/root 共用内部实现，登记保守 approximate fidelity；身份、metadata、CPU p=1/2 shape/value 与 Var 方法定点 3 项通过。2490ec9a 将 `logcumsumexp` 提升为 numerical 模块级稳定函数，让 Var/root 共用内部实现，登记保守 approximate fidelity；身份、metadata、CPU 1-D/2-D dim 与 Var 方法定点 3 项通过。1512f92c 将 `quantile` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity 并明确 NumPy CPU fallback/dtype/out/device 限制；模块身份、metadata、CPU q=0/.5/1 与 dim/keepdim 对拍 3 项通过。80b41079 将 `nanquantile` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity 并明确 NumPy CPU fallback/float32 与 device/layout/interpolation/out 限制；模块身份、metadata、CPU NaN q/dim/keepdim 对拍 3 项通过。7f2cbb83 将 `std_mean`/`var_mean` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity 并明确 correction/keepdim 限制；身份、metadata、CPU 基础值/tuple shape 定点 3 项通过。b1568308 将 `mv` 提升为 numerical 模块级稳定函数，保留 out identity 与 shape/size 错误语义并登记保守 approximate fidelity；身份、metadata、CPU value/out/Var 委托/非法输入定点 4 项通过。0d4828fb 将 `addmm` 提升为 numerical 模块级稳定函数，保留 alpha/beta 并登记保守 approximate fidelity；身份、metadata、CPU 默认/缩放及 Var 委托对拍 3 项通过。1cca275c 将 `mm` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，保持现有 2-D matmul 与 out/device/layout/dtype 限制；身份、metadata、CPU NumPy shape/value 及 Var 委托定点 3 项通过。d8dbbb44 将 `trapz`/`trapezoid` 提升为 numerical 模块级稳定包装，登记保守 approximate fidelity，保留 x/dx/dim/out 语义；身份、metadata、CPU 1-D/2-D NumPy 对拍、Var 委托及 out identity 定点 4 项通过。a291194d 将 `masked_select` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，保持 1-D 选择与 out/device/layout/dtype 限制；身份、metadata、CPU 2-D bool-mask NumPy 值及 Var 委托定点 3 项通过。8139d685 将 `narrow` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，保留正/负轴与 Var 委托；身份、metadata、CPU 正/负 `dim/start` NumPy slice 对拍定点 3 项通过。ee5adc5b 将 `tile` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，保留 tuple/list dims 与 Var 委托；身份、metadata、CPU NumPy.tile shape/value 定点 3 项通过。a2e86707 将 `diff` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，保留 n/dim/prepend/append 语义；身份、metadata、CPU 1-D/2-D NumPy 对拍与 Var 委托定点 3 项通过。9ac4ae4f 将 `square` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，保持元素级 x*x 与 Var 委托；身份、metadata、CPU NumPy 值定点 3 项通过。8647dc4d 将 `pairwise_distance` 提升为 numerical 模块级稳定函数，登记保守 approximate fidelity，复用原生 nn p-norm/keepdim 语义；身份、metadata、CPU p=2 值及 keepdim shape 定点 3 项通过。16333333 将 `amax`/`amin`/`count_nonzero` 收回 `jittor/misc/reductions.py` 原生 owner（compat 里那三个闭包是逐行复制），提升为薄转发的模块级稳定对象并登记 approximate fidelity；同时给 `install_methods` 的 `_axis_to_dim` 适配器加 `_torch_accepts_axis` 跳过标记，否则它会重新包一层、让模块级对象与 Var 方法不是同一个对象（这条对后续 max/min/argmax/argmin/norm/std/var 每个 cohort 都适用）；顺带修 `cosine_similarity` 的 fidelity 文案与断言长期对不上的红；CPU 4 项通过。9cba7d68 将 `cumsum`/`cumprod` 提升为模块级稳定对象，`out=` 依赖的 retained-view 写回器由 install 通过模块级句柄交接（同 numerical 的 `_vmap_runtime_impl` 做法），**并首次给 7.03 的 cohort 加 CUDA 数值对拍**（`instantiate_device_type_tests` 设备参数化）：CPU 15 passed、CUDA 15 passed，实测 4096 元素 float32 cumsum 两侧相对差 1.4e-06、并行前缀和比顺序扫描更接近 float64 参考，整数/bool 路径两侧逐位相同，判为后端求和顺序并登记进 fidelity。50876abf 将 `sort`/`argsort`/`topk`/`median` 提升为模块级稳定对象，CPU 13 passed、CUDA 13 passed；实测 512 元素重复键的 argsort **indices 两侧不同、values 逐位相同**，测试改为钉「values 跨设备逐位相等 + indices 能取回自己的 values + 键互异时 indices 才逐位相等」。d94c5cbd 把 `sign`/`trunc`/`frac`/`exp2`/`log10` 归一到 `installers/core.py` 单一 owner，**修掉 `torch.sign` 的静默错 dtype**——core 与 tensor 两份实现让 `torch.sign(int32)` 返回 float32 而 `Tensor.sign()` 返回 int32（真 PyTorch 2.12 实测两者都是 int32）；修前 3 failed、修后 5 passed。a7dcae1c 将 `nan_to_num`/`logaddexp` 提升为模块级稳定对象，把 clamp 实现的窄 posinf 偏差单独写成用例钉住；CPU 7 项通过。d1535282 将 `outer`/`tensordot`/`repeat_interleave` 从转发 wrapper 改为再导出原生 owner，恢复 `torch.repeat_interleave is jittor.repeat_interleave` 与 pickle 身份，`tests/structure/test_misc_structure.py` 由 2 failed 变 9 passed。**两个 installer 已单独闭合（7.03 第一次有「某个 installer 彻底空了」这件可验证的事实，而不是又一批 cohort）**：`796b8e43c` 清空 `_install_reductions`，内嵌 def/class **14→0**、lambda **13→0**；`d5740ee7d` 清空 `_install_module_methods`，内嵌 def/class **40→0**、lambda **6→0**。两者的 install 现在只做绑定，计数可复跑 `python agent/skills/torch-api-cohort-promotion/count_installer_closures.py`（`CLEARED` 标记＝nested 与 lambda 同时为 0），且各有一条 AST 测试防止下一波往空 installer 里塞新闭包。`d5740ee7d` 的搬迁手法：7 个「install 会覆盖掉的原始方法」在 import 时捕获成 `_ORIG_MODULE_*`（晚查会自指递归）；捕获 `self` 的嵌套 dispatch 提到模块级显式收 `self`、外层 `functools.partial` 绑定；pipelining 阈值由闭包 cell 改为模块级 `_pipeline_state` 因而可读出可复位；带原生 owner 的 `if not hasattr` 守卫照抄，不覆盖 jittor 原生 `Module.half` 等。12 个 Module API 登记 approximate fidelity。**归约/排序相关的 API 都过了 CUDA 那一层**：`_install_reductions` 的 cohort 见上；`_install_module_methods` 无归约但 `to`/`cuda`/`cpu` 就是 residency 迁移面，用 `instantiate_device_type_tests` 在 CPU 与 CUDA 各跑一遍，真实 CUDA 105 passed（含跨设备 state_dict 往返与 `zero_grad` 两种 `set_to_none` 在两侧的行为）。**修掉一处静默错**：`zero_grad(set_to_none=False)` 此前一律把 `.grad` 置 None，真 PyTorch 2.12.1 留下的是同 shape 同 dtype 全零张量；因为梯度裁剪与累积都写成 `if p.grad is not None`，参数被静静跳过、不报错也不改数（修前 1 failed / 修后 7 passed）。**另查明一处跨域差异按口径记给 owner 未自行改**：只要进程里还活着任何一个 optimizer 对象，`backward()` 就把梯度路由进它、`p.grad` 保持 None（开关是 `jt._active_optimizers`），真 torch 无论有无 optimizer 都填 `.grad`，属 optimizer/autograd 桥。剩余范围（同口径 AST 实测于 `d5740ee7d`，此前记的 `_install_tensor_methods` 76 与 `_install_cuda` 67 均已漂移）：`_install_nn_extras` 135、`_install_tensor_methods` 83、`utilities.install` 81、`_install_cuda` 80、`_install_lr_scheduler` 68、data 的 `install` 64、`_install_distributed` 64、`_install_optimizers` 45、`core.install_misc` 34 尚未迁移，整条 7.03 按剩余面保持待领 |
| 7.04 | 激活显式、一次性、可查询 | 已合并 | compat | `f704b9d4`：删除 argv 源码嗅探与 `jt.flags` 代理，部署/运行时入口统一为幂等 `activate()`（`enable` 仅同对象别名），公开不可变 `activation_status()`；HOME/TMPDIR/NCCL/严格数学环境只在显式 preflight 中准备，并接入 `EXPLICIT_REQUIRES_GRAD` 策略。聚焦 bootstrap 41 passed，策略接线 1 passed，部署静态测试 15 passed；结构实算单例仍命中既有 JIT IR 重复行号失败，非本项激活回归。 |
| 7.05 | install 事务化 | 待领 | | 第200波已合入四个 family：`4ecfb14f` 修失败路径——`install()` 的 `rollback()` / `release()` / 弹出 ledger 句柄三步没有 `finally`，而 `rollback()` 正是最可能抛的一步（外部改写就抛 `TransactionConflict`），于是类级 RLock 永久被占、其他线程的 install 无限阻塞，死掉的 ledger 仍在 `context.state` 里被运行期写入口找到；`rollback()` 另在第一个冲突处就抛出去，而回滚是逆序走的，更早的全部改写留在生效状态，现改为继续回滚其余条目、把撑不掉的一次列全、状态置为 `failed` 而不是留在 `open`。`dcbbedf3` 把六份 inline 的「现在有没有事务在记账」查找收归 `transaction.py` 的 `active_transaction()`/`set_flag`/`set_env`/`set_attr`——六份里只有 `core.py` 那份检查了事务状态，而 `record()` 拒绝已关闭的事务，所以一次失败安装留下的 ledger 会让此后每一次 `torch.zeros(device="cuda")` 抱 `RuntimeError: transaction is rolled_back`；顺带消去 `factories.py`/`tensor.py` 两份逐字节相同的 `_set_use_cuda`，**第134波引入的 `test_cleanup_structure` 跳文件重复实现红已修掉**。`0e336b58` 把 vLLM arming finder 的 undo 从 `remove(f) if f in sys.meta_path else None` 改成 owner-aware，并发外部替换不再静默通过。`c2fa74d8` 把剩余写入口钉成**分类闭集**（`tests/structure/test_compat_write_entry_points.py`）：AST 扫出 compat 全树 53 个进程全局写入口，ledger 17 / pre-ledger 12 / runtime 11 / pending 8 / deployed-payload 5，新增未分类或表里有而树里已无都报红（两个方向均造过反例）。**剩余四处（即 `PENDING`，未覆盖全部写入口所以保持待领）**：(1) `external_backend.py` 的 source-root import 用整表快照恢复 `sys.path`/`sys.modules`，丢弃并发写者的条目而不是报告，需 owner-aware 条目或子进程隔离；(2)(3) `vllm/__init__.py` 与 `vllm/flash_attn.py` 的 `install()` 由 arming finder 在首次 `import vllm` 时触发，那时安装事务已关闭；(4) `shim/cpp_extension/torch_utils.py:load` 发布的扩展模块寿命长于安装。后三者需一本寿命等于「已布防的 hook」而不是「安装」的账（同第143波对 activation 的结论）。**看板此前记的「`tests/structure` 2 条既存失败」已过时：`803e3785` 上实测 15 failed / 872 passed**，多数属 ACL/miniz/rocm/pytest 合同，本波后 15 failed / 881 passed 与改前逐条同集合（去掉 `test_cleanup_structure`、新增 `test_vllm_compat_structure` 的公开入口白名单一项）。 |
| 7.06 | 依赖单向化 core→tensor→nn/optim→distributed→fsdp→适配器 | 已合并 | 兼容层分区 | 27c4bdeb |
| 7.07 | 第三方库补丁搬出 compat/ | 待领 | | **协调者实测（2026-09-07）：验收未满足，判据可一行复现。** `JITTOR_TORCH_SHIM=1` 下 `orig=builtins.__import__; import jittor; import torch; builtins.__import__ is orig` → **False**，即 `builtins.__import__` 确实仍被替换着。前置 `7.05` 仍待领（剩四处写入口未进 ledger），而本条要求改走 module_patcher 的 entry point 并带版本断言。**这条判据应当直接做成测试**——它现在是文档里的一句话，没有任何东西在守它 |
| 7.08 | `torch.dtype` 改真正的对象而非 str 子类 | 待领 | | 9aaedba9 已合入 `torch.backends` 映射；dtype 真对象、完整 C++ dtype 边界迁移和占位 dtype 的计算/分配拦截仍需整体完成 **协调者实测（2026-09-07）：验收未满足，判据可一行复现。** `JITTOR_TORCH_SHIM=1` 下 `isinstance(torch.zeros(3).dtype, str)` → **True**。`type(x.dtype).__name__` 已经是 `dtype`，所以有一个名为 `dtype` 的类，但它**仍然继承 str**，故判据不成立。`9aaedba9` 合入的是 `torch.backends` 映射，与本条的 dtype 真对象无关。剩余面按计划正文：dtype 真对象、完整 C++ dtype 边界迁移、占位 dtype 的计算/分配拦截。**这条判据应当直接做成测试** |
| 7.09 | `torch.library` | 已合并 | compat | 99901e6c、d0a782a0。按张量真实驻留选择 CPU/CUDA 并排除 Meta，`register_autograd` 真正接入且模型特判移出通用注册层；线程局部 autocast dtype policy 进一步选择 AutocastCPU/CUDA，嵌套禁用与退出恢复普通路由。独立 PyTorch oracle 一致，CPU dispatch 8 passed、1 个未分配 CUDA 节点 skipped |
| 7.10 | `torch.compile`/`jit.trace`/`jit.script` 保留 pass… | 已合并 | 兼容层分区 | 3d898ece。语义参数拒绝、permissive allowlist/audit 与 ShapeProp ImportError 验收均有测试 |
| 7.11 | autograd 语义 | 已合并 | compat | `2ec34693`：`Var.is_leaf` 转发内核 `is_backward_leaf`，`Var.grad_fn` 对叶子返回 None、对非叶子返回 node/op/name 代理；shim autograd 语义 20 passed，core backward-leaf 查询 20 passed。requires_grad 策略差异仍归 7.12。 |
| 7.12 | 独立 torch 包 | 待领 | | **内核前置 `2.25` 已就位（提交 `c6e62ba1`、`781d4188`）**；`b2782315`/`06eba9aa`/`219c5b44`/`b3225844`/`ad46690d`/`696e5088`/`756a0fb6`/`41eb41c2`/`330d0a4c`/`9bbf87aa`/`d52f02ac`/`b84498c4`/`040b39cf`/`20cea34c`/`fda7501c`/`49906c5b`/`c6409827`/`e8cf42d6`/`b77217e8` 已将 leaf、retained、optimizer、requires_grad 状态收进 owner，并支持显式 activation 发布独立 TorchNamespace、事务替换、模式锁定、子模块、registry root、根条目回滚、不完整发布 fail-closed、父级闭包预检、root alias 隔离、alias 预检回滚和标准 ModuleSpec、元数据隔离、删除 ownership；stop_grad 会清理 owner 强引用，状态/autograd 定向 37 passed，namespace/transaction 定向 13 passed。这仍只是状态所有权前置，独立 torch 包的完整 requires_grad/模块边界与聚合验收继续待领。 **协调者实测（2026-09-07）：两半都未满足，且第一半差得比预想远。** (1) `JITTOR_TORCH_SHIM=1` 下 `torch is jittor` → **True**（判据要求 False）——shim 目前直接把 jittor 模块本身当作 `torch`，所以「独立 torch 包」在模块身份这一层还没开始。(2) `grep _torch_` 于 `python/jittor`（compat 外）实测 **4 个文件 19 处**，不是 0。逐处看下来是**三个具体契约**，不是散落的命名残留：**`_torch_parameter`**（`nn/modules/parameter.py` 3 处写、`_runtime/core_api.py` 4 处读，标记「这个 Var 是 torch Parameter」的跨模块布尔）、**`_is_torch_0d`**（`misc/indexing.py` 7 处，torch 的 0 维索引语义漏进了核心索引）、**`_torch_registration_semantics`**（`core_api.py`）；`misc/tensor_ops.py` 那 1 处只是注释引用。所以这一半是**可枚举、可闭合**的三项，建议作为独立一波先做，不必等模块身份那一半。另见 `5.24`：还有三个**不带 `_torch_` 前缀**的根命名空间契约（`_current_optimizer`、`_active_optimizers`、`_transform_getitem_to_index_depth`），本条的 grep 抓不到它们，两条应同波做 |
| 7.13 | FSDP2 | 待领 | | 已合入 37c0aed4、c0e6e1ae、48da7360、873dd5cf；仍缺峰值显存达标、复用原生 optimizer 更新逻辑与 DeviceMesh 真实分组 |
| 7.14 | vLLM 边界检查把 `torch` 视作 jittor 别名 | 已合并 | 兼容层分区 | 178be65a |
| 7.15 | `_rebuild_tensor_v2` 按 stride 还原或报错 | 已合并 | | 7e7877c8 |
| 7.16 | compat/ 内 129 个 `except: pass` 与 258 个宽泛 except … | 已合并 | 兼容层分区 | 72dbc22d（+ 一次修复：`93b48a8e` [4.02 3/3] rebase 时把 cuda.py/types.py 整段解回 7.16 之前，`swallowed()` 24→0、16→0，全树违规回到 47 条；已用三方合并恢复，见提交说明） |
| 7.17 | `runtime.enable()` 只把 shim 的 site 目录加进 sys.path … | 已合并 | 兼容层分区 | d5c769fb |
| 7.18 | 布局收尾 | 待领 | | |
| 7.21 | `compat/vllm` 只经公开入口使用 jittor | 已合并 | gatecheck | `71adc134`。新增 `jt.nn.qk_rms_norm_rotary` 与 `jt.nn.has_qk_rms_norm_rotary`，`compat/vllm/layers.py` 不再 import `jittor.nn.backends.hooks`。拆成两个入口是为了让非融合路径不必为「问一句有没有」先付一次 cos/sin cache 的 cast。`tests/nn/test_serving_ops.py` 12 passed（新增 3 条覆盖无后端／后端按序收到全部实参／后端拒绝输入）、vllm 与 nn 结构合计 29 passed。本机无 CANN/NPU，ACL 融合分支未实机执行，行为与改前一致（两处都在 hook 为 None 时短路） |
| 7.19 | 精度策略接线：Jittor 一档、torch 两档，底层 matmul/conv 分字段 | 待领 | | 依赖 8.03、7.08；需保持 shim 的卷积与 matmul 语义分离 |
| 7.20 | fp32 RNN 默认精度与 torch `cudnn.allow_tf32` 映射 | 待领 | | 依赖 8.03、7.19；需 CPU 递推与真实 CUDA 对拍 |
| 8.01 | 描述符与 workspace 一律 RAII | 已合并 | cudabk | afb08e88 |
| 8.02 | 集合通信走通信流加事件依赖，支持 `GroupStart/End` 桶化 | 并入 硬件验收 | dist | NCCL 部分已合并并有两卡证据；HCCL 同步优化按用户授权并入 Ascend 910B3/CANN 硬件验收，保留上机清单 `agent/manuals/hccl-on-device-verification.md`。 |
| 8.03 | 精度策略收敛 | 已合并 | cudabk | dab0690c |
| 8.04 | cuDNN 9 | 已合并 | cudabk | 7580b6e7（RNN v8 API）+ 9f2e7b80（版本闸门与 wheel 栈） |
| 8.05 | MKL | 待领 | device/gates | **部分交付，两条验收里只达成「能力表可查」，「CPU 卷积每调用开销下降」未达成**。已做四件：(1) **测试先修——原来 13 条 MKL 用例全是死的**：`jt.compile_extern.mkl_ops` 走 `library_attribute` → `get_library_ops("mkl")` **不带 `load=True`**，是「加载了没有」而非「取库」，惰性加载器不到 `nn.functional.matrix` 不触发。实测 `tests/backends/cpu` 5 条全部 `AttributeError: 'NoneType' object has no attribute 'mkl_conv'`（class 上那个 `use_mkl` skipIf 拦不住，该 flag 默认 True）、`tests/ops/test_mkl_batched_matmul.py` 8 条全部 skip 且理由不含缺硬件字样；`tests/ops/test_matmul.py` 更糟——结论随命令行顺序变，单独跑 5 failed/2 passed，先跑一个会加载 oneDNN 的文件再跑它 3 failed。新增 `tests/_helpers/onednn.py` 的 `requires_onednn()`（`load=True` 取库，取不到才 skip 且理由是真实原因，编译错误会说是编译错误）。改后 `tests/backends/cpu` **13 passed**（含新增 8 条）、`test_mkl_batched_matmul` **7 passed/1 skip**、`test_matmul.py` **3 failed/4 passed** 且不再随顺序变。(2) **能力表 dtype 声明可查**：`OpCapabilityRegistration` 加 `dtypes`，`@pyjt(backend_capability_dtypes)` 暴露；实测 `("cpu","matmul")==["float32"]`、`("cuda","matmul")==["float32","float64","float16","bfloat16"]`，审计说的「无处声明这种能力差异」有了落点。声明不参与选择（能力签名不统一，无通用 dtype 提取），仍由 `supports` 谓词决定，配防漂移测试：逐 dtype 跑 CPU matmul 看实际执行的实现，断言「声明集合==oneDNN matmul 真跑起来的集合」（实测 f32→`mkl_matmul`，f64/f16/bf16→通用 kernel）。**把 MKL 声明改空则三条红**。坑：`auto_convert_64_to_32` 默认 1，`jt.array(float64)` 得到 float32 Var，不关掉这个 flag 的「float64 用例」量的是 float32，与任何声明都一致。(3) **前向 prop_kind 与反向 hint 统一**（计划外那条发现的落点）：前向原为 `forward_inference`+`convolution_auto`，两个反向的 hint 为 `forward`+`convolution_direct`，**prop_kind 与 algorithm 两处都不一致**，oneDNN 给 hint 挑的 layout 可能与真实前向不同、每处不一致换成一次额外 reorder；三个文件统一为 `forward_training`+`convolution_auto`。核心里没有 train flag，按 is_train 选择做不到，留作剩余。(4) **库名不再钉死版本**：v2.2 装了 `libdnnl.so` 与 `libmkldnn.so` 别名两套，v3 只剩 `libdnnl.so`，而「装上了没有」/「可用吗」/「链什么（`-lmkldnn`）」三处全写别名，一棵正确的 v3 树会被报成「下载了但认为没装上」；改成一处 `mkl_library_layout()` 探测、链接名取实际存在的那个，v3 布局用合成目录测（**必须这么测：oneDNN 从 v3 起不再发布预编译二进制，v3.12/v3.13 各版 release assets 为空，已核实**）。验证：native CPU `tests/backends/cpu` 13 passed、torch 模式 16 passed/5 skipped（conv 4 条按写明理由 skip：断言的 conv tuner relay 是 native 语义，Torch 模式下卷积不走 conv tuner 认得的 reindex 形式；该目录不在 `TORCH_MODE_PATHS`，无门禁以该模式跑它）、CUDA `test_cuda_op_capabilities`＋4.13 矩阵＋`backends/cpu` 共 73 passed；结构 876 passed/16 failed 与改前逐条同集合。**剩余（下一位接手）**：① **迁 v3 API 未做**——v3 删了 `*::desc`，代码改动必须与 v3 库同时到位，而上游不再发布预编译 v3、本机各处只有 2.2.0，要验证得先决定「从源码编 oneDNN 怎么进安装路径」，这是设计决定；写一个编不到的 v3 分支等于写没人编过的代码。② **primitive/pd/reorder 按形状缓存未做**，故「每调用开销下降」未达成，`mkl_conv_op.cc:119-120` 仍每次 `jit_run` 重建 engine/stream/desc/pd/reorder；前一位建议「缓存写在 v2 API 上会被 v3 迁移全部重写」成立，但 v3 既阻塞，先在 v2 上做是有实际收益的，且 `prop_kind`/`algorithm` 已统一、pd 构造集中在每文件一处，改动面比之前小。③ matmul 仍只 fp32（`dnnl_sgemm` 签名是 float-only，要宽需换 `dnnl::matmul` primitive），现按 (2) 声明而非补齐。④ `tests/ops/test_matmul.py` 剩 3 条先于本改动存在的独立缺陷：`test_backward` 是 `liveness_info` 内存计数断言、`test_backward_once` 断言 relay 日志恰 1 条而循环实际产生 6 条（**oneDNN 未加载时是 0 条，所以这条在 CPU 上从来没通过过，两种情况都是失败**）、`test_matmul_example` 断言 `c.shape == [1]` 实得 `[]`（0 维/1 维，与 MKL 无关） |
| 8.06 | ACL 去样板 | 待领 | device | 5be5fa15 建立 BaseOpRunner 统一 workspace/execute/error/可选同步尾部并迁 unary；86b31e14 迁 binary；b7c763bd 迁 ternary SWhere；b1d7bd5 迁四个单步 reduce owner；251e3e96 迁 Cumsum；c59e3948 迁 MatMul；90050ccb 迁 Expand；51103861 迁 Floor；9a5a8ac4 迁 NanToNum；5e831df1 迁 Triu；88d4d35f 迁 Sigmoid forward；b76f1b16 迁 Transpose/Permute；a180c691 迁 Softmax forward；072e05b9 迁 Embedding forward；15668ea3 迁 Roll；11481922 迁 Gather forward；80c5e565 迁 ClampTensor；6e1d462c 迁 Stack；9c348801 迁 Flip；9db57798 迁 Scatter；530bbc8f 迁 Concat；1c4b32d9 迁 SplitWithSize；69e5974a 迁 Nonzero；11fe7012 迁 Range；faac2700 迁 Dropout forward；910e2c49 迁 LeakyReLU forward；830907ff 迁 ArgReduce max/min；a293b615 统一 Random uniform/normal launcher；055cb64b 迁 UpsampleNearest2d forward；553b5ec1 迁 SiLU forward；600ee169 迁 BatchMatMul；8251d29d 迁 RotaryPositionEmbedding forward；71bab738 迁 Maxpool forward；16a89606 迁 Avgpool forward，保留 descriptors/poolCeil/divisor 与同步策略且 backward 不动；静态合同 35 passed，补 Ascend 910B3 上机说明。本机无 CANN/NPU，仍待实机验证，未铺其余 family、胖 AclOpFunctions、op_idx_map、属性 data 通道与描述符缓存。 Host-only 前置 `e2731c0f`/`2dcc3448` 已建立 `acl_data` schema/normalizer 与结构负向合同 11 passed；C++ decoder、属性 owner、descriptor cache 和 910B3 实机仍待。第158波收口标准 owner：迁 SWhere、SigmoidBackward、BatchNorm 前反向（此前十波「已穷尽」的记录不成立，这四个仍自己驱动 execute 并保留 `LOG_PRINT` 后 return 的静默失败），BatchNormBackward 的 outMask 仍在 launch 同步之后释放；把作废已久的 `checkRet == 65` 计数断言换成「除 reduce prod 与 KVCacheMemcpy 两处豁免外没有 family 自己发 aclnn execute」的闭集不变量（该断言自 `5be5fa15` 起红穿约四十个提交）；新增 `agent/skills/acl-host-syntax-check` 无 CANN 主机 TU 语法检查（全树 43 源文件 + 68 launcher ABI 断言通过，含两次反向对照）。静态合同 88 passed。**剩余范围**：data-channel 的 C++ 解码入口、胖 `AclOpFunctions` 类型擦除、属性走 data 通道（`softmax.dim`/`triu.diagonal` 等 30+ 处仍拼进源码）、aclTensor 描述符按形状缓存、`acl_op_exec.cc` 的 "cu" 前缀删算子改注册表。**待实机**：Ascend 910B3 + CANN，`npu-smi` 前置与禁止 CPU fallback 检查见 `docs/guides/ascend-910b.md`；本机无 CANN/NPU，未做也不得声称硬件验证。第159波 **launch 尾部这一 family 闭合**：登记的「65 处」实测是 70 处（在 `5be5fa15~1` 上按「executeOp 体内自己处理 workspace 查询失败／自己 `mallocWorkSpace`／自己发 `(workspaceAddr, workspaceSize, executor, aclstream)`」清点，71 个 owner 里 70 个带尾巴，唯一例外 KVCacheMemcpy 没有 aclnn dispatch；按片段数是 76 处 `mallocWorkSpace` 或 69 处 `syncRun();`；「65」既不是 owner 数也不是片段数）。收完最后 9 个 owner（Upsample 前反向、GroupNorm 前反向、ArgReduce、TruthReduce、reduce prod 三条路径），**实测归零 71/71**；此前记为「有意保留」的 reduce prod 不成立——分步只要异步，`launch(ret, f, false)` 即可，中间张量释放前的屏障是另一条无条件 `aclrtSynchronizeStream`。顺带修掉三处静默失败：ArgReduce 查询失败原本打印后 return 留下未初始化的值与下标；reduce prod 的 `ret = aclnnProd(...)` 赋值了却从不读，失败即把未动过的输出缓冲当归约结果返回；TruthReduce 的 `throw` 统一成 `LOGf`。等价性用 `agent/scripts/acl_launch_program.py` 把每个 owner 归约成 (查询, execute 入口, 同步策略, 失败处理) token 流做改前改后 diff：**69/71 个 owner 的 execute 入口序列与同步策略逐字相同**，另 2 个（Random、reduce）是同一组入口重新分组，唯一系统性差异是失败处理收敛到 `LOGf`。门禁从计数式换成不变量式并要求两个 ACL 根各自非空（机制 `tests/_helpers/acl_launch_tails.py`，断言 `tests/structure/test_acl_launcher_contract.py`）；5 个反例全部报红，其中把 launcher 换成 workspace 查询只有 `--check-launchers` 挡得住。桩 SDK 44 个源文件 TU 全过、launcher ABI 断言 70 个。静态合同 102 passed。**设备侧零执行**：`tests/backends/npu` 本机 `164 skipped, 0 executed -- no acl found`，四条上机确认项与精确命令见 `agent/manuals/deferred-hardware.md` 的 Ascend/CANN 一节。**另两个 family 的实测现状**：`op_idx_map` 那条**已经闭合**（`448aa10a` 删除，全树 0 处引用，`test_acl_op_idx_map_is_removed_without_touching_reduce_dispatch` 钉住「删表但保留 `acl_op_exec.cc` 的显式 op_idx 9/13」，看板此前未记）。**只剩 `AclOpFunctions` 类型擦除**，实测面比登记的大：`acl_jittor.h` 现在 765 行，`std::function` 成员 **77 个**（登记写约 40）、构造重载 **39 个**、表项 **103 条**，`aclOpFuncMap` 仍是 `static` 定义在头文件第 344 行，**44 个 TU 各构造一份完整拷贝**。动它的连带面：它正是桩 SDK 下必然 ambiguous 的那个文件，`agent/skills/acl-host-syntax-check` 按文件名过滤掉了它的诊断，类型擦除之后那条过滤应当能删掉——这本身就是改对了的判据。无硬件依赖，本机可做到编译级 |
| 8.07 | conv 族共享描述符与计划层 | 已合并 | cudabk | 947f5223（反向只留 C++ 一份）+ 47f91130（计划请求一个构造函数） |
| 8.08 | `ProcessGroup` 对象替代全局唯一 communicator | 已合并 | dist | 82410549（NCCL env/file 与 MPI bootstrap 双卡通过；HCCL 对称实现未在 Ascend 真机验证） |
| 8.09 | NCCL | 已合并 | dist | f2d9c291, 95a1c956 |
| 8.10 | `distributed/launch.py:102-107` 改 `wait(timeout)… | 已合并 | dist | 925850b3 |
| 8.11 | 图同构优先：reduce 保留全尺寸输出，broadcast 去 rank 相关别名，flat 策略可配置 | 已合并 | dist | a1e769d5 |
| 8.12 | 算子内不再复用全局 jit key 缓冲做缓存键 | 待领 | cudabk | `988fd825`、`0f7046c8` 已将串行与并行 JIT cache lookup 从线程局部 JK 缓冲改为自有字符串键，静态合同通过；六条 cuDNN legacy cache 的 POD key、per-device 生命周期和完整 CUDA 验收仍待领。**前置 3.02 已解除**（`9cdae20fc`）：全局 JK 缓冲不再是「2 MB、无边界检查、靠守护页兜底」的东西，超长键抛可捕获异常而不是杀进程，所以「算子自带缓存键」这件事不必再兼容那个缓冲的溢出语义。**顺带**：3.03（`0ad6c7504`）留下了这条要的形状——`utils/jit_cache_map.h` 是一张有容量上限、有 LRU、键自有 `string` 的表，六条 cuDNN legacy cache 的 per-device POD key 可以直接用它而不必各写一份 |
| 8.13 | cuTT 计划未命中时的 `cudaDeviceSynchronize` 删除或降流同步 | 已合并 | cudabk | c0d2cc5c |
| 8.14 | Corex | 已合并 | 见备注 | 代码半已闭合（协调者 2026-09-07 复核）：前置 `4.12` 已合并，`process_acl` 全树 0 处，`corex_compiler.py` 里那份同名改写已随之消失；该文件现在**没有 `check()`**，只有只读的 `discover()`，路径经 `corex_home` 实参或 `COREX_HOME` 解析、默认 `/usr/local/corex`。**验收「探测无副作用」此前并未被证明**——原断言只比较一个临时目录**顶层**的 `os.listdir`，而 `check()` 当年的问题恰恰是去**跑**编译器，那条断言对进程派生、写模式 `open`、`os.environ`、深层文件写入全都是盲的。现已补成 `side_effect_recorder` 同时看这四类加 cwd 与递归文件树快照，并加 `TestTheGuardCanNoticeSideEffects` 把三类副作用注入 `discover()` 的副本、断言守卫报得出来；牙齿本身也验过（去掉守卫里的 env 比较后立刻报 `env side effect went unnoticed`，1 failed / 4 passed）。`tests/backends/corex` 5 passed。**本机无 Corex/Iluvatar 卡**，四条真机验收命令与判据已登记进 `agent/manuals/deferred-hardware.md` 的 Corex 一节 |
| 8.15 | 多机 rendezvous | 并入 多机硬件验收 | dist（2/N） | 已有 TCP/File/NCCL 单机前置；两机/HCCL/NCCL 子组与跨机失败验收按用户授权并入多机硬件验收。 |
| 8.16 | 多机启动器 | 并入 多机硬件验收 | | `torchrun --nnodes` 等真实两机启动参数和 runner 验收并入多机硬件范围。 |
| 8.17 | 跨机网络与诊断 | 并入 多机硬件验收 | | 跨机网络、掉线和 watchdog 需要两台机器，按用户授权并入多机硬件范围。 |
| 8.18 | 多机 checkpoint | 并入 多机硬件验收 | | 两机保存/四机加载和分片 checkpoint 真实验收并入多机硬件范围。 |
| 8.19 | 布局收尾 | 待领 | | |
| 8.20 | 手写 GroupNorm CUDA 多搬一份全尺寸中间量 | 待领 | | 2026-09-06 由 3.22 口径对齐后派生：forward 额外物化全尺寸 `xhat` 给反向，PyTorch 从 `X/mean/rstd` 重算。实测占归约类差距的 75%（1.5–1.8 ms），是 3.22 未达成的主因，而它不在代码生成里 |
| 9.01 | `import jittor` 不编译不下载 | 待领 | | 361d59b2、c4b21762、cf3835ee、51d0439f 已合入 4/N。1/N+2/N：native import 不再探测 Torch 或无条件调用 NCCL/cuTT/MKL setup；显式分布式请求仍 fail-closed，CPU float32 batched matmul 首次按需 MKL，只读 HOME 配合可写 JITTOR_HOME 可离线导入。3/N（cf3835ee）归因：看板此前的 1.332 s 是 `nvcc_path=""` 的 CPU-only 配置（复现 1.325/1.335 s），CUDA 配置同树是 2.457 s；「40.015 s 冷编译」复现 39.96 s/176 TU，但触发条件是**在 CUDA 配置已热的同一 JITTOR_HOME 里切到 CPU-only**（cfg 指纹不同），不是空缓存。1.325 s 里最大的一项是核心编译在无事可做时的固定开销 0.906 s（68%）：run_cmds 把 176 条命令发进 16 进程 Pool 做空转缓存校验 0.542 s、gen_jit_flags 纯 Python 剥注释扫 176 个 .cc 后写出逐字节相同的头 0.212 s、pyjt 0.104 s。4/N（51d0439f）：核心编译收进 `compiler.build_core()`，加构建戳（src/+extern/ 每文件 mtime_ns+size、编译要素、产物 stat、编译顺序，原子写），戳一致整步跳过、不一致走原完整校验。**热缓存 import CPU-only 1.332→0.413 s（达标），CUDA 2.457→1.545 s（未达标）**；剩余已归因为 extern 自定义算子 49 条命令空转校验 0.351 s 与无条件 import cupy 0.369 s。**冷缓存（空 JITTOR_HOME 68.3 s）与换配置两种情形不变，import 仍编译整个核心**，「移到显式 bootstrap 或首次算子调用」只完成前一半。离线只读 HOME+可写 JITTOR_HOME 两配置均可 import 且算对。门禁（基线 534d375d，三套均逐条 A/B）：tests/compiler 定向 8 passed；native 452→459 passed / 21→20 failed（多出的 7 passed 是新增测试，无新增失败）、torch shim 155 passed/10 failed 完全相同、structure 15 failed/491 passed 相同；CUDA `dtype_coverage` 6 passed、`test_torch_compat_cuda_tf32` 2 passed、`network_training_parity` 8 skipped（既有基线），`tests/backends/cuda` 64 passed/21 failed、`tests/ops/test_ops.py`（shim）172 passed/35 failed，改前改后逐条相同。四处进程 abort 全为**既有**且改前改后中止在同一 nodeid：native `test_complex64_linalg::test_svdvals`、torch `test_torch_compat_autograd::test_a_second_call_does_not_steal_the_first_calls_context`、CUDA `test_cudnn_rnn_dropout::TestCudnnRnnReserveSpace::test_reserve_space_is_queried_once_per_configuration`（即 156/157 波的 cuDNN RNN）、shim CUDA `test_ops.py` 在 `test_reference_getitem_step_float64` 之后。5/N（`d23f9bba`）：核心那份构建戳的做法推广成通用的 `product_build_stamp_path()`/`product_build_is_current()`/`compile_if_stale()`，`compile_custom_ops`（**公开 API，签名未变**）与 `libcuda_extern` 都改为戳一致整步跳过。**热缓存 import CUDA 1.28→0.80 s，达标**（同负载 12–15 对照，改前 1.226/1.253/1.279/1.286/1.378、改后 0.797/0.804/0.812/0.827/0.830）；CPU-only 0.52→0.38 s；热 import 的编译扇出 60 条命令→**0 条**。自定义算子那份戳记的是显式列出文件的 stat + 调用方各 `-I` 目录（含 `extra_flags` 手写的）递归扫描 + `core_source_signature()`；CUDA SDK 与树内目录故意不扫（前者由 cuda key 分区，后者已被 core 签名覆盖，照扫是 90 ms/次 import）。`2bf369dc` 的 `import cupy` 惰性化（0.369 s）此前已合入。另落地 `JITTOR_NO_BUILD=1` + `python -m jittor_utils.bootstrap`（含 `--check`）：import 路径上任何要编译的动作抛 `compiler.BuildNotAllowed` 并指名 bootstrap，`setup_cub` 的 `except Exception` 显式 re-raise 不再把它降级成警告。离线只读（只读 HOME + 可写 `JITTOR_HOME` + 代理指死端口 + `JITTOR_NO_BUILD=1`）CPU-only 与 CUDA 两配置均 import 成功且算对（sum 12.0、matmul 512.0），这次是**证明了没编译**而不只是 import 成功。三套门禁逐条 A/B（基线 `803e3785`，同机同核）：native `tests/compiler` 19 failed/355 passed → **18** failed/375 passed（唯一差异是 2.13 冻结 `jittor_path` 后一直红的 `test_source_signature_sees_same_size_edits_and_new_files` 已修，无新增失败）、native `tests/core` 21 failed/643 passed **逐条相同**、torch shim `tests/structure` 16 failed/869 passed **逐条相同**、CUDA `tests/backends/cuda` 42 failed/220 passed **逐条相同**、CUDA `tests/compiler` 19 failed/397 passed → 18 failed/417 passed（同一条既有失败已修，其余逐条相同）。定向：该文件 29 passed（native/CUDA/torch 三套各一遍）、`tests/distributed/test_distributed_init_failure.py` 4 passed（6.B04 显式分布式仍 fail-closed）、CUDA JIT matmul 与 cudnn/cublas 装载正常（4.12 的 `configure_accelerator_compiler` 未被绕过）。**缺口 2 仍未完成**：冷缓存与换配置下 import 照旧编译整个核心（本波实测换配置/戳失效 49.0 s / 199 TU，空缓存此前 68.3 s），因为 `compiler.py` 模块体里 `build_core()` 紧接着就是 `import jittor_core`，而 `flags`/`Var`/全部算子都来自那个模块对象，要真惰性化必须推迟这一句——整卡改动。核心编译**没有**落到首次算子调用（现首次算子调用仅 0.037–0.042 s，可作为将来搬过去的对照基线）。**热缓存两配置均已达标，但「移到显式 bootstrap 或首次算子调用」只完成可断言契约这一半，保持待领** |
| 9.02 | `install_cuda.py:113-122` 的 `os.execl` 自重启删除，用 d… | 已合并 | 构建 | 6b45c078 |
| 9.03 | 构建期失败一律抛带上下文的 `RuntimeError`，不用 LOGf/裸 assert | 已合并 | 构建 | 9197c8c6 |
| 9.04 | 依赖跟踪改用编译器的 `-MD -MF` | 已合并 | 构建 | 65a2dc12（clean_cache 从一份布局定义生成）、2569fe3b（依赖跟踪、SHA-256、主机名/`-march=native`/git 分支/路径哈希位数，一个提交只让大家重编一次）。依赖跟踪走的是「扫描器认识 `#ifdef`」而不是 `-MD -MF`：`process()` 兼着 JT_XXX 宏发现（必须在编译前）与依赖跟踪（只能在编译后），拆开才可能用 depfile，已登记为 9.21 |
| 9.05 | 下载安全 | 已合并 | 构建 | e111ebcc |
| 9.06 | 删 cutlass 下载 | 已合并 | 构建 | 50673d69 |
| 9.07 | import 过程不反向写环境变量 | 已合并 | build | `d141d8c2`：导入不再写 `os.environ["cc_path"]`，保留模块内 `jittor_utils.cc_path`；子进程回归 1 passed，完整 bootstrap 前16节点无失败。 |
| 9.08 | 新架 GPU | 已合并 | 构建 | 2d71f792 |
| 9.09 | `cuda_wheel` 失败时 LOG.w 出原因，strict 为默认 | 已合并 | 构建 | c63dd809 |
| 9.10 | 2.0 版本策略 | 已合并 | | 77dcc747 |
| 9.11 | release 的 platform-validation 阶段跑 selftest | 已合并 | 构建 | 2af4658e |
| 9.12 | `extern/rocm/rocm_cache.tar.gz` 的预编译 .o 改从源码构建，或… | 已合并 | 构建 | 46cc77d5（源码不在本仓库，做不到从源码构建；按任务允许的第二条补了来源说明与字节钉定，并写清要怎样才算可接受） |
| 9.13 | README 加「首次运行会发生什么」 | 已合并 | 构建 | dad3cd26 |
| 9.14 | 一次性的构建前置条件检查 | 已合并 | 构建 | b2bd11fd（审计那 17 个失败点：4 个可操作 → 15 个「是」、2 个「部分」、0 个「否」） |
| 9.15 | noxfile | 已合并 | 构建 | 84c7f766 |
| 9.16 | `agent/scripts/check_repo_layout.sh` 收缩为少数真会复发的检… | 已合并 | build | 94944d28。266 行降到 68 行，删除历史路径/根目录清单与全树 grep，保留运行时资源、生成物污染、实验目录、模块包冲突和文档治理；0.18 s，相关结构测试 20 passed |
| 9.17 | 死代码 | 已合并 | 构建 | f99250bb |
| 9.18 | `disable_lock=1` 启用时明确告警并纳入缓存指纹 | 已合并 | build | 801dd80d。启用时打印并发损坏警告并进入独立构建配置指纹；默认锁定配置保持原缓存名。相关两文件 28 passed |
| 9.19 | 布局收尾 | 待领 | | |
| 9.20 | asm_tuner 非原子写 .s，并发编译读到截断汇编 | 已合并 | build | 1919b035。`pass_asm()` 写进带 pid 的临时文件后 `os.replace`；inode 回归 1 passed，原四 worker Dataset 复现用例 1 passed |
| 9.21 | 拆掉手写预处理器最后一块：process() 双职责分离 + depfile | 已合并 | build | 9a5f4e7c 拆出 `JT_*` 宏声明；237d6460 删除手写 include scanner，GCC/Clang 用原子发布的 `-MD/-MF` depfile，asm/dlink 仅首段编译保留依赖参数，MSVC 走独立 `/showIncludes` 构造与解析（单元契约覆盖，未做 Windows 实机）。宏展开/失活条件修前 1 failed，修后定向 5 passed、C++ TEST 通过、CPU 9 passed/1 skipped、实机 CUDA `-dc` dlink 通过 |
| 9.22 | 并发编译同一个算子读到写了一半的 `.so` | 已合并 | build | c4bbdd72。`cache_compile` 对 asm/dlink wrapper 的最终产物也用私有临时名加 rename，`.key` 同样原子替换；修前两个 wrapper inode 契约均失败，修后 4 passed，冷 CPU 聚焦 6 passed，实机 CUDA 普通 JIT 与 `-dc` dlink 均通过 |
| 9.23 | `run_child_script(timeout=N)` 不收孙进程 | 已合并 | bindings | 17e43c9a（进程组 + `os.killpg` + 有界 drain）。**更正**：任务描述里"`communicate()` 继续等"在 CPython 3.11 上不成立（3.11 的 `subprocess.run` 超时后只 kill+wait，不重新 drain，已实测）；稳定复现的是整棵子孙进程留存，默认 `timeout=600` 的用例因此要等满 10 分钟才失败 |
| 10.01 | `tools/run_test_suite.py` 拆成 `nox -s full` 周期性调度… | 已合并 | gates | `5501d0b6` 加入稳定 `nox -s full` 完整 CPU 门禁入口，保留 `cpu` 兼容 session；CPU workflow 改调用 `full`，结构合同确认委托和调度入口，定向 2 passed |
| 10.02 | 默认 `nox` 含 cpu 数值测试，或把默认改名为 static | 已合并 | gates | `151c5856`：`nox.options.sessions` 默认列表加入 `cpu`；新增 AST 结构合同确认默认数值门禁存在，定向 1 passed |
| 10.03 | optional/rocm/mpi/nccl 四个 session 排上 runner 或在文档… | 已合并 | gates | a1668aca。CUDA 可由维护者添加 `ci:cuda` 标签触发 PR 真机门禁；当前 runner 能力不覆盖 optional 依赖、ROCm、MPI 与双卡 NCCL，四项在测试支持矩阵中明确为 Manual，结构规则防文档/调度漂移。相关结构 22 passed |
| 10.04 | 假绿清理 | 已合并 | gates | 74cace5f。6 个首行 `return` 改严格预期失败并登记，4 个 `skipIf(True)` 清零；两条内存契约用短循环 RSS 上限进入 slow 层，负向自测证明真实保留会失败；AST 全树规则禁止复发。内存 2 passed，规则/负向 9 passed，旧禁用项 3 xfailed、4 prerequisite-skipped |
| 10.05 | 按 skip 原因分桶统计并在 CI summary 输出，对「本环境应能跑却 skip」设阈值 | 已合并 | gates | `1a423a16`：`tests/conftest.py` 按固定优先级（accelerator/backend/mpi/torch/network/manual/other）汇总 skip reason，CI summary 输出稳定 bucket；`JITTOR_TEST_REQUIRE_EXECUTION=1` 下 `other>0` fail-closed。`tests/structure/test_gate_scope.py` 合成重叠/unknown/阈值节点 2 passed。 |
| 10.06 | `expect_error` 带 `exc_type` 与 `match` | 已合并 | gates | 6753062d 严格 helper；49503f95 与 01536ba4 为全部 34 处旧调用固定异常类型/消息，AST legacy 计数 0，并修复 CUDA `Var` 误调用导致的假绿；796b5338 增加 OpInfo `ErrorInput`/生成式错误电池与覆盖率门禁，46/227（20.3%）。聚焦 helper 4 passed、六个最终调用节点 6 passed、OpInfo 错误电池 47 passed，真实 CUDA 负向节点 1 passed |
| 10.07 | Unary/Binary/Reduction 用 `OpDTypes.supported` | 已合并 | gates | 4af5fbcd。TestCommon 覆盖每个 OpInfo 声明的全部 dtype，BF16 输入保持原生 bfloat16；两条修前契约各失败，修后输入生成 7 passed |
| 10.08 | 已复现缺陷用 `xfail` 而非 `skip` | 已合并 | gates | d7f87e28。OpInfo `xfail` 改为 `pytest.mark.xfail(strict=True, raises=...)`；fft/ifft/rfft 不再以 `supports_autograd=False` 静默绕过，六个 CPU gradcheck/gradgradcheck 节点稳定复现 float64 输出无法 reinterpret 为 complex64 并全部 xfailed，monkeypatch 修复探针产生 XPASS(strict) 且退出非零。NPU crash/hang 隔离 skip 与数学、数值 harness、环境前置 skip 保持不变 |
| 10.09 | 公开 API 与 OpInfo 差集作为 structure 门禁一项 | 已合并 | gates | 8b76a79f。纠正审计把 stub 类方法/重载混进 536 个公开算子的口径；计划点名的 12 个高频项现全部结构化归属：nonzero/unique/einsum/rms_norm 新增独立 NumPy OpInfo，其余 8 项绑定可解析的现存 nodeid 与不适用通用 OpInfo 的理由。门禁修前准确报 4 项缺口，修后 1 passed；四个 CPU reference 节点 4 passed。注册表现为 231 实例/204 distinct name |
| 10.10 | gradcheck 加「故意写错导数应当失败」的负向自测 | 已合并 | gates | 3e83594d。故意把平方的 backward 写成 `3*x`，`gradcheck` 必须抛 `Jacobian mismatch`；定向 1 passed，相关结构 15 passed |
| 10.11 | 设备对拍加 dtype 轴 | 已合并 | gates | 4bf5830c。双方支持时生成 float32/int8/int16 轴，整数逐位比较，浮点容差按 `sqrt(reduce_size)*eps` 下限缩放，CuPy linalg 探针失败硬失败；真实 CUDA sum-int8/max-int16 2 passed |
| 10.12 | `retry` 装饰器记录并上报重试次数 | 已合并 | gates | 402d09ef。恢复成功与最终失败均报告准确 retries/attempts，并暴露调用、最近和累计重试计数；保留原异常并支持 kwargs。聚焦结构 8 passed |
| 10.13 | marker 真正建立 `-m "not slow"` 快门禁或删除 | 已合并 | gates | 821bb6ba。新增 AST 合同确认 smoke 的 native/torch 两次 `_run_pytest_once` 均传入 `-m not slow`，且 `SLOW_FILES` 仍被 gate 覆盖；定点结构节点 1 passed |
| 10.14 | notebook 门禁按 topic 参数化 | 已合并 | gates | 828bc272。fence/materialize/CPU smoke 生成独立 topic nodeid，smoke 共享模块级缓存；33/117 个 skip-execution 单元均带原因标签且低于 35% 上限。50 tests collected，结构/标签 20 passed，单 topic materialize 1 passed |
| 10.15 | 速度 harness 记录并断言两侧线程数、亲和掩码与精度策略 | 已合并 | gates | 9047897a。runner报告实际线程环境、affinity、runtime线程数与精度，harness要求两侧一致；速度类默认至少10次。纯结构契约3 passed、运行文件语法检查通过，未执行大模型 |
| 10.16 | 提供计时 API | 已合并 | pyother | f9f3a23d。`jt.benchmark` 冻结并预先物化输入池，至少一次 warmup 不计时，每轮递归保留 tuple/list/dict 全部输出并强同步后采样；无 Var 输出直接拒绝，返回不可变秒级统计。CPU 回归 3 passed，覆盖跨轮输入复用、嵌套输出、CSE/死码与未物化假快 |
| 10.17 | 异步错误 | 待领 | | 前置 3.01 结构拆分已落地：发射循环与它的异常处理（`check_op_async_error`）现在整段在 `exec_runner.cc`，环形缓冲的写入点就是那个循环 |
| 10.18 | 结构测试预算转向核心 | 已合并 | `d6f17450`、`c586b7fc7` | `d6f17450` 增加 native CPU gate 对 traversal/liveness/autograd 核心属性测试的结构合同，定向 11 passed；这是门禁前置，结构预算整体迁移仍待领。前置 3.01 的结构拆分已落地：「执行器计划的属性测试」现在有了可断言的对象——`ExecPlan`（`exec_plan.h`）是纯下标的值类型，拓扑序、段划分、`range` 与 `fuse_ops` 的一致性都能脱离设备直接断言。**2026-09-06 完成：验收原话「核心属性测试进 CPU 门禁」的三类（图不变量、liveness、执行器计划）全部落地且在原生 CPU 门禁里跑。** 先量（独立提交 `5fbeb8847`，`tools/measure_core_test_balance.py`）：核心 C++ 329 文件 44934 行、`tests/core` 81 文件 14374 行，比值 0.20 → **0.32**，但**构成才是问题**——按实际点名的 API 归类，dtype/数值 69 文件 12918 行 684 用例，而图与 liveness 只有 **10 文件 2328 行 141 用例**、执行器只有 **7 文件 1397 行 68 用例**；翻倍长出的七千行几乎全在数值面上，所以「多写 tests/core 的行」不是解。另报下界：179 个核心头文件里 **14 个**在整个 `tests/` 加 `src/tests` 里没被任何文件点名。**位置选择是本条最可复用的一点**：`src/tests/*.cc` 的 `JIT_TEST` 由 `gen_jit_tests()` 自动变成 `jt.tests.*`、再由 `tests/compiler/test_jit_tests.py` 自动变成 pytest 节点，而 `tests/compiler` 0.04 之后已在门禁里，**不用改任何门禁配置**。新增 `test_exec_plan_properties.cc`（9 条）与 `test_node_liveness_properties.cc`（5 条），`jt.tests` 72 → **86**；执行器那 9 条断「计划自身一致且按它声明的顺序可执行」（下标范围、`range`/`fuse_ops`/`queue` 自洽无空段、每个算子**至少**排进一个段、段间与段内均为拓扑序、每个批内 var 恰好一个批内生产者、建计划不执行任何东西、同图两次同计划），**刻意不录快照**——`count_fuse` 有权改主意什么该融合，快照会把启发式冻住并把一次正常调整变成一片红。图与 liveness 在 `tests/core/test_core_invariant_properties.py`（11 条），含 `dump_all_graphs` 边对称/下标范围/无环，以及 `graph_check` 必须**区分「校验通过」与「跳过了」**（6.C21）：标志打开前返回 0、打开后新建节点返回 > 0，并钉住「`swept` 不是进程存活节点数」这条既有限度（实测 10 扫过对 17 存活）。**抓到真的不变量违反**：详见本表 `2.10` 行，缺陷未修（属他人范围且 `var_holder.cc` 被 5.02 占用），用 `..._leaks_nothing_new`（绿，多一个形状就红）加 `..._leaks_nothing_at_all`（`xfail(strict=True)`，修好即 XPASS 变红强制回访）两条落地，再加一条钉住「每次恰好 2 个」。**层内成本：C++ 15 条整文件 2.49 s、每条 < 5 ms、边际约 0.05 s；Python 11 条 1.88 s（1.38 s 是那一个子进程）；合计约 1.9 s，对 native 半边 1592.9 s 层内工作量为 +0.12%。两个都进 smoke，不进 nightly**，按 gate-tier-budget §2「默认包含」；**本波没有新增任何推迟项**。三套门禁：原生 CPU 51 failed/1031 passed/172 skipped/2 xfailed，CPU torch 15 failed/918 passed/2 skipped/2 xfailed，CUDA 42 failed/231 passed/35 skipped/1 xfailed（42 failed 与 3.01 基线逐数相同，passed 220 → 231）；CUDA 推前烟测通过。**「没把旧结论挤掉」是机械判定**：`gate_conclusion_diff.py` 在同一次构建、同一条命令上做 A/B（baseline 只 `--ignore` 新 py 文件、`--deselect` 14 个新 C++ 用例），两轮 1045.9/1038.7 s 都是热的，结果 `collected 1231 → 1256`、`failed 51 → 51`、`skipped 172 → 172`、`passed 1007 → 1031`、`xfailed 1 → 2`，带 `--expect-new` 点名那 25 个新用例后 **`IDENTICAL` 退出 0**，不带时报 25 条差异且非零退出（默认保持严格）；**没有一条 `STATUS x -> y`、没有一条 `CONCLUSION LOST`**。torch/CUDA 失败清单里没有一条点到本波新文件，`tests/structure` 那 5 个扫描类失败逐条查过点的都是既有项。**⚠ 一条自我更正**：第一次留的原生基线是**冷缓存**的（79 failed/979 passed 对改后 51/1031），28 条「差异」全是 `test_import_bootstrap_laziness` 等「热缓存下不该重编」类断言，机械证据是基线日志有一行 `Compiling jittor_core` 而改后 0 行——起因是预热用了 `nvcc_path=/usr/local/cuda/bin/nvcc` 而门禁口径是 `nvcc_path=""`，**那是另一个 cfg 缓存目录**。唯一同 nodeid 的真差异是 `test_relu_memopt` 两轮都红、数字 75 → 43，即 3.01 警告的存活 Var 绝对断言漂移。**顺带补掉判据工具的两个缺口**：（1）`gate_conclusion_diff.py` 新增 `--expect-new`，否则加测试的人只能用眼睛扫差异列表；修前 `unrecognized arguments` 退出 2，回归 `test_gate_conclusion_record.py::TestExpectNew` 4 条（含「既加了测试又丢了旧结论必须仍非零退出」的反向用例）。（2）**用它时量出插件把被 deselect 的 nodeid 也算进 `collected`**：钩子用的是 `pytest_collection_modifyitems`，而插件在 `pytest_configure` 里注册、**先于 pytest 自己对该钩子的实现**跑，`--deselect`/`-k`/`-m` 正是在那个实现里剔项的，于是 14 个被 deselect 的用例被逐条报成 `COLLECTED BUT NO CONCLUSION`（实测 `collected=1245 concluded=1231`）——**核心信号为无害的事报警，会被读者学会跳过**，且让串行与 xdist 两种记录不可比（xdist 那条钩子一直报 deselect 之后的 ids）。改用 `pytest_collection_finish` 读 `session.items`；回归 `TestDeselectionIsNotCollection` 3 条**修前 3 failed、修后全文件 10 passed / 7.71 s**。沉淀 skill `core-invariant-property-tests`，并给 `gate-tier-budget` 加 §1.2bis（`nvcc_path` 换的是 cfg 缓存目录）与 §5 的 `--expect-new` 用法。**把 C++ 桥接接进门禁当场兑现了审计那句「零执行」的代价：变基到 `e5e353644` 后第一次跑就有 4 条红，全在 `test_op_register.cc` 的 native provider registry 系列（另一分区 `c1a67c91b`..`b8398291b` 那串重构带进来的），且在任何门禁里从未被执行过。** 已排除是本波造成：单独只跑这 4 条（`-k native_op_registry`，其余 84 条 deselect）同样 4 red，与我新增的 `PlanProbeOp` 是否先注册无关；两条提交也都没碰 `op_register.*`（`git diff e5e353644 HEAD` 对这些路径为空）。三个互相独立的根因，断的都是 registry 自身的行为：`test_op_register.cc:267` 观察者收到的生命周期事件过不了自己的 `event.valid()`（`..._lifecycle_consumer_boundary` 与 `..._scopes_transfer_teardown_ownership` 撞同一句）；`op_register.cc:210` 的 `..._provider_dispatch_boundary` 把 op 注册进**局部** `NativeOpRegistry` 却用自由函数 `get_op_id()` 查**全局** `op_registry()` 单例，名字不可能在那里；`test_op_register.cc:414` 的 `..._registration_scope_is_identity_checked` 在 scope 退出后被替换的 provider 仍注册着。**本波不修**（`ops/op_register.{h,cc}` 属他人工作集，且三条断的是 registry 行为不是脚手架），处置是 `test_jit_tests.py:KNOWN_BROKEN_TESTS` 隔离成 `xfail(strict=True)`；**strict 是要点**，用 skip 则对方修好这边毫无反应、隔离永久留着，strict 下修好即变红逼着删条目。反向也实测过：把本来就绿的 `op_register_reads_and_writes_the_same_key` 临时写进隔离表，报 FAILED（XPASS strict）而非放过。另加 `test_the_quarantine_names_only_tests_that_exist`，防用例改名后隔离条目空指向。隔离后该组合 **98 passed / 2 skipped / 5 xfailed / 2.99 s**（5 = 这 4 条 + liveness 那条理想不变量）。**剩余**：`test_op_register.cc` 那 4 条待 registry owner 修（修好即在此变红）；2.10 的 backward 多释放未修、`test_zmem_leak{,2,3}` 仍红；**没有削减 `tests/structure`**——前一波已实测那不是 smoke 的杠杆（9.4%），0.19 的「< 2000 行」是另一个目标、由 0.25 承接改判 **协调者判定已合并（2026-09-07），三项证据都自己重跑过。** 验收原话是「核心属性测试进 CPU 门禁」，不含削减行数：(1) `tests/structure/test_gate_scope.py` **14 passed**，含把核心属性测试钉在 CPU 门禁里的那条合同；(2) `tests/core/test_core_invariant_properties.py` + `test_traversal_state_isolation.py` **16 passed / 1 xfailed**；(3) `tests/compiler/test_jit_tests.py` **88 passed / 2 skipped / 4 xfailed**，四条 registry 缺陷确实是 `xfail(strict=True)` 隔离——strict 是要点，owner 修好即变红逼着删条目，用 skip 则永久静默。**最后一条阻塞理由已消解**：备注里「没有削减 `tests/structure`」指向的是 `0.19` 的「< 2000 行」，而那条验收已由 `0.25` 改判删除（换成「枚举必须双向报红」，理由是行数分不清公开 API 快照与过期清单）。**两处剩余明确属他人**：`test_op_register.cc` 那 4 条归 registry owner（已 strict 隔离，修好即在此报红）；`2.10` 的 backward 多释放与 `test_zmem_leak{,2,3}` 归 `2.10`。本条不再等它们 |
| 10.19 | 每个带 `grad()` 的后端算子有对 CPU 参考的梯度单测 | 代码半闭合，余项并入 硬件验收 | `a55d66b6d`、`e5eaacfc2`、`c6b9e967f` | 清单齐全：枚举 **60** 条后端梯度（C++ `::grad()` 28 + Python `jt.Function.grad` 32），与源码树逐条相等。原 26 项 inventory 漏了 34 条——扫描根只有 `backends/cuda/kernels` 与 `python/jittor/extern`（`backends/rocm/libraries` 两条在根之外），且只认 `.cc`（`backends/acl/kernels/ops` 23 条与 `backends/cuda/kernels/nn` 9 条 Python 梯度在语言之外），而断言写的是「总数 == 26」，漏掉的部分同时不在分子也不在分母；这与 `10.21` 记的「后端搬进 `backends/` 后一条没扫到而总数看着健康」是同一形状。现扫描根按后端拆开并**逐个断言非空**，corex 单列为「应为空」但同时断言目录真实有源文件。**24 条本机真跑**（CUDA 22 + oneDNN 1 + CPU 1），**36 条硬件延迟**，七种 `kind` 全部登记进 `agent/manuals/deferred-hardware.md` 新增一节。CUDA 侧唯一缺口是 `softmax_cuda.py` 两条 streaming 反向（既有用例最长 2049 列只走 register 核），已补 6 长度 × log/plain 共 12 组对 float64 与 CPU 双参考；其余 22 条 CUDA 梯度逐条对 CPU 重跑，**未发现梯度 bug**。**两个会让对拍静默失去分辨力的数值坑**：余切必须取正（标准正态余切下行归约抵消，往 streaming 注入 0.1% 误差仍全过，改正余切后立刻报红）；131072 列上 CUDA 反向对 float64 偏差 3.3e-5 而 jittor CPU 是 1.5e-1，长行上拿 CPU 当紧容差参考会得出「CUDA 核不准」的反向结论。反例在真树上双向验过：加假算子（`.cc` 与 `.py` 各一）报红、移走 `rocprim_cumsum_op.cc` 报红、扫描根写错报红。三套门禁：原生 CPU 19 passed/3 skipped、`JITTOR_TORCH_SHIM=1 tests/structure` 15 failed/922 passed（与基线同 15 条，无一属本任务）、`tests/backends/cuda` 5 failed/274 passed（基线 269，同 5 条既有失败）。**余项属真窟窿而非单纯缺卡**，七条即使有卡也测不到反向：`HcclAllGatherOp::grad()` 仍 `LOGf << "not implemented"`；`RocprimCumsumOp` 全树无任何用例；`FloorIntACL`/`IndexACL`/`NonzeroACL`/`StackACL`/`TriuACL` 只有前向用例。七条逐个列名并由门禁钉住，补齐前删不掉。 |
| 10.20 | 给测试提供受支持的内省 API，替代 283 处 `jt.flags.*`、137 处 `com… | 待领 | | **本波（pyops）未实现，只做了数字复核，形状因此改变；记录在 `043d19be2` 与 `codebase-audit/07-architecture.md`。** 注意：上一波留下的未提交工作曾被当作本条，实际是 `2.22`（新增文件自己的标题写的就是 `[2.22]`），已按 `2.22` 收口，本条仍待领。**三个数字全漂了**，两个向上一个向下，测试树本身从 357 个文件长到 631 个：`jt.flags.*` 审计 283 → 实测 **653** 处（134 个文件，其中 **266 处是写**、387 处是读，单 `use_cuda` 就 **233** 处）；`compile_extern.*`／`jt.compiler.*` 审计 137 → 实测 **201** 处（92 个文件）；触碰下划线名或 `__dict__` 的测试文件 审计 127 → 实测 **64**（唯一变好的一项，`_torch_*` 影子属性清理的结果）。**复核同时改变了这条任务的形状**：原条目把三者当同一个问题，实测是三个——（1）`use_cuda` 的 233 处里绝大多数问的不是「当前运行目标」而是「这台机器能不能跑加速器」（`if not jt.flags.use_cuda: skip`），这是**能力查询**，`2.13` 的 `jt.config`／`jt.runtime` 报告的是策略而不是能力，**答不了它**，这才是本条真正缺的东西；（2）266 处**写**按定义不属于只读的内省 API，它们要的是可恢复作用域，即 `2.13` 已交付的 `jt.runtime.scope`，本条不该为它们再开一组可写入口；（3）201 处 `compile_extern.*` 全是「这个库在不在」（`compile_extern.cudnn_ops is not None` 这类），也是能力查询，而 `compile_extern` 的属性是 `globals()` 注入的，静态分析、`.pyi`、IDE 全部看不见。**所以内省 API 的形状应当是建在 `jt.config`／`jt.runtime` 之上的只读三层**：能力（后端/库/设备可用性，替代第 1、3 类）、有效策略（转发 `jt.runtime.context` 与 `jt.config`，替代 387 处读）、计数（`exec_called`、`stat_allocator_total_*`、存活 Var/Op）；写一律不进这个命名空间。**障碍**：这是开放式替换面（226 个文件），单波关不掉，接手前先决定「本波只交付 API 加能力层，还是同时迁调用点」，后者需要按目录分批 |
| 10.21 | import 方向做成 lint 规则 | 部分合并 | mem | `a82fd5b9` import 环门禁 + `7627bc0ae` mypy 扩包。**环这项已闭合**：审计记的三个真环实测一条都不在了（`jittor_utils`⇄`compiler` 无任何框架 import；`var_holder.h` 不再 include `executor.h`；`node.h` 无 pyjt/tracer），实测还在的是 Python 模块级 3 个 SCC 共 164 模块（157 `jittor` 包门面／4 `jittor_utils` 内部／3 vendored `einops`）与 1 个 42 头的 ACL C++ 头环。`tools/lint/check_import_layering.py` 三条契约接在 `tests/structure/test_import_layering.py` 与 `nox -s imports` 上，扫 384 模块（backends/ 86）、996 边、5 个 package-dir 根无一为空；三条反例在真树上各报红一次并点名。不用 import-linter：grimp 静态解析看不见 `backends/` overlay，实测只见 294/384 模块、`jittor.backends.*` 仅 1 个，会零违规通过。**mypy 这项部分推进**：7 个文件 → 34；本波只扩 `python/jittor_utils` 整包（26 模块，按目录写入 files 故新模块自动进覆盖），29 条错误全按真实类型修掉、无 `# type: ignore`、未放宽配置；顺带修好 base 上就红的 `nox -s typing`（noxfile.py 13 errors → 0）。**剩余面**：`python/jittor` 2514 errors／165 files（checked 269），`backends/` 79 errors／32 files（checked 87），合计约 2593 条／197 文件待清 |
| 10.22 | 多机门禁 | 并入 多机硬件验收 | | 两节点 DDP/FSDP 和掉线门禁按用户授权并入多机硬件范围。 |
| 10.23 | 布局收尾 | 待领 | | |
| 10.24 | fixture 契约按真实来源解析 | 已合并 | gatecheck | `c8ce8760`。原判据拿函数参数与一份只含 pytest 内置 fixture 的冻结清单比较，7 处使用自有 fixture 的合法测试全被误判成「命名成 test 的 helper」；改为按真实来源解析（内置 + 本模块声明 + 沿树 conftest + 该函数的 parametrize argname）。修前 27 passed 1 failed、修后 28 passed；另造反例（参数无人提供的 `test_*` helper）确认判据未被放宽 |
| 11.01 | 删已被取代的绕过与死路径 | 已合并 | gatecheck | 逐项核查（无需新代码，六项在各自前置里已经完成）：`nn/backends/cudnn.py` 的 `_CudnnConv2d` 全树 0 处（8.07）；`change_function()` 0 处（4.11）；`process_acl` 0 处（4.12 今日的 ACL/ROCm 源转换移除）；`asm_tuner.py` 文件已删且引用 0 处；`var_holder.cc` 的十层硬编码下标链已换成 `cascade_setitem_root` 的循环加 `needs_cascade_setitem()`；`event_queue` 无「cause hang」标注、`executor.cc` 无注释掉的 `run_sync`，且有 `tests/structure/test_event_queue_contract.py` 钉着（3.19 选的是「修好并加测试」而不是删除）。**唯一剩余项 `process_jittor_source` 交由 4.12**：它现在只有 `compiler.py:61` 一处赋值给 `BuildContext.transform_sources`，而 `transform_sources` **零消费者**，即已成死代码，可安全删除 |
| 11.02 | 已提前为 0.20 | 并入 0.20 | | |
| 11.03 | 单文件异常拆分 | 待领 | | |
| 11.04 | 关键接口写成显式契约 | 待领 | coreops | `8e1ad4bd`、`fa2d8523`、`029fa9a4`（随 3.01）已交付 `executor.h` 那一份：79 行，写明一次批的承诺、执行序按 `Op::order`、`device_sync`/`weak_sync` 语义、不可重入、两个分配器与 `last_is_cuda` 的所有权；新增的 `exec_plan.h`/`exec_runner.h` 同样以契约起头。**仍待**：`allocator.h`（58 行下挂 8 个实现）、compat 28k 行、`Installer`/`Backend` 协议类型统一 19 处 install 与 7 处 check |

## 增量证据（按波次）

### 第101波增量证据（2026-09-04）

- `2.19`：`ba2f4077` 将 cuBLAS matmul 内维不匹配从 `ASSERTop` 改为可捕获的 `USER_CHECKop`，补负向结构合同；定向 3 passed。2.19 仍是聚合任务，未改为已合并。
- `7.03`：`94df46f7` 将 `complex`、`view_as_complex`、`view_as_real` 提升为 numerical 模块级稳定对象，登记 approximate fidelity；CPU identity/metadata/value 定向 2 passed。其余 tensor/nn/module family 仍待领。
- `0.15`：RingBuffer 修复后，独立 Dataset worker 监管两个 nodeid 在临时缓存下 2 passed/65.68 s；完整 smoke 仍约 390 s，任务保持待领。

**7.08 只做了三分之一，另两项仍待领**（兼容层分区，2026-09-03）：

- 已做：**`torch.backends.*` 映射表格化并单测** — `9aaedba9`。六种拼写合成两条状态。`fp32_precision` 原本是四个 backend 对象上的字面量 `"ieee"`（`_PrecisionBackend` 的**类属性**），读不反映 tf32 已打开、写它什么都不做；`get_float32_matmul_precision()` 读的则是一个`matmul.allow_tf32` 从不更新的独立字符串。四条缺陷都用探针在旧实现上逐条实测复现过，不是推断。
- 未做：**`torch.dtype` 改真正的对象** — **未动，且不建议顺手做**。`types.py` 的 `class dtype(str)` 里 str 继承是**承重**的，文件自己写明了理由：jittor 的 C++ 类型分发构造器要求 str/NanoString；而且 jittor **自己的 Python 代码**（`contrib.concat`、`linalg`、`nn`）会`str(var.dtype)` 再把结果直接喂回 C++ 分发。所以「入口处一次转换」要求先把**每一个** dtype 跨进 C++ 的边界找全再改；改一半会让错误的 dtype 静默流进算子。这是本任务里唯一一条「做一半比不做更糟」的，应整块领、单独排期。
- 未做：**占位 dtype 参与计算时抛 `NotImplementedError`**。占位清单在 `types.py:_make_dtypes` 的 specs 里已有注释标出。难点不在识别而在拦截点：这些 dtype 对象**必须**继续存在且可作字典键（transformers/safetensors/torchao 在 import 期就按它们建表），所以只能在「真的参与计算或分配」那一步抛，不能在被引用时抛。


**7.13 已合入四部分，其余待领**（兼容层分区，2026-09-03）：

- 已做：`37c0aed4` 按实现身份识别优化器；`c0e6e1ae` 修复 Torch 模式下 NCCL preflight；`48da7360` 拒绝未遵守的 mesh 并跨 rank 归约梯度范数；`873dd5cf` 在分片后释放聚合的 `full_param`。
- 未做：释放后峰值仍未低于未分片，每步仍增长 16 个分片大小的 Var；需继续沿已证实的引用环线索定位。
- 未做：`optimizer.py` 仍自行实现 SGD/Adam/AdamW 数学，没有做到“复用 Jittor optimizer、只替换梯度来源”。
- 未做：DeviceMesh 真实分组与多维切片依赖 8.08，当前仍明确拒绝。


### 2026-09-04 第四十三波补充证据

- `8.06`：`1e8e90c6` 为 `aclnn.h` 增加 `#pragma once`，新增重复包含静态合同，1 passed；本机无 CANN/NPU，仍待 Ascend 910B3 实机。
- `2.19`：`45f77257`/`f76e3b90` 将 CUB argsort/arg_reduce 的 offsets dtype 边界改为 `USER_CHECK` 并记录 int64 负向与双 nvcc TU 语法通过；本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`8647dc4d` 将 `pairwise_distance` 提升为模块级稳定对象并登记 conservative approximate fidelity；身份、metadata、CPU p=2/keepdim 三节点通过。

### 2026-09-04 第四十四波补充证据

- `8.06`：`553b5ec1` 将 SiLU forward owner 接入共享 launcher，backward/Swish/SwiGlu 保持原路径；结构合同 31 passed，本机无 CANN/NPU，仍待 Ascend 910B3 实机。
- `2.19`：`a3890dd9`/`b8e1f592` 将 cuDNN convolution forward 格式边界改为 `USER_CHECK`；结构合同与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录，累计 80 处。
- `7.03`：`4a31179c` 将 `cosine_similarity` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，三节点动态测试因首次 JIT 编译过久终止，未宣称通过。

### 2026-09-04 第四十五波补充证据

- `8.06`：`600ee169` 将 BatchMatMul 接入共享 launcher，保留 `cube_math_type` 与同步策略；结构合同 32 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`67e710b7`/`aae33e6a` 将 cuDNN convolution backward-x 格式边界改为 `USER_CHECK`；结构合同与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录，累计 81 处。
- `7.03`：`93cd6a53` 将 `svd` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，动态三节点因首次编译过久终止，未宣称通过。

### 2026-09-04 第四十六波补充证据

- `8.06`：`8251d29d` 将 RotaryPositionEmbedding forward 接入共享 launcher，保留三输入与同步策略；结构合同 33 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`0a0e820e`/`aad3ba0c` 将 cuDNN convolution backward-w 格式边界改为 `USER_CHECK`；结构合同与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录，累计 82 处。
- `7.03`：`dc8cdfcb` 将 `svd_lowrank` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第四十七波补充证据

- `8.06`：`71bab738` 将 Maxpool forward 接入共享 launcher，保留 descriptors、`poolCeil`、同步策略及 Avgpool/backward 原路径；静态合同 34 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`37004fe0`/`b48f8af7` 将 cuDNN conv3d 输入 rank 边界改为 `USER_CHECKop`；结构合同与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录，累计 83 处。另：`broadcast_to` 源码实际 5 个检查，但 dimension map 仍期望 2（shape map 期望 5），相关门禁仍 1 failed，待专门修复。
- `7.03`：`cfe67a7e` 将 `pca_lowrank` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第四十八波补充证据

- `8.06`：`16a89606` 将 Avgpool forward 接入共享 launcher，保留 descriptors、`poolCeil/divisor`、同步策略及 backward/其他 pool owner；静态合同 35 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`9d77a5a7`/`cf177243` 将 cuDNN conv3d backward-x 权重 rank 边界改为 `USER_CHECKop`；结构合同与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录，累计 84 处。
- `7.03`：`727b440a` 将 `nan_to_num_` 提升为 numerical 稳定 in-place 对象并登记 conservative approximate fidelity；`py_compile`/diff-check 通过，因既有 NaN/Inf JIT abort 风险未运行动态测试。

### 2026-09-04 第四十九波补充证据

- `8.06`：`ba8e2621` 将 TruthReduce all/any 接入共享 launcher，保留双路径异常处理与同步策略；静态合同 36 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`e81ef514`/`a7f45f1f` 将 cuDNN conv3d backward-w 输入 rank 边界改为 `USER_CHECKop`；结构合同与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录，累计 85 处。
- `7.03`：`602a813f` 将 `sparse_coo_tensor` factory 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态测试未运行。

### 2026-09-04 第五十波补充证据

- `8.06`：`230c0b69` 将 Conv2d forward 接入共享 launcher，保留 group/bias/descriptor 与同步策略，backward 不变；静态合同 37 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`496dd510`/`e7b10858` 将 cuDNN conv3d backward-x dy rank 边界改为 `USER_CHECKop`；结构合同与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录，累计 86 处。
- `7.03`：`32064314` 将 `randint_like` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十一波补充证据

- `8.06`：`e86ccd11` 将 RmsNorm forward 接入共享 launcher，保留 `eps`、双输出与同步策略，gradient owner 不变；静态合同 38 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`f07cb966`/`54a88b42` 将 cuDNN conv3d backward-w 的 dy rank 边界改为 `USER_CHECKop`；结构合同与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录，累计 87 处。
- `7.03`：`d9c7c6a2` 将 `det` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十二波补充证据

- `8.06`：`faf6745e` 将 RmsNormGrad 接入共享 launcher，保留多输入、双输出与同步策略，gradient owner 不变；静态合同 39 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`85ae0688`/`b5e00107` 将 cuDNN conv3d 权重 rank 边界改为 `USER_CHECKop`；结构合同与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录，累计 88 处、四十一组证据。
- `7.03`：`9c469b37` 将 `inverse` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十三波补充证据

- `8.06`：`3581db5d` 将 Softmax backward 接入共享 launcher，保留 `dim` query 与同步策略；静态合同 40 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`1910f343`/`53db0066` 将 CUB argsort 的 x/indexes rank 边界改为 `USER_CHECK`，累计 89 处、四十二组证据；结构合同与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`8bc2791e` 将 `take_along_dim` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十四波补充证据

- `8.06`：`5697f619` 将 Embedding backward 接入共享 launcher，保留 `numEmbeddings`、`paddingIdx`、`scaleGradByFreq` 与同步策略；静态合同 41 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`4fe6f687`/`4f605d00` 将 CUB argsort 循环内 x/indexes shape 边界改为 `USER_CHECK`，累计 90 处、四十三组证据；结构合同与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`48c6fd73` 将 `log1p` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 看板一致性修复与并发补充

- 修复 7.14 之后任务表的结构：补回计划中遗漏的 `0.22`、`2.24`、`7.19`、`7.20` 四行，移除 8.06 下只有两列的 `8.06 note` 伪行；增量证据继续集中在本文件末尾的波次小节。
- 主线已包含的并发提交：`166010a8`（CUB argsort offsets rank，2.19）与 `ccbc6132`（`reciprocal` 稳定对象，7.03）；两项保持原任务“待领”状态，作为前置证据记录。

### 2026-09-04 第五十五波补充证据

- `8.06`：`0b149241`/`a12a2fbe` 将 Dropout backward 接入共享 launcher，保留 `scale` query 与同步策略；静态合同 42 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`166010a8`（CUB argsort offsets rank）已进入主线，结构/TU 证据已记录，累计 91 处、四十四组证据；本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`ccbc6132` 将 `reciprocal` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；静态身份/metadata、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十六波补充证据

- `8.06`：`4f414054`/`14c30c38` 将 RotaryPositionEmbedding gradient 接入共享 launcher，保留四输入、三输出 query 与同步策略；静态合同 43 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`36502b8e`/`fcaa6cce` 将 CUB argsort offsets 长度边界改为 `USER_CHECKop`，并补齐此前 `166010a8` 漏记；累计 92 处、四十五组证据，结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`742f1595` 将 `lerp` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十七波补充证据

- `8.06`：`f34ecce4`/`393a5f70` 将 Conv2d backward 接入共享 launcher，保留三输出 gradient query、descriptor cleanup 与同步策略；静态合同 44 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`193d5171`/`22ccddc5` 将 CUB arg-reduce offsets rank 边界改为 `USER_CHECKop`，累计 93 处、四十六组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`25142db7` 将 `softmax` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十八波补充证据

- `8.06`：`1874f7ed`/`f92d4ffd` 将 UpsampleNearest2d backward 接入共享 launcher，保留 output/input-size RAII descriptor 与同步策略；静态合同 45 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`4c64067e`/`050da89a` 将 CUB arg-reduce offsets 长度边界改为 `USER_CHECKop`，累计 94 处、四十七组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`d6bd24f1` 将 `log_softmax` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第五十九波补充证据

- `8.06`：`2d8f415b` 将 LeakyReLU backward 接入共享 launcher，保留 negativeSlope、selfIsResult、scalar cleanup 与同步策略；静态合同 46 passed，本机无 CANN/NPU，仍待实机。
- `8.06`：`c33196b3` 将 SiLU backward 接入共享 launcher，保留同步策略；静态合同 47 passed，本机无 CANN/NPU，仍待实机。

### 2026-09-04 第六十波补充证据

- `8.06`：`d87bbd09` 将 Swish forward 接入共享 launcher，保留同步策略；静态合同 48 passed，本机无 CANN/NPU，仍待实机。

### 2026-09-04 第六十一波补充证据

- `8.06`：`744f6c6d` 将 Swish backward 接入共享 launcher，保留同步策略，SwiGlu 未迁；静态合同 49 passed，本机无 CANN/NPU，仍待实机。

### 2026-09-04 第六十二波补充证据

- `8.06`：`1f1ffec3` 将 LayerNorm forward 接入共享 launcher，保留 normalizedShape、eps、三输出与 descriptor cleanup；静态合同 51 passed，本机无 CANN/NPU，仍待实机。
- `8.06`：`ca40d0d6` 将 LayerNorm backward 接入共享 launcher，保留 normalizedShape/outMask、三输出 query 与 descriptor cleanup；静态合同 52 passed，本机无 CANN/NPU，仍待实机。
- `8.06`：`8e772a5b` 将 SwiGlu 接入共享 launcher，保留同步策略；静态合同 50 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`fbc69232`/`a4adb24b` 将 cuDNN RNN LSTM mode 用户边界改为 `USER_CHECKop`，累计 95 处、四十八组证据；结构合同与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`b98cde25` 将 `relu` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第六十波补充证据

- `8.06`：`d87bbd09`/`55a81e8e` 将 Swish forward 接入共享 launcher，保留同步策略；静态合同 48 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`b1c604af`/`7c6420d3` 将 cuDNN RNN 非 LSTM mode 边界改为 `USER_CHECKop`，累计 96 处、四十九组证据；结构合同与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`2bdc68a0` 将 `torch._shape_as_tensor` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第六十一波补充证据

- `8.06`：`744f6c6d` 将 Swish backward 接入共享 launcher，保留同步策略，SwiGlu 未迁；静态合同 49 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`6afd44df`/`b20ea9e2` 将 cuDNN RNN `proj_size==0` 用户边界改为 `USER_CHECKop`，累计 97 处、五十组证据；结构合同与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：本波复核剩余 API 后仅 `vmap` 仍是复杂闭包，已有原生 owner 的 API 不重复包装；未产生安全代码提交。

### 2026-09-04 第六十二波补充证据

- `8.06`：`8e772a5b`/`012dddf4` 将 SwiGlu 接入共享 launcher，保留同步策略；静态合同 50 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`57c6cd92`/`53895c66` 将 cuDNN RNN 第二处 `proj_size==0` 边界改为 `USER_CHECKop`，累计 98 处、五十一组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`7a7ae622` 将 `outer` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第六十三波补充证据

- `8.06`：`1f1ffec3`/`f74043b9` 将 LayerNorm forward 接入共享 launcher，保留 `normalizedShape`、`eps`、三输出与 descriptor cleanup；静态合同 51 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`aae2f5bc`/`79269b83` 将 cuDNN conv3d 分组通道 shape 边界改为 `USER_CHECKop`，累计 99 处、五十二组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`9a9011ce` 将 `isin` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第六十四波补充证据

- `8.06`：`ca40d0d6`/`2e92d162` 将 LayerNorm backward 接入共享 launcher，保留 `normalizedShape`、`outMask`、三输出 query 与 descriptor cleanup；静态合同 52 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`935bb1a9`/`bae4711f` 将 cuDNN RNN backward-x LSTM mode 边界改为 `USER_CHECKop`，累计 100 处、五十三组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`b5dc26d7` 将 `tensordot` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第六十五波补充证据

- `8.06`：`3f0b8c7d`/`3c0f2115` 将 GroupNorm forward 接入共享 launcher，保留 group/eps、三输出 query 与同步策略；静态合同 53 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`b3826005`/`ae78d185` 将 cuDNN RNN backward-x `proj_size==0` 边界改为 `USER_CHECKop`，累计 101 处、五十四组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`e0bc5294` 将 `repeat_interleave` 提升为 numerical 稳定对象并登记 conservative approximate fidelity；身份/metadata 静态测试、`py_compile`/diff-check 通过，动态 JIT 未运行。

### 2026-09-04 第六十六波补充证据

- `8.06`：`016fc62d`/`eb1e89cd` 将 GroupNorm backward 接入共享 launcher，保留 output-mask、group 属性、三输出 query 与 cleanup；静态合同 54 passed，本机无 CANN/NPU，仍待实机。
- `8.06`：`fc849c10` 将 MaskedSelect 接入共享 launcher，保留双输入 mask query 与同步策略；静态合同 57 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`35664df5`/`d3e786e2` 将 cuDNN RNN backward-x 非 LSTM mode 边界改为 `USER_CHECKop`，累计 102 处、五十五组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：本波复核剩余候选仅有复杂 `vmap` 闭包，未强行拆分，保持无新增代码提交。

### 2026-09-04 第六十七波补充证据

- `8.06`：`c4f0447c`/`fca8451c` 将 Avgpool backward 接入共享 launcher，保留 `countIncludePad/divisorOverride`、descriptor cleanup 与同步策略；静态合同 55 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`76dc9dc3`/`6baf9dd5` 将 cuDNN RNN backward-x 第二处 `proj_size==0` 边界改为 `USER_CHECKop`，累计 103 处、五十六组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：复核剩余 API 后仅 `vmap` 为复杂闭包，本波无安全小切片提交。

### 2026-09-04 第六十八波补充证据

- `8.06`：`a9d73aae`/`efb1b758` 将 Maxpool backward 接入共享 launcher，保留 pool descriptors、`poolCeil`、输出处理、cleanup 与同步策略；静态合同 56 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`408b4832`/`1de5551e` 将 cuDNN conv 输入 rank 边界改为 `USER_CHECKop`，累计 104 处、五十七组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`9bd71961` 新增 `agent/design/vmap-owner-plan.md`，记录复杂 vmap 的 owner、Runtime 依赖、迁移边界与后续 CPU 验收节点；本波仅设计前置，未宣称实现完成。

### 2026-09-04 第六十九波补充证据

- `8.06`：`fc849c10`/`77e1d30d` 将 MaskedSelect 接入共享 launcher，保留双输入 mask query 与同步策略；静态合同 57 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`ceabd84c`/`3096b2f0` 将 cuDNN conv 权重 rank 边界改为 `USER_CHECKop`，累计 105 处、五十八组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`30cd207f`、`27866dc2` 细化 `vmap` owner 的可验证契约与验收节点；仅设计前置，未修改 runtime，未宣称实现完成。

### 2026-09-04 第七十波补充证据

- `8.06`：`18fca063`/`029795fa` 将 Index 接入共享 launcher，保留 index query 与同步策略，SliceV2 未改；静态合同 58 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`44a80c8a`/`3259631f` 将 cuDNN conv 分组通道 shape 边界改为 `USER_CHECKop`，累计 106 处、五十九组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`ed9b2010` 补充 `vmap` owner 提取协议、AST 完成门禁与 `VmapContext` 约束；仅设计前置，未修改 runtime。

### 2026-09-04 第七十一波补充证据

- `8.06`：`2e27d71b`/`e2b6e3f0` 将 SliceV2 接入共享 launcher，保留 begins/ends/steps/axes descriptors 与同步策略，Index/其他 owner 未改；静态合同 59 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`6727dd57`/`eb4db9b4` 将 cuDNN conv backward-x 权重 rank 边界改为 `USER_CHECKop`，累计 107 处、六十组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`abaa242a`、`afa756bc` 连续补充 vmap 设计契约与 unsupported AST 静态门禁；仅设计/门禁前置，未修改 runtime，未宣称实现完成。

### 2026-09-04 第七十二波补充证据

- `8.06`：`ff26ab02`/`7457382d` 将 StridedSliceAssignV2 接入共享 launcher，保留 gradient memset 分支与 slice descriptor handling；静态合同 60 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`92a66390`/`5e720411` 将 cuDNN conv backward-x dy rank 边界改为 `USER_CHECKop`，累计 108 处、六十一组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`41236df4`、`5a8b0115` 补充 vmap context 夹具契约、提取顺序、绑定与回滚步骤；仅设计前置，未修改 runtime。

### 2026-09-04 第七十三波补充证据

- `8.06`：`73b71c7d`/`b124efbf` 将 InplaceMaskedScatter 接入共享 launcher，保留 tracked base-to-output memcpy 依赖与同步策略；静态合同 61 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`1e3bab6e`/`f112976b` 将 cuDNN conv backward-w 输入 rank 边界改为 `USER_CHECKop`，累计 109 处、六十二组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`05e9f37f` 补充 vmap 评审证据清单，覆盖 AST、closure/global、fidelity、聚焦节点与 skip 归因；仅设计前置，未修改 runtime。

### 2026-09-04 第七十四波补充证据

- `8.06`：`4eb360d7`/`2dc68144` 将 IndexPutImpl 接入共享 launcher，保留 index tensor-list handling 与同步策略，IndexPutImplAccumulate 未改；静态合同 62 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`5596563f`/`2702f7c6` 将 cuDNN conv backward-w dy rank 边界改为 `USER_CHECKop`，累计 110 处、六十三组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`9ee118a0` 补充 vmap unsupported 行为矩阵，覆盖 extent/nested dim/非 bool/depth callback/out_dims；仅设计前置，未修改 runtime。

### 2026-09-04 第七十五波补充证据

- `8.06`：`f353076a`/`1cc7aa53` 将 IndexPutImpl accumulate 接入共享 launcher，保留 tracked output memset 与 index tensor-list dependency；静态合同 63 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`241ab528`/`7f7c9bbc` 将 cuDNN RNN 推理阶段输入 rank 边界改为 `USER_CHECKop`，累计 111 处、六十四组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`ba76983c` 明确 vmap 仅做组织重构，不新增 kernel/设备传输/优化，并定义 CPU/CUDA/ACL 分层验收与 skip 归因；仅设计前置。

### 2026-09-04 第七十六波补充证据

- `8.06`：`3dd89256`/`90d73767` 将 AdamWList 各项更新接入共享 launcher，保留 fused D2D copy checks 与唯一同步点；静态合同 64 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`4858b0a2`/`1a9acf32` 将 cuDNN RNN 输入通道 shape 边界改为 `USER_CHECKop`，累计 112 处、六十五组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`f6d3a435` 明确 vmap 稳定签名、内部 callback 注入和 unsupported kwargs 拒绝；仅设计前置，未修改 runtime。

### 2026-09-04 第七十七波补充证据

- `8.06`：`5f989f16`/`24a9e438` 将 FlashAttention forward 接入共享 launcher，保留 prefix/qstart/kvstart RAII descriptors 与同步策略，backward/KV-cache 未改；静态合同 65 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：本波复核剩余 CUDA/CUDNN/CUB/NCCL 断言均属内部不变量或后端运行失败，没有新增安全用户边界迁移。
- `7.03`：`d316a706` 补充 vmap AST 门禁输出契约，定义计数、禁止捕获、unsupported guard 与 fail-closed 证据格式；仅设计前置。

### 2026-09-04 第七十八波补充证据

- `2.19`：`a29e0f81`/`c9866b48` 新增后端内部断言分类文档与结构门禁，1 passed；用户边界累计保持 112 处、65 组，不改运行时错误语义。
- `7.03`：`d90a716a`、`9ee118a0`、`05e9f37f`、`41236df4`、`5a8b0115`、`ba76983c`、`f6d3a435`、`9bd71961`、`ed9b2010`、`d316a706`、`abaa242a`、`afa756bc` 逐步补充 vmap owner、context、unsupported、后端分层与 AST 评审契约；未修改 runtime，未宣称实现完成。
- `8.06`：本波暂未新增 ACL family；上一波 FlashAttention backward 已在前面 canonical 记录中，保持无 CANN/NPU 实机验证声明。

### 2026-09-04 第七十九波补充证据

- `8.06`：`e1470830`/`4e1f6ba0` 将 IncrementalFlashAttention 接入共享 launcher，保留 block-table、actual-sequence、cache-view cleanup 与同步策略，KVCacheMemcpy 未迁；静态合同 67 passed，本机无 CANN/NPU，仍待实机。
- `2.19`：`7a24ca0b`/`7c1565d2` 将 cuFFT jit_prepare unsupported dtype 边界改为 `USER_CHECK`，累计 113 处、六十六组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`e8260779` 明确 vmap 版本兼容、kwargs 策略与退出标准；仅设计前置，未修改 runtime。

### 2026-09-04 第八十波补充证据

- `8.06`：本波复核确认标准 workspace/query/execute/sync owner 已全部迁移；剩余 KVCacheMemcpy 为逐 token `aclrtMemcpyAsync` 专用路径，不纳入通用 launcher。
- `2.19`：`040e44a0`/`d251d738` 将 CUBLAS matmul 输入 rank 边界改为 `USER_CHECK`，累计 114 处、六十七组证据；结构与 nvcc TU 通过，本机 CUDA 可用，负向见 2.19 行末运行记录。
- `7.03`：`7d8fdd37` 补充 vmap 无可变全局、幂等 install、失败回滚与资源释放门禁；仅设计/门禁前置。

### 2026-09-04 第八十一波补充证据

- `2.19`：`91718e98`/`c041b0a7` 将 cuDNN RNN 权重查询内部断言纳入分类文档与结构门禁；1 passed，不改变用户错误语义或累计数。
- `7.03`：`79455300` 固定 vmap 首门禁夹具（seed=17、简单映射与 nested bool shape），要求记录 fixture/NumPy 期望/unsupported 矩阵；仅设计前置，未修改 runtime。
- `8.06`：复核确认标准 launcher owner 已穷尽，剩余 KVCacheMemcpy 等专用路径不纳入通用 launcher，本波无代码提交。

### 2026-09-04 第八十二波补充证据

- `8.06`：`6905864a` 明确标准 workspace/query/execute/sync owner 已迁移完毕，剩余 KVCacheMemcpy 为逐 token 专用 memcpy 路径，不纳入通用 launcher；本波无代码提交。
- `2.19`：`a29e0f81`/`be526722`/`8382497d` 新增后端内部断言分类文档与 CUDNN RNN bias/descriptor 门禁，结构门禁 1 passed；不改变用户错误语义，用户边界累计保持 114 处。
- `7.03`：`1abf0d75`、`d6626bd0`、`41236df4`、`5a8b0115` 补充 vmap 固定数据、context 夹具、handoff 证据模板等设计契约；未修改 runtime，未宣称实现完成。

### 2026-09-04 第八十三波补充证据

- `2.19`：`27cc72f2`/`23d70b26` 补充 CUB 状态内部断言分类门禁与说明文档；不改变运行时语义，用户边界累计保持 114 处。
- `7.03`：`de512c43` 补充 vmap AST 实现草案，定义 module/install/nested 计数与 binding 行号提取伪代码；仅设计前置，未修改 runtime。
- `8.06`：只读确认标准 launcher owner 已穷尽，KVCacheMemcpy 等专用 memcpy 路径不纳入通用 launcher，本波无代码提交。

### 2026-09-04 第八十四波补充证据

- `2.19`：`24848098`/`0bfb854e` 将 CUBLAS 测试入口返回码内部断言纳入分类门禁；1 passed，不改变运行语义或用户边界累计。
- `7.03`：`30f5e2de` 细化 vmap context 泄漏 AST 门禁，覆盖默认参数、注解、decorator、closure 白名单与模块全局扫描；仅设计/门禁前置。
- `8.06`：只读确认标准 workspace/query/execute/sync owner 已全部迁移，KVCacheMemcpy 等专用路径不纳入通用 launcher，本波无代码提交。

### 2026-09-04 第八十五波补充证据

- `2.19`：`54e8d545`/`4229cba5` 将 CUDNN 测试入口返回码内部断言纳入分类门禁；1 passed，不改变运行语义或用户边界累计。
- `7.03`：`40aed528` 补充 vmap fidelity registry 静态门禁，校验 implementation identity、approximate level、context/backend detail 与重复 install；仅设计/门禁前置。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 等专用路径不纳入通用 launcher；本波无代码提交。

### 2026-09-04 第八十六波补充证据

- `2.19`：`d269a52a`/`8595e479` 收束后端内部断言分类说明与门禁，覆盖 CUDNN/CUBLAS/CUB 状态路径；不改变运行语义或用户边界累计。
- `7.03`：`d97d5620` 补充 vmap 发布检查清单，覆盖 clean import、重复 install identity、board/handoff 链接、回滚与无缓存产物；仅设计/门禁前置。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 等专用路径不纳入通用 launcher，本波无代码提交。

### 2026-09-04 第八十七波补充证据

- `2.19`：`8af5fd8d` 精确约束 Cutt 返回码内部断言并纳入分类门禁；不改变运行语义或用户边界累计。
- `7.03`：`2b7b64ca` 补充 vmap 变更控制与 reviewer sign-off，明确 code/doc 分离、owner/context/unsupported 审阅与冲突规则；仅设计前置。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 等专用路径不纳入通用 launcher，本波无代码提交。

### 2026-09-04 第八十八波补充证据

- `2.19`：`30bfdb6e`/`e853b873` 将 CUDNN RNN descriptor 内部断言纳入分类门禁；1 passed，不改变运行语义，用户边界累计保持 113 处。
- `7.03`：`ce1cbb5c` 补充 vmap metadata 兼容契约，固定 `_jittor_vmap_base`/`_jittor_vmap_specs` 身份、形状与嵌套层级；仅设计前置，未修改 runtime。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 等专用路径不纳入通用 launcher，本波无代码提交。

### 2026-09-04 第八十九波补充证据

- `2.19`：`46ab1e17`/`20225dca` 精确约束 cuDNN plan `ASSERT(ok)` 内部断言计数并纳入门禁；1 passed，不改变运行语义或用户边界累计。
- `7.03`：`06899cc0` 补充 vmap `in_dims/out_dims` 的 int、None、tuple/list、负轴归一化矩阵；仅设计/门禁前置，未修改 runtime。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径，本波无代码提交。

### 2026-09-04 第九十波补充证据

- `2.19`：`9f6afe17`/`143cb8e3` 补充 CUB 测试入口内部断言精确计数门禁；结构门禁 1 passed，不改变运行语义或用户边界累计。
- `7.03`：`9ad2c132` 补充 vmap 嵌套 metadata 深度契约，固定 specs 追加、base identity 与 batch shape 顺序；仅设计前置，未修改 runtime。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径，本波无代码提交。

### 2026-09-04 第九十一波补充证据

- `2.19`：`bf5fef1e`/`a7bc9595` 收紧 CUBLAS/CUDNN 测试入口内部断言精确计数门禁；结构门禁 1 passed，不改变运行语义或用户边界累计。
- `7.03`：`ac38772d` 补充 vmap kwargs 兼容矩阵，明确 in/out_dims 归一化、randomness/chunk_size unsupported、未知 kwargs TypeError 与静态门禁；仅设计前置。
- `8.06`：只读确认标准 ACL launcher owner 已全部处理，KVCacheMemcpy 保持专用路径，本波无代码提交。

### 2026-09-04 第九十二波补充证据

- `2.19`：`8c2ebaa8`/`92216275` 将 CUB 测试 CUDA 状态断言收紧为精确计数门禁；结构门禁 1 passed，不改变运行语义或用户边界累计。
- `7.03`：`b233cf6a` 补充 vmap 标量/zero-dim 输出契约，固定 singleton 归一化、nested batch 轴与 `out_dims` 形状；仅设计前置，未修改 runtime。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径，本波无代码提交。

### 2026-09-04 第九十三波补充证据

- `2.19`：`66fd400d`/`5599b11b` 精确约束 cuDNN convolution `best_algo_idx!=-1` 内部断言计数并纳入门禁；1 passed，不改变用户边界或运行语义。
- `7.03`：`aa882756` 补充 vmap autograd 契约，明确 loop/stack 梯度、bool fast path 非微分边界与 CPU gradient 节点草案；仅设计前置。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径，本波无代码提交。

### 2026-09-04 第九十四波补充证据

- `2.19`：`9d2752e2`/`9ad1a807` 精确约束 cuDNN 3D convolution `best_algo_idx!=-1` 内部断言计数并纳入门禁；1 passed，不改变用户边界或运行语义。
- `7.03`：`ae5623e8` 补充 vmap 并发契约，明确 re-entrant 调用、context 生命周期隔离及线程安全 probe；仅设计前置，未修改 runtime。
- `8.06`：只读确认标准 ACL launcher owner 已穷尽，KVCacheMemcpy 保持专用路径，本波无代码提交。

### 2026-09-04 第九十五波补充证据

- `10.13`：`821bb6ba` 新增 smoke AST 快门禁，确认 native/torch 两次运行均传入 `-m not slow` 且 `SLOW_FILES` 仍被 gate 覆盖；定点结构节点 1 passed，任务已完整关闭。
- `2.19`：`903b8d3f`/`2a55244b` 精确约束 CUDNN backward-x `best_algo_idx!=-1` 内部断言计数；结构门禁 1 passed，不改变用户边界累计。
- `7.03`：`17a15406` 补充 vmap 资源边界/取消契约，覆盖 footprint 上限、超大 extent、异常清理和 hold-vars/fidelity 泄漏；仅设计前置。

### 2026-09-04 第九十七波补充证据

- `0.15`：只读核对 RingBuffer GIL/有界等待与长文件拆分方案。当前 `test_children_died` 需要 Linux timed-wait/GIL 安全实现；两个长 compat 文件无法在保持 nodeid/import 语义与 `loadfile` 覆盖的前提下安全拆分降时。smoke 仍约 390s、预算约 446s，未提交代码，任务保持待领。

### 2026-09-04 第九十八波补充证据

- `0.15`：`23814b9a` 已实现 Linux `RingBuffer::wait_pop_for`、等待阶段 GIL 安全拆分，并将 `test_children_died` 改为有界 timeout；worker-death 聚焦节点 1 passed（约 106s，含核心重编）。`SLOW_FILES` 尚未移除，smoke `<300s` 尚未重测，任务继续保持待领。

### 2026-09-04 第九十九波补充证据

- `0.15`：`d3f4853e` 修正 `wait_pop_for` 的单次绝对 deadline、EINTR/非零状态处理，并在恢复 GIL 后保留原异常（不再把 stop/其他错误统一改写为 timeout）；`ring_buffer.cc` 与 `py_ring_buffer.cc` TU 语法检查通过。固定 timeout 作用范围、Dataset 专用轮询、`SLOW_FILES` 与 smoke `<300s` 仍待专项验收，任务保持待领。

### 2026-09-04 第九十九波补充证据（修正版）

- `0.15`：`876ec09c` 恢复通用 `pop()` 无限等待/兼容行为，新增 Dataset 专用 `pop_for(timeout_ms)`；`wait_pop_for` 单次 deadline、伪唤醒/EINTR/stop/其他异常处理及 GIL 边界已修正，协议合同与 killed-worker 聚焦节点各 1 passed（后者 19.72s）。`SLOW_FILES` 尚未移除，smoke `<300s` 尚未重测，任务继续待领。

### 2026-09-04 第九十六波补充证据

- `10.05`：`1a423a16`/`f7f33f5b` 固定 skip reason bucket 优先级、CI summary 和 `other>0` fail-closed；合成结构测试 2 passed，任务已完整关闭。
- `0.15`：RingBuffer GIL/有界等待方案完成只读审计，当前未提交代码；仍需 Linux timed-wait/GIL 安全实现、worker death 聚焦节点与 smoke 重新测量，不能标完成。
