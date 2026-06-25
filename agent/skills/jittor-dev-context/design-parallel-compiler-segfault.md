# Task #8：parallel 编译器 VarRelayManager segfault — 根因假设 + 修复方案

> 状态：🟡 根因**假设**已提出（2026-06-25 调研），**未验证、未修**。workaround `jt.flags.use_parallel_op_compiler=0` 在用。这是核心并发改动，高风险，必须全套验证后才动。

## 现象
编译 MBConv/depthwise/SE 密集的模型（MobileNetV3 / EfficientNet b0-b3）时，默认 parallel op 编译器堆损坏崩溃。**CPU + CUDA 都触发**：
- Fault PC：`VarRelayManager::get_op_relay_info`（CUDA）/ `VarRelayManager::get_relay_src`（CPU）
- `parallel_compiler.cc:354 Segfault happen`，glibc "corrupted double-linked list"，worker 线程 C10/C12
- 与已修 inf/nan GIL bug（`64de9c07`）不同根因。

## 根因假设（**待验证**，调研读码得出，可能不全对）
1. `OpCompiler::do_compile`（`op_compiler.cc:1097`）用 `jittor::lock_guard`——**进程级文件锁** + 全局 `_has_lock` 标志（`lock.h:24-35`：`if(_has_lock) return;` 已持有则跳过 `lock()`）。
2. `parallel_compiler.cc:241` **主线程**先 `lock_guard lg` 获锁（`_has_lock=1`），再 `threads.launch_all` 启动 worker。
3. worker 进 `do_compile` → `lock_guard` 见 `_has_lock==1` → **跳过加锁** → 多 worker **无互斥**并发跑 `do_compile_inner`（含 `TunerManager::tune()→add_relay_group()` 写 `relay_groups`、`get_jit_src()→get_op_relay_info()` 读）。
4. 并发改 `relay_groups`（`opt/var_relay.cc:35-112` / `114-175`）+ 非原子全局计数（`VarRelayGroup::~VarRelayGroup` 里 `Var::number_of_lived_vars++`）→ 堆/链表损坏。
5. MobileNetV3/EfficientNet 大量独特 shape op → 高并发 + 频繁 relay → 竞态暴露。

## 调研建议的修复（方案 A：mutex 替代文件锁）
`lock.h` 加 `std::mutex compile_lock` + `compile_lock_guard`；`op_compiler.cc:1097` 改用它；`parallel_compiler.cc:241` 删主线程文件锁持有。

## ⚠️ 为什么不能贸然套用（接手者必读）
- **可能废掉并行**：若 `do_compile` 被 mutex 完全串行化，并行编译的意义没了。需搞清 jittor 并行编译到底并行的是什么（codegen prep？还是 nvcc/g++ 子进程？）——若慢的是子进程编译、codegen prep 本就该串行，那串行 do_compile 不损失并行度；否则会。
- **多进程缓存安全**：文件锁是**跨进程**缓存写保护需要的，不能简单删。
- **死锁前科**：`64de9c07` 修 GIL 时发现"单纯加锁会死锁"——并行编译器的锁交互很微妙。
- 假设本身存疑：jittor 并行编译用了很久，若 do_compile 真的全程无互斥，应该**经常**崩而非只在复杂模型崩 → 说明要么假设不全对，要么竞态只在特定共享态（relay_groups/全局计数）高并发时才损坏。

## 接手验证流程（务必）
1. **稳定复现**：parallel 编译 MobileNetV3，确认必崩 + 抓 backtrace 确认在 VarRelayManager。
2. 先用**最小侵入**验证假设：给 `VarRelayManager::relay_groups` 的读写 + `add_relay_group` 加**细粒度 mutex**（只锁这个共享态），别动全局文件锁/并行结构 → 复编译看是否还崩。这比方案 A 安全。
3. 若细粒度锁解决且并行度不退 → 优先。否则再评估方案 A/B。
4. 全套门禁：复现崩 → 修后不崩（多次）→ 无死锁（stress + 超时）→ 并行编译耗时不显著退化 → 多进程并发编译不坏缓存 → `test_torch_compat` 171/0 → MobileNetV3/EfficientNet 默认 parallel 跑通。
5. 过了才 commit。守 G1。
