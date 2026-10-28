# MiniMax-H3 运行退出时的「corrupted double-linked list」

- 状态：已修复，多次真实运行验证
- 日期：2026-09-14
- 基线提交：`c6b56a61`（两处修复 `f43b90d6`、`c6b56a61`，基于 `7f7c8c0e`）
- 验证范围：MiniMax-H3 tiny 32x32 t2va（`run-tiny-parity.sh`，VAE 在 autocast 下为
  float32）在 CUDA 上连跑四次、一次 ASAN 构建的全流程，以及 `tests/core` 中 fused op、
  relay 与 node 相关文件在 CPU 上；卡型原文未记
- 维护者：核心运行时与编译器维护者
- 复查条件：liveness 队列、free buffer 或已编译融合算子缓存的生命周期变化

## 现象

每次 MiniMax-H3 tiny 32x32 t2va 运行都正常结束——mp4、音轨与汇总都已写出——然后在
解释器退出时以 glibc 的 `corrupted double-linked list` 和状态 134 中止。它指出的
地址每次运行都会漂移（`~jit_cache_map`、`~SFRLAllocator`、某个 PyAV/gnutls
析构函数、一个裸 `free`）。这种漂移是「后面的某次 free 走到了一个已被覆盖的 chunk」
的特征，而不是上述任一位置本身的故障。

## 如何定位

**CUDA 在 AddressSanitizer 下确实能初始化。** 早先的尝试得出了不能的结论
（「visible device count is 0」），但那是 ASAN 的 shadow-gap 与 TLS 拦截破坏了驱动
探测，而不是驱动本身。加入

```
ASAN_OPTIONS=detect_leaks=0:protect_shadow_gap=0:intercept_tls_get_addr=0:\
detect_odr_violation=0:halt_on_error=0:abort_on_error=0:\
log_path=/tmp/asan-report:print_stacktrace=1
```

就能把设备带起来，完整流水线在 ASAN 下运行，核心与 JIT 算子都被插桩
（`cc_flags=-fsanitize=address -fno-omit-frame-pointer -g -O1`、
`JITTOR_NO_DEEPBIND=1`、`LD_PRELOAD` 指向 gcc 的 `libasan.so`）。flash-attn 扩展
无法在 ASAN 下加载，因此 attention 回退到原生路径；tiny 形状足够小，可以承受。

随后 ASAN 报告第一次非法访问。这次写操作：

```
heap-use-after-free, WRITE of size 8
  std::vector<pair<Node*, void (Node::*)()>>::emplace_back
  Node::release_both_liveness            node.cc:302
  VarPtr::~VarPtr                        var.h:161
  VarRelayGroup::~VarRelayGroup          var_relay.cc:33
  FusedOpContext::~FusedOpContext        fused_op.h:27
  jit_cache_map<shared_ptr<FusedOpContext>>::~jit_cache_map
  __run_exit_handlers
freed by:
  ~vector<pair<Node*, void (Node::*)()>>  (the same liveness_queue)
  __run_exit_handlers
```

## 根因

静态销毁顺序。`node.cc` 中的 `liveness_queue` 与 `free_buffer` 是命名空间作用域的
vector，因此它们的析构函数在 `__run_exit_handlers` 期间运行，其相对于已编译融合
算子缓存析构函数的顺序只由链接器决定。该缓存的析构函数拆解 `VarRelayGroup`，其
`VarPtr` 释放 liveness——而 `release_both_liveness` 会向已被销毁的队列追加。这次
追加把一个 pair 写入已释放的存储，正是它破坏了分配器的 chunk 元数据；glibc 很久
之后在某个无关的 free 上以 `corrupted double-linked list` 检测到。

此前的六处 teardown 修复没有解决这一点，因为它们都没有改变退出时 teardown 所写入
容器的生命周期。

## 修复

让这两个容器在整个进程生命周期内存活，这也是 `graph_mutation_mutex`（`node.cc`）
与 cleanup 回调列表（`utils/log.cc`）已采用的模式：

- `free_buffer` 是一个返回泄漏堆 vector 的函数（`node.h` 将其声明为函数，
  `SetupFreeBuffer` 调用它）；
- `liveness_queue` 是绑定到泄漏堆 vector 的引用。

退出时不回收任何东西，因此没有任何退出时的写入会落到已释放存储中。

## 验证

- `run-tiny-parity.sh`（32x32，124 帧，VAE 在 autocast 下为 float32）连跑四次：
  **每次运行都是 exit 0、没有 `corrupted`**；修复前同一命令以该消息退出 134。每次
  运行仍写出由 VAE 解码的真实 44325 字节 mp4。
- ASAN 运行不再报告任何退出时访问；剩下的唯一报告是下面编译期的 relay
  use-after-free，与 teardown 无关。
- `tests/core/test_fused_op.py`、`test_new_fused_op.py`、`test_fuser.py`、
  `test_var_relay_members.py`、`test_node.py` 在 CPU 上通过。
  （`tests/core/test_core.py` 与 `test_auto_flush.py` 在此环境中存在既有失败：
  `test_var_holder` 断言一条旧的 `'int' object has no attribute 'shape'` 消息，
  而代码现在说的是 `ndim`；另有若干测试需要可选库；它们各自的通过或失败与本改动
  无关。）

## 第二份报告：编译期的 relay use-after-free（在 `c6b56a61` 中修复）

退出时故障消失后，ASAN 的下一个报告是另一类编译期问题。
`VarRelayManager::add_relay_group` 调用 `op->set_inputs(new_inputs)` 断开被 relay
的源与其目标之间的链接，随后 `release_inputs` 释放了 `FusedOp::vars` 与 relay 算子
自身的 Var 成员仍在引用的输入。`ReindexOp::jit_prepare` 之后读取 `x->ns`，因此融合
key 可能由已释放内存构建。H3 视频 VAE 通过 `ConvTuner::forwardTune` 到达此处。

`removed_input_vars`——已收集这些输入的字段——现在拥有它们（`vector<VarPtr>`）
而不是借用，正如 `relayed_pairs` 已经拥有 relay 源一样。

- 完整的 tiny 流水线**在 ASAN 下干净运行**（exit 0，无报告文件），这也在插桩构建
  上确认了上面的退出时修复。
- 带此修复的非 ASAN tiny 运行以 0 退出且没有 `corrupted`，其
  `video`/`audio`/`cond`/`latents` tensor 与不带修复的运行**逐 bit 相同**——改动只
  涉及生命周期，不涉及结果。
- 记录为 `agent/manuals/known-issues.md` 中的 `KI-COMPILER-005`，现已修复。

原始 ASAN 报告与运行日志位于 `$JITTOR_LAB_ROOT/_state/h3/` 与
`/tmp/asan-report.*`，不入库。
