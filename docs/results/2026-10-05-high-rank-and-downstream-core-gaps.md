# 张量维度上限 10 → 32，及下游评测暴露的四项核心缺口

- Status: 已修复并验证（CPU + 真实 CUDA RTX 4090），提交于 `6a850ea5`
- Date: 2026-10-05
- Baseline: `316e1fe7`
- Owner: Jittor 核心维护者
- Review when: `type/nano_vector.h` 的打包格式、JIT key 的维度编码、
  reduce/broadcast 掩码宽度、`KernelIR::to_string` 的 for 循环输出、
  `pyjt_compiler.py` 的数字协议槽，或 `linalg` 的 `numpy_code` 前向发生变化

## 结论

一次下游量子线路库的实测评测给出了五项归属 Jittor 核心的问题，按优先级全部处理：

| 优先级 | 问题 | 处理 |
| --- | --- | --- |
| P0 | 张量最多 10 维（第 11 维直接报 `NanoVector exceeds its ten-entry ...`） | 上限提到 32 维；超出时给出明确错误 |
| P0 | 复数 `jt.norm` 不可用（`maximum` 没有复数 kernel，编译期报错） | `norm` 改为 `sqrt(sum(abs(x)**2))`、结果为实数；复数 `maximum`/`minimum` 改为构造期的明确错误（与 torch 一致） |
| P1 | `Var * obj` 遇到不认识的对象直接抛 RuntimeError，`obj.__rmul__` 没有机会执行 | 生成的二元数字协议槽在无匹配重载时返回 `NotImplemented` |
| P1 | 秩亏 QR 的梯度：CPU 抛 `LinAlgError`，CUDA 静默得到 NaN | 根因是 CuPy 的 `solve`/`inv`/`cholesky` 对奇异矩阵返回 NaN；改为两端都抛 `LinAlgError` |
| P1 | 默认 `use_mkl` 下，机器没有 cmake 时第一次 CPU matmul 现场构建 oneDNN 失败 | 拿不到 oneDNN（无 cmake、无现成构建、无显式路径）时，mkl 库报告为不可用并给一次警告，CPU 乘法走通用 kernel |

## P0：维度上限

上限不只是 `NanoVector` 一处。放开以后依次暴露了下列静默假设，每一处都有回归用例：

1. **`NanoVector` 打包格式**：两个 64 位字，4 位条目数 + 10×6 位结束偏移，值总共
   64 位。现在超出时溢出到引用计数、写时复制的堆块（条目数字段标记为 15），
   打包快路径不变；缩回能打包的大小时会重新打包，所以追加构造出来的向量仍然每种
   内容只有一种表示。Python 侧持有的 `NanoVector` 补了 `__dealloc__`。
2. **打包格式自身的旧 bug**：结束位恰好是 64 的条目在 6 位字段里存成 0，并进位到
   下一字段；之后再 push 会以 `pre_offset=0` 覆盖数据（实测 stride `1024` 读成
   `1092`）。旧代码在这之后不会再 push，所以没暴露；现在 0 统一解释为 64。
3. **JIT key 编码**：每维的 key 名用 `JK::hex1`/`<< int` 写成十六进制，而模板
   `NAME@i` 按十进制展开，第 10 维起对不上；`hex1` 的值又是 mod 16。现在名字里的
   下标用 `JK::dec`，`=` 值用完整 `JK::hex`，`:` 值（`IV`/`IO`、code_op 的
   `_dim`）用十进制；fused op 的 `var_info` 加了分隔符。
4. **掩码宽度**：`ReduceOp`/`BroadcastToOp` 的掩码是 `uint16`，17 维全归约返回了
   shape `[2]`（**静默错误**）。改为 `uint32`；`1<<i` 改为无符号移位，
   `reduce_tuner` 的 `(1<<j)<=mask` 循环在第 31 位时不会终止，一并改掉。
   32 位掩码就是 32 维上限的来源，由 `Var` 构造与 `set_shape` 统一检查。
5. **固定容量数组**：`StackVector<>` 默认 10 槽且越界不检查（getitem 时栈溢出、
   `double free`），改为 32 槽并越界即断言；cuTT 计划键 10 → 32；reindex 的
   `@e{i}(i0,...)` 不再用 `'0'+i` 拼字符。
6. **nvcc 编译时间**：cudafe++ 对“在 for-init 里声明变量”的嵌套深度是指数的。
   独立复现：深 14 层 2.1 s、16 层 11 s、17 层 30 s（每层约 ×2.5），rank-20 归约
   核 10 分钟内编不完；把声明放到外层块 `{ T i = 0; for (; ...) }` 后 20 层 0.75 s，
   g++ 无此问题（20 层 0.08 s）。`KernelIR::to_string` 对外层已有 ≥8 层循环的
   循环使用这一形式，浅层循环生成的源码逐字不变。

## 验证

环境：conda `jittor`，g++ 11.5，CUDA 12.2，RTX 4090。每条线独立
`JITTOR_HOME`/`cache_name`。日志在 `$JITTOR_LAB_ROOT/_state/tc-gaps/`（未版本化）。

| 检查 | CPU | CUDA |
| --- | --- | --- |
| 维度探针：n = 10/11/12/16/17/20 × 21 种操作（构造、逐元素、全/部分归约、keepdims、transpose、reshape、getitem/setitem、Var 索引、广播、where、argmax、argsort、concat、matmul、grad、复数） + rank 32 | 127/127 | 127/127 |
| rank 33 | 明确报错 | 明确报错 |
| 新增 `tests/core/test_high_rank_tensors.py`、`tests/ops/test_binary_reflected_and_complex_order.py`、`tests/linalg/test_singular_input_raises.py` | 16 passed | 16 passed |
| 新增 `tests/build/test_onednn_availability.py` | 4 passed | — |
| C++ JIT 测试（含 `nano_vector` 新增溢出/写时复制/64 位边界用例） | 97 passed / 2 skipped / 4 xfailed | — |
| 受影响算子与绑定的定向回归（binary/reduce/broadcast/transpose/reindex/getitem/setitem/argsort/code/where、dlpack、pyjt 协议、错误消息、全部 linalg） | 409 passed / 3 skipped | — |
| `tools/run_test_suite.py --tier core`（默认配置） | native 98 + torch 212 passed，0 failed | native 128 + torch 212 passed；唯一失败是冷缓存下子进程 600 s 超时，热缓存单独复跑通过 |
| `tests/structure`（Torch 模式） | 修正两处计数台账后，仅剩 1 项既有失败（见下） | — |

热缓存下 `test_high_rank_tensors.py` 用时约 9 s，留在默认 tier。

## 已知边界与后续

- **既有失败，非本次引入**：`tests/structure/test_pytest_contract.py::test_test_modules_avoid_collection_time_backend_side_effects`
  指向 2026-09-30 的 `tests/linalg/test_linalg_higher_order.py`（收集期调用本地 helper）；
  `test_process_mode_contract` 的一项只在全量结构运行中失败、单独运行通过。
- `--tier core` 以 `use_mkl=0` 运行时 `test_core.py` 的三项内存/调度断言失败——它们
  假定 oneDNN matmul 路径，默认配置下通过；与本次改动无关。
- `Var @ obj` 仍不回退到 `obj.__rmatmul__`：`__matmul__` 是 Python 层绑定，
  安全的回退判据（下游对象常带 `.shape`）需要单独设计。（2026-10-09 已处理，见
  [下游 P1 报告](2026-10-09-downstream-p1-allocator-protocol-transpose.md)。）
- QR 秩亏梯度现在两端一致地报错，而不是给出有限值。torch 的 `linalg.qr` 反向
  同样对奇异 R 不给出有限梯度；需要有限梯度的下游应自带正则化的 QR 反向。
- 32 维上限由 32 位掩码决定；需要更高维时要把掩码与 Python 侧参数一并加宽。
- ROCm/NPU 未验证（无硬件）。
