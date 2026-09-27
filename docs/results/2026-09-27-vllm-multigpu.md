# vLLM 双卡验证与 CPU 控制通信

- 状态：进行中；双卡可完成生成，但 Qwen TP=2 数值正确性尚未通过。
- 日期：2026-09-27。
- 基线：`539ce90ee68d6772ffa2b780203f889e67f4d9ec` 加本报告同提交的修改；开工时
  `origin/chk=61294cd14673ba60f4072542a2e73ea2c8f8c509` 已是祖先。
- Owner：vLLM adapter / Torch compatibility maintainers。
- 复查条件：通信组、设备放置、worker 启动、vLLM 或 NCCL 版本变化。

## 范围与环境

单机两张 RTX 4090，vLLM 0.24.0、Transformers 5.5.3、Python 3.10.21；
Jittor 使用 CUDA 12.2，独立 oracle 为 PyTorch 2.11.0+cu130。
模型为 Qwen3-0.6B，FP16、TP=2、eager、FlashAttention、关闭 custom all-reduce，
模型上下文 512，单次生成 32 token。保留 Jittor 既有 CUDA 默认设备。

卡上有其他任务，本轮仅检查正确性；生成耗时不作为双卡性能或加速比。
原始日志、JSON、源码清单、独立缓存位于未版本化目录
`$JITTOR_LAB_ROOT/_state/vllm-multigpu/20260927/`。

## 已确认的修复与验证

1. 请求 `gloo` 的 CPU 通信组此前被错误处理为 MPI/NCCL。保留 singleton 与真实
   FileStore/TCPStore 双进程失败证据后，在原生 ProcessGroup 下补充 Store 驱动的
   CPU 控制通信，Torch 层负责公开 API 路由。
2. 销毁 CPU 子组不再关闭 WORLD 的 Store。实际 TCP 测试覆盖正常/反序组、服务器
   位于不同 rank、延迟清理、重复交换、调用顺序不一致、关闭后的访问和有界键数量。
3. CPU 控制通信没有反向传播实现，`distributed.nn.all_reduce` 明确拒绝此路径，
   避免静默丢失梯度；GPU 原生 collective 不被替换成 CPU 计算。
4. 补齐 `get_group_rank`：真实 vLLM 暴露缺失后，公开 API 测例先 RED，再补齐
   WORLD/非连续子组映射与无效组错误。独立 PyTorch 三个 CPU rank 对照通过。
5. 默认 `env://` 动态初始化保留同一个 Store 到 WORLD 销毁；不再让 bootstrap
   临时创建、关闭后由 CPU 子组重建通信服务。先保留生命周期 RED，再验证身份和清理。
6. PyNccl 使用 `ByteTensor` 交换通信 ID，暴露负数字节转换和旧式 CPU 构造器
   放置错误。CPU/真实 CUDA 测例先失败，再修复旧式构造器；同时保留并修复 CUDA
   构造器委托产生的回归。现代 factory 和全局默认设备不变。
7. 模型预热的 `_apply_write_kernel` 暴露第二个 worker 的请求缓冲区落在 GPU 0。
   `get_cuda_view_from_cpu_tensor` 适配此前硬编码复制到设备 0，现改为当前设备。
   新增实际 StagedWriteTensor 测例先得到 1 failed / 3 passed，修复后四项通过。
   原生 vLLM 的 UVA 可以标为 cuda:0，却实际指向所有卡可访问的映射主机内存；
   Jittor 的显存副本不能沿用这一标签约定，因此必须核查真实指针和执行结果。

最初 11 项 host/owner 测试通过；加入 autograd 拒绝测例后 13 项通过；加入 rank
映射和默认 Store 生命周期后相关组合 17 项通过。原生 host transport 与 ProcessGroup
测试共 14 项通过。两套真实 CPU FileStore/TCPStore 数据交换结果一致。
最终版本的 StagedWriteTensor 两套环境均 4 passed；同份代码单卡 Qwen 与 OPT
各生成 32 token，分别与独立 PyTorch 对照完全一致。这是小样例回归，不代表重新
完成此前全部单卡验收矩阵。
旧式类型构造的独立对照：两套环境均为 CPU 30 passed / 7 GPU skipped，CUDA
37 passed。覆盖有符号字节、溢出拒绝、NumPy 转换、设备、别名及梯度。

`multigpu_collectives.py` 两侧真实双卡均通过：NCCL 求和 `[3,6]` → CPU 求和、
gather、广播、对象交换 → 销毁 CPU 子组 → NCCL 求和 `[21,42]`。CPU 指针为 host
memory，GPU 指针分别属于逻辑设备 0/1；Jittor 默认 CUDA 与 `use_cuda=1` 全程保持。
该测试直接复现了默认 Store 重建导致的挂起/端口冲突，修复后不再重建服务。

独立双 GPU NCCL all-reduce 在两套环境均通过：rank 0 输入 `[1,2]`，rank 1 输入
`[2,4]`，两端均得到 `[3,6]`。通过 CUDA driver 指针属性确认实际设备为逻辑 0/1。
PyTorch Qwen TP=2 已生成 32 token；每个 worker 的 226 个参数、28 个 KV cache
及 32 次 forward 设备检查通过。Jittor 在修复 metadata 放置后也完成 32 token，
两 rank 的参数、KV cache、forward 和实际加载二进制检查通过，但第 4 个 token 起
与 oracle 不同，出现重复词；因此不是正确性验收通过。同份代码的 Qwen TP=1
32 token 与 oracle 完全一致。另行测试真实 vLLM PyNccl 的 all-reduce、all-gather、
broadcast、新建/复用输出、FP16/FP32、输入变化及 rank 延迟，共 256 项数值检查通过。
此前通过的 Jittor 原生 NCCL 求和不能替代这条 ctypes 通信路径的验证。

排查过 Python Stream 的句柄 0 与原生 PTDS 句柄 2 不同这一假设：实际 CUDA 事件
测例中，current/default/new 三种公开 stream 都等待 PTDS 的受阻写入；显式
nonblocking stream 负对照能提前完成，因此测例有效，但没有复现顺序错误。
未凭句柄数值不同修改生产代码。另一次实际 TP=2 诊断同时设置
`JITTOR_TRITON_FAST_SYNC=0`、`JITTOR_TRITON_SYNC_AFTER_LAUNCH=1`，仍从第 4 个
token 起发生同样的重复词；这项强同步没有解决错误，也不能单凭它排除所有异步问题。
该诊断正常生成 32 token，但收尾记录 worker 异常退出与共享内存清理警告，
不将退出码 0 等同于完整生命周期验收。下一步固定两 rank 的相同输入前缀，
比较逐步 input IDs、position、logits 和采样后请求状态，定位首次偏差。

## 运行隔离与尚未承诺的范围

- CPU `gloo` 名称映射到本仓 Store 控制传输，不是 libgloo 二进制绑定；不声称支持
  混合 PyTorch/Jittor rank 互通、全部 Gloo API 或训练。
- 外部启动器使用每轮唯一的 `JT_NCCL_ROOTINFO_FILE`。历史默认路径的 `.pgN`
  文件可能被后续运行读取；本轮一次子组初始化失败，换独立路径后通过该阶段。
  通用旧文件隔离机制仍需专门修复，不能以启动器隔离宣称已解决。
- spawn 会继承父进程缓存搜索路径。验证工具在启动 worker 时移除这些继承路径，
  并检查实际导入二进制所在目录；通用编译器搜索顺序仍需另行回归。
- 使用独立工作目录，防止 shim 自动扫描公共临时目录中的无关扩展。
- 尚未验收四卡、多机、PP、Graph/compile、量化或多卡吞吐。

可复用入口、环境和设备证据要求见
[vLLM 验证 skill](../../agent/skills/vllm-torch-compat/SKILL.md) 中的双卡小节。

## 仓库检查

CPU `tools/run_test_suite.py --tier core`：310 passed、44 skipped、1 xfailed；
跳过项不计入 CUDA 覆盖。适配层生命周期 11 passed，真实命名空间结构检查 12 passed。
最终结构全集为 1365 passed、8 skipped、4 failed；失败均为原有的 child-process
两项、pytest collection 两项（未改动的 executor thread / H3 decode 测试）。新增报告
最初触发索引行数约束，已压缩并通过文档治理复验。布局与生成 manifest 检查通过。
远端 1165 个生产文件 SHA256 与本地候选清单一致；保留清单与测试原始日志。
旧 adapter 注册 fixture 与真实 vLLM 导入不能在同一进程重复注册；将两组测试隔离后，
既有 backend 26 项通过。原始混跑错误保留，不把它记为通过。
