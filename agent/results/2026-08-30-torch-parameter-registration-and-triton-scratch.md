# torch 的参数注册语义、Triton 尾部 scratch 参数与向量化归约的浮点顺序

- Status: 十一道门全绿（native 739 / CPU Torch 1513 / CUDA+oracle 1809 / `nox`
  的 `cuda` `structure`(245) `cpu` `tutorials` `optional` `packaging` `mpi`
  `nccl` 均 `EXIT=0`）；vLLM 7B 与 PyTorch 输出逐 token 一致，速度 1.22-1.24x
- Last reviewed: 2026-08-30
- Baseline: `f10e9480`
- Owner: Torch compatibility and downstream integration maintainers
- Review when: 参数注册规则、Triton 版本、CUDA 显存查询或 CPU 归约的浮点顺序变化

## 结论

三处 shim 缺口被下游大模型推理暴露出来，各自都超出 vLLM 的范围：

1. **参数注册语义**。Jittor 是"赋值即参数"，torch 只认 `nn.Parameter`。shim 下两套
   约定必须同时成立，判据是**谁在赋值**，不是被赋值对象的类。
2. **Triton 尾部 scratch 参数**。3.7 在 kernel 参数表末尾追加 `global_scratch` 与
   `profile_scratch`；桥接只建模了一个前置指针，安全网因此拦下所有这类 kernel。
3. **`torch.cuda.mem_get_info` 是固定的 64 GiB 桩**。推理框架按它规划显存预算。

另外，`test_cpu_results_are_unchanged` 的失败不是缺陷：CPU 归约现在会被向量化，
浮点求和顺序随之改变。原因已用生成的 kernel 与同旗标的 C++ 复现逐位坐实。

## 一、参数注册：判据是赋值方，不是类

### 现象

vLLM 加载 7B 权重后拒绝启动：

```text
ValueError: Following weights were not initialized from checkpoint:
  {'model.layers.2.self_attn.attn.v_range', ...}
```

### 根因

`vllm/model_executor/layers/attention/attention.py:117` 写的是

```python
layer.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
```

torch 里这只是个普通属性；shim 下 `Module.__setattr__` 把任何公开 Var 赋值都标记成
参数，于是它进了 `named_parameters()`，vLLM 的加载器把它当成"检查点里应该有、却没被
初始化"的权重。

十行即可复现，两侧确实分叉：

| | shim（修前） | 真 PyTorch |
|---|---|---|
| `named_parameters` | `['w', 'plain']` | `['w']` |
| `state_dict` | `['buf', 'plain', 'w']` | `['buf', 'w']` |

影响面远不止 vLLM：优化器会去更新非参数，`state_dict` 会多出键。

### 为什么不能"只认 Parameter"

shim 下 `nn.Linear` 就是 Jittor 的类，权重是普通赋值。改成只认 `nn.Parameter`，
ViT 的 200 个参数会几乎全部消失。

### 第一版规则错在哪

先按**模块类的归属**分流（类定义在 `jittor.` 之外就走 torch 规则）。这条规则让
生态对拍掉了一个权重：

```text
no counterpart for saved weights: ['decoder.embed_positions.weight']
```

`WhisperPositionalEmbedding` 是 transformers 定义的类，但它的 `weight` 是
**Jittor 自己的 `nn.Embedding.__init__`** 用普通赋值声明的。类的归属和赋值代码的
归属是两回事。

### 第二版规则也错了：判据不能是"谁在赋值"

第二版改按**赋值方所在模块**分流（`sys._getframe(2)`）。它修好了 Whisper，但仍然
是"默认降级、除非有理由保留"，于是又咬了两次，两次都是**静默丢参数**：

- vLLM 的 `self.bias = Parameter(torch.empty(...))`——适配器替换过的构造函数丢了
  标记，bias 被降级、加载器从不填充，未初始化内存进了矩阵乘（见第五节）；
- `tests/distributed/test_fsdp2_nccl.py` 里 `self.output_bias = jt.ones((3,))`——
  纯 Jittor 写法，只因进程里装了 shim 就被降级，`state_dict` 的 numel 由 18 变 15。

### 最终规则：只在有正面证据时降级

只有 shim 自己的 `torch.tensor` 产出的 Var 会被标记 `_jt_plain_tensor`，
`Module.__setattr__` 只降级带此标记且不带 `_is_torch_parameter` 的值：

- `layer.v_range = torch.tensor(...)` → 不是参数（正是要修的那条）；
- 其它一切照旧是参数，**权重不可能被静默丢掉**；
- `torch.ones` / `torch.zeros` 不能用作信号——它们**就是** Jittor 自己的函数，
  标记它们会把 Jittor 各层用赋值声明的权重一并降级；
- 代价是 `layer.x = torch.zeros(3)` 仍会被当成参数。这是已知的不完整，
  但方向是安全的一侧。

`register_parameter` 也补上了标记：torch 里它就是"显式声明为参数"的调用。

验证：

| 场景 | 结果 |
|---|---|
| torch 作者的类 + `nn.Parameter` / 普通张量 | 与 PyTorch 逐项一致 |
| torch 作者的类**继承** Jittor 层（Whisper） | `weight` 保留 |
| 测试里用 `jt.ones` 声明的属性（FSDP2） | 保留，numel 18 |
| Jittor 自己的 `nn.Linear` | `weight`, `bias` 均在 |
| 原生 Jittor（不开 shim） | 赋值即参数，不变 |
| ViT 真实模型 | 两侧同为 200 / 200 |

## 二、Triton 3.7 的尾部 scratch 参数

`_topk_topp_kernel` 报"打包 8 个参数、kernel 声明 10 个"，被桥接的安全网拦下。

Triton 的 nvidia 后端在 `driver.c` 的 "Add scratch objects" 处，把
`global_scratch`、`profile_scratch` 两个指针**追加在参数表末尾**；3.2 两者都没有。
桥接原先假设 scratch 是**前置**的 param 0，且只有一个。

改成由元数据驱动：元数据暴露哪个字段，就追加哪个指针；字段存在但尺寸为 0 时
kernel 仍然声明该参数，此时传空指针（与 Triton 自己的行为一致）。分配大小也补上了
Triton 那份 `grid_size * num_ctas` 的缩放。

顺带确认了一件事:`PassManager::run failed` **与 Jittor 无关**——在真 PyTorch 环境
里把 Triton 降到 3.2.0，同一 kernel 在同一行挂同样的错。

## 三、`mem_get_info` 返回真实显存

原先是 `lambda *a, **k: (64*1024**3, 64*1024**3)`。torch 的语义是 `cudaMemGetInfo`：
整卡的 free/total，包含其它进程、CUDA context 和 Jittor 池里已持有但空闲的块。
vLLM 用 `total*util - (total-free) - 激活峰值` 规划 KV cache，读到虚构的 64 GiB
就会算出无意义的预算。

改为调用 `cudaMemGetInfo`，取不到 cudart 时退回 Jittor 的 `total_cuda_ram`。
（`memory_reserved` 的映射本来就是对的：实测 `total_cuda_used` 跟踪的是**池**，
free 之后不降、`jt.gc()` 之后才降。）

## 四、向量化让 CPU 归约的求和顺序变了（不是缺陷）

`test_cpu_results_are_unchanged` 失败：2^18 个标准正态求和，相对误差 1.5e-4，
超过 1e-4 的容差。

生成的 kernel 是标量顺序累加——accumulator pass 把 `yp[yid]` 提成了局部变量，
去掉了每次迭代的内存往返，编译器因此可以重排。Jittor 用的旗标是 `-Ofast -march=native`，
`-ffast-math` 允许重结合。用同样旗标编译同一个循环：

| 求和方式 | 结果 |
|---|---|
| `-O2` 普通循环（朴素顺序） | 42.199001 |
| `-Ofast -march=native` 普通循环 | **42.191078** |
| 显式 8 路部分和 | **42.191078** |
| float64 参照 | 42.197387 |

与 Jittor 的 42.191078 逐位相同，即 AVX2 的 8 路向量化求和。

这组数据由 209000 的总量抵消到 42（条件数约 5000），固定 `rtol` 考的是数据运气而非
实现——朴素顺序恰好更近，8 路恰好更远。测试改为浮点求和的教科书误差界
`|err| <= log2(n) * eps * sum|x|`，实测 0.0063 远在界内。

## 五、vLLM 7B：十个阻碍全部跨过，输出正确

Qwen2.5-7B-Instruct 现在给出 `' Paris. Which of'`（token `[12095, 13, 15920, 315]`）。

| # | 阻碍 | 归属 |
|---|---|---|
| 5 | `v_range` 被当成参数 | Jittor（已修，见第一节） |
| 6 | `gate_up_proj.weight` KeyError | 第 5 条的回归，已修 |
| 7 | Triton `PassManager::run failed` | 环境（Triton 3.2 太旧） |
| 8 | scratch 参数个数不符 | Jittor（已修，见第二节） |
| 9 | KV cache 预算为负 | 配置（收小 profiling 批量） |
| 10 | 前向输出全 NaN | **第 5 条的第二次回归**，已修 |

### 第 10 条：未初始化的 bias 进了矩阵乘

定位过程是逐层缩小的：

1. 按层量 NaN：第 0 层 attention 输出干净，第 1 层输入干净却输出全 NaN。
2. 把那次调用的输入落盘、在 vLLM 之外原样回放——**离线复现**，说明是数据不是图上下文。
3. 看数据：Q/K/V 的量级是 **1e34**。bf16 装得下，但 `q·k` = 1e68 在 fp32 溢出成
   `inf`，softmax 于是给出 NaN。所以真正的问题在上游。
4. 改量 absmax 而不是 NaN，逐子模块回溯：`L1.input_ln` 输出 24.75（正常），
   `L1.qkv_proj` 输出 **1.038e34**——而它的 weight absmax 是 0.668。
5. 查 bias：`in_params=['weight']`，**bias 根本不在 `named_parameters()` 里**，
   加载器从不填充它，`torch.empty` 的未初始化内存（1e34 / 2.6e36 / 4.3e37）直接
   参与计算。第 0 层碰巧是 0，所以只有第 1 层起才爆。全模型只剩 114 个参数。

根因是第 5 条改动的第二次回归。vLLM 的 linear 写的是
`self.bias = Parameter(torch.empty(...))`，而适配器把 shim 的 `Parameter` 元类
`__call__` **整个替换**成自己的实现，从不设 `_is_torch_parameter`。第 6 条我只补了
`register_parameter` 那条路径，这条直接赋值的路径还漏着。

修在适配器：它既然替换了 shim 的 Parameter 构造函数，就得守住那个契约。
修后参数数 `114 -> 199`，28 个 qkv bias 全部就位，量级 27~171，输出正确。

### 教训

这条规则的失败模式是**静默丢参数**——只有下游数值爆炸才暴露。契约本身是对的
（torch 里普通 Tensor 属性同样不是参数），但任何绕开 `nn.Parameter` 构造参数的
第三方代码都会踩中。凡是替换 shim 构造函数的适配器，都必须保留标记。

过程中还发现基准脚本自己漏了 `bs.patch_vllm()`，适配器的 attention/MoE 补丁因此
全是空转，走的是 vLLM 原版 FlashAttention。

## 六、打包门禁此前就是红的

`nox -s packaging` 有两处，都与今天的改动无关：

1. sdist 里混进了 `docs/locales/**/LC_MESSAGES/*.mo`。这些是 docs i18n 构建从被
   跟踪的 `.po` 编译出来的产物，git 不跟踪、`.gitignore` 也已排除，但
   `recursive-include docs *` 会把本地构建留下的任何东西一并打包，而现有的 prune
   规则够不到 `LC_MESSAGES/`。加一条 `global-exclude *.mo`。
2. wheel 内容基线停在 `3a3b04fa`，此后多次功能提交（acl 算子、mkl batched matmul、
   各 CUDA backend、opt pass、fsdp2 state dict）让它报出 24 处新增、118 处内容变化。
   刷新前逐项核对：wheel 的 **817 个成员全部**对应仓库中被 git 跟踪的源码，零意外
   内容，新增与变更也都落在 `jittor/` 源码目录内。

刷新后的基线按文件哈希锁定内容，所以每次改动已发布的源码都要随之刷新一次。

另：`nox -s optional` 一度报 `4 failed`（torchmetrics），是并发高负载下的超时——
单轮 torchmetrics 就要 18 分钟，而每测超时是 600s。空载重跑 `EXIT=0`。

## 七、分布式两道门：三个真实缺陷

`nox -s mpi` 与 `nox -s nccl` 此前都是红的，与本轮改动无关：

1. **`nccl_test` 编译不过**。`helper_cuda.h` 把 `_cudaGetErrorEnum(ncclResult_t)`
   守在 `#ifdef NCCL_H_` 后面，只有 `nccl.h` 先于它包含时才存在；而它自己的
   include guard 会让"后来的包含"变成空操作，于是任何更早拉进 `helper_cuda.h` 的
   翻译单元里，`checkCudaErrors(ncclResult_t)` 都落到 `cudaError_t` 的重载上。在
   `nccl_wrapper.h` 的 `nccl.h` 之后补一条与顺序无关的声明（`check` 里的调用是
   依赖名，ADL 在实例化点能找到）。
2. **`nccl_test` 运行时段错误**。`jit_run` 第一行 `output->ptr<T>()[0] = 123;`
   在**主机端写设备指针**，信号码正是 "Invalid permissions"。改用 `cudaMemcpy`。
3. **NCCL 初始化失败**：`peer access is not supported between these two devices`。
   这台机器的 8 张 RTX 4090 之间**完全没有 P2P**（`nvidia-smi topo -p2p r` 全是
   CNS），而 NCCL 在这种板子上硬失败而非回退。判据放在 `launch.py`：每个 rank 只
   看得到一张卡，唯有分发之前才有完整设备列表；探测到没有任何一对可互访就设
   `NCCL_P2P_DISABLE=1`（`overwrite=0`，操作者显式设置优先）。

修后 `nox -s mpi` `EXIT=0`，`nox -s nccl` `EXIT=0`（`4 passed`）。

## 八、vLLM 7B 与 PyTorch 的速度对拍

参照物是同版本 vLLM 0.11.0 跑在真 PyTorch 2.8.0+cu128 上（`--target` 隔离安装，
transformers 固定在 4.56.2）。两侧设置完全一致：Qwen2.5-7B-Instruct、bfloat16、
`enforce_eager=True`、`gpu_memory_utilization=0.80`、`max_num_batched_tokens=512`、
关闭 prefix cache，3 次预热 + 21 次计时取中位。

**输出逐 token 一致**：两侧都是 `' Paris. Which of'` / `[12095, 13, 15920, 315]`。

### 批量扩展的缺陷及修复

| batch | 修前 | 修后 | PyTorch | 修前比值 | 修后比值 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0833 | 0.0833 | 0.0673 | 1.24x | 1.24x |
| 8 | 0.1525 | **0.0904** | 0.0727 | 2.10x | **1.24x** |
| 32 | 0.3841 | **0.1057** | 0.0864 | 4.44x | **1.22x** |

修前 Jittor 几乎随批量线性变慢，PyTorch 却几乎不变。拆开量：batch 32 时
`max_tokens=1` 用 0.307s、`max_tokens=4` 用 0.372s，即 decode 约 21.5ms/步且**不随
批量增长**（融合 decode kernel 生效：一次 8 序列生成里 `hit=84 / miss=28`，28 次
miss 正是 prefill 的 28 层），**prefill 独占 0.285s**，而 PyTorch 同批量 prefill 约
0.026s。

根因是适配器的 `flash_attn_varlen_paged` 对 prefill 走**逐序列的 Python 循环**，
每序列每层约 6 个 kernel，batch 32 就是 28×32 次迭代。改为：当各序列的 query 与
key 长度一致（同长 prompt 批总是如此）时，一次 gather + 两次批量 matmul + 一个可
广播的 mask 完成；参差批量仍走原循环。与循环路径逐元素对拍最大差 `1.49e-08`；
另用 5 条长度各异的 prompt 端到端复核，两侧 8 个 token 逐一相同，回退路径无损。

### 剩下的 1.24x：GPU 在等 CPU，不是 kernel 慢

先前把它记成"小 kernel 慢"是不准确的，更正如下。

PyTorch 侧用 `torch.profiler` 取到权威分解：**单次生成 GPU 总忙时 63.2ms**，其中
**GEMM 占 97.0%（61.3ms）**，非 GEMM 只有 **1.88ms**；它的融合 kernel 每个仅
`0.2~0.6us`（RMSNorm 0.20us、SiluAndMul 0.63us、RoPE 0.22us、KV 写 0.17us）。

Jittor 的 profiler 把同类算子报成 `7~15us`。同一算法不可能差 30~70 倍——那是
**CPU 侧每算子耗时**，不是 GPU kernel 时间。GPU 利用率采样证实了这点：

| | GPU 利用率 | wall | 推得的 GPU 忙时 |
|---|---|---:|---:|
| PyTorch | 峰值 96%（profiler 口径 94%） | 67ms | 63.2ms |
| Jittor | 73~82% | 83ms | ~65ms |

**两侧的 GPU 实际工作量相当（62~65ms）**，Jittor 慢在 GPU 有约 20% 的时间空等
CPU 派发，即每次生成约 18ms 的 CPU 时间没有被 GPU 执行掩盖。合成基准也印证方向：
同样"一个大 GEMM + 8 个小算子"，PyTorch 掩盖了 73% 的小算子 CPU 成本
（266+94 → 292），Jittor 只掩盖 52%（276+40 → 296）；而 Jittor 的 8 个小算子本身
只要 39.6us，比 PyTorch 的 94.3us **便宜 2.4 倍**。

顺带做了一项确定的优化：这些融合 kernel 每次调用都要把约 1.4KB 的 CUDA 源码用
`%` 重新格式化一遍。改为按参数记忆化（`_cuda_inference.cached_source`，覆盖
rms_norm / rope / swiglu / kv_cache / packed_qkv 共 9 处），单次调用省 `1.7~1.9us`
（RMSNorm `13.76 -> 11.91us`、SwiGLU `9.36 -> 7.66us`），端到端 `0.0833 -> 0.0826s`
（约 1%）。诚实地说，这印证了归因：省下 CPU 时间只按比例兑现，因为瓶颈是 CPU 派发
总量，而不是某一处热点。

### 逐一排除的杠杆

配置层与 Python 层能碰的都试过了，逐条记下以免后来者重走：

| 试探 | 结果 |
|---|---|
| `VJ_SYNC_EVERY` 1→28（适配器刷新窗口） | `0.0827~0.0839s` 无规律，同步点不是杠杆 |
| `VJ_EAGER=1`（关惰性执行） | `0.159s`，**慢一倍**——惰性执行是承重的 |
| `use_parallel_op_compiler=0` / `check_graph=0` / `para_opt_level=3` | 无变化 |
| `gopt_disable=1` / `para_opt_level=0` | 运行失败 |
| `JITTOR_TORCH_CUDA_EMPTY_CACHE` gc vs noop | 无变化，`empty_cache` 不在热路径 |
| shim 的张量操作（view/reshape/transpose/slice/contiguous） | 与 PyTorch **持平**，多数更快 |
| 通用小算子派发 | Jittor **便宜 2.4 倍** |
| 解码 shape 的 GEMV | Jittor 在两个大 shape 上**更快** |

也就是说：GEMM 不慢、算子派发不慢、张量包装不慢、同步点不是问题。剩下的开销落在
Jittor 执行器从"算子创建"到"kernel 启动"之间的 C++ 环节——每步解码的图结构完全相同
却被反复重建，而 Jittor 没有 CUDA graph 那样的捕获重放能力（源码里只有
`gopt_disable` / `para_opt_level` 这类旋钮，没有 graph capture）。

要真正抹平这 18ms，需要把每算子的 CPU 成本降到接近零——也就是 CUDA graph 级别的
捕获重放，属于运行时层面的能力，本轮没有做。

（测量陷阱记录：Jittor 的惰性图会做公共子表达式消除与死代码消除，用固定输入或不
保留输出的微基准会得到荒谬的数字——`qkv_proj` 一度量出 9.4us，而它要读 33MB 权重，
意味着 3.5TB/s。上面所有 Jittor 微基准都改用了互不相同的输入并保留输出引用。）

**结论：vLLM 的"能跑"与"对拍一致"已达成，"速度不更慢"未达成**——各批量下稳定在
约 1.22-1.24x，且已定位为 CPU 派发未充分重叠，而非算子实现慢。

## 复现

```bash
# 参数注册的两侧对拍
JITTOR_TORCH_SHIM=1 python -c "..."   # 见 tests/compat/torch/_torch_compat_checks.py

# 归约的求和顺序
g++ -Ofast -march=native red.cc && ./a.out
```
