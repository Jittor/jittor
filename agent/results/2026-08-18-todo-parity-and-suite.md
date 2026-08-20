# 对拍框架、下游库覆盖与测试套件模式切分

日期：2026-08-18

代码基线：`3a3b04fa`（分支 `2.0`）

环境：`jt311`（Python 3.11.15，torch-shim 部署）、`jt312b`（Python 3.12.13，真实
PyTorch 2.12.1+cu126 与 torchvision 0.27.1+cu126），4x RTX 4090，CUDA 12.2/cuDNN 8。

## 结果

### 常见网络对拍（前向 + 反向）

新增 `tests/models/test_network_parity.py`。权重从真实 PyTorch 侧拷入 Jittor 侧，
比较输出、输入梯度和逐参数梯度，并断言实际比较的梯度数量，避免名称不匹配时退化成
只比前向。

| 网络 | CPU | CUDA | 比较的梯度数 |
| --- | --- | --- | ---: |
| ResNet-18（torchvision） | 通过 | 通过 | 63 |
| ViT | 通过 | 通过 | 33 |
| GPT-2 | 通过 | 通过 | 29 |
| 扩散 UNet | 通过 | 通过 | 45 |

容差：CPU 前向 2e-4 / 反向 2e-3，CUDA 前向 1e-3 / 反向 6e-3。

CPU 那一列同样修正过：`TestNetworkParityCPU` 原先不设 `use_cuda`，在有显卡的机器上
跑的是 GPU。现在设备由基类的 `use_cuda` 属性显式给出，两个子类各占一个值。上表是
修正后在 `jt312b`（真实 PyTorch 与本仓库 Jittor 同进程）上重跑的结果，8 条全过。

### 下游库数值对拍

新增 `tests/compat/torch/test_ecosystem_parity.py`。真实 PyTorch 与 Jittor 都要占用
`torch` 命名空间，无法同进程比较，因此同一个用例分别在两个解释器里执行：真实
PyTorch 侧生成权重与参考值，Jittor 侧加载同一份权重复算。权重按
`named_parameters + named_buffers` 传递并断言两侧完全覆盖——`state_dict` 不一定完整
（ms-swift 的 tuner 只报告 adapter），按 `state_dict` 传会让两侧骨干各自随机初始化。

CPU 与 CUDA 各 12 个用例全部通过：transformers 的 gpt2/llama/bert/vit/t5/whisper、
diffusers 的 UNet2DModel 与 DiTTransformer2DModel、peft LoRA、ms-swift LoRA、
mmcv `ConvModule`、mmengine `BaseModule`。

两侧运行完后都把自己实际使用的设备回报给对拍框架断言。Jittor 在有 GPU 的机器上
默认 `use_cuda=1`，对拍框架的 CPU 分支原先只是没去动这个标志，于是"CPU 用例"实际
上是 Jittor 跑显卡对 PyTorch 跑 CPU——**CPU 这一半此前从未真正测到**。修正后重跑，
24 条仍然全部通过，CPU 正确性这才算有依据。

误差度量按整次比较的梯度量级设下限。注意力 key bias 的梯度在数学上恒为零，两侧都只
有 1e-8 量级的浮点噪声，按该张量自身最大值归一化会把它放大成 1.96e-2 的假失败。

ms-swift 4.5.2 与 peft 0.19 的 LoRA `Linear` 签名不兼容，**真实 PyTorch 2.12 上报
完全相同的 `TypeError`**，属于上游版本组合问题；peft 0.17.1 下两侧都能跑通。

### 测试套件模式切分

Torch 兼容模式是进程级的，会改变惰性执行、归约默认值与梯度语义。`conftest` 之前在
选择 `tests` 目录时给整个套件打开该模式。同一批文件在两种模式下的对比：shim 模式
13 失败，原生模式 4 失败。

| 会话 | 修正前 | 修正后 |
| --- | --- | --- |
| Torch 模式 | 36 failed，4 errors | 6 failed，0 errors，1447 passed |
| 原生模式 | 收集阶段即报错，运行到 70% 无汇总退出 | 收集干净 1624 条 |

### 整轮结果（本轮，4x RTX 4090 机器，CUDA 构建）

| 会话 | passed | failed | skipped | 其他 |
| --- | ---: | ---: | ---: | --- |
| Torch 模式 | 1712 | 26 | 184 | 4 xfailed，3h02m |
| 原生模式 | 1083 | 137 | 357 | 8 xfailed，2 errors |

原生这一列是分目录跑出来的：`backends`/`compiler`/`core`/`ops` 四个目录里存在顺序
相关的硬崩溃，整目录一次跑会在中途 abort 掉，因此这四个目录改为按文件各起一次
pytest；其余目录整目录跑。`distributed`/`opinfo`/`system` 没有收集到用例。按文件跑
时只有两个文件仍然 abort：`compiler/test_tracer.py` 与 `core/test_setitem.py`
（后者是工作区里未提交的改动，未纳入本轮）。

原生这 137 条失败绝大多数是 CUDA 构建特有的。把失败最多的三个文件
（`test_parallel_pass`、`test_transpose_op`、`test_where_op`，CUDA 构建下合计 25 条
失败）放到 CPU-only 构建上重跑，只剩 3 条失败、20 条通过、20 条跳过，主因就是
KI-LOG-001：CUDA 构建下几乎捕获不到算子日志，而这些用例正是靠匹配日志来断言
relay 或 pass 是否生效。

`tests/ops` 下只有 `test_ops.py` 使用 OpInfo，其余断言的是原生惰性执行行为，已移出
Torch 模式。`tools/run_test_suite.py` 分两个会话跑完整套件并汇总。

## 修正的缺陷

| 缺陷 | 影响 | 证据 |
| --- | --- | --- |
| shim 预处理把 `use_mkl` 设为 0 | 去掉 `mkl_conv`/`mkl_matmul` relay，CPU 卷积 1.4ms→156ms、512x512 矩阵乘 0.6ms→11ms | `test_torch_bootstrap.py::test_preflight_leaves_onednn_enabled` |
| `reshape`/`view` 丢弃 `shape=`/`size=` 关键字 | diffusers DiT unpatchify 直接崩 | `test_torch_compat_reduce_shape.py::TestReshapeKeywordShape` |
| `state_dict` 不接受 `destination`/`prefix`/`keep_vars` | ms-swift 适配后的模型无法保存 | `test_torch_compat_serialize.py::TestStateDictTorchKeywords` |
| `home()` 把 `JITTOR_HOME` 写回共享配置，且原地重写 | 一次隔离运行变成整机默认；并发启动读到截断 JSON 直接崩 | `tests/compiler/test_jittor_home.py` |
| `print_trace` 无限等待 gdb | gdb 挂住或被崩溃收集器接管时整个进程一起卡死 | `test_tracer.py::test_gdb_backtrace_wait_is_bounded` |
| Python 变量追踪器在 3.11+ 直接 `LOGf` | `trace_py_var` 全部功能不可用 | `test_trace_var.py`，3.11 与 3.12 各 5 passed |
| 13 个带参数的模块级 `test_*` 辅助函数 | 被 pytest 收集后因缺 fixture 报错 | `test_pytest_contract.py::test_module_level_helpers_are_not_named_like_tests` |
| 结构门禁与布局脚本按文件系统扫描 | 跑过 lint 或 import 过包的工作区被判失败 | 布局脚本 4 种情况验证：干净 0、根目录多余文件 1、notebook 产物 1、`__pycache__` 0 |
| 下载外部数据集的用例无标记 | 主机不可达时阻塞到 900 秒超时而非快速跳过 | `network` 标记 + `--network` 开关 |
| `manual` 标记无人消费 | notebook 冒烟测试在整轮第 1100+ 条时把进程打死 | 整树运行跳过、显式路径仍运行 |
| 冷启动把 `jit_utils_core` 加载成两份 | 该库里定义的每个 flag 和整个日志捕获缓冲区各有两份，`log_capture_scope` 冷缓存下恒返回空 | `tests/compiler/test_cold_start_runtime.py` |
| `cutt_test` 从主机端给显存指针赋值 | cuda_managed_allocator 默认关闭，必然段错误，整轮原生会话在第 50 条左右被打死 | `tests/backends/cuda/test_cutt.py` 1 passed |
| relay 单测用 `get_allocator()` 却按主机指针读写 | 同上，有显卡时写进显存；崩溃后 gdb 崩进 apport，进程停在 ptrace_stop 永不返回 | `jt.tests.fused_op_relay_matmul()` 通过 |
| conv tuner 把 cuDNN 不支持的 filter 布局 relay 过去 | 生成引用未定义标识符的 kernel，运行期表现为通过空函数指针调用 | `tests/backends/cpu/test_mkl_conv_op.py` 不再崩溃 |
| 套件运行时 gdb 回溯无上限 | 一条崩溃用例把整轮会话挂死而不是记为失败 | `tools/run_test_suite.py` 关闭 gdb_path |
| OpenMP 按逻辑 CPU 起线程 | SMT 机器上每核超额订阅一倍，产生与规模无关的固定开销：同一次批量调用 64 线程 437us、128 线程 5955us | `tests/compiler/test_openmp_threads.py` |
| CPU 批量矩阵乘没有 oneDNN 通路 | Transformer 每层两次注意力乘法走通用内核，37 GFLOPS 对 oneDNN 的 1800 GFLOPS；BERT CPU 单步 11.4s | `tests/ops/test_mkl_batched_matmul.py` |
| 对拍框架的 CPU 分支不关 `use_cuda` | Jittor 跑显卡对 PyTorch 跑 CPU，CPU 那一半从未真正测到，且把 1.8x 慢报成 20x 快 | `tests/compat/torch/test_ecosystem_device_selection.py` |
| 计时步只同步不取值 | 惰性图里没人索要的梯度不求值，切片与求和还会被融合裁剪，BERT 报 0.049s 而非 0.141s | 同上，运行侧回报设备并完整读回梯度 |
| `pytest_ignore_collect` 声明了 pytest 9 已移除的 `path` | pluggy 拒绝整个 conftest，3.12 侧 oracle 会话收集阶段即失败 | `tests/models/test_network_parity.py` 在 jt312b 上 8 passed |

### 冷启动重复加载运行时库

冷启动必须先编出 `jit_utils_core` 才能 import 它，而 `cache_compile` 只有在这个模块
可 import 之后才能写缓存键，于是原来的顺序是：编译 → import → 再编译一次生成键。
第二次编译重新链接了一个已经被本进程映射的库；链接器写输出前会先 unlink，所以进程
持有的 inode 与磁盘上的不再相同，随后 `jittor_core` 解析依赖时把同名文件又加载了
一遍。`/proc/self/maps` 里能直接看到两个 inode，其中一个标着 `(deleted)`。

后果不是崩溃而是静默失效：`log_capture_start()` 打开的是 A 份的开关，算子却往 B 份
里写。冷缓存下 `log_capture_scope(log_v=1000)` 一条都拿不到，热缓存下同样的写法有
195 条，这解释了此前记为 KI-COMPILER-003 的全部现象。

改为在一个短命子进程里生成缓存键，父进程随后只 import 一次已经定稿的文件。修复后
冷启动只有一个映射，`log_capture_scope` 拿到 195 条覆盖 18 个文件，tuner 输出 5 条，
缓存键正常写出因此后续运行不会重复编译。

## Python 3.12

`jittor.selftest` 在 3.12 上通过（forward `(1.0, 4.0, 9.0)`，gradient `(2.0, 4.0, 6.0)`）。
完整套件在 3.11 与 3.12 上的失败集合逐条 diff：118 条共有，3.12 独有 4 条，3.11 独有
7 条。**没有 3.12 特有的框架性破坏。**

## 教程

新增三个纯计图、CPU 一分钟内跑完的教程，已纳入离线冒烟集：`transformer.md`（复制
任务 100% 准确率）、`diffusion.md`（完整 DDPM，双峰分布覆盖 48.7%/51.3%）、
`lora.md`（低秩适配，103x 改善且合并后最大差 2.4e-6）。

## 速度

### 测量方法本身的两个错误

先前这一节的所有 CPU 数字都不成立，原因有两个，两个都会把 Jittor 报得更快：

1. **CPU 用例跑在 GPU 上。** Jittor 在有显卡的机器上默认 `use_cuda=1`，两个对拍
   框架的 CPU 分支都只是没去动这个标志。BERT-base 单步因此报成 0.06s，真实 CPU
   数字是 11.4s。
2. **计时步没有真正算完。** Jittor 是惰性的，GPU 同步之后仍有没人索要的梯度未求
   值；而切片或求和会被融合进生产它的 kernel，只算被读到的那部分——`grad[0]` 与
   `grad.sum()` 都测不出真实代价。计时步现在完整读回每个梯度，PyTorch 在
   `backward` 返回时本来就已经算完，两侧拷贝的字节数相同。

现在运行侧会把实际使用的设备回报给对拍框架断言。

### CPU 上的两个性能缺陷

**批量矩阵乘没有 oneDNN 通路。** `MatmulTuner` 只识别二维的
broadcast+multiply+reduce 形式（源码里写死 `bcop1->shape.size() != 3` 且两个操作数
都必须是二维），`nn.matmul` 的批量分支又只在 `jt.flags.use_cuda` 为真时才走 cuBLAS。
于是 CPU 上每一次批量矩阵乘都落到通用 reindex 内核，而 Transformer 每层的两次注意力
乘法都在这条路上。按算子剖分 BERT 单步：DIM=5 的 broadcast 融合内核合计约 7.5s，占
10.6s 的七成；`mkl_matmul`（各个 Linear）只有约 1.2s。新增 `mkl_batched_matmul`
（含反向）后接上该分支。

**OpenMP 按逻辑 CPU 起线程。** OpenMP 自己的默认是每个逻辑 CPU 一个线程，SMT 机器上
等于把每个核心超额订阅一倍。代价不是缓慢下降而是断崖——同一次批量调用 64 线程
437us、128 线程 5955us，与问题规模无关的固定开销。PyTorch 默认取物理核心数，Jittor
此前不设置。改为默认物理核心数后（显式 `OMP_NUM_THREADS` 一律优先）：

| 形状 | 修复前 | 修复后 |
| --- | ---: | ---: |
| 2048x768 @ 768x768 | 406 GFLOPS | 1841 GFLOPS |
| 96x256x64 @ 96x64x256 批量 | 5955us | 380us |

注意力形状（8x12x256x64 @ 8x12x64x256）上三种实现的吞吐：通用 reindex 内核
37 GFLOPS、`mkl_batched_matmul` 1800 GFLOPS、PyTorch 2490 GFLOPS。

新算子的前向与两个梯度都与通用路径逐元素比对过（含四种转置组合与批维广播），见
`tests/ops/test_mkl_batched_matmul.py`；线程默认值见
`tests/compiler/test_openmp_threads.py`。两项修复之后 24 条下游库对拍仍然全过。

### 真实尺寸单步（空载，两项修复之后）

| 用例 | CPU: PyTorch | CPU: Jittor | 修复前 | CUDA: PyTorch | CUDA: Jittor |
| --- | ---: | ---: | ---: | ---: | ---: |
| large_convnet | 0.750s | 1.001s (1.33x) | 1.82x | 0.030s | 0.037s (1.21x) |
| large_transformers_vit | 0.669s | 2.123s (3.17x) | 10.3x | 0.113s | 0.144s (1.27x) |
| large_transformers_llama | 1.314s | 5.082s (3.87x) | 14.5x | 0.267s | 0.297s (1.11x) |
| large_transformers_bert | 0.825s | 3.799s (4.61x) | 13.9x | 0.148s | 0.162s (1.09x) |
| large_transformers_gpt2 | 1.577s | 7.293s (4.63x) | 13.6x | 0.183s | 0.299s (1.63x) |
| large_diffusers_unet2d | 0.599s | 2.962s (4.94x) | 8.8x | 0.061s | 0.141s (2.31x) |

CUDA 1.09x–2.31x，CPU 1.33x–4.94x。再剖分一次 BERT，DIM=5 的热点已经消失
（`mkl_batched_matmul` 只剩 80ms），剩下的差距分散在 softmax、layernorm、gelu 这些
访存受限的逐元素融合内核上，没有单一热点，需要的是融合质量与带宽层面的工作，不是
再补一个 relay。

`JITTOR_ECOSYSTEM_SPEED_RATIO` 可以把比值变成断言；默认只打印，避免共享机器上的
测量抖动造成假失败。

## 未解决

- Torch 模式剩余 5 条整轮失败在单独运行时全部通过，属于顺序依赖；其中两条是
  installer 幂等性断言，需要查清是哪个前置用例改动了 `torch.*` 模块图。已确认把
  嫌疑最大的 bootstrap、install context、compat mechanisms 与这两条放在一起跑
  （88 passed）并不能复现，需要更大范围的二分。
- CPU 上仍比 PyTorch 慢 1.33x–4.94x。批量矩阵乘与线程数修好之后不再有单一热点，
  时间分散在逐元素融合内核上，属于融合质量与访存带宽的问题。
- `tests/ops/test_matmul.py` 在 CUDA 构建下 6 条失败、在 CPU-only 构建下全过，起因
  是 CUDA 构建几乎捕获不到算子日志，见 KI-LOG-001。与本轮改动无关（把改动全部撤回
  后失败集合完全相同）。
