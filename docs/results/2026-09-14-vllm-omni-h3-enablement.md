# MiniMax-H3 经 jittor + torch-compat + vLLM-Omni：shim 缺口

- 状态：接受。单卡与两卡 TP2 的 vLLM-Omni 请求端到端正确：单卡首个完整 `fl2va`
  请求 `GENERATE` 由 710.7 s 降到 39.6 s，OpenAI 服务与离线路径持平；TP2 的噪声画面
  （§34）、粉色标题卡（§35）与部署侧视频噪声（§48–§49）都已修复，去掉临时绕过后
  4/4 段视频正常。**仍开放**：参考 VAE 解码比真 PyTorch 慢 1.33x（§46–§47），以及
  文末「未结事项」
- 日期：2026-09-14 至 2026-09-22（速度 2026-09-15，TP2 噪声画面 2026-09-17，部署噪声
  2026-09-22）；2026-10 压缩归档
- 基线提交：最终结论对应 `2198d438`（§49 的修复为 `a60d12f9`）；各节修复的提交见正文
- 验证范围：NVIDIA H20 96 GiB，单卡与两卡 TP2；vLLM-Omni MiniMax-H3 recipe（层级
  offload），`FLASH_ATTN` 走经 shim 构建的官方 flash-attention 扩展；对照为同机真
  PyTorch（§33 的 diffusers 路径、§37–§47 的 VAE 解码类）。同机有其他租户长期占满
  全部 GPU，绝对时间只能当上限、比值才可引用（§40）。未覆盖：ROCm、NPU、TP>2、量化
- 维护者：Torch 兼容层维护者
- 复查条件：offload 路径、`torch.device` 放置、conv/pow op-type 表、`.data`/视图处理、
  `torch.as_strided`/`empty_strided`、matmul dtype 路由、triton bridge 的 guard 与同步
  策略、`fuse_op_limit` 默认值或执行器的跨线程计算流交接（`backend_compute_stream_*`）
  变化

由约 6,700 行实验室笔记压缩而成；日记、原始日志与被撤回的假设都在 git 历史中
（`git log -p -- <this file>`）。保留章节号是因为代码注释、测试与其它文档会引用它们。
实验室 harness（`serve-vllmomni.sh`、`run_tp2_probe.sh`、`probe_*.py`、
`env-jittor.sh`）位于 `$JITTOR_LAB_ROOT/minimax-h3/`，运行状态位于
`$JITTOR_LAB_ROOT/_state/h3/`。

## 问题

`jittor` + `JITTOR_TORCH_SHIM=1` + `vllm-omni` 能否在不修改 vLLM-Omni 的情况下运行
MiniMax-H3 推理？参考运行是单张 H20，采用 `recipes/MiniMaxAI/MiniMax-H3.md` 中的
层级 offload 配置（DiT 与 Qwen3-VL 编码器无法同时驻留：约 62 GiB + 约 63 GiB 的
BF16 权重）。下文每一行都给出 jittor 侧的缺陷、其修复，以及存在时的回归测试。

## 各节发现

### 引擎构建与首个请求（§1–§11）

| § | 缺陷 | 修复 / 证据 |
| --- | --- | --- |
| 1 | `TCPStore` 绑定 `127.0.0.1`，而客户端在多地址 `localhost` 上拨 `::1` | 绑定客户端拨打的第一个地址；`tests/distributed/test_process_store.py::TestHostnameRendezvous` |
| 2 | `with torch.device("cpu"):` 没有改变分配的默认设备 | 工厂放置时查询线程局部 device-context 栈 |
| 3 | op-type 表由运行时的 `use_cuda` flag 选择，于是主机单元拿到仅 CUDA 的 `pow`、半精度 `abs` 与半精度比较 | `OpByType::expand_op(args, is_cuda)` 改由单元自身的 `JIT_cuda`/`JIT_cpu` 决定 |
| 4 | `nn.Conv3d` 拒绝 `padding_mode` | shim 用请求的模式补零，再以无 padding 方式卷积 |
| 5 | 缺少 `torch.get_device_module` | 已添加 |
| 6 | `tensor.data = other` 拷贝了元素 | 用 `_update` 重新绑定，与 torch 替换数据的行为一致 |
| 7 | 改变大小的重新绑定在惰性记录的视图上中止 | 记录的视图只在仍然放得下时才重新推导；`tests/core/test_transpose_view_staleness.py` |
| 8 | 缺少 `torch.as_strided` / `torch.empty_strided` | 已添加（基于 gather 的窗口；连续分配） |
| 9 | 跨设备 `copy_` 把目标搬到源所在设备 | 先在目标设备上实体化源 |
| 10 | 核心辅助函数按环境放置分配 | `placement_scope_like`；`*_sdp_enabled` getter；可调用的 bool `dtype.is_complex` |
| 11 | 混合 fp32/fp16 的 `linear` 未匹配任何 cuBLAS 行，回退到实体化的外积（在 `512x2048x6144` 下，53.6 ms 对 fp16 的 0.49 ms） | 将该对解析到 autocast 计算 dtype 并重试 relay（`c34b1240`） |

完成 §1–§11 后，一个 `fl2va` 请求可以完成（124 帧，每声道 165,600 个音频采样，
96 GiB 卡上峰值 24.3 GiB）；§11 是让它变快的关键。

### 服务、正确性与单卡速度（§12–§18）

| § | 缺陷 | 修复 / 证据 |
| --- | --- | --- |
| 12 | 在 stub 中 `from torchaudio.functional import melscale_fbanks` 失败 | 为 `torchaudio.*` 增加 `_AnyFinder`；`vllm-omni serve ... --omni --num-gpus 1` 在 39.5 s 内返回片段，离线为 39.6 s |
| 13 | TP2 启动：必须导出 `JITTOR_TORCH_DISTRIBUTED_AUTO_INIT`；无 mpirun 的 NCCL 需要 `JT_BUILD_NCCL_INCLUDE_PATH`/`_LIB_PATH`；`make_cache_dir` 竞态；存在 store 时 `JT_NCCL_ROOTINFO_FILE` 未被推导；子组 communicator 创建在错误设备上；在外部 context 上 `record_event`；`share_with` 未做范围检查；扩展 `device()`/`CUDAGuard` 固定在 index 0 | 前两项是环境设置，其余在 jittor 或 shim 中修复；`compat/tests/torch/test_cpp_extension_device_index.py`（6/6） |
| 14 | `linspace` 没有精确落在 `end` 上，于是每个步数 >= 4 的请求都因 `sigma_next >= 0` 失败 | 固定端点；`tests/ops/test_random_op.py`。实验室此前还在 recipe 要求 50 步时用了 2 步。shim 的 VAE 解码与真 torch 的偏差为 0.30/255（256x256）与 0.31/255（512x512） |
| 15 | 256x256、50 步下的速度：DiT offload 时 3.35 s/步，驻留时 1.68 s/步 | 第一杠杆是驻留，第二是 attention 后端（§16） |
| 16 | `int(cu[1])` 把 `cu_seqlens` 停放在主机上，而 shim 的「已在 cuda 上」捷径信任元数据，于是 `FLASH_ATTN` 拒绝它；`TORCH_SDPA` 在 256x256 以上丢失打包的多文档边界 | 捷径要求真实驻留；`compat/tests/torch/test_native_tensor_placement.py::test_a_device_var_parked_on_the_host_is_moved_back_by_to_device`。832x480：10.09 s/it，`TORCH_SDPA` 为 25.2 s/it |
| 17 | 扩展构建缓存忽略头文件改动，返回过期的 `.so` | 在命令中加入 shim ABI 头文件摘要（`-DJTORCH_SHIM_ABI=`）；`TestShimHeadersInvalidateTheBuildCache` |
| 18 | 孤立 worker 与残留服务器产生看似错误的结果 | 运维问题：按进程树停止，先查 `ps` 再归咎 jittor |

### TP2 设备与线程缺陷（§19–§32）

| § | 缺陷 | 修复 / 证据 |
| --- | --- | --- |
| 19 | 间歇性 TP2 启动挂起：rank 0 在 `ncclCommInitRank` 内持有 GIL，而其 store 服务器必须应答 rank 1 | 在集合通信前加两阶段 barrier；`tests/distributed/test_nccl_store_rendezvous.py` |
| 20 | 六个生成的直接 flash-attn 入口发出 `CUDAGuard device_guard{0}` | 从输入 tensor 取 guard；`compat/tests/torch/test_flash_attn_compat.py::TestPackedEntryDeviceGuard` |
| 21 | 设备 1 上有一个不可读的 index buffer | 仅诊断：context 已死，该读取是受害者 |
| 22 | `ArrayOp::run` 通过空分配器释放 | 跳过该 free；`tests/core/test_array.py::TestArrayOpDoesNotAssumeAnOutputAllocator` |
| 23–24 | 四个 loader 线程破坏融合批次（`fused_op.cc:89`）；请求阶段本身是单线程的 | 由 `SERIALISE=1` 确认；在 §29 修复 |
| 25 | 怀疑 `transpose` 步长撒谎 | 已撤回：CUDA transpose 是急切的；尝试的 `set_storage_strides` 破坏了数值并已回滚 |
| 26 | `accelerator_current()` 是进程全局的，而 CUDA 的当前设备是线程局部的，于是 loader 线程在设备 0 上运行 | 首次使用时绑定调用线程；`tests/runtime/test_runtime_device_state.py::test_current_device_binds_the_calling_thread` |
| 27 | triton bridge 在 legacy stream 上启动；`jt.unique` 与 `*_like` 家族使用环境设备；`device_raw_ptr` 迁移到环境设备 | `_launch_stream()`；`device_scope_like`；`tests/ops/test_unique_paths.py::TestUniqueOffTheAmbientDevice`、`tests/core/test_array.py::TestLikeConstructorsKeepTheDevice` |
| 28 | flash-attn 构建标识未覆盖 shim 头文件 | 在标识中加入 `shim_hdrs=` 摘要 |
| 29 | 一个 batch node 被另一线程在 planner 之下释放 | 执行器为每个已规划的 var 持有一个 `VarPtr`，外加一份边快照：20/20 次 loader 竞态运行干净（keep-alive、加锁与仅快照三种变体各自失败） |
| 30 | 部署的核心缺少每线程 stream 集合；`_as_strided` 越界 gather | 实验室重新部署；`as_strided` 现在根据存储校验视图 |
| 31 | `Tensor.to(1)` 静默丢弃设备索引 | 裸 int（不是 `bool`）就是设备索引；`compat/tests/torch/test_multi_device.py::TestMultiDeviceFacade::test_to_and_cuda_with_an_index` |
| 32 | `_Storage.nbytes()`/`set_` 描述的是 tensor 而不是其分配；triton driver 固定在设备 0 上并在实体化前切换 context | 按字节寻址的存储（`74bfe44c`）；每设备 driver 与 context 顺序（`95ad1ea2`）；`compat/tests/triton/test_triton_backend.py::TestDriverIsPerDevice` |

§32 之后，TP2 在 256x256、2 步、`FLASH_ATTN` 下完成：不 offload 为 495.3 s，使用
默认文本编码器 offload 为 50.1 s，零非法地址。

### 画面正确性（§33–§35）

- **§34 TP2 帧是噪声。** shim 的 `torch.Generator` 只保留种子并从 jittor 的全局流
  取数，因此两个 rank 去噪了不同的 latent。现在每个 generator 拥有自己的流
  （`0ea3448c`）；`compat/tests/torch/test_generator_streams.py`。2 步下 TP1 对
  TP2：block SSIM 0.972–0.985（TP1 对自身：0.942–0.985）。
- **§35 粉色标题卡。** triton bridge 的过度读取 guard 把带步长的 `chunk` 视图当作
  连续视图回弹，静默破坏了两个 rank 上的 AdaLN 调制内核。现在只对连续操作数采取
  回弹；`TestGuardedBounceRequiresContiguous`。block 输出对 diffusers 的 cos 从
  0.18744 升到 0.99978，四段端到端片段都遵循提示词。

## 速度

### 对比 diffusers 路径上的真 torch（§33）

`infer_h3.py`，512x512，124 帧，6 步，`--vae-dtype float16`，各用一张 GPU：

| 阶段 | jittor，冷 | jittor，热 | torch | 热 / torch |
| --- | --- | --- | --- | --- |
| dit | 61.41 | 25.92 | 23.35 | 1.11x |
| text_encoder | 17.25 | 3.77 | 10.18 | 0.37x |
| vae.video | 66.50 | 15.69 | 6.79 | 2.31x |
| vae.audio | 36.90 | 5.64 | 0.28 | 20.1x |
| **generate_seconds** | **253.69** | **92.34** | **75.94** | **1.22x** |

冷运行承担 JIT 编译成本，不是速度数字。服务器路径，832x480 下 8 步：TP1 309.2 s，
TP2 180.2 s。

### 服务器请求（§36–§40）

服务器的视频 VAE 是缺口，而在 `autocast(fp16)` 下它就是 Triton bridge：每个操作数
一次设备排空（部署的核心缺少 `device_ptr_ready`、阻塞式 bounce 拷贝、一个等待
barrier），然后每次启动一次启动后的 `synchronize`。

| 512x512，8 步，经 `/v1/videos` 的 `t2va` | TP1 | TP2 |
| --- | --- | --- |
| §35 之后 | 118.9 s | 130.1 s |
| §38 之后（无逐操作数排空） | 85.7 s | 85.5 s |
| §39 之后（shim 中无逐启动等待） | **78.7 s** | **77.0 s** |

每次启动 16.65 → 2.72 ms（§38）与 2.65 → 0.515 ms（§39）；测试
`tests/backends/cuda/test_device_ptr_ready.py` 与 `TestLaunchFollowsItsProducers`。
§40 表明剩余请求时间都是设备工作；便宜的路子（128 位向量化 1.10x、
`auto_flush_ops`、`--enforce-eager`、autocast 权重转换缓存约 2%）都已逐项测量并
关闭。

### VAE 解码探针（§41–§47）

Autocast VAE 解码，`autocast(fp16)` 下的 fp32 权重，latent `(1, 24, 37, 32, 32)`。
§41–§43 使用服务器侧解码探针（`probe_vae_checksum.py`、
`probe_decode_cprofile.py`）：

- §41 让 cuBLASLt linear 感知 amp（scale 类型 `CUDA_R_32F`）：GEMM 减半，解码不变
  （17.51 → 17.57 s）；
- §42 记忆化逐算子 shim 簿记（`get_install_context`、owner 查找）：17.51 → 12.41 s；
- §43 后端缓存命中不再重走项目树：11.50 → 9.93 s。

从 §44 起，参考解码在两个 runtime 下运行（`probe_decode_base_repeat.py`，重复取
最小值）：

| 步骤 | shim | torch | 比值 |
| --- | --- | --- | --- |
| §44 bridge barrier 命名其操作数（`TestTheLaunchBarrierNamesItsOperands`） | 10.14 s | 6.67 s | 1.52x |
| §45 设置 `JITTOR_FLASH_ATTN_JITTOR_SRC` 后 SDPA 到达融合后端 | 8.30 s | 6.67 s | 1.248x |
| §47 bias 转换到计算 dtype（15.97 s），然后 `fuse_op_limit=16`（`1345192d`） | 8.86 s | 6.64 s | 1.33x |

- **§45 数值。** 融合路径与 torch 的差为 rms 1.543e-03，组合路径为 1.567e-03；
  切换在数值上是中性的。
- **§46 省略基线无效。** `return value` 让 jittor 剪掉 `q`/`k`。用 `consume` 分支
  后，1.61 s 的差距是 1.09 s 的非 attention 设备工作加上 0.53 s 的 attention
  （kernel 0.29，wrapper 0.23）；尺寸扫描显示差距随 tensor 尺寸增长，即它在设备侧。
- **§47 bias 修复带来 2x 成本，因为 fuser 在无界的情况下重新计算。** 在统一半精度
  下，`count_fuse` 构建了多达 361 个算子的 kernel（是算子执行次数的 5.5x）。
  `fuse_op_limit` 只限制半精度组：limit 0 → 15.97 s，16 → 8.87 s，8 → 8.62 s；
  fp32 不受影响（25.94 s 对 torch 的 28.69 s）。该界限低于解码自身的重复噪声
  （3.0e-04）；`tests/codegen/test_fuse_op_limit.py`、`tests/core/test_fuser.py`。

## 部署侧的噪声（§48–§49）

- **§48 症状与陷阱。** ComfyUI/vLLM-Omni 部署在约 70% 的请求中返回均匀随机的
  `uint8` 帧。任何读取 tensor 的探针都会强制求值并修复运行，因此四个「修复」在
  挂上 tracer 时都被评为干净；检查器还会放过全黑、空与降质的片段。有效的临时绷带
  是 `H3_PREP_MODE=varsync`（在量化器输入上调用 `VarHolder::sync(true, false)`）。
  配对的三分支测试把线程作为变量：在生产线程上同步 0/6 失败，在消费线程上同步
  3/6，不同步 4/6。过程中修复了两个 Python 线程下的崩溃点：编译失败处理不再解引用
  已死的 `Op*`（`841cca38`），batch hold 覆盖了它即将编译的算子（`a5b6a29c`）。
  `tests/core/test_nonsink_holder_evaluation.py` 固定了调查指向的不变式（`sync_all`
  只清扫 sink holder，而解码持有的 Var 会在未完成时被读回）；它并不能复现该失败。
- **§49 根因。** `compute_stream()` 返回 `cudaStreamPerThread`，即每个调用线程一个
  不同流，而图及其 buffer 是进程全局的；第二个线程的 kernel 可能在一个线程仍在
  写入其输入时运行。`backend_compute_stream_acquire`/`_release`
  （`src/runtime/backend_streams.cc`）现在在 `run_exec_plan` 末尾把上一个发起者的
  事件交给下一个线程（`a60d12f9`）；单线程进程不付出任何代价。跨线程解码：
  12/12 全 NaN → 0/12（最差 rel 0.0023，同线程对照 0.00166）。去掉绕过后部署：
  4/4 片段落在干净区间内。对 `src/` 与 `backends/` 中每个 `thread_local` 的审计
  未发现第二个实例。

## 门禁与复现

本次工作中看到的门禁失败都在同一提交的干净 worktree 上核对过：
`tests/structure/backends/comm/test_comm_resource_layout.py::test_legacy_runtime_resource_trees_are_absent`
失败仅因为本地未跟踪的 `python/jittor/extern`（全新 checkout 会通过），而
`test_compat_write_entry_points.py` 的 `unclassified` 列表是 gitignore 的
`compat/build/lib/**` 树。

```bash
cd $JITTOR_LAB_ROOT/minimax-h3 && source ./env-jittor.sh
# shim tests must be collected through the installed package
PYTHONPATH=$REPO/python "$VENV/bin/python" -m pytest --pyargs jittor.compat.tests.torch.test_multi_device
GPU=0 PORT=8100 ATTN=FLASH_ATTN ./serve-vllmomni.sh      # shipped single-GPU server
./run_tp2_probe.sh                                       # TP2 loop, retries known startup flakes
"$VENV/bin/python" probe_decode_base_repeat.py 5         # reference VAE decode, repeated minimum
"$VENV/bin/python" probe_decode_sdpa_decompose.py        # section 46
H3_FUSE_DUMP=1 "$VENV/bin/python" probe_loader_race.py threads 4 6   # section 29
```

实验室 venv（`$JITTOR_LAB_ROOT/_state/h3/venv-jittor`）持有 `python/jittor` 的
**副本**加上 `jittor-torch`：仓库中的改动在重新部署前不起作用，而 `src/` 的改动
还需要重建核心并清空 JIT 缓存（JIT key 既不覆盖编译 flag，也不覆盖 codegen 前缀）。

## 未结事项

- 参考 VAE 解码比 torch 慢 1.33x：1.09 s 非 attention 设备工作与 0.53 s attention
  （§46–§47）。需要内核工作，而不是配置。
- 带 mask 的 SDPA 调用从未到达融合后端（每次服务器生成 50 次，§45）；需要一个
  感知 mask 的融合内核。
- 两个*并发* Python 线程下 `FusedOp::update_ops` 的 use-after-free 是与 §49 不同
  的缺陷；`tests/core/test_executor_python_threads.py` 用 `JT_TEST_THREAD_RACE=1`
  复现它。相关台账条目：KI-EXEC-005、KI-EXEC-007。
- `torch.backends.cudnn.version()` 在 shim 下返回 `None`（§45），因此 vLLM-Omni
  的 `cudnn_version >= 90500` 路由在 Blackwell 上会与 torch 不同。
- 服务器在成功生成后于关闭期间中止（`SIGABRT`）（§45）；尚未归因。
- vLLM-Omni 导入顶层 `flash_attn_interface`，而 shim 只提供
  `flash_attn.flash_attn_interface`（§16）；实验室使用了一个别名。
- `Tensor.to(memory_format=...)` 被接受但被忽略（§48）；`lt_linear_cuda` 与可移植
  linear 在结果 dtype 上不一致（§47）；`check_graph=1` 标记出一个孤立的音频 Var
  （§48）。
- ROCm 的驱动以 §26 为 CUDA 修复的同一进程全局方式缓存 `hipGetDevice`；因缺少
  硬件未处理。

## 值得保留的经验

- 读取惰性 tensor 会对其求值：挂上的探针可能修复它所测量的运行，而只认识过去失败
  的检查器会把新的失败判定为通过。
- 停止消费某个操作数的消融分支也会测到剪枝（§46）；相同的输入会折叠图，未被读取
  的结果会被剪掉。
- 在同一进程中交错运行各分支，并引用重复最小值；绝不在共享机器上读 `nvidia-smi`
  利用率。正确的回退可以无限期地掩盖坏掉的快路径（§41）。
