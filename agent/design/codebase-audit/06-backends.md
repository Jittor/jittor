# 后端实现与分布式（extern/）

**核心判断**：每个第三方库的封装都是"一段从官方 example 抄来的调用序列"，而不是一个
有生命周期、有错误契约、有能力声明的适配层。四条贯穿 8 个 CUDA 库、ACL 和两套集合
通信的共性缺陷：(1) 资源以"进程内一个全局句柄加每次调用重建描述符"管理，析构统一
放在静态对象里且用会抛异常的宏检查，导致退出期 terminate、异常路径必泄漏；
(2) 错误检查宏有四套语义（抛 / 只打印一次 / 打印后继续 / 打印后 return），其中三套
把失败变成"输出未定义但流程继续"；(3) 精度策略在 3 份 cuBLAS 拷贝、2 份 cuDNN 卷积
拷贝、MPI/NCCL/HCCL 三份 dtype 表里各自演化，已出现互相矛盾的分支；(4) 分布式层
只有一个全局 communicator、一条默认流、零超时、零失败传播，初始化失败会静默退化
成单卡。ACL 的 13.6k 行里 67 个 OpRunner 的 71 个 executeOp 中有 65 个是同一段 8 行
尾巴的复制。

## 库句柄与资源生命周期
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 静态析构里调用会抛异常的 checkCudaErrors，析构默认 noexcept 故 terminate | `check()` 用 LOGf（`inc/helper_cuda.h:124-132`），LOGf 即 throw（`src/utils/log.h:152-158,163`）；抛点 `cublas/src/cublas_wrapper.cc:28`、`cudnn/src/cudnn_wrapper.cc:43`、`curand/src/curand_wrapper.cc:37`、`cusparse/src/cusparse_wrapper.cc:26`、`nccl/src/nccl_wrapper.cc:139`、`mpi/src/mpi_wrapper.cc:263` | 这正是审计 F5 记录的退出期 "terminate called without an active exception"：进程退出时 CUDA 上下文已拆，Destroy 返回非 0 直接 abort，掩盖真正的错误 | 析构路径用只记录的 peek，句柄销毁改显式 shutdown() | 关键 |
| 局部 RAII 对象的析构也抛，栈展开中抛出即 terminate | `cudnn/inc/cudnn_rnn_descriptor.h:68`、`:99`，二者都是 jit_run 的局部量 | RNN 执行中任一错误抛出，展开时再抛，进程直接死且无栈信息 | RAII 析构一律不得抛 | 关键 |
| 库句柄是单例全局，无 device 维度、从不 SetStream | `cublas_handle`/`cudnn_handle`/`cusparse_handle`/`gen`（各 `*_wrapper.cc:12~16`）；全仓无 cublasSetStream/cudnnSetStream | 句柄绑定在 dlopen 时的当前 device 上；多卡与多流在库层是硬约束 | per-device 惰性表加每次执行前 SetStream | 主要 |
| cuTT plan 缓存永不销毁永不淘汰 | 缓存 `cutt/ops/cutt_transpose_op.cc:66`；`cutt_wrapper.cc:30-32` 的析构只打一行日志 | 每个新的（维度,形状,置换,dsize）组合永久占一份 plan 显存 | 加容量上限与析构清理，与 cuFFT 对齐 | 主要 |
| cuFFT 每次建计划泄漏一个 handle | `cufft/ops/cufft_fft_op.cc:79-80`：cufftCreate 之后 cufftPlanMany 覆盖了句柄，前一个从未 Destroy | 每个新形状泄漏一份 plan 含 workspace 显存 | 删掉 cufftCreate 或改 MakePlanMany | 主要 |
| cuFFT 缓存无上限、字符串键、不含 device | `cufft_wrapper.cc:15` `unordered_map<string,cufftHandle>`；键在 `cufft_fft_op.cc:74` 拼串 | 形状多样的 FFT 把显存耗在 plan 上 | 与 cudnn_conv_plan.h 的 POD 键统一 | 次要 |
| 异常路径必泄漏描述符与 workspace | `cudnn/ops/cudnn_conv_op.cc:122-125` 建 4 个描述符、`:366` 分配 workspace，销毁在 `:380-386`；中间任一抛出即全泄漏。RNN 同理（`cudnn_rnn_op.cc:159-164` vs `:216-224`） | 一次 cuDNN 失败加上层 catch 就变成句柄泄漏累积 | 描述符与 workspace 一律 RAII（`cudnn_conv_plan.h:85` 的 Desc 已示范） | 主要 |
| RNN 的 work_space 未初始化即被读 | `cudnn_rnn_op.cc:179` `void *work_space;` 仅当 size>0 时赋值（`:182-183`），但 `:226` 无条件 `if (work_space)` 判断并 free | size==0 时读未初始化指针并交给分配器释放 | 初始化为 nullptr | 主要 |
| infer_shape 里创建 cuDNN 描述符且全部泄漏 | `cudnn_rnn_op.cc:91-97` 为 seq_length 个 xDesc 调 Create，函数结束前无 Destroy | 训练 RNN 每次形状推断泄漏 seq_length 个描述符；形状推断本不该碰设备 API | reserve size 改由 op 内缓存计算 | 主要 |
| ACL workspace 是只增不减的私有全局，失败时留下悬垂指针 | `acl_jittor.cc:132-156`：aclrtFree（`:151`）→ 更新 size（`:152`）→ aclrtMalloc 失败则 return（`:154`），此时指针指向已释放内存 | 一次 OOM 后所有后续 aclnn 算子写入已释放显存；且该缓冲不经 Jittor 分配器，内存统计看不见 | 失败时置 nullptr 并抛；workspace 改走 exe.temp_allocator | 关键 |

## 每次调用的重复工作
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| cuDNN 卷积走 backend plan 快路径时仍每次建 4 个描述符再销毁 | `cudnn_conv_op.cc:122-125` 建、`:143-222` 全套 Set、快路径命中后 `:243-246` 再销毁 | plan 缓存把执行降到 12 µs，但描述符构造的 3 µs 与若干 Set 仍是每调用成本 | ConvPlanRequest 直接从 Var 形状构造，快路径不建 legacy 描述符 | 主要 |
| conv3d 完全没迁移，且与 conv2d 共用同一个全局算法缓存、键格式不同且缺 dtype | `cudnn_conv3d_op.cc:82` 自己的定义被注释掉，`:89` 引用 `cudnn_conv_op.cc:104` 的 fwd_algo_cache；conv3d 键（`:219-222`）无 dtype 无 workspace_ratio | fp32 与 fp16 的同形状 conv3d 共享算法选择；max_workspace_ratio 改变后 3D 缓存不失效；还共享 100 条容量预算 | conv3d 迁到 plan 缓存，之前至少给键加 dtype 与命名空间 | 主要 |
| MKL 卷积每次重建 engine、stream、primitive desc 和 reorder 缓冲 | `mkl/ops/mkl_conv_op.cc:119-120`、`:148-156`、`:162,170,175` | CPU 卷积每调用开销全在建原语；reorder 缓冲的 malloc/free 在内存统计之外 | 按形状与参数缓存 primitive 与 pd | 主要 |
| cuSPARSE 每次调用 cudaMalloc/cudaFree | `cusparse/ops/cusparse_spmmcsr_op.cc:64,68` | cudaFree 是同步调用，每次 SpMM 做一次全设备同步 | 改用 exe.temp_allocator（同文件其它算子已用） | 主要 |
| 算子执行期复用全局 JIT key 缓冲 | `cufft_fft_op.cc:72-73`、`cutt_transpose_op.cc:102-103`、`cudnn_conv_op.cc:276-277`、`cudnn_conv3d_op.cc:217-218` 都在 jit_run 里 get_jk 后 clear | 同一个全局缓冲既服务执行器又服务算子内缓存键，不可重入不可多线程，每次调用做字符串拼接 | 缓存键改 POD 哈希 | 主要 |
| ACL 每个算子每次调用重建 aclTensor 描述符加两次 vector 堆分配 | `aclops/base_op_acl.cc:49-68`、`:84-104`、`:70-82`；`utils.cc:110` 每次算 strides | 39 个算子无一例外，NPU 侧的 CPU 开销与 cuDNN legacy 卷积同类 | 描述符按形状与 dtype 缓存，指针用 aclSetTensorAddr 更新 | 主要 |
| cuTT 计划未命中时做全设备同步 | `cutt_transpose_op.cc:115` cudaDeviceSynchronize | 首次遇到的每个转置形状清空流水 | 删除或降为流同步 | 次要 |

## 错误处理：四套语义
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| peek 用进程级布尔锁，只报告第一次异步 CUDA 错误 | `inc/helper_cuda.h:107,115-116`；`peek_logged` 定义在 `src/utils/log.cc:24`，从不复位 | 第二次及以后的异步错误（含退出期报出的真正 kernel 失败）全部静默；`fetch_op.cc:57-59`、`array_op.cc:38-40`、`cuda_dual_allocator.h:101` 都走这条路 | 去掉全局闩，改限频或按 call site 去重 | 主要 |
| CUFFT_CALL 只 fprintf 然后继续 | `cufft/inc/cufft_utils.h:71-85`；CUDA_RT_CALL 同（`:53-68`） | cufftPlanMany 失败后仍把无效句柄写进缓存（`cufft_fft_op.cc:85`）并继续 Exec，结果是未定义输出而非报错 | 统一用 checkCudaErrors | 主要 |
| cudaGetLastError() 被显式用来清错误 | `cutt_transpose_op.cc:77` | 前面任何 kernel 的异步错误在这里被吞掉，之后归因到无关算子 | 删除；确需清理时记录被丢弃的错误码 | 主要 |
| ACL 的 checkRet 打完日志就返回，调用方无条件继续 | `aclops/base_op_acl.cc:115-124`（`CHECK_RET(..., return)` 在 void 函数里等于什么都没做）；消费点 `binary_op_acl.cc:125-132`：checkRet 后立刻用可能是垃圾的 workspaceSize 与 executor 调 executeFunc | GetWorkspaceSize 失败到用未初始化 executor 执行到输出未定义，且不抛异常故 fallback 也接不住。65 处同一模式 | checkRet 改为抛；executeOp 骨架收进基类 | 关键 |
| run() 的非 group 分支不检查 find() 是否命中 | `aclops/base_op_acl.cc:142-149` 直接 executeOp(it)，而 group 分支（`:131-136`）检查了 | 名字拼错或算子未注册则解引用 end() 迭代器，UB | 两条分支合并统一检查 | 关键 |
| CreateAclTensor 恒返回 0，所有错误检查都是死代码 | `aclops/utils.cc:124,151` 恒 return 0；检查点 `base_op_acl.cc:66,102` | aclCreateTensor 返回 nullptr 从不被发现；且 setupInputDesc 提前 return 时 inputTensors 短于 in_.size()，cleanupDesc（`:74-77`）会越界读 | 返回真实状态 | 主要 |
| HCCL 算子的宏在错误时 return，算子静默不写输出 | `hccl/inc/hccl_wrapper.h:33-47`（LOGe 加 return），用在 `hccl_all_reduce_op.cc:53-57` 的 jit_run 里 | 一次集合通信失败到输出保持未初始化到训练继续跑出 NaN，其它 rank 挂死 | 集合通信失败必须抛并让所有 rank 快速失败 | 关键 |
| sync_run 这个默认开启的调试开关在 ACL 上是空实现 | 定义 `src/misc/cuda_flags.cc:23`（默认 1，doc 写 Enable per-op-sync）；实现 `aclops/base_op_acl.cc:106-113` 整个函数体被注释掉 | 文档承诺的逐算子同步定位手段不存在 | 实现或删掉 flag | 主要 |

## 精度与数值语义
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| cuBLAS 精度选择逻辑有 3 份拷贝且已彼此矛盾 | `cublas_matmul_op.cc:110-130`、`cublas_batched_matmul_op.cc:138-158`、`cublas_acc_matmul_op.cc:99-117`；前两者 fp16 用 `use_tensorcore ? COMPUTE_16F : COMPUTE_32F`，第三者（`:109-110`）无条件 COMPUTE_16F | 同一个 fp16 矩阵乘的累加精度取决于图里选中哪个算子（与 PyTorch 默认的 fp32 累加不一致）。cublas_acc_matmul 全仓只有一个测试调用，是近乎死代码的第三份拷贝 | 抽出统一的 compute type 选择；删除或合并第三份 | 主要 |
| algo 的三目判断写反 | `cublas_matmul_op.cc:122,125` 与 `cublas_batched_matmul_op.cc:150,153`：`use_tensorcore ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP` | 开 tensorcore 反而选 DEFAULT，关掉反而选 TENSOR_OP；与同文件 CUDA<11 分支（`:137-143`）逻辑也相反 | 修正并加单测覆盖 4 种组合 | 主要 |
| use_tensorcore 是 0–3 多义编码，与两个 tf32 flag 交叉决定结果 | `src/init.cc:23-25`；消费点 `cublas_matmul_op.cc:113-126`、`cudnn_conv_op.cc:193-200`；`>=3` 表示 fp32 输入降到 fp16 | 一个整数同时表示是否用 tensorcore 和降到哪种精度，且对 fp32 与 fp16 含义不同 | 收敛为 float32_matmul_precision 三档，matmul 与卷积共用 | 主要 |
| conv3d 的 benchmark 判据与 conv2d 不同 | conv2d `cudnn_conv_op.cc:274` `cudnn_benchmark != 0`；conv3d `cudnn_conv3d_op.cc:215` `(<0 或 >0) 且 !has_fp16_or_bf16` | 同一设置在 2D/3D 上语义不同；3D 的半精度卷积永不测量 | 统一判据 | 次要 |
| cuDNN RNN 描述符硬编码 CUDNN_DATA_FLOAT | `cudnn_rnn_descriptor.h:94`、`:105`、`:135`；而张量描述符用 `getDataType<Tx>()`（`cudnn_rnn_op.cc:162`） | fp16/bf16 RNN 的张量与 RNN 描述符 dtype 不一致；权重空间按 fp32 计算 | 按实际 dtype 参数化 | 主要 |
| RNN 的 dropout 状态每次调用重建并用全局 seed 重新初始化 | `cudnn_rnn_descriptor.h:47-66` 在构造里 SetDropoutDescriptor(get_seed())；该对象是 jit_run 的局部量（`cudnn_rnn_op.cc:177`） | 每步 RNN 的 dropout 掩码序列相同，既不正确也很慢（SetDropoutDescriptor 是毫秒级） | dropout 状态按 (dropout,seed) 缓存跨调用复用 | 主要 |
| cuSPARSE 计算类型硬编码 fp32，alpha/beta 也是 float | `cusparse_spmmcoo_op.cc:49-50,61`、`cusparse_spmmcsr_op.cc:50-51,62,67`，尽管上面已算出 dtype | fp64 稀疏乘法的 alpha/beta 与计算类型都错 | 用算出来的 dtype | 主要 |
| COO 变体不申请 buffer 就调 cusparseSpMM | `cusparse_spmmcoo_op.cc:51-62`：bufferSize 查询被注释掉，直接传 NULL | 需要外部 buffer 的算法路径上是 UB；与 CSR 变体行为不一致 | 恢复 bufferSize 查询 | 主要 |
| curand 越界写一个元素，且只支持 fp32/fp64 | `curand/ops/curand_random_op.cc:42-48`：注释明说 curand 不支持偶数所以加 1，靠"分配器会给奇数块"兜底；`@define(TT,…)` 对非 fp32 一律展开成 Double | 依赖分配器内部实现的越界写；fp16 随机数会得到类型不匹配的编译错误而非清晰报错 | 用临时 buffer 处理奇数长度；显式拒绝不支持的 dtype | 主要 |
| MPI 的 fp16 归约在 x86 与 ARM 上数值不同，且 x86 路径无 CPUID 检查 | `mpi/src/mpi_wrapper.cc:113-163`：x86 用 `_mm256_cvtph_ps`（需 F16C+AVX，运行期无检测），ARM 手写路径把非规格化数刷成 0（`:87-89,97-99`），x86 路径不刷 | 同一 all-reduce 在不同架构给出不同结果；老 x86 机器直接 SIGILL | 统一标量参考实现加可选 SIMD 与运行期检测 | 主要 |

## 分布式：通信器、同步点与失败模式
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 全局唯一 communicator，无 process group | `nccl/inc/nccl_wrapper.h:45`；`hccl/src/hccl_wrapper.cc:24` 同 | DDP 与张量并行无法共存；无法做子组 all-reduce | 引入 ProcessGroup 对象 | 主要 |
| 所有集合通信硬编码默认流（最后一个参数 0） | `nccl_all_reduce_op.cc:57`、`nccl_broadcast_op.cc:56`、`nccl_reduce_op.cc:56`、`nccl_all_gather_op.cc:62`、`nccl_reduce_scatter_op.cc:61` | 通信与计算不可能重叠，DDP 的梯度 all-reduce 完全串在反向后面；也无 GroupStart/End，梯度不能分桶 | 通信流加 event 依赖，桶化 | 主要 |
| HCCL 每次集合通信前后各做两次全设备/流同步 | `hccl_all_reduce_op.cc:53-57`、`hccl_all_gather_op.cc:61-65`、`hccl_reduce_op.cc:54-58`、`hccl_broadcast_op.cc:50-58` | 每次通信 4 次同步，NPU 多卡训练流水被彻底打断 | 删掉，依赖流序 | 关键 |
| NCCL 的 env/file rendezvous 无超时无失败路径 | `nccl/src/nccl_wrapper.cc:89-104`：非 0 rank 轮询 6000 次后不做任何检查直接往下走，rf 为 null 时拿未初始化的 id 去 InitRank；对照 HCCL 同段（`hccl_wrapper.cc:70-85`）有超时判断 | 文件没写出来则用垃圾 unique id 建通信器，挂死或诡异错误 | 与 HCCL 版本对齐并抽成共享实现 | 主要 |
| NCCL 在静态构造器里做 cudaSetDevice 加阻塞式建通信器 | `nccl_wrapper.cc:57,78,104`（`static nccl_initer nccl_init;` 在 `:144`） | HCCL 注释明说这样会挂死 import jittor（`hccl_wrapper.cc:97-100`）故改成显式 init，NCCL 侧没跟进 | NCCL 也改显式初始化 | 主要 |
| 无通信超时、watchdog、ncclCommAbort | 全仓无 ncclCommAbort、无 timeout 设置 | 一个 rank 崩溃其余 rank 永久挂起且无诊断 | 超时加 abort 加明确错误 | 主要 |
| 启动器按 rank 顺序 wait，不做失败传播 | `distributed/launch.py:102-107` 顺序 p.wait()，只有 finally 里才 kill | rank 3 崩溃、rank 0 挂死则启动器永远等 rank 0 | 用 wait(timeout) 轮询，任一非零退出立刻 kill 全部 | 主要 |
| 分布式初始化失败静默退化为单卡 | `compile_extern.py:972-973` `except Exception: LOG.w("HCCL setup failed…")`；`fsdp2/common.py:20-31` 的 world_size/rank 用 `try/except: return 1/0` | 一个 4 卡任务变成 4 个独立单卡任务各自训练，看起来能跑但完全错 | 分布式一旦被请求，初始化失败必须硬失败 | 关键 |
| MPI 只认 OpenMPI 的环境变量 | `mpi/src/mpi_wrapper.cc:216` `getenv("OMPI_COMM_WORLD_SIZE")` | MPICH / Intel MPI / srun 启动时静默单卡 | 同时识别 PMI_/SLURM_ 或要求显式声明 | 主要 |
| MPI 的 int64 数据类型映射成 MPI_DOUBLE_INT | `mpi_all_reduce_op.cc:85`、`mpi_broadcast_op.cc:70`、`mpi_reduce_op.cc:84` | MPI_DOUBLE_INT 是 MAXLOC 用的二元组（12/16 字节）不是 int64，按 count=num 传会读越界并给出垃圾结果 | 改 MPI_INT64_T；三份 dtype 表合并 | 关键 |
| var_broadcast 忽略 root 参数 | `mpi_wrapper.cc:271-280`：签名收 root，第 278 行硬编码 0 | 任何非 0 root 的广播行为错误且无提示 | 修正 | 主要 |
| 同一进程内 rank/world_size 有三份来源 | C++ 全局 `mpi_world_rank`（`mpi_wrapper.cc:165-168`）、Python `compile_extern.rank/world_size`（`:934-944`，由三条分支覆写）、`jt.rank` | 任何一条分支漏改就出现 C++ 认为 rank 0、Python 认为 rank 2 的静默不一致 | 单一来源 | 主要 |
| nccl_reduce 在非 root rank 上把输出清零 | `nccl_reduce_op.cc:57-58`；CPU 版同 `mpi_reduce_op.cc:93-94` | 掩盖"非 root 输出无意义"的语义，且每个 rank 都分配了全尺寸输出 | 非 root 不分配输出 | 次要 |
| mpi_broadcast 在 infer_shape 里按 rank 决定是否 share_with | `mpi_broadcast_op.cc:47-51` | 图的别名结构因 rank 而异，跨 rank 图不再同构 | 别名决策移出形状推断 | 次要 |
| FSDP2 分片策略由硬编码经验常数决定 | `fsdp2/common.py:114-123`：`world_size<=2 or total_numel<=1_000_000` | 在 3 卡或 1.1M 参数处行为突变，常数来自一次特定实验 | 变成可配置策略 | 次要 |
| 每个 rank 用独立 JIT 缓存目录 | `distributed/launch.py:90` `env["cache_name"] = f"{backend}{rank}"` | N 卡任务把整套 kernel 编译 N 次，磁盘与首步时间乘 N | 共享缓存加文件锁，或 rank 0 预热 | 主要 |

## ACL 三件套：样板量化
一个 ACL 算子由三部分组成：`*_op_acl.cc` 的 OpRunner、`*_op.py` 的 jt.Function、
以及 `_code.py` 在**每次调用时**用 Python f-string 拼出的一段 C++ 源码（`aclops/_code.py:47-59`）。

| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 67 个 OpRunner 类 / 71 个 executeOp，其中 65 个是同一段 8 行尾巴的复制 | `aclops/*.h` 中 67 处 `class *OpRunner`（分布在 32 个头文件）；`aclops/*.cc` 中 71 处 `::executeOp`；65 处紧接同样的 checkRet / mallocWorkSpace / executeFunc / CHECK_RET / syncRun | 每个副本独立地忽略 checkRet 失败；改一处错误处理要改 65 处 | 尾巴收进 `BaseOpRunner::launch()`，executeOp 只负责绑定参数 | 主要 |
| AclOpFunctions 是装了约 40 个 std::function 成员的胖结构，每条表项只用其中一个 | `acl_jittor.h:33-345`（成员 `:36-127`，35 个构造重载 `:146-344`）；表 `:347` | 每条表项约 1.3 KB 且 39/40 是空的；新增一种签名要改结构体加构造重载 | 类型擦除的单一 launcher | 主要 |
| 这张表 static 定义在头文件里 | `acl_jittor.h:347` `static std::unordered_map<...> aclOpFuncMap` | 40 个 .cc 各有一份完整拷贝，各自在静态初始化期构造上百个胖结构；跨 TU 修改互不可见 | 改 extern 加一个 .cc 定义，最好换注册宏 | 主要 |
| op_idx_map 是 60 行手工维护的死表 | `aclops/utils.cc:42-104` 定义、`utils.h:15` 声明，全仓无读取点 | 纯维护负担，也说明缺少算子清单的真实机制 | 删除，真正需要的是 supported_ops() | 次要 |
| 算子属性被拼进 C++ 源码，运行期值变成编译 key | `aclops/_code.py:47-59` 生成源码；`pool_op.py:96-106` 把 kernel/stride/padding/dilation/ceil_mode 全部 f-string 插入；共 30+ 处 | 每个不同参数组合编译一个独立 .so；参数来自运行期（如自适应池化）时无界膨胀 | 属性走 jt.code 的 data 通道（`_code.py:60` 已有） | 主要 |
| 若干 ACL 算子静默把输入升到 fp32 | `silu_op.py:24`、`softmax_op.py:62`、`sigmoid_op.py:62`、`relu_op.py:19,41`、`norms_op.py:86`、`where_op.py:119-121` | bf16/fp16 模型在这些点悄悄变 fp32 并沿图传播，与 PyTorch 语义不符且拖慢 | 明确支持的 dtype，不支持时报错而非升精度 | 主要 |
| fallback 前的不变量检查作用在错误的变量上 | `acl_op_exec.cc:274` `for (auto in : op->inputs()) ASSERT(in->mem_ptr);` 中的 op 是函数形参（FusedOp），真正的当前算子在 `:278` 才用 `auto op = queue.front()` 遮蔽声明 | "即将执行的算子输入都已分配"这条不变量从未被检查 | 调整声明顺序，避免同名遮蔽 | 主要 |
| 用名字前缀 "cu" 从全局算子表里删算子 | `acl_op_exec.cc:621-632` | 判据是字符串前缀而非能力声明 | 显式的后端算子注册表 | 次要 |
| aclnn.h 无 include guard，含 120 个第三方头 | `acl/aclnn/aclnn.h:1` 首行是 include 而非 pragma once | 重复包含依赖第三方头自己的 guard，编译期开销大 | 加 pragma once | 次要 |

## 其余后端
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| MKL 用 oneDNN 2.2.0（2021）且用 v3 已移除的 API | 版本 `compile_extern.py:51,62,66,69`；API `mkl_conv_op.cc:153-156` 的 `convolution_forward::desc` | 与拒绝 cuDNN 9 同类：CPU 后端钉死在 4 年前的库上 | 迁到 v3 API 后放开版本 | 主要 |
| MKL matmul 只支持 fp32 | `mkl_matmul_op.cc:28` `ASSERT(a->dtype().dsize()==4)` | CPU 的 fp64/fp16/bf16 矩阵乘掉回通用元算子，而 CUDA 侧全支持，且无处声明这种能力差异 | 补齐或在能力表里声明 | 主要 |
| MKL 卷积一律用 forward_inference | `mkl_conv_op.cc:153` | 训练路径用推理 prop_kind，不保证与 backward 配套 | 按 is_train 选择 | 次要 |
| Corex 的探测函数带副作用且路径写死 | `corex/corex_compiler.py:86` isdir("/usr/local/corex")、`:68` 硬编码 home、`:88` 在 check() 里调 install() 改全局编译器配置 | 检查是否可用会改全局状态；非标准安装路径不被支持 | check() 只读，路径可配置 | 次要 |
| Corex 的源码改写函数沿用 ACL 的名字 | `corex/corex_compiler.py:31` `string process_acl(...)` | 两个后端的文本改写实现互相拷贝，命名都没改 | 随后端注册表一起删除 | 次要 |
| ROCm 只有 374 行，算子全部来自被改写的 CUDA 源码 | `rocm/rocm_wrapper.h`（150 行，只补了 rocprim 的 argmax/argmin）、`rocm_compiler.py`（154 行） | ROCm 正确性完全取决于文本替换是否覆盖每一个 CUDA 调用；`#ifndef IS_ROCM` 散布在 `cublas_wrapper.h:28`、`cudnn_wrapper.h:42`、`cudnn_conv_op.cc:189,224` 等处是唯一契约 | ROCm 自己实现并注册，不再吃改写产物 | 主要 |

## 优先级
- **先修会静默出错的**：MPI 的 MPI_DOUBLE_INT、ACL 的 checkRet 空实现与 find() 未检查、
  mallocWorkSpace 失败后的悬垂指针、分布式初始化失败退化单卡。四条都属于"不报错但结果错"。
- **再修决定分布式能否用的**：HCCL 每次通信 4 次全设备同步、集合通信全在默认流、
  无超时无失败传播。
- **再做结构性收敛**：cuBLAS 三份精度逻辑合一与 use_tensorcore 语义收敛、
  ACL 65 份 executeOp 尾巴收进基类与 AclOpFunctions 去胖结构、conv3d 迁 plan 缓存。
- **析构不得抛**是一条可一次性扫干净的规则，成本极低，能立刻消除退出期 terminate 噪音。
