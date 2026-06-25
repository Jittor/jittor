I'll integrate these 28 deep-audit JSONs into a comprehensive second-round work list. Let me analyze the data and produce the Markdown.

# 第二轮(更深)工作清单 · Jittor 双后端深审汇总

> 覆盖 28 个子系统深度审计，合计 **300+** 条独立发现。本清单聚焦**正确性 / 内存安全 / 跨后端对偶**，按工作项编号去重合并。
> 🔁 = **跨后端对偶**（N 卡 ↔ 华为昇腾 都要改）；⚠️ = 数据静默损坏 / 段错误风险。

---

## 🔥 深审 Top 20

| # | title | maps_to | severity | location | both_backends | work_item |
|---|-------|---------|----------|----------|:---:|-----------|
| 1 | RopeACL.execute 返回原始输入而非旋转结果（整算子变 no-op）| RopeACL.execute | **critical** | rope_op.py:80-81 | 🔁 风险 | ALL_TODO-001 |
| 2 | reduce 乘法梯度 `dout*out/v` 含零元素时 0/0→NaN | reduce-multiply-grad-nan | **critical** | reduce_op.cc:344-347 | — | ALL_TODO#17 |
| 3 | argmax/argmin CUDA CUB `forward()` 夺取输出所有权，破坏反传 | arg-reduce-cuda-forward-ownership | **critical** | arg_reduce_op.cc:74-82 | — | ALL_TODO#17 |
| 4 | RNN backward 描述符 dtype 用 Ty 而非 Tx + dx/dhx/dcx 硬编码 float32 | cudnn_rnn_backward dtype | **critical** | cudnn_rnn_backward_x_op.cc:52-56,112-113 | 🔁 | ALL_TODO_CUDNN_002/003/014 |
| 5 | cuSPARSE SpMM 硬编码 CUDA_R_32F，无视 fp16 输入 | spmm-dtype-hardcode | **critical** | cusparse_spmmcsr/coo_op.cc:50-68 | 🔁 (CSR↔COO) | #2/#12 |
| 6 | HCCL all_gather 反传完全缺失（return nullptr）| DDP all_gather backward | **critical** | hccl_all_gather_op.cc:38-41 | 🔁 (NCCL 同样) | #21 |
| 7 | HCCL reduce 非 root rank 输出未清零→静默数值损坏 | DDP reduce non-root | **critical** | hccl_reduce_op.cc:35-59 | 🔁 (NCCL 已清零) | #21 |
| 8 | swap_to_disk fwrite 成功判定反置 `if(res==1)`→文件句柄损坏+丢数据 | SwapIO | **critical** | swap.cc:52 | 共用代码 | BUG_SWAP_FWRITE_LOGIC |
| 9 | Allocation 移动构造未清空源字段→double-free / UAF | allocator move | **critical** | allocator.h:39-41 | 全后端 | ALL_TODO#16 |
| 10 | CudaDualAllocator free_ids 无锁并发→双重分配/越界 | dual-alloc race | **critical** | cuda_dual_allocator.h:34-64 | NV-only(查 ACL) | ALL_TODO#16 |
| 11 | topk()/sort() 错误解包 `idx,_=_argsort()`（实际返回单 Var）→立即崩溃 | topk-sort-unpack | **critical** | torch_compat.py:623,634 | 🔁 | torch_compat_1 |
| 12 | torch.load `_from_portable` 丢失 dtype，全部回落 float32 | load-dtype-loss | **critical** | torch_compat.py:1543-1551 | 🔁 | torch_compat_3 |
| 13 | Function._grad() ret 与 input_mask 索引错配→IndexError/错梯度 | gradient_computation | **critical** | __init__.py:1979-1980 | 🔁 | TODO_GRADIENT_INDEXING |
| 14 | Module.npu() 误设 `use_cuda=1` 而非昇腾标志 | device_configuration | **critical** | __init__.py:1411-1413 | 🔁 | TODO_NPU_DEVICE_FLAG |
| 15 | Dataset MPI 散播末批不均时索引重叠→多卡训练数据重复 | data-parallel scatter | **critical** | dataset.py:509-517 | 🔁 (NCCL+HCCL 共用) | #8 |
| 16 | TernaryOp infer_shape 用 min/max 误拒合法广播形状 | broadcast validation | **critical** | ternary_op.cc:64-71 | — | ALL_TODO#002 |
| 17 | BroadcastTo VLA `int64 zz[zdim]` 栈溢出（高维张量）| stack overflow | **critical** | broadcast_to_op.cc:131-136 | — | ALL_TODO#008 |
| 18 | ReindexOp 用户索引表达式未校验→OOB 内存访问 | unbounded index expr | **critical** | reindex_op.cc:134-137 | — | ALL_TODO#012 |
| 19 | CUDA abs 用整型 `::abs` 截断浮点→数据损坏 | cuda-abs-int | **critical** | common_op_type.cc:19 | 🔁 (CPU:86 正确) | #17 |
| 20 | FusedOp 图边 hex2/hex1 编码 >256 op / >16 维静默截断→拓扑错乱 | operand-encoding-overflow | **critical** | fused_op.cc:178-186 | 🔁 | FOP-001/FOP-002 |

---

## 📑 按 ALL_TODO 编号分节（去重合并）

### #17 — 修复 Jittor 核心算子正确性 Bug（最大集合，跨多文件）

> 该编号在多个子系统中复用，下按文件域归并。

#### 17.A 二元算子梯度 / dtype（binary_op.cc）
- **max/min 在相等点梯度错误** 🔁语义：`x==y==z` 时应返回 `(dout/2, dout/2)`，实返回 `(0,dout)`/`(dout,0)`。`binary_op.cc:524-530`。同 bug 在 reduce_op.cc:354-359、reindex_reduce_op.cc:55-60（仅取首个 tie，应均分/求和）。→ 合并为「**相等点/并列 max-min 梯度均分**」统一修复。
- **MOD y-梯度缺 dout 乘子**：`make_unary(a,ns_negative)` 应为 `-dout*floor(x/y)`。`binary_op.cc:516-522`。
- **POW x-梯度硬编码 int32(1)**：`y-1` 常量应与 y 同 dtype。`binary_op.cc:537-539`。
- **dirty_clone_broadcast 在反传期复制广播节点改图结构**（DAG 别名风险）。`binary_op.cc:476-489`。

#### 17.B 一元 / cast / 类型系统（unary_op.cc + common_op_type.cc）
- ⚠️🔁 **CUDA abs 用整型 ::abs**（Top 20 #19）：`common_op_type.cc:19` 需 `::fabsf/::fabs`；CPU:86 正确。
- ⚠️🔁 **sigmoid 硬编码 1.0f**：float64 精度丢失；CUDA 用 `::expf`、CPU 用 `std::exp` 行为分歧。`:41`(CUDA)/`:108`(CPU)，并见 fp16_op_type.cc:64,103。
- ⚠️🔁 **CUDA erfinv 硬编码 ::erfinvf**：float64 截断；CPU `_erfinv` 正确。`:43` vs `:110`。前向/反向需精度对齐。
- **cast→bool 反传语义违规**：`unary_op.cc:892-893` 把 bool 梯度 cast 回 float，破坏恒等梯度；非 float cast 静默丢梯度（应显式报错）。
- **log 梯度 dtype 不匹配**（输入曾被 upcast）`unary_op.cc:901-902`。
- **float16/bf16 sigmoid 截断阈值错**（非 float32 默认 300）`:41,108`。
- **round/floor/ceil_int 对 float16 精度丢失**（应先升 float32）`:26-28/93-95`。
- **mod 宏变量名 `$1` vs `@Tx` 不一致** `:53/115`。
- **erf/erfinv/sqrt 梯度浮点常量精度丢失**；tan/asin/acos 梯度奇点/域未校验（NaN 未捕获）`unary_op.cc:917-937,991-1001`。
- **bool 输出 `((bool)…)` 双重 cast** `common_op_type.cc:161`。

#### 17.C 规约 / arg 规约（reduce_op.cc + arg_reduce_op.cc + nano_string.h）
- ⚠️ **乘法规约梯度 NaN**（Top 20 #2）`reduce_op.cc:344-347`。
- ⚠️ **argmax CUDA forward() 夺所有权**（Top 20 #3）`arg_reduce_op.cc:74-82`。
- **AMP 寄存器 bit 32 未定义**（nano_string.h 仅到 bit 16），f16 升精无条件触发 `reduce_op.cc:260,296`（命名 `reduce16_intermediate_not_use_32` 与逻辑反向，见 #6/低级）。
- 🔁 **argmax/argmin 输出 int32，应 int64**（CUB 可能返 int64，CPU 回落 int32→跨后端不一致）`arg_reduce_op.cc:88`。
- **mean 未标 float-promoting**：int 输入→int 输出，应为 float `nano_string.h:267-277` + 梯度 dtype 失配 `reduce_op.cc:349-352`。
- **bool 规约 dtype 构造器 1/2 不对称**（int32 vs bool），与 torch all/any→bool 冲突 `reduce_op.cc:287-291,312`。
- **infer_shape 原地改 keepdims_mask**（违反只读契约）`:329-333`；**空输入 count/rcount 除零** `:384-385`；mean rcount 标量对部分规约错 `:141`。

#### 17.D 三元 / where / broadcast / reindex（ternary/where/broadcast_to/reindex*）
- **TernaryOp infer_shape min/max 误拒广播**（Top 20 #16）+ 条件 dtype 未校验布尔语义 `ternary_op.cc:64-93,85-93`。
- ⚠️ **WhereOp CUDA signed→uint 转换 UB + atomicInc 2^30 回绕无边界检查**（缓冲溢出）`where_op.cc:130,167,87,133,185`；CPU int64 vs CUDA int32 计数不一致 `:49-54,253,263`；`-cond->num` 无符号翻转 `:49-54`；prefix_sum 硬编码 32-lane warp `:97-109`。
- ⚠️ **BroadcastTo VLA 栈溢出**（Top 20 #17）+ signed/unsigned 下溢 `:138-160` + bcast_mask 非幂等原地变更 `:150-151` + grad 用未初始化 mask `:112-116`。
- ⚠️ **ReindexOp 用户索引表达式 OOB**（Top 20 #18）+ move 语义误用 `:35-70` + grad 访问未初始化 shape `:76-79`；**ReindexReduce 缺 init 宏 / max-min 梯度只取一个** `:115-120,55-60`。

#### 17.E getitem / setitem（getitem_op / setitem_op + var_slices/nano_vector）
> 编号体系 ALL_TODO#1001-1015，独立列出。
- ⚠️🔁 **#1001 atomicAdd 缺 half/bf16**（fp16 scatter-add 编译失败）`setitem_op.cc:363`。
- 🔁 **#1002 负步长反传 negtive_set_none mask 破坏 stop** `var_slices.h:50-59`。
- ⚠️🔁 **#1003 first_oid_of_var/var_dim 未初始化→越界** `getitem_op.cc:77-78,399`。
- 🔁 **#1004 负整数索引修正未写回** `:140-146`。
- ⚠️ **#1005 ODIM==1 时 dstride 未初始化** `setitem_op.cc:324-325`。
- **#1006 bmask off-by-one** `:104-107`；**#1007 getitem grad 无条件 ns_add 缺重叠分析** `:423-427`；**#1008 setitem 除法梯度操作数反置** `:176-178`；**#1009 FOV/VD 宏无 guard** ；**#1010 Slice.fill 负步长断言不足** ；**#1011 负步长形状有符号当无符号** `:169`；**#1012 rtnum=0 二次循环除法未定义** ；**#1014 broadcast mask 索引错** ；**#1013/#1015 atomicAdd 截断 / 字符串切片无文档**。

#### 17.F 核心算子杂项（array/transpose/code/numpy_code，ALL_TODO_001-016）
- ⚠️🔁 **_001 axes 位移整型溢出**（`1<<i` 无边界）`transpose_op.cc:64`、`fuse_transpose_op.cc:61`；ACL transpose_op_acl.cc:44 同根因（aclCreateIntArray 不校验）。
- **_002 check_vary_shape XOR 逻辑误拒合法动态形状** `code_op.cc:25`。
- ⚠️ **_003 NumpyCodeOp grad backward[v_index] 无边界** `:120`；**_012 void*→Var* 不安全转换** `:132,136`。
- 🔁 **_005 transpose axes 越界访问 x->shape**（mask 校验集合而非个体）`transpose_op.cc:69`；ACL 同样不校验。
- **_004 ACL keepdims 未用（Transpose/Reduce 语义混淆）** transpose_op_acl.cc:45；**_006 ArrayOp memcpy 无 null 检查** `:73,80`；**_007 拒绝 0-d 标量** `:57`；**_008/_009 JIT key dec3/hex1 计数/维度溢出** ；**_010 FuseTranspose OpType 依赖可变上下文** 。

#### 17.G 执行器（executor.cc，ALL_TODO#1-16，独立编号）
- ⚠️🔁 **#6/#8/#15 GPU 迁移冗余 + 非原子 use_cuda + 缺前置同步**（数据损坏/泄漏）`:627-632,601-610,628`。
- **#1/#3 sync_ptr/weak_sync 迭代器越界与排序假设** `:178-198`；**#2/#14 custom_data 位布局复用未清零** `:84-86,268-274,535-537`；**#4/#9/#13 load_fused_op 边索引/null input** `:55-89,72-88,79-82`；**#5 union-find 无环检测** `:283-292`；**#7/#10 liveness 释放顺序与提前 free** `:654-679`；**#11/#12/#16 sharegraph/range/var_fused 边界** 。

#### 17.H Fused-op 编译器（fused_op.cc + op_compiler.cc，FOP-001~014）
- 🔁 **FOP-001/002/010/014 hex2/hex1 编码溢出**（Top 20 #20）：>256 op、>16 维静默截断，executor.cc:86 同源。jit_key.h:153-155 加边界检查。
- **FOP-003 custom_data 位运算溢出** ；**FOP-004 loop_options 悬垂指针** ；**FOP-005 context 未初始化** ；**FOP-006 全局 use_cuda 竞态** ；**FOP-012 jit_key 字符串 UAF** ；**FOP-007/008/009/011/013 迭代器/空 vector/step==0 死循环/relay 越界/alias 校验** 。

#### 17.I 类型系统（JitKey/NanoString，ALL_TODO_001-008）
- 🔁 **_001 ACL binary_dtype_infer 硬编码 ns_add**（应传 op）`binary_op.cc:445`——与 17.A 中 ACL-DTYPE-PROMOTION 同条，**合并**。CUDA 不受影响（无该路径）。
- 🔁 **_002 scalar dsize 条件覆盖错**（both-scalar 时丢信息）`nano_string.h:223-241`，dtype_infer 与 binary_dtype_infer 同 bug。
- 🔁 **_003 jit_key 256-bit 非对齐访问**（ARM64 严格对齐 UB）`jit_key.h:134-145`。
- 🔁 **_004 pow() Windows/非 Windows float16 cast 不一致** `common_op_type.cc:45-50`。
- _005 sigmoid 精度（同 17.B）；_006 AMP 命名反向；_007 fp16 cast 链冗余；_008 dsize==0 映射无校验。

#### 17.J 优化 Pass（loop 变换 + atomic/cache/func/vectorize，14+16 条）
- 🔁 **merge_loop_pass loop_id 无分隔符拼接→`'01'` 与 `'1'+'0'` 碰撞静默错合并**（critical，CUDA 与 CPU 共用路径）`merge_loop_pass.cc:30-33`。
- **reorder/split/loop_to_func 越界**：reorder insert 无 `choice<=order.size()` 校验 `:24-25`；split_loop 计数整型溢出 `:23-25`；loop_to_func `split().at(0/1)` 未校验 + auto 类型推导对嵌套模板失败 `:87-95`。
- **float_atomic_fix match() 失败静默 + fp16/bf16 跳过无 atomicCAS** 🔁(NV-only) `:37-49,82-84`；check_cache substr 下溢 `:60-74`；fake_main ops[i] 越界 `:63-79`；expand_empty_block erase 失效 `:23-28`；replace_for_num ASSERT 硬崩；**expr.cc simplify 缺 0*x/1*x 化简 + match 后置条件 + 负数括号** `:557-560,69,590-599`。

#### 17.K 核心变量与图（var/var_holder/grad/graph，BUG_001-015）
- ⚠️🔁 **BUG_005 ItemData fp16 缓冲溢出**（int64 当 fp16）`var_holder.cc:215-224`，ROCM/CUDA 分歧。
- ⚠️ **BUG_001/006 allocator null 解引用** `:202-206,205,237`；**BUG_002 NodeFlags bit 冲突**（`_th_require_grad` 与 `_is_scalar` 同 `_n+5`）`node.h:47-48`；**BUG_003 Var::alloc 未初始化 allocation** ；**BUG_007 SetitemOp 无类型校验强转** ；**BUG_008 graph liveness 复用同计数器** ；**BUG_011 VarPtr 拷贝构造悬垂引用** ；**BUG_009/012/013 grad dtype/null 校验** ；**BUG_015 hold_vars 迭代器竞态**。

#### 17.L cuBLAS/cuSPARSE / cuFFT-NCCL 中归入 #17 的条目
- 🔁 **cuBLAS batched_matmul 4D+ 批/步长假设 + batch_size 溢出 + trans 标志脆弱字符串比较** `cublas_batched_matmul_op.cc:113-166`。
- 🔁 **cuBLAS acc_matmul stride_a 被注释禁用 + lda/ldb 在 trans 调整后计算→错位** `cublas_acc_matmul_op.cc:113-119`。
- **cuSPARSE 稀疏索引无越界校验 + dBuffer 错误路径泄漏** `cusparse_spmmcsr_op.cc:47,58-72`。

---

### #16 — 内存分配 / Swap / 内存信息（mem/allocator/*, swap.cc, mem_info.cc）

> SFRL/Temp/NFEF/CudaDual/Aligned + swap，合并两份审计（17+12 条）。

- ⚠️ **Allocation 移动构造未清源**（Top 20 #9）`allocator.h:39-41`；**unique_ptr 构造器未初始化 size/allocation** `:42-43`；**构造器重复赋 allocator + 异常不安全** `:44-46`。
- ⚠️ **CudaDual free_ids 竞态**（Top 20 #10）`cuda_dual_allocator.h:34-64` 🔁(查 ACL 等价)；**ref_cnt 下溢无校验** `:56-64`。
- ⚠️ **swap fwrite 反逻辑**（Top 20 #8）`swap.cc:52` + 文件句柄循环损坏 `:48-57`；**fopen null 未检查（CPU 路径）** 🔁(CUDA 路径有 CHECK，CPU 无) `:61,194`；**cudaMemcpy 缺 checkCudaErrors** `:50,184`；静态 buffer 泄漏；孤儿 swap 文件；force-evict swap_timestamp 未更新 `:76-132`。
- **SFRL：occupied_id_mapper 越界 `:289-304`（Temp 同 `:73,90`）；should_split 无符号下溢 `:166-172`；block 合并未更新 map key `:197-216`；ID_LIMIT off-by-one `:73`；erase_occupied 无边界 `:81-86`。Temp：share_with `ASSERT(false)` 硬崩 `:115-118`；free() 变量遮蔽 `:90`。NFEF freed 无界增长。Aligned macOS 对齐不保证 + null 未检查。**
- **mem_info 除零**（空 allocator/temp/var 统计）`:148-155,185-192,231-250`。

---

### #21 — DDP / 分布式训练（NCCL ↔ HCCL 高度对偶）🔁

> 本节几乎全部为跨后端对偶，**N 卡与华为成对修复**。

| 子项 | NCCL（N 卡）| HCCL（华为）| 状态 |
|------|------------|------------|------|
| all_gather 反传缺失 | nccl_all_gather_op.cc:39-41 | hccl_all_gather_op.cc:38-41 | 🔁 两侧均 stub，应 all_reduce(dout) |
| reduce 非 root 清零 | ✅ cudaMemsetAsync :58 | ❌ 缺失 :35-59 | 🔁 HCCL 补 aclrtMemset |
| bf16 dtype | ✅ 有 | ❌ 四算子全缺 | 🔁 HCCL 补 ncclBfloat16 等价 |
| broadcast 反传 | ✅ nccl_reduce :32-34 | ❌ 误返 broadcast :25-27 | 🔁 HCCL 改 hccl_reduce |
| 不支持 dtype @else | 部分缺 | 全缺→宏展开 garbage | 🔁 加 @else 报错 |
| 错误宏传播 | checkCudaErrors 抛异常 | ACLCHECK/HCCLCHECK 静默 return | 🔁 HCCL 传播异常 |
| 初始化校验 | world_size/rank 未校验 :50-84 | 同 | 🔁 双侧加边界校验 |
| 懒初始化检查 | 缺 | 缺 ASSERT(hccl_inited) | 🔁 |
| reduce_op 参数校验 | 硬编码 ncclSum | @strcmp 无 @else | 🔁 |
| rootinfo 文件 IO | fwrite 返回值未查 / 超时 id 未初始化→hang | — | NCCL 侧 |

- **nccl_all_reduce 缺 bf16 宏**（与 reduce/broadcast/all_gather 不一致）`:47-54`。
- **集合算子硬编码 stream 0**（绕过 event_queue，多流同步 bug）🔁 `nccl_*:56-57, cufft:84`。

---

### #3 — 复数 dtype / FFT（cuFFT + torch_compat/shim）

- ⚠️🔁 **cuFFT JK cache key 漏 inverse 标志**：fwd 与 ifft 同形状复用同 plan→结果静默错 `cufft_fft_op.cc:72-76`。
- **plan 创建错误未校验即入缓存** `:79-85`；dtype 只接受 float32/64 拒复数 `:43-45`；batch_size 未校验 >0；ASSERT(false) 应优雅报错。
- ⚠️ **torch_shim FFT 丢虚部**（两分支都只返 res.real）`torch__init__.py:951` + 硬编码 float32；fft2/ifft2 是恒等 no-op `:958-959`。

---

### #7 — fp16/bf16 混合精度

- 🔁 **ACL binary Add/Sub 缺 bf16 标量 case**（alpha 留 nullptr）`binary_op_acl.cc:43-99`，utils.cc:18-19 已识别 ACL_BF16。
- **get_init_var_rand 忽略 dtype 参数恒转 float32** `nn.py:163-164`；RMSNorm.weight 恒 float32 `torch_compat.py:791`。
- **NCCL reduce memset(0) 对 fp16/bf16 语义**（数值正确但应 dtype-aware）`nccl_reduce_op.cc:56-58`。

---

### #12 — 清晰错误信息 / 异常安全

- **cuBLAS/cuSPARSE get_dtype 静默回落 fp32**（int8 等无报错）🔁 `cublas_wrapper.h:24-33, cusparse_wrapper.h:28-37`。
- **cuBLAS/cuSPARSE handle 在无设备时未初始化→null 解引用** `:19-21`。
- **ACL get_dtype default 静默返 ACL_FLOAT** `utils.cc:42-45`；base_op checkRet 对 null err msg 解引用 `base_op_acl.cc:115-124`。
- **vectorize_pass ASSERT 无诊断上下文** `:37-62`；HCCL CHECK 宏静默 return（见 #21）。

---

### #19 — 图 / 代数优化

- **expr.cc simplify 缺 0*x→0、1*x→x** `:557-560`；match 后置条件假设 `:69`；check_cache 正则不全 / 参数无范围校验 `:34-45,84-98`。

---

### #6 / #10 — 测试覆盖 / 健壮性

- bool 规约测试在 gen_data 转 int32 掩盖 dtype 不一致 `test_reduce_op.py`；use_movnt_pass 未测试 🔁(NV-only)；collate_batch 空批无 guard `utils.py:26-27`。

---

### #8 — 分布式启动 / DataLoader

- ⚠️🔁 **Dataset MPI 散播重复**（Top 20 #15）`dataset.py:509-517`。
- 🔁 **compile_extern 未定义 mpi_compile_flags**：NCCL `:623`、HCCL `:690`（成对）。
- **worker daemon 无 join 资源泄漏 / 关闭时序竞态 / __del__ 与 reset 清理不一致** `:85-93,365-426`；launcher 顺序 wait + `or` 丢多 rank 错误码 `:77-82`；nproc 未校验设备数。

---

### ACL 后端专项（多套自有编号，未并入全局 #）

> Sigmoid/SiLU/Softmax/Binary、where/pool/norms/index、Flash-Attention/RoPE/Cumsum/SetItem、acl_op_exec 等。

#### 激活与二元（softmax/sigmoid/silu/binary）
- 🔁 **softmax dim 误 cast 为 aclDataType**（前向）softmax_op_acl.cc:41；后向 :65 正确。
- 🔁 **binary alpha 标量缺 bf16 + 无条件 aclDestroyScalar(nullptr)** binary_op_acl.cc:43-99,121。
- **sigmoid/silu/softmax 强制 float32 前向但后向用 grad_output.dtype→dtype 失配**（bf16/float64 训练破坏）sigmoid_op.py:62/72-79, silu_op.py:62/64-79, softmax_op.py:62-87。

#### where/pool/norms/index/base（ALL_TODO_037-050）
- ⚠️ **NonzeroACL 反传形状不兼容**（critical）where_op.py:74-75；Nonzero 用 `!=0.0` 强制浮点比较 :65；**WhereACL 对 condition 返回梯度应为 None** :148。
- **资源泄漏**：BatchNormBackward 缺 aclDestroyBoolArray :66-82；LayerNorm normalizedShape 未销毁 :92-109；LayerNormBackward 异步未同步即销毁 :140-142。
- 🔁 **maxpool indices int32 应 int64**（torch 兼容）pool_op.py:124,163；AvgpoolBackward 参数顺序错 :166。
- **BatchNormBackward 缺 momentum 参数** :64-68；base run() 非 group 算子查找无校验→解引用 end() :127-150；workspace 异步复用竞态 acl_jittor.cc:26-45。

#### Flash-Attn/RoPE/Cumsum/SetItem（ALL_TODO-001~018）
- ⚠️ **RopeACL 返回原始输入**（Top 20 #1）+ freq_sin/cos 参数交换 rope_op.py:61 vs acl_compiler.py:84 🔁(若加 NV 版须对齐)。
- **FlashAttention 中间输出 shape 硬编码 8 / output_dtypes 用字符串 'float' / scale dtype 不随 q** flashattention_op.py:113,165,73-138。
- **CumsumACL.grad 用 ReduceAttr 做 Flip（语义错）+ prod_dim 硬编码** cumsum_op.py:84-86。
- **InplaceMaskedScatter/IndexPutImpl 张量参数顺序疑误** setitem_op_acl.cc:48,77。

#### acl_op_exec / acl_jittor / utils（SHADOW/TODO_*，14 条）
- ⚠️🔁 **try_exec_and_fallback_cpu 变量遮蔽 `op`→悬垂引用** acl_op_exec.cc:203,221…362。
- 🔁 **reduce op_idx=999（乘法不支持）静默继续→executeOp switch 不匹配** :281-283（查 CUDA 等价）。
- 🔁 **get_dtype 缺 bf16 / 默认静默 fp32** binary_op_acl.cc:43-98, utils.cc:42-45；**stride 假设连续布局未校验** utils.cc:116-129；**transpose fallback 误用 ReduceAttr** :341-356；**random offset 竞态** :395-408；**workspace 溢出无界** acl_jittor.cc:28-29；error_code rfind 无 npos 检查。

---

### cuDNN 专项（ALL_TODO_CUDNN_001-017）

- ⚠️ **所有 Conv/RNN workspace allocation 未初始化**（workSpaceSize==0 时传垃圾给 free）_001/006/007。
- ⚠️🔁 **RNN backward dtype 系列**（Top 20 #4）_002/003/004/005/008/014：xDesc/dxDesc 用 Ty 应 Tx、dx/dhx/dcx 硬编码 float32、weight desc 硬编码 CUDNN_DATA_FLOAT；_014 明确两后端结构相同。
- ⚠️ **Conv stride int 溢出（>2GB）** _009 `:131`；**RNN backward memset 用元素数而非字节** _010 `:161`。
- **proj_size 未实现阻塞 v8/v9** _011/015；描述符清理非异常安全 _012；benchmark 静默回退 _013；workspace size 用 dxDesc 应 xDesc _016。

---

### nn.py / torch_compat / torch_shim / linalg（应用层）

- **nn.py**：Conv3d kernel 维序错 `:1181-1345`；nll_loss `ignore_index>0` 应 `>=0` + 1D 索引 `target[0]` `:456-462`；logsumexp 数值不稳 `:550-551`；Bilinear bias=False、Embedding padding_idx、PReLU 广播维硬编码、tensordot bitmap、LayerNorm affine=False 类型；缺 RReLU/GLU/EmbeddingBag/AdaptivePooling。
- **torch_compat**（17 条）：topk/sort 解包（Top 20 #11）、_backward 忽略 gradient、load 丢 dtype（Top 20 #12）、_rebuild_tensor_v2 忽略 stride、get_default_device 忽略 set、缺 item()/expand()/where()、bf16 位移重解释错、complex 仅 stub。🔁 共用 wrapper。
- **torch_shim**（18 条）：FFT 丢虚部（critical，见 #3）、clip_grad_norm 恒 L2（critical）、ConcatDataset 越界（critical）、缺 torch.linalg/torch.special、F8 朴素转换丢精度、safetensors 全量加载、各 LR/Sampler 边界。
- **linalg.py**（19 条）：SVD backward 高/宽矩阵维度失配 + 实矩阵多余 conj `:266-285`、复数 EIG/SVD 无 backward、QR 仅方阵、缺 lstsq/lu/matrix_exp/rank/cond、pinv 病态无稳定化。

---

### 核心 Python（__init__ / compiler，TODO_*）

- 🔁 **Function._grad 索引错配**（Top 20 #13）`:1979-1980`；**Module.npu 误设 use_cuda**（Top 20 #14）`:1411-1413`。
- **dlopen_flags 漏 RTLD_DEEPBIND（复制粘贴到 import_flags）** compiler.py:944-946；profile_mark/flag_scope 原地改 dict；flatten off-by-one `:727`；single_process_scope 缺 bk_mpi_state；std/var 偏差不一致。

---

## 🔁 跨后端对偶（N 卡 ↔ 华为昇腾）汇总速查

成对修复，**改一侧必须同步另一侧**：

| 对偶项 | N 卡侧 | 华为/ACL 侧 |
|--------|--------|-------------|
| RNN backward dtype（Top 20 #4）| cudnn_rnn_backward_x_op.cc | 等价 MIOpen/ACL 封装 |
| all_gather 反传缺失（Top 20 #6）| nccl_all_gather_op.cc:39-41 | hccl_all_gather_op.cc:38-41 |
| reduce 非 root 清零（Top 20 #7）| ✅ nccl :58 | ❌ hccl :35-59 |
| broadcast 反传 | ✅ nccl_reduce | ❌ hccl 误返 broadcast |
| bf16 集合算子/标量 | ✅ nccl | ❌ hccl 四算子 + binary_op_acl |
| 错误宏传播 | checkCudaErrors 抛 | ACLCHECK/HCCLCHECK 静默 |
| Dataset MPI 散播重复（Top 20 #15）| 共用 dataset.py | 共用 dataset.py |
| compile_extern mpi_compile_flags | NCCL :623 | HCCL :690 |
| dtype_infer/binary ns_add 硬编码 | CUDA 走 JIT（无此路径）| ACL binary_op.cc:445 |
| ACL-DTYPE-PROMOTION | operation-specific（正确）| ACL 硬编码 ns_add |
| axes 位移溢出 / 越界 | transpose_op.cc:64,69 | transpose_op_acl.cc:44 |
| get_dtype 静默回落 fp32 | cublas/cusparse_wrapper.h | acl utils.cc:42-45 |
| atomicAdd half/bf16 | setitem_op.cc:363 / float_atomic_fix | IndexPutImpl（ACL 内部处理） |
| CUDA abs ::abs（fp 损坏）| common_op_type.cc:19 | — (CPU:86 正确，参照修复) |
| handle 无设备未初始化 | cublas/cusparse_wrapper.cc | — |
| 集合算子硬编码 stream 0 | nccl_* | hccl_*（应核查） |
| torch_compat/shim wrapper | 共用 Python 层 | 共用 Python 层 |
| Function._grad / Module.npu | 共用 | 共用（npu 误设 use_cuda）|
| merge_loop loop_id 碰撞 | CUDA 分支 | 同 pass，昇腾继承 |
| reduce 乘法 op_idx=999 | 查 cuda_reduce 等价 | acl_op_exec.cc:281 |

---

## 建议处置顺序

1. **P0 — 静默数据损坏 / 训练直接出错**：Top 20 全部 + #21 DDP 对偶（reduce 非 root、all_gather 反传、broadcast 反传）+ swap fwrite 反逻辑 + reduce 乘法 NaN + RopeACL no-op。
2. **P0 — 立即崩溃**：topk/sort 解包、Function._grad 索引、cuDNN workspace 未初始化、VLA 栈溢出。
3. **P1 — 内存安全/UAF/竞态**：allocator 移动构造、CudaDual 竞态、occupied_id_mapper 越界、executor custom_data 复用、fused_op jit_key UAF。
4. **P1 — dtype/精度**：RNN backward dtype 对偶、cuSPARSE/cuBLAS fp16、CUDA abs/sigmoid/erfinv、ACL 激活 float32 强制。
5. **P2 — API gap / 数值稳定 / 错误信息**：torch.linalg/special、缺失 nn 模块、logsumexp、get_dtype 报错、清理顺序与文档。