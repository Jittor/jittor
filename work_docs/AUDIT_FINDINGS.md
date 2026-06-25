# Jittor dev 仓库审计汇总工作清单

## 🔥 Top 20 最高杠杆

| title | maps_to | severity | location | work_item |
|---|---|---|---|---|
| Inverted logic in swap_to_disk() write success check | #16 | critical | src/mem/swap.cc:51-55 | 修复 `if(res==1)` 反向的 fwrite 成功判断，应为 `if(res!=1)` |
| Unchecked fopen() return value in swap operations | #16 | critical | src/mem/swap.cc:61,194 | 在 swap_to_disk()/move_with_swap() 文件 I/O 前对 fopen() 做空指针检查 |
| Missing bounds validation for occupied_id_mapper[] access | #16 | critical | src/mem/allocator/sfrl_allocator.cc:291,314 | 在 free()/share_with() 访问前加 `ASSERT(allocation < ID_LIMIT)` |
| Race condition in CudaDualAllocator free_ids management | #16 | critical | src/mem/allocator/cuda_dual_allocator.h:46-64 | 为 alloc/free 加互斥锁，避免多线程 free 列表损坏/双重分配 |
| Allocation constructor redundant assignment bug (uninitialized allocation field) | #16 | critical | src/mem/allocator.h:44-46 | 删除重复 allocator 赋值，确保 allocation 由 alloc() 结果初始化 |
| Silent fallback to ACL_FLOAT on unsupported dtype | #17 | critical | extern/acl/aclops/utils.cc:43-44 | unsupported dtype 改为抛异常而非静默回退 ACL_FLOAT |
| RopeACL returns wrong values (inputs instead of results) | #6 | critical | extern/acl/aclops/rope_op.py:81 | 返回计算结果而非未修改的输入张量 |
| Typo in Upsample.align_corners attribute name | #1 | critical | nn.py:2286 | 修正 `align_cornerss` → `align_corners` |
| to_tensor tuple conversion creates generator instead of list | #17 | critical | transform/__init__.py:416 | 生成器表达式改为列表推导 `[to_tensor(p) for p in pic]` |
| CenterCrop passes float arguments to crop expecting int | #17 | critical | transform/__init__.py:396 | top/left 强制转 int 后传入 crop() |
| Hard PIL import break at module level | #15 | critical | dataset/dataset.py:18; transform/__init__.py:13 | PIL 改为可选导入并给出友好报错；文档化硬依赖 |
| Bare except catches system exits (worker crashes hidden) | #10 | critical | dataset/dataset.py:271,283,417 | bare except 改为 `except Exception`，保留 KeyboardInterrupt/SystemExit 并记录 traceback |
| torch.fft.fft2/ifft2 are identity no-ops | #3 | critical | torch_shim/torch__init__.py:958-959 | 用真实实现替换 identity lambda，或抛 NotImplementedError |
| FFT numpy wrapper loses imaginary part | #3 | critical | torch_shim/torch__init__.py:951 | _np_fft_wrap 返回复数输出，勿丢弃虚部 |
| No native complex dtype limits linalg/FFT effectiveness | #3 | critical | linalg.py (throughout) | 实现原生 complex64/128 dtype（FFT/复数 linalg 的根本阻塞） |
| argmax/argmin int64 cast fails on Python 3.13 (asm_tuner import) | #13 | critical | torch_compat.py:577-588 + utils/asm_tuner.py:15 | 修复 asm_tuner 的 jittor_utils 导入，或将 int64 cast 推迟到导出期 |
| cumsum(bool) cast broken on Python 3.13 (same root cause) | #13 | critical | torch_compat.py:1294-1302 + asm_tuner | 修复 asm_tuner + unary.cast 编译 |
| Disabled stream synchronization hides data dependency bugs | #19 | critical | extern/acl/aclops/base_op_acl.cc:106-113 | 初始化时强制流内顺序执行；workspace 增长时做 hazard 检测；动态 shape 测试 |
| Workspace allocation fails silently on shape changes without recompile | #19 | critical | extern/acl/acl_jittor.cc:26-45 | 将 shape 加入 JIT key；分发前校验 workspace ≥ 所需；加 cache miss 计数 |
| No JIT recompile on tensor shape change for ACL ops | #19 | critical | extern/acl/acl_op_exec.cc:443-467 | get_jk() 纳入输入/输出 shape 哈希；executeOp 前做 shape 校验 |

---

## #17 修复 Jittor 全部 bug

整体状况：分布最广的一类，覆盖核心算子（梯度/dtype/切片）、优化 pass（越界/off-by-one/缓冲区溢出）、内存分配器、ACL dtype 处理与异常安全，是工作量与风险最集中的编号。

**核心算子（src/ops）**
- mod 梯度丢失 dout 因子，反向传播错误 — `src/ops/binary_op.cc:516-522` — high
- maximum/minimum 梯度对并列值分配错误（应按 PyTorch 平均） — `src/ops/binary_op.cc:524-531` — high
- reduce mean 梯度在 v->num==0 时除零/NaN — `src/ops/reduce_op.cc:350-352` — medium
- TernaryOp(where) 混合 dtype 缺少 cond 为 bool 的校验 — `src/ops/ternary_op.cc:42-46` — high
- getitem 负索引归一化：越界检查在调整之后，下溢静默通过 — `src/ops/getitem_op.cc:140-146` — high
- BinaryOp 在 ACL 上静默 auto-cast，与 CUDA/CPU 报错行为分歧 — `src/ops/binary_op.cc:443-451` — high
- float16/bfloat16 mean 精度恢复被 amp_reg 位 32 静默关闭 — `src/ops/reduce_op.cc:259-265,295-301` — medium
- pow 梯度对负底数无定义域检查，产生 NaN — `src/ops/binary_op.cc:532-541` — medium
- getitem 负 step 切片输出 shape 计算错误（被 clamp 成空） — `src/ops/getitem_op.cc:168-174` — high
- mod 对 x 的梯度为 dout，需注释/gradcheck 说明正确性 — `src/ops/binary_op.cc:516-522` — high
- SetitemOp 复合操作（+=/-=/*=//=）缺少 dtype 提升 — `src/ops/setitem_op.cc:40-56,134-192` — medium
- WhereOp CUDA 前缀和对大数组数值不稳定 — `src/ops/where_op.cc:97-140` — medium
- BroadcastToOp::infer_shape 使用栈上 VLA，不可移植 — `src/ops/broadcast_to_op.cc:134-137` — medium
- reduce_dtype_infer 对 bool 输入静默改 int32，与 torch 语义分歧 — `src/ops/reduce_op.cc:286-290` — medium

**优化 pass（src/opt）**
- reorder_loop_pass 循环索引插入 off-by-one（缺 `choice<=order.size()` 校验） — `python/jittor/src/opt/pass/reorder_loop_pass.cc:24-26` — high
- merge_loop_var_pass substr 越界（条件应为 `>=6`） — `python/jittor/src/opt/pass/merge_loop_var_pass.cc:74-77` — high
- loop_to_func_pass 未保护的 `.at(0)/.at(1)`（split 结果可能仅 1 元素） — `python/jittor/src/opt/pass/loop_to_func_pass.cc:89-94` — high
- float_atomic_fix_pass match 结果未判空即 `.at(0/1)` — `python/jittor/src/opt/pass/float_atomic_fix_pass.cc:43-49` — high
- loop_var_analyze_pass 维度跟踪不完整，静默漏维 — `python/jittor/src/opt/pass/loop_var_analyze_pass.cc:118-142` — high
- split_loop_pass 大 factor 整型溢出（缺上界） — `python/jittor/src/opt/pass/split_loop_pass.cc:23-25` — medium
- reorder_loop_pass 无子循环时静默跳过 — `python/jittor/src/opt/pass/reorder_loop_pass.cc:40-51` — medium
- check_cache_pass `substr(sp-4,5)` 未校验 `sp>=4` — `python/jittor/src/opt/pass/check_cache_pass.cc:62-72` — medium
- replace_for_num_pass `ASSERT(found)` 硬失败，应优雅跳过 — `python/jittor/src/opt/pass/replace_for_num_pass.cc:39` — medium
- expr to_string 负整数缺括号产生歧义表达式 — `python/jittor/src/opt/expr.cc:592-599` — medium
- merge_loop_pass loop_id 无分隔符拼接致静默错误合并 — `python/jittor/src/opt/pass/merge_loop_pass.cc:30-36` — high
- expand_empty_block_pass insert 越界 + erase 后索引偏移 — `python/jittor/src/opt/pass/expand_empty_block_pass.cc:23-28` — medium
- fake_main_pass `ops[i]` 无边界检查 — `python/jittor/src/opt/pass/fake_main_pass.cc:75` — medium

**内存分配器与 swap（src/mem）**
- erase() 中块删除后访问 `cur->second->id`（use-after-free） — `src/mem/allocator/sfrl_allocator.cc:187-189` — high
- should_split() `block->size - size` 整型下溢 — `src/mem/allocator/sfrl_allocator.cc:166-172` — high
- TempAllocator::share_with() 无条件 `ASSERT(false)` 硬失败 — `src/mem/allocator/temp_allocator.cc:115-118` — high
- id_mapper 解引用后未判空块指针致 segfault — `src/mem/allocator/sfrl_allocator.cc:289-304` — high
- swap 文件 I/O 中 cudaMemcpy 缺 checkCudaErrors — `src/mem/swap.cc:50,184` — high
- TempAllocator::free() 内层变量遮蔽外层 block — `src/mem/allocator/temp_allocator.cc:90` — medium
- Allocation 移动构造未清 o.allocation/o.allocator — `src/mem/allocator.h:39-41` — high
- allocation_size() GPU 内存检查无符号下溢 — `src/mem/allocator/sfrl_allocator.cc:150-154` — high
- swap_timestamp 在 alloc_with_swap() 中从不自增，OOM 无法解决 — `src/mem/swap.cc:76-132` — high
- NFEFAllocator freed 列表无界增长 — `src/mem/allocator/nfef_allocator.cc:29-30` — high
- SFRL 块合并不更新 map key 致分配失败 — `src/mem/allocator/sfrl_allocator.cc:197-216` — high
- macOS aligned_allocator 不保证对齐 — `src/mem/allocator/aligned_allocator.cc:18-21` — medium
- SFRL free() 未校验 size 参数与 block->size 一致 — `src/mem/allocator/sfrl_allocator.cc:289-304` — medium

**ACL 后端与 dtype**
- FlashAttention 中间输出硬编码 `float`，bf16/fp16 丢精度 — `extern/acl/aclops/flashattention_op.py:165` — high
- Nonzero 硬编码 `0.0` 比较，整型张量错误 — `extern/acl/aclops/where_op.py:65` — high
- get_dtype() 缺 float64 case，回退 ACL_FLOAT — `extern/acl/aclops/utils.cc:12-46` — medium
- Pool maxpool 索引硬编码 int32（torch 用 int64） — `extern/acl/aclops/pool_op.py:124` — low
- get_dtype 对 uint64/uint32 默认回退 ACL_FLOAT 致数据损坏 — `extern/acl/aclops/utils.cc:12-45` — medium
- Workspace size 不匹配在执行期未检测 — `extern/acl/aclops/norms_op_acl.cc:41-50` — high
- CreateFakeTransAclTensor 缺 rank 校验 — `extern/acl/aclops/utils.cc:139-155` — low
- aclCreateTensor 返回值未校验空指针 — `extern/acl/aclops/utils.cc:112-135` — critical
- Workspace malloc 多线程不安全 — `extern/acl/acl_jittor.cc:26-44` — critical

**ACL 资源/异常安全（acl-perf 分支）**
- aclCreateScalar 异常路径无清理致泄漏（多处仅 index_op 安全） — `extern/acl/aclops/index_op_acl.cc:48-50,66-68` — high
- use_cuda flag pinning 不完整，ACL-only op 边界遗漏 — `src/op_compiler.cc:776-785` — medium
- cudaDeviceSynchronize 临时绕过未修根因（run_sync 死锁） — `src/executor.cc:706` — medium
- finish pending_liveness TODO 标注潜在 bug — `src/executor.cc:673` — low
- aclnn op 异常时无显式 tensor/scalar 清理（cleanupDesc 漏调） — `extern/acl/aclops/base_op_acl.cc:127-150` — medium

**CUDA wrappers**
- cuDNN conv stride 用 int32 易溢出，应 int64/size_t — `extern/cuda/cudnn/ops/cudnn_conv_op.cc:131` — medium
- cuFFT plan 创建失败未检查即缓存致悬挂句柄 — `extern/cuda/cufft/ops/cufft_fft_op.cc:79-85` — low
- cuDNN 描述符清理失败时早期描述符仍占用（缺 RAII） — `extern/cuda/cudnn/ops/cudnn_conv_op.cc:307-310` — low

---

## #9 torch 自定义算子直接迁移

整体状况：torch_compat/nn 层缺失大量常用张量方法与 functional 算子，外加 ACL 算子覆盖与编译能力短板；多为 API 缺口（缺方法直接 AttributeError），补齐成本通常较低。

**缺失张量方法**
- 缺 tensor.expand()/expand_as() — `torch_compat.py` — medium
- 缺 tensor.squeeze()/unsqueeze() — `torch_compat.py` — medium
- 缺 masked_select/tile/narrow — `python/jittor` — low
- 缺 torch.randperm/permutation — `python/jittor/__init__.py` — medium
- 缺 torch.seed 别名 — `torch_compat.py` — low

**缺失 functional 算子（nn.py）**
- 缺 F.conv1d / F.conv_transpose1d — medium
- 缺 F.adaptive_avg_pool2d / AdaptiveAvgPool2d — medium
- 缺 F.softsign / Softsign — low
- 缺 F.glu — low
- 缺 F.prelu（仅有 module） — `nn.py:337` — low
- 缺 F.rrelu / RReLU — low
- 缺 F.pairwise_distance — low
- 缺 F.cosine_similarity — low
- 缺 F.embedding_bag / EmbeddingBag — low
- 缺 F.normalize — low
- GroupNorm 缺 3D（5D 输入）支持 — `nn.py:822` — low

**torch_compat 行为/返回类型**
- SVD 返回类型不清（namedtuple vs tuple） — `torch_compat.py:1660-1663` — medium
- Generator.seed() 返回值不符 torch — `torch_compat.py:391-403` — medium
- GradScaler.step() 应始终返回 None — `torch_compat.py:249-257` — medium
- backward 不支持 inputs 参数（选择性梯度） — `torch_compat.py:1371` — low
- F.embedding 不支持 padding_idx — `torch_compat.py:467` — medium
- 复数 dtype 仅为 stub — `torch_compat.py:69-88` — low
- sort() namedtuple 名与 torch.return_types 不符 — `torch_compat.py:558,633-635` — low
- SVD 缺 some 参数 — `torch_compat.py:1660-1663` — medium
- topk sorted 参数被忽略 — `torch_compat.py:620-630` — low
- F.linear 未包装 bias=None 处理 — `torch_compat.py:464` — low
- clip_grad_norm_ 空参数列表边界返回类型不一致 — `torch_shim/torch__init__.py:120-143` — low
- int8 量化 matmul 缺失（PyTorch parity） — `extern/cuda/cublas/ops/cublas_matmul_op.cc:30` — medium
- 批量 matmul 假定 ndim>=3，不支持 2D — `extern/cuda/cublas/ops/cublas_batched_matmul_op.cc:62-89` — low

**ACL 算子覆盖/编译**
- ACL 多维 ReduceMultiply 不支持（op_idx=999 回退 CPU） — `extern/acl/acl_op_exec.cc:282` — medium
- flex_attention 运行时抛 NotImplementedError，无降级 — `torch_shim/torch__init__.py:220-221` — medium
- 多行宏展开不支持，自定义 CUDA op JIT 错误 — `src/op_compiler.cc:211` — low
- ACL 输入张量连续性未校验 — `extern/acl/aclops/base_op_acl.cc:49-68` — medium
- ACL 算子覆盖不足（102 vs 400+ torch ops；einsum/linalg/FFT/NMS） — `extern/acl/acl_jittor.h:339-443` — high

---

## #12 所有核心报错清晰可排查

整体状况：报错普遍缺少维度/dtype/设备/算子上下文，且多处不支持 dtype 时静默回退 fp32 造成数值分歧；亟需统一错误码注册表与结构化异常元数据。

- BinaryOp 广播 shape 不匹配报错未指明出错维度 — `src/ops/binary_op.cc:432-433` — medium
- ArgReduceOp::grad reshape 前缺 shape 兼容性断言/报错 — `src/ops/arg_reduce_op.cc:104-105` — medium
- pass manager 缺 pass 依赖跟踪，关键 pass 被跳过无报错 — `python/jittor/src/opt/pass_manager.cc:92,96,120` — medium
- vectorize_pass 断言缺上下文（哪个 split_id/stride） — `python/jittor/src/opt/pass/vectorize_pass.cc:48-49` — medium
- CudaDeviceAllocator 静默捕获异常回退 managed，隐藏根因 — `src/mem/allocator/cuda_device_allocator.cc:20-34` — medium
- used/unused_memory 在 display_memory_info 无锁读取致撕裂 — `src/mem/allocator/sfrl_allocator.cc:243-286; mem_info.cc:147-155` — high
- cuBLAS get_dtype 不支持 dtype 静默回退 fp32 — `extern/cuda/cublas/inc/cublas_wrapper.h:31` — medium
- cuSPARSE get_dtype 不支持 dtype 静默回退 fp32 — `extern/cuda/cusparse/inc/cusparse_wrapper.h:36` — medium
- 卷积 groups 不整除输入通道时断言信息不清 — `extern/cuda/cudnn/ops/cudnn_conv_op.cc:68` — low
- cuBLAS alpha/beta fp16 精度不匹配（兼 #2） — `extern/cuda/cublas/ops/cublas_matmul_op.cc:64-104` — high
- cuSPARSE computeType 硬编码 CUDA_R_32F（兼 #2） — `extern/cuda/cusparse/ops/cusparse_spmmcsr_op.cc:62,67` — high
- 'Not a valid call' 报错无函数签名 — `python/jittor/pyjt_compiler.py:713` — high
- ACL op 'not supported' 缺操作上下文（兼 #11） — `extern/acl/acl_op_exec.cc:291,359,422` — high
- checkRet 报错无 op 名/shape/dtype 上下文（兼 #11） — `extern/acl/aclops/base_op_acl.cc:115-124` — medium
- exec_mapped_acl_ops 'Unsupported operation type' 无 op 名/降级提示 — `extern/acl/acl_op_exec.cc:412-424` — medium
- ACL 错误码无 Python 可读名称映射 — `extern/acl/acl_error_code.cc` — high
- 'Wrong inputs arguments' 未区分解析/运行时错误 — `python/jittor/pyjt_compiler.py:695-741` — high
- ReindexOp 报错缺越界/溢出上下文（兼 #11） — `python/jittor/src/ops/reindex_op.cc:41-57,83-90` — medium
- HCCL/ACLNN 错误宏静默 return，不向 Python 传递（兼 #11） — `extern/acl/hccl/inc/hccl_wrapper.h:33-57` — high
- numpy_code_op/code_op shape 校验晚且无提示 — `python/jittor/src/ops/numpy_code_op.cc:40,69` — medium
- parallel_compiler 多线程错误聚合未区分 — `python/jittor/src/parallel_compiler.cc:331` — medium
- reshape 报错未指明哪个负维度 — `python/jittor/src/ops/reshape_op.cc:45,50` — low
- ACL op 缺集中 dtype/shape 支持注册表与校验 — `extern/acl/aclops/` — high
- py_converter.h 类型转换失败无运行时提示 — `python/jittor/src/pyjt/py_converter.h:678-679,710` — medium
- 无集中错误码注册表（C++/CUDA/ACL） — `python/jittor/src/` — high
- torch_shim 错误澄清器正则解析 [Reason] 脆弱 — `torch_shim/torch__init__.py:908-938` — low
- check_op_async_error 缺 dtype/device/allocator 信息 — `src/executor.cc:129-176` — medium
- get_dtype uint64/uint32 默认静默回退 fp32（兼 #17） — `extern/acl/aclops/utils.cc:12-45` — medium
- 通用错误信息掩盖 op 特定失败（'xxx' 占位） — `extern/acl/aclops/base_op_acl.cc:123` — high
- op 属性类型不匹配致 nullptr 解引用 — `extern/acl/aclops/norms_op_acl.cc:38-56` — high
- CPU 回退未校验 dtype 兼容性 — `extern/acl/acl_op_exec.cc:97-128` — medium
- BaseOpRunner checkRet 吞掉 aclnnStatus 不传播（兼 #17） — `extern/acl/aclops/base_op_acl.cc:115-124` — medium
- torch-compat dtype stub 不校验后端计算能力 — `torch_compat.py:62-91` — low

---

## #16 优化显存管理

整体状况：内存子系统是 critical 缺陷的重灾区（已上 Top20 多条），此外存在性能性问题（碎片、未释放 swap 文件）及缺用户级显存调优 API。本节列除 #17 已归类内存 bug 之外的 #16 项。

- 缺 CUDA 错误检查的 swap 文件 I/O（cudaMemcpy） — `src/mem/swap.cc:50,184` — high（同见 #17 内存段）
- swap 文件清理失败未升级，孤儿文件堆积 — `src/mem/swap.cc:143-144,200-201` — medium
- 缺用户级显存上限/缓存控制 Python API — `src/mem/allocator.cc, src/mem/swap.cc` — medium（兼 #12）
- JT_SAVE_MEM swap 机制缺文档与验证（TODO 未完成） — `src/mem/swap.h:40-65` — medium
- BFS 调度器每次 sync 的 per-op 开销未量化（兼 #19） — `src/executor.cc:200-260` — high

> 内存安全 bug（inverted swap check、未检 fopen、occupied_id_mapper 越界、DualAllocator 竞态、构造函数未初始化等）已归入 Top20 与 #17「内存分配器与 swap」段，此处不重复。

---

## #19 优化统一计算图特性

整体状况：聚焦 ACL 图执行性能（流同步、JIT shape key、workspace 校验）与执行器调度开销；多为 perf 风险，含若干 critical（已上 Top20）。

- Expr::simplify() 缺代数化简（0*x→0, 1*x→x） — `python/jittor/src/opt/expr.cc:557-560` — high
- check_cache_pass 赋值运算符正则不完整 — `python/jittor/src/opt/pass/check_cache_pass.cc:35` — low
- use_cuda_managed_allocator 跳过输出迁移行为不一致 — `src/executor.cc:596-600` — medium
- cuDNN benchmark 因 workspace 比例超限静默禁用 — `extern/cuda/cudnn/ops/cudnn_conv_op.cc:230,239` — low
- 不支持 cuDNN packed layouts（v8+ perf） — `extern/cuda/cudnn/ops/cudnn_conv_op.cc:139-203` — low
- ACL fallback exec_and_fallback_cpu 仍有 per-op 流同步 — `extern/acl/acl_op_exec.cc:149` — high
- 图优化重触发可能性能回退（无迭代上界） — `src/executor.cc:252-260` — medium
- top_weak_sync 可能跳过必要的最终同步 — `src/executor.cc:178-198` — medium
- var_fused 弱/强 share 切割阈值（魔数 32）无界且未配置 — `src/executor.cc:468-469` — low
- ACL fallback 每次重建依赖图，无缓存 — `extern/acl/acl_op_exec.cc:152-189` — medium
- H2D memcpy 在热路径用阻塞调用 — `extern/acl/acl_op_exec.cc:273` — high
- inplace masked scatter 缺数据依赖同步 — `extern/acl/aclops/setitem_op_acl.cc:45-47` — high
- 流内顺序执行未保证 — `extern/acl/acl_jittor.cc:88` — medium
- 不支持动态 rank（0-d↔n-d shape 变化） — `extern/acl/aclops/base_op_acl.cc:49-68` — medium
- BFS executor dispatch 每 sync 开销未量化（兼 #16） — `src/executor.cc:200-260` — high

> critical 项（disabled stream sync、workspace 静默失败、ACL 无 shape 重编译）已在 Top20。

---

## #2 cuDNN/cuBLAS 版本与混合精度对齐

整体状况：CUDA 库 wrapper 大量落后于现代 cuDNN/cuBLAS（v8/v9、bf16、loss scaling、projection RNN），阻塞 LLM 训练与混合精度 parity。

- 缺 cuDNN8/9 版本支持（RNN） — `extern/cuda/cudnn/inc/cudnn_rnn_descriptor.h:149` — high
- cuBLAS alpha/beta fp16 精度不匹配（兼 #12） — `extern/cuda/cublas/ops/cublas_matmul_op.cc:64-104` — high
- cuSPARSE dtype 硬编码 CUDA_R_32F（兼 #12） — `extern/cuda/cusparse/ops/cusparse_spmmcsr_op.cc:62,67` — high
- cuFFT 假定输入为复数但未校验 dtype（兼 #3） — `extern/cuda/cufft/ops/cufft_fft_op.cc:43-94` — high
- cudnnGetConvolutionForwardAlgorithm_v7 已弃用（cuDNN9 缺口） — `extern/cuda/cudnn/ops/cudnn_conv_op.cc:257` — medium
- RNN backward 输出强制 float32，破坏混合精度（兼 #7） — `extern/cuda/cudnn/ops/cudnn_rnn_backward_x_op.cc:52-56` — medium
- ROCm cuBLAS 路径缺 bfloat16 支持 — `extern/cuda/cublas/ops/cublas_matmul_op.cc:88-116` — medium
- 缺 PyTorch 式混合精度 loss scaling 支持（兼 #7） — `extern/cuda/cublas/ops/cublas_matmul_op.cc:88-123` — medium
- RNN proj_size 始终断言为 0，阻塞投影 LSTM/GRU — `extern/cuda/cudnn/ops/cudnn_rnn_op.cc:34,49` — low
- int8 量化 matmul 缺失（兼 #9） — `extern/cuda/cublas/ops/cublas_matmul_op.cc:30` — medium

---

## #3 复数类支持

整体状况：复数支持是 torch.fft/linalg/special 全家桶的根本阻塞——原生 complex dtype 缺失导致 FFT 退化为 identity/丢虚部、linalg 复数反向未实现，且 torch.linalg/torch.special 模块整体缺位。

**根本阻塞与 FFT 正确性（已上 Top20）**
- torch.fft.fft2/ifft2 为 identity no-op — `torch_shim/torch__init__.py:958-959` — critical
- FFT numpy wrapper 丢失虚部 — `torch_shim/torch__init__.py:951` — critical
- 无原生 complex dtype 限制 linalg/FFT — `linalg.py (throughout)` — critical

**FFT 覆盖**
- FFT 仅支持 2D+CUDA，缺 1D/ND 与实值变体（rfft/fftn 等） — `nn.py:3096-3103` — high
- ComplexNumber.fft2/ifft2 缺 CPU 回退 — `nn.py:3297-3301` — medium
- 缺 fftfreq/rfftfreq/fftshift/ifftshift — `torch_shim/torch__init__.py` — low
- cuFFT 假定复数输入未校验（兼 #2） — `extern/cuda/cufft/ops/cufft_fft_op.cc:43-94` — high

**linalg 模块与复数反向**
- torch.linalg 模块未在 torch_shim 暴露 — `torch_shim/torch__init__.py` — high
- complex_eig backward 抛 NotImplementedError — `linalg.py:86` — high
- complex_svd backward 抛 NotImplementedError — `linalg.py:198` — high
- 缺 lstsq/lu/matrix_exp/matrix_rank/cond 等 linalg 函数 — `linalg.py` — medium
- SVD full_matrices 参数不支持 — `linalg.py:223` — low
- QR 不支持非方阵 — `linalg.py:606-608` — low
- cholesky 等维度语义文档不清 — `linalg.py:524-603` — low

**special 模块**
- torch.special 模块整体缺失 — `torch_shim/torch__init__.py` — high
- 缺 torch.special.logsumexp — `torch_compat.py` — medium
- 缺 torch.special.softmax/log_softmax — `torch_compat.py` — medium
- 缺 torch.special erf/erfc/erfinv — `torch_compat.py` — medium
- 缺 torch.special.gammaln/digamma — `torch_compat.py` — low
- 缺 torch.special.polygamma/psi — `torch_compat.py` — low

**复数 dtype stub（acl-perf 分支）**
- 复数 dtype 仅定义无 compute op，应在 Var 创建期报错 — `torch_compat.py:69-71` — medium

---

## #1 模型库与层修复

整体状况：条目少但含 critical（Upsample 拼写错误已上 Top20）；其余涉及 ViT/模型库相关层级修复（本审计集中表现为 nn.py 单点 bug）。

- Typo `align_cornerss` → `align_corners`（已上 Top20） — `nn.py:2286` — critical

---

## #21 双卡/并发验证

整体状况：聚焦多卡与并发场景下的资源安全与测试覆盖；条目少但触及 critical 级线程安全问题。

- Workspace malloc 多线程不安全（加锁或线程局部缓冲） — `extern/acl/acl_jittor.cc:26-44` — critical（同见 #17 ACL 段）
- 无 all_gather 测试覆盖（op 存在但从未使用） — `python/jittor/test/test_nccl_ops.py` — medium
- Dropout 缺确定性/可复现 seed 控制 — `nn.py:569,603` — low

---

## #20 分布式状态与 collective 语义

整体状况：MPI-free DDP 路径的 collective 语义与 bootstrap 状态管理存在正确性缺口（非 root 输出未清零、world_size 不校验），多为中高危正确性/技术债。

- NCCL reduce 非 root 用 memset(0) 对非 float 类型不正确 — `extern/cuda/nccl/ops/nccl_reduce_op.cc:56-58` — high
- HCCL reduce 非 root 完全不清零输出（数据泄漏） — `extern/acl/hccl/ops/hccl_reduce_op.cc:56` — high
- compile_extern in_mpi 状态在 env/file 与 MPI bootstrap 路径间混淆 — `python/jittor/compile_extern.py:825-831` — medium
- NCCL/HCCL 不校验各 rank world_size 一致性 — `nccl_wrapper.cc:50-56; hccl_wrapper.cc:49-52` — medium
- HCCL 两套 bootstrap 路径未统一（技术债） — `python/jittor/compile_extern.py:836-845` — low
- NCCL/HCCL 平台流不对称（NCCL stream 0 vs HCCL aclstream） — `nccl/ops/*.cc; hccl/ops/*.cc` — low

---

## #15 DataLoader / 启动器 / rendezvous 健壮性

整体状况：DataLoader 缺 PyTorch 生态关键参数（collate_fn/worker_init_fn 等），NCCL/HCCL rendezvous 与启动器在失败路径上缺诊断与清理；含 critical（PIL 硬导入已上 Top20）。

**DataLoader API 缺口**
- 缺 collate_fn 参数 — `dataset/dataset.py:619-629` — high
- 缺 worker_init_fn 参数 — `dataset/dataset.py:619-629` — medium
- 缺 persistent_workers/pin_memory 参数 — `dataset/dataset.py:107-128` — medium
- MNIST/EMNIST 硬编码 RGB 转换无选项 — `dataset/mnist.py:80,176` — low

**测试覆盖**
- 无 PIL 缺失的优雅处理测试 — `test/test_dataset.py, test_transform.py` — medium
- 无损坏图片文件测试 — `test/test_dataset.py` — medium

**rendezvous/启动器健壮性**
- NCCL rank0 写 rootinfo 失败无错误处理，其他 rank 挂起 — `extern/cuda/nccl/src/nccl_wrapper.cc:66-67` — high
- NCCL 非 rank0 读 rootinfo 超时无诊断信息 — `extern/cuda/nccl/src/nccl_wrapper.cc:70-81` — high
- 启动器 rank spawn 失败无清理（进程泄漏） — `python/jittor/distributed/launch.py:58-68` — medium
- 启动器早退时日志句柄泄漏 — `python/jittor/distributed/launch.py:58-68` — medium
- 启动器未校验 -n 等于检测到的设备数 — `python/jittor/distributed/launch.py:40-68` — medium
- 启动器 SIGINT 后未等待优雅退出 — `python/jittor/distributed/launch.py:70-86` — medium

---

## 其他编号（次要，简列）

以下编号未进入主排序但有零散条目：

**#11 防段错误/硬化**
- 整体状况：执行器/编译器多处指针解引用与位打包缺边界检查，易致 segfault。
- free_var 中 allocator 为空时空指针解引用 — `src/var.cc:47` — high
- load_fused_op custom_data 移位无边界检查 — `src/executor.cc:84` — high
- top_weak_sync 迭代器递减无边界检查 — `src/executor.cc:189` — high
- 共享输入 op 融合校验不完整（TODO） — `src/executor.cc:442` — medium
- 多输出 op liveness 跟踪不完整（TODO） — `src/executor.cc:673` — medium
- use_cuda flag JIT 编译期竞态 — `src/op_compiler.cc:776-785` — medium
- FusedOp 边编码无溢出检查 — `src/executor.cc:88; src/fused_op.cc:181` — medium
- get_op_var_by_name stoi 后缺边界检查 — `src/op_compiler.cc:37-45` — medium
- loop_options 解引用前未判空 — `src/op_compiler.cc:809-810` — medium
- custom_data var 索引编码静默截断 — `src/fused_op.cc:102,111` — medium
- shared_id 向量跨 root 迭代未重置 — `src/executor.cc:420,427,439` — medium
- event_queue flush 异常路径缺清理 — `src/executor.cc:702-712` — medium
- weak sync 含无反向路径校验的 var — `src/executor.cc:193-196` — medium
- Cumsum 用错属性类（GatherAttr） — `extern/acl/aclops/cumsum_op.py:65` — medium
- aclGetRecentErrMsg() 返回空指针未检查 — `extern/acl/aclops/base_op_acl.cc:119` — medium
- ReduceMul op_idx=999 占位致执行崩溃 — `extern/acl/acl_op_exec.cc:281-283` — medium
- 无 ACL 设备张量校验 — `extern/acl/aclops/base_op.h:36-47` — medium
- HCCL/ACL 惰性初始化隐藏 setup 错误至首个 op — `extern/acl/hccl/inc/hccl_wrapper.h:68-70` — medium
- broadcast/expand 负/零维度缺校验 — `src/ops/broadcast_to_op.cc:84` — medium
- reindex_reduce_op 动态 shape 缺边界检查 — `src/ops/reindex_reduce_op.cc:66-67` — low
- 弱 share 阈值魔数 32 未文档化/不可配 — `src/executor.cc:468` — low
- max/min dispatch 与 keepdims kwarg 冲突脆弱 — `torch_compat.py:592-618` — medium
- Var.__reduce__ 访问未定义 .data 破坏 pickle — `__init__.py:2161` — high
- single_process_scope 全局状态变更无线程安全 — `__init__.py:334-350` — high
- flag_scope dict 合并就地修改调用方输入 — `__init__.py:166-171` — medium
- compile_custom_ops dlopen_flags=None 未初始化 — `compiler.py:665-676` — medium
- profile_mark.compile_options 缺类型校验 — `__init__.py:316-321` — low

**#10 报错可读性 + torch-compat 行为（Python 层）**
- 整体状况：大量 torch-compat 语义分歧、eval/deepcopy 不安全、缺 .item()/.to()/where() 等核心 API，及 docstring 示例未测试；以 API 缺口与可读性为主。
- 关键项：缺 torch.eye()（`torch_compat.py:661-665`，high）、缺 Var.to()（`torch_compat.py:1324-1327`，high）、缺 Var.item()（high）、缺 torch.where()（high）、nn.Module.to() 不递归（high）、F.softmax dim=None 默认错误（`torch_compat.py:459`，high）、torch.cat 全空输入返回错误（`torch_compat.py:426-433`，high）、topk 不校验 k 上界（`torch_compat.py:620-630`，high）、F.relu 缺 inplace（`nn.py:166`，high）、F.dropout 缺 training 参数（`nn.py:578`，high）。
- 其余 medium/low：__deepcopy__ 可变默认参数/缺 memo（`dataset.py:420,434`）、eval 无错误上下文（`compiler.py:229; misc.py:1318`）、_invert bool 矛盾逻辑（`torch_compat.py:1317-1322`）、device= kwarg 被静默忽略（`torch_compat.py:640-665`）、Var.data 非代理破坏链式（`torch_compat.py:1329-1342`）、Generator.set_state no-op、is_available 硬编码 flags、num_workers>0 被忽略、load_parameters 形状不匹配仅警告、std/var keepdim 命名不一致、RMSNorm weight 未注册 Parameter 等。

**#7 bf16/混合精度**
- 整体状况：bf16 在 ACL 二元算子、NCCL/HCCL all_reduce、多处强制 fp32 转换中缺失，阻塞 bf16 训练与 DDP。
- ACL Add/Sub 缺 bf16 标量处理 — `extern/acl/aclops/binary_op_acl.cc:39-98` — high
- Softmax/Sigmoid/SiLU 强制 fp32 丢 dtype — `softmax_op.py:62; sigmoid_op.py:62; silu_op.py:62` — high
- NCCL all_reduce 缺 bf16 — `nccl/ops/nccl_all_reduce_op.cc:47-54` — high
- HCCL all_reduce 缺 bf16 — `acl/hccl/ops/hccl_all_reduce_op.cc:37-44` — high
- BCELoss/BCEWithLogitsLoss 用已弃用 size_average — `nn.py:409,484,496,515` — high
- bfloat16 检测依赖运行时字符串匹配脆弱 — `extern/acl/aclops/utils.cc:11-17` — medium

**#8 backward / collective 完善**
- 整体状况：all_gather 反向在 NCCL/HCCL 均未实现，分布式 glue 缺 scatter/reduce_scatter，及若干同步/perf 项。
- NCCL/HCCL all_gather 反向未实现 — `nccl_all_gather_op.cc:39-41; hccl_all_gather_op.cc:38-41` — high
- collective 用默认 stream 0 — `nccl/ops/*.cc` — medium
- HCCL broadcast root 冗余 memcpy — `hccl_broadcast_op.cc:52-54` — medium
- 缺 mpi_all_gather/mpi_reduce_scatter — `__init__.py:2250-2251` — low
- MPI broadcast 子进程 TODO 未修 — `dataset/dataset.py:475-476` — medium
- .data 用于标量提取（resize area） — `nn.py:1994,1997` — medium

**#6 测试覆盖**
- 整体状况：负 step 切片、pass 交互、all_gather、损坏图片、复数 backward 等多处缺回归测试；含 ACL 正确性 bug（stride/rope 已分别归 #17/#6）。
- 负 step 切片无回归测试 — `src/ops/getitem_op.cc` — high
- pass 排序/交互/IR 有效性无测试 — `pass_manager.cc:66-123` — high
- use_movnt_pass 未测试的 pragma 注入 — `use_movnt_pass.cc:18` — medium
- CreateFakeTransAclTensor 转置 stride 计算 bug — `extern/acl/aclops/utils.cc:133-148` — medium
- H2D/D2H memcpy 缺错误检查 — `extern/acl/acl_op_exec.cc:273` — high
- Array op 不处理 0 维张量 — `extern/acl/acl_op_exec.cc:270-274` — low
- linalg batch 反向公式未对照 torch 验证 — `linalg.py:247-286 等` — medium

**#13 checkpoint/平台兼容**
- 整体状况：Python 3.13（asm_tuner 导入）阻塞 argmax/cumsum cast（已上 Top20），cuDNN RNN 权重布局与 .pt 互通缺失，及 PEP 667 exec/locals 隐患。
- cuDNN RNN weight offset 非 torch API，阻塞 checkpoint 互通 — `extern/cuda/cudnn/inc/cudnn_rnn_descriptor.h:151-152` — medium
- Python 3.13 PEP 667 exec/locals 修复范围过窄 — `compiler.py:764-772` — medium

**#4 / #5 / #14 / #18（nn 单点）**
- #4 CrossEntropyLoss mean 全忽略时除零 + 权重覆盖 mask — `nn.py:402,383-385` — high
- #5 Dropout mask 语义不清 — `nn.py:573,604` — medium
- #14 CrossEntropyLoss 无效目标处理破坏（mask 被覆盖） — `nn.py:383-385` — high
- #18 Conv 权重初始化选择未文档化（与 resnet 不一致） — `nn.py:984` — low