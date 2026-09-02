# 架构与代码组织（横切）

**如果重写这个框架**，值得原封保留的只有三件事：统一惰性图加融合执行的执行模型
（`Executor::run_sync` 的 BFS→并查集融合→拓扑排序这条主线在实测里 kernel 时间已优于
PyTorch）、JIT 代码生成与 loop-transform pass 管线（`src/opt/` 82 文件 7589 行，是真正
的差异化资产）、以及按域拆分的 Python 包骨架。必须换掉的是**把绑定/构建/兼容三层的
关注点焊进核心数据结构的做法**：`Node` include Python tracer、C++ 里有 `th_mode`、
后端改写 compiler 全局、`use_acl` 只是 `use_cuda` 的别名、构建脚本用正则解析 C++ 源码。
还必须换掉**三套手工 liveness 计数**和**把执行触发放在 VarHolder 构造函数里**这两个
决定——它们是那类 bug 的共同根因，也是"设备成为 Var 属性"绕不过去的障碍。

## 核心抽象是否成立
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| Node 把图节点、内存对象、Python 追踪对象合成一个类型，最底层类型反向依赖绑定层 | `src/node.h:10` include `pybind/py_var_tracer.h`；`:169` 析构调 `trace_data.release_node(this)` | 图数据结构无法脱离 CPython 单测或复用；tracer 改动重编译全树 | 追踪改为节点生命周期的观察者接口，由 pyjt 层注册 | 主要 |
| 三套 liveness 计数是抽象泄漏 | `node.h:120-133` 用注释规定 f1/f2/f3、b1/b2/b3、p1/p2 八条规则；`need_free()` 是三者手写布尔式；own/release_{forward,backward,pending,both} 共 8 个手工配对方法 | 规则只存在于注释里，新持有者必须手工维护三个计数；Tapes double free 与 auto_flush 的 tape 排除都是漏掉一条规则 | 用显式的根集合加可达性模型替代三计数，或至少封成带不变量断言的类型 | 关键 |
| NodeFlags 位域被 Var 和 Op 复用且**位重叠** | `node.h:44-48`：`_th_require_grad=_n+5` 与 `_is_scalar=_n+5` 同一位；注释自陈两种布局共用该位，读通用 Node 的调用者必须先区分种类 | 一个 uint32 承载两套互不兼容语义，泛型代码读 flag 必然错，无编译期保护 | Var/Op 各持自己的 flag 类型 | 主要 |
| Op 同时是图节点/代码生成单元/执行单元/反向定义/JIT-key 提供者 | `op.h:44-62` 一个类 12 个虚函数：grad/grads/infer_shape/run/jit_prepare/do_jit_prepare/do_prepare/do_run_after_prepare/do_run/duplicate/compile_optimize/graph_optimize，外加 get_jit_key/get_hash_name/statistics | 加一个后端算子必须实现四个关注点；三个 do_* 的分层只在注释里 | 拆成 OpDef、Kernel、Codegen 三个接口由注册表组合 | 主要 |
| VarHolder 既是 Python 生命周期又是用户 API 门面又是执行触发点 | 54 个 @pyjt 标注（全树 146 个的 37%）；`var_holder.cc:65-88` add_hold_vars 在构造函数里调 auto_flush 并在急切模式调 sync | 创建持有变量带有可能发射整段图的副作用；错误不能从构造函数抛出，于是有 flush_suspended 这个补丁状态（`executor.h:34-35`） | 提前发射做成执行器显式接口，由 Python 侧调度调用 | 关键 |
| Var::allocator 字段被类型双关 | `var.h:56` share_with 把 `Var*` 存进 `Allocator*`；`var.cc:116-128` 靠 mem_ptr==nullptr 区分两种含义 | 任何在分配前读 `var->allocator->is_cuda()` 的代码都是 UB；设备属性无处安放 | 分成 allocator 与 share_src 两个字段 | 主要 |
| "元算子"名不副实 | `src/ops/` 30 个算子中真正的元算子只有 unary/binary/ternary/reduce/broadcast_to/reindex/reindex_reduce 七个；其余是 getitem/setitem/argsort/candidate/copy/clone/where 等具体算子，还有一个融合优化器 kernel `fused_adamw_op.h:7` 直接住在核心 | 元算子是宣传语（README:9,197）而非代码里的边界，核心因此不断长出非元算子 | 明确划出 meta 层与 composite 层 | 主要 |
| Executor 是单函数上帝对象 | `executor.cc:200-719`，run_sync **一个函数 520 行**，含 BFS、并查集融合、两次拓扑排序、内存分配、换出磁盘、CPU/GPU 迁移、并行编译、JIT key 构造、发射、profiling、NaN 检查、异步回溯；`executor.h` 只有 8 个成员且 4 个是流水线补丁状态 | 无法缓存调度结果、无法多流、无法插入设备选择 | 拆成 Planner（图到执行计划，可按结构哈希缓存）与 Runner | 关键 |

## 分层与依赖方向
实际依赖图中的三个**真环**：
1. `jittor_utils` ⇄ `jittor.compiler`：`jittor_utils/__init__.py:731,807,810,869` 里 import jittor，而 compiler.py 顶部 import jittor_utils。所谓底层工具包实际依赖它服务的框架。
2. `Executor` ⇄ `VarHolder`：`var_holder.h:11` include executor.h；`executor.cc:24` include var_holder.h（另有 13 个核心 .cc 包含 var_holder.h）。
3. `Node` ⇄ pyjt tracer：`node.h:10`。

**反向依赖**（下层被上层改写）：
- 三个后端模块 import 时直接改写 compiler 模块全局共 22 处：`rocm_compiler.py:145-153`、`acl_compiler.py:134-139`、`corex_compiler.py:67-77`；`acl_compiler.py:76` 更往 `compiler.extra_core_files` 追加核心源文件。
- 核心算子按字符串名认识可选后端：`ops/transpose_op.cc:43` has_op("cutt_transpose")、`random_op.cc:25`、`argsort_op.cc:43`、`where_op.cc:28`、`arg_reduce_op.cc:45`；全树 127 处 get_op_info。
- 优化器层写死后端算子名：`opt/tuner/conv_tuner.cc:371-395` 在 mkl_conv 与 cudnn_conv 间选；`matmul_tuner.cc:103` 同理。
- C++ 核心内含 Torch 兼容概念：`var.cc:26` `DEFINE_FLAG(uint8, th_mode)`、`op.cc:142`、`node.h:47`、`mem/mem_info.cc:73`，核心共 16 处 th_mode。

**跨层直接调用**：`nn/functional/matrix.py:45,51,64,106,112,207,219` 直接读 `jt.compile_extern.cublas_ops`；`attention.py:40` 读 `jt.compiler.is_cuda`。违反 source-architecture.md 明写的方向。全树 17 个 Python 模块 import compile_extern，其中 8 个在 nn/ 下。

| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 核心通过全局字符串注册表反向依赖可选后端 | 5 处 has_op 加 tuner 两文件 4 处写死名字 | 加后端要改核心；缺后端只能靠名字查不到静默降级，无法枚举支持集 | OpRegistry 分派表，核心只查表不认名字 | 关键 |
| 构建层与核心层互为输入 | jittor_utils ⇄ compiler 环；后端改写 compiler 全局 22 处；extra_core_files 由后端追加 | import 顺序即行为；无法在不 import jittor 的情况下工具化 | jittor_utils 降为纯函数库禁止 import jittor；后端配置返回 BuildConfig 值 | 主要 |
| Torch 兼容概念下沉到 C++ 核心 | `var.cc:26`、`op.cc:142`、`node.h:47`、`mem_info.cc:73` | 核心梯度语义有两套，任何核心改动要考虑 Torch 模式 | th_mode 上移为 autograd 策略对象 | 主要 |
| 兼容层拥有核心对象 | `__init__.py` 末行 `_core_api.flags = flags`；`compat/shim/control.py:169` wrap_flags 无条件替换 jt.flags（原生模式也执行）；`compat/runtime.py:73` `sys.modules["torch"] = root_module` | jt.flags 在任何进程里都是 compat 代理；Torch 模式下 torch 与 jittor 是同一模块对象 | flags 代理逻辑归核心；torch 用独立模块对象 | 主要 |
| import 即编译即联网下载 | `compiler.py:1437` 顶层 compile jittor_core；`compile_extern.py:975-985` 顶层 setup_nccl/cutt/cutlass/mkl 到 download_url_to_local | 直接违反 source-architecture.md 的「Module imports must not compile kernels, download assets, or silently install external packages」；离线只读环境不可用；锁自死锁由此而来 | 编译与下载移到显式 bootstrap 或首次算子调用 | 主要 |

## 模块边界：文档声明 vs 代码现实
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| nn/ 目录契约已被破坏 | source-architecture.md 声明 nn/ 只含 modules/functional/backends/utils/attention.py；实际根下另有 13 个文件 2389 行，其中 6 个是 CUDA kernel 包装（rms_norm_cuda 342、rope_cuda 229、kv_cache_cuda 224、packed_qkv_cuda 181、swiglu_cuda 54、_cuda_inference 50）按文档应在 backends/；kv_cache_acl.py 276 行是 ACL 专属却在公共包里 | 同类算子散在 nn/、nn/backends/、extern/cuda/、src/ops/ 四处 | 给 nn/ 加 exact-entry 门禁 | 主要 |
| 结构门禁把现状固化成契约 | `tests/structure/test_nn_structure.py:48-52` 逐个 import_module 具体文件名，`:170-274` 断言其导出名 | 门禁保护的是现在长什么样而非应该长什么样，重构被自己的测试锁死 | 结构测试断言规则而非清单 | 主要 |
| `python/jittor/utils/` 是杂物间 | 11 文件 1477 行无共同职责：asm_tuner/dumpdef/dlink_compiler（编译器资源）、gen_pyi/local_doc_builder（仓库工具应在 tools/）、pytorch_converter.py 718 行（源码翻译器应在 compat）、converter_server.py（其启动脚本却在 tools/services/legacy/）、nvtx、jtune、tracer | 根目录 exact-entry 门禁把 utils 整体放行，杂物在门禁内侧继续堆积 | 拆散 | 主要 |
| `src/misc/` 是 C++ 侧杂物间，最核心的状态住在里面 | 25 文件 12233 行占 src 的 26%：vendored miniz 9214 行、类型系统 nano_string/nano_vector 655 行、NaN 检查、`misc/cuda_flags.cc:19-24` 定义 use_cuda/device_id/sync_run。同时另有 `src/type/` 879 行也是类型系统 | 类型系统一分为二；最重要的全局状态在名为 misc 的目录里 | miniz 移出；nano_* 并入 src/type/；cuda_flags 并入 Runtime 对象 | 主要 |
| agent/ 树不在任何布局文档里却承载布局门禁 | 163 文件 2.3 MB；repository-layout.md 的 Target Layout 不含 agent/，但引用 `agent/scripts/check_repo_layout.sh` 作为验收条件；docs/ 里 9 处反向链接到 agent/results/ | 两套并行的架构文档，读者不知哪套权威 | design 并入 docs/architecture/，脚本并入 tools/ | 次要 |
| 死目录与死文件 | `tests/system/` 下只有 legacy/*.sh 零个 .py；`src/utils/flags.cc` 27 行全在注释里但仍被 flag 扫描器读到 | 门禁与扫描器把死代码当活代码 | 删除 | 次要 |
| 结构靠测试而非构造保证 | tests/structure 22 文件 8071 行占测试树 11%；test_nn_structure.py 单文件 1912 行 | 架构约束的成本落在测试维护上且会固化现状 | 少数可执行规则替代逐项清单 | 主要 |

## 重复与一致性
| 问题 | 证据（量化） | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| ACL 算子样板复制粘贴 | aclops/ 115 文件；32 个 .py 加 39 对 .cc/.h；**17 个 .py 共享同一段 12 行代码块**，另一段块跨 16 个文件。4325 行 py 加 5074 行 cc/h 实现 39 个算子 | 每加一个算子复制三个文件；模板 bug 要改 17 处 | AclOp 基类加声明式算子表 | 主要 |
| cuDNN/MKL 卷积族复制 | conv{,3d}_{,backward_x,backward_w} **6 个文件共享同一 12 行块**；其中一段同时出现在 cudnn_conv_backward_w 与 mkl_conv{,_backward_x,_backward_w} 4 个文件 | backend API 迁移只做了 2D，conv3d 三处仍是 legacy，正因为它们是复制品而非共享实现 | 抽出 conv 描述符与计划的共享层 | 主要 |
| 反向定义在 C++ 与 Python 各存一份且 Python 那份是绕过 | `nn/backends/cudnn.py:8-16` 手写 `_CudnnConv2d(jt.Function)`，注释写 autodiff through the raw op is broken in this build；该缺陷已修但绕过代码仍在 | 真值有两份且不同步，修复不会自动生效 | 修复后删掉绕过并补 CPU 参考对拍 | 主要 |
| 后端适配无共同接口，三份各写各的 | 三个 *_compiler.py，22 处 `compiler.X = ...`；add_backend 只要求 check()，install_extern/post_process 用 hasattr 探测 | 每个后端实现不同子集且无契约测试 | BackendRegistry 契约加契约测试 | 关键 |
| 一个概念的多种表达：CUDA kernel 的四种存放方式 | ① `extern/cuda/<lib>/ops/*.cc` 31 个；② `nn/backends/*_cuda.py` 10 文件 2079 行内嵌 cuda_src 字符串；③ `nn/*_cuda.py` 6 文件；④ `src/ops/` 的 `#ifdef HAS_CUDA` 分支。全树 **54 个 .py 含 cuda_src 共 14755 行** | 找一个算子的实现要搜四个地方 | 一个位置加一个注册表 | 主要 |
| flag 重复定义 | 78 个 flag 中 12 个定义在两个文件，因为 `src/utils/flags.cc` 是注释掉的死文件而正则扫描器不认注释 | 生效的默认值取决于 glob 顺序 | 删死文件；flag 扫描改预处理后扫描或宏注册 | 次要 |
| 安装/校验函数各写各的 | install 在 19 个模块各自定义、install_parity 在 6 个 installers、check 在 7 个模块 | 无共同签名与错误约定 | 定义 Installer/Backend 协议类型 | 次要 |

## 公共 API 与内部实现的边界
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 根命名空间由 5 次星号导入叠加，无封闭承诺 | `__init__.py:57-58`、`:86-88`、`:105`、`:124-125`；`__init__.pyi` 只声明 238 个顶层名 | 无法判断哪些名字是承诺的；.pyi 与实际不一致 | 显式 __all__；.pyi 从 __all__ 生成并加一致性门禁 | 主要 |
| 78 个 flag 全部公开可读写且不分类 | 构建期只读（cc_path/cc_flags/nvcc_flags/jittor_path/python_path/cache_path）、运行期设备、调试、数值策略、分配器混在同一命名空间 | 用户可以在运行中改 cc_flags；框架自己也这样用 | 分为 jt.config（启动期只读快照）与 jt.runtime | 主要 |
| jt.compile_extern.* 是运行时 globals() 注入却被当公共分派依据 | `compile_extern.py:269-270,380-381,394-395,415-416`；`nn/functional/matrix.py` 6 处据此分派；测试引用 137 次 | 静态分析、.pyi、IDE 全部失效 | 后端能力查询接口 | 主要 |
| 影子对象模型：205 个 `_torch_*` 属性名 | 全树 205 个不同名字（python/jittor 内约 100）：_torch_grad(34 处)、_torch_index_parent、_torch_data_owner、_torch_leaf_params(16 处)、_torch_force_cpu(17 处) 等 | 无类型无清理点无所有权；静默丢权重与叶子被裁剪两类 bug 都是这个模型的必然产物 | 一个显式的 TorchTensorState 对象 | 关键 |
| 10 个 `jt._*` 名字是跨模块契约 | _torch_leaf_params、_active_optimizers、_current_optimizer、_torch_retained、_torch_sdpa_flash_stats、_transform_getitem_to_index_depth、_acl_clamp、_C、_torch_compat_install_context/complete | 单进程单线程假设固化；模块间通过根命名空间通信 | 收进显式 Runtime 对象 | 主要 |
| 测试大量依赖内部细节 | 283 处 jt.flags.*、137 处 compile_extern/jt.compiler.*、127 个测试文件触碰下划线名或 __dict__ | 内部重构必然触发大面积测试改动，实际冻结了内部实现 | 给测试提供受支持的内省 API | 主要 |
| torch 与 jittor 是同一个模块对象 | `compat/runtime.py:73`；`compat/torch/__init__.py:187` install(torch) 的实参就是 jittor 根模块 | Torch 模式进程里原生 Jittor 语义被就地改写；repository-layout.md 声称的"不改变无关进程的原生 API"只在进程间成立 | 独立的 torch 模块对象只做委托 | 主要 |

## 代码规模与分布
| 层 | 文件 | 行数 | 说明 |
| --- | --- | --- | --- |
| C++ 核心 src/ | 292 | 46735 | 去掉 vendored miniz（9214）后 **37521**；misc/ 12233 占 26%，ops/ 7962，opt/ 7589 |
| pyjt 绑定 | 12+1 | 2482+965 | pyjt_compiler.py 解析 `// @pyjt` 注释生成绑定，全树 146 个标注其中 54 个在 var_holder.h |
| extern 后端 | 257 | 28790 | ACL 13646 / CUDA 11601 / MKL 1810 / MPI 887 / ROCm 748 / Corex 97 |
| Python 运行时与 API | 736 | 68846+8003(.pyi) | compat 28198 / nn 11630 / models 4436 / misc 3419 / _runtime 2615 |
| jittor_utils | 18 | 3555 | |
| 测试 | 357 | 72309 | compat 20542 / backends 8223 / **structure 8071** / core 7355 |

| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 单函数异常 | run_sync 520 行 | 见核心抽象节 | 拆 Planner/Runner | 关键 |
| 单文件异常（框架自身） | misc/tensor_ops.py 2874；_runtime/core_api.py 2614；installers/nn.py 2454；installers/tensor.py 2413；shim/backends/flash_attention.py 2086（大半是内嵌 CUDA 字符串）；compiler.py 1500；acl_compiler.py 1397；op_compiler.cc 1171；opt/expr.cc 1180 | 这些是域而不是文件 | 按域拆分继续下推一层 | 主要 |
| 层厚薄失衡：接口薄实现厚 | executor.h 40 行接口对 executor.cc 744；mem/allocator.h 58 行抽象下挂 8 个实现 2173 行；compat/ 28198 行没有任何契约定义 | 抽象没有承载设计意图 | 关键接口写成显式契约 | 主要 |
| "后端"这个抽象不成立（规模差 18 倍） | ACL 13646 行 133 文件 39 算子；ROCm 748 行 0 算子（靠文本改写复用 CUDA 源码，含把 `run_pass<FloatAtomicFixPass>();` 替换成字面量 WTF 让该 pass 编译失败） | 两个后端没有共同形状，无法有共同契约与矩阵测试 | 注册表加分派表，删除文本改写 | 关键 |
| 测试分布与风险倒挂 | 核心 C++ 37.5k 行对 tests/core 7355 行（0.20）；compat 28198 对 20542（0.73）；结构测试 8071 行超过 tests/core | 最难改最容易出隐性 bug 的一层测得最少 | 结构测试预算转向核心执行器与图不变量的属性测试 | 主要 |
| 工具链覆盖极窄 | `pyproject.toml:70` ruff 仅 E4,E7,E9,F,UP006,UP007；`:80-91` mypy 只覆盖 **7 个文件**（占 820 个 Python 文件的 0.9%），其中 2 个在 agent/、2 个是结构测试 | 静态工具无法承担任何边界约束，全部压给结构测试 | 先把 import 方向做成 lint 规则 | 主要 |
