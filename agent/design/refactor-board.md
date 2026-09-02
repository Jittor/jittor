# 整改看板

一行一个任务，与 [refactor-plan.md](refactor-plan.md) 的编号对应。领任务把状态改成「进行中」并写名字，
完成改成「已合并」并填提交号；推送冲突说明别人先领了。状态只有四种：待领 / 进行中 / 已合并 / 并入 X。

## 门禁 agent 的最新结果

| 提交 | 原生 | CPU torch | CUDA | 失败用例 / 责任任务 |
| --- | --- | --- | --- | --- |
| 0e5e031b（分支起点） | 775 passed | 1595 passed | 进行中 | — |

## 热点文件占有

第一波（2026-09-02 21:xx 派出）：九个 agent 并行，各自独占一张卡/一段核/一个 worktree
（`/home/zy/jittor-lab/refactor/<分区>`，分支 `wk/<分区>`，推送到 `2.0-refactor`）。

| 分区 | 当前占有者 | 任务 |
| --- | --- | --- |
| 核心节点 | — |  |
| 执行器 | — |  |
| 代码生成 | — |  |
| 类型与日志 | coreops (6.C01/05/06/07/09/30) | GPU3 c24-35 |
| 内存 | mem (6.C10–6.C20) | GPU2 c12-23 |
| 绑定 | bindings (6.C02/22/23/24/25/27/28/29) | GPU0 c0-11 |
| Python 核心 | — |  |
| Python 算子 | pyops (6.P03–6.P09) | GPU5 c36-47 |
| Python 其他 | pyother (6.P10–6.P22) | GPU6 c64-75 |
| 兼容层 | compat (7.01) | CPU c96-103 |
| CUDA 后端 | cudabk (6.B05/07/08/09/12/13/14) | GPU7 c76-87 |
| ACL/ROCm/Corex | — |  |
| 分布式 | dist (6.B01/03/04/06/10/11/15) | CPU c88-95 |
| 构建 | — |  |
| 门禁 | gates (0.01–0.04/06/12/13/18/19) | CPU c104-111 |

## 任务

| 编号 | 任务 | 状态 | 负责 | 提交 |
| --- | --- | --- | --- | --- |
| 0.01 | `TestGradients` 改用 `only_for=("cpu",)` 显式实例化 | 待领 | | |
| 0.02 | 设备过滤后 bases 为空或方法数为 0 时生成器直接 raise | 待领 | | |
| 0.03 | `tests/compiler/test_jit_tests.py` 进 CPU 门禁，并断言 … | 待领 | | |
| 0.04 | 门禁改为「整个 `tests/` 减显式排除清单」，排除项必须写理由 | 待领 | | |
| 0.05 | 生态对拍进 nightly | 待领 | | |
| 0.06 | `make_tensor` 种子改为 `hash(nodeid, shape, dtype)` … | 待领 | | |
| 0.07 | 缓存路径追加构建配置指纹 | 待领 | | |
| 0.08 | 锁统一为一种类型、一个 fd | 待领 | | |
| 0.09 | 探测结果落盘 `cache_path/probe.json` | 待领 | | |
| 0.10 | 写缓存前检查可用磁盘空间，不足时给明确错误 | 待领 | | |
| 0.11 | 「jit_utils 已更新请重跑」改非零退出码 | 待领 | | |
| 0.12 | 14 处在用例里裸赋值 `jt.flags.*` 且无 tearDown 的测试改 `flag_… | 待领 | | |
| 0.13 | conftest 的模式由显式环境变量决定，删除 `sys.argv` 嗅探 | 待领 | | |
| 0.14 | `_session_env` 不再 `os.environ.copy()` | 待领 | | |
| 0.15 | 门禁分两层 | 待领 | | |
| 0.16 | `test_device_parity.py` 按算子分片并行，不再在 `setUpClass`… | 待领 | | |
| 0.17 | `pyproject.toml` 的 `pythonpath` 改由 conftest 按环境变… | 待领 | | |
| 0.18 | 门禁每条目断言至少执行 1 个非 skip 用例 | 待领 | | |
| 0.19 | 结构测试从「精确清单」改成「规则」 | 待领 | | |
| 0.20 | 布局收尾 | 待领 | | |
| 1.01 | 把 `utils/data.gz` 解出的 `data.cc` 还原为可读的五个翻译单元 | 待领 | | |
| 1.02 | `op_compiler.cc:30-69` 用正则给 `ParallelPass` 输出打补丁… | 待领 | | |
| 1.03 | 查明 `SharedReducePass` 在约 4900 个归约 kernel 里零命中的触发… | 待领 | | |
| 1.04 | `ReduceTuner::run` 不再对 CUDA 直接返回 | 待领 | | |
| 1.05 | 布局收尾 | 待领 | | |
| 2.01 | Var 与 Op 各持自己的 flag 类型 | 待领 | | |
| 2.02 | 删除 `Node::custom_data` | 待领 | | |
| 2.03 | `tflag` 全局计数器加魔数改为 epoch 对象或局部集合 | 待领 | | |
| 2.04 | `Var::allocator` 去类型双关 | 待领 | | |
| 2.05 | 真正的 0 维张量 | 待领 | | |
| 2.06 | 边表由 list 加反向迭代器改 SmallVector，按下标 O(1) | 待领 | | |
| 2.07 | `hold_vars`/`sync_ptr` 析构里 `std::next(end())` 的 … | 待领 | | |
| 2.08 | `Node` 不再 include `pybind/py_var_tracer.h` | 待领 | | |
| 2.09 | `th_mode` 从 C++ 核心上移为 autograd 策略对象 | 待领 | | |
| 2.10 | 三套 liveness 计数 | 待领 | | |
| 2.11 | `VarHolder` 不再是执行触发点 | 待领 | | |
| 2.12 | 打破 `Executor ⇄ VarHolder` include 环 | 待领 | | |
| 2.13 | 执行相关全局状态 | 待领 | | |
| 2.14 | `src/misc/` 拆散 | 待领 | | |
| 2.15 | NanoString | 待领 | | |
| 2.16 | 类型提升表 | 待领 | | |
| 2.17 | 算子身份用注册期整型 id | 待领 | | |
| 2.18 | 算子注册表惰性初始化 | 待领 | | |
| 2.19 | 错误分两档 | 待领 | | |
| 2.20 | 信号处理器只做 `write` 与 `_exit`，符号化交给预建 helper 进程 | 待领 | | |
| 2.21 | `DEFINE_FLAG_WITH_SETTER` 先赋值再调 setter，签名收新旧两值 | 待领 | | |
| 2.22 | 环境变量统一 `JT_` 前缀 | 待领 | | |
| 2.23 | 布局收尾 | 待领 | | |
| 3.01 | `Executor::run_sync` | 待领 | | |
| 3.02 | jit key 结构化 | 待领 | | |
| 3.03 | 三张 kernel 缓存表键改 `string` | 待领 | | |
| 3.04 | 求 jit key 改纯函数 | 待领 | | |
| 3.05 | 删除算子构造期回调执行器 | 待领 | | |
| 3.06 | 并行编译器修到可信 | 待领 | | |
| 3.07 | 执行器在设备等待段释放 GIL | 待领 | | |
| 3.08 | KernelIR 结构化 | 待领 | | |
| 3.09 | 死代码消除不再按「语句含 `void` 一词」删除 | 待领 | | |
| 3.10 | 算子内标识符改名走结构化成员表并先做合法性校验，替代三个硬编码白名单与 `op{i}_` 盲目前… | 待领 | | |
| 3.11 | 生成源码里的结构体字节偏移改显式 setter，成员表用宏声明 | 待领 | | |
| 3.12 | `float_atomic_fix_pass.cc:76-80`、`fake_main_pass… | 待领 | | |
| 3.13 | 循环维度身份用整数向量，`range10` 不再被拆成 `range1*range0` | 待领 | | |
| 3.14 | 两个同名 pass | 待领 | | |
| 3.15 | 一次编译只解析一遍 | 待领 | | |
| 3.16 | `token_replace_all` 不再用 CHECK 抛异常做循环终止 | 待领 | | |
| 3.17 | 只用于代码生成的 JIT 区段与普通 C++ 分离 | 待领 | | |
| 3.18 | 删掉 `asm_tuner` 链路 | 待领 | | |
| 3.19 | `event_queue` 异步基础设施修好并加测试，或删除 | 待领 | | |
| 3.20 | 执行器提供「提交部分图」显式接口，`jt.grad` 与 `Function` 回调降开销，让反… | 待领 | | |
| 3.21 | 每算子建图成本 | 待领 | | |
| 3.22 | CUDA 归约块内树形归约 | 待领 | | |
| 3.23 | 融合逐元素 kernel 带宽效率 | 待领 | | |
| 3.24 | 布局收尾 | 待领 | | |
| 4.01 | 分配器 id 空间随分配器实例走，不再是进程静态 2M 单例 | 待领 | | |
| 4.02 | 合并多卡 | 待领 | | |
| 4.03 | `BackendRegistry` | 待领 | | |
| 4.04 | `OpRegistry` | 待领 | | |
| 4.05 | Python 分派表 | 待领 | | |
| 4.06 | `jt.flags.backend_fallback ∈ {error, warn, allow… | 待领 | | |
| 4.07 | 后端配置改为返回 `BuildConfig` 值 | 待领 | | |
| 4.08 | 流与事件模型 | 待领 | | |
| 4.09 | per-device 库句柄 | 待领 | | |
| 4.10 | CUDA kernel 存放位置统一 | 待领 | | |
| 4.11 | ACL 改为注册表后端 | 待领 | | |
| 4.12 | 删除 `process_jittor_source` 与 `process_acl` | 待领 | | |
| 4.13 | 跨后端契约矩阵 | 待领 | | |
| 4.14 | `Module.cuda(i)`/`npu(i)`/`x.to(...)`/`x.cpu()` … | 待领 | | |
| 4.15 | 布局收尾 | 待领 | | |
| 5.01 | 114 个 `foo_` 就地方法改白名单显式声明 | 待领 | | |
| 5.02 | 视图与存储模型 | 待领 | | |
| 5.03 | 转置隐藏标记 | 待领 | | |
| 5.04 | 参数模型 | 待领 | | |
| 5.05 | `eval()`/`train()` 只切 `is_train`，冻结统一由 `requires… | 待领 | | |
| 5.06 | hook 存实例级有序字典，多 hook、prepend/always_call 生效、可移除 … | 待领 | | |
| 5.07 | `jt.Function` 每次调用创建一次性上下文对象，实例无状态 | 待领 | | |
| 5.08 | `flag_scope` 的备份改局部栈，`__call__` 每次新建 scope | 待领 | | |
| 5.09 | 29 处融合 kernel 的启用条件由全局 `no_grad` 改为「输出不需要梯度」 | 待领 | | |
| 5.10 | 索引与计数统一 int64 | 待领 | | |
| 5.11 | `amp_reg` 位常量命名导出，一律 `\ | 待领 | | |
| 5.12 | matmul 四条路径共用能力表，dtype 用枚举不用子串 | 待领 | | |
| 5.13 | `unique` | 待领 | | |
| 5.14 | `Var.scatter` 改非就地 | 待领 | | |
| 5.15 | `.half()`/`.float16()` 删死的 amp 分支 | 待领 | | |
| 5.16 | `state_dict(to="torch")` 用 `from_numpy`，不强制 floa… | 待领 | | |
| 5.17 | 同一概念合并 | 待领 | | |
| 5.18 | 同一概念合并 | 待领 | | |
| 5.19 | 被静默忽略的参数改为传非默认值时 warn 或 raise | 待领 | | |
| 5.20 | import 期副作用删除 | 待领 | | |
| 5.21 | 六个 monkeypatch 安装器写成显式有序清单并加断言 | 待领 | | |
| 5.22 | `nn` facade 不导出 39 个下划线名，内部用模块局部名不经 `jt.nn.*` 晚绑… | 待领 | | |
| 5.23 | 根命名空间显式 `__all__` | 待领 | | |
| 5.24 | 10 个 `jt._*` 跨模块契约 | 待领 | | |
| 5.25 | `python/jittor/utils/` 拆散 | 待领 | | |
| 5.26 | 布局收尾 | 待领 | | |
| 6.C01 | `.item()` 对无符号 dtype | 待领 | | |
| 6.C02 | `PySlice_Unpack` 返回值检查，三个变量初始化 | 待领 | | |
| 6.C03 | 整数提升 | 并入 2.16 | | |
| 6.C04 | 含 `void` 语句被删 | 并入 3.09 | | |
| 6.C05 | 融合边号 ≥256 回绕 | 待领 | | |
| 6.C06 | `grad.cc:65-68` 判空对象改为 `dx` | 待领 | | |
| 6.C07 | 缺失梯度默认报错 | 待领 | | |
| 6.C08 | `grad.cc:146-261` 两趟遍历合一趟并快照结构，删无边界游标 | 待领 | | |
| 6.C09 | `backward()` 可重复 | 待领 | | |
| 6.C10 | CUDA 分配钩子两张 map 用 `find` 加显式错误，释放后 `erase` | 待领 | | |
| 6.C11 | CPU 分配失败抛异常，返回值必须检查 | 待领 | | |
| 6.C12 | `cuda_device_allocator.cc:32-37` 的 managed 回退放到 … | 待领 | | |
| 6.C13 | 零字节分配不返回伪指针 `0x10` | 待领 | | |
| 6.C14 | SFRL | 待领 | | |
| 6.C15 | `migrate_to_cpu/gpu` 迁移前检查 share_with 关系，整组迁移或拒绝 | 待领 | | |
| 6.C16 | fetch 跨流 | 待领 | | |
| 6.C17 | `TempAllocator` 删遮蔽基类的 `used_memory`/`unused_mem… | 待领 | | |
| 6.C18 | CachingBlock 保存底层 allocation 并原样回传，不再传 0 | 待领 | | |
| 6.C19 | 每个分配器一把锁并覆盖 `gc()` | 待领 | | |
| 6.C20 | swap | 待领 | | |
| 6.C21 | 检查 `NODE_MEMCHECK` 外 `check_graph` 静默空转 | 待领 | | |
| 6.C22 | pyjt 关键字参数 | 待领 | | |
| 6.C23 | `is_type<NanoString>` 收窄 | 待领 | | |
| 6.C24 | 带实例 `__dict__` 的类型加 `Py_TPFLAGS_HAVE_GC` 与 trave… | 待领 | | |
| 6.C25 | 生成绑定补 `catch (...)` | 待领 | | |
| 6.C26 | `pyjt_compiler.py` 的 C++ 解析 | 待领 | | |
| 6.C27 | `Var.data` 返回的 numpy 视图 base 指向包裹该次 allocation 的… | 待领 | | |
| 6.C28 | 生成带「已构造」标志的 `tp_new` 或 `tp_dealloc` 先检查 | 待领 | | |
| 6.C29 | 标量转数组的全局 `tmp_data` 改自带 buffer | 待领 | | |
| 6.C30 | `helper_cuda.h` 的 `peek` 去掉进程级闩 `peek_logged` | 待领 | | |
| 6.P01 | 转置标记陈旧 | 并入 5.03 | | |
| 6.P02 | Function 实例复用、no_grad 泄漏、tied weight 参数集合 | 并入 5.07、5.08、5.04 | | |
| 6.P03 | H1 分组 conv3d 的 ww reindex 形状顺序 | 待领 | | |
| 6.P04 | H2 Pool3d `return_indices` 内核第三层循环变量 | 待领 | | |
| 6.P05 | H3 Pool3d CUDA 反向用 `pout_shape` 作上界 | 待领 | | |
| 6.P06 | H4 MaxUnpool2d/3d 在 `stride != kernel_size` 时用原始… | 待领 | | |
| 6.P07 | H5 eigh 反向 `dout` 全零时写零 | 待领 | | |
| 6.P08 | H6 `_autograd_grad` 的 zip 用过滤后的 `new_grad_output… | 待领 | | |
| 6.P09 | H7 irfft 对实数输入与显式 `n` 的处理走 `:68-73` 的判别函数 | 待领 | | |
| 6.P10 | H8 ReduceLROnPlateau 每轮从初始 lr 计算 | 待领 | | |
| 6.P11 | H9 `unique(return_counts=True, return_inverse=Fa… | 待领 | | |
| 6.P12 | H10 Adan 的 `clip_grad_norm` 移出 param_group 循环 | 待领 | | |
| 6.P13 | H11 `zero_grad` 清缓冲而非只翻标志 | 待领 | | |
| 6.P14 | H12 Adam 偏差修正用每 param 的步数 | 待领 | | |
| 6.P15 | H13 worker 异常不再变成给父进程发 SIGINT | 待领 | | |
| 6.P16 | H14 `mp_log_v` 做 int 转换 | 待领 | | |
| 6.P17 | H15 Pillow 版本用元组比较 | 待领 | | |
| 6.P18 | H16 `Dataset.__deepcopy__` memo 存对象不存 id | 待领 | | |
| 6.P19 | H17 `LogitRelaxedBernoulli` 返回 logit | 待领 | | |
| 6.P20 | H18 `ComplexNumber.__rsub__` 虚部符号、`__imatmul__` … | 待领 | | |
| 6.P21 | H19 稀疏卷积重复坐标 CPU/CUDA 语义统一 | 待领 | | |
| 6.P22 | H20 `to_dense` 对 COO 重复索引求和 | 待领 | | |
| 6.B01 | MPI 的 int64 改 `MPI_INT64_T` | 待领 | | |
| 6.B02 | ACL | 待领 | | |
| 6.B03 | HCCL 宏错误时抛而非 return | 待领 | | |
| 6.B04 | 分布式一旦被请求，初始化失败硬失败 | 待领 | | |
| 6.B05 | cuBLAS `use_tensorcore` 三目判断写反 | 待领 | | |
| 6.B06 | `var_broadcast` 用传入的 root | 待领 | | |
| 6.B07 | cuDNN RNN | 待领 | | |
| 6.B08 | cuSPARSE | 待领 | | |
| 6.B09 | curand 奇数长度用临时 buffer 不越界写 | 待领 | | |
| 6.B10 | MPI fp16 归约统一标量参考实现加可选 SIMD 与运行期 CPUID 检测 | 待领 | | |
| 6.B11 | ACL 六个算子静默把输入升到 fp32 | 待领 | | |
| 6.B12 | `cutt_transpose_op.cc:77` 的 `cudaGetLastError()`… | 待领 | | |
| 6.B13 | cuFFT `cufftCreate` 后被 `cufftPlanMany` 覆盖的句柄泄漏 | 待领 | | |
| 6.B14 | conv3d 三算子迁到 backend plan 缓存 | 待领 | | |
| 6.B15 | MPI 同时识别 PMI_/SLURM_ 环境变量或要求显式声明 | 待领 | | |
| 6.B16 | `sync_run` 在 ACL 上实现或删 flag | 待领 | | |
| 6.B17 | 析构不得抛 | 待领 | | |
| 7.01 | 「看起来支持其实空操作」一律改为实现或抛 `NotImplementedError`，需显式 `… | 待领 | | |
| 7.02 | DDP 真实梯度同步 | 待领 | | |
| 7.03 | 每个 torch API 一个模块级一等对象加保真度标注 | 待领 | | |
| 7.04 | 激活显式、一次性、可查询 | 待领 | | |
| 7.05 | install 事务化 | 待领 | | |
| 7.06 | 依赖单向化 core→tensor→nn/optim→distributed→fsdp→适配器 | 待领 | | |
| 7.07 | 第三方库补丁搬出 compat/ | 待领 | | |
| 7.08 | `torch.dtype` 改真正的对象而非 str 子类 | 待领 | | |
| 7.09 | `torch.library` | 待领 | | |
| 7.10 | `torch.compile`/`jit.trace`/`jit.script` 保留 pass… | 待领 | | |
| 7.11 | autograd 语义 | 待领 | | |
| 7.12 | 独立 torch 包 | 待领 | | |
| 7.13 | FSDP2 | 待领 | | |
| 7.14 | vLLM 边界检查把 `torch` 视作 jittor 别名 | 待领 | | |
| 7.15 | `_rebuild_tensor_v2` 按 stride 还原或报错 | 待领 | | |
| 7.16 | compat/ 内 129 个 `except: pass` 与 258 个宽泛 except … | 待领 | | |
| 7.17 | `runtime.enable()` 只把 shim 的 site 目录加进 sys.path … | 待领 | | |
| 7.18 | 布局收尾 | 待领 | | |
| 8.01 | 描述符与 workspace 一律 RAII | 待领 | | |
| 8.02 | 集合通信走通信流加事件依赖，支持 `GroupStart/End` 桶化 | 待领 | | |
| 8.03 | 精度策略收敛 | 待领 | | |
| 8.04 | cuDNN 9 | 待领 | | |
| 8.05 | MKL | 待领 | | |
| 8.06 | ACL 去样板 | 待领 | | |
| 8.07 | conv 族共享描述符与计划层 | 待领 | | |
| 8.08 | `ProcessGroup` 对象替代全局唯一 communicator | 待领 | | |
| 8.09 | NCCL | 待领 | | |
| 8.10 | `distributed/launch.py:102-107` 改 `wait(timeout)… | 待领 | | |
| 8.11 | `nccl_reduce`/`mpi_reduce` 非 root 不分配输出 | 待领 | | |
| 8.12 | 算子内不再复用全局 jit key 缓冲做缓存键 | 待领 | | |
| 8.13 | cuTT 计划未命中时的 `cudaDeviceSynchronize` 删除或降流同步 | 待领 | | |
| 8.14 | Corex | 待领 | | |
| 8.15 | 多机 rendezvous | 待领 | | |
| 8.16 | 多机启动器 | 待领 | | |
| 8.17 | 跨机网络与诊断 | 待领 | | |
| 8.18 | 多机 checkpoint | 待领 | | |
| 8.19 | 布局收尾 | 待领 | | |
| 9.01 | `import jittor` 不编译不下载 | 待领 | | |
| 9.02 | `install_cuda.py:113-122` 的 `os.execl` 自重启删除，用 d… | 待领 | | |
| 9.03 | 构建期失败一律抛带上下文的 `RuntimeError`，不用 LOGf/裸 assert | 待领 | | |
| 9.04 | 依赖跟踪改用编译器的 `-MD -MF` | 待领 | | |
| 9.05 | 下载安全 | 待领 | | |
| 9.06 | 删 cutlass 下载 | 待领 | | |
| 9.07 | import 过程不反向写环境变量 | 待领 | | |
| 9.08 | 新架 GPU | 待领 | | |
| 9.09 | `cuda_wheel` 失败时 LOG.w 出原因，strict 为默认 | 待领 | | |
| 9.10 | 2.0 版本策略 | 待领 | | |
| 9.11 | release 的 platform-validation 阶段跑 selftest | 待领 | | |
| 9.12 | `extern/rocm/rocm_cache.tar.gz` 的预编译 .o 改从源码构建，或… | 待领 | | |
| 9.13 | README 加「首次运行会发生什么」 | 待领 | | |
| 9.14 | 一次性的构建前置条件检查 | 待领 | | |
| 9.15 | noxfile | 待领 | | |
| 9.16 | `agent/scripts/check_repo_layout.sh` 收缩为少数真会复发的检… | 待领 | | |
| 9.17 | 死代码 | 待领 | | |
| 9.18 | `disable_lock=1` 启用时明确告警并纳入缓存指纹 | 待领 | | |
| 9.19 | 布局收尾 | 待领 | | |
| 10.01 | `tools/run_test_suite.py` 拆成 `nox -s full` 周期性调度… | 待领 | | |
| 10.02 | 默认 `nox` 含 cpu 数值测试，或把默认改名为 static | 待领 | | |
| 10.03 | optional/rocm/mpi/nccl 四个 session 排上 runner 或在文档… | 待领 | | |
| 10.04 | 假绿清理 | 待领 | | |
| 10.05 | 按 skip 原因分桶统计并在 CI summary 输出，对「本环境应能跑却 skip」设阈值 | 待领 | | |
| 10.06 | `expect_error` 带 `exc_type` 与 `match` | 待领 | | |
| 10.07 | Unary/Binary/Reduction 用 `OpDTypes.supported` | 待领 | | |
| 10.08 | 已复现缺陷用 `xfail` 而非 `skip` | 待领 | | |
| 10.09 | 公开 API 与 OpInfo 差集作为 structure 门禁一项 | 待领 | | |
| 10.10 | gradcheck 加「故意写错导数应当失败」的负向自测 | 待领 | | |
| 10.11 | 设备对拍加 dtype 轴 | 待领 | | |
| 10.12 | `retry` 装饰器记录并上报重试次数 | 待领 | | |
| 10.13 | marker 真正建立 `-m "not slow"` 快门禁或删除 | 待领 | | |
| 10.14 | notebook 门禁按 topic 参数化 | 待领 | | |
| 10.15 | 速度 harness 记录并断言两侧线程数、亲和掩码与精度策略 | 待领 | | |
| 10.16 | 提供计时 API | 待领 | | |
| 10.17 | 异步错误 | 待领 | | |
| 10.18 | 结构测试预算转向核心 | 待领 | | |
| 10.19 | 每个带 `grad()` 的后端算子有对 CPU 参考的梯度单测 | 待领 | | |
| 10.20 | 给测试提供受支持的内省 API，替代 283 处 `jt.flags.*`、137 处 `com… | 待领 | | |
| 10.21 | import 方向做成 lint 规则 | 待领 | | |
| 10.22 | 多机门禁 | 待领 | | |
| 10.23 | 布局收尾 | 待领 | | |
| 11.01 | 删已被取代的绕过与死路径 | 待领 | | |
| 11.02 | 已提前为 0.20 | 并入 0.20 | | |
| 11.03 | 单文件异常拆分 | 待领 | | |
| 11.04 | 关键接口写成显式契约 | 待领 | | |
