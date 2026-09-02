# 核心 C++ 运行时（src/）

`python/jittor/src/` 46605 行，89 个 DEFINE_FLAG，486 处 ASSERT/CHECK，62 处 LOGf。

**四条根本问题**。第一，`Node` 是所有子系统共用的可变涂鸦板：一个 32 位 flags 字（Var 与
Op 两套语义叠在同一批位上且已真的撞位）、一个无类型 `custom_data` 整数、一个全局单调
`tflag` 时间戳，被图遍历、融合划分、拓扑排序、自动微分轮流覆写，谁在什么阶段能读写
哪个字段只写在注释里。第二，本该是结构化数据的东西全是文本：jit key 是 2 MB 线程局部
字符缓冲（溢出靠 guard page 段错误发现）、KernelIR 用 `string type` 加 `map<string,string>`
表示 IR、算子身份用 `name()` 字符串比较（25 处含正确性判据）、Python 绑定用正则解析
C++ 头文件。第三，错误处理只有一档：ASSERT/CHECK/LOGf 全部抛 runtime_error，出现在析构、
信号处理器和编译工作线程里；而真正该报错的地方（缺失梯度、溢出 dtype、被吞掉的关键字
参数）反而静默返回错值。第四也是最要命的：**这一层最微妙的部分不在源码树里**。

## 可审计性：核心的一部分不在源码树里
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 5 个翻译单元只以混淆形式存在于二进制资源里 | `fuser.h`/`node.h`/`opt/pass/{atomic_tuner,shared_reduce,parallel}_pass.h` 有头无 .cc；`compiler.py:1402-1429` 把 `utils/data.gz`（437 KB）解压成 data.cc、编译、然后 `os.remove(data_s_path)`。**已亲自核实**：解压后 569 行 1.5 MB，标识符全部替换成 `x10364` 形式，字符串十六进制转义，宏改名，并插入 `_P(...)` 噪声行由 `src/utils/vdp`（内容仅 `#define _P(...)`）消除；`#include` 行未混淆，含 node.h/fuser.h/opt/pass/*。编译出的 `data.o` 导出 31 个符号，含 `jittor::count_fuse`、`jittor::Node::free`、`own_forward_liveness`。最近改动是提交 fe74d1f5「polish computing graph liveness」。`use_data_gz=0` 的回退分支指向 `__data__` 路径，仓库里不存在 | `Node::free`、own/release_{forward,backward,pending}_liveness、count_fuse 全部不可读不可 diff 不可单测。「三套 liveness 计数」与融合策略本身无法审计；「图不变量只存在于代码里」在这里升级为「不在能读的代码里」 | 把这 5 个文件还原进源码树。任何 2.0 重构（多设备、执行计划缓存、流水化）都要改 liveness 与融合，绕不开 | 关键 |

## 节点模型：一个字段承担多种含义
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| **`_th_require_grad` 与 `_is_scalar` 是同一个位** | `node.h:47-48` 两者都等于 11。都设在 Var 上：`var_holder.cc:210`（start_grad）、`:157-158`（assign_var）设前者；`ops/array_op.cc:59`、`reduce_op.cc:338`、`unary_op.cc:893`、`broadcast_to_op.cc:169`、`pyjt/py_array_op.cc:158` 设后者。读取点 `binary_op.cc:472,479`、`ternary_op.cc:47` | `p.requires_grad_(True)`（shim 里遍布 optimizers 与 fsdp2）使每个参数在 binary_dtype_infer 里被当成标量：`nano_string.h:263` 让 f64 参数被降到 f32；`:271` 走 has_scalar 分支**直接跳过 amp_prefer32/16 覆盖**，即 **AMP 在所有涉及参数的算子上静默失效**；`binary_op.cc:477-496` 还给每个碰参数的 f32 加减挂上 `-O3` loop option，改变 jit key 并丢掉 -Ofast。反方向：`jt.array(1.0)` 设了 _is_scalar，于是被当 torch 叶子标记传播 | 位号由枚举连续生成并加 static_assert；Var 与 Op 的 flag 拆成两个类型 | 关键 |
| AmpGradGuard 把 flags 高位当 amp 位灌进全局 | `grad.cc:51` `amp_reg \|= (op->flags.flags >> NodeFlags::_prefer_32)` 右移 16 位无掩码；而 `op.cc:56` 入口是有掩码的 | bit 22-25（_custom_flag、_requires_grad_disabled、_requires_grad_snapshot、_first_order_only）被移进 amp_reg 的 bit 6-9。当前 amp 只用 bit 0-5 故无害；新增第 6 个 amp 位时任何 getitem/setitem 都会在反向里伪造该位 | 出入口用同一掩码 | 主要 |
| custom_data 一个 int 被至少五套算法轮流覆写含位打包 | `node.h:148`；`executor.cc:269,273,537`；`fused_op.cc:81-84` 注释说明同一字段 bit0=不可融合、bit1=已访问、bit≥2=var 下标；`graph.h:115,143`；`grad.cc:127`；`memory_profiler.cc:88-94` 不得不备份恢复它 | 任意两个遍历交错就互相破坏。`grad.cc:120` 刚用它存完入度，`:124-129` 立刻改写成 gvars 下标，正确性依赖两段之间不能插入任何其他遍历 | 遍历用局部 vector 按 node id 索引；删除 custom_data | 主要 |
| tflag 是全局计数器加魔数，函数之间靠读全局值握手 | `node.h:88`；`grad.cc:105,121` 读上一个 bfs_forward 用掉的值；`:161` `op->tflag = 0` 当"别再反向"标记；`graph.cc:77` 用 `!= -1` | 调用顺序即契约且不可嵌套。`op.cc:135` 的重入 run_sync 会推进 tflag_count，任何持有旧值的外层遍历随即失效 | 遍历标记用局部集合或 epoch 对象 | 主要 |
| Var::allocator 字段被类型双关成 Var* | `var.h:48` share_with 把 Var* 存进 Allocator*；`var.cc:118-124` 再还原。调用点 12 处。同一 allocation 字段此时存的是**字节偏移**，alloc 之后又变成 sfrl 的 **block id** | share_with 与 alloc 之间任何读 `var->allocator->is_cuda()` 的路径（`executor.cc:594,604`、`var_holder.cc:255,300`）都会对一个 Var 发虚函数调用；目前靠不成文不变量躲开 | 加独立的 share_src 与 share_offset 字段 | 主要 |
| 没有 0 维张量，标量由 shape==(1,) 冒充 | `var.cc:107` `if (shape.size()==0) shape.push_back(1);`；`array_op.cc:56-59` 只按 shape==(1,) 判定 _is_scalar | `np.array([2.0],f64) * f32_tensor` 被按标量提升成 f32 而 NumPy/PyTorch 给 f64；`var_holder.h:317-319` 与 `var_holder.cc:260-262` 两处注释掉的 "state_dict only has one element / save wrong" 是同一问题的疤 | Var 需要真正的 0 维 shape | 主要 |
| 边表是 list 加反向迭代器，按下标访问是 O(i) | `node.h:150-151`、`:162-172` 的 `while (i--) iter++`；`setitem_op.cc:61,170,179,204` 按下标取输入 | 每条边两次 malloc；UNet 一步约 1500 kernel 数千 var，仅边表就是几万次分配 | 改 SmallVector，索引 O(1) | 次要 |
| hold_vars 全局链表加 sync_ptr 全局迭代器，析构里 `std::next(end())` | `var_holder.cc:30-31`、`:141-142` 与 `:115-116`；`release_from_holders()` 在 `:121` 把 iter 设成 end()，随后析构再执行同一句 | `iter==end() && sync_ptr==end()` 时是 UB（libstdc++ 绕回 begin()），`top_weak_sync` 随即立刻 break，weak sync 静默不再工作 | sync_ptr 改序号或哨兵 | 次要 |

## 执行器、融合与并行编译
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 全局 kernel 缓存持有指向栈对象的指针 | `executor.cc:539` `FusedOp fused_op;` 是局部量；`fused_op.cc:223` 命中缓存时把全局表里的 context 重新指向这个栈对象。并行编译路径更远：`parallel_compiler.cc:194` 拷进 unique_ptr，`:265-266` 让 context 指向它，函数返回后该对象销毁而 context 留在全局表 | 编译完到首次执行之间 context->vrm.fop 悬垂；重入执行会让两个 FusedOp 争抢同一 context | context 不持有 fop；relay 信息编译时固化成 POD | 主要 |
| 算子构造期回调执行器 | `op.cc:127-137` 动态形状算子里 `exe.run_sync(...)`。Executor 有跨次运行的成员状态 | 建图期间同步执行整段图；嵌套调用覆写外层 last_is_cuda、推进 tflag_count、清空线程局部 jit key 缓冲 | 形状推断走主机侧计算；执行器提供显式提交接口 | 主要 |
| 计算 jit key 会**永久改写**算子的能力位 | `op.cc:196-197,215,228` 在 do_jit_prepare 里 `flags.set(_cuda,0)`/`set(_cpu,0)`；`fused_op.cc:170-176` 同。而 do_prepare 在错误路径会被再调一次（`executor.cc:685`、`parallel_compiler.cc:203,297`） | 求一个键是有副作用的：一个 CPU/CUDA 双支持算子在 use_cuda==0 时被 prepare 后 _cuda 位永久丢失，之后即使打开 use_cuda 也只走 CPU。这也是 `op.cc:316-320` 必须维护两个键的根因 | key 计算改纯函数；后端选择结果放执行计划不写回节点 | 主要 |
| 对外暴露的 CUDA 分配钩子用两张永不清理的全局 map 且缺失即当 0 | `executor.cc:721-733`：free 里 `size_map[ptr]`/`allocation_map[ptr]` 未命中时默认构造 0 然后按 size=0 释放；两张表**从不 erase** | 外部库（cupy 等）传入的任意指针会以 size 0 被释放，sfrl 的 erase_occupied(0) 拿到错误的块；每次分配泄漏两个 map 项 | 用 find 加显式错误；释放后 erase | 主要 |
| 并行编译器：非原子的 has_error、无锁的 error_msg、按插入序耦合的下标、超时即放行 | `parallel_compiler.cc:226-227` `static volatile int has_error; static string error_msg;`，`:308-309` 工作线程无锁写；`:369` `map.holder.at(...)` 依赖 operator[] 插入序与任务序完全一致而 string_view_map 无此契约；`:113-130` wait_all 用 try_lock 探测，5 秒后打印 "Compile thread timeout, ignored." 直接放行，而 func 是 `[&]` 捕获栈上对象 | 数据竞争加悬垂栈引用。另有挂死路径：catch 里再次 do_prepare（`:297`）若二次抛出，异常逃出 func 被 `SimpleThread::run` 吞掉后线程**永久退出**且 has_error 未置位；全部线程如此则主线程在 `:327` 上以 1 秒睡眠无限自旋 | has_error 用 atomic；error_msg 加锁；等待改 join/future | 主要 |
| 全局计数器被"按律恢复" | `parallel_compiler.cc:240,351` 编译前备份 lived_vars/lived_ops 编译后赋回；`fused_op.cc:122,140` 用 --/++ 让 FusedOp 不计入 | 这两个计数驱动 `var_holder.cc:74` 的急切执行阈值；恢复期间主线程若也在建图则计数永久错位 | 计数器改成分配器职责或删掉该启发式 | 次要 |
| 执行器全程持有 GIL | 全仓仅两处 GIL 操作且都是为编译线程；run_sync（含 `executor.cc:705` 的 cudaDeviceSynchronize）、.item()、.numpy() 路径无 Py_BEGIN_ALLOW_THREADS | 一次 sync 期间整个解释器停摆；use_threading flag 只关掉 top_weak_sync 并不解决。对 vLLM 这类需要后台调度线程的场景是硬串行点 | 在设备等待段释放 GIL | 主要 |

## JIT 键与代码生成：用文本代替结构化数据
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| jit key 是 2 MB 无边界检查的字符缓冲，溢出靠段错误发现 | `jit_key.h:16-18` `char buffer[2*1024*1024]`；所有 operator<<（`:129-198`）直接写 buffer 无容量检查。唯一防线是 `jit_key.cc:36-43` 对最后一页 mprotect PROT_NONE；命中后在 **SIGSEGV 处理器里**执行 LOGf（`utils/log.cc:304-308`），即在信号上下文里分配内存、写 cerr 并抛 C++ 异常。判定窗口写死 4 KB，16 KB 页平台上提示丢失。析构还原时给该页加上了 PROT_EXEC | 键溢出表现为进程崩溃而非可捕获错误；信号处理器本身是 UB。每个线程 2 MB TLS 加一次 mprotect | key 改结构化，或至少加长度检查并抛正常异常 | 关键 |
| 融合图的边在 jit key 里用 2 位十六进制编码算子号，≥256 静默别名 | `fused_op.cc:181` `jk << hex2(i) << hex1(j) << hex2(k) << hex1(l)`；`jit_key.h:153-166` hex1 取模 16、hex2 只输出低 8 位。而 i/k 可以是 `ops.size() + iv_id`（`executor.cc:86`，注释写着 prevent iv_id jit key overflow——只修了一个方向）。全仓无融合规模上限断言 | 含 100 个算子加 160 个外部输入 var 的融合段边号超过 255 后回绕；两个结构不同的融合算子映射到同一个键，find 命中，**执行错误的已编译 kernel 静默错结果** | 边编码用变长十六进制或结构化哈希 | 关键 |
| 三张全局 kernel 缓存表永不回收且 key 生存期靠巧合 | `op.h:66-68`、`fused_op.h:27`；`string_view_map`（`misc/string_view_map.h:30-54`）没有 erase，operator[] 把键 emplace_back 进 vector<string> 再取 string_view——**vector 扩容会移动这些 string，SSO 短串（≤15 字节）的数据随对象搬走，此前所有 string_view 键悬垂**。目前只因 jit key 都长于 15 字节而没炸；clang/ACL 下 string_view 直接 typedef 成 string，行为随编译器而变 | 动态形状负载下三张表无界增长；FusedOpContext 也从不释放 | 键直接用 string；缓存加容量上限与 LRU | 主要 |
| KernelIR 的节点类型和全部语义属性都是字符串，缺失键静默变空串 | `opt/kernel_ir.h:23,33`；`kernel_ir.cc:50-52` `get_attr` 是 `return attrs[s];`——拼错的属性名不报错，静默插入空串。pass 间契约是 14 个字符串字面量跨 13 个 pass 文件硬编码，唯一文档是头文件注释 | pass 之间的顺序与属性依赖没有任何机制保障；漏设一个属性表现为少一层优化或直接算错且不报错 | IR 节点类型用 enum，属性用带类型结构体，依赖显式声明由 pass manager 校验 | 主要 |
| 正确性 pass 在名字解析失败时静默跳过 | `opt/pass/float_atomic_fix_pass.cc:76-80` `catch (...) { return; }`；`fake_main_pass.cc:91-95` 同形 | 该 pass 是修浮点原子 max/min 正确性的；名字解析失败时它什么都不做，生成的 kernel 结果错误且无提示 | 解析失败必须是错误 | 主要 |
| 以 `_` 开头的 loop option 不进 jit key | `fused_op.cc:206` | 两个只在下划线选项上不同的配置共用同一份编译产物 | 全部入键或设置时拒绝 | 次要 |
| 生成的 kernel 源码里烧进 C++ 结构体字节偏移并被缓存 | `opt/var_relay.cc:189-193` 把 offset 写进生成源码，而这些偏移由 **`compiler.py:431-440` 用正则扫头文件**得出。生成结果长期复用，jit key 不含结构体布局 | 算子结构体加一个成员、换编译器 ABI 或某个 -D 改变条件成员，缓存里的 kernel 就往错误成员上写指针——静默内存破坏 | relay 走显式 setter；成员表用宏声明 | 主要 |
| 算子身份靠 name() 字符串比较 25 处，含正确性判据与 UB | `misc/nan_checker.cc:57-58`（决定跳不跳 NaN 检查）、`var_holder.cc:49,79`（决定不流水/不急切）、`op_compiler.cc:146,148,943,1116`、`loop_var_analyze_pass.cc:32,176,259,274,285`、`conv_tuner.cc:118-124,254,257`。`var_holder.cc:402-404` 的 fast_strcmp 直接比较 8 字节，在 `:414` 对 name() 使用——名字短于 8 字节时越界读 | 重命名一个算子会静默关掉某项优化或某个安全检查；fast_strcmp 是 UB | 算子身份用注册期整型 id | 主要 |
| 三个 pass 只有 .h 没有 .cc，且用正则在最终 C++ 文本上打补丁 | `op_compiler.cc:30-69` 用正则修的正是这几个无源码 pass 的语义 bug；`:914-933` 与 `:1074-1076` 的 opX_ 盲目前缀与硬编码白名单构成一条无人校验的隐含契约；`kernel_ir.cc:865-871` 的 check_unused 会删掉任何含 void 标识符的语句 | 格式一变补丁静默失效；清缓存后无法重新链接 | 随第一节的源码还原一并处理 | 关键 |

## 自动微分
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 对可能为空的 VarPtr 解引用（两处） | `grad.cc:65-68` 判空的是**输入** x，解引用的是**结果** dx。返回 nullptr 是常规路径：`op.cc:77-80`、`binary_op.cc:607`（mod/floor_divide/位运算）、`unary_op.cc:939,941,1060`（floor/round/ceil）、`reduce_op.cc:363`、`broadcast_to_op.cc:113`。另一处 `grad.cc:262` | 设了 compile_options 后任意 `jt.floor(x)` 进反向即段错误；amp level 3 下另一条同类崩溃 | 判空对象改成 dx/grad | 主要 |
| 缺失梯度只是一条按 var 名去重的全局警告 | `op.cc:78`；`grad.cc:76-82` 以 `v->name.c_str()` 为键的进程级 map，永不清空 | 无名 var 的名字都是空串，第一条警告之后**所有后续缺失梯度静默无声**，训练照常收敛到错误结果 | 缺失梯度默认应报错 | 主要 |
| 两趟遍历用一个无边界检查的游标同步 | `grad.cc:146-183` 与 `:187-261`；而 `n_o = op->outputs().size()`（`:198`）读的是第二趟时刻的出边数，第一趟写入时用的是当时的 | 两趟之间出边数一变（grad 会建新算子、Op::forward 会加输出）游标即错位，越界读 | 两趟合成一趟并快照结构 | 主要 |
| backward() 不可重复：反向会永久摧毁前向图的可导性 | `grad.cc:281-294` retain_graph=false 时对 gvars 调 set_stop_grad；而 `var_holder.cc:183-186` 注释明写 stop_grad 是 intentionally permanent | 第二次对同一张图反向不报错，只静默得到零梯度 | 图释放与停止求导分开 | 主要 |

## 类型系统
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 名字写入 16 字节槽位无边界检查 | `misc/nano_string.cc:221-227` 逐字节拷到 `__ns_to_string + index*16`（ns_max_len=16）。当前最长名 greater_equal 13 字节，只剩 2 字节余量 | 新增任何 ≥16 字符的算子或 dtype 名会静默覆写下一条目的名字 | 加 ASSERT | 主要 |
| 索引只有 7 位（128 项），当前已用 71，无越界检查 | `nano_string.h:107`；FOR_ALL_NS 71 项；`nano_string.cc:194` 的 set 自带掩码超过 128 静默回绕；表却按 256 开 | 加到第 129 个类型时两个不同的 NanoString 变得相等，to_cstring 返回错名 | 位宽加到 8 并加注册期断言 | 主要 |
| dtype 集合是编译期宏，外部无法注册 | `nano_string.h:15-92` 的 FOR_ALL_NS；`nano_string.cc:36-44` 的属性 map 在 init_ns 里打补丁 | 后端要加 fp8/int4 必须改核心头文件重编全树；与后端注册表方向冲突 | dtype 表改运行期注册 | 主要 |
| **.item() 对无符号 dtype 返回未初始化的高位字节** | `var_holder.cc:284` `ItemData data;`（POD 不初始化）；`:305` memcpy 只写 dsize 字节；`pyjt/py_converter.h:496-513` 只列了有符号与浮点，uint8/16/32/64 全部落到 `:513` 的 PyLong_FromLongLong（读满 8 字节） | `jt.array(np.uint8([200])).item()` 返回随机大整数，**静默错值不报异常**。图像、mask、量化权重是重灾区 | `ItemData data{};` 值初始化并补齐无符号分支 | 关键 |
| ROCm 上 bf16 的 .item() 只改 dtype 不做转换 | `var_holder.cc:272-278` 的 bf16 分支被 `#ifndef IS_ROCM` 包住，而 `:279` 的 `data.dtype = ns_float32;` 在 ifndef 之外无条件执行 | ROCm 上 bf16 标量读出来是位模式被当作 f32 解释，静默错值 | 转换与 dtype 改写放同一分支 | 次要 |

## 内存与分配器
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| CPU 分配失败无人检查 | `aligned_allocator.cc:16-28` 不判空；`var.cc:126-128` 当 bool 返回；`executor.cc:576-578` **丢弃返回值**。唯一断言在 `#ifdef NODE_MEMCHECK` 内 | CPU OOM 表现为 kernel 写空指针段错误而非 OOM 错误。另：aligned_alloc 要求 size 是 32 的倍数而 Var::size 一般不是，glibc 容忍其他 libc 是 UB | alloc 失败抛异常；返回值必须检查 | 主要 |
| OOM 回退分支在 LOGf 之后，永不可达 | `cuda_device_allocator.cc:32-37`：LOGf 之后才是 cudaMallocManaged，而 LOGf 会抛异常 | 打开 managed_fallback 并不会回退到统一内存，只会换一句错误信息 | LOGf 改 LOGw | 主要 |
| 零字节分配返回伪指针 0x10，释放按 size 判断 | `cuda_device_allocator.cc:25,41` | 零元素 Var 的 mem_ptr 非空看起来已分配；形状若在 alloc 与 free 之间被改过，伪指针会被送进 cudaFree 或真指针被泄漏 | 用真正的空分配语义 | 主要 |
| 块 id 空间是全进程静态单例，2M 上限，索引前不查范围 | `sfrl_allocator.h:35-38` 静态 16 MB 指针数组；`sfrl_allocator.cc:81-86` 先索引再断言非空 | CPU/CUDA device/host/dual/temp 所有分配器共用一个 id 空间——多设备必须重做这一层。Var::allocation 只在配上正确的 Var::allocator 时才有意义而二者是否匹配无处校验；越界 allocation（例如 share_with 遗留的字节偏移）先越界读再断言 | id 表随分配器实例走；索引前断言 | 主要 |
| fork 之后子进程的编译线程池是幽灵 | `parallel_compiler.cc:107-112` create_threads 有 `if (threads.size()) return;` 守卫；`init.cc:102-109` 的 jt_init_subprocess 只重置三样东西，不碰线程池、hold_vars、jit_ops、tflag_count | DataLoader worker 里线程对象存在但线程不存在，wait_all 对每个线程各等 5 秒后放行，这批算子退回串行编译。表现为 worker 首次编译长时间停顿 | fork 后重建或禁用线程池 | 主要 |
| swap 文件名用静态初始化的 pid，且 D2H 拷贝不查错 | `mem/swap.cc:26` 静态初始化的 pid（fork 后仍是父进程的）；`:40` 用它拼文件名；`:50` cudaMemcpy 返回值丢弃；`:46` 函数内 `static char* buffer = new char[8MB]`（非线程安全且泄漏） | 开 save_mem 时父子进程写同名 swap 文件（var id 也因 total_node 继承而重合）互相覆盖，静默错数据 | 文件名用运行期 pid 加唯一 token | 主要 |
| swap 的设计说明就是一张未完成的 TODO | `mem/swap.h:37-58` 注释列着 handle cutt / handle cupy / search share_with / search migrate / disable dual allocator / handle foreign allocator。而 `var.cc:37-38,51-53` 的**每一次** Var 释放都无条件先判 save_mem | 一个自认未完成的特性接在最热的释放路径上 | 完成前用编译开关隔离 | 主要 |

## 绑定层：生成器与 CPython 协议
共同特征是**生成器不报错、生成的 C++ 能编译、行为悄悄不对**。

| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 特定签名下关键字参数被**静默丢弃** | `pyjt_compiler.py:106-109` 只有非 VarHolder* 参数进 kw_args_id；`:111-113` 该表为空时长度检查退化成只看 n，完全不读 kw；`:166` 的 fill_with_kw 也不生成。生成结果 `gen/pyjt_jit_op_maker.cc:1537-1551`（detach）：`if (n<=0 && n>=0) { … }` | `v.detach(non_blocking=True)`、`x.to(memory_format=...)` 之类静默成功并走默认语义。对 torch/vLLM 兼容层是持续的静默错误来源 | 长度检查计入 kw；未消费的 kwname 报错 | 主要 |
| 关键字路径的转换错误检查排在转换之前 | `pyjt_compiler.py:193-196` 把检查追加在 func_args_convert 末尾，而 `:219-223`、`:714-717` 的展开顺序使检查排在 `PyLong_AsLong` 之前，之后再无检查；底层转换本身全裸 | `x.sum(dim=2**40)` 溢出返回 -1 并置 OverflowError 被完全忽略，沿 dim=-1 归约，**结果静默错误**；遗留异常稍后在无关位置爆出。位置参数路径有检查、关键字路径没有 | 检查移到关键字填充之后 | 主要 |
| 重载选择按**位置**探测 args[tid]，关键字顺序会改变重载结果 | `pyjt_compiler.py:150`；FASTCALL 下 `n < tid+1` 时 args[tid] 实际是某个关键字的值 | `x.sum(dim=1, keepdims=True)` 正常；`x.sum(keepdims=True, dim=1)` 抛 "Not a valid keyword: dim"。`**kwargs` 转发下是随机失败 | 先按 kwname 映射槽位再做类型检查 | 主要 |
| `is_type<NanoString>` 近乎万能匹配并吞掉 __getattr__ 异常 | `py_converter.h:238-250`：字符串、类型、可调用对象、或带 .type 属性的对象都算 | 任何函数或类抢先匹配 NanoString 重载然后在转换里 ASSERT，而 matched_overload 已在转换**之前**置真，错误信息指向算子而非参数类型。PyTorch 的 `Tensor.type()` 是标准 API，shim 一旦补上就会引爆 | 收窄 is_type；matched_overload 移到转换成功之后 | 主要 |
| PySlice_Unpack 返回值被丢弃，用未初始化栈值构造 Slice | `py_converter.h:158-167` 三个变量未初始化，返回值丢弃；CPython 在 step==0 或溢出时 return -1 且不写这三个值 | `a[::0]`、`a[::2**70]` 把栈垃圾当切片边界送进 getitem/setitem（越界读写风险） | 检查返回值 | 关键 |
| 带实例 __dict__ 的类型不参与 GC | `pyjt_compiler.py:874` 设了 tp_dictoffset，`:876` `tp_flags = Py_TPFLAGS_DEFAULT` **无 HAVE_GC**，也无 traverse/clear | `v.foo = v`，或 shim 里 `t.grad`/`t._base` 这类反向指针形成的环**永不回收**，连带整张计算图与显存 | 加 GC 标志与 traverse/clear | 主要 |
| 生成绑定只有 `catch (const std::exception&)` | `pyjt_compiler.py:723`；生成结果 325 处 catch exception、0 处 `catch (...)`。而 `pyjt/pyjt_console.h:531,533` 自己 `throw new std::runtime_error(...)`（抛的是**指针**不匹配） | 非 std::exception 派生的异常穿过 extern "C" 边界即 terminate，Python 侧连 traceback 都没有 | 补 catch(...)；throw new 改 throw | 主要 |
| 生成器的 C++ 解析是字符扫描，多种合法写法静默生成错代码 | `pyjt_compiler.py:72-86` split_args 只数尖括号不数圆括号，且 `>` 会让 presum 变 -1 使**所有后续参数不再切分**；`:321-338` find_bc 不跳过字符串与注释；`:400` 遇到第二个 = 崩在构建期；`:389` 让 `VarHolder *foo(...)` 变成 `*foo`；`:215-218` 把 __getitem__ 的类型检查与转换整段清空 | 当前头文件恰好都躲过了，任何新增绑定都在雷区里 | 换 libclang 或至少加括号计数与断言 | 主要 |
| Var.data 返回的 numpy 视图只钉住 Python 包装对象 | `var_holder.h:310-321` 返回 `{this, var->mem_ptr, …}`；`py_converter.h:471-488` 用 SetBaseObject 指向 VarHolder 的 PyObject。而 `var_holder.cc:219-229` 的 assign 会释放旧 var 换新 var | `a = v.data; v.assign(other); a[0]` 读已释放内存 | base 指向包裹该次 allocation 的胶囊 | 关键 |

## 错误处理与失败模式
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 错误只有一档：486 处断言与 62 处 LOGf 全部抛异常，包括析构与信号处理器 | `utils/log.h:152-174`；出现在信号处理器（`utils/log.cc:307`）、生成的 tp_dealloc（异常时跳过 tp_free 并泄漏实例字典）、编译工作线程 | 参数写错与内部状态损坏用同一种机制上报；析构里抛异常导致对象泄漏并覆盖正在传播的 Python 异常 | 分成用户错误与内部不变量两条通路；析构不得抛 | 主要 |
| 执行失败的处理路径本身会二次抛出 | `executor.cc:681-693` catch 里再调 do_prepare，而它正是形状与类型错误的抛出点且有副作用；`parallel_compiler.cc:203,297` 同形 | 原始异常若来自 do_prepare，catch 内会再次抛出：executor 里异常逃出 catch，编译线程里被吞掉后线程退出 | 错误路径缓存首次算出的 key | 主要 |
| check_graph 在 release 构建里静默不检查 | `graph.cc:74-82` 遍历 lived_nodes，而它只在 `#ifdef NODE_MEMCHECK` 下填充 | 打开 check_graph=1 得到虚假的安全感；三套 liveness 唯一的一致性校验在正式构建里是空转 | 校验逻辑与 memcheck 解耦 | 次要 |
| 信号处理器不是 async-signal-safe 且检查窗口写死 | `utils/log.cc:268-322` 内有 cerr、dladdr、fork addr2line、LOGf，最后 exit(1) 触发 atexit 与静态析构而其他线程仍在跑。`parallel_compiler.cc:246,353` 读的 segfault_happen 在 exit 前一行才置位 | 崩溃诊断本身可能二次崩溃或挂死；协作退出路径是死代码 | 处理器内只做 write 与 _exit | 次要 |
| 已知损坏的机制留在代码里 | `executor.cc:704-707` 把 event_queue.run_sync 注释掉写着 "TODO: run_sync cause hang"；`event_queue.h:26` 用 volatile 当同步原语。`ops/tape_op.cc:38-44` 注释 "this is still not enough… please find a better solution" | 异步执行的基础设施存在但不可用；执行器异步化没有可复用的底座 | 修好并加测试或删除 | 次要 |
| 算子注册表的键不对称且 73 处依赖静态初始化顺序 | `ops/op_register.cc:34` 按 op_info.name 存，而 `:15,38,43` 按截断到第一个点之前的名字查——注册带点的名字后永远查不到。`op_register.h:20-28` 用 RTTI 在 void* 上分派。全仓 73 处命名空间作用域的 `static auto make_xxx = get_op_info(...)` 而注册本身也是静态初始化 | 两组静态初始化的相对顺序未定义；若查询先于注册，ASSERT 在 main 之前抛出即 terminate 且无诊断。跨 .so 的 type_info 比较也可能静默不匹配 | 注册表改惰性初始化；查询延迟到首次使用；构造函数签名用编译期校验 | 主要 |

## 补充：代码生成与优化 pass（文本当作 IR）

第二轮审计在同一层补出的发现，均已单独核实其中三条。

| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 死代码消除按"语句里出现 void 这个词"整条删除 | `opt/kernel_ir.cc:865-871`：取出标识符后 `if (var=="void") { if (type=="") { code=""; break; } }`。本意只删 `(void)count;`（`reduce_op.cc:405`、`transpose_op.cc:123` 依赖此行为）；触发点 `remove_intermediate_pass.cc:39,45`、`merge_loop_var_pass.cc:151`。**已核实** | `memset((void*)zp,0,n);` 这类语句从融合 kernel 里凭空消失，能编译通过，结果静默错误 | 解析时结构化识别 `(void)expr;` 形态并打属性，不在文本里搜关键字 | 关键 |
| 算子内可用的标识符是隐含白名单，违反即生成非法 C++ | `op_compiler.cc:914-933` 三个硬编码集合：`members{x,y,z,cond,output,extras}`、`scalar_members{left,right}`、`unchanged{for,const,auto,int,float,bool,CHECK,void,if,true,false,Op,Var,Node,...}`；`:1074-1076` 其余标识符一律加 `op{i}_` 前缀 | `size_t`、`int64`、`uint`、`nullptr`、`return`、`else`、`while`、`static` 都不在白名单，会变成 `op0_return`。任何人给含这些写法的算子加 `set_type(OpType::element)` 就会得到与真实原因完全脱节的编译错误 | 用结构化成员表做重命名，改名前做合法性校验 | 关键 |
| 用 `std::regex` 在最终 C++ 文本上打补丁，修一个没有源码的 pass 的语义 bug | `op_compiler.cc:30-69` 的 `fix_parallel_thread_ranges` 匹配 `^(\s*)int (tn[0-9]+) = (get_thread_range_log\(thread_num_left, [^;]+\));\s*$` 并重写成累加形式；触发条件是 `:1156` 的子串嗅探；产生方 `ParallelPass` 无源码（见第一节） | CPU 线程划分的正确性依赖另一个不可读 pass 输出的确切文本格式（空格、变量名、行尾分号）。格式一变即静默不匹配；单测 `tests/compiler/test_parallel_pass.py:124-129` 断言的正是打完补丁之后的文本 | 在 IR 层修正累积逻辑 | 关键 |
| 两个 pass 注册同一个名字，`get_pass` 用 C 风格下转型 | `opt/pass/unroll_pass.h:13` 与 `expand_empty_block_pass.h:13` 都是 `Pass("expand_empty_block")`；`pass_manager.h:54` 的 emplace 不覆盖，`:62` `return (T*)iter->second;` 无校验 | `exclude_pass="expand_empty_block"` 同时关掉两个且无法单独关 unroll；一旦有人按 UnrollPass 取就是无声的类型混淆 | 改名；pass 按类型索引 | 主要 |
| 循环维度的身份用名字字符串表达，`range10` 二义 | `merge_loop_var_pass.cc:22-24` 用 `str.size()==6` 判断"是单个 range"；`:74-82` 逐字符拆分把 `range_b` 展开成 `range1*range0`；`:128` 新 id 是字符串拼接 | 维度数 ≥10（7 维张量加几次 split 即可）时 `range10` 被拆成 `range1*range0`，循环上界完全错误且能编译通过；`size()==6` 的保护同时失效 | loop id 用整数向量，名字只在输出时生成 | 主要 |
| 一次编译至少重跑 2 遍完整 pass pipeline，每遍从文本重新解析 | `opt/pass_manager.cc:47` 构造即 `all(oc->get_src())` 把整份生成的 C++ 重新解析成 KernelIR；`tuner_manager.cc:35-38` 与 `:57-59` 各一遍，`jit_searcher.cc:33-35` 每个候选一遍；而 `jit_searcher.cc:58-61` 的 timeout 字段声明了却全文无人读取，`reorder_tuner.cc:22-24` 的候选是 N! 量级 | 首次执行每个融合算子付两次"全文解析加 25 个 pass 加全树 to_string" | tuner 只改 loop_options 不重跑 pass；一次解析后 clone IR | 主要 |

## 补充：内存与分配器（第二轮）

| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| SFRL 的 free/share_with 直接索引未清零的静态映射表，先解引用后校验 | `sfrl_allocator.cc:24-25` `new CachingBlock*[ID_LIMIT]`（默认初始化，元素不确定）；`:291-292`、`:313-315` 直接 `occupied_id_mapper[allocation]` 后解引用；对照 `:82,90` 的 erase/get 是有非空断言的。free 完全忽略传入的 mem_ptr 与 size，只信 allocation | 二次释放时 id 可能已重新分配给别的块，静默释放错误的块使两个 var 拿到同一段内存；id 未复用则读到从未写过的垃圾指针（数组没清零，非空校验形同虚设）；allocation 超过 2^21 直接读表外 | `new ...[N]()` 清零；三重校验 | 关键 |
| alloc 的"必须写回 allocation"契约被 5 个分配器违反，`Var::allocation` 从无初值 | `aligned_allocator.cc:16-28`、`cuda_device_allocator.cc:24-38`、`cuda_host_allocator.cc:19-24`、`cuda_managed_allocator.cc:20-25`、`nfef_allocator.cc:24-26` 均不写形参；`var.h:27` 是 Var 里唯一没有初值的成员 | `getitem_op.cc:515-518` 与 `setitem_op.cc:336-341` 用 `allocator` 与 `allocation` 相等判断"已就地共享，跳过 memcpy"。在 `use_sfrl_allocator=0` 或 `use_nfef_allocator=1` 下两边都是未初始化残值，相等即误判，**静默错数**而非崩溃 | 每个 alloc 实现必须写 allocation；给初值；别名判断改用显式 share 关系 | 主要 |
| migrate_to_cpu/gpu 静默解除 share_with 建立的别名 | `allocator.cc:167-173,194-200` 新分配加 memcpy 加 free 原块；`var.cc:121` 共享路径把偏移覆盖成父块的 allocation，迁移代码无从得知"我是某块的子区间"；调用点 `executor.cc:593-610` | getitem/setitem inplace 与 fused_adamw 建立的别名，在混合 CPU-CUDA 图里被单方面迁移后断开，通过一个别名的写对另一个不可见 | 迁移前检查共享关系，整组迁移或拒绝 | 主要 |
| fetch 的跨流内存复用只做了一半的顺序保证 | `fetch_op.cc:48` 建非阻塞流；`:121-122` 只有 stream 等默认流，`:156-159` 没有反向的默认流等 stream；`:103` 不做设备同步；`sfrl_allocator.cc:293-301` 源 var 释放后块立即回 free list 且无流完成跟踪 | 默认流上的下一批 kernel 可以覆盖 mem_ptr 而 stream 上的拷贝还没执行，fetch 到被覆盖后的数据。这比"依赖单流"更强：这里确实有第二条流 | 拷贝后记 event 让默认流等待 | 主要 |
| TempAllocator 用同名成员遮蔽基类的统计字段 | `allocator.h:17` 基类声明 used_memory/unused_memory；`temp_allocator.h:29` 重新声明；`swap.cc:90,92,107,126` 全部通过基类指针读 | 通过基类指针看到的 TempAllocator 用量恒为 0，`cpu_mem_limit`/`device_mem_limit` 对所有 workspace 分配（cuDNN conv、cub sort/where 等 20+ 调用点）完全失效；而 `mem_info.cc:184-193` 用派生类指针所以显示正确——"显示正常、限额不生效" | 删掉派生类里的重复声明 | 主要 |
| 缓存回收把 allocation=0 传给底层分配器，使分层配置不可嵌套 | `sfrl_allocator.cc:183`、`temp_allocator.cc:93,100,116`；底层真正返回的 allocation 在 `:260` 拿到后被 `:284` 覆盖丢弃。而 id 从 1 开始，`occupied_id_mapper[0]` 永远是未初始化垃圾 | `allocator.cc:103-122` 的分层设计明确鼓励嵌套，当前默认配置只是侥幸没踩到 | CachingBlock 保存底层 allocation 并原样回传 | 主要 |
| 回收策略的正常路径实际是死代码，全局单锁且覆盖不全 | `sfrl_allocator.cc:241` 一把 mutex 供所有 SFRL 实例共用；`:307-310` gc() 不加锁却改同样的结构，而 `allocator.h:53` 把 gc_all 暴露给 Python；`temp_allocator.cc:41-110` 完全无锁；`allocator.cc:45` 使 free_ratio=1，令 `:227` 的条件恒假——每次缓存未命中都调的 free_all_sfrl_allocators 是纯开销的空操作 | 锁给了错误的安全感；CPU 与 GPU 分配互相串行；OOM 前的行为只由异常路径决定 | 每个分配器一把自己的锁并覆盖 gc()；free_ratio 改小或删掉这条已死策略 | 主要 |

## 补充：绑定层与失败模式（第二轮）

| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 构造失败后在零初始化对象上跑 C++ 析构 | `pyjt_compiler.py:875` 无条件 `tp_new = PyType_GenericNew`（内存 memset 为 0），`:876` 无 GC 标志；tp_init 无匹配重载时 return -1（`:722`）→ CPython 立即调 tp_dealloc → `py_ring_buffer.cc:241-243` → `ring_buffer.cc:73` 在检查 init **之前**无条件解引用。**已核实 tp_new 与 tp_flags** | 一行 `jittor_core.RingBuffer()` 即段错误。VarHolder 侥幸安全（`var_holder.cc:141` 有判空）是巧合不是设计 | 生成带"已构造"标志的 tp_new，或让 tp_dealloc 先检查 | 关键 |
| 标量转数组共享一个非线程局部的全局 union | `numpy.h:125-131`/`numpy.cc:56` `tmp_data_t tmp_data;` 全局非 thread_local；`py_converter.h:363-374` 返回它的地址而非拷贝 | 一次调用里出现两个标量参数时后者覆盖前者；跨线程只靠 GIL 侥幸串行，而并行编译器正是释放 GIL 的地方 | 标量走自带 buffer | 主要 |
| 整数提升用"取最大字节数加与运算"替代提升格，静默溢出 | `nano_string.h:251-254`（浮点分支之外）：`dsize_ = max(...); is_unsigned = x.is_unsigned() && y.is_unsigned();`。**已核实** | `uint8 + int8` 得 int8（NumPy 给 int16），`uint8(200)+int8(1)` 得 −55 无警告；`uint32+int32` 得 int32（应 int64）；`uint64+int64` 得 int64（NumPy 给 float64） | 混合符号时单独提一档 dsize，达上限退到 float64 | 关键 |
| 信号处理器整条路径非 async-signal-safe 且以抛异常收尾 | `log.cc:250-254` 里 LOGe 走 ostringstream 即 malloc；`:268-269` cerr；`:307` LOGf 即 throw；`:314` print_trace 里 fork+execvp 与 system；`:321` exit(1) 触发 atexit 遍历 cleanup_callback 执行显存释放；`:204,206,689` 的标志是普通 bool/int 而非 `volatile sig_atomic_t` | 崩在 malloc 内部时处理器再进 malloc 即死锁，进程挂死而非给出崩溃报告 | 处理器内只做 write 与 _exit，符号化交给预建的 helper 进程 | 关键 |
| 环境变量解析失败静默回退默认值，唯一的警告还会被吞掉 | `log.h:180-196` 把"解析成功"编码为"再读一次失败"：`export log_v="1 "` 的尾随空格使 peek 不设 failbit 从而静默回退；`log.cc:173` 在 log_silent 下吞掉这条警告 | 写错的环境变量表现为"设置无效"且无提示 | 用 from_chars 加全串消费校验，启动期配置 fail fast | 主要 |
| `DEFINE_FLAG_WITH_SETTER` 的 setter 在赋值**之前**被调用 | `log.h:228-242` `set_##name(v) { setter_##name(v); name = v; }`；绕过证据 `tracer.cc:137-139` 的 setter 必须手工回写才能让另一个 setter 看到新值；`log.cc:441` 的 setter 抛异常时赋值被跳过而用户以为设置成功 | setter 看到的永远是旧值，每个有副作用的 setter 都要自己打补丁 | 先赋值再调 setter，签名改成收新旧两值 | 主要 |
| `token_replace_all` 用异常做循环终止 | `str_utils.cc:227-239` 的正常终止条件是 `:187` 的 CHECK 抛出，即每次调用必然抛一次并走完整的格式化与构造；同一个 catch 还吞掉真正的错误 | 源码改写静默失败并返回未改写的源码；每次调用付一次异常开销 | 用返回值表达"无更多匹配" | 次要 |

## 补充：CUDA 归约没有任何优化路径（2026-09-02 实测）

追查 UNet 剩余 3% 时发现的，是第一节「核心的一部分不在源码树里」的一个具体代价。

| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 归约调优器对 CUDA 直接返回，CUDA 归约完全不调优 | `opt/tuner/reduce_tuner.cc:14` `if (fo->flags.get(NodeFlags::_cuda)) return;` 是 `ReduceTuner::run` 的第五行。其后的 split/order 候选（`:53-58`）只对 CPU 生效 | CUDA 上归约的循环切分与顺序没有任何搜索，只能拿 `ParallelPass` 给的默认配置 | 给 CUDA 归约一套自己的候选，或说明为什么不需要 | 主要 |
| `SharedReducePass` 在整个负载里从未生效 | 统计四个 JIT 缓存共约 4900 个生成的归约 kernel：含 `atomicAdd` 的有相当比例，含 `shared_reduce_add` 的 **0 个**。该 pass 在 `pass_manager.cc:117` 确实被调用，但它只有 `.h` 没有 `.cc`——实现在 `utils/data.gz` 里，无法查明它的触发条件 | 空间维归约退化为「每线程私有累加后直接 `atomicAdd` 到全局」。以 UNet 一个实测 kernel 为例（`reduce OP_add DIM_4 REDUCE_c`，16 次/步、每次 52.9us）：3072 个输出地址、每输出 64 个线程各做一次原子加，合计约 19.6 万次原子加，64 路争用；PyTorch 对应的 `reduce_kernel` 走共享内存树形归约，每输出只写一次，平均 3.7us | 需要一条可读的块内归约实现。在 `data.gz` 还原之前，写新 pass 无法与既有 pass 协调 | 关键 |

这两条合起来解释了 UNet 剩余差距里 kernel 侧的大部分：卷积改走 backend 计划缓存后
Jittor 的卷积与 GEMM 已比 PyTorch 快 2.57ms，而逐元素与归约合计慢 3.44ms。

