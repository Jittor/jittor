# Torch 兼容层（python/jittor/compat/）

**核心判断**：这一层最根本的问题不是覆盖度不够，而是**它根本不是一层**。
`site-packages/torch/__init__.py` 最后一行是 `sys.modules[__name__] = _jittor`
（`shim/resources/torch_init.py:16`），即 `torch is jittor`：不存在独立的 Torch
对象模型，只有被就地改写的 Jittor 模块和被就地改写的 `Var` 类。三个结构性后果：
(1) Torch 与 Jittor 语义冲突处（`.data`、dtype、Parameter、0 维）只能靠进程级模式
开关加逐对象 Python 标记属性区分，标记要每个算子包装器手工传播，漏一个就是静默
错误；(2) 兼容层无法卸载、无法事务化、无法并存，install 失败后进程停在半改写状态；
(3) Torch 语义已反向渗入内核（`_runtime/core_api.py`、`misc/indexing.py`、
`nn/modules/parameter.py`），compat/ 的目录边界与真实依赖边界不重合。
28k 行里有大量"签名齐全、语义为空"的 API，共同特征是**不报错、结果错**。

## 部署与激活
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 用脚本源码的文本嗅探决定是否接管整个进程 | `shim/preflight.py:18` `_ENTRY_MARKERS` 含 `"import jittor as torch"`；`:101-116` 读 argv[0] 前 64KB 做子串匹配；`jittor/__init__.py:23` 每次 import 无条件调用 | 纯 Jittor 脚本因注释里出现该短语就被改写 HOME/TMPDIR/XDG_CACHE_HOME/LD_LIBRARY_PATH、置 use_nccl=0、给 nvcc 加 --fmad=false；反向也成立，改个入口写法就静默失活 | 激活必须是显式唯一信号，删除源码嗅探 | 关键 |
| import 期改写进程级环境变量且不可逆 | `preflight.py:365` 覆盖 HOME；`:347` 设 use_nccl=0；`:367` 覆盖 TMPDIR；`_add_nvcc_flags` 强加 `--fmad=false` | 改 HOME 影响 HF token/git/ssh；use_nccl=0 与分布式承诺矛盾；关闭 FMA 是全局数值与性能决策却无记录 | 环境准备移到显式启动器，库 import 只读环境不写 | 关键 |
| 至少三条互不等价的 torch 模式入口 | 部署式 `torch_init.py:15`；`shim/runtime.py:enable()`；`shim/control.py` 的 `jt.flags.torch_shim=1` 赋值副作用 | 语义随入口而变；`tensor.py:904-908` 每次读 `.data` 都要现场判定处于哪种模式 | 收敛为单一 activate()，模式是不可变进程属性 | 主要 |
| flags 被代理对象包住，热路径每次走 Python `__getattr__` | `shim/control.py:TorchShimFlagsProxy.__getattr__`，注释自承解码一步要读 flags 数千次 | 全进程 flag 读取变慢，只为承载一个布尔的赋值副作用 | 激活做成函数调用，flags 保持原生对象 | 次要 |

## install 的事务性与失败模式
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 命名空间事务只覆盖 sys.modules，不覆盖真正的改写面 | `compat/torch/__init__.py` 的 `_restore_namespace()` 只恢复 torch* 模块项；install 同时改写 jittor 模块属性、Var 类字典、`builtins.__import__`、`sys.meta_path`、os.environ、jt.flags | 任一步失败后停在半改写状态，此后行为不可预测 | 要么真正可回滚（`module_patcher.restore_method` 已有该模式但 installers 没用），要么明确硬失败 | 主要 |
| 可选步骤失败被永久记为 failed 且无人看见 | `context.py:run_optional` 吞掉所有异常；`InstallReport` 只进 `context.reports`，`compat/runtime.py:52` 收集后无输出 | 可选面失败后报错出现在离病因很远的地方 | 失败必须 warn 一次并可通过 API 查询 | 主要 |
| install 全程无锁无重入保护 | `context.py` 无锁（对比 `module_patcher.py` 有 `_LOCK`）；flag setter 可从任意线程触发 | 多线程下竞争产生半装配状态 | 一次性锁加幂等哨兵 | 次要 |

## 张量语义：视图/存储/叶子/0 维
至少四条独立的手工标记链在维持本应由类型系统保证的语义。

| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| `view()` 其实是 reshape，不共享存储 | `installers/tensor.py:2243` `Var.view = _torch_reshape` | `y=x.view(-1); y[0]=1` 不改 x；torch 对非连续张量会报错，这里永远成功 | 需要真正的 base+offset+stride 存储对象 | 关键 |
| `is_leaf` 恒 True、`grad_fn` 恒 None | `tensor.py:1260-1261`、`1266-1267` | 所有基于微分图内省的分支走错：`if not t.is_leaf`、`assert loss.grad_fn is not None`、梯度检查点、PEFT 冻结判断全部静默取反 | 由 Var 是否有 grad 输入边真实回答，内核已有该信息 | 主要 |
| `Var.backward(gradient=...)` 参数被完全忽略 | `tensor.py:1059` 签名接受，函数体 1059-1180 再无引用；`autograd.py:269` 把 grad_tensors 转发进来 | `y.backward(w)` 算的是无权重梯度，数字错且无提示 | 立即实现或显式抛异常 | 关键 |
| retain_graph 默认值使自身注释失效 | `tensor.py:1059` 默认 False，`:1062` 的 `if retain_graph is None` 分支永不触发 | `backward(create_graph=True)` 仍释放图 | 默认改 None | 主要 |
| 0 维张量靠 `_torch_0d` 属性模拟，只在约 10 处传播 | `misc/indexing.py:8-16` 产生标记；`tensor.py:1317/1328/1336/1352/1361/1659/1673` 手工透传 | `x[0]` 是 0 维、`x[0]+1` 不是、`x.sum()` 不是；shape/pickle/广播随之分叉 | 0 维应是 Var 的形状能力 | 主要 |
| 视图写回靠三条并行标记链递归回写 | `tensor.py:866-873` 建链，`:517-536` 递归逐级回写，`:397/437/478` 三个 `_torch_data_owner` 分支 | 每层回写在惰性图上多建节点；只覆盖基本索引；写路径与 CPU/GPU 驻留判定耦合 | 有存储模型前至少合并成一个 `_View(base, path)` | 主要 |
| 反向叶子是三个进程级 id 键强引用字典 | `jt._torch_leaf_params`（六处独立填充：`nested.py:254-256`、`tensor.py:944`、`nn.py:1228/1257/1399/1544`）、`jt._torch_retained`、`jt._active_optimizers` | 叶子集合取决于谁先调用过 parameters()（`nn.py:1399` 注释自承 bert 曾 0/39 个梯度）；Var 不可弱引用故强引用加手工 prune | 叶子由 requires_grad 加图连通性决定；内核需提供反向可达查询 | 主要 |
| Parameter/buffer 语义实现在内核里，由全局模式位开关 | `compat/torch/__init__.py` 结尾 `_core_api._torch_registration_semantics = True`；`_runtime/core_api.py:1402-1405,2155-2173,2201-2207`；`nn/modules/parameter.py:44/136/143/166` | compat/ 不是边界：内核有 torch 分支且行为随全局布尔改变 | 统一注册规则，删除模式位 | 关键 |

## dtype 与 device 映射
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| `torch.dtype` 是 str 子类 | `compat/torch/types.py:14-88` | `isinstance(x.dtype,str)` 为 True（torch 为 False）；`x.dtype=="float32"` 为 True；f-string 得到 `float32` 而非 `torch.float32` | 真正的 dtype 对象，入口处一次转换 | 主要 |
| 26 个 dtype 是存在但无算子的占位 | `types.py:_make_dtypes` specs | 可以 cast 到 float8 得到无定义行为 | 参与计算时抛 NotImplementedError | 次要 |
| device 只是标签：set_device 空操作、current_device 恒 0 | `installers/cuda.py:214-215`；tensor.py 的 `_device` 恒返回 cuda:0 | device_count 返回真实卡数而 set_device 无效 | 落地前对 i!=0 应报错 | 关键 |
| Stream/Event 全部空壳 | `cuda.py:301-322`：wait_stream/wait_event/record_event 返回 None，query 恒 True，`Event.elapsed_time` 恒 0.0 | 用 event 计时得到 0 ms；依赖 stream 顺序的代码在单流上碰巧正确 | elapsed_time 应报错 | 主要 |
| `torch.backends.*` 映射不透明 | `cuda.py:681-712`：allow_tf32 的 getter 走 `__getattribute__`，setter 被 `_jittor_cudnn_init` 门控且 `:704-706` 初始化赋值会穿过 setter；`cudnn.conv.fp32_precision` 是字面量 "ieee"；set_float32_matmul_precision 不影响 cudnn | 同一语义三个入口两套状态 | 一个精度策略对象加表格化单测 | 主要 |

已修：7.01（49d41acf `torch.cuda.set_device(i!=0)` 报错、`Event.elapsed_time`
改为真实计时并对未开 `enable_timing` / 未 record 报错；Stream 的
`wait_stream`/`wait_event` 保持空操作并注明理由——jittor 把所有逻辑流串行到
一条物理流上）。`torch.dtype` 是 str 子类、26 个占位 dtype、
`torch.backends.*` 三条属于 7.08。

## 看起来支持其实是空操作（本层风险最集中处）
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| `torch.autocast` 完全空操作 | `compat/torch/grad.py:55-77`；`installers/core.py:237-238` `is_autocast_enabled→False` | 混合精度训练静默跑成 fp32，显存速度数值全变而 loss 曲线看着正常 | 实现或进入上下文时 warn | 关键 |
| `load_state_dict` 忽略 strict，恒返回空 missing/unexpected | `installers/nn.py:1353-1371` 末行 `return _IncompatibleKeys([],[])` | strict=True 对错配 checkpoint 不报错，静默加载残缺权重 | 计算真实 key 差集 | 关键 |
| `torch.load` 忽略 weights_only 与 map_location；未知类退化成空类型 | `serialization.py:312-313` 注释即 "(ignored)"；`:275-294` `find_class` 兜底 `return type(name,(),{})` | weights_only 是安全保证，忽略等于恢复任意代码执行；未知类变空壳后 checkpoint 加载成功但对象是错的 | 受限 unpickler 白名单；未知类必须抛异常 | 关键 |
| `_rebuild_tensor_v2` 丢弃 stride | `serialization.py:260-268` 只取切片后 reshape | 保存自非连续视图的权重被静默读错值 | 按 stride 还原或报错 | 主要 |
| DataLoader 记下 num_workers 等后全不用 | `installers/data.py:215-247` 恒建单进程迭代器，`_MultiProcessingDataLoaderIter` 是 pass | 数据流水线静默降级为串行，容易被误读为框架慢 | 实现多进程取数否则 warn | 关键 |
| SummaryWriter 全部方法返回 None | `installers/utilities.py:407-433`，该分支恒生效 | 训练日志静默全部丢失 | 转发到真实 tensorboard，找不到就报错 | 主要 |
| `tree_map`/`tree_map_only` 不递归 | `utilities.py:571-572` `tree_map = lambda f,x: f(x)`；同处 `:528` 的 tree_flatten 却是真递归 | `tree_map_only(Tensor,to_dev,batch)` 原样返回，批数据没搬到设备且无错误 | 用已有的 flatten/unflatten 实现 | 关键 |
| `nn.init.dirac_`/`sparse_` 是恒等 | `installers/nn_init.py:131,146` | 依赖这些初始化的模型从 empty 垃圾开始训练 | 实现或抛异常 | 主要 |
| checkpoint 直接调用函数 | `installers/data.py:311-312` | 显存保证没了，use_reentrant 被忽略 | 至少文档化 | 次要 |
| `swa_utils.update_bn` 空操作 | `lr_scheduler.py:385` | SWA 权重配陈旧 BN 统计量，精度静默下降 | 实现 | 次要 |
| `has_torch_function` 恒 False，TorchFunctionMode 惰性 | `installers/cuda.py:600-609` | 张量子类与 device mode 被静默绕过 | 声明不支持并报错 | 主要 |
| `torch.library.opcheck` 返回 None | `library.py:249` | 用户的算子正确性测试无条件通过 | 抛 NotImplementedError | 主要 |
| set_default_device 空操作而 get_default_device 报告真实设备 | `installers/core.py:485-488` | set/get 自相矛盾 | 一起实现或一起报错 | 次要 |

已修：7.01（ff395ecc `torch.autocast`；b7c12ddc `load_state_dict(strict=)`；
0446217e `torch.load(weights_only=, map_location=)` 与 `find_class` 兜底；
47012a27 DataLoader `num_workers` 与 `checkpoint` 的显存代价；
49d41acf `Var.backward(gradient=)`、`tree_map`/`tree_map_only`、SummaryWriter、
`dirac_`/`sparse_`、`update_bn`、`has_torch_function`/TorchFunctionMode、
`opcheck`、`set_default_device`）。统一开关见 `jittor/compat/stub_policy.py`：
默认抛 `NotImplementedError`，`JITTOR_TORCH_ALLOW_STUB=1` 或
`torch.compat_allow_stub(True)` 才降级为原来的静默行为并 warn 一次。
负向测试在 `tests/compat/torch/test_torch_compat_unimplemented.py`。
`_rebuild_tensor_v2` 丢弃 stride 一条未动，属于 7.15。

## 自定义算子、编译与自动微分
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| dispatch key 被完全折叠，最后注册的实现赢 | `compat/torch/library.py:69-70` `next(reversed(list(self._implementations.values())))` | `impl(...,"CPU")` 后注册 "CUDA" 则 CPU 张量跑 CUDA kernel；`impl(...,("CPU","CUDA","Meta"))`（`:212-223`）让 Meta 实现排最后，每次真实调用返回 fake 结果 | 按驻留与 dtype 选择；Meta 必须排除 | 关键 |
| `register_autograd` 记录的 backward 从不被调用 | `library.py:235-236` 写入 `_backward`/`_setup_context`，全仓无读取点 | 前向若脱离 tape，梯度静默为零 | 用 jt.Function 包住前后向 | 关键 |
| custom_op 里硬编码具体模型的算子名 | `library.py:176-191` 对 `"transformers::grouped_mm_fallback"` 特判并丢弃用户实现 | 通用注册 API 里的模型特例，同名注册被静默替换 | 移到 integrations 适配点 | 主要 |
| `torch.compile`/`jit.trace`/`jit.script` 吞掉全部参数返回原对象 | `installers/compiler.py:34-36,122-123` | `fullgraph=True` 这种断言语义被静默接受；jit.trace 返回未 trace 的模型 | 保留 pass-through 但对语义性参数报错 | 主要 |
| `torch._inductor`/`fx.*`/`_dynamo.*` 交给 permissive finder 伪造 | `compiler.py:92,104-107` 加 `compat/permissive.py:27-37`：任意属性到调用返回 None 的类 | `from torch.fx.passes... import ShapeProp` 成功、实例化成功、调用返回 None；permissive.py 自己的文档就写明会掩盖缺口 | 只覆盖已知的 import-time 引用清单 | 主要 |
| `ctx.needs_input_grad` 只统计位置参数 | `installers/autograd.py:127-131` | 关键字调用的 Function 拿到错位标志 | 按签名归一 | 主要 |
| `autograd.grad(create_graph=)` 被折进 retain_graph | `autograd.py:237,244` | create_graph=False 仍返回可微张量；两参数组合与 torch 相反 | 分开处理 | 主要 |
| 多输出隐式求和 | `autograd.py:233` `sum(o.sum() ...)` | torch 会报只能对标量隐式求导，这里静默按 ones 处理 | 对齐 torch 报错 | 次要 |
| `_sum_grad_to` 元素数不匹配时返回零 | `autograd.py:93-94` | 掩盖真正写错的 backward | 至少 warn | 次要 |
| saved_tensors 无版本计数 | `autograd.py:62-69` | 原地修改后反向静默用新值（torch 会报错） | 记录版本号 | 次要 |

## 分布式与 FSDP2
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| DDP 是纯转发壳，torch 惯用写法下梯度从不同步 | `installers/nn.py:132-137` 无 hook 无 bucket 无初始广播，`no_sync()` 是 nullcontext；同步只存在于 `optim/base.py:206-215` 的 `opt.step(loss)` 路径，而 `loss.backward(); opt.step()` 走 `tensor.py:1152` 自填 grads；tensor.py/optimizers.py 内 mpi_all_reduce 命中数为 0 | 标准 torch 训练循环在 N 卡上训出 N 个互不相同的模型，无任何报错；初始参数也未广播 | backward 完成点做真实 all-reduce，或构造时直接抛 NotImplementedError | 关键 |
| `dist.nn.all_reduce` 是恒等 | `installers/distributed.py:657` | 可微 all-reduce 的值和梯度都错 | 实现或报错 | 关键 |
| 分布式 checkpoint 读写是恒等 | `distributed.py:703-706` `checkpoint.load/save = lambda state_dict=None,*a,**k: state_dict`；`:747` `_write_item` 为 None | `dcp.save()` 什么都不写并返回成功，静默数据丢失 | 报错 | 关键 |
| `new_subgroups_by_enumeration` 恒返回 WORLD | `distributed.py:510` | 子组集合通信静默扩散到全体 rank | 与 `new_group`（`:439-442` 已正确拒绝）一致抛异常 | 主要 |
| DeviceMesh 退化：`mesh["dp"] is mesh["tp"]`、`get_group()→None` | `fsdp2/dtensor.py:36-37,67-68,73-77` | 2D 并行静默塌缩成一维 | 报错 | 主要 |
| `DTensor.full_tensor()` 直接返回本地分片 | `fsdp2/dtensor.py:187-188`、`:151-152`；`compat_types.py:85-89` parallelize_module 只打标记；ColwiseParallel 等是空子类 | TP 程序跑通但每个 rank 用 1/N 权重算全量结果 | 未实现的并行策略必须抛异常 | 关键 |
| FSDP2 分片真实但显存不降 | `fsdp2/shard.py:361-366` 明确不释放 `entry.full_param`；`:327` 写入的 `true_fsdp_flat_full_param` 全仓无清除点；`grad_sync.py:200` 常驻全尺寸梯度 | 峰值约等于未分片，FSDP 核心收益丧失 | 让反向不再需要同一个 Var 对象后释放 | 主要 |
| `fully_shard(mesh=...)` 接受 mesh 但按全局 world 分片 | `fsdp2/api.py:180-181` 存下、`shard.py:229-231,74` 用 `common._world_size()` | 混合并行下 8 卡全参与分片与 reduce-scatter，数值错 | 未支持的 mesh 直接拒绝 | 关键 |
| FSDP 下 `clip_grad_norm_` 是 rank 本地的 | `compat/torch/grad.py:149-162`、`installers/nn.py:197-203` 均无跨 rank 归约 | 每个 rank 用偏小范数裁剪，训练轨迹与 torch 不同 | 加跨 rank 平方和归约 | 主要 |
| TCPStore/FileStore 是进程内字典 | `distributed.py:770-788` | 基于 store 的 rendezvous 静默成功 | 报错 | 主要 |
| tensor 层与 fsdp2 层双向耦合 | `tensor.py:1096-1122,1167-1172` 直接 import fsdp2；`distributed.py:220-223` 反过来 import `fsdp2.common`；`fsdp2/installer.py:49` import `compat.torch.context`；optimizers.py 中 31 处 fsdp 引用 | 三个包互为下层，无法单独理解或替换；`nn.py:1237-1242` 与 `shard.py:383-398` 两处 hook 同时生效 | 单向依赖 core→tensor→nn/optim→distributed→fsdp | 主要 |
| FSDP 路径重写了一套 SGD/Adam，靠类名子串识别优化器 | `fsdp2/optimizer.py:93-133`、`:136-144`、`:170-172` | 同一数学两份实现；自定义 Adam 子类落到 NotImplementedError | 复用 jittor optimizer，只替换梯度来源 | 主要 |

已修：7.01（46bc9ea7 DDP 在 world_size>1 时构造即报错，真实同步仍是 7.02；
9053a7c0 `dist.nn.all_reduce`、分布式 checkpoint 读写、
`new_subgroups_by_enumeration`、DeviceMesh 的取轴与 `get_group`、
`DTensor.full_tensor`、TCPStore/FileStore）。判据是「单 rank 下本来就是恒等」：
单进程路径保持精确，只有真的会算错的多 rank 情形才拒绝。
其余条目（FSDP2 显存、mesh 分片、clip_grad_norm_ 跨 rank、双向耦合、
优化器重写）属于 7.06/7.13。

## vLLM / shim / 模块补丁的边界
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| vLLM 边界规则形同虚设：因为 torch is jittor，走 `torch.` 就绕过全部检查 | `tests/structure/test_vllm_compat_structure.py:83-97` 只检查根名为 jt/jittor 的私有属性；而 `compat/vllm/__init__.py:57,67` `import torch` 后 `torch.__version__ = api_version` 就是在改 `jittor.__version__` | 只用公共 API 的适配器在 import vLLM 时静默改掉框架版本号；规则测试给出虚假安全感 | 边界检查须把 torch 视作 jittor 别名 | 主要 |
| `builtins.__import__` 被串接两次以打第三方库补丁 | `installers/utilities.py:61-76`（transformers）、`:196-209`（torchmetrics） | 每次 import 多走两层 Python；不可卸载；两条链互相覆盖时行为取决于安装顺序 | 统一走已有的 module_patcher | 主要 |
| 兼容层改写第三方库的私有函数 | `utilities.py:103-191` 替换 torchmetrics 的 `_bincount`、`dim_zero_cat`、`_safe_divide` | 版本升级会静默失效或静默算错；性能补丁混进兼容层 | 移出 compat/ 作为带版本断言的 integration | 主要 |
| `runtime.enable()` 把项目目录插到 sys.path[0] | `shim/runtime.py:95-97` | 项目里的 types.py/copy.py 会遮蔽标准库 | 只加 shim 的 site 目录 | 次要 |
| `_ensure_dir` 在 import 期无保护 mkdir | `preflight.py:142`，只读 HOME 下 import jittor 直接抛 PermissionError | 受限环境完全不可 import 且错误信息与真实原因无关 | 捕获并降级为明确诊断 | 次要 |

## 代码结构与测试
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 整个 API 表面是几个巨型函数里的闭包 | `_install_tensor_methods` 1386 行 79 个内嵌 def/class；`_install_nn_extras` 1003 行 126 个；`_install_module_methods` 667/40；`_install_cuda` 623/67；data 的 install 267/55 | 没有任何一个 torch API 是可单独 import、单独测试、单独查阅的对象；无法自动生成实现程度覆盖表，上面那张空操作清单只能靠人肉阅读得到 | 每个 API 一个模块级函数加注册表，install 只做绑定 | 关键 |
| 文件拆分是机械搬运，没有引入接口 | 三个 installers 文件首行文档字符串都是 "source moved from the former monolithic installer without changing the compatibility semantics" | 已模块化是表象，单体的所有耦合原样保留 | 拆分应以可独立测试的实现单元为粒度 | 主要 |
| 错误处理以静默吞咽为主 | 全 compat/ 统计：356 个 try，258 个宽泛 except，其中 **129 个是 `except: pass`** | 一次失败的标记传播、一次失败的 dtype 还原都不留痕迹 | 吞咽必须限定异常类型并至少 debug 打点 | 主要 |
| 结构测试钉死实现细节而非契约 | `tests/structure/` 8071 行 23 文件；`test_torch_compat_structure.py:37-100` 用 AST 校验 sys.modules 赋值写法；`test_vllm_compat_structure.py:66-68` 断言每文件不超过 300 行、`:57-62` 断言文件名集合 | 改文件名、拆超过 300 行的文件都会红；同时那 14 条静默空操作无一被覆盖 | 保留边界类断言，删除行数与文件名断言，预算移到与真 PyTorch 的行为对拍 | 主要 |
| 真正缺失的测试类别 | `tests/compat/torch/` 20k 行里没有针对空操作的负向测试：无 autocast 生效性、无 `load_state_dict(strict=True)` 报错、无 `backward(gradient=)` 数值、无 dispatch key 路由测试 | 已修复的缺陷（vmap 曾是空操作，`numerical.py:245-246` 有记录）说明这类缺陷反复发生 | 每个未实现 API 都要有断言它抛异常的测试 | 关键 |

## 架构判断
**不可持续，但不是因为 monkeypatch 本身。** 用 monkeypatch 装配命名空间是可行的工程
手段；不可持续的是**把 torch 定义成 jittor 本体**（`torch_init.py:16`）。这一个决定
同时造成了：语义冲突只能用全局模式位解决、无法回滚、边界检查失效、Torch 语义反向
渗入内核。在此前提下每新增一种用户写法就是一个新漏洞。

重做时边界应这样划：
1. **torch 是独立的包，不是 jittor 的别名。** `torch.Tensor` 持有 `jt.Var`（或是其真子类），
   requires_grad/is_leaf/grad_fn/0 维/视图/存储都是这个类型的字段。今天散在
   `_torch_index_parent`、`_torch_data_owner`、`_torch_0d`、`_torch_leaf_params`、
   `_torch_retained`、`_is_torch_parameter`、`_jittor_torch_force_cpu` 里的东西
   都是这个缺失类型的碎片。
2. **内核只暴露能力，不暴露 torch 分支。** 内核需补三样：带 stride/offset 的存储视图、
   按值指定上游梯度的 grad、从某节点反向可达的 requires_grad 叶子查询。补上后
   `_torch_registration_semantics`、indexing 的 0 维分支、tensor_ops 的 CPU 驻留分支都可删。
3. **每个 API 一个一等对象加保真度标注**（exact/approximate/unimplemented），
   install 只做绑定；unimplemented 默认抛异常，需显式 allow_stub 才降级为静默。
4. **依赖单向化**：core→tensor→nn/optim→distributed→fsdp→适配器。
5. **激活显式、一次性、可查询**：删除 argv 嗅探与 import 期环境改写。
6. **第三方库补丁搬出 compat/**，走 module_patcher 的 entry-point 机制，
   禁止串接 `builtins.__import__`。

其中 (3)(4)(5)(6) 主要是重新组织与删减，可先于 (1) 落地；尤其 (3) 能在不改架构的
前提下把"静默错误"这一整类风险转成"明确报错"。
