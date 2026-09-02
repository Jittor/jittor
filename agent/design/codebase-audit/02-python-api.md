# Python API 层（不含 compat/extern/src）

**核心判断**：这一层没有对象模型，只有一堆约定。张量没有视图与存储概念（切片、view、
expand 全是拷贝，所有"就地"操作实为 assign 整个 Var），设备不是张量属性（`model.cuda(1)`、
`x.to('cuda')` 都只是改一个进程全局 flag），参数身份不是类型而是散落在 Var 与 Module 上的
26 个临时标记属性，后端选择不是 dispatch 层而是 57 处手写的 `if jt.flags.use_cuda`。
因为没有模型，每加一个需求就只能新加一条并行路径和一个新标记：同一个 AvgPool2d 有两份
实现且数值不同、reduction 参数在同一个文件里有四种解析方式、nn.Conv2d 与
nn.functional.conv2d 是两份独立抄写的 reindex 代码。这些并行路径的差异从不被交叉验证，
本次审计仅在 CPU 上就复现出 5 处静默算错结果的缺陷。维护成本的主要来源不是代码量，
而是**没有一处地方定义"什么是参数""什么是就地""在哪个设备上"，每条新代码都要重新
回答一遍且答案各不相同**。路径相对 `python/jittor/`。

## 张量语义：没有视图与存储模型
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| 切片/view/expand 全部返回独立拷贝，写回不生效 | 运行时验证：`y=jt.arange(6); v=y[1:4]; v.assign(v+100)` 后 y 不变；`r=z.view(2,3); r.assign(0)` 后 z 不变。`_runtime/core_api.py:722` `Var.view = Var.reshape`；`misc/shape_transforms.py:122` expand 即 broadcast；写回机制 `src/var_holder.cc:406-441` 只支持连续整数下标链且硬编码 10 层上限 | 所有 `x[i:j] = f(x[i:j])` 风格代码静默失效；expand 物化整个张量显存高一个量级；chunk/split 的 docstring 明写 view 与实现直接矛盾 | 给 Var 引入显式 storage（base+offset+strides），所有原地操作作用于 storage | 关键 |
| 114 个 `foo_` 就地方法由"首参数名启发式"在 import 期批量生成 | `__init__.py:135-154`：遍历 Var.__dict__，首参名属于 {x,input,self} 即生成 `k+"_"`。运行时得到 114 个，含 tolist_、isnan_、sort_、cpu_、norm_、nonzero_、peek_、argmax_ | `x.isnan_()` 就地把 float Var 变成 bool；`x.norm_()` 就地把 shape [3] 变成 [1]；`x.sort_()` 抛 RuntimeError；`_` 后缀在本仓库不表示就地 | 改白名单显式声明；对返回非 Var 的函数禁止生成 | 关键 |
| `Var.scatter`（无下划线）实际是就地的，与 torch 相反 | `misc/tensor_ops.py:2030` `return x.setitem(...)`；仓库自己在 `:2039-2041` 注释里承认它 in place 所以 scatter_add 要先 clone | `y = x.scatter(...)` 污染 x；同族三个 API 三种语义 | scatter 内部先 clone | 主要 |
| **转置结果携带隐藏标记，基张量被改写后 matmul 返回陈旧结果** | 标记 `_runtime/core_api.py:752-760`；消费 `nn/functional/matrix.py:118-121,180-232`。运行时复现（CPU/mkl）：`at=a.transpose(0,2,1); a.assign(zeros); jt.nn.matmul(at,b)` 期望 [4,4,4,6] 实得 [0,0,0,0]，而 `at.numpy()` 仍是正确值 | **静默算错**。触发条件极常见：任何 `w_t = w.transpose(...)` 后 optimizer step、load_parameters 或 114 个 foo_ 之一改写了 w。CUDA/cublas 路径同构 | 转置融合基于图结构而非可变对象属性，或在快路径校验版本号 | 关键 |
| dtype 提升唯一规则是"字节宽度取 max"，无 promotion lattice | `src/misc/nano_string.h:259-279,217-227`。偏离 torch：`uint8 * (1/255.)` → **float16**；`int64 * 2.0` → float64（torch float32）；`float16+bfloat16` → bfloat16（torch float32）；`int32+uint32` → int32 | 视觉前处理精度直接崩；GPU 上 float64 掉进 1/32 吞吐。Python 层用一次性 cast 局部打补丁而不是修表，未打补丁处（rad2deg/deg2rad）仍中招 | 引入区分 kind 与 width 的提升表，删除局部 cast | 关键 |
| 索引与计数类返回 int32 | `src/ops/where_op.h:30`(nonzero)、`misc/tensor_ops.py:1850`(randperm)、`:1265`(topk 空输入)、`pool/core_2d.py:108`(pool indices)，`:198` 显式 assert 不超过 2^31 | 与 torch int64 不匹配；超过 2^31 元素静默溢出 | 索引统一 int64 | 主要 |

## Module 与参数模型
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| parameters/named_parameters/state_dict/named_buffers/_buffers 是**五份独立抄写**的 DFS | `_runtime/core_api.py:1466,1615,1525,2221,2212`。parameters 按 id 去重，named_parameters 按**名字**去重。运行时验证：权重绑定时 `len(parameters())=3` 而 `len(named_parameters())=4` | 优化器与 transformers/peft/checkpoint 对同一模型看到不同的参数集合；`_buffers` 返回全部 Var（验证得 5 个键）而 named_buffers 只返回 3 个 | 一份带策略参数的遍历，其余作为视图 | 关键 |
| parameters/state_dict 是查询方法但会改名，判据是字符串长度 | `:1518` `if base + len(leaf) > len(p.name()): p.name(...)`；`:1590` 同 | 参数名字取决于先在哪一层调用了 parameters()，checkpoint 的 key 不稳定 | 名字由遍历路径产生，不写回 Var | 主要 |
| "是不是参数"由 6 个标记加 3 个名字集合共同决定，jittor 自家 BatchNorm 绕过了这套机制 | 标记见 `:2145-2211`；`nn/modules/normalization.py:33-39` 用 `object.__setattr__(buf,"is_buffer",True)` 而非 register_buffer。运行时验证：`bn.running_mean = jt.zeros(4)` 后 running_mean 出现在 parameters() 里 | 这正是 _buffer_names 机制要防的失效模式，却在自家 BN 上仍然存在 | 参数与缓冲区应是类型不是标记 | 主要 |
| Module.eval() 对所有参数 stop_grad，train() 靠 id(p) 索引的备份还原 | `:2094-2123`。运行时验证：子模块 eval 后父模块 train，子模块参数永久保持 stop_grad | 与 torch 语义不符（torch 的 eval 只影响 BN/dropout）；备份键为 id，Var 被替换或 GC 后 id 复用会还原到错误对象；"可训练吗"同时有四种表示 | eval/train 只切 is_train；冻结统一由 requires_grad 表达 | 主要 |
| 每种 hook 只能注册一个，prepend/always_call 被接受后忽略；hook 安装是**类级别**且不可撤销 | `:1841-1861`、`:1833-1839`（`_place_hooker` 交换 cls.__call__）。运行时验证：连续注册两个 forward hook 只有第二个触发；给一个 Linear 实例装 hook 后此后**每个** Linear 实例都走 hook 路径 | 与 torch 的多 hook OrderedDict 契约不兼容（accelerate/peft/transformers 都依赖）；`hasattr(cls,"__hooked__")` 沿 MRO 查找，给基类装过后子类不再安装 | hook 存实例级有序字典 | 主要 |
| .half()/.float16() 里的 amp 分支是死代码，且不转换 buffer | `:2281-2292`：`self._amp_level = -1` 紧接 `if self._amp_level >= 0:` 恒假，其下的整个 __half_call__ 不可达。运行时验证：`bn.half()` 后 weight=float16 而 running_mean=float32 | torch 的 half 转换参数与浮点 buffer，此处只转参数 | 删除死分支；转换范围含浮点 buffer | 主要 |
| `model.cuda(device)`/`npu(device)` 丢弃设备号 | `:1678-1685` 只设 `flags.use_cuda = 1`；`misc/tensor_ops.py:2849-2853` 同（npu 设的还是 use_cuda）；`:2604` `Var.cpu = clone`（完全没有设备迁移） | 多卡在 Python API 层面无入口；`x.cpu()` 是谎言 | 与设备模型一并处理 | 关键 |
| `x.to()` 的语义取决于 kwargs 书写顺序，且修改全局 flag | `misc/tensor_ops.py:2606-2618` `args += tuple(kargs.values())` 后只看 args[0] | `x.to(device='cuda', dtype=jt.float16)` 静默丢 dtype，反序则丢 device；`x.to('cuda')` 把**整个进程**切到 CUDA | 按 torch 签名显式解析 | 主要 |
| state_dict(to="torch") 强制 float32 | `:1607` `torch.Tensor(v.numpy())` | int/bool buffer 被转成 float32 | 用 from_numpy | 次要 |

## jt.Function 的契约
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| Function 实例把中间量存在 self 上，**同一实例被调用两次会静默改写第一次的反向** | `:2380-2418`（input_mask/output_mask 与用户的 self.x 同为实例属性）。运行时验证：`f=Mul(); o1=f(a,b); o2=f(a,c); jt.grad(o1,a)` 得到 10（应为 2），无任何警告 | 文档鼓励的 `func = MyFunc.apply` 安全，但 `f = MyFunc(); f(x); f(y)` 同样自然且静默算错。全仓 20+ 个子类用 self.saved | 每次调用创建一次性上下文对象，实例无状态 | 关键 |
| apply 接受关键字参数，__call__ 拒绝 | `:2440-2442` vs `:2380`。运行时验证：`F.apply(x, alpha=2)` 抛 TypeError | 签名撒谎 | __call__ 接受并透传 | 次要 |
| no_grad 下 Function 完全跳过 taping；29 处融合 kernel 以全局 no_grad 为启用条件 | `:2381-2382`；`grep getattr(jt.flags,"no_grad",0)` 29 处；`nn/backends/cudnn.py:41-44` fp16/bf16 卷积仅在 no_grad 下走 cuDNN | "是否推理"由全局 flag 而非 requires_grad 决定：`model.eval()` 不设 no_grad，因此 eval 模式下若无显式 `with jt.no_grad()`，全部 29 个融合 kernel 都不生效；fp16 训练的卷积退回 reindex 路径（其 docstring 自承会物化约 30GB 中间量） | 启用条件改为"输出不需要梯度" | 主要 |
| flag_scope 把备份存在实例上，`@jt.no_grad()` 装饰器递归调用后**永久泄漏 no_grad=1** | `:104-112`（__call__ 复用同一 scope 实例）加 `:137-167`（self.flags_bk）。运行时验证：递归调用后 `jt.flags.no_grad` 仍为 True | 此后整个进程的自动微分静默失效，训练 loss 不下降且无报错。非线程安全非可重入 | flags_bk 改局部栈；__call__ 每次新建 scope | 关键 |
| register_hook 用 swap 原地替换 Var 且不返回可移除 handle | `:2460-2483`；同文件 `:1336` 已有 _RemovableHandle 仅供 Module hook | 同一概念两套返回契约；Var hook 无法卸载 | 统一 | 次要 |

## 后端分派：一个全局开关，18 种写法
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| Python 层 57 处直接判 use_cuda，共 18 种不同守卫写法 | 排除 compat/extern/src/测试后 57 处；形态含 `use_cuda and jt.cudnn`、`use_cuda and jt.compiler.is_cuda`、`use_cuda and not has_acl and not has_rocm`、`not (use_cuda and no_grad)` 等 18 种。另有 41 处 is_cuda/has_acl/has_rocm/use_acl 判断 | 没有 dispatch 层，加一个后端要改 98 处；守卫条件差异从不被交叉验证 | 按（算子,设备,dtype）注册的分派表 | 关键 |
| matmul 有四条路径，守卫条件互不一致且用**字符串子串**判 dtype | `nn/functional/matrix.py:99-112`（cublas 2D）、`:196-215`（cublas batched）、`:216-232`（mkl batched 仅 float32）、`:233-256`（reindex）。2D 路径检查两个操作数的 complex，batched 只检查 a；`"float" in dtype` 会匹配 bfloat16/float64 | 同一个 matmul 在不同设备与 dtype 下走四种累加顺序与精度；bmm_transpose 不加 amp_reg 而 matmul 加 | dtype 用枚举；四条路径共用能力表 | 主要 |
| amp_reg 的位常量散落 6 个文件无命名常量无文档，concat 用赋值而非按位或 | `_runtime/core_api.py:417-424,459-466,471`；`matrix.py:54,176`；`convolution.py:52`；`modules/convolution.py:147`；`misc/concatenation.py:49` `amp_reg=4`（**赋值**，清掉用户设的偏好） | jt.concat 是最高频算子之一，会在其作用域内静默改写全局 AMP 策略，且其内部根本没有 reduce 算子；array() 的 amp 逻辑要求 numel!=1 而 random() 不要求，同一段逻辑抄了两遍且不同 | 常量命名导出；一律 `|=` | 主要 |
| unique 在 CUDA 上有四条路径，其中一条强制回落 CPU，而 CPU 内核的比较器把元素截断成 int | `misc/tensor_ops.py:761-788`；`:766-769` 用 `bool(...all())` 在惰性图中做**主机同步**；`:815` `int lhs = @input_flatten(a,i)` | 注释称 CPU 路径是 correct 实现，实际 float/int64 的排序键被截断，去重后可能残留重复值，而 CUDA 上恰恰**必然**走进这条路径 | 比较器用 input_flatten_type；分派移入算子层 | 关键 |
| cumsum CPU 走 numpy 主机回调、CUDA 走 CUB，两套实现两套反向 | `:1370-1373`，CPU 实现 `:1313-1331` | 主机同步打断惰性图；两条路径 dtype 行为不同；`assert(dim >= -1 ...)` 只允许 -1 一个负数 | 统一为一个 scan 算子 | 主要 |
| isnan/isinf/isfinite 的非 ACL 内核把值 cast 成 float | `:2576-2586` `"isinf(float(x))"` vs ACL 路径 `:2567-2574` | float64 的 1e300 在 CPU/CUDA 上被判为 inf、在 ACL 上不是——**同一 API 跨后端结果不同** | 用 in0_type | 主要 |
| Conv 在 __init__ 时按当时的 use_cuda 决定是否用 depthwise 路径 | `nn/modules/convolution.py:98-99` | CPU 上构造再 cuda() 的模型永远不用 depthwise kernel；DepthwiseConv 只好重写 __call__ 在 CPU 上绕过整个 Function tape | 分派移到 execute | 主要 |

## 同一概念的多份实现
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| **两个公开的 AvgPool2d，数值不同** | `pool/layers.py:20` vs `nn/modules/pooling.py:11`（其 docstring 自承 Unlike jittor.pool.AvgPool2d this honours count_include_pad exactly）。运行时验证：同参数下 `jt.nn.AvgPool2d` → [2.5,4,8.5,10]，`jt.pool.AvgPool2d` → [1.11,2.67,5.67,10]。且 **jt.nn.AvgPool3d 是未修正的那份** | jt.nn 命名空间内 2D 与 3D 遵循两套语义 | 删除旧实现，jt.pool 转发 | 关键 |
| nn.Conv2d.execute 与 nn.functional.conv2d 是两份独立抄写且已分叉 | `modules/convolution.py:130-195` vs `functional/convolution.py:46-99`。模块版 `ww.compile_options = xx.compile_options = {"G":G,"C":C}`，functional 版只有 `{"G":G}`；模块版有输出尺寸校验，functional 版没有。而 `Conv._conv_forward` 调用的是 **functional 版** | 同一层的两种调用方式生成不同 JIT kernel；BatchNorm/LayerNorm/GroupNorm 同样各抄一份，其中 `nn.functional.batch_norm(training=True)` 根本不走融合 kernel | module 层只做参数管理，一律委托 functional | 关键 |
| BatchNorm 训练路径在 sync 与非 sync 分支用两套归一化数学（**2026-09-03 补充第三处，比原描述严重**：sync 分支的统计量用 `E[x²]-E[x]²`，非 sync 分支用两遍法 `mean((x-mean)²)`。float32 只有约 7 位有效数字，一旦 `var/mean²` 掉到 1e-7 以下，两个平方项就在它们仅有的位数上相等，**差出来的基本是舍入噪声**——而这个公式只在 MPI 打开时才跑，所以这处精度崩塌**只在分布式作业里出现**。取证要点：用 mean 0 / std 1 的良态输入，两个公式相等到 1e-5、缺陷完全藏住；必须用 mean 100 / std 0.05 这种「均值远大于标准差」的输入才会以约 7e-2 的相对误差暴露。**这是「同一概念两份实现」这类缺陷的通用取证要点：默认参数与良态输入下两份实现往往恰好相等，要专门去找让它们分开的那个输入**） | `modules/normalization.py:56-71` | 开不开 MPI 决定 BN 的数值与反向公式 | 统一 | 主要 |
| reduction 参数在同一个文件里有四种解析方式、三种错误行为 | `functional/loss.py:32-38`（静默 none）、`:53`（委托）、`:56`（无 reduction）、`:113/134/153`（显式 raise）。运行时验证：`reduction='MEAN'` 时 cross_entropy 静默返回未归约、l1_loss 静默返回 mean、mse_loss 抛深层 RuntimeError | 拼错一个字符串三个损失三种结局，其中两种不报错 | 一个 _reduce helper | 主要 |
| models/ 内 SqueezeExcitation ×4、StochasticDepth ×4、ConvNormActivation ×3，默认值互不兼容 | `efficientnet.py:99`(SiLU)/`regnet.py:68`(ReLU)/`maxvit.py:134`(ReLU)/`mobilenet_v3.py:64`（第二个位置参数含义相反）；`convnext.py:28`=`efficientnet.py:40`=`maxvit.py:72` 逐字节相同 | **已导致架构缺陷**：`efficientnet.py:86-94` 把 activation_layer=None 替换成 SiLU 后 `if activation_layer is not None` 恒真，于是声明"project (1x1, no activation)"的投影层被强加 SiLU，b0–b7 全系列与 torchvision 不符，而 regnet 的同名类处理是对的 | 抽到 models/_utils.py 单份实现 | 关键 |
| 分布类一半不继承 Distribution；sample() 是否 detach 分两派 | `distributions.py:221/253/298/343/378/407` 为裸类；加 stop_grad 的 `:312,543,562,853`，不加的 `:657`(Beta)、`:698`(Gamma)、`:766`(Dirichlet)、`:804`(LogNormal)、`:899`(MultivariateNormal) | isinstance 对最常用的 Normal/Categorical 为 False；五个分布的 sample 会把梯度接回参数（RL/VI 中缓慢跑偏且不报错）；kl_divergence 对未覆盖的 7 个分布**静默返回 None** | 全部继承基类，sample 在基类实现一次 | 主要 |
| 两套契约互斥的 LR scheduler 并存且相互污染 | 新式 `optim/schedulers.py:5-43`（写 pg['lr']）vs 旧式 `optim/legacy_schedulers.py:115-119`（写全局 optimizer.lr）；两者都可达。Optimizer.step 用 `pg.get("lr", self.lr)` | 用过一次 LambdaLR 之后旧式 scheduler 的行为从"改全局"翻转成"改每组"——**行为取决于历史**。旧式内部也不自洽：StepLR.get_lr 是死代码，MultiStepLR 的 get_lr 与 update_lr 各施加一次 gamma | legacy 继承新基类 | 主要 |
| normalize 两份语义不同的同名实现 | `misc/tensor_ops.py:604`（加性 eps，docstring 默认值与签名不符）vs `nn/functional/vector.py:39`（对分母 clamp，torch 语义）；jt.normalize 指向前者 | 同名不同义 | 合并 | 主要 |
| init.py 里两张互相矛盾的 gain 表、两套 fan 算法 | `:430-443`（无 selu，未知 key 抛裸 KeyError）vs `:656-675`（有 selu，抛 ValueError）；fan 计算 `:339/392/536` 用循环 vs `:428` 用 numel | `kaiming_uniform_(w, nonlinearity='selu')` 直接 KeyError | calculate_std 调用 calculate_gain | 次要 |
| linalg.py 里两个 helper 复制了 12 次 | `:55,69,102,142,180,196,255,305` 等 | 改一处要同步 12 处 | 提到模块级 | 次要 |

## 参数被接受后静默忽略（均已确认函数体内无引用）
| 位置 | 被忽略的参数 | 后果 | 严重度 |
| --- | --- | --- | --- |
| `nn/functional/activation.py:8,32,178,279` | relu/leaky_relu/silu/mish 的 inplace | 显存敏感代码以为省了内存 | 次要 |
| `nn/functional/normalization.py:31-45`、`modules/normalization.py:104-127` | instance_norm 的 running_mean/var/momentum；InstanceNorm 的 momentum/sync/is_train | 有 running stats 的 InstanceNorm 静默退化 | 主要 |
| `models/resnet.py:96` | zero_init_residual | ResNet-50+ 收敛行为与 torchvision 不同 | 主要 |
| `linalg.py:455`、`:670` | svd 的 compute_uv/driver；inv_ex 的 check_errors（info 恒 0，而 docstring 明说调用方靠 info==0 建掩码） | 奇异矩阵掩码永远拿不到 1 | 主要 |
| `misc/tensor_ops.py:2192` | ctc_loss 的 zero_infinity | inf loss 时梯度不清零 | 主要 |
| `misc/tensor_ops.py:1263`、`:150` | topk 的 sorted、sort 的 stable | | 次要 |
| `init.py:448,479` | kaiming_* 的 generator | 用户以为控制了复现性 | 次要 |
| `dataset/dataset.py:150-155` | pin_memory/persistent_workers（注释明写 mirror torch's DataLoader semantics，实际 worker 永远常驻） | 默认值 False 的行为恒等于 torch 的 True | 主要 |
| `autograd/functional.py:327-331` | vjp/jvp 的 strict：报错分支依赖 grads_i is None，而 jt.grad 对无关输入返回**零 Var** | strict=True 是静默失效的死开关 | 主要 |
| `fft/__init__.py:194-199` | fftfreq/rfftfreq 的 dtype/device | 恒返回 float32 | 次要 |

统计：范围内 106 处无消息 assert、28 处 `except Exception:`、7 处 `except: pass`。

## 导入期的全局副作用与 monkeypatch 顺序
| 问题 | 证据 | 后果 | 修改方向 | 严重度 |
| --- | --- | --- | --- | --- |
| import jittor 重置 Python/numpy/cupy 的全局随机种子 | `misc/tensor_ops.py:1879-1880` 模块顶层 set_global_seed，内部调 random.seed/np.random.seed/cupy.random.seed，末尾裸 except | 用户在 import 之前设的 np.random.seed(0) 被静默抹掉 | 删掉顶层调用 | 关键 |
| import jittor.dataset 永久替换 PIL.Image.open | `dataset/dataset.py:39-40` 加 `dataset/utils.py:57-61`（setattr 无卸载接口） | 全进程的 PIL.Image.open 变成非函数对象（signature/pickle/wraps 全失效） | 改显式 opt-in 的 contextmanager | 关键 |
| 六个独立的 monkeypatch 安装器，顺序隐式且相互依赖 | `nn/__init__.py:104` → `__init__.py:109` → `:111` → `:137-154` → `:162` ACL → `:178/196` HCCL/NCCL → `:203` compat → `:215` optim → `:219`。全仓 169 处 `Var.x = ...`（8 个文件） | 顺序即语义无处声明；`nn/functional/softmax.py:35-38` 注释明说 Backend integrations replace the public symbol at runtime——monkeypatch 是**契约**而非补丁 | 安装序列显式写成有序清单并加断言 | 主要 |
| install_full_reduce_fast_path 只替换 Var.sum/mean，不替换 jt.sum/jt.mean | `nn/backends/full_reduce_cuda.py:147-177`。运行时验证：`jt.sum is jt.Var.sum` 为 False | `x.sum()` 走 CUB 两阶段（可复现），`jt.sum(x)` 走 atomicAdd（不可复现且慢 70 倍）——同一语义两种数值 | 快路径装在算子层 | 主要 |
| nn facade 把 39 个下划线私有名导出为公开属性，内部一律通过 jt.nn.* 晚绑定回调 | dir(jt.nn) 中 39 个下划线名；`nn/backends/cudnn.py:100` 在自己文件里用 `jt.nn._CUDNN_3D_HALF_DTYPES` 引用 4 行之上定义的常量 | 私有公开边界消失；任何对 jt.nn.X 的替换会改变无关函数的内部行为 | 内部调用用模块局部名 | 主要 |
| 星号导出泄漏 np/math/time/Sequence 且代码**依赖**这个泄漏 | tensor_ops 无 __all__ 使 jt.np/jt.math/jt.time 成为公开属性；`shape_transforms.py:29` `isinstance(shape[0], jt.misc.Sequence)`、`:53` `jt.misc.np.array(...)` | 任何加 __all__ 的清理会直接打断 repeat() | 先修依赖再加 __all__ | 次要 |

## 已确认会静默算错的缺陷（并行路径从不交叉验证的直接结果，全部 CPU 可复现）
| # | 缺陷 | 证据 | 严重度 |
| --- | --- | --- | --- |
| H1 | 分组 conv3d 在无 cuDNN 时必定失败 | `functional/convolution.py:173-178` ww 的 reindex 形状写成 [...,oh,ow,od,Kh,Kw,Kd] 而 xx（`:164`）是 [...,od,oh,ow,Kd,Kh,Kw]，索引 i7/i8/i9 随之错位。运行时：`Check failed xshape(3) == yshape(2)` | 关键 |
| H2 | Pool3d 的 return_indices 内核第三层循环条件用错变量 | `pool/core_3d.py:84` `for (int r = k4; q < k4_; ++r)`（应为 r < k4_）→ 无上界，越界读或死循环 | 关键 |
| H3 | Pool3d 的 CUDA 反向用输入形状当循环上界（CPU 反向与 2D 版都是对的） | `core_3d.py:162-165` 用 out_shape4/3/2，而 launch 配置 `:170-172` 与 CPU 反向 `:193-198`、`core_2d.py:147-148` 都用 pout_shape | 关键 |
| H4 | MaxUnpool2d/3d 在 stride != kernel_size 时用错误行宽解码索引 | `pool/unpool.py:60-67` 用 xshape3（池化后宽度），前向编码用的是原始宽度（`core_2d.py:82`）；相等分支 `:57` 用的正是正确的 yshape3 | 主要 |
| H5 | eigh 反向在 dout 全零时不写输出缓冲，返回未初始化内存 | `linalg.py:581-586` 的 `if np.any(dout):` 无 else；对照 slogdet `:1127-1128` 显式 copyto 0 | 关键 |
| H6 | _autograd_grad 用错变量，vjp/jvp 的种子梯度与输出错位 | `autograd/functional.py:190-202`：new_grad_outputs 构造后从未被读，`:202` 的 zip 用的是未过滤的 grad_outputs | 关键 |
| H7 | irfft 对实数输入静默算错（Var.real 对实数返回自身），显式传 n 时重建长度也用错 | `fft/__init__.py:150-162`；同模块 `:68-73` 已有判别函数但 irfft 绕开了它 | 关键 |
| H8 | ReduceLROnPlateau 把全局 lr 降 factor^N | `optim/legacy_schedulers.py:69-77` 循环内每轮从被改过的 optimizer.lr 重读 old_lr；而 jittor 的 param group 默认就是没有 lr 键，所以这是默认路径 | 关键 |
| H9 | `unique(return_counts=True, return_inverse=False)` 静默丢弃 counts | `misc/tensor_ops.py:973-981` 缺分支 | 关键 |
| H10 | Adan 在 param_group 循环内调用全局的 clip_grad_norm | `optim/algorithms/adan.py:70-75` 加 `optim/base.py:102-112`（遍历全部 group）→ 裁剪被复合 N 次 | 关键 |
| H11 | zero_grad 只翻标志位不清缓冲，导致 step 后的 clip_grad_norm 静默空转 | `optim/base.py:152-153` 加 `:100`；post_step（`:266`）每步都调 zero_grad | 关键 |
| H12 | Adam 用全局 n_step 做偏差修正 | `optim/algorithms/adam.py:62,75` 加 `optim/base.py:221`；而 `base.py:155-187` 的 docstring 正在推荐梯度累积写法 | 关键 |
| H13 | worker 里任何异常都变成给父进程发 SIGINT | `dataset/dataset.py:321-326` | 关键 |
| H14 | mp_log_v=0 反而打开调试日志 | `dataset/dataset.py:37` `os.environ.get("mp_log_v", 0)` 返回字符串 "0"（真值）；相邻的 `:46` 却做了 int 转换 | 关键 |
| H15 | Pillow 版本用字符串比较，Pillow ≥10 全部判断反向 | `transform/function_pil.py:541` `PILLOW_VERSION < "5.2.0"`（"10.4.0" < "5.2.0" 为真）、`:652` `PILLOW_VERSION[0] >= '5'`（'1' >= '5' 为假） | 关键 |
| H16 | Dataset.__deepcopy__ 把 id(obj) 写进 memo | `dataset/dataset.py:468` `memo[d] = id(obj)`（应为 obj） | 关键 |
| H17 | LogitRelaxedBernoulli = RelaxedBernoulli（torch 中前者返回 logit）；RelaxedOneHotCategorical 继承离散 log_prob 且 base_dist 自引用 | `distributions.py:546,549-562` | 主要 |
| H18 | ComplexNumber.__rsub__ 虚部符号错、__imatmul__ 操作数顺序反 | `nn/legacy_complex.py:174-180,237-244` | 主要 |
| H19 | 稀疏卷积对重复坐标 CPU 保留首个、CUDA 由 atomicCAS 竞争决定；neighbors 缓存只校验 shape | `sparse/convolution.py:74` vs `:137-138`；`:232-235`（而 docstring 正在鼓励复用缓存） | 主要 |
| H20 | spmm 先转稠密再 matmul（源码标注 TODO）；to_dense 对 COO 重复索引是覆盖而非求和 | `sparse/coo.py:66-72,60` | 主要 |

## 优先级
1. **先修 5 个会静默算错且入口常见的**：转置标记陈旧（任何 optimizer step 都可触发）、Function 实例复用、no_grad 泄漏（会让整个训练静默失效）、tied weight 下参数集合不一致、unique。五条都不需要架构改动，但当前没有任何测试能发现它们。
2. **再补 H 类孤立缺陷**：都是单点修复，但每条都说明这条路径从未被跑过，修的同时补对应对拍用例。
3. **然后做三件消除结构性成本的事**：视图与存储模型（是 114 个 foo_ 方法、转置标记、compat 的回写链三个问题的共同根因）；参数模型（把 26 个标记收敛为类型，五份遍历收敛为一份）；dispatch 层（把 98 处判断收进注册表，这是多卡落地的前置条件）。
4. **可以立刻做的清理**：import 期重置随机种子与 monkeypatch PIL.Image.open 是两行删除。被静默忽略的参数应统一改为传非默认值时 warn 或 raise。
