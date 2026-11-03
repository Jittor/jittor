# 源码架构与模块边界

[本栏目目录](index.md) · [English](source-architecture.en.md)

先用本页确定改动属于哪个模块，再查对应实现契约。源码 checkout 与打包边界见[仓库布局](repository-layout.md)。

| 要修改的内容 | 详细契约 |
| --- | --- |
| 原生运行时、状态与提交 | [原生运行时、状态与提交](runtime/native-runtime.md) |
| 算子注册、分派与后端回退 | [算子注册、分派与后端回退](runtime/operator-dispatch.md) |
| Torch 前端的归属与 dtype 边界 | [Torch 前端的归属与 dtype 边界](compatibility/frontend-boundaries.md) |
| 后端构建与运行时资源 | [后端构建与运行时资源](backends/backend-build.md) |

契约保留原复查基线 `e3c369acb`；整理文档不刷新硬件验收状态。

## 原则

1. **一个物理归属者。** 一个公开领域就是一个常规包；它**不再有第二棵藏着真实实现的
   私有树**。
2. **导入跟随归属。** 实现的元数据与回溯信息报告**拥有该代码的那个模块**。再导出模块
   不递归改写 `__module__`。
3. **组合保持浅层。** 包的 `__init__.py` 负责组合与导出公开名字；大块实现住在有意义的
   子模块里。
4. **运行时路径是契约。** 编译器加载的 C++/CUDA 资源只能随着显式的编译器与打包迁移一起
   移动。
5. **兼容是分层的。** 框架原生能力、可复用的兼容机制、导入 shim 和下游集成，各有不同的
   归属者。
6. **每次搬动都保持行为。** 重构要保留公开名字、承诺过的可调用对象标识、支持范围内的
   pickle 行为、后端分派，以及定向回归覆盖。

## 当前的领域划分

```text
python/
├── jittor/
│   ├── __init__.py              # 根组合与运行时初始化
│   ├── __init__.pyi             # 公开根的类型标注面
│   ├── _core/                   # 原生 Python API 的实现域
│   │   ├── api.py               # 编译核心引导之后的显式组合
│   │   ├── var.py               # 张量构造、运算与 Var 绑定
│   │   ├── module.py            # Module 与参数/缓冲归属
│   │   ├── function.py          # 自定义自动微分上下文与梯度钩子
│   │   ├── hooks.py             # 可移除句柄与钩子支持
│   │   ├── flags.py             # 原生作用域与共享运行时状态
│   │   ├── arg_policy.py        # 显式的不支持/忽略参数策略
│   │   └── diagnostics.py       # 日志、性能分析、进程作用域与退出清理
│   ├── _runtime/
│   │   ├── core_api.py          # _core.api 的同对象历史别名
│   │   ├── composition.py       # 显式的命名空间发布
│   │   ├── install_order.py     # 有序的安装器注册与校验
│   │   └── state.py             # 注入的原生 Flags 视图，不做引导期导入
│   ├── serialization/
│   │   └── native.py            # 原生 save/load 与安全 pickle 实现
│   ├── build/                   # 编译器/引导的实现归属
│   │   ├── compiler.py          # 启动状态与构建编排
│   │   ├── codegen.py           # 原生绑定/注册的源码生成
│   │   ├── compilation.py       # 编译与自定义扩展操作
│   │   └── utils/               # 独立工具，以 jittor_utils 名义导入
│   ├── contrib/                 # 贡献算法与组合助手
│   │   ├── ccl/                 # 连通域标记
│   │   ├── loss3d/              # Chamfer 与推土机损失
│   │   ├── math_util/           # gamma 函数与共享原生资源
│   │   └── einops/              # 内置的张量表达式算法
│   ├── nn/                      # 神经网络公开 API
│   │   ├── modules/             # 有状态的 Module 实现
│   │   ├── functional/          # 无状态的张量函数
│   │   ├── backends/            # cuDNN 与只读的钩子调用适配
│   │   ├── utils/               # weight norm 等构造助手
│   │   └── attention.py
│   ├── autograd/                # 函数式自动微分
│   ├── fft/                     # 可导的原生 FFT 命名空间
│   ├── linalg/                  # 分解、求解、范数与缩并
│   ├── distributions/           # 概率分布族与共享约束
│   ├── init/                    # 初始化族与共享的 fan/gain 规则
│   ├── ops/                     # 张量、索引、归约与形状实现
│   ├── misc/                    # 已弃用的同对象张量 API 门面
│   ├── pool/                    # 已弃用的同对象池化 API 门面
│   ├── optim/                   # 优化器门面与算法模块
│   ├── sparse/                  # COO 稀疏张量与稀疏卷积
│   ├── compat/
│   │   ├── torch/               # 规范的 Torch 风格 API 兼容
│   │   ├── fsdp2/               # 分布式 FSDP2 兼容
│   │   ├── triton/              # Triton API 桥接与部署命令
│   │   ├── shim/                # Torch shim 运行时与部署命令
│   │   ├── module_patcher.py
│   │   └── external_backend.py
│   ├── selftest.py              # 安装后的冒烟测试
│   ├── tools/                   # 用户工具，含 benchmarking.py
│   ├── backends/                # 源码 checkout 的包路径桥接
│   └── distributed/             # 原生启动、rendezvous 与通信助手
└── jittor_utils/                # 通向 jittor/build/utils 的独立导入桥
```

## 包组合契约

下面两个公共领域由各自的规范归属者组合，不另建平行实现树。

## 神经网络 API

`jittor.nn` 是公开包。有状态的层在 `nn.modules`，无状态运算在 `nn.functional`，可选的
加速路径在 `jittor.backends.cuda.kernels`。`nn.backends` 保留的是**调用适配器，不是 CUDA
实现**。公开的再导出必须指向规范实现对象；只有当包装器确实在强制某条真实的 API 契约时
才成立。

依赖方向是：

```text
nn.modules -> nn.functional -> Jittor 张量/核心运算
nn.backends ----------------> 显式的后端/编译器接口
```

**functional 模块不得导入有状态的层实现。** 后端适配器必须保持可选，并在其工具链不可用
时以**可执行的能力错误**失败。

## 杂项与池化 API

`jittor.ops` 拥有张量、形状、索引与组合运算。历史的 `jittor.misc` 导入仍是**已弃用的
同对象门面**，包括旧的子模块路径。公开名字与 pickle 全局符号解析到规范实现；
**`misc` 下不存在第二棵可编辑的数学实现树。**

池化的数学与参数校验在常规的 `nn.functional.pooling` 包里，按平均池化、2D/3D 核心、
自适应、1D 和反池化分成不同归属者。`nn.modules.pooling` 只保存构造参数并调用这些无状态
实现；**functional 调用不会构造临时 Module**。CPU/CUDA 的生成源码保持单一来源，后端发射
构造器停在既有的注册边界上。

`jittor.pool` 及其历史子模块是**已弃用的再导出**。`nn.modules.pooling_legacy` 以原类名
保留三个历史类：`AvgPool2d` 和 `AvgPool3d` 为旧 pickle 保留其转发层状态，而
`AdaptiveAvgPool2d` 保留其固定窗口算法——**后者刻意不同于当前 NN 的重叠分箱算法；
一次布局搬动不会静默改变它的数值规则。** 自适应窗口的中间量是函数局部变量，不是持久的
Module 状态。

历史开关 `pool_use_code_op` 的唯一归属者是 `nn.functional.pooling._state`。`jt.pool`、
`jt.nn` 和 functional 池化包暴露的是对**同一个值**的实时读写，包括临时的属性覆盖/还原。
**后端执行绝不读取被复制过的门面值。** NN 在加载旧门面之前先构造它的 functional 与
module 归属者，因此这条兼容导入不会造成引导循环。

`jittor.sparse` 在不同子模块中分别拥有坐标格式稀疏张量和稀疏神经网络 kernel。历史名字
`jittor.nn.sparse` 是 `jittor.sparse.convolution` 的同对象别名；`jittor.nn` 与
`jittor.nn.functional` 再导出那些规范可调用对象。

`jittor.autograd` 拥有函数式自动微分。`jittor.fft` 拥有原生 Jittor 与 Torch 模式共享的
可导 FFT/shift/frequency 命名空间。拼接与索引在 `jittor.ops`，池化在 `jittor.nn`，
优化后的 softmax 在 `jittor.backends.cuda.kernels.nn`，权重归一化在 `jittor.nn.utils`。
**历史的根级拼写只是导入别名**，不保留物理源文件或包装实现。

## 导入与初始化规则

Torch 的 dtype 对象及其原生/NumPy 消费点遵循[前端 dtype 边界](compatibility/frontend-boundaries.md#dtype-边界)。

- 模块导入**不得**编译 kernel、下载资源、修改源码 checkout，或静默安装外部包。
- **注册必须幂等。** 重新导入一个兼容模块不得把同一个可调用对象包两次，也不得创建第二个
  模块对象。
- 可选依赖的检查发生在**运算边界**，除非导入期发现本身就是那个 API。
- 宽泛的异常处理器可以标注并**重新抛出**失败；**不得**把一个只安装了一半的兼容接口转成
  表面上的成功。
- 昂贵的导入不进入仅做收集的结构测试。
- 同一文件里的顶层定义**不得**被后面的定义静默替换。跨文件的相同实现同样被扫描；保留的
  重复需要属于一个窄的、经过评审的类别，例如独立的部署入口、后端代码生成模板、模型局部的
  架构块，或历史的序列化读取器。

## 重构流程

每次模块搬动：

1. 清点定义、赋值、导入、注册和消费者；
2. 显式定义规范目的地以及任何兼容别名；
3. 搬动一个**连贯的领域切片**，不夹带无关的行为改动；
4. 尽可能**机械地**比对搬动前后的定义集合与公开导出；
5. 测试导入标识、公开调用、动态分派、适用时的序列化，以及相关的 CPU/加速器行为；
6. 删除过渡源码路径，并把它加进结构门禁；
7. 在**同一次改动中**更新长期文档与活跃链接。

**搬动之后不得保留两份可编辑的实现。** 兼容必须委托给规范对象，并且有退出条件。

## 验收

一次源码布局改动在满足下列条件时可以接受：

- 导入与公开名字保持其有文档记载的行为；
- 不残留任何历史实现树或根级兼容文件；
- wheel 含有全部必需的运行时资源，并排除仅供仓库使用的测试/工具；
- `bash tools/check_repo_layout.sh` 通过；
- 定向测试、结构测试和每一个受影响的后端门禁都通过；
- 任何**刻意的不兼容**都记入发布说明。
