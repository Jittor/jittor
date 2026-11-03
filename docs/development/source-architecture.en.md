# Source architecture and module boundaries

[Section contents](index.en.md) · [中文](source-architecture.md)

Use this page to decide where a change belongs, then follow the relevant implementation contract. For checkout and package boundaries, read [repository layout](repository-layout.en.md).

| Change | Detailed contract |
| --- | --- |
| Native runtime, state and submission | [Native runtime, state and submission](runtime/native-runtime.en.md) |
| Operator registration, dispatch and fallback | [Operator registration, dispatch and fallback](runtime/operator-dispatch.en.md) |
| Torch frontend ownership and dtype boundaries | [Torch frontend ownership and dtype boundaries](compatibility/frontend-boundaries.en.md) |
| Backend builds and runtime resources | [Backend builds and runtime resources](backends/backend-build.en.md) |

The contracts retain their original review baseline `e3c369acb`; reorganizing this text does not renew hardware validation.

## Principles

1. **One physical owner.** A public domain is an ordinary package; it **no longer has a second
   private tree hiding the real implementation**.
2. **Imports follow ownership.** Implementation metadata and tracebacks report **the module
   that owns that code**. Re-export modules do not recursively rewrite `__module__`.
3. **Composition stays shallow.** A package's `__init__.py` composes and exports public names;
   large implementations live in meaningful submodules.
4. **Runtime paths are contracts.** C++/CUDA resources loaded by the compiler move only
   together with an explicit compiler and packaging migration.
5. **Compatibility is layered.** Framework-native capabilities, reusable compatibility
   mechanisms, import shims and downstream integrations each have a different owner.
6. **Every move preserves behaviour.** A refactor preserves public names, promised callable
   identity, pickle behaviour within the supported range, backend dispatch, and targeted
   regression coverage.

## Current domain partitioning

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

## Package composition contract

The public NN and tensor domains below are composed from their canonical owners.

## Neural network API

`jittor.nn` is the public package. Stateful layers live in `nn.modules`, stateless operations
in `nn.functional`, and optional accelerated paths in `jittor.backends.cuda.kernels`.
`nn.backends` holds **call adapters, not CUDA implementations**. Public re-exports must point
at the canonical implementation object; that holds only when the wrapper really enforces a real
API contract.

The dependency direction is:

```text
nn.modules -> nn.functional -> Jittor 张量/核心运算
nn.backends ----------------> 显式的后端/编译器接口
```

**A functional module must not import a stateful layer implementation.** Backend adapters must
stay optional and fail with an **actionable capability error** when their toolchain is
unavailable.

## Miscellaneous and pooling APIs

`jittor.ops` owns tensor, shape, indexing and composition operations. The historical
`jittor.misc` import remains a **deprecated same-object facade**, including the old submodule
paths. Public names and pickle global symbols resolve to the canonical implementation; **there
is no second editable math implementation tree under `misc`.**

Pooling mathematics and argument validation live in the ordinary `nn.functional.pooling`
package, split into separate owners for average pooling, 2D/3D cores, adaptive, 1D and
unpooling. `nn.modules.pooling` only keeps constructor arguments and calls these stateless
implementations; **a functional call does not construct a temporary Module**. CPU/CUDA generated
source keeps a single source of truth, and backend emission constructors stop at the existing
registration boundary.

`jittor.pool` and its historical submodules are **deprecated re-exports**.
`nn.modules.pooling_legacy` keeps three historical classes under their original class names:
`AvgPool2d` and `AvgPool3d` keep their forwarding-layer state for old pickles, while
`AdaptiveAvgPool2d` keeps its fixed-window algorithm -- **the latter deliberately differs from
the current NN's overlapping-bin algorithm; a layout move does not silently change its
numerical rule.** The adaptive-window intermediates are function locals, not persistent Module
state.

The sole owner of the historical switch `pool_use_code_op` is
`nn.functional.pooling._state`. `jt.pool`, `jt.nn` and the functional pooling package expose
live reads and writes of **the same value**, including temporary attribute override and
restoration. **Backend execution never reads a copied facade value.** NN constructs its
functional and module owners before loading the legacy facade, so this compatibility import
does not create a bootstrap cycle.

`jittor.sparse` owns coordinate-format sparse tensors and sparse neural-network kernels in
separate submodules. The historical name `jittor.nn.sparse` is a same-object alias of
`jittor.sparse.convolution`; `jittor.nn` and `jittor.nn.functional` re-export those canonical
callables.

`jittor.autograd` owns functional automatic differentiation. `jittor.fft` owns the
differentiable FFT/shift/frequency namespace shared by native Jittor and Torch mode.
Concatenation and indexing are in `jittor.ops`, pooling in `jittor.nn`, the optimized softmax in
`jittor.backends.cuda.kernels.nn`, and weight normalization in `jittor.nn.utils`.
**Historical root-level spellings are import aliases only**, with no physical source file or
wrapper implementation kept.

## Import and initialization rules

Torch's dtype objects and their native/NumPy consumption points follow the [frontend dtype boundary](compatibility/frontend-boundaries.en.md#dtype-boundary).

- A module import **must not** compile kernels, download resources, modify the source checkout,
  or silently install external packages.
- **Registration must be idempotent.** Re-importing a compatibility module must not wrap the same
  callable twice or create a second module object.
- Optional dependencies are checked at the **operation boundary**, unless the import-time
  discovery itself is that API.
- A broad exception handler may annotate and **re-raise** a failure; it **must not** turn a
  half-installed compatibility interface into an apparent success.
- Expensive imports do not enter collection-only structural tests.
- A top-level definition in the same file **must not** be silently replaced by a later
  definition. Identical implementations across files are scanned as well; retained duplicates
  must belong to a narrow, reviewed category, such as an independent deployment entry point,
  backend code-generation templates, model-local architecture blocks, or historical
  serialization readers.

## Refactoring process

Every module move:

1. inventory definitions, assignments, imports, registrations and consumers;
2. explicitly define the canonical destination and any compatibility aliases;
3. move **one coherent domain slice**, carrying no unrelated behavioural changes;
4. compare the definition sets and public exports before and after **mechanically** where
   possible;
5. test import identity, public calls, dynamic dispatch, serialization where applicable, and the
   relevant CPU/accelerator behaviour;
6. delete the transitional source path and add it to the structural gate;
7. update the long-term documentation and active links **in the same change**.

**Two editable implementations must not survive after a move.** Compatibility must delegate to
the canonical object, and must have exit criteria.

## Acceptance

A source-layout change is acceptable when all of the following hold:

- imports and public names keep their documented behaviour;
- no historical implementation tree or root-level compatibility file remains;
- the wheel contains all required runtime resources and excludes repository-only tests/tools;
- `bash tools/check_repo_layout.sh` passes;
- targeted tests, structural tests and every affected backend gate pass;
- any **deliberate incompatibility** is recorded in the release notes.
