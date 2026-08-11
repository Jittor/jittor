# Jittor 源码架构与渐进式拆分规范

## 目标

Jittor 的公开 API、JIT 资源路径和 Torch 兼容行为已经被大量项目依赖，源码整理
不能采用一次性搬目录的方式。本规范要求用“稳定 facade + 私有实现包”逐步拆分
巨型模块，同时持续降低导入环、控制模块体积，并保持已有对象身份、pickle 路径和
`sys.modules` 注册不变。

当前非测试 Python 源码约 181 个文件、67,620 行，前 12 个文件占约 45.9%。首要
问题不是文件数量，而是根入口、领域实现、运行时安装器和第三方兼容逻辑混在同一
模块中。重构必须让职责边界和依赖方向逐批变清晰，不能只把一个大文件机械切成
若干互相循环的小文件。

## 目标结构

```text
python/
  jittor/
    __init__.py             # 稳定公开 facade 和 composition root
    nn.py                    # 稳定公开 facade
    pool.py                  # 稳定 legacy pooling facade
    optim.py                 # 稳定公开 facade
    compiler.py              # 稳定公开 facade
    torch_compat.py          # 稳定 Torch 兼容 facade
    misc.py                  # 稳定杂项公开 facade
    torch_fsdp2_compat/      # 稳定 FSDP2 兼容入口
    _bootstrap/              # 仅标准库：环境、路径、MPI 前置处理
    _runtime/                # core loader、编译器和后端控制
    _api/                    # Var、Module、Function 和基础 API 实现
    _nn/                     # 神经网络私有实现
    _pool/                   # legacy pooling 私有实现
    _misc/                   # 杂项算子的领域实现
    _optim/                  # 优化器私有实现
    _torch_compat/           # Torch 兼容私有实现
    _torch_fsdp2/            # FSDP2/DTensor 兼容实现
    torch_shim/              # 保持现有公开导入路径
    src/                     # 运行时 JIT C++ 源码，保持实体路径
    extern/                  # 运行时头文件和第三方资源，保持实体路径
  jittor_utils/              # 独立底层包，不反向依赖 jittor facade
```

私有目录使用前导下划线，避免与已有 `nn.py`、`optim.py` 等公开模块发生
“同名文件与包”冲突。公开 facade 可以长期存在；迁移完成不以删除 facade 为目标。

## 依赖方向

允许的主依赖方向为：

```text
stdlib
  -> _bootstrap / jittor_utils 底层
  -> _runtime
  -> _api
  -> _nn / _optim 等领域实现
  -> _torch_compat
  -> 公开 facade 负责最终组合与重导出
```

具体约束：

1. 新私有模块不得在模块作用域 `import jittor` 或 `from jittor ...`。
2. 私有包内部只允许一级相对导入；`from ..` 会重新依赖部分初始化的根包。
3. 必须访问运行时根模块时，通过 composition root 显式绑定的窄接口完成。
4. `jittor_utils` 的底层模块不得反向导入 `jittor`。
5. 不得增加现有强连通导入分量；每批迁移应让环数量保持或下降。

`jittor._torch_compat.runtime`、`jittor._nn.runtime` 和 `jittor._pool.runtime` 当前
分别提供最小运行时代理。
它们只解决拆分期间的初始化顺序，不应演化成无边界的 service locator；后续应按
实际依赖继续收窄协议。私有实现若依赖可被后端运行时重绑的公开符号，必须经 facade
动态解析，不能静态捕获同包实现；`log_softmax -> nn.softmax` 是现有示例。
公开类迁入私有模块时，元数据恢复还必须递归覆盖类自身的方法、property 和 decorator
wrapper，避免只修复类的 pickle 路径而让方法反射来源发生漂移。

## Facade 契约

移动实现时必须同时保护以下可观察行为：

- 公开导入路径和符号名称不变。
- facade 与实现对象保持同一身份，不创建无意义的包装对象。
- 类和函数的 `__module__`、`__qualname__` 及 pickle 路径保持兼容。
- `Var.__module__ == "jittor"` 等既有序列化约定不变。
- `jt.nn.SGD is jt.optim.SGD` 等历史重导出身份不变。
- Torch shim 注册的 `torch.*` 模块对象和安装顺序不变。
- 初始化期间的副作用保持幂等，不提前导入可选依赖。

新的业务实现不得继续直接追加到 facade。facade 只允许组合、适配、显式重导出和
必要的兼容元数据处理；超过模块行数预算时必须先拆分再增加功能。

## 资源与打包边界

`python/jittor/src`、`python/jittor/extern` 不是普通源码杂物。编译器在运行时按
实际文件路径递归读取 C++ 源码、头文件和数据资源，因此第一阶段不得移动或改成
抽象资源接口。

当前 `setup.py` 使用显式 package 列表。`find_packages()` 会遗漏或改变部分现有
边界，namespace discovery 又会把 C++ include 目录误识别为 Python 包，因此在
完成打包清单审计前不得直接切换自动发现。每新增私有包必须：

1. 显式加入 package 列表。
2. 从仓库外构建 wheel。
3. 检查所需 Python/C++ 资源存在。
4. 拒绝 `.pyc`、`__pycache__`、日志、实验目录和未知运行产物。
5. 在隔离目录安装 wheel，并从安装结果冷启动导入。

## 测试结构

`python/jittor/test` 暂时保留公开包路径，避免破坏 README、Docker 和外部脚本中
的 `python -m jittor.test.*`。总入口只负责选择、调度和汇总；可测试逻辑放入
`test/_runner.py`，必须满足：

- 文件过滤发生在导入测试模块之前。
- skip marker 对整个测试名生效。
- 子进程失败、超时、启动异常和 unexpected success 使总入口返回非零。
- 子进程使用参数数组，禁止 `shell=True` 拼命令。
- 历史环境变量 `seperate_test` 的拼写继续兼容。

结构测试与数值测试分开。结构测试负责导入方向、facade 身份、pickle、模块体积和
打包声明；数值/设备测试继续负责 CPU、CUDA、NPU 行为。

## 渐进迁移顺序

### 第一批：Torch 兼容层

已建立 `_torch_compat/`，按类型与设备、梯度、nested tensor、序列化、纯函数、
优化器和 scheduler 拆分。`torch_compat.py` 保留安装顺序和公开 facade。

### 第二批：`nn.py`

已建立 `_nn/`，先抽取 29 个低耦合纯函数，按 activation、loss、softmax 和 vector
分为 4 个实现模块。第二批当时让 `nn.py` 保留所有 `Module` 子类、别名、`Var`
绑定、optimizer 重导出和后端集成点，先验证 facade 模式，再处理含参数状态和设备
快路径的区域。

normalization、RNN、convolution 的 functional/backend 和主要公开类、padding、
`nn.py` 的 corrected average pooling 覆盖层，以及 legacy `pool.py` 已按风险顺序
完成拆分。参数容器协议稳定前不移动根 `Module`；DepthwiseConv、ACL 重绑和别名
网络仍必须按领域整体审计，不能只搬 Python 类壳。

### 第三批：normalization

已将 `batch_norm`、`instance_norm`、`group_norm`、稳定反向 helper 和完整 CUDA
LayerNorm no-grad kernel 拆到 `_nn/normalization.py` 与
`_nn/layer_norm_cuda.py`。四个 `Module` 类、别名、`fp32_guard` 和装饰后的
`layer_norm` 留在 facade，保留 ACL 对 `LayerNorm.execute` 的原位补丁、Torch shim
类身份、MPI 状态和公开 helper monkeypatch 行为。

### 第四批：RNN

已将 `LSTMCell/RNNCell/GRUCell`、`RNNBase` 与 `RNN/LSTM/GRU` 分别拆到
`_nn/recurrent_cells.py`、`_nn/recurrent_base.py` 和
`_nn/recurrent_layers.py`。参数创建顺序、递推调度、投影、双向/多层状态布局和
完整 cuDNN 权重 flatten/执行路径随领域整体迁移。facade 直接重导出同一类对象，
私有方法通过 `jt.nn` 动态解析 `init`、`matmul_transpose`、`relu` 和 `dropout`，
保留 Torch shim 与后端运行时重绑语义。

### 第五批：convolution

已完成 functional/backend 第一阶段：`conv1d/2d/3d`、三种 transpose functional、
2D cuDNN `Function` 和 3D 半精度保护分别拆到 `_nn/convolution.py`、
`_nn/convolution_transpose.py` 与 `_nn/convolution_cudnn.py`。facade 直接重导出同一
函数/类对象并恢复 `jittor.nn` 元数据；私有实现通过 `jt.nn` 动态解析 `_pair/_triple`、
cuDNN helper、`conv2d` 和 `conv_transpose`，保留 ACL 后置 wrapper、公开 monkeypatch
及 `conv/conv_transpose2d` 别名语义。

### 第六批：convolution 类

已将 `Conv/Conv1d`、`Conv3d` 与 `ConvTranspose/ConvTranspose3d` 分别拆到
`_nn/convolution_layers.py`、`_nn/convolution_3d_layers.py` 与
`_nn/convolution_transpose_layers.py`。facade 直接重导出同一类对象，递归恢复类和
方法的 `jittor.nn` 元数据；`_pair/_triple`、`init`、`DepthwiseConv`、cuDNN helper
及 functional 均经 `jt.nn` 动态解析。

`Conv1d` 必须动态构造公开 `jt.nn.Conv`，保留 ACL 类 wrapper。`Conv1d_sp` 留待与
`Linear` 同批迁移；`DepthwiseConv` 继续由 `jittor.depthwise_conv` 拥有并保持原
pickle 路径；pooling 与 Torch shim 后置创建的 `ConvTranspose1d` 不在本批迁移。

### 第七批：padding

已将 `pad`、`ReflectionPad2d`、`ZeroPad2d`、`ConstantPad1d/2d/3d` 与
`ReplicationPad2d` 整体拆到 `_nn/padding.py`。facade 直接重导出同一对象，并递归
恢复函数、类和方法的 `jittor.nn` 元数据；`nn.functional.pad` 与
`jittor.attention.pad` 继续和 `nn.pad` 保持同一身份。

本批保留 `pad=` 关键字别名、四种 functional mode、类/实例 pickle、属性顺序与空
`state_dict`。ACL 当前不重绑 padding，所有路径仍使用 `Var.reindex`；若未来新增
后端 wrapper，必须同步审计 attention 的静态导入和 Torch functional 的创建顺序。

### 第八批：pooling 覆盖层

已将 `nn.py` 中 corrected `adaptive_avg_pool2d/avg_pool2d` 与
`AdaptiveAvgPool2d/AvgPool2d` 拆到 `_nn/pooling.py`。facade 直接重导出同一对象，
用 19 个显式名称替代 `from jittor.pool import *`；functional、`nn.modules`、反射、
pickle 与实例字段契约保持不变。

两个 functional 必须经 `jt.nn` 动态构造公开类，`AvgPool2d.execute` 必须动态解析
`jt.nn._pair`，保留公开 monkeypatch 行为。`jittor.pool` 中语义不同的 legacy
average pooling，以及 `Pool/Pool3d/MaxPool*/1D/3D/Unpool` 仍是独立对象；ACL
继续只包装直接的 `nn.Pool`。

### 第九批：legacy pooling

已把 777 行 `pool.py` 按 2D core、3D core、1D、adaptive、wrapper 和 unpool
拆入 `_pool/`，公开 facade 降到 38 行。22 个迁移定义严格 AST 22/22 等价，
27 个历史公开名称、反射/pickle、实例状态、`nn.functional/modules` 身份和
`pool_use_code_op` 的执行时动态读取均由结构契约锁定。

私有实现只依赖 `_pool.runtime`，跨定义调用经 `jt.pool` 动态解析，不反向依赖
`jt.nn`。ACL post-process 仍修改公开开关并只包装直接的 `nn.Pool`；真实 NPU
尚未执行，因此当前只声明源码和动态边界等价，不声明 ACL 数值验证。

### 第十批：`misc.py`

已将 `repeat/chunk/expand` 与 `atleast_1d/2d/3d`、`cartesian_prod/block_diag`
分别拆入 `_misc/shape_transforms.py` 和 `_misc/shape_composition.py`，`misc.py`
从 3,066 行降到 2,805 行。8 个公开对象直接重导出，并通过 `jt.misc` 动态解析可被
monkeypatch 的公开符号；归一化后 AST 8/8 等价。

后续按 reduction/scatter、sequence/indexing 继续拆分。`repeat_interleave` 当前有
独立的 CUDA/ACL 路由与梯度覆盖风险，不在纯结构提交中顺带移动或修改。

### 第十一批：FSDP2 兼容层

已删除根包中 2,175 行的 `torch_fsdp2_compat.py`，改为 56 行的稳定公开包 facade，
并按 runtime、DTensor/config、shard、gradient、optimizer、FSDP API 和 installer
拆入 11 文件的 `_torch_fsdp2/`。原 79 个公开名称、70 个 callable 签名/反射/pickle、
37 个 `torch.distributed.*` 深层注册和动态 monkeypatch 边界保持不变。

拆分后所有 facade 导出 helper/type 的运行时依赖都经 facade 解析，installer 只依赖
runtime；`DeviceMesh` 重装传播、wrap 安装后 patch、FSDPModule 同步 hook 和 collective
hook 由结构测试锁定。`jt/np/os/types` 等历史公开模块对象保持身份，但不是可替换 hook。

真实 2-GPU NCCL API 与单步训练已经和官方 PyTorch 参考对拍。HCCL all-gather、CPU
MPI all-gather 和多卡 checkpoint 仍是既有能力缺口；没有真实 NPU 验证，不得把源码
边界审计写成 ACL/HCCL 已通过。

双卡结果只覆盖 FSDP-managed 参数的 NCCL shard/unshard/grad/optimizer 路径。通用
DTensor/tensor parallel 仍主要是 API/元数据兼容；MixedPrecision、CPUOffload 和
ShardedGradScaler 也主要是 surface compatibility。Torch distributed 对外状态与内部
FSDP rank/world 口径分裂、distributed checkpoint 静默不落盘的风险，必须先实现真实
语义或 loud reject，不能从结构测试通过推导为完整能力。

### 第十二批：`torch_shim/torch__init__.py`

按 nn、optim、cuda、distributed、data 和 stub 注册拆 shim；保持模块对象身份、
注册顺序和只读 extension 安装时机。该文件必须和 `torch_compat.py` 中的安装器共同
设计，不能分别切分后再补对象身份。

### 第十三批：剩余 `nn` 类

整体审计并迁移 `Linear/Conv1d_sp/DepthwiseConv` 及相关别名、ACL wrapper 和 Torch
shim 类快照。不得只移动类壳而静态捕获可被后端替换的公开构造器。

### 第十四批：启动与运行时

抽出 `_version.py` 和仅标准库的 `_bootstrap/`，再逐步分离 core loader、compiler
和 backend controller。根 `__init__.py` 最后收敛为严格排序的 composition root。

## 顶层审计与后续优先级

2026-08-11 对 `python/jittor` 顶层全部 Python 文件重新盘点后，后续工作按以下顺序
推进。公开 facade 留在根包是兼容设计；应移走的是大段实现、第三方补丁和运行时控制。
能力正确性高于目录施工：先修 Torch distributed 状态、checkpoint 静默 no-op，并为
HCCL/CPU MPI 补通信或显式拒绝，再执行下面的源码拆分顺序。

| 优先级 | 当前文件 | 主要职责债务 | 下一步 |
| --- | --- | --- | --- |
| P0 | `torch_compat.py`（8,683 行） | installer、tensor methods、misc 注册和大量 `torch.*` 模块构造 | 继续拆 `_torch_compat`，与 shim 同步设计 |
| P0 | `misc.py`（2,805 行） | 103 个函数、3 个类、71 个模块级绑定 | reduction/scatter，再 sequence/indexing |
| P0 | `nn.py`（2,442 行） | 62 个函数、43 个类及后端静态引用 | 剩余 nn/attention/DepthwiseConv |
| P0 | `monkeypatch_ops.py`（1,024 行） | 多个第三方项目补丁混在根包 | 保留根 facade，实现迁入 `_third_party_compat/` |
| P0 | `__init__.py`（2,851 行） | bootstrap、core loader、API 和安装副作用混合 | 最后拆 `_version/_bootstrap/_runtime/_api` |
| P1 | `linalg.py`、`distributions.py` | 多数学领域与 Torch 注册混合 | 按领域拆分，保留 facade |
| P1 | `compiler.py`、`compile_extern.py` | JIT、缓存、外部后端控制高度耦合 | 启动边界稳定后迁入 `_runtime` |
| P2 | `init.py`、`optim.py`、`contrib.py` 等 | 领域实现和全局安装副作用尚未分离 | 在 P0/P1 边界明确后处理 |

完整盘点、指标和已知 FSDP2 后端缺口见
`agent/results/2026-08-11-source-architecture-misc-fsdp2-refactor.md`。

### 暂缓项

- 不移动 `src/`、`extern/`。
- 不一次性迁移全部测试到仓库根 `tests/`。
- 不直接改用 namespace package 自动发现。
- 不在结构重构提交中顺带改变算子语义或设备实现。

## 每批验收

每个拆分提交至少完成：

1. 旧实现与新实现的 AST 或人工逐块等价审查。
2. facade 对象身份、模块元数据和 pickle 契约测试。
3. 私有模块导入方向与行数预算检查。
4. 受影响 CPU 测试和至少一个真实加速设备回归。
5. 仓库外 wheel 构建、内容检查和隔离安装导入。
6. `git diff --check` 与 `agent/scripts/check_repo_layout.sh`。
7. 更新本规范、阶段报告和项目上下文。
