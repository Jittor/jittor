# Jittor 源码架构迁移记录与领域包规范

> **决策更新（2026-08-11）**：长期目标已经由
> [`docs/architecture/repository-layout.md`](../../../docs/architecture/repository-layout.md)
> 锁定。本文早期采用的“公开文件 facade + `_xxx/` 私有实现包”是已完成批次的迁移
> 记录，不再是目标架构；`jittor.test` 也不再作为长期公开包保留。新工作必须按领域
> 包与四层兼容架构推进。

## 目标

Jittor 的公开 API、JIT 资源路径和 Torch 兼容行为已经被大量项目依赖，源码整理
不能采用一次性搬目录的方式。前十一批先用“稳定 facade + 私有实现包”降低巨型
模块风险；后续把这层脚手架收敛为常规领域包，同时保持公开 import、旧 checkpoint
可加载、Torch shim 注册顺序和 JIT 实体资源路径。

当前非测试 Python 源码约 181 个文件、67,620 行，前 12 个文件占约 45.9%。首要
问题不是文件数量，而是根入口、领域实现、运行时安装器和第三方兼容逻辑混在同一
模块中。重构必须让职责边界和依赖方向逐批变清晰，不能只把一个大文件机械切成
若干互相循环的小文件。

## 目标结构（2026-08-11 决策）

```text
python/
  jittor/
    __init__.py             # composition root
    nn/                     # __init__.py 为公开组合入口
      modules/              # 有状态模块实现
      functional/           # functional 实现
      attention.py
    misc/                   # 杂项领域包
    pool/                   # pooling 领域包
    optim/                  # optimizer 领域包
    compat/
      torch/                # Torch API 兼容
      shim/                 # 部署 shim
      fsdp2/                # FSDP2 / DTensor
      triton/               # Triton bridge
      module_patcher.py     # 通用补丁注册机制
      external_backend.py   # 通用外部后端发现机制
    selftest.py             # 安装后最小自检
    src/ extern/ utils/     # JIT 运行时资源，实体路径保持
    math_util/ other/
  jittor_utils/             # 独立底层包，保持 jittor 的同级兄弟
```

**推翻的旧决策**：不再用前导下划线包规避 `nn.py` 与 `nn/` 的同名冲突。迁移时
直接把公开文件转换为同名包，由 `__init__.py` 承接公开入口。`_nn/_misc/_pool/
_torch_compat/_torch_fsdp2` 是前十一批形成的临时脚手架，阶段 3 和阶段 4 必须消化，
不得继续复制 runtime proxy 或元数据伪装模式。

## 依赖方向

允许的主依赖方向为：

```text
stdlib / jittor_utils 底层
  -> core runtime 与领域实现
  -> nn / misc / pool / optim 领域包
  -> compat 能力与通用机制
  -> 领域 __init__.py 和根 composition root
```

具体约束：

1. 领域实现依赖具体领域接口，不通过无边界的根模块 service locator。
2. `jittor_utils` 的底层模块不得反向导入 `jittor`。
3. `compat` 可以依赖稳定领域能力；核心领域不得依赖下游项目补丁。
4. TRELLIS、Gaussian Splatting 和 Transformers 版本粘合放在可选发行包，主仓库
   只保留注册机制和通用能力。
5. 不得增加现有强连通导入分量；每批迁移应让环数量保持或下降。

阶段 3 已删除 `jittor._torch_compat.runtime`、`jittor._nn.runtime`、
`jittor._misc.runtime` 和 `jittor._pool.runtime`，对应实现已收敛到真实领域包。
`jittor._torch_fsdp2.runtime` 是唯一剩余的旧形态代理，随 shim/installer 收尾迁移。
新实现的 `__module__` 使用真实模块路径，旧 pickle 通过同对象兼容别名加载，不再递归
改写所有 callable 元数据。

## 迁移兼容契约

移动实现时必须同时保护以下可观察行为：

- 公开导入路径和符号名称不变。
- 领域 `__init__.py` 与实现对象保持同一身份，不创建无意义的包装对象。
- 新反射元数据使用真实实现路径；迁移前 pickle 必须有 fixture 验证和兼容加载路径。
- `Var.__module__ == "jittor"` 等既有序列化约定不变。
- `jt.nn.SGD is jt.optim.SGD` 等历史重导出身份不变。
- Torch shim 注册的 `torch.*` 模块对象和安装顺序不变。
- 初始化期间的副作用保持幂等，不提前导入可选依赖。

新的业务实现不得继续直接追加到组合入口。`__init__.py` 只允许组合、适配和显式
重导出；实现进入有语义的 modules/functional 子模块。

## 资源与打包边界

`python/jittor/src`、`python/jittor/extern` 不是普通源码杂物。编译器在运行时按
实际文件路径递归读取 C++ 源码、头文件和数据资源，因此第一阶段不得移动或改成
抽象资源接口。

阶段 1 已用 `pyproject.toml`、`find_packages`、`MANIFEST.in` 和结构测试替换显式
短清单与深度通配。四个断裂父包补齐后，35 个含 `__init__.py` 的目录全部由 regular
package discovery 覆盖，C++ include 目录不会被当作 namespace package。每次后续
打包变更必须：

1. 通过包发现完整性断言。
2. 从仓库外构建 wheel。
3. 检查所需 Python/C++ 资源存在。
4. 拒绝 `.pyc`、`__pycache__`、日志、实验目录和未知运行产物。
5. 在隔离目录安装 wheel，并从安装结果冷启动导入。

## 测试结构

**推翻的旧决策**：`python/jittor/test` 不再作为长期公开包路径。阶段 5 已新增
`jittor.selftest` 承接安装脚本、Docker 和环境脚本的最小自检，并把仓库测试迁到
顶层 `tests/`，由 pytest 收集。后续修改必须保持：

- 共享工具收敛到 `tests/_helpers/`，不再跨测试模块隐式导入。
- 文件过滤发生在导入测试模块之前，marker 对完整测试名生效。
- 测试选择不依赖 `listdir()` 索引或目录枚举顺序。
- 子进程失败、超时、启动异常和 unexpected success 使入口返回非零。
- C++ `test.h` 只随编译 include 消费者一起移动。

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

## 后续迁移阶段

前十一批的事实和验证证据继续有效，但后续顺序改由仓库布局 RFC 统一管理：

1. 先修正打包清单、资源声明、部署复制和 wheel 基线。
2. 建立 ruff、mypy、pre-commit、nox、分层 CI、现代发布链、容器与 ASV。
3. 把 `nn.py + _nn/`、`misc.py + _misc/`、`pool.py + _pool/` 和 Torch compat
   配对合并为领域包，删除重复 runtime proxy 和 `preserve_facade_origins`。
4. 按能力、机制、根因、项目粘合四层重构兼容代码。`monkeypatch_ops.py` 不再拆入
   新的 `_third_party_compat/`；通用能力/根因进入 Jittor，粘合进入可选发行包。
5. 已新增 `jittor.selftest`，测试已迁到根 `tests/` 并切换 pytest。
6. 把脚本、demo、notebook 收敛到 `tools/` 与 `examples/`，删除有证据的死资产。
7. 在新领域包形态下同批处理 Torch shim/installer、剩余 nn 类和根 composition root。
8. 完成文档、国际化、教程和治理文档现代化。

能力正确性仍高于目录施工：Torch distributed 状态、checkpoint 静默 no-op、HCCL/
CPU MPI 通信以及通用 DTensor/TP 的能力债务必须真实实现或 loud reject，不能由结构
测试代替。完整旧盘点见
`agent/results/2026-08-11-source-architecture-misc-fsdp2-refactor.md`。

### 保持原位的运行时边界

- `src/`、`extern/`、`utils` 中被 C++ exec 的文件、`math_util/src` 和
  `other/code_softmax.py` 在消费者协议修改前保持实体路径。
- `python/jittor_utils` 保持为 `jittor` 的同级兄弟。
- 不使用 namespace package 自动发现；包发现只识别含 `__init__.py` 的目录。
- 结构迁移与算子语义变更分别验证；阶段 4 的根因/能力变更必须追加数值和设备证据。

## 每批验收

每个迁移提交至少完成：

1. 旧实现与新实现的 AST 或人工逐块等价审查。
2. 公开导入、对象身份、真实模块元数据和旧 pickle 兼容测试。
3. 领域模块导入方向、职责和体积检查。
4. 受影响 CPU 测试和至少一个真实加速设备回归。
5. 仓库外 wheel 构建、内容检查和隔离安装导入。
6. `git diff --check` 与 `agent/scripts/check_repo_layout.sh`。
7. 更新本规范、阶段报告和项目上下文。
