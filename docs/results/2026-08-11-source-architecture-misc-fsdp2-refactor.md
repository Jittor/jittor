# 源码架构重构第十、十一批：misc 形状域与 FSDP2 兼容层

## 结论

本批不再只检查已经显眼的实验目录，而是对 `python/jittor` 顶层 Python 文件做了
完整盘点。盘点确认原先 2,175 行的 `torch_fsdp2_compat.py` 是不应继续存在于根包的
单文件实现：它同时承担 DTensor、参数切分、梯度同步、优化器、FSDP API、Torch
模块注册和安装编排。该文件现已删除，公开入口改为 56 行的
`torch_fsdp2_compat/__init__.py`，实现拆入 11 文件的 `_torch_fsdp2/` 私有包。

同时完成 `misc.py` 第一阶段拆分：8 个 shape transform/composition 函数迁入
`_misc/`。公开名称、对象身份、反射、pickle、Torch shim 和运行时 monkeypatch
语义保持不变。本批不改变算子或设备语义。

## 顶层全量盘点

盘点覆盖 `python/jittor` 顶层全部 `.py` 文件，并结合行数、顶层定义、模块注册、
生产引用和后端耦合判断优先级。公开 facade 可以留在根包；问题是把大段实现、第三方
补丁或运行时控制器继续堆在 facade 中。

| 优先级 | 文件 | 当前规模 | 主要问题 | 建议归属 |
| --- | --- | ---: | --- | --- |
| P0 | `torch_compat.py` | 8,683 行 | 兼容实现仍有 77.6% 留在根文件；`install`、tensor method 与 misc 安装器合计数千行，并创建大量 `torch.*` 模块 | 继续拆到 `_torch_compat/`，与 shim 注册共同设计 |
| P0 | `misc.py` | 2,805 行 | 仍有 103 个函数、3 个类和 71 个模块级绑定，第一阶段后仍保留旧文件 89.2% | 依次拆 reduction/scatter、sequence/indexing |
| P0 | `nn.py` | 2,442 行 | 仍有 62 个函数、43 个类及 38 处生产静态引用 | 按 Linear、DepthwiseConv、attention 等领域迁入 `_nn/` |
| P0 | `__init__.py` | 2,851 行 | bootstrap、core 编译、基础 API、Module/Function 与 Torch 安装混合；初始化风险最高 | 最后拆 `_version`、`_bootstrap`、`_runtime`、`_api` |
| P0 | `monkeypatch_ops.py` | 1,024 行 | Transformers、PEFT、ms-swift、FlexGEMM、DINOv3、TRELLIS 补丁混在根包 | 保留根 facade，实现拆到 `_third_party_compat/` |
| P1 | `linalg.py` | 1,602 行 | 46 个函数覆盖 complex、分解、范数、求解和 einsum | 按数学领域拆分，保留公开 facade |
| P1 | `compiler.py` / `compile_extern.py` | 1,514 / 943 行 | 编译、缓存、外部后端和运行时状态高度耦合 | 在启动层稳定后迁入 `_runtime/` |
| P1 | `distributions.py` | 921 行 | 14 个函数、27 个类，还创建 Torch constraints 路径 | 按 distribution/constraint 拆分 |
| P2 | `pyjt_compiler.py` | 946 行 | `compile_src` 单函数 587 行 | 独立解析与生成阶段 |
| P2 | `init.py` / `optim.py` | 737 / 659 行 | 初始化函数、Var 绑定和优化器类仍集中 | 分别迁入 `_nn`/`_optim` |
| P2 | `attention.py` / `depthwise_conv.py` | 596 / 324 行 | 与剩余 nn、后端重绑共同演进 | 与剩余 nn 类同批处理 |
| P2 | `contrib.py` | 346 行 | 重复 concat 定义并带全局 indexing 补丁 | 拆实现并显式管理安装副作用 |

`torch_shim/torch__init__.py` 另有 2,296 行。它虽然不在根目录，却与
`torch_compat.py` 的模块注册共享对象身份和安装顺序，不能把两边各自机械切开。

能力正确性优先于下一轮目录施工：先处理下文列出的 distributed 状态分裂、checkpoint
静默 no-op，以及 HCCL/CPU MPI 未实现通信或入口拒绝。完成这些显式边界后，源码整理
顺序为：

1. 继续 `misc.py` 的 reduction/scatter 与 sequence/indexing。
2. 拆 `torch_compat.py` 的安装器，并把 `monkeypatch_ops.py` 实现迁入
   `_third_party_compat/`，根模块保留兼容 facade；同时审计 Torch shim 注册。
3. 处理剩余 nn、attention、DepthwiseConv，再处理 linalg、distributions、init、
   optim 和 contrib。
4. compiler 与根 `__init__.py` 最后处理，避免过早改变冷启动和 JIT 资源路径。

## 第十批：misc 形状域

新增结构：

```text
jittor/
  misc.py                      # 稳定公开 facade 与剩余实现
  _misc/
    runtime.py                 # 最小运行时绑定与元数据恢复
    shape_transforms.py        # repeat / chunk / expand
    shape_composition.py       # atleast_* / cartesian_prod / block_diag
```

`misc.py` 从 3,066 行降到 2,805 行。8 个函数直接重导出同一对象；对原源码和新的
动态 `jt.misc` 解析做归一化后，AST 为 8/8 等价。私有实现不在模块作用域导入
`jittor`，跨公开符号调用通过 facade 动态解析。

`repeat_interleave` 有既有 CUDA/ACL 分支和反向覆盖问题，本批有意不迁移，避免把
行为修复混入结构提交。

## 第十一批：FSDP2 兼容层

新结构：

```text
jittor/
  torch_fsdp2_compat/
    __init__.py                 # 56 行稳定公开 facade
  _torch_fsdp2/
    runtime.py                  # 根模块代理与递归元数据恢复
    shard_common.py             # shard 公共计算
    dtensor.py                  # DTensor 与 placement
    config.py                   # mixed precision / CPU offload 等配置
    shard_runtime.py            # 参数 unshard/reshard 与通信路径
    grad_sync.py                # 梯度收集与同步
    optimizer.py                # sharded optimizer 适配
    fsdp_api.py                 # fully_shard / FSDP API
    compat_types.py             # Torch 兼容类型
    installer.py                # torch.distributed 模块注册
```

原 2,175 行根文件已删除。11 个私有文件共 2,571 行，最大实现模块 394 行；增加的行数
主要来自显式导出边界、运行时代理和兼容元数据恢复，而不是新增算法。

关键契约：

- 原有 79 个公开非下划线名称全部保留，不新增会改变 wildcard 行为的 `__all__`。
- 70 个公开 callable 的签名、`__module__`、`__qualname__` 和 pickle 契约保持不变；
  契约摘要为
  `ca8aea5689aa5280fcd65aa3157274e7f2cd0d2ee3e0d607ec14880d28e71e73`。
- 37 个 `torch.distributed.*` 深层模块注册保持幂等；
  `_install_fsdp2_distributed is install`。
- 原文件 136 个顶层定义/绑定与私有导出加 facade 元数据计划归一化后 136/136 对齐。
- 136 个 facade 导出 helper/type 的运行时裸引用已经清零；梯度、optimizer、DTensor、
  shard、FSDP API 和 installer 都从公开 facade 动态读取可替换符号，保留 monkeypatch。
  `jt/np/os/types` 等历史意外公开的模块对象保持身份，但不定义为可替换 hook。
- 嵌套类型和安装到虚拟 Torch 模块的对象继续显示历史模块路径。

真实双卡验证发现并修复了一处仅多卡会触发的问题：元数据恢复曾尝试直接修改 bound
method 的 `__module__`。现在会跟随 `__func__` 恢复底层函数，并由结构测试锁定。

## 验证

| 范围 | 结果 |
| --- | --- |
| misc 定向结构/数值 | CPU 54/54；CUDA 54/54 |
| FSDP2 结构/行为 | CPU 20/20；CUDA 20/20 |
| 合并源码 CPU 回归 | 108/108 |
| CUDA Torch compat 总入口 | 172/172 |
| 源码 Torch shim 身份 | misc 8/8、Var 3/3、FSDP/DTensor 4/4 |
| 真实 2-GPU NCCL API | 2/2 rank，8/8 DTensor、full shape 与 summon shape，单 rank local numel 29 |
| 真实 2-GPU NCCL 单步训练 | 6/6 case；初始 forward 最大误差 `5.36441802979e-07`，更新后 `4.76837158203e-07`，参数最大误差 `1.86264514923e-09` |
| wheel | 1,053 条目；`_misc` 4、`_torch_fsdp2` 11、公开 FSDP2 facade 1；旧根文件不存在；无缓存、构建目录或实验目录 |
| wheel SHA-256 | `07bb86a8647484bae1a4015aba84ef3187c1d07f4e48eceaae9d6385fb0b871c` |
| 隔离安装 | 108 项中 105 通过、3 项按环境/源码条件跳过；shim 身份 8/8、3/3、4/4 |
| 仓库检查 | `check_repo_layout.sh`、diff whitespace、禁止目录和仓库缓存检查通过 |

双卡训练覆盖 small/medium/large 三种模型、两个 rank，并与官方 PyTorch 参考结果对拍。
日志和产物均位于仓库外：

```text
/home/zy/projects/jittor-lab/_state/verify/source-refactor/fsdp2-multigpu
/home/zy/projects/jittor-lab/_state/verify/source-refactor/combined-final-5eb5d6
```

wheel 从暂存树 `5eb5d64f7f3019223ac5a71e21d5312adaa50d29` 构建。随后只把本段
artifact SHA/路径写回报告，该报告不属于 `setup.py` 的打包输入；20 个本批 Python
文件与最终索引逐字节核对一致，两处用户未暂存文件在 wheel 中仍取 `HEAD` 内容。

## 能力矩阵与已知边界

| 能力 | 当前状态 | 证据或风险 |
| --- | --- | --- |
| FSDP-managed 参数的 NCCL shard/unshard/grad/SGD | 已真实验证 | 2 GPU、2 rank、small/medium/large、6/6 case 与官方 PyTorch 参考对拍 |
| 通用 DTensor / tensor parallel | 仅 API/元数据兼容 | `distribute_tensor/full_tensor/redistribute` 与 local-shape/parallelize 多数不执行真实通信，不能从 FSDP 双卡结果外推 |
| MixedPrecision / CPUOffload / ShardedGradScaler | 主要为 surface compatibility | 当前多为记录配置或 no-op，没有验证真实混精、offload 或 scale/unscale 语义 |
| FSDP state/optimizer checkpoint | 不完整 | state/optim API 多为 delegate/pass-through，忽略部分 state-dict 配置 |
| distributed checkpoint | 存在静默风险 | `torch_compat.py` 的 save/load/_write_item 当前不实际落盘；应改为真实实现或 loud reject |
| activation checkpoint | 仅兼容入口 | wrapper 为 identity，`checkpoint` 直接执行，不提供真实重计算省显存语义 |
| Torch distributed 状态 | 与内部 FSDP 状态分裂 | shim 可能报告 initialized=False、rank=0、world=1 或 no-op collective，而 FSDP 内部按 `jt.world_size/rank` 真切分；第三方可能走错分支 |
| HCCL true-FSDP | 未支持 | 会进入 true-FSDP，但 all-gather 未接已有 HCCL op |
| CPU MPI true-FSDP | 未支持 | 会进入 true-FSDP，但没有可用的 MPI all-gather |

- 本机没有 Ascend/NPU 环境，没有执行真实 HCCL；不得把源码审计表述为 NPU 已通过。
- HCCL 当前会进入 true-FSDP，但 `_all_gather_shards` 尚未接入已有
  `hccl_all_gather`，首次 unshard 可能失败。
- CPU MPI 也可能进入 true-FSDP，但当前没有 `mpi_all_gather` 实现。
- checkpoint 与 Torch distributed 状态问题可能造成第三方训练代码静默选择错误分支
  或误以为已保存。按项目的 loud-crash 原则，应优先实现真实语义；暂时做不到时必须
  显式拒绝，而不是继续 no-op。
- 仓库原有 13 个 FSDP2 测试均为单 rank 或 mock 通信；本批执行了真实双卡外部验证，
  后续应把可维护的多卡用例接入持续回归。
- `repeat_interleave` 的既有设备路由和梯度覆盖问题仍需单独修复。

这些问题在本批结构拆分前已经存在。本批只确保拆分不改变现有计算行为，并将风险
显式立账；下一步应先补 distributed 状态与 checkpoint 的真实语义/loud reject，再
处理 HCCL/CPU MPI 和持续多卡测试，然后继续其余目录拆分。
