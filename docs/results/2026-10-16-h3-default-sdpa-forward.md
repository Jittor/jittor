# H3 默认 BF16 SDPA 前向精度

- 状态：前向改进已验证；公共原语梯度与整模型 AD 缺口仍未关闭。不授予整模型 L2/L3 或 CUDA 广度准入。
- 日期：2026-10-16
- 测量源码：`67969d159e4f9033bac2ffc864a66a1b2a6a41c1`（干净）
- 保留的归因基线：`57b5f57ae220d1d8af1e6880ffcde6e362969110`
- 维护者：原生 attention、CUDA attention 与 Torch 前端维护者
- 复查条件：attention 选择、cuDNN/前端/运行时、固定的 H3 recipe，或真实 Torch 默认 provider 变化。

## 范围与未改动的预算

这里测量实际上游 `MiniMaxH3DiTModel`：一个主干 block、一个 refiner、hidden 维度 32、33,268 个参数标量、36 个参数与 37 个初始 state 条目。Omni 固定在 `446c2b5474dabffccf9a0034836eeab77c62b4b8`，vLLM 固定在 `98dff2a81d747d1dba01a47f939f48c3526d4206`。初始权重使用种子 3；输入使用种子 7 与 19。六个输出是每个输入种子的 video、audio 与 loss。

原始前向指标仍为
`max(abs(candidate-reference)) / max(1e-8, max(abs(reference))) <= 0.002`，
逐数组。绝对误差或 `allclose(atol=.002, rtol=.002)` 是另一种判据。正式比较不使用任何 oracle attention 覆盖、更小模型或改动输入。

## 归因与实现

在 source57 上，全部 37 个初始 state 与真实运行时逐字节匹配，但六个输出中有四个不满足原缩放预算。缩放误差在种子 7 下为 `.001200732/.005837109/.017018842`，在种子 19 下为 `.001815902/.004493984/.002656590`，按 video/audio/loss 顺序。

一个真实的算子观察者保留这些输出。真实默认 Torch 在 BF16 `[1,4,2,8]` refiner 与 `[1,4,6,8]` main-attention 张量上使用 Flash SDPA。第一个 refiner 的 Q/K/V 输入完全一致，但其受保护的 FP32-opmath 输出相差一个 BF16 ULP；后续 main-attention 输入此时已经不同。独立 CUDA 原语用未归一化的 BF16 概率乘积复现真实 fused 前向舍入。真实 MATH 与归一化 BF16 概率产生不同的输出。强制 MATH 仅用于诊断。

生产改动复用既有原生 cuDNN provider。严格的布尔原生 `allow_reduced_precision` 选项默认 false，保留原生公共 `.002` 前向 / `.01` 梯度 opmath 策略。Torch 前端请求该选项；它不增加任何 attention 数学。新准入限制在无 mask、非 causal、无 dropout 的 BF16 四维张量，batch/head 数匹配，K/V shape 相同，head 维度 8/16/24，query 长度 1..64，key/value 长度 2..64。mask、causal、dropout、GQA、更长序列与单 key 调用保留其原有受保护路径。单个 key 的解析 Q/K 梯度恰好为零；不引入抵消下限来在那里主张 fused 等价。

## 干净源码的正式结果

在干净的测量源码上，六个默认 attention 输出在 candidate 与真实 Torch 之间**字节相等**，缩放误差为零。全部 37 个初始 state 条目也与原始 source57 参考匹配。真实六数组 NPZ 哈希仍为
`e11269e381af6b183917886a9b71be847b957ba9bccd3081c4b3eb1365005bb9`；
其初始 state NPZ 仍为
`5e0e32c5633363e79de48076c6faf59df6a2e691c69a768c2b458a5e8c7d077e`。

单独的观测运行保留每侧六个未观测输出。全部 16 对内部 Q/K/V/output 数组逐字节一致。真实侧执行四次 Flash 前向调用；candidate 执行四次实际原生 cuDNN 前向调用，使用物理 sequence-major 布局 7 与 BF16 操作数。前向探针在每侧记录 43 个实际 CUDA 指针；candidate 的 fallback 增量为零。

完整 eval/train AD 审计在**两个**运行时上都仍不完整。每个 worker 显式请求全部 36 个参数与四个浮点输入的梯度，写出其实际数组，然后在严格完整性检查下以非零码退出。对每个种子与模型模式，只有四个 terminal-head 参数梯度存在；32 个参数梯度与全部四个输入梯度缺失。每个 worker 写出 14 个数组（跨两个种子共六个前向与八个可用 head 梯度）。缺失的梯度不被零替代，仅 head 梯度不是整模型训练证据。

CPU 探针也仍为负：上游默认抛出 `No CUDA GPUs available`；candidate 抛出 `Fallback Gloo backend is not available`。两者都在原始失败之前写出完整的安装/平台 packet。它们不提供 CPU 数值对应项或 CPU `jtrun` 覆盖。

## 公共梯度边界

新的 provider 测试使用既有 cuDNN BF16 `.02` 契约。**这不建立 Torch 公共默认的原始 `.01` 梯度契约。**
一个单独的四 fixture 前后 CUDA 比较对每个 dQ/dK/dV 数组保留原 `.01` 指标：

| Fixture | 旧受保护数学 dQ | 新 cuDNN dQ | 新 dK | 新 dV |
| --- | ---: | ---: | ---: | ---: |
| seed7 refiner | .025103401 | .007352941 | .006849315 | 0 |
| seed7 main | .004629630 | .003086420 | .005681818 | 0 |
| seed19 refiner | .324621212 | **.018181818** | .001470588 | 0 |
| seed19 main | .004360465 | .005813953 | .004739336 | 0 |

十二个旧梯度数组中有九个、十二个新数组中有十一个通过 `.01`；不存在从旧通过变为新失败的梯度数组。seed19 refiner 的 dQ 仍未关闭。全部四个新原语输出与默认 Flash 字节相等。该表是前向/原语证据，不是对公共梯度缺口或 H3 缺失梯度总账的豁免。后续的反向调查使用其自身的源码与产物；它不得重标本测量源码。

## 平台与源码证明

卡为 RTX 4090 D，SM89，驱动 580.178.04。真实侧是独立二进制
Torch **2.13.0+cu130**，二进制 SHA256
`819a187ea3b1f4d781200fe73fba3264689835d2285f7e6d10b986257a802dce`。
live 映射报告真实 CUDART 13000 / cuDNN 92000，candidate 为 CUDART 12020 / cuDNN 92400。candidate 使用 NVCC 12.2.140、GCC 11.5 与实际 NVIDIA cuDNN frontend 1.30.0 wheel。真实平台 packet 中可用的编译器同样是 NVCC 12.2；它不标识 Torch 的构建工具链。两个实际库栈分别记录，不称为同一工具链。

两个运行时都独立校验完整的已安装 Omni/vLLM 载荷、METADATA、RECORD 与原始 wheel 哈希。Omni wheel 是**本地源码构建**的构件，不是 PyPI 发行：
`c6a43c816c95f99f751dd9aa18c36ff94461d508da002a2a328c543e14b2c0de`。
固定构件守卫与源码闭包仍然生效。candidate 的 13 个源绑定部署 facade 文件与兼容 metadata 被显式标识为兼容实现，而非真实 Torch/媒体/Flash 包。前端 wheel 哈希为
`98d3b5e3ef9c951a69ca36978f7b421e744895a810167577d1f57f3926a0202f`，
其全部 897 个已安装载荷成员被独立重新哈希。

## 回归与保留的负证据

未改动的原生 CUDA opmath/MemoryEfficient/cuDNN 套件通过 41 项测试；新的 CUDA 策略测试通过 5 项，邻近兼容测试通过 7 项，FP32 provider 测试通过 3 项，均无跳过。新的 CPU opmath 测试通过 10 项；邻近 CPU MemoryEfficient 测试通过 9 项，另有两个仅 CUDA 跳过。维护的 CPU core 门禁通过 322 项，42 跳过与一个既有 xfail。最终结构检查通过 1,698 项，四个跳过。Ruff、布局与 diff 检查通过。这些单测运行记录其当时的基础 SHA 加上精确的七文件 WIP 哈希，与最终生产字节匹配；只有正式 H3 worker 主张干净的测量 SHA。

全部失败尝试保持独立：第一次 CPU 单测调用中继承的 CUDA 后端；两个初始路由观测断言；一次 editable core/compat 命名空间安装失败；以及第一次真实 CUDA 平台探针在 PATH 上缺少实际编译器。显式环境修正、compat 模式 core 安装与真实 provider 调用观测解决了这些问题，未改动模型数学或数值预算。loader token 在实际发现期间从 `(1,0,0)` 变为 `(1,1,2)`，缓存命中不会重新发布 `LAST_FUSED_KERNEL`；测试校验真实 `_forward` 调用，而不是制造该标记。第一次封条包含其仍未关闭的控制日志；最终 v2 封条重新哈希已完成的产物并排除其自身 live 日志。

## 复现与封存证据

以下路径都是 `$JITTOR_LAB_ROOT` 下**未版本化**的：

- State：`_state/omni-forward-fix/20261016/`。
- 不可变测量 worktree：`worktrees/omni-forward-fix`，干净 source679。
- 正式驱动：`run-formal-v2.py`、`run-remaining-formal-v1.py`；每个 case 有
  精确的 argv/环境、执行前源码哈希、终态与日志。
- `formal-forward-audit-v1.json`：独立完整 installed-source 与六数组
  重算；`formal-gradient-ledger-v1.json`：八行 runtime/mode/seed。
- `formal-observer-audit-v1.json`：真实 provider、物理操作数、live 映射
  库、12 个未改动最终输出与 16 对内部字节相等数组。
- `public-gradient-budget-audit-v1.json`：从 48 个已保存数组独立重算全部 32 项
  前后比较，保留 `.002/.01` 预算。
- 最终 `evidence-manifest-v2.json`：9,180 个产物 / 4,476,538,455 字节与
  2,907 个当前 Git blob，包括符号链接目标文本。SHA256：
  `816360ba7f1975b4b7620c698e6e2f6219e1cc7c0c8dfcba331d25f09362d365`。

要对照原始测量树独立重算前向证据，
使用带 NumPy 的 host 解释器并写入新的外部输出：

```bash
python "$JITTOR_LAB_ROOT/_state/omni-forward-fix/20261016/audit-formal-v1.py" \
  --root "$JITTOR_LAB_ROOT/worktrees/omni-forward-fix" \
  --oracle "$JITTOR_LAB_ROOT/_state/omni-forward-fix/20261016/formal-genuine-cuda-v2" \
  --candidate "$JITTOR_LAB_ROOT/_state/omni-forward-fix/20261016/formal-candidate-cuda-v2" \
  --output "$JITTOR_LAB_ROOT/_state/<new-independent-audit>.json"
```

原始 source57 归因、setup 失败、source679 数组与缺失 AD 总账保持区分。本报告不主张完整 H3 训练、性能、engine/pipeline 支持、CPU 支持或 CUDA 广度准入。
