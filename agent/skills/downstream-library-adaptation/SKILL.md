---
name: downstream-library-adaptation
description: 接入或验证一个下游 Torch 生态库时的流程与准入标准。用于判断某个断点该修 jittor 核心、修 jittor.compat.torch 还是写 adapter，用于建立可信的双解释器对拍环境，用于决定跑哪些 device，以及用于新 adapter 进仓的准入检查。适用于 transformers/diffusers/peft/mmcv 这类纯 `import torch` 消费者，也适用于 vLLM 这类需要独立 adapter 的库。
---

# 适配一个下游库

**这件事必须走本流程。** 难点从来不是"跑通"，而是证明这次对拍是有意义的：
仓库已经三次踩在"绿了但什么都没证明"上，下面每一条约束都对应一次实际事故。

## 0. 判据：三个去处

先分流，再动手。这是整套标准的地基：

| 遇到的东西 | 去哪 | 判据 |
| --- | --- | --- |
| 能力本身 jittor 没有 | **jittor 核心** | 真缺功能，不是拼写问题 |
| 能力有，拼写/签名不同 | **`jittor.compat.torch`** | compat 只做适配；**出现第二套实现就是放错了** |
| 这个库私有的实现细节 | **adapter** | 它根本不是 torch API |

[`torch-compatibility-principles`](../../../docs/architecture/torch-compatibility-principles.md)
和 [`project-context`](../../manuals/project-context.md) 写了前两行；第三行原先只用
排除法说"不属于 core"，没给地址，这是 adapter 失控的根源。

**默认答案是"不需要 adapter"。** 绝大多数库走第二行，最终产物是生态门禁里的一个
case，零适配代码。

**第三行的可执行形式是「可重定位」**：把这段代码搬到仓外当第三方包发布，还能一模一样
工作 → 合法 adapter；搬出去就不行（用了私有属性、改了框架状态）→ 它属于前两行。

**adapter 想动框架状态的那一刻，说明该回 compat 开公开接口。** 正面样本：
`adapters/jittor_adapters/vllm/bootstrap.py` 原先写 `torch.__version__ = api_version`，
而 `torch` 就是 jittor，这一句把框架版本号改给了**所有用户**；正确做法是 compat 暴露
`compat_report_torch_api_version` 让 adapter 去问。

## 1. 定级

不定级就是无底洞。声称到哪一级，就要拿出哪一级的证据：

| 级 | 声称 | 证据 |
| --- | --- | --- |
| L1 | 能 import、能前向 | 单解释器 smoke |
| L2 | 与 PyTorch 数值一致 | 双解释器，forward + 全部 param grad + 全部 input grad |
| L3 | 能训 | 多步 loss/参数轨迹 |
| L4 | 性能达标 | 真实尺寸，ratio 不超过门禁上限 |
| L5 | 真机零 fallback | `fallback_count == 0` 加后端断言 |

## 2. 裸跑，收集断点

**不要先写适配代码。** 把库直接架在 shim 上跑，逐条记录断点，按第 0 节分流。
这是 verify-then-fix 用在适配上：约 75% 的"兼容层搞不定"经不起复现。
**只有落在第三行的才允许变成 adapter 代码。**

## 3. 环境合法性（要看数字就必须先过）

四条硬检查，过不了就**停在这里记「未验证 + 环境前置」**，不许往下看任何数字：

1. **oracle 不是 shim**：`assert not hasattr(torch, '_torch_compat_install_context')`。
   `noxfile.py` 的 `ecosystem` session 已经在起手做这件事。
2. **装之前先查 pin**。新库钉 `torch==X` 是常态；会换掉 oracle 的 torch 就**不装**，
   按 torch 大版本另建 oracle 环境。装进去等于作废全部生态对拍。
3. **两侧下游库版本一致**。harness 断言 `_versions()` 相等，并区分"能共享 site"与
   "ABI tag 不同各装一份"。
4. **离线固定输入**：`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；上游 checkout 钉
   commit SHA 并核对 remote URL。否则是拿两份不同权重在对拍。

## 4. 写 case，进生态门禁

按 [`_ecosystem_cases.py`](../../../compat/tests/torch/_ecosystem_cases.py) 的契约：
case 是个吃 `torch` 吐 `(model, inputs)` 的函数，**永远不知道自己跑在哪个 runtime 上**；
seeding、权重传递、序列化归 runner。这让"同权重同输入"是结构保证，不靠人记得。

- 正确性用 tiny config，快、确定、不联网；
- 真实尺寸留给 speed 那半，玩具尺寸量的是框架开销不是计算。

注册进 `CASES` 即自动获得全套断言：缺任何一个梯度即失败（抓断链）、forward/backward
容差按全场最大量级取 floor（防"数学上就是 0 的梯度被噪声放大成假 bug"）、两侧 device
相同、`tf32` 与 `runtime_conditions` 相同、`fallback_policy == "error"` 且
`fallback_count == 0`。

**没进 nightly 的库等于没接入**——下次谁改一行 shim 就悄悄坏了。

## 5. Device 阶梯：只跑 ① 和 ③

| | 跑什么 | 和谁比 | 回答 |
| --- | --- | --- | --- |
| ① | CPU | Jittor CPU vs **独立 PyTorch CPU** | 兼容层/算子对不对 |
| ③ | device | Jittor `<dev>` vs **真 PyTorch / `torch_npu` 同 device** | 加速卡声明成立吗 |

CUDA 目标 = ① + ③(cuda)；NPU 目标 = ① + ③(npu)。**① 是所有目标 device 共享的，
只跑一次**，所以新库先在 CPU 上收敛完最省。

**不做「Jittor CPU vs Jittor device」这一步。** 它的参考物是我们自己，而 ①③ 的参考物
是外部；两端都被外部钉死之后，两端一致是推论不是新证据。更关键的是**它恰好是静默
fallback 能骗过的那一个**：NPU 真发生 CPU 回退时，它会完美一致——最该报警的场景正是
它必然全绿的场景。

两个边界，别误删：

- **③ 拿不到时**（目标卡上的原生下游栈装不起来），退回用 Jittor CPU 对 Jittor device，
  **必须配 `fallback_count == 0`**，并在报告里标成**降级证据**，不能声称"与原生栈一致"。
- **核心算子门禁里的 device parity 层不要动**。
  [`test-system`](../../../docs/testing/test-system.md) 第 3 层跨 device 跑整个 OpInfo
  库，那里逐算子的外部 device oracle 根本拿不到，它是唯一可规模化的手段。删的是**新库
  接入流程**里的那一步，不是仓库已有的 parity 层。

工程约束：**一个 device 一个进程**（全局 flag 移动整张图，同进程混 device 不可能）；
**一个 device 一套 `JITTOR_HOME`**（JIT 文件锁）；**绝不从加速卡起手**，否则红了分不清
是兼容层 bug 还是后端 kernel bug。没硬件的后端（ROCm/Corex）诚实记"未验证"，
不拿 CPU fallback 冒充支持。

## 6. Adapter 准入清单

只有第 0 节第三行还剩东西时才建 adapter。目录形态是自足单元，`cp -r` 能整体搬走：
代码与它的 `tests/` 同住。进仓必须齐：

1. **断点表 + 分流声明**：为什么这不是 compat 缺陷、也不是 jittor 缺功能；
2. **继承共享 relocatable 契约**：只 import 公开入口白名单、不碰私有属性、不给
   `jt`/`jittor`/`torch` 任何属性赋值、扫描用 `rglob` 防新子包绕过、**规则本身带负向
   测试**、扫描排除 `tests/`；
3. **版本矩阵 fail-closed**：用
   [`_common.py`](../../../adapters/jittor_adapters/_common.py) 的 `require_version` 与
   `SUPPORTED_VERSIONS`，**不要自己造**；未验证版本显式拒绝，不能默默 arm 一个 finder
   去骗没验过的库；
4. **独立发行**：删掉 `adapters/` 整棵树，主仓其余门禁结果不变；
5. **真机验收报告**，`Review when:` 必须包含 **adapter 依赖的 compat 公开入口变更**
   （`owned_runtime_hook`、`transaction`、`module_patcher`）；
6. **库缺席的诚实 skip 类别**：库名进 `ENVIRONMENT_SKIP_PATTERNS`，并按
   `REAL_TORCH_PATTERNS` 的做法加 `JITTOR_REQUIRE_<LIB>=1`——普通机器 skip 且可见，
   验收机上"没装"从环境事实变成配置错误。**没有这个类别，真机测试根本进不了树**：
   `JITTOR_TEST_REQUIRE_EXECUTION=1` 会把"未安装"判成 unexplained 直接红；
7. **（stub 类）算子覆盖清单**：把 `<lib>._C` 做成 importable-but-empty，字面意思是
   告诉下游"kernels 都在"；缺一个算子它不报错，直接拿错结果跑到输出。上机时 hook 一次
   记录下游实际要了哪些算子，固化进仓，和我们提供的做差集。

**压力阀**：adapter 数量要被 justify，不是来一个库加一个。每个都是钉死在某个上游版本
上的负债。只给头号目标库开；不是头号目标却出现同样形态，正确答案是**"不支持，显式
报错"**。配套上收规则：**三个 adapter 在做同一件事，那件事就该进 compat。**

## 7. 双轨验证

| 轨 | 跑什么 | 在哪 | 能证明 |
| --- | --- | --- | --- |
| 离线契约 | stub 掉外部库，验注入/事务/生命周期/所有权 | **门禁内**，每次 PR | 我们的逻辑自洽 |
| 真机验收 | 真库、真卡、公开 API 端到端 | 门禁外的验收机 | 它真的跑起来了 |

**两轨都要，谁也替不了谁。** 契约全绿不代表库能跑；真机跑通一次也不代表下次改了
compat 还行——所以第 6 节第 5 条的过期条件是必须的。

## 8. 落盘

- dated `docs/results/YYYY-MM-DD-topic.md`：Status / Date / **Baseline commit** / Owner /
  **Review when**，记环境、命令、结果、边界；
- 真实 gap 进 [`known-issues`](../../manuals/known-issues.md)：owner、可执行证据、
  workaround、退出条件；
- 性能**永远报告，只在 nightly 断言**——墙钟上界是唯一能被机器负载单方面搞红的断言，
  假红会教人忽略那个同时还在查数值的门禁。

## 常见假绿

每一条都发生过：

1. **GPU 机器上"跑 CPU"默认不是 CPU。** Jittor 没有 per-tensor device，全局 flag 在有
   GPU 的机器上**初始就是开的**。只是"不请求 CUDA"的对拍，实际是 Jittor 在加速卡上对
   PyTorch 在 CPU 上——数字还对（两边都正确），但 CPU 那一半根本没跑，而且把 1.8x 变慢
   报成了 20x 加速。必须**显式关**。
2. **shim 对自己。** 缺 oracle 时用例会自我 skip（对的），于是丢了 oracle 的 nightly 会
   为它唯一存在的理由报成功。用 `JITTOR_REQUIRE_REAL_TORCH=1` 把这类 skip 变成失败。
3. **签名齐全的 no-op。** 接受全部参数、返回可信值、什么也不做。参见
   [`torch-shim-noop-audit`](../torch-shim-noop-audit/SKILL.md) 的三问：函数体读了每个
   参数吗？返回值依赖输入吗？用户会发现它什么都没干吗？**tiny case 通过对拍，不代表库
   实际用到的 API 面是真的。**
4. **只会 skip 的门禁条目，看起来和通过一模一样。** 227 个算子的反向公式就是这样在三次
   全绿里保持未验证。

## 相关

- 对拍 harness 与梯度调试：[`jittor-torch-diff`](../jittor-torch-diff/SKILL.md)
- 算子级独立参考：[`jittor-op-parity-oracle`](../jittor-op-parity-oracle/SKILL.md)
- 多卡/多后端验证：[`multi-device-verification`](../multi-device-verification/SKILL.md)
- 协作与 JIT 并发规则：[`collaboration`](../../manuals/collaboration.md)
