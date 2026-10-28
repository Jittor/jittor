# 既有 PEFT 与 Swift CPU nightly 门禁

- 状态：workflow 接线加本地执行；未触发也未观察到任何 GitHub 运行。
- 日期：2026-10-15
- 被测源码：`08a3ee3ca981855db55e3d7a38d7ba6503b73e7e`
- 在最终 helper 改动之前集成的上游基线：`0fb3920d4a44f37fa6cea7fa856374f0bacf73b4`
- 维护者：Jittor 兼容性维护者
- 复查时机：workflow 镜像/Python、包 pin、供给流程、已安装源码契约或任一既有门禁发生变化时。

## 入口与范围

`.github/workflows/ecosystem.yml` 保留既有的通用 `ecosystem` job，并为 `peft` 和
`swift` 新增独立的 `maintained-cpu` matrix job。定时与手动 workflow 事件使用相同的
job。每个 job 调用既有的完整 maintained `ecosystem_peft` 或 `ecosystem_swift`
会话；缺失溯源、库、oracle 或要求的执行都会报错。不新增任何 adapter 或模型/优化器数学。

在健康的 Linux Python 3.11 安装与原生 C++ 工具链上：

```bash
python3.11 tools/provision_ecosystem_nightly.py --library swift \
  --state-root "$JITTOR_LAB_ROOT/_state/ecosystem-nightly/<run>/swift" --run
python3.11 tools/provision_ecosystem_nightly.py --library peft \
  --state-root "$JITTOR_LAB_ROOT/_state/ecosystem-nightly/<run>/peft" --run
```

使用新的 state 根。CI 使用 `RUNNER_TEMP`；检出、wheel、环境、HOME、
TMPDIR、模型缓存、编译缓存和日志都留在源码检出之外。
可选的只读源码/wheel 缓存参数保留精确源码检查和完整依赖解析。
它们不会复制已安装环境或 JIT 二进制。正式执行之前源码检出必须干净。

## 环境与安装证明

每个库获得三个全新环境：controller 使用规范的 nox/build 依赖；
candidate 和 oracle 各自安装完整的真实运行时依赖闭包。运行时 setuptools 80.9.0 遵循
官方 Torch wheel 的 `<82` 要求；规范 setuptools 83 保留在独立的 controller 中。
三者都运行 `pip check`。oracle 必须加载真实的 Torch `2.11.0+cpu` 及其二进制扩展。
只有 candidate 激活仓库官方的可编辑 Torch shim。CUDA 被显式隐藏，规范的与遗留的 CUDA
标志均为零。两个实际 CPU Torch wheel 的 SHA256 均为
`8a56a8c95531ef0e454510ba8bbd9d11dc7a9000337265210b10f6bfeacdd485`。

既有源码仍为 PEFT 0.17.1 和 Transformers 4.56.2，确切的官方 Git SHA 与 wheel SHA256
值记录在 `_peft_scorecard.PINS` 中。Swift 4.5.2 由干净检出
`ff5128777dd3dc5538f1a95fbcc29442c327d7d2` 构建，不被描述为 PyPI wheel。NumPy 1.26.4、
Accelerate 1.10.1 以及既有的数值预算保持不变。Swift 还固定 PyArrow 19.0.1，
并在首次 JIT 之前实际检查一次 Arrow/NumPy 往返和 Dataset 构造。wheel 文件、解析后的
lock、干净上游清单以及已安装载荷/METADATA/RECORD 标识仍保存在未纳入版本控制的证据中。

独立的安装前缀暴露了既有 Swift 校验器的一个假阴性：绝对包根和依赖来源被跨运行时比较。
修复保留原始路径，各自在本侧校验实际包根和完整已安装载荷，并从跨运行时相等比较中只
排除 `package` 和每个依赖的 `origin`。版本、每个内容哈希和所有其他标识字段保持精确。
完整的四依赖集合为必需。真实的独立前缀 fixture 以及缺失 origin/
内容/版本/wheel/元数据/根/键集合的反例覆盖了这一改动。没有放宽任何数值阈值。

## 本地执行

Swift 的全新供给使用 `af34c07aef9725423ba08b4b1aef647e4f8a3645`。其 workflow 与
provisioner 字节在最终测量源码 `08a3ee3...` 处完全相同；最终的 helper 验证在这些任务
私有的 candidate/oracle/controller 环境中运行。新输出记录两个 SHA 以及该任务指纹缓存的
正常复用；它不声称第二次冷构建，也不重新标注更早的失败运行。

| 检查 | 实际结果 |
| --- | --- |
| Swift 既有 FULL SFT CPU 节点 | 1 通过 / 0 跳过，97.47 秒 |
| Swift 三环境 `pip check` | 全部通过 |
| Provisioner 与 Swift 契约宿主测试 | 77 通过 / 0 跳过 |
| PEFT 既有完整 CPU 门禁 | 通过；177 组数组对；最大绝对误差 `1.4543533325195312e-05`，缩放相对误差 `3.135175185365948e-07` |
| 布局 | 通过 |
| 完整 `tests/structure` + `compat/tests/structure` | 1621 通过 / 4 跳过 / 3 失败：缺少单元 ASV 以及待定的中央 toctree/MANIFEST 登记 |
| 仅在私有单元目标中供给依赖后原 ASV 失败项 | 1 通过 / 0 跳过 |

Swift 节点执行已注册的 tiny Llama loader/template 与真实的
`SwiftSft.main()`/`Seq2SeqTrainer.train()`、三次 SGD 更新、全部十二个可训练参数梯度
和一个嵌入中间梯度，使用三种实际输入变体与一个独立的二进制 Torch oracle。整型 token
输入不请求输入梯度。这是有界的数值训练用例，不是整库覆盖。

PEFT 会话运行六个真实 worker：每个运行时执行基线、seed 7 输入扰动和三步训练。
基线/扰动请求全部 29 个参数梯度；训练保留八个可训练 LoRA 参数和 21 个冻结参数，
检查全部三次更新，并使冻结参数保持字节一致。其原有的严格收集器接受全部 177 组数组对。
这是既有的完整 tiny LoRA 范围。

完整结构运行保留了原有的三处失败。新运行时中缺少 ASV，因为它不是下游运行时依赖；
把规范的 `asv==0.6.6` 及其真实闭包安装到**独立的仅单元目标**中，同时保持两个被测环境
不变，使那个确切的失败节点通过。其余节点是
`test_docs_structure.py::TestDocsStructure::test_every_page_is_reachable_and_every_toctree_entry_exists`
和 `test_packaging_structure.py::TestPackagingStructure::test_manifest_covers_runtime_trees_without_cache_payloads`。
中央 results toctree/MANIFEST 登记与最终的集成结构验证仍由集成维护者负责；
本地完整运行不被声称为全绿。

## 保留的失败尝试

| 尝试 | 实际的终态观察 |
| --- | --- |
| Swift v1 | 官方 Git HTTPS 传输失败；无数值结果 |
| Swift v2/v4 | 在准备经过验证的私有重放输入时，依赖传输在 JIT 之前被有意中断 |
| Swift v3 | pip 拒绝规范 setuptools 83，因为官方 Torch 的运行时依赖要求 `<82` |
| Swift v5 | 离线安装错误地把两个 fsspec 版本请求为根；通过解析最初声明的根修复 |
| Swift v6 | 真实的 PyArrow 26 二进制需要 NumPy 2，尽管 pip 元数据兼容；保留了原有的 NumPy 1 协议 |
| Swift v7 | 真实的独立前缀训练到达元数据比较，因不同的安装路径被拒绝；该失败节点不声称任何成功的数值结果 |

全部尝试保留其原始源码/日志/退出标识。引导修复与已验证下载的复用仅限于任务私有输入；
共享安装位置为只读。v7 最终环境的首次 CPU JIT 从空缓存串行执行。一个未使用的复制缓存
在任何 v7 JIT 之前被撤回，并如实记录；没有使用任何复制的二进制来声称新的冷构建。

## 证据与限制

未纳入版本控制的证据位于
`$JITTOR_LAB_ROOT/_state/ecosystem-nightly/20261015/`。它包含 `swift-v7/provision.json`、
`swift-final-08a/{measurement,commands,completed}.json`、既有 nox 日志、
`peft-v1/`（含其既有六 worker 数组与验证记录）、原始尝试、真实包前缀复现记录以及
`source-08a-complete.json`（2,867 个 Git blob，含符号链接目标文本）。最终清单绑定
选定的原始日志/证明和源码快照；环境、wheel、编译二进制和缓存被排除在该清单与仓库
之外。`manifest.json` 绑定 **313 个文件 / 29,123,587 字节**，SHA256 为
`14f583115e57e2b616173731ae62c6126acfbc762093cc7aabb9e51ee693c4bf`。

本改动接入了 nightly 执行，并仅在本地验证上述 Linux CPU 会话。它不证明已完成一次
GitHub 定时运行，也不证明 CUDA/NPU/ROCm/Windows、性能、分布式/完整 checkpoint 工作流
或对整个上游库的支持。既有的纳入版本控制的 CPU/CUDA scorecard 保留其原始测量 SHA 与
广度。专门的 VERL/Transformers/MM nightly 集成仍是独立的后续工作；
这项涉及两个库的改动并没有关闭全部 R32。

## 主集成验证

集成维护者同步了已发布的
`cd7d45ab99e086f2b027be35bffa66d3b33efdd7`，并保留上文的原始被测源码标识。
干净集成源码 `c4935e9061964f17fa7d8ac4fb9930e1b84f5bfd` 增加了既有门禁、源码契约、
发现机制以及中央 results/MANIFEST 登记。全部 73 个既有 nox 函数 AST 未改动；
只新增了 `ecosystem_openmmlab_installed`。

在该集成源码上，供给、Swift、distribution 名称、已安装源码/版本证明、OpenMMLab、
共享覆盖率、VERL、PEFT 和 Transformers 的宿主契约通过 **228 / 0 跳过**。
规范 core 通过 **322 / 42 跳过 / 1 项既有 xfail**，原生与 Torch 两种模式的退出码
均为零。完整 `tests/structure` 加 `compat/tests/structure` 在 235.97 秒内通过
**1622 / 6 跳过**。六项跳过保留其加速器/已声明/未安装原生分发的边界；
不推断任何安装或硬件主张。布局、生成的清单、Ruff 和空白检查通过。

维护者独立重新哈希了全部 313 个 nightly 证据文件和全部 1033 个 OpenMMLab 证据文件：
无不匹配。原始失败尝试与测量/源码标识予以保留。c4935 上的全新隔离既有运行记录在
下文；集成检查不会把原始数值结果重新标注为 c4935 测量。未纳入版本控制的集成记录位于
`$JITTOR_LAB_ROOT/_state/maintained-gates-main-integration-cd7/`。


## c4935 的全新执行与独立评审

2026-10-16，从干净检出 `c4935e9061964f17fa7d8ac4fb9930e1b84f5bfd` 重跑了完整的
既有会话，使用六个新建环境（每个库三个）、独立的空编译缓存以及未改动的公共
provisioner。经过验证的官方 wheel/源码输入以只读方式复用；
未复制已安装环境和 JIT 二进制。全部六条记录的 `pip check` 命令返回零。
CPU 执行显式隐藏 CUDA，oracle 加载了真实的 Torch 2.11.0+cpu 二进制。
Swift 环境还通过了其实际的 Arrow/NumPy/Dataset ABI 预检。

Swift 的完整 FULL SFT 维护节点通过 **1 / 0 跳过**，152.10 秒。
PEFT 的完整六 worker 门禁在原始预算下通过 **177 组数组对**，
最大绝对误差 `1.4543533325195312e-05`，缩放相对误差
`3.135175185365948e-07`。集成维护者针对冻结的 c4935 源码和真实的新原始数组独立重跑了
完整的原始 PEFT 收集器：基线/扰动梯度、全部三次更新、八个可训练参数和 21 个未改变的
冻结参数通过。这些是测试时长和数值结果，不是性能测量或整库支持。

新的封存位于未纳入版本控制的
`$JITTOR_LAB_ROOT/_state/ecosystem-nightly-main-recheck/20261015/manifest.json`：
**125 个文件 / 22,361,671 字节**，SHA256 为
`3097d96ff54ba684f9690b048b9a62e36ee2c44ce219fd9fd5ee72f61cd210a7`。
集成维护者独立重新哈希了每个封存文件，无不匹配。PEFT 的完整原始数组、阶段元数据和
最终验证予以保留。Swift 遵循未改动 harness 的临时数组清理；其新的保留证据是执行日志与
完整源码/安装日志，而不是导出的新共享 scorecard。该限制不重写更早的 Swift CPU/CUDA
数值证据。

原始的 08a3 测量与 313 文件封存保持不变。新的源码标识和 2,877 文件的 c4935 Git 快照
相互独立。独立的中央 PEFT 重算与封存检查位于
`$JITTOR_LAB_ROOT/_state/ecosystem-nightly-main-recheck-root-ffbb/`。未触发也未观察到
任何 GitHub 定时运行；本次 CPU 复查不授予 CUDA/NPU、整库覆盖或其余库的 nightly 门禁。
