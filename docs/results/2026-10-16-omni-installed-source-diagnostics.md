# Omni 已安装源码 H3 诊断

- 状态：实际源码/产物绑定已验证；原始整模型数值与梯度协议失败。
- 日期：2026-10-16。
- 被测源码：干净的 `57b5f57ae220d1d8af1e6880ffcde6e362969110`。
- 维护者：下游验证与 vLLM adapter 维护者。
- 复查条件：固定版本 wheel/源码、生成的版本模块、模型派发或梯度契约发生变化。
- 平台信息生产者后续：干净的 `c18033fef8c44bfa4045a4108ba67128e7a5ee7e`；与原始诊断队列分开。

维护版 H3 探针现在会在导入模型之前绑定完整的实际 Omni 安装，包括非 Python 资源、原始
源码构建的 wheel、实际 METADATA 和 pip RECORD。它保留原始的混合 BF16/FP32 模型、固定
权重与输入、观察到的设备指针以及缺失梯度台账。这强化了[更早的 autograd 诊断](2026-10-15-h3-autograd-boundary.md)；
它不确立通过评测的模型、引擎、流水线或训练评分卡。

## 实际安装与被测配方

上游仍为 vLLM `98dff2a81d747d1dba01a47f939f48c3526d4206`（官方 0.29.0 wheel）和 Omni
`446c2b5474dabffccf9a0034836eeab77c62b4b8`。Omni 的源码身份是
`source@446c2b5474dabffccf9a0034836eeab77c62b4b8`；其实际 SCM 分发版本为
`0.29.0rc2.dev58+g446c2b547`。两个值都未被改写。
本地源码构建的 Omni wheel SHA256 为
`c6a43c816c95f99f751dd9aa18c36ff94461d508da002a2a328c543e14b2c0de`。
它不是 PyPI 二进制。其载荷是完整的 1593 个受跟踪包文件加上生成的 `_version.py`；实际的
pip RECORD 字节，包括 CRLF 和合法未哈希的生成字节码条目，保持原样。

独立解释器是真正的二进制 Torch 2.13.0+cu130。候选的核心、兼容性前端和 adapter 来自被测
的干净检出。双方执行原始的 one-main/one-refiner H3 模型，隐藏层大小 32，33268 个参数，
37 个初始参数/缓冲区数组，权重种子 3，输入种子 7/19。初始权重在各运行时之间逐字节
相同。原始张量保留其实际的 BF16/FP32 dtype；既有的 FP32 数组序列化保持不变。TF32 与
cuDNN benchmark 被禁用，matmul 精度为最高，所有线程控制为 1。CUDA UUID 分配和每个被
观察的 CUDA 指针都会被检查；候选记录物理派发和零回退。

这些是失败诊断，而非完整的公共平台记录：未来要纳入评分卡，仍需完整的 worker
驱动/编译器/provider/图模式区块。该诊断封存也不对 Torch/provider wheel 闭环作出完整
声明。一个独立的真实 Torch provider 预检在执行前重新哈希了其 12117 个官方 wheel 载荷
文件和实际 pip RECORD。

## 原始结果予以保留

| 执行 | 实际结果 |
| --- | --- |
| 真实 CPU 前向/评估梯度/训练梯度 | 三次非零退出：上游 CUDA 能力初始化需要 CUDA GPU |
| 带维护版私有 facade 的候选 CPU | 三次非零退出：Gloo 后端不可用 |
| 真实与候选 CUDA 前向 | 双方均执行；37 个初始数组逐字节一致；6 个输出中有 4 个超出原定的缩放 0.002 预算 |
| 真实与候选 CUDA 评估/训练梯度审计 | 双方两种模式均非零退出：两个种子下 36 个请求的参数梯度中有 32 个、全部四个请求的输入梯度均缺失 |

原始前向度量是 `max(abs(candidate-reference)) / max(1e-8, max(abs(reference)))`。一个
中间 `allclose` 审计使用了不同的度量，作为对原始预算的无效解读予以保留。修正后的审计
复现了历史上的缩放误差，并保留那四项失败。
34 个真实 CUDA 前向/部分梯度数组、全部初始权重和操作路径也与原始干净的
`521314943b480517b6fb422c09359fc521081a8f` 诊断逐字节一致。没有任何 MATH oracle、
强制整模型 FP32 转换或仅头部训练替代所请求的协议。

更早的失败另行保留：一个初始的严格 FP32 观察者错误地拒绝了模型原本的混合精度，六次候选
尝试在能输出安装包之前加载了真实的 Torchvision。修正后的观察者保留原始 dtype；维护版
私有 facade 解决了部署问题。这些失败未被跳过或覆盖。最终队列包含 18 条执行记录和 12 个
实际安装包。

## 生成模块反例与当前守卫

整合过程用真实的本地 wheel 复现了一个残余的源码证明缺陷：向 `_version.py` 添加可执行
载荷、重建 wheel 并安装到新的私有前缀，会得到一个自洽的 1594 文件安装，版本相同且
RECORD 正确，而先前的守卫接受了它。当前的有界配方在上述被审阅的本地 wheel SHA256 之外，
还固定了完整源码与安装检查。不同的合法构建需要经过审阅的来源。真实回归在修复前失败、
修复后通过；当前的仅主机守卫接受全部 12 个未改动的原始安装包，并拒绝那个自洽的生成
模块替换。这是后处理修复；原始被测源码和原始 JSON 保持不变。

## 复现与证据

使用维护版[探针 runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/vllm-omni-torch-compat/SKILL.md)，
显式指定原始 wheel 路径、独立解释器、私有 home、经验证的 facade 以及按设备分离的
benchmark 缓存。在 CUDA 之前如实请求 CPU；使用 `--mode grad-audit --model-mode eval` 和
`--model-mode train` 保留完整的缺失梯度失败。既有输出和源码树产物会被拒绝。

未纳入版本控制的诊断证据位于
`$JITTOR_LAB_ROOT/_state/omni-installed-source/20261016/`。
`diagnostic-cohort-seal-v3.json`，SHA256
`d91c955a5c662418c110ae6859c1ba60dd8cbdfb3a8fde29ed8fa6dcc3c7b911`，
绑定 2895 个原始 Git blob 和 8206 个原始/实际已安装包文件（844358101 字节）。原始失败
尝试、真实配方等价性、`forward-numerical-audit-v4.json` 与
`gradient-ledger-audit-v3.json` 均予以保留。
当前整合的反例、单测回归和 12 包产物复核位于
`$JITTOR_LAB_ROOT/_state/omni-main-integration-e7/20261016/`。

中央整合方独立重新哈希了该完整封存，并检查了每个原始 Git 树条目、145 个历史真实数组/
初始状态、37 个跨运行时初始状态和 290 次物理张量观察。验证：54 项定向主机检查，CPU 核心
322 passed / 42 skipped / 1 个既有 xfail，扩展结构 1628 passed / 6 skipped；布局、manifest
和 Ruff 通过。全部 74 个 nox 函数体保持不变。这些检查验证源码证明和保留的失败诊断，
而非一个通过评测的整模型协议。

## 共享平台信息生产者后续

一次独立的干净 `c18033fef8c44bfa4045a4108ba67128e7a5ee7e` 执行检查维护版探针是否在模型
导入之前以及最终结果中输出其共享平台区块。原始 `57b5f57` 包保持不变，且没有该区块。
新的真实 CPU 运行输出一个安装/平台包，随后以上游同样的 CUDA 要求非零退出。新的真实
CUDA 运行完成了原始前向配方：全部六个输出数组和全部 37 个初始参数/缓冲区数组都与原始
真实 CUDA 诊断逐字节相同。全部 43 次张量观察在分配的逻辑设备上都有实际的 CUDA 指针；
仅凭清单不作为放置证明。

新的 worker 记录实际 OS/CPU、加速器型号与 UUID 分配、驱动、Torch CUDA 构建、可用编译器
版本、解析到的 NVIDIA 发行版、精度与执行标志。CUDA worker 报告驱动 580.178.04、Torch
2.13.0+cu130、可用 NVCC 12.2 和 GCC 11.5；可用的 NVCC 不被声称为构建 Torch 所用的编译器。
TF32 与 cuDNN benchmark 保持禁用，matmul 精度为最高，参考执行使用 eager。本后续验证的是
真实生产者元数据，而非新的候选一致性、完整的 provider wheel 来源或整模型准入。原始缩放
误差失败与缺失梯度仍未关闭。

Accelerate 现在使用同样的字段构造器。从记录的观察输入重放全部 48 个原始 `c7c2d37`
worker 头部可复现每个字段；当前公共采集器仍接受两条单 rank 记录，并阻止两条未配对的
CUDA FSDP2 记录，拒绝数为零。这是格式/语义重放，不是新的 Accelerate 数值运行，也不是
追溯性的源码更新。

后续验证：121 项定向检查，CPU 核心 322 passed / 42 skipped / 1 个既有 xfail，扩展结构
1628 passed / 6 skipped；布局、manifest 与 Ruff 通过。全部 74 个 nox 函数体未变。新的
实际记录、源码前后检查以及独立的数组/平台/安装审计未纳入版本控制，位于
`$JITTOR_LAB_ROOT/_state/runtime-platform-main-53d/20261016/`。
`integration-seal-v1.json`，SHA256
`de15cc28fa35f49306ffac6a43ae57e1ce58a074cc277805b1c90ab8dcbf8671`，
绑定 2905 个被测源码 Git blob 和 1635 个证据/实际 Omni 安装文件（33193538 字节）；每个
被封存文件都被独立重新哈希。编译器缓存和完整的 Torch/provider 内容不在本次后续封存范围
之内。
