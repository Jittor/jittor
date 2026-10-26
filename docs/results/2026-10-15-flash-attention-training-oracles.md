# FlashAttention 公共 CUDA 训练与独立 PyTorch 对拍

- 状态：48 项历史 CUDA 比较和 12 个既有 CUDA 训练用例通过。十二条 L3 记录通过
  共享 schema；由于缺少匹配的独立上游 CPU 执行，广度仍为 0/7。
- 被测 Jittor 源码：干净检出 `71c5be74ef6991b75fa79d73fe52eb04c0319d59`。
- 维护者：原生 CUDA 与 Torch 兼容性维护者。评审日期：2026-10-15。
- 复查时机：上游 pin、扩展 ABI、公共 facade、原生反向、优化器算术、精度协议
  或比较工具发生变化时。

## 既有训练与受控扰动

干净检出 `06ed62ad728fae399ccbdbadec7a2441a13eaed9` 无跳过地运行了
`compat/tests/torch/test_ecosystem_flash_attention.py` 中的全部 12 个节点。六个
公共训练 API 各自在 head 维度 32 和固定 noncausal attention 下执行 FP16 与 BF16。
每个节点都对基线输入和一个 `0.125` 噪声扰动比较了独立 Torch 与 Jittor。
Q/K/V、目标、因果性和形状保持不变；输入哈希与实际变化的输出在两个运行时上都独立
证明了扰动确实生效。

24 条配对的三步轨迹包含 720 次数组比较和 216 次独立的更新比较。全部通过预先声明的
各 dtype 原始预算。包含更新在内的最大归一化误差为
`0.002433830240340736`。记录了实际 CUDA 放置、零后备、全部梯度、FP32 主副本、
精度回读以及 58 个受追踪的上游 Python 文件哈希。上游与独立二进制 Torch 的 pin 和
平台如下文所述保持不变。这是正确性证据；并发作业使得无法进行速度验收。

可复用收集器在调用共享 scorecard 审计器之前，会独立复核原始数组、不可变的比较
元数据、源码标识和输入扰动。其测量 schema 接受 12 条 CUDA L3 记录，拒绝零条。
**共享加速器广度仍为 0/7**，因为该指标要求匹配的独立上游 CPU 比较。上游
FlashAttention 不提供 CPU 执行；我们没有编造任何 CPU 证据或豁免。另一项独立的
二进制 Torch 探针显式隐藏 CUDA、在 CPU 上分配 FP16 Q/K/V，并调用真实固定版本的
公共函数。Torch 对 CPU 后端拒绝了 `flash_attn::_flash_attn_forward`；完整结果
作为未纳入版本控制的 `official-cpu-boundary.json` 保留在同一 state 根目录下。
实际执行的 CUDA API 范围为七个固定公共 API 中的六个，KV cache 未测量。

原始 `gate-schema-complete/` 产物和生成的 scorecard 未纳入版本控制，
位于 `$JITTOR_LAB_ROOT/_state/flash-attention-audit/` 之下。更早的干净检出
`799af7118b352d5ac4621c5d657934dd2f6acc71` 门禁也通过了 12 个 CUDA 节点；
其原始文件保持不变。随后补充了规范的训练元数据和源码哈希，并在 `06ed62ad7` 上
重跑全部 12 个节点；没有修补旧元数据。下文历史 48 用例协议保留其原始源码 pin。

## 历史执行范围

两个独立解释器都导入了公共 FlashAttention API。参考方使用二进制 Torch
`2.11.0+cu128`，确认其真实的 `_C` 扩展，且没有导入 Jittor。其 FlashAttention
Python 包来自干净的官方 `Dao-AILab/flash-attention` 检出，版本 `2.7.4.post1`，
提交 `5231d95fe13733fb534c01895f7ea88c6a6c7793`。参考扩展依据该上游 `setup.py`
中的全部 85 条源码声明构建，使用上游 CUDA flags，未修改源码。在数值比较之前，
实际的 CUDA 前向/后向 smoke 已通过。

候选方使用 Jittor 的公共 Torch facade 以及原生所需的上游 CUDA 扩展桥接。
每个部署的 facade Python 文件都与被测仓库资源一致。探针检查了实际 CUDA 放置、
真实 half/BF16 输入输出 dtype、FP32 主副本与梯度、有限值、参数更新以及零后备守卫。
导入扩展没有被当作执行的证据。

自动重新生成的上游 `attention_api` 分母有七个成员。此处执行了六个，
**在固定的上游 SHA 上为 6 / 7**，每个都覆盖八种场景：

| 公共 API | fp16 场景数 | BF16 场景数 |
| --- | ---: | ---: |
| `flash_attn_func` | 4 | 4 |
| `flash_attn_kvpacked_func` | 4 | 4 |
| `flash_attn_qkvpacked_func` | 4 | 4 |
| `flash_attn_varlen_func` | 4 | 4 |
| `flash_attn_varlen_kvpacked_func` | 4 | 4 |
| `flash_attn_varlen_qkvpacked_func` | 4 | 4 |

场景维度为 head 维度 32/64 以及 noncausal/causal 执行。causal 场景还会把某一个
可微输入元素扰动 `0.125`。这一配对并不构成因果性固定下的独立受控扰动比较。
`flash_attn_with_kvcache` 仍不在本训练探针范围内。
这些计数不构成对内部 API、全部 attention 特性、生产模型规模或整个库的覆盖。

每次运行创建三个 FP32 可训练 Q/K/V 张量和一个独立的 FP32 可微输入，把 attention
操作数转换为请求的 fp16/BF16 dtype，并以学习率 `0.01` 执行三次真实 SGD 更新。
稠密张量形状为 `(2, 11, 2, head_dim)`；打包的变长输入长度为 `[4, 7]`。
固定的 NumPy seed `20261015` 在两侧提供相同的非零权重、输入和目标。dropout 为零。
每个运行时为每个场景保留 30 个数组：初始参数以及每一步的输出、loss、输入梯度、
全部参数梯度和更新后的参数。

## 平台与验收

| 字段 | 候选方 | 独立参考方 |
| --- | --- | --- |
| 源码/运行时 | 被测 SHA 上的 Jittor 2.0.0 | 二进制 Torch 2.11.0+cu128 |
| 实际加速器 | 一块可见的 NVIDIA GeForce RTX 4090 | 同一块分配的 GPU |
| CPU / OS | AMD EPYC 9554；Linux 6.1.188，glibc 2.34 | 同一主机 |
| 驱动 | 580.178.04 | 580.178.04 |
| NVCC | 12.2.140 | 12.8.93 |
| C/C++ 编译器 | GCC 11.5.0 | GCC 11.5.0 |
| 精度 | TF32 禁用，最高的 float32 matmul 精度 | 相同的请求策略 |
| 线程 / 输入 | 两个运行时线程；离线输入 | 相同 |
| 图模式 | 默认开启原生重放 | 普通 Torch 执行 |

每个运行时的原始 JSON 保留编译器输出、NVIDIA wheel 版本、完整驱动/设备清单、
实际二进制路径/哈希和原生 flags。原生 `use_acl` 是 `use_cuda` 的遗留别名；
其记录值不声称 NPU 执行。主机上的其他工作使用了不同的设备/缓存，因此这些结果
只确立正确性，不作出速度或整进程内存主张。

比较器的限值在首次匹配比较之前就已提交：fp16 前向/状态最大归一化误差 `0.003`，
梯度/更新 `0.006`；BF16 两者均为 `0.03`。归一化使用每个参考张量的最大绝对值，
下限为 `1e-8`。初始参数必须精确一致。九次参数更新中的每一次也单独比较，
因此被省略的小幅更新无法藏在大初始参数背后。

全部 **48 项比较通过**，覆盖**每个运行时 1440 个数组**和
**1872 次张量/更新检查**。最大归一化误差为：

| 字段 | fp16 | BF16 |
| --- | ---: | ---: |
| Attention 输出 | 0 | 0 |
| 全部参数与输入梯度 | 0 | 0 |
| Loss | 0.0000292143 | 0.00147610 |
| 更新后的参数值 | 0.0000000828791 | 0.0000000414392 |
| 单独测量的参数更新 | 0.00305811 | 0.00304879 |

输出和每个梯度数组都逐位一致；每个梯度字段均非零。Loss 与优化器更新的一致性
受原始限值约束，而非声称逐位相同。比较器还验证了匹配的协议、干净源码提交、
脚本哈希和 NPZ 哈希。其 21 项宿主契约覆盖缺失梯度/更新、脏源码、后备报告、
协议不匹配、损坏数组以及 Python `-O` 关闭断言的情形。

## 复现与证据

使用 [FlashAttention skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/flash-attention-torch-compat/SKILL.md)
中的隔离解释器与命令。以相同的 API/dtype/head 维度/causal/扰动参数，对每个运行时
各运行一次已提交的 `flash_training_probe.py`，然后对两个 NPZ 文件调用
`compare_training.py`。请保留被测检出；使用更新的工具或运行时需要新的证据。

未纳入版本控制的证据位于
`$JITTOR_LAB_ROOT/_state/flash-attention-audit/formal-71c5/`：
`candidate/`、`reference/`、`comparison/` 和 `accepted-evidence.json`。
验收清单绑定 385 个原始文件，SHA256 为
`def6b6f28daa4ad89184a6f4ba63c2b678578e9dfaf066413d910d441a163d91`。
脚本 SHA256 为
`c5e1ea376038e4e04da968599ac69bd3a2618815864e43a82b0aa7a7259469aa`。
同级的 `oracle-build.json` 和 `logs/oracle-build-v3.log` 保留完整的构建协议和
实际二进制 smoke 结果。更早失败的工具链搭建尝试与脏源码诊断单独保留。

上游 FlashAttention 没有可供此处比较的 CPU CUDA 扩展执行。CPU facade 契约与原生
CPU 回归是另外的证据。跨框架的 dropout RNG 一致性、KV-cache 推理、完整模型、
性能、NPU 与多 GPU attention 均未被本结果认证。
