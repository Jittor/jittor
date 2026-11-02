# Safetensors torch compat 开发记录

> 本地维护者交接记录，不随 Jittor 仓库提交。原始证据位于 `/home/cgq/jittor-lab/_state/safetensors/`。

## 基本信息

- `safetensors==0.6.2`；Oracle PyTorch `2.6.0+cu124`；Jittor shim `1.3.11.0`。
- Transformers `4.56.2`、PEFT `0.17.1`；基线 `54cd55cc`；更新日期 2026-09-14。

## 阶段历史

2026-09-14 的 L0/L1 fixture 验证了 save/load、safe_open、metadata、key、shape、dtype、scalar、empty tensor 和逐值读取。两侧文件 SHA 不同属于 header/字节布局差异，不是张量语义差异。

同日修复 `compat/torch/serialization/safetensors.py` 的 shared alias 错误契约：保存前检查 storage address，重复 storage 抛 `RuntimeError`。读取端加入有界 header 长度、文件大小、UTF-8 和 JSON object 校验，损坏文件 fail-closed 抛明确兼容异常。

PEFT adapter `.../pytorch_v2/trained/second/adapter_model.safetensors` 两侧均加载 5 个 key，完整逐张量 shape、dtype 和原始字节 SHA 一致。CUDA 小检查中 `load_file(device="cuda:0")` 两侧均成功，返回张量实际位于 `cuda:0`。

## 当前等级与边界

当前等级：L0 PASS、L1 PARTIAL、L2 LIMITED、L3 OPEN、L4 LIMITED、L5 PARTIAL。non-contiguous 保存仍开放：shim transpose/as_strided 在序列化边界前物化为 contiguous，不能复现 PyTorch 的 `ValueError`。C5 暂跳过该项，退出条件是 native/pyjt 提供共享 storage 的 stride+offset view 构造 API。

L2 仅覆盖 PEFT adapter 的完整 key/shape/dtype/value SHA 与加载；未完成完整 Transformers 模型继续使用。L4 只有真实 CUDA `cuda:0` 加载证据。L5 只有 PyTorch 约 256 MiB 单进程缓存读取，未形成有效 Jittor 对照，不宣称性能等价。

## 入口与下一步

- 基础 fixture：`_state/safetensors/l0_fixture/run_fixture.py`。
- 非连续 probe：`noncontiguous_probe.py`，日志 `_state/safetensors/noncontiguous_probe/{pt,jt}/out.log`。
- 两环境分别 source `transformers_compat/env_pytorch.sh`、`env_jittor.sh`，使用独立 `RUN_ROOT`。

后续由 native view owner 评估 C5 退出条件，再补模型级 round-trip、CUDA 下游访问和连续大 fixture 双端吞吐/RSS/hash/多进程稳定性。禁止依赖错误的 `is_contiguous()` 或把 CPU fallback 当真实设备证据。

## 2026-09-14 详细验证记录

### L0 环境与版本

PyTorch 环境通过 `transformers_compat/env_pytorch.sh` 启动，Jittor 环境通过 `env_jittor.sh` 启动；两侧均使用离线缓存和独立 `HOME`、`HF_HOME`、`JITTOR_HOME`、`TMPDIR`。两个 site-packages 的 safetensors 版本均为 0.6.2，调用同一 Rust 序列化协议。

### L1 基础 API

基础 fixture 覆盖三维普通张量、标量、空张量、metadata、key 枚举、shape/dtype 和逐元素读取。连续数据在 CPU 上双侧一致，真实 CUDA `device="cuda:0"` 读取也成功。文件 header 的键顺序和字节 hash 可以不同，不作为语义失败。

shared alias 初始会被 shim 接受；修复后通过 storage address 检查抛出 `RuntimeError`。损坏的 8-byte header 初始触发 `MemoryError`；修复后通过长度、文件边界、UTF-8 和 JSON object 检查抛 `_SafetensorError`。这些检查没有宽泛捕获 I/O 异常。

non-contiguous 是独立未完成项。`noncontiguous_probe.py` 显示 PyTorch transpose 保留 alias 和 stride，Jittor transpose 的 storage address 改变且 `is_contiguous()` 结果不可靠；因此保存错误契约仍不一致。

### L2 PEFT 证据

固定文件 `_state/PEFT/adapter_lifecycle_20260913/pytorch_v2/trained/second/adapter_model.safetensors` 在两环境完整加载 5 个 key。逐 tensor 的 shape、dtype 和 CPU 原始字节 SHA 全部一致；这构成 LIMITED 证据。该文件来自既有 lifecycle 实验，未把 optimizer pickle 互通外推为 safetensors 能力。

### L3 与 L4 边界

尚未完成模型级 adapter save→load→继续前向或训练，因此 L3 OPEN。L4 LIMITED 仅表示真实 CUDA load 的返回张量均位于 `cuda:0`；没有据此声称完整下游推理、训练或多设备覆盖。

### L5 性能边界

PyTorch 完成约 256 MiB 连续 fixture 的 warmup 后三次单进程读取，记录吞吐和 RSS。Jittor 因 view/contiguity 语义问题未形成有效同条件对照；没有多进程、峰值显存或大文件性能结论，所以 L5 仅 PARTIAL。

## C1-C5 分流

C1（纯上游格式限制）：safetensors 对非连续和共享存储的拒绝是上游契约，shim 需要复现错误类型。

C2（compat 签名适配）：metadata、key 和 device 参数已在 shim 中适配并通过基础测试。

C3（组合语义）：PEFT adapter state-dict 的 key、dtype、shape 和字节内容已完成对拍。

C4（实现缺陷）：transpose/as_strided 应保留共享 storage view，但当前实现物化结果；若 native 有能力，应由 view owner 修复。

C5（ABI/backend 缺口）：当前未发现公开的 pointer+stride+offset 共享 view 构造入口，因此 non-contiguous 暂按 C5 跳过。协作动作：需同步至“严重问题”文档，并经全体开发者确认后再修改相关公共实现。退出条件是 native VarHolder/pyjt 提供并验证该构造 API。

## 交接计划

后续维护者先运行 `noncontiguous_probe.py` 核对真实 stride、storage offset、alias 和写回，再决定 C4 实现或 C5 维持。随后补 Transformers/PEFT 模型级 round-trip、CUDA 下游访问，以及连续 64 MiB 以上 fixture 的双端单/多进程吞吐、RSS、hash 和重复读取稳定性。所有性能数字必须匹配设备、dtype、字节数、warmup 和 repeat 条件。
