# Ascend 910B3：冷缓存启动、ACL 执行与 NPU 门禁

- 状态：在维护的 NPU 门禁范围内接受（`397 passed, 9 skipped`）；9 个显式 skip
  对应的能力边界仍开放，见下文「仍开放的边界」
- 日期：2026-08-28（2026-09-01 复查）
- 基线提交：`532c250b`
- 验证范围：一张 Ascend 910B3（Linux aarch64），驱动/`npu-smi` 25.5.1，CANN 9.0.0，
  Python 3.9.25，NumPy 1.26.4，pytest 7.4.4；从空的 Jittor/ACL 编译缓存冷启动，
  先串行预热，再跑完整 `nox -s npu`。未覆盖：ACL float64、训练全流程、分布式 NPU、
  ROCm
- 维护者：昇腾后端维护者
- 复查条件：CANN 或驱动版本变化、`nox -s npu` 的用例集合变化、下文任一 skip 被
  移除或新增

## 结论

当前源码在真实 910B3 上从冷缓存完成了验证：探测 CANN 与 ACL 设备、执行 float32
matmul 探针，并覆盖 ACL 后端、扩展算子、索引、227 项 OpInfo、负整数 floor-divide
与 NaN 比较回归。设备证据不依赖「导入成功」或 CPU fallback：基线时 ACL matmul 回归
同时断言算子在 ACL 上编译、没有回退到 CPU；当前 `tests/backends/acl` 的 autouse
fixture 改用 `forbid_backend_fallbacks()` 的回退计数，不再读日志。9 个 skip 全部是
已登记的能力边界，没有一个是「环境没准备好」。

## 复现

```bash
export CANN_SET_ENV=<CANN 安装目录>/set_env.sh
export JITTOR_CI_PYTHON=<昇腾环境的 python>
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>
python -m nox -s npu
```

session 先跑 `npu-smi info` 与带精确期望值的 matmul 探针，再按进程模式分组跑
`NPU_TESTS`（`noxfile.py`），并设置 `JITTOR_TEST_REQUIRE_ACL=1`、
`JITTOR_TEST_ACCELERATOR_MIN_EXECUTED=1`，缺卡时整组自我 skip 不会读成绿。首次
编译（核心、ACL 扩展、模型算子）须串行完成后再跑整套门禁；缓存放在仓库外，用独立
`cache_name`。

## 门禁结果（基线提交）

| 阶段 | 结果 |
| --- | ---: |
| ACL 设备与 float32 matmul 探针 | passed |
| `tests/backends/acl/test_acl.py` | 41 passed |
| `tests/backends/acl/test_acl_torch_compat.py` | 16 passed |
| `tests/backends/acl/test_aclop.py` | 112 passed, 2 skipped |
| `tests/backends/acl/test_acl_indexing.py` | 4 passed |
| `tests/ops/test_ops.py`（OpInfo） | 220 passed, 7 skipped |
| NPU floor-divide 定值向量与广播 | 2 passed |
| NPU float32 NaN/Inf 谓词 | 1 passed |
| NPU float32 融合/非融合 NaN 比较 | 1 passed |
| 合计 | 397 passed, 9 skipped |

（基线时这些文件位于 `tests/backends/npu/`，现已移到 `tests/backends/acl/`。）
floor-divide 覆盖 uint8/int8/int16/int32/int64 定值向量与 int64 广播，对照 NumPy；
NaN 比较覆盖六种比较形式在同一/不同 Var、融合/非融合下的结果。

## 先复现、再修的缺陷

- **冷启动**：CANN 环境脚本在 `bash -u` 下因未设变量失败；Torch 预检把 CUDA 专用
  的严格数学 flag 交给 `ccec`；Nox 的 3.11 配置助手被 3.9 进程继承，编出 3.9 无法
  导入的 `cpython-311` 核心扩展。三处均已修：先初始化可选变量再 source CANN，导入前
  识别昇腾环境并去掉不兼容 flag，配置助手按 `JITTOR_CI_PYTHON` 解析。
- **ACL 算子语义**：真实设备 OpInfo 暴露 gather、permute、1-D matmul、负/花式
  getitem、index_select、embedding max-norm、searchsorted、affine-grid、NaN/Inf
  谓词、ceil-mode 池化与 Torch `where` 的缺失或错误路径，以及 dtype 桥依赖过期的
  NanoString 数值、缺 complex64。修法是把公开拼写路由到对应 ACL 实现、规范化负索引、
  使用稳定的 NanoString 常量。asin/acos 的 float32 误差用一条 `1e-4` 的定向容差
  覆盖，没有放宽其它算子。
- **惰性求值下的 kthvalue 与 max/min**：样本先建 float64 Var 再 cast，触发不支持的
  ACL float64 回退；`max(dim).values` 曾经由 argmax+gather 重建而在长序列中错位。
  现在样本按目标 dtype 进入，values 走 CANN ReduceMax/ReduceMin，argmax/argmin 只
  负责下标；不再需要 `JT_SYNC=1`。
- **布尔掩码赋值**：`x[mask] = scalar` 曾因用不支持的布尔归约计数而静默成为空操作；
  标量 masked-scatter 不再对掩码做归约，张量源保留精确长度检查。
- **跨设备卷积对照**：旧 `test_conv` 把惰性的随机图从 ACL scope 直接拿到 CPU scope
  复用，两侧消费的不是同一份输入。改为从固定 NumPy 快照分别建 Var 后，前向与两个
  梯度在 910B3 上通过；没有放宽容差，也没有给卷积加回退。

## Transformers 推理探针（同一台机器）

Transformers 4.56.2 + Torch 兼容层，CPU 反序列化 checkpoint 后显式迁到 NPU，
SDPA、KV cache、batch 1。这是推理正确性探针，不是吞吐基准。

| 证据 | Qwen3-8B float32 |
| --- | ---: |
| 参数量 | 8,190,735,360 |
| 进程设备显存（加载后） | 32,376 MB |
| 稳态 prefill / 单 token 生成 | 0.1144 s / 0.1282 s |
| 生成 token | ID 19（`4`） |
| ACL 融合注意力命中 / 未命中 | 216 / 0 |
| CPU 回退 / 生成期 CPU 编译算子 | 0 / 0 |

后续同机复验：Qwen3-0.6B 完整 logits 对原生 `torch_npu` SDPA 最大绝对误差
`3.80e-05`、argmax 与 top-20 集合一致；bfloat16 下 0.6B 与 8B 重复生成结果逐 token
一致，参数保持 bfloat16 且驻留设备；0.6B BF16 SDPA（prefill 用
FlashAttentionScoreV2、单 token decode 用 IncreFlashAttentionV4）3 次预热、10 次
采样为 0.53622 s，原生 `torch_npu` 为 0.52247 s。FP16 融合 SDPA 仍 fail-closed。

## 仍开放的边界

在当前树（`e3c369acb`）上核对，以下 skip 仍在、对应条目仍开放：

| skip | 原因 | 登记 |
| --- | --- | --- |
| 窄整数 `sum`/`max`/`min`；原生 bool `all_`/`any_` | 缺 ACL 原子重载 / 未在 NPU 验证的逻辑归约 | `KI-BACKEND-001` |
| float32 组合 `atan2` | 可触发 vector-core 异常终止进程 | `KI-BACKEND-002` |
| complex `irfft` | 600 s 内不返回 | `KI-BACKEND-003` |
| 两条原生 FlashAttention 前反向 | `jt.nn.FlashAttention` 不可用时跳过 | — |

这些 skip 不能改成同进程 xfail：`atan2` 会终止进程、`irfft` 会挂住。基线时另有一条
float32 `prod` skip，现在 `prod` 经 CANN `aclnnProd`/`aclnnProdDim` 执行，当前树已无
该 skip。本报告同时是 `KI-OPS-002`（整数 floor-divide，ROCm 待验）与
`KI-SEMANTICS-003`（浮点比较，NPU 其余 dtype 与 ROCm 待验）的 NPU 侧证据。

不在本报告范围：通用 ACL float64（CPU 回退不算 NPU 证据）、`arg_reduce` 反向、
BF16 训练、分布式 NPU 与其它下游项目，各自另有门禁。用户侧的安装、探针与排错见
[昇腾 910B 指南](../guides/ascend-910b.md)。
