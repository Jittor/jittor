# bf16 autocast 与 GradScaler 边界验证

- 状态：本报告范围内已修复；兼容层 fidelity 保留 `APPROXIMATE`。
- 日期：2026-10-14。
- 基线：`cded3102037edc3663b8f2054a6ebc06da6ebfd2`；修复提交由本文件所在的提交确定。
- 维护者：Torch 兼容层与 AMP 维护者。
- 范围：真实 CPU、NVIDIA RTX 4090 CUDA；未验证 ROCm、ACL/NPU。
- 复查条件：AMP 寄存器、节点标志布局、卷积 dispatch、GradScaler 状态机或参考 Torch 版本变化。

## 已确认的问题与修改

修复前，所有输入为 FP32 的 `torch.autocast(dtype=torch.bfloat16)` 矩阵乘法产生
FP16，输入 `1e5` 时得到 NaN。新增独立 bf16 preference bit，并把 Op 保存和恢复的
AMP 字段扩为七位，使退出 autocast 后执行的反向也保留该请求。保持旧 `prefer16`
规则，显式 bf16 请求同时适用于全 FP32 和混合 FP32/FP16 操作数。

CPU oneDNN 卷积入口只能返回 FP32，因此低精度区域使用现有可移植卷积实现。
CUDA 转置卷积的 bias 也转换到计算 dtype，避免 bf16 结果被 FP32 bias 提升。
实测矩阵乘法、linear、conv2d、conv_transpose2d 的输出均为 bf16，输入、权重和
bias 的梯度均在声明设备上执行。新增回归通过 `forbid_backend_fallbacks()` 禁止
后端回退，并在同步后检查输出与各梯度的实际驻留位置。

独立真实 PyTorch 2.11.0+cu128 的 CPU/CUDA 进程确认了以下 GradScaler 行为，
本实现与其对齐：

| 边界 | 对齐后的行为 |
| --- | --- |
| scale=0、输入梯度=0 | IEEE reciprocal 产生 NaN；不会抛 Python 除零异常；有限性检查读取除法前的梯度 |
| 输入梯度有限、除以小 scale 后溢出 | 本次 step 仍执行，不把新产生的 Inf 重新判为输入溢出 |
| scale=0.25、输入梯度=Inf | 跳过 step，scale 回退至 0.125；Torch 模式保留梯度供调用者检查 |
| CPU float64 梯度=1e300 | 保留 float64 有限性检查域，不因降为 float32 而误判 |
| CUDA float64 梯度=1e300 | 与 Torch CUDA 的 float 参数有限性检查一致，判为非有限并跳过；该差异已实际对拍 |
| growth 导致 float32 scale 溢出 | 保留原 scale，重置 growth tracker |
| 初始 scale 为 ±Inf、NaN | 保留 IEEE 数值语义；不进行额外的有限值限制 |
| 手动 update | 接受 0、低于 1、Inf、NaN 和不带梯度的同设备标量 tensor；拒绝向量形状 |
| 重复 unscale/step、未初始化、无梯度检查 | 按 Torch 的阶段和初始化约束报错；每个 optimizer 独立记录 |
| FP16 梯度 | unscale 前拒绝，保持梯度原值 |

原生与 Torch 模式共用数值算法；原生 optimizer 在跳过 step 后清理已消费梯度的
既有行为保留。Torch 模式由调用者负责 `zero_grad()`。设备状态路径也覆盖了
scale=0、回退低于 1、除法溢出和 growth 溢出，已有 CUDA capture/replay 回归通过。

## 验证方法与结果

源码与独立 Torch oracle 分进程运行。Python 3.11 私有 venv 使用 pytest 7.4.4、
pytest-timeout 2.3.1；Jittor 编译器为 GCC 11.5、CUDA 12.2.140。Torch oracle
确认 `_C` 为独立二进制扩展，并实际创建和计算 CPU/CUDA tensor。原环境缺失的
Python 依赖补到实验区私有 venv，未修改共享环境。各运行的 HOME、JITTOR_HOME、
TMPDIR 与原始日志位于 `$JITTOR_LAB_ROOT/_state/refactor-goal/amp/`，首次 JIT 串行。

定向选择如下，`JITTOR_TEST_DEVICES=cpu,cuda`；源码模式通过仓库 Torch bootstrap
激活兼容前端：

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q \
  compat/tests/torch/test_torch_amp_boundaries.py \
  compat/tests/torch/test_torch_amp_fidelity.py \
  compat/tests/torch/test_torch_amp_training_loop.py \
  compat/tests/torch/test_torch_compile_replay.py::TestCompileTrainingCuda::test_a_gradient_scaler_skips_overflows_on_the_device

JITTOR_TORCH_SHIM=0 PYTHONPATH=python python -m pytest -q \
  tests/nn/test_grad_scaler.py tests/runtime/test_amp_reg_bits.py \
  tests/core/test_node_flag_layout.py
```

- 兼容层：103 passed，0 skipped；包含新数值与 API 边界和真实 CUDA capture/replay。
- 原生层：29 passed、1 既有 xfailed，0 skipped。既有预期失败为 float16 loss 与
  Python float 相乘的弱标量提升问题（`KI-DTYPE-004`），不属于本次修复。
- 另以真实 CPU/CUDA 验证 1×1、1×2、2×1 矩阵在全 FP32 输入下返回 bf16。
- 节点布局和 AMP bit 定向检查通过；import layering 保持 165 个循环内模块，未增长。
- `bash tools/check_repo_layout.sh` 与 `git diff --check` 通过。
- 完整结构选择为 1427 passed、4 skipped、3 failed（177.06 秒）。一项为基线
  `DropoutACL` 梯度登记过期；另两项要求把此新增报告接入结果目录 toctree 并更新
  `MANIFEST.in`，由主任务集中维护这些共享文件后复验。两项 skip 是该私有 venv
  未安装 core distribution 的包解析检查，另两项为既有声明跳过；不用于声明完整
  结构门禁已通过。新增 AMP 与节点布局检查均通过。
- 主任务整合后统一运行 core 与相关回归；本子任务没有重复运行全量 full。

## 保留的精度与行为边界

### 主树整合复验（2026-10-15）

主树同步基线为 `ad47b58143f5c54e83114f923e23c0c69322130a`。AMP 与完整 RNG
恢复一起整合后，以独立 CPU/Torch 缓存串行首编译完成以下复验：

- `tools/run_test_suite.py --tier core --serial-compile`：322 passed、42 skipped、
  1 既有 xfailed；原生与 Torch 两个子进程均返回 0。
- 真实 CUDA 的 AMP、capture/replay、RNG 与 checkpoint 联合回归：127 passed，
  0 skipped；声明 CUDA 的计算在 CUDA 上执行。
- 原生 AMP、节点标志与新增 ACL 验收用例选择：29 passed、4 skipped、1 既有
  xfailed。四个 skip 均因没有 ACL 硬件，不能视为 NPU 梯度通过。
- 完整 `tests/structure`：1432 passed、6 skipped、0 failed。补齐报告索引与
  manifest，并让梯度清单识别 helper 内 CodeOp 反向后，之前的结构失败均已消除。
  两个 skip 为该源码解释器未安装 core distribution 的解析检查；其余为声明或
  加速后端能力跳过。
- 仓库布局与 `git diff --check` 通过。

未版本化日志位于 `$JITTOR_LAB_ROOT/_state/refactor-goal/distributed/logs/`，
分别为 `amp-rng-integration-core.log`、`amp-rng-integration-cuda.log`、
`amp-rng-integration-native.log` 与 `amp-rng-integration-structure.log`。
CUDA 预热最初使用已移除的 `setup_cublas` 接口失败；改用现有
`probe_library(..., load=True)` 后实际库加载、CUDA 计算及上述联合回归通过。

### 精度与行为限制

这里的 bf16 是真实 dtype 和指数范围，未改为 FP16 近似。计算仍不保证与 Torch
逐位相同：CPU 可移植卷积反向存在额外 bf16 中间舍入，示例权重梯度为 892928，
独立 Torch 为 897024，误差约 0.46%，回归使用 1% 容差；矩阵乘法和 linear
示例的输出、输入和权重梯度使用精确断言，CUDA cuDNN 示例也得到 897024。

Autocast 仍基于全局 AMP 寄存器，缺少 Torch 按算子的完整 allow/fallthrough
策略、各 device 独立上下文和权重转换缓存；例如部分 FP32 add 也可能被降低。
GradScaler 仍通过 optimizer bridge 与一次 host 可见归约实现常规检查，未实现
Torch 私有 `_check_inf_per_device` 与多设备梯度复制协议。此报告不把这些边界
升级为 `EXACT`，也不声明未覆盖的算子、shape 或设备已完成兼容。
