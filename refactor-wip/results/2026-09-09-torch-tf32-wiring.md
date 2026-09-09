# Torch tf32 控制项的归属与执行接线（KI-PRECISION-001 复核）

- Status: KI-PRECISION-001 撤销——前提已过期；契约注释、回归测试与生态口径已修正
- Date: 2026-09-09
- Baseline commit: 复现与归因在 `df6c221db`，结论在 `f2692a2eb` 上逐条复验；
  行为变更来自 `abd299137`
- Owner: Torch 兼容与 CUDA 精度维护者
- Review when: `_PRECISION_FIELDS` 的归属再次迁移、
  `compat/torch/frontend.py::_frontend_precision_policy` 改变它读的字段、
  或 native 弃用覆盖项（`cuda_allow_tf32` / `cuda_allow_cudnn_tf32`）语义变化

## 结论

KI-PRECISION-001 记的现象属实，它的结论不成立。

`torch.backends.cuda.matmul.allow_tf32 = True` 之后 `jt.flags.cuda_allow_tf32`
确实还是 0——但**这一条写入是生效的**：同一次进程里的 fp32 CUDA matmul 实际调用
`CUBLAS_COMPUTE_32F_FAST_TF32` / `CUBLAS_GEMM_DEFAULT_TENSOR_OP`，关掉它就回到
`CUBLAS_COMPUTE_32F` / `CUBLAS_GEMM_DEFAULT`。条目里"写入被接受、读回确认、matmul
仍用旧精度"的后半句是推断，不是实测，实测不成立。

真正过期的是**契约**：`abd299137`（7.19/7.20，已合并）把 Torch 前端的精度策略改成
前端自有的两档字段，与 native `jt.flags` 弃用覆盖项互不改写；它更新了架构文档和
`test_torch_backends_tf32.py`，却漏掉两处：

1. `installers/cuda/api.py` 里那张映射表的注释仍写着"每种拼写都是
   `flags.cuda_allow_tf32` 的 view"——**正是这段注释让本缺陷看起来像"已关闭缺陷的回归"**；
2. `test_torch_compat_cuda_tf32.py` 的两条测试仍断言旧归属，因此变红。

所以本次是"契约迁移漏改注释与测试"，不是第四条 polish 回归；但它与那三条同源：
`98c8ee94`（第217波，条目里的怀疑对象）**不是**成因。

## 环境

单卡 NVIDIA GeForce RTX 4090（`CUDA_VISIBLE_DEVICES=7`），CUDA 12.2.140，
nvcc 12.2.140，g++ 12.3.0，Python 3.11.15，sm_89。
解释器 `jt311`；`PYTHONPATH=compat/shim/resources:python`（不走 editable 安装）；
`JITTOR_TORCH_SHIM=1`。独立 `HOME`/`JITTOR_HOME`/`TMPDIR`/`XDG_CACHE_HOME` 与
`cache_name=tf32-wiring-run1` 位于 `$JITTOR_LAB_ROOT/_state/tf32-wiring/run1/`
（未版本化）。所有精度判据都取自各 op 在调用库之前打的 `vvv` 选择日志。

## 复现与实测

`jt.runtime.scope(use_cuda=1, float32_matmul_precision="highest",
use_tensorcore=0, cuda_allow_tf32=0, cuda_allow_cudnn_tf32=0)` 下，
六种拼写逐一读写，每次都跑一个真实 fp32 CUDA matmul / conv2d：

| 写入 | 六种拼写读回 | 实际执行 |
| --- | --- | --- |
| `cuda.matmul.allow_tf32 = True` | matmul 三种一致为 tf32 | `high` / `CUBLAS_COMPUTE_32F_FAST_TF32` / `..._TENSOR_OP` |
| `cuda.matmul.fp32_precision = "tf32"` | 同上 | 同上 |
| `set_float32_matmul_precision("high")` | 同上 | 同上 |
| 以上三种写 False/`"ieee"`/`"highest"` | matmul 三种一致为 ieee | `highest` / `CUBLAS_COMPUTE_32F` / `CUBLAS_GEMM_DEFAULT` |
| `set_float32_matmul_precision("medium")` | `get_...()` 读回 `medium` | `medium` / `CUBLAS_COMPUTE_32F_FAST_16BF` / `..._TENSOR_OP` |
| `cudnn.allow_tf32 = True` | cudnn 三种一致为 tf32 | `high` / `CUDNN_DATA_FLOAT` / `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` |
| `cudnn.conv.fp32_precision = "tf32"` | 同上 | 同上 |
| `cudnn.rnn.fp32_precision = "tf32"` | 同上 | 同上 |
| 以上三种写 False/`"ieee"` | cudnn 三种一致为 ieee | `highest` / `CUDNN_DATA_FLOAT` / `CUDNN_FMA_MATH` |

同一进程内：`jt.flags.cuda_allow_tf32` / `cuda_allow_cudnn_tf32` /
`float32_matmul_precision` 全程不动，native `jt.matmul` 全程 `highest`；
反向也成立——`jt.runtime.scope(cuda_allow_tf32=1, cuda_allow_cudnn_tf32=1)`
不能把一条显式 Torch op 抬到 tf32。这两条隔离都是 7.19/7.20 明写的契约。

`jt.flags.cuda_allow_tf32 = 1` 让 native 侧两个读数都变 1（条目里"标志通路是活的"
属实），但它只抬 native Runtime-following 调用的档位，与 Torch 前端策略无关。

## 归因

- 代码考古：`9aaedba9`（7.08）建立映射表，键值是 `jt.flags` 名；
  `abd299137`（7.19/7.20）把 `_TF32_FLAGS` 改名 `_PRECISION_FIELDS`，值改成
  `CudaRuntimeState.matmul_precision` / `cudnn_precision`，
  `_tf32_set` 不再写 `jt.flags`。表上方那段注释**一字未改**。
- 反向补丁定位（不重编核心）：
  `git show abd299137 -- compat/torch/installers/cuda/api.py | git apply -R`
  之后 `test_torch_compat_cuda_tf32.py` **2 passed**；还原后回到 2 failed。
  该 hunk 就是这两条失败的唯一成因。
- `98c8ee94`（条目里的怀疑对象）与 `271ad81d3`、`94fd022cf` 都不触及这条通路。

## 改了什么

1. `compat/torch/installers/cuda/api.py`：只改注释。映射表改写成"六种拼写 ↔
   `CudaRuntimeState` 的两个 tier 字段"，并写明这两个字段是前端自有策略、
   由 `frontend.py` 带进 op 构造、与弃用的 native 覆盖项双向隔离，以及
   **"写入被接受且能读回"不构成生效证据，能钉住的是库调用日志里的 compute type**。
2. `compat/tests/torch/test_torch_compat_cuda_tf32.py`：两条按旧归属断言 flag 的
   测试改写成五条按**执行**断言的测试（见上表），覆盖六种拼写、三档 matmul tier
   与两个方向的隔离。
3. `compat/tests/torch/_ecosystem_runner.py` / `_ecosystem_harness.py`：
   tf32 报告增加 `matmul_precision` 档位字符串，并写明这两侧比较的语义。
4. `agent/manuals/known-issues.md`：删除 KI-PRECISION-001，头部 Baseline 改回
   `715009c02`。

顺带修掉的：旧文件的两条测试在断言失败处中断后不恢复前端 tier，把 tf32 泄漏给了
同一进程里后跑的文件（见下面的基线对照）。新文件用 `addCleanup` 恢复。

`compat/torch/` 下**没有行为改动**：本条不是代码缺陷。

## 回退验证（三个反例都变红）

按 `agent/skills/structure-rule-has-teeth`，改真实文件、跑真实 nodeid、立即还原：

| 反例 | 新测试文件 | 旧读回式测试 |
| --- | --- | --- |
| `_tf32_set` 丢弃写入（原本要抓的东西） | **3 failed / 2 passed** | — |
| `frontend.py::_frontend_precision_policy` 恒返回 `(0, 1)`——存下来但没人执行，即条目描述的那种缺陷形状 | **5 failed** | `test_torch_backends_tf32.py` **13 passed** |
| `_tf32_set` 顺带写 `jt.flags`（= `abd299137` 之前的行为，也就是条目要求的"修复"） | **1 failed**（隔离那条） | — |

第二行是关键：**旧的读回式断言对这种缺陷完全无感**，新的执行式断言全红。

## 基线对照

同机同缓存，Torch shim + 真实 CUDA，逐条 A/B（`-p no:randomly`）。任务进行中
共享 checkout 的 HEAD 由 `df6c221db` 前进到 `f2692a2eb`（他人提交），下表两侧
都在 `f2692a2eb` 上重跑过，数字与 `df6c221db` 上一致：

| 口径 | 改前（`df6c221db`） | 改后 |
| --- | --- | --- |
| `test_torch_compat_cuda_tf32.py` | 2 failed | 5 passed |
| `test_torch_backends_tf32.py` | 13 passed | 13 passed |
| 精度与生态定向集合（7 个文件） | **3 failed / 38 passed / 42 skipped** | **0 failed / 44 passed / 42 skipped** |

定向集合 = `test_torch_compat_cuda_tf32` + `test_torch_backends_tf32` +
`test_precision_policy_isolation` + `test_basic_api_precision_memory` +
`test_ecosystem_parity` + `test_ecosystem_device_selection` +
`test_ecosystem_speed`。42 skipped 全部是生态对拍（`real_torch_python is not
configured`），因此 harness 的改动在本机没有可执行门禁覆盖——它只是给两侧对称
地多报一个字段。

改前那 3 条里的第三条不是一条独立的既存缺陷，而是**旧测试文件的状态泄漏**：
`test_basic_api_precision_memory::TestBasicPrecision::test_matmul_and_bmm_match_numpy`
单独跑 11 passed，把旧的 `test_torch_compat_cuda_tf32.py` 排在它前面就 3 failed。
旧文件那两条测试在断言失败处中断，`finally` 只恢复了 native scope，**没有恢复前端
tier**，于是把 matmul 留在 tf32 上泄漏给后面的文件——那条 k=48 的 fp32 matmul
因此对 NumPy 差到 1.3e-4 相对误差（float32 该有的量级是 3.5e-7，实测同一用例在
干净进程里就是 3.5e-7）。新文件在 `addCleanup` 里恢复前端 tier，这条连带失败随之
转绿。

`compat/torch/installers/cuda/api.py` 的 diff 逐行核对过：**改动全部是 `#:` 注释**，
没有一行可执行代码变化。

## 生态对拍 harness 的那条断言

`_ecosystem_harness.py` 断言两侧 `tf32` 报告相同，值来自
`_ecosystem_runner.py::_configure_tf32`，它读回的是**各自前端**的开关，
Jittor 侧就是兼容层存下来的两个 tier。

结论：**这条断言有效，不需要改成读 native flag**。因为 Jittor 侧存下来的 tier
正是它自己每个 op 构造时捕获、执行时恢复的策略（本报告上表即证据）；改成读
`jt.flags.cuda_allow_tf32` 反而会读到一个与 Torch op 执行无关的值，把一条有效
断言变成假的。

有一个**真实的弱点**已一并收紧：原报告只比较布尔，而 torch 的"tf32 开着"是三档，
两侧可以布尔相同而一侧在 bf16 累加。现在同时报告并比较
`torch.get_float32_matmul_precision()` 的档位字符串。

仍然存在、本次未处理的暴露面：harness 只覆盖**走 Torch 前端**的算子。若某个
adapter 直接调 `jt.nn.*`，那条调用跟随 native 策略（默认 `highest`），而真实
torch 侧默认允许 cuDNN tf32——差异方向是更精确、更慢，不是错误结果，但对拍的
性能比较会因此偏保守。

## 边界

- 只在单卡 RTX 4090 / sm_89 / CUDA 12.2 上验证；未做 NPU、ROCm、多卡与全量门禁声明。
- `tests/structure`（Torch shim）本次为 **71 failed / 1203 passed / 2 skipped**，
  但这个数字**不可归因**：同一 checkout 上他人的文档迁移与新增测试同时在进行，
  失败集中在 ACL 注册、error categories 以及仍指向已迁走的 `docs/results/`
  的看板契约。逐条核对过：没有一条失败提到本次改动的文件。
- 判据是库选择日志，不是数值差异：允许 tensor-op 只是让相应 engine 可选，
  cuDNN 仍可能自行挑 FMA，数值判据在这里不成立（见
  [float32 精度策略](../../docs/notes/float32-precision-policy.md)）。
- 未改任何执行路径，因此不重复 7.19/7.20 的数值验收，那批结论见
  [前端精度隔离](2026-09-08-frontend-precision-isolation.md)。
