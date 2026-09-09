# 昇腾 910B：配置与验证

本指南把 Jittor 2.0 的源码 checkout 配置到昇腾 910B 系列设备上，并验证运算**确实
经由 ACL 执行**。昇腾驱动、固件和 CANN 本身的安装不在 Jittor 范围内——请从一台已装好
厂商运行时与编译器的主机开始。

## 已验证的基线

维护中的门禁在以下组合上验证通过：

| 组件 | 验证版本 |
| --- | --- |
| 设备 | 昇腾 910B3 |
| 架构 | Linux aarch64 |
| `npu-smi` 报告的驱动 | 25.5.1 |
| CANN toolkit 与 `ccec` | 9.0.0 |
| Python | 3.9.25 |
| NumPy | 1.26.4 |
| pytest / pytest-timeout | 7.4.4 / 2.3.1 |

这些版本是**复现过的基线**，不是"其它 CANN 版本都不兼容"的断言。Jittor 声明支持
Python 3.7 到 3.13；对于厂商包可能有更窄 Python 约束的昇腾环境，3.9 到 3.11 是实际
可用区间。

## 检查设备与工具链

导入 Jittor 之前先确认驱动看得到一块健康的设备：

```bash
npu-smi info
```

找到 CANN 的环境脚本并在当前 shell 里加载。**把位置做成可配置的**，不要把主机专属
安装路径写死：

```bash
export ASCEND_HOME=/path/to/Ascend/cann-9.0.0
export CANN_SET_ENV="$ASCEND_HOME/set_env.sh"

# 某些 CANN 的 set_env.sh 在 `set -u` 下要求这两个变量已存在。
: "${LD_LIBRARY_PATH:=}"
: "${CMAKE_PREFIX_PATH:=}"
source "$CANN_SET_ENV"

ccec --version
```

`npu-smi info` **必须在与 Python 相同的执行环境里**成功。容器中请按厂商容器运行时的
说明把所需的昇腾设备和驱动库透传进去。

## 安装源码 checkout

建一个专用 Python 环境，以 editable 方式安装。NPU 验证门禁还需要 SciPy 与钉住版本的
pytest 工具：

```bash
python -m pip install -e .
python -m pip install \
  "numpy==1.26.4" \
  "scipy==1.13.1" \
  "pytest==7.4.4" \
  "pytest-timeout==2.3.1"
```

从仓库根目录运行。不做 editable 安装时，把 checkout 的 Python 包放在最前面：

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

## 隔离 JIT 状态

Jittor 在首次使用时编译核心与算子。把可变状态放到 checkout 之外，并给每个并发运行
**不同的 `JITTOR_HOME` 或 `cache_name`**：

```bash
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
run_root="$JITTOR_LAB_ROOT/_state/ascend-910b/manual"
mkdir -p "$run_root"/{home,jittor-home,tmp,xdg-cache}

export HOME="$run_root/home"
export JITTOR_HOME="$run_root/jittor-home"
export TMPDIR="$run_root/tmp"
export XDG_CACHE_HOME="$run_root/xdg-cache"
export cache_name=ascend_910b_manual
```

**首次 JIT 或扩展编译要串行执行。测试与基准不得共享编译缓存。** 更换 CANN、驱动、
宿主编译器或 ACL 源码之后要换一个新的 cache name——**不要用陈旧的二进制去诊断新的栈。**

## 选择设备

导入 Jittor 之前把进程限制到分配给你的设备上：

```bash
export ASCEND_RT_VISIBLE_DEVICES=<分配到的设备>
```

可见设备在进程内会被重新编号。**不要选一块已经被别的负载占用的设备。**

## 跑一次真实的 ACL 探测

**只验证 import 成功不能证明 NPU 支持。** 下面这个探测要求 ACL 存在，打开加速器标志，
做 float32 矩阵乘，并检查独立参考结果、设备驻留和原生回退计数器：

```python
import numpy as np
import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks

assert getattr(jt.compiler, "has_acl", 0), "ACL was not detected"
jt.flags.use_acl = 1
jt.flags.use_cuda = 1

a_np = np.arange(12, dtype=np.float32).reshape(3, 4)
b_np = np.arange(20, dtype=np.float32).reshape(4, 5)

fallback_before = jt.core.backend_fallback_count()
with forbid_backend_fallbacks():
    result = jt.matmul(jt.array(a_np), jt.array(b_np))
    jt.sync_all(True)
    assert result.location() == "device"
    actual = result.numpy()

np.testing.assert_allclose(actual, a_np @ b_np, rtol=1e-5, atol=1e-5)
assert jt.core.backend_fallback_count() == fallback_before
print("ACL matmul passed")
```

把探测保存为 checkout 之外的 `probe_acl.py`，并且**只在 source 过 CANN 之后**运行：

```bash
backend_fallback=error python probe_acl.py
```

即使选中的后端是 ACL，Jittor 也用 `use_cuda` 作为通用的加速器执行标志。因此
**`has_acl` 才是区分昇腾与 CUDA 的判据。**

### 运行时回退策略

`jt.runtime.backend_fallback` 接受 `error`、`warn`、`allow`。**每个验证进程启动前都在
环境里设 `backend_fallback=error`。** `warn` 和 `allow` 是显式的调试策略，
**不是 NPU 的验收模式**。只有预检阶段判定不支持时才可以请求回退；launcher、分配、
编译或执行的异常必须在清理后向上传播，**绝不能改成 CPU 执行重试**。

`jt.core.backend_fallback_count()` 统计回退**尝试**次数，包括被 `error` 拒绝掉的请求。
在同一个进程内比较工作前后的值。`forbid_backend_fallbacks()` 临时选中 `error`，并且
在正常退出时**拒绝计数器有非零增量**——即使内层调用方捕获了那次拒绝也一样。函数体
抛出的异常原样向上传播。

**实体化（`numpy()`/fetch）或 `jt.sync_all(True)` 必须发生在作用域内**：它不会自动
同步。在作用域内创建、离开后才执行的惰性张量**不在覆盖范围内**。

定向测试节点必须把该作用域套在运算和同步外面，并断言 NPU 执行/驻留以及独立的数值或
梯度。**收集成功、skip 或"日志里没有某条消息"都不是这种证据。** SDK 与 launcher 日志
只用于诊断——不要靠 grep CPU 编译或回退消息来判断验证是否通过。设备计算之外的
CPU 参考/checkpoint 工作本身不算后端回退。

`tests/backends/acl/conftest.py` 会在 fixture 进入时 Jittor 已加载的情况下自动为每个
测试装上这道守卫。它保留原始的 setup/call 异常，而不是用计数器错误替换掉。测试仍然
必须在返回前同步或取值——**fixture 不会冲刷待定工作**。`tests/backends/acl/` 之外的
独立脚本和测试不继承该 fixture，必须显式进入 `forbid_backend_fallbacks()`。如果
Jittor 是在 fixture 进入之后才首次导入，该测试也需要显式作用域。

## 逐算子同步检查

在昇腾 910B3 上，先 source CANN 环境、确认设备健康、选好分配到的设备，再运行：

```bash
source "$CANN_SET_ENV"
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=<分配到的设备>

set -o pipefail
backend_fallback=error sync_run=1 python -m pytest -q -s \
  tests/backends/acl/test_acl.py::TestACL::test_float32_matmul_runs_on_acl \
  2>&1 | tee "$TMPDIR/acl-sync-run.log"
```

`sync_run=1` 让每个 `BaseOpRunner` 在发射后立刻等待 `aclstream`。同步失败时 Jittor 会
抛出包含算子名、数值返回码和解码后 ACL 错误的异常。保留完整日志，并提取归因行：

```bash
rg "aclrtSynchronizeStream failed" "$TMPDIR/acl-sync-run.log"
```

**只有当 ACL 执行断言通过、且 `forbid_backend_fallbacks()` 下
`backend_fallback_count()` 增量为零时，这次运行才有效。** 测试必须在作用域内同步；
日志仅供诊断。`execute launcher failed` 是失败归因的证据，**不能替代运行时计数器检查**。

诊断完成后，用关闭逐算子同步的方式重跑同一个定向节点，确认常规异步路径仍然发射在
ACL 上：

```bash
set -o pipefail
backend_fallback=error sync_run=0 python -m pytest -q -s \
  tests/backends/acl/test_acl.py::TestACL::test_float32_matmul_runs_on_acl \
  2>&1 | tee "$TMPDIR/acl-async-run.log"
```

`JT_SYNC=1` 是另一个作用于整个执行器的编译期诊断开关。它**不能替代**上面
`BaseOpRunner` 的 `sync_run=1` 检查，而且切换它可能重建 JIT 缓存。

## 验证 ACL 张量与 workspace 归属

ACL runner 显式创建张量描述符，并从 Jittor 的临时分配器取得共享的 aclnn workspace。
在 910B3 上 source CANN、检查设备、记录显存，然后运行一个通过若干次正常矩阵乘把
workspace 撑大的进程：

```bash
source "$CANN_SET_ENV"
export ASCEND_RT_VISIBLE_DEVICES=<分配到的设备>
npu-smi info | tee "$TMPDIR/before-workspace.txt"

set -o pipefail
backend_fallback=error sync_run=1 python - <<'PY' 2>&1 | tee "$TMPDIR/workspace-normal.log"
import numpy as np
import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks

assert getattr(jt.compiler, "has_acl", 0), "ACL was not detected"
jt.flags.use_acl = 1
jt.flags.use_cuda = 1
fallback_before = jt.core.backend_fallback_count()
with forbid_backend_fallbacks():
    for width in (64, 128, 256):
        value = np.arange(width * width, dtype=np.float32).reshape(width, width)
        result = jt.matmul(jt.array(value), jt.array(value))
        jt.sync_all(True)
        assert result.location() == "device"
        actual = result.numpy()
        np.testing.assert_allclose(actual, value @ value, rtol=2e-4, atol=2e-2)
assert jt.core.backend_fallback_count() == fallback_before
print("ACL workspace normal path passed")
PY

npu-smi info | tee "$TMPDIR/after-workspace.txt"
```

**只有数值正确、设备驻留、且原生回退尝试增量为零时**，这次正常运行才被接受。Python
进程退出后它必须不再出现在 `npu-smi` 里；对比 `before-workspace.txt` 与
`after-workspace.txt` 确认 workspace 被释放，而不是被某个孤儿进程占着。

**不要在共享 NPU 上人为制造无界分配。** 当已有负载自然复现出 workspace 失败时，保留
它的日志并提取归因：

```bash
rg "ACL workspace allocation failed" "$TMPDIR/workspace-failure.log"
```

错误必须报告 `workspace requested bytes`、`workspace allocator` 以及底层的分配失败。
之后在新进程里的小探测必须仍然能通过且无 CPU 回退——否则说明那次失败的分配没有把
全局 workspace 留在可重试的空状态。

三类 runner 失败现在会指明阶段与算子，而不是带着无效的 executor 继续、或误查外层
融合图：

- `aclnn workspace-size query failed` —— 含算子名、返回码、解码后的 ACL 状态和 CANN
  最近的错误文本；
- `ACL operator has no registered launcher` —— runner 名字不在 ACL 函数表里
  （分组与非分组 runner 同理）；
- `current fused operator input is not allocated` —— 指出输入不变量失败的那个队列项，
  而不是外层的融合运算。

在 910B3 上，请把这些行连同周边的 SDK/launcher 日志和原生回退尝试增量一起保留。
**上面的正常 matmul 与 workspace 命令中不应出现任何这类失败诊断**；注入或自然复现的
失败必须在 execute 调用用上无效 executor **之前**就让该 ACL runner 停下。

## 运行维护中的 NPU 门禁

NPU 的 nox 会话会建立隔离状态、检查 `npu-smi`、跑一次真实 ACL matmul 探测，然后执行
维护中的后端与 OpInfo 测试。nox 自身用 Python 3.11，而 `JITTOR_CI_PYTHON` 指向预置的
昇腾 Python 环境：

```bash
python -m pip install -r requirements/dev-tools.txt
export CANN_SET_ENV=/path/to/Ascend/cann-9.0.0/set_env.sh
export JITTOR_CI_PYTHON=/path/to/ascend-python/bin/python
export ASCEND_RT_VISIBLE_DEVICES=<分配到的设备>
backend_fallback=error python -m nox -s npu
```

该会话从 `JITTOR_CI_PYTHON` 解析 `python_config_path`，而不是从 nox 自己的解释器解析。
若直接启动时设置了 `python_config_path`，它必须指向**同一个 Python 版本**的 config
助手——版本不匹配的助手会产生硬件 Python 无法导入的扩展后缀。

直接运行同一批核心测试：

```bash
export JITTOR_TEST_DEVICES=npu
export backend_fallback=error
"$JITTOR_CI_PYTHON" -m pytest -v --timeout=600 \
  tests/backends/acl/test_acl.py \
  tests/backends/acl/test_aclop.py \
  tests/backends/acl/test_acl_indexing.py \
  tests/ops/test_ops.py
```

**不要把通过的 CPU 回退当成 NPU 覆盖。** 新增或修复一个运算时，要补一条捕获 ACL 执行、
或以其它方式证明所声明设备完成了计算的定向断言。

## 用 Transformers 跑 Qwen3-8B

维护中的手动探测让本地 Qwen3 checkpoint 经由 Jittor Torch shim 与 Transformers 4.56.2
运行。安装可选的模型依赖，并把 checkpoint 放在源码树之外：

```bash
python -m pip install "transformers==4.56.2" "jinja2==3.1.6"
export QWEN3_MODEL=/path/to/Qwen3-8B
export JITTOR_TORCH_SHIM=1

backend_fallback=error python tests/backends/acl/manual/run_qwen3_transformers.py \
  --model "$QWEN3_MODEL" \
  --dtype bfloat16 \
  --max-new-tokens 8 \
  --runs 3
```

**只在完成上面的 CANN、设备选择和缓存隔离配置之后**运行该命令。探测会在 CPU 上加载
权重、显式把模型迁移到可见 NPU、在模型驻留时打印 `npu-smi`、在
`forbid_backend_fallbacks()` 下用 KV cache 做贪心 eager 注意力生成，并拒绝生成期间的
任何原生回退尝试。**CPU 上的 checkpoint 反序列化是预期行为，不作为模型前向的证据。**

已验证的 Qwen3-8B checkpoint 有 8,190,735,360 个参数。float32 在一块 64 GB 910B3 上
占用 32,376 MB 显存。float32 与 bfloat16 都报告参数驻留在加速器上且
`has_acl=use_acl=use_cuda=1`。**历史上的零回退日志报告必须在当前分支上用原生计数器和
作用域重新验证。** 维护中的 bfloat16 八 token 请求在 `[19, 13, 151645]` 处停止，每次
重复运行都解码为 `4.`。用 `--dtype float32` 可跑最初的单 token 探测。
**这些是正确性探测，不是吞吐基准。**

## 当前限制

维护中的 910B 门禁**刻意跳过**下列已复现的缺口，而不是让它们中止或卡住进程：

- 低于 32 位的整数 `sum`、`max`、`min`，以及布尔 `all`、`any` 缺少完整的 ACL 归约
  路径——可能的话把输入提升到受支持的位宽；
- 组合出来的 float32 `atan2` 可能触发向量核异常；
- 复数 `irfft` 可能卡死；
- 原生 FlashAttention 测试需要可选的 `jt.nn.FlashAttention` 实现，缺失时跳过；
- Qwen3 bfloat16 已验证的是 eager、无梯度的贪心推理。融合 ACL SDPA 另在 Qwen3-0.6B 上
  以 FlashAttentionScoreV2 prefill 和 IncreFlashAttentionV4 decode 验证。Qwen3-0.6B
  float32 的 eager 前向、因果 LM loss 与反向也在零回退下验证通过；当前优化后的 RoPE
  结果需要一个外部 Transformers 模块补丁把 Qwen3 路由到 `jt.nn.rotary_emb`。优化器
  更新、BF16 训练、Qwen3-8B 的 BF16 SDPA 与训练、采样、量化以及其它模型家族仍是**独立
  的能力门禁**；
- ACL 不提供通用的 float64 算子覆盖，因此 **float64 回退不被接受为 NPU 运算的证据**。

float16/float32 的 `arg_reduce` 前向与值输出反向是维护中的 ACL 能力：前向用 CANN
MaxDim/MinDim，反向把上游梯度散射到选中的第一个下标；当前的真实设备验证必须通过
原生策略与计数器拒绝回退尝试。

完整、单轴和多轴的 `prod` 使用 CANN `aclnnProd`/`aclnnProdDim`，多轴归约被降低为有序的
单轴设备归约。float32 的前向/反向以及 uint8/int8/int16/int32/int64 的前向在真实 NPU 上
与独立 NumPy 参考吻合。重新验证需要设备驻留，且守卫内的计算回退尝试为零。

可执行证据与退出条件见
[活跃问题总账](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md)。

## 排错

**ACL 未被检测到**：在同一个 shell 里核对 `CANN_SET_ENV`、`ccec --version`、Python
架构和 CANN 库路径。**不要捕获并忽略 CANN 的注册或编译错误**——不完整的后端配置必须
显式失败。

**异步设备错误报告在导致它的那行 Python 之后**：只用最小复现重跑，并加上

```bash
export JT_SYNC=1
export trace_py_var=3
```

这些开关是诊断用途且会拖慢执行，**定位到出错运算后就去掉**。通用的 JIT 与显存诊断见
[调试指南](debugging.md)。

各 ACL 算子族迁移到共享 launcher 的逐条状态属于整改期记录，见
[昇腾迁移记录](../../refactor-wip/architecture/ascend-migration-notes.md)。
