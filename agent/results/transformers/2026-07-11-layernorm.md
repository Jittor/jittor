# LayerNorm CUDA / torch_compat 性能审计

> 状态：✅ 静态深挖完成；🟡 CUDA profiler 与实现验证待主任务统一占卡执行
> 日期：2026-07-10
> 环境：`conda jt311`；本轮按约束未运行 GPU kernel，所有新缓存均位于
> `${JITTOR_LAB_ROOT}/jittor_transformers_perf/runtime/`。

## 1. 结论摘要

1. corrected benchmark 中 `layernorm_16x128x768` 的 `2.30-2.42x` **确实命中**了
   现有 no-grad CUDA fast path，不是通用组合式 LayerNorm 的结果。命中证据包括：
   benchmark 在 `torch.no_grad()` 内调用 `F.layer_norm(x, (768,), w, b, 1e-5)`；所有参数均为
   `jt.Var`、输入为 fp32、归一化维只有最后一维；同一 runtime 中在结果写入前已生成对应
   `CodeOp` 的 `.so`，源码正是 `_layer_norm_no_grad_cuda` 的单 kernel 实现。
2. 该结果不能简单归因为“kernel launch 太多”：no-grad fast path 本身只有 **1 次 CUDA launch**。
   差距由两部分共同构成：
   - **host 构图**：fast path 每次仍在 Python 侧做 shape 遍历、两个 reshape、CUDA 源码字符串
     构造和 `CodeOp` 建图。纯构图、不 sync/不读数据的探针中，热态中位数约
     **11.89 us/call**（20 组，每组 10 次，范围 11.65-18.98 us）。corrected benchmark 的
     Jittor 总时延为 36.70 us，故 host 构图不是可忽略项。该数值不能直接从总时延相减，
     但足以说明只改 CUDA 算术未必能把 wall time 拉到 torch 水平。
   - **kernel 质量**：当前 kernel 固定 128 threads，全部 scalar load/store，均值、中心化方差、
     输出阶段共读取输入 3 遍；归约使用 shared-memory tree，没有 warp shuffle。缓存 `.so` 的
     SASS 为 20 registers、512 B shared memory、4 个 `BAR.SYNC`，未见 `SHFL` 或 vector
     load/store。torch 2.12.1 对这个 N=768 对齐形状走 `vec_size=4` 的 vectorized Welford
     单 kernel，128 threads（32x4 warps），输入只在统计和输出阶段读取 2 遍。
3. **训练路径完全不命中该 fast path**。它走 `_ln_normalize` 的稳定 `jt.Function`，前向为多个
   reduce/elementwise 边界，反向还要分别计算 `dx/dweight/dbias`。当前实现保存完整 `xhat`
   和每行 `rstd`；相比保存 `x + mean + rstd` 的 fused 方案，每层额外常驻一个与输入同大的
   `xhat`。对本 benchmark 形状，单个 fp32 `xhat` 约 6 MiB。
4. 训练优化不能退回 `E[x^2]-E[x]^2` 或依赖巨大项抵消。现有一阶反向修复的核心约束是：

   ```text
   dx = rstd * (g - mean(g) - xhat * mean(g*xhat))
   ```

   小方差与极小 eps 必须继续用随机投影上游梯度验证。当前 LayerNorm 已明确不支持二阶导，
   因而 CUDA fused `jt.Function` 只保持一阶导并不会新增二阶导回归，但必须保留现状的
   “不支持”语义，不能静默给零二阶导。
5. 安全扩展专用 CUDA fused 路径是可行的。优先顺序应是：先补 profiler 与 ACL dispatch
   guard，再优化 no-grad kernel，最后做训练 forward/backward fused Function。CPU 保留组合
   原语 fallback，NPU 保留现有 aclnn LayerNorm/LayerNormBackward，不引入新依赖。

## 2. torch_compat 到 LayerNorm 的实际调用链

### 2.1 Functional 路径

`python/jittor/torch_compat.py:1826` 直接执行：

```python
F.layer_norm = nn.layer_norm
```

因此 `import jittor as torch` 后的 `torch.nn.functional.layer_norm` 没有额外 tensor copy、同步或
Python 数学 fallback；它直接进入 `python/jittor/nn.py:1351` 的 `@fp32_guard` 包装函数。

### 2.2 Module 路径

`torch.nn.LayerNorm` 直接引用 `jittor.nn.LayerNorm`。`execute()` 在
`python/jittor/nn.py:1342` 先尝试 `_layer_norm_no_grad_cuda`，未命中再进入 `_ln_normalize`。

`torch.no_grad()` 与 `torch.inference_mode()` 最终都把 `jt.flags.no_grad` 置 1，所以二者均能
触发 fast path。仅调用 `model.eval()` 不会关闭 autograd，因此 **eval-only 不命中**；即使输入和
参数都已经 `stop_grad`，只要全局 `no_grad` flag 没开，也不会命中。

### 2.3 现有 fast path 命中矩阵

| 条件 | 命中情况 | 说明 |
|---|---:|---|
| CUDA + `no_grad/inference_mode` | 必须 | `eval()` 本身不够 |
| `x.dtype == float32` | 必须 | 原生 fp16/bf16 直接 miss；AMP 可能先整体 cast 到 fp32，再增加输出 cast |
| `len(normalized_shape) == 1` | 必须 | 多维 normalized shape 即使内存连续也 miss |
| `x.shape[-1] == normalized_shape[0]` | 必须 | benchmark 的 768 满足 |
| weight 与 bias 都是 Var | 命中 | dtype 可混合，kernel 内显式 cast affine 到 float |
| weight 与 bias 都是 Python scalar | 命中 | 可由 `JITTOR_LAYERNORM_SCALAR_FAST=0` 关闭 |
| weight 为 Var、bias 为 scalar/None | miss | `LayerNorm(..., bias=False)` 的常见路径 |
| weight/bias 只有一个是 Var | miss | functional 的独立可选 affine 组合也 miss |
| normalized shape 为多维 | miss | 可安全 flatten trailing normalized dimensions 后扩展 |
| NPU/ACL + no-grad functional | **有风险** | ACL 中 `use_cuda==1`，当前 helper 没排除 `use_acl`，可能误选 ACL 不支持的 `jt.code` |

最后一项是现有 dispatch 风险：ACL compiler 只 monkey-patch 了 `nn.LayerNorm.execute` 到原生
`LayerNormACL`，没有同步替换已绑定的 functional `F.layer_norm`。因此新增或重构 CUDA 路径时，
首个条件必须显式包含 `not jt.flags.use_acl`（或等价的真实 CUDA backend 判断）。

## 3. 为什么 corrected no-grad benchmark 仍慢约 2.3x

### 3.1 benchmark 是可信的，但测的是 wall time

对应数据：

| backend | fp32 latency |
|---|---:|
| torch | 0.015137 ms |
| Jittor | 0.036700 ms |
| ratio | 2.424x |

benchmark 使用不同的预分配 input slot，并保留计时区全部输出，因此没有被 Jittor lazy CSE
错误地只执行最后一个输出。warmup 后同步，首编译也不在计时区内。

但计时使用 `perf_counter()` 包围“Python 调用 + lazy 建图 + 最终 sync”，不是 CUDA event 的纯
kernel 时间。Jittor 会先构完 10 个 `CodeOp`，随后在 sync 才启动 GPU；host 建图与 device 执行
基本串行。PyTorch eager 的 host dispatch/launch 则能与前一 kernel 的执行部分重叠。因此
36.70 us 既包含 kernel 差距，也包含 runtime 调度模型差距。

本轮 host-only 构图探针（未 sync、未读取 Var、没有运行 device kernel）：

| 路径 | 热态 host 构图中位数 |
|---|---:|
| no-grad CUDA fast path | 11.89 us/call |
| 通用 `_ln_normalize` forward | 62.79 us/call |

通用路径更高是因为每次会在 `_ln_normalize` 内定义一个局部 `jt.Function` class，并创建多个
reduce/broadcast/binary/unary/tape 节点。绝对值会受 Python allocator 和存活图规模影响，应用于
性能结论时只能看量级，不能当作 GPU latency 直接相减。

### 3.2 当前 Jittor 单 kernel 的结构

对 `(16,128,768)`：`M=2048` rows、`N=768` hidden，launch 为：

```text
grid = 2048 blocks
block = 128 threads
shared = float buf[128]
```

每行执行：

1. scalar 读取 X，求 sum；
2. shared-memory tree reduce，得到 mean；
3. 再次 scalar 读取 X，求中心化平方和；
4. 再次 shared-memory tree reduce，得到 variance/rstd；
5. 第三次 scalar 读取 X，同时读取 gamma/beta，写 Y。

源码每个归约写了 1 次初始 `__syncthreads()` 加 7 层 tree barrier；nvcc 对 sm89 优化后，缓存
二进制中总共仍有 4 个硬件 `BAR.SYNC`。更重要的是它没有 warp shuffle，也没有 `float4`/128-bit
load-store；每个线程为每个阶段循环处理 6 个 scalar。

按全局访存指令的名义数据量（不扣 L1/L2 affine cache）估算：

```text
X 三遍读取        18 MiB
Y 一遍写入         6 MiB
gamma/beta 读取   12 MiB（实际大部分会命中 cache）
```

### 3.3 同版本 PyTorch CUDA 路径

本机真 torch 为 `2.12.1+cu130`，git revision
`7269437d655783a26cba32aa88195b741ff496aa`。对应官方
[`layer_norm_kernel.cu`](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/native/cuda/layer_norm_kernel.cu)
的 aligned fast path具备：

- `vec_size=4`，X/Y/gamma/beta 全部满足 16-byte alignment 且 N%4==0 时启用；
- 128 threads，组织为 `(warp_size=32, 4 warps)`；
- Welford 在线统计，warp 内用 shuffle、warp 间只交换少量 shared 数据；
- 统计阶段和输出阶段各读取一次 X，共 2 遍；
- 一个 kernel 同时输出 Y、mean、rstd；
- float/half/bfloat16 均以合适 accumulation type 计算。

N=768 可整除 4，普通 CUDA allocator 返回的 input/output/affine buffer 也满足 alignment，故该
benchmark 形状会走 vectorized path。PyTorch 不是靠“少一次 launch”胜出，而是靠更少的 scalar
指令、更少一遍 X 读取和更高效的 warp 归约。

## 4. 训练、一阶反向与中间张量

### 4.1 当前通用 forward

`_ln_normalize` 的逻辑是：

```text
mean = reduce_mean(x)
var  = reduce_mean((x-mean)^2)
rstd = 1 / sqrt(var+eps)
xhat = (x-mean) * rstd
y    = xhat * weight + bias
```

Jittor 可以把相邻 elementwise 节点融合，但两个 reduction、保存给 backward 的 `rstd/xhat` 以及
Function tape 会形成物化边界。静态图至少包含两个 row reduction，并通常需要分别物化 rstd、
xhat 和 affine output；精确 CUDA launch 数需要 profiler 确认，不能仅按 Python op 个数相加。

另一个可直接修的低风险候选：当前 `rstd = 1.0 / jt.sqrt(...)` 因 `1.0` 的 promotion 生成了
`float32 -> float64 -> divide -> float32` 图。换成现有 `jt.rsqrt(...)`（实现为整数 `1 / sqrt`）
可保留 float32。CPU-only 数值探针覆盖输入 scale `1, 1e-2, 1e-3, 1e-5`：候选与当前 forward
最大差 `2.38e-7`，随机投影 dx 相对 float64 参考的最坏误差 `1.62e-7`。这只是候选证据，仍需
CUDA/NPU/torch 对拍后才能落地。

### 4.2 当前 backward

令 `dy=dL/dy`，当前图执行：

```text
g  = dy * weight
db = sum_rows(dy)
dw = sum_rows(dy * xhat)
mg  = mean_hidden(g)
mgx = mean_hidden(g * xhat)
dx = rstd * (g - mg - xhat * mgx)
```

这里至少有 4 个逻辑 reduction（db、dw、mg、mgx）与一个 dx elementwise 输出。Jittor 可能把
`g` 生产者分别融入消费者，但这也意味着 `dy*weight` 可能被重复计算/读取。PyTorch CUDA 通常
用一个 row-wise dx kernel 加一个按 feature 归约 gamma/beta 的 kernel，并按 M/N 选择 tile 或
partial reduction。

当前 Function 为 backward 保存：

- 原始 x（tape input）；
- 完整 xhat（与 x 同大小）；
- 每行 rstd；
- affine 外层还需要 weight 与 dy。

更合理的 fused forward 只保存 x、mean、rstd，backward 内重算
`xhat=(x-mean)*rstd`。对 fp32 `(2048,768)`，可少保留约 6 MiB/LayerNorm 的 xhat；大模型多层
训练时这一点可能比单算子微秒数更有价值。

### 4.3 稳定性硬约束

不得采用以下“看起来少一次 reduction”的形式：

```text
var = mean(x*x) - mean(x)*mean(x)
```

它正是历史小方差 bug 的来源。允许的实现为：

- Welford；或
- 先求 mean，再对中心化 `(x-mean)^2` 求和；若为常见小 N 专门化，可把第一遍加载的 X 留在
  registers 中，从而保持两阶段数学但只读一次 global X。

backward 必须直接实现稳定闭式，不得让 autodiff 展开出 `rstd^3` 巨项再依赖抵消。验证损失必须
使用 `sum(y * random_upstream)`；`sum(y^2)` 对标准 LayerNorm 接近常数，会产生退化的近零真梯度。

## 5. 建议实现

### P0：先完成测量与 dispatch 修正

1. 给 benchmark 增加四组独立指标：
   - wall time（保留 corrected slots/outputs 逻辑）；
   - Jittor profiler 的 kernel time 与 launch count；
   - host-only graph build time；
   - peak allocated/reserved memory。
2. 增加 fast-path hit/miss 统计或测试钩子，至少区分：not no_grad、backend、dtype、normalized dims、
   affine combination、shape mismatch。
3. CUDA helper 首先排除 `jt.flags.use_acl`，避免 NPU functional no-grad 误进 `jt.code`。
4. 将 `1.0 / sqrt` 改 `jt.rsqrt` 作为独立小补丁验证，不与 fused kernel 大改绑在一个提交。

### P1：优化 no-grad CUDA kernel

建议保留现有 helper 的 fallback，新增两个 tier：

1. **常见 hidden 专门化（优先 128/256/512/768/1024/2048）**
   - 16-byte 对齐时用 `float4` load/store；动态检查 pointer alignment；
   - 128 threads，warp shuffle + 少量 shared warp partial；
   - X 先载入 registers，reduce mean，再用 registers 求中心化 variance，最后直接写 output；
   - N=768 时每线程最多保存 2 个 float4，register pressure 可控，X 只读一遍；
   - gamma/beta 分别可选，支持 weight-only、bias-only、全 scalar 和全 Var。
2. **通用/大 hidden 路径**
   - vectorized Welford 统计 + vectorized output，两遍 X；
   - 对齐/尾部不满足时回落 scalar；
   - threads 根据 N 分档，避免 N<=32 时固定 128 threads 的浪费。
3. 多维 `normalized_shape` 可在 trailing shape 全匹配且连续时 flatten 为
   `N=prod(normalized_shape)`，weight/bias 同样 flatten；否则回落通用组合路径。

单纯替换归约 kernel 后仍有约 12 us/call 的 Python 建图量级。若 GPU-only kernel 已接近 torch、
wall time 仍落后，应把该路径下沉为一个原生 Jittor `LayerNormCudaOp` 或缓存化的单 meta-op，减少
每次 f-string/reshape/CodeOp 构造。该 op 仍是 Jittor 元算子扩展，不需要也不应链接 torch 底层。

### P2：训练 fused forward/backward

用模块级 `jt.Function`（不要每次调用定义局部 class）实现真实 CUDA 分支：

```text
execute(x, gamma, beta):
    一个 multi-output jt.code -> y, mean, rstd
    self 保存 x/mean/rstd/gamma，返回 y

grad(dy):
    kernel A: 每行 reduce sum(g)、sum(g*xhat)，输出 dx
    kernel B: 按 feature tile reduce，输出 dgamma/dbeta
```

关键点：

- `kernel A` 直接使用当前稳定闭式；可对齐时 vectorize X/dY/gamma/dX；
- `kernel B` 不建议用每 row 原子累加到 gamma/beta：那会增加 zero-init launch、争用和非确定性；
  应做确定性 tile reduction，大 M 时允许 partial buffer + 第二阶段 reduction；
- 根据 input mask 跳过未请求的 dx/dgamma/dbeta，尤其 frozen affine；
- weight-only/bias-only/无 affine 均要支持；
- CUDA 分支之外继续调用 `_ln_normalize`；NPU module 继续调用 ACL native implementation；
- 当前二阶导已 unsupported，新 Function 保持一致并在测试元数据中明确，不静默返回错误二阶导。

### P3：fp16/bf16 与 AMP

当前 helper 只接 fp32。AMP 下 `fp32_guard` 可能先把整个输入 cast 到 fp32、fast kernel 后再 cast
回低精度，形成额外 kernel 和显存流量。完成 fp32 验证后应增加：

- fp16/bf16 vector load/store；
- fp32 accumulation 的 mean/variance/rstd/backward reductions；
- 输出保持输入 dtype，参数梯度保持参数 dtype/既有 Jittor 约定；
- 避免为了命中 fast path 而物化整张 fp32 input；
- mixed `fp32 input + fp16 affine` 保留现有显式 cast，防止 `float * __half` nvcc 歧义回归。

## 6. 验证矩阵

### 6.1 Dispatch / API

| 维度 | 必测项 |
|---|---|
| 入口 | `nn.LayerNorm`、`F.layer_norm`、部署 torch shim 两条路径 |
| grad mode | `no_grad`、`inference_mode`、grad enabled、仅 `eval()`、参数 frozen 但 grad enabled |
| affine | gamma+beta、gamma only (`bias=False`)、beta only functional、无 affine、scalar 1/0 |
| normalized shape | `(N,)`；`(H,W)` flatten fast path；trailing shape mismatch 应清晰 fallback/error |
| layout | 直接连续输入、transpose/reindex 后物化输入、对齐与故意非 vector-friendly N |
| backend | CPU fallback、CUDA fused、NPU module native、NPU functional |

### 6.2 Shape / dtype / 数值

建议至少覆盖：

```text
M(rows): 1, 8, 128, 2048, 65536
N(hidden): 1, 16, 32, 64, 128, 256, 768, 1024, 4096, 4100
dtype: fp32, fp16, bf16
mixed affine: fp32 X + fp16 gamma/beta；低精度 X + fp32 affine
eps: 1e-5, 1e-6, 1e-12
distribution:
  N(0,1)
  N(0,1e-2), N(0,1e-3), N(0,1e-5)
  constant row
  nonzero large mean + small variance（按 fp32 可表达范围构造）
```

精度检查：

- forward 对真 torch CUDA；同时以 float64 NumPy 参考监控统计误差；
- backward 分别检查 dx/dgamma/dbeta，使用随机投影 upstream；
- fp32 小方差至少维持现有 `test_norm.py` 门槛，并目标做到 net relative `<=1e-4`；
- fp16/bf16 按 accumulation 误差设独立容差，不能用放宽容差掩盖 NaN/Inf；
- constant row、极小 eps 的所有输出/梯度必须 finite；
- gradgrad 继续明确 expected unsupported，不能从 loud failure 退化为 silent zero。

### 6.3 性能与显存

每个关键 shape 同时记录：

1. end-to-end wall time；
2. CUDA event / profiler kernel time；
3. launch count 与每个 kernel 名；
4. achieved bandwidth、register/shared-memory、occupancy；
5. host graph-build time；
6. forward-only、forward+dx、forward+dx+dgamma+dbeta；
7. peak memory 与 saved tensor 大小；
8. 首次 JIT latency和 warm steady-state 分开。

验收不能只看 `M=2048,N=768`。建议目标：常见 fp32 inference 的 GPU-only kernel 接近 torch
`<=1.15x`，wall time `<=1.25x`；训练先以 launch 数、saved memory 和端到端 Transformer block
收益为主，防止单算子快但整图因额外 materialization 变慢。

## 7. CPU / NPU 兼容风险与依赖

- **CPU**：不应进入 CUDA `jt.code`。保留 `_ln_normalize` 组合原语可维持功能和现有一阶精度；
  `jt.rsqrt` 小补丁需要 CPU 回归。
- **NPU**：`jt.code` 在 ACL 上不支持。必须用 `use_acl` guard；module 保留现有
  `aclnnLayerNorm/aclnnLayerNormBackward`。functional no-grad 的现有误 dispatch 需要单独回归。
- **NPU affine edge cases**：现有 ACL wrapper 对 `bias=False` 的 scalar bias 也值得纳入矩阵，不能因
  CUDA 重构顺手改变 module 参数/梯度注册方式。
- **二阶导**：CPU/CUDA 现状均不支持 stable norm Function 的 gradgrad；本优化不扩大承诺。
- **新依赖**：不需要。可复用仓库已有 CUB 与 `jt.code` multi-output/`jt.Function` 机制。
- **JIT 体积**：hidden 专门化会增加编译变体。只专门化高频 hidden，其他走通用 kernel；记录
  cold compile 时间和 cache 大小。

## 8. 最终判断

现有 `2.3x` 不是“fast path 没命中”，而是“命中了一个仍较朴素、且 Python 构图成本显著的
单 kernel fast path”。优先做 GPU profiler 可以把约 36.7 us 拆成 host 与 kernel 两部分；随后
用 vectorized register-cache/Welford + warp reduction 优化 inference，并用 multi-output
`jt.Function` 补训练 forward/backward。只要严格限定真实 CUDA backend、保留组合/ACL fallback，
并锁住小方差随机投影一阶梯度，这条扩展路径是可控且符合 Jittor 元算子设计的。
