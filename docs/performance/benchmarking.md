# 性能基准测试

Jittor 用 ASV 0.6.6 保存按提交索引的耗时与内存结果。套件**刻意使用当前 Python 环境**
（`--python=same`）：为每个历史提交新建环境的话，测到的主要是 Jittor 的编译与安装
成本，而不是稳态运行性能。

## 可靠的本地计时

本地微基准用 `jt.benchmark`，不要直接把 `perf_counter()` 套在一个惰性操作外面：

```python
import jittor as jt

inputs = [jt.randn(1024, 1024) for _ in range(4)]
result = jt.benchmark(
    lambda value: (value * value).sum(),
    inputs,
    warmup=2,
    repeat=10,
)
print(result.median, result.samples)  # 单位：秒
```

输入池在计时前被快照并实体化，然后按轮转顺序取用。**预热是强制的**且不计入返回的
样本，因此首次使用的编译落在计时区间之外。被测可调用对象必须返回属于被测工作的
**全部**输出；嵌套的 tuple、list、dict 都支持。每一轮都持有全部返回的 Var，并在停表
前做一次针对性的设备同步。

这些规则避开三种常见的假结果：

- **公共子表达式消除（CSE）**：在上一次的惰性输出仍然活着时反复构造同一个表达式，
  会复用那张图。`jt.benchmark` 在构造下一轮之前先实体化并释放上一轮；操作会修改
  输入时，请用多个常驻池条目。
- **死代码消除**：没有被引用的惰性输出可以不执行就被丢掉。该 API **拒绝**不返回任何
  `jt.Var` 的调用，并把每个返回的 Var 一直持有到同步完成。
- **未实体化的工作**：只对 Python 建图提交计时，既没测到 CPU 执行也没测到异步的设备
  完成。每个样本都包含那次针对性的同步。

样本包含 Python 建图加执行。**跨提交的留存结果用下面的 ASV 套件**，并且把正确性检查
与计时分开。

## 缓存隔离

**ASV 与单元测试套件绝不能编译进同一个 Jittor 缓存。** 每个基准的 setup 都在导入
Jittor 之前校验这两个变量：

```bash
export JITTOR_HOME="${JITTOR_ASV_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/jittor-asv}"
export cache_name="asv-local"
export ASV_PYTHONPATH="$PWD/python"
```

`cache_name` 必须以 `asv-` 开头，解析出的 `JITTOR_HOME` 必须含有 `asv` 路径成分。
**缓存缺失或被复用是硬错误，不是 ASV skip。** 并行的 CPU、CUDA、NPU 任务要用不同的
`JITTOR_HOME` 目录或不同的 `cache_name`。

ASV 启动一个已有环境时会清掉普通的 `PYTHONPATH`，所以 `ASV_PYTHONPATH` 是显式传给
基准进程的源码树路径；nox 会话会自动把它设为 `python/`。这些会话同时把 Jittor 编译
缓存和生成的 ASV 状态放到 checkout 之外。`ASV_RESULTS_DIR` 与 `ASV_HTML_DIR` 可以为
原始 JSON 和发布报告选择持久的外部位置。

## 规范的 nox 运行

安装钉住版本的开发工具，记录维护中的 CPU 选择：

```bash
python -m pip install -r requirements/dev-tools.txt
python -m nox -s benchmark -- \
  --bench '^(operators|optimizer_step)\.'
```

在打了标签、装好钉住版本 CUDA 基准依赖的 CUDA 主机上：

```bash
python -m nox -s benchmark_cuda -- \
  --bench '^(tiny_llama|optimizer_step)\.'
```

两个会话都执行完整的 ASV 流程：`check`、注册机器、对**确切的当前提交** `run` 并采样、
与最近的缓存祖先（或显式的 `ASV_COMPARE_BASE`）`compare`，然后 `publish` HTML 报告。
`ASV_COMPARE_FACTOR` 控制回归倍数，必须大于 1。首次运行会以自身为基线自举。
**只有结果 JSON 与 `index.html` 都存在，这次运行才算成功。**

## 生态对比

`tests/compat/torch/test_ecosystem_parity.py` 与 `test_ecosystem_speed.py` 对比**两个
独立进程**——Jittor 的和二进制 PyTorch 的。两个进程都必须在加载下游库之前拿到各自的
`torch` 命名空间。

只有在 CPython ABI 兼容（通常是相同的主次版本）时它们才共享一个 package site：site
目录里带 ABI 标签的扩展模块，意味着 3.12 的参考解释器无法导入 3.11 的 site，而且会在
第一个编译型依赖处以一个不相干的错误失败。ABI 不同时，用
`JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE` 指定独立 oracle 的 site，两边各导入自己的
一份。**共享 site 时版本与来源都必须相等；分开 site 时版本必须相等，来源本就应当不同。**
CUDA 运行会显式对齐 matmul/cuDNN 的 TF32，可选的 cuDNN autotuning 对两个运行时都启用。

正确性张量会被**立即拷贝**，以免之后的优化器/梯度写入改动某个 NumPy 视图。计时的
训练会预分配多个常驻输入槽和一个 loss 权重张量。Jittor 持有每一个被请求的梯度并对
这些 Var 显式同步；**两个运行时都不在计时窗口内做逐梯度的 D2H。**

带掩码的 SDPA 性能必须保住"整行被掩掉"的语义。CUDA 上维护的注意力路径把显式掩码
交给 softmax kernel 的 `zero_all_neg_inf` 模式，而不是另建一个行有效性归约加两张三元
图。纯因果注意力不需要该模式，因为每一行都含有自己的对角元素。**性能剖析对比必须同时
检查图的行数和墙钟时间。**

结果标签必须与 checkout 一致。nox 会话拒绝脏工作树，除非调用者刻意设
`ASV_ALLOW_DIRTY=1`——那只用于本地调查，**不是可发布的证据**。

CPU 参数在没有 CUDA 时也能导入并执行。当 Jittor 或真实 PyTorch 无法在 CUDA 上执行时，
CUDA 参数抛出显式的 ASV skip。真实 PyTorch 是**可选的 oracle**，需要为目标平台单独
安装；缺失或解析到 shim 的 `torch` 会被报告为 skip，**绝不报告为零耗时结果**。CI 中
至少要有一个强制的 Jittor CPU 用例真正执行——**全部参数都 skip 的运行不可接受。**

## CPU 线程绑核

任何 CPU 对比都要绑核。Jittor 与 PyTorch 默认都不设置 OpenMP affinity，在多核主机上
调度器会在两次测量之间迁移线程：同一个 Jittor ViT 步在六次运行里测出
`0.5555`~`0.7074s`——那是**相差 25% 的两个簇，不是一段离散**。设置

```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

后，四次运行收敛到 `0.5165`~`0.5195s`。绑核对**两个运行时**都值 `10`–`30%`，所以
不绑核的对比可以把结论倒过来：某个扩散 UNet 不绑核读作 `0.90x`、绑核读作 `1.16x`，
因为在那个模型上 PyTorch 从绑核中获益比 Jittor 更多。

读 CPU 数字的人要记住两条推论：

- **不绑核测出的比值不构成证据。** 两边都绑核、多次运行取中位数，并说明是哪种配置
  产生的这个数字。
- 用 `/proc/stat` 观察到的"核在忙"是负载**实际达到**的并行度，不是上限。在断言配额
  限制了什么之前，先看 `/sys/fs/cgroup/cpu.max`。

## 记录选定的修订

已有环境无法让 ASV 构建任意历史。只在与结果标签一致的 checkout 里运行套件，并记录
那个确切的提交：

```bash
commit=$(git rev-parse HEAD)
asv --config benchmarks/asv.conf.json run --python=same --set-commit-hash "$commit"
```

做对比时，建**一个专用的基准 worktree**，只把这个 worktree 切到那两三个被评审的修订，
每次切换后运行上面的命令。保持相同的编译器、加速器、依赖版本、缓存预热策略和 ASV
结果目录。**不要跑 `ALL` 或无界的修订区间。** 两个选定提交都有结果之后，用
`asv --config benchmarks/asv.conf.json compare <base> <candidate>`。

这些直接命令要从仓库根目录运行。配置由 `benchmarks/` 拥有，其 `repo: ".."` 相对于
配置文件解析，而 `benchmark_dir` 和输出目录相对于工作目录解析。维护中的
`nox -s benchmark` 入口会生成一份使用绝对仓库、基准和外部状态路径的配置，因此移动
源配置不会把 CI 产物重定向进 checkout。

## CI 留存与节奏

CPU 工作流在基线 CPU 容器里跑 `nox -s benchmark`：可用时恢复先前的结果 JSON，记录
当前提交，比较并发布，然后保存更新后的结果缓存。原始 JSON 与生成的 HTML 作为留存的
CI 产物一起上传。

**CUDA 基准刻意不作为 PR 门禁。** CUDA 工作流在其真实设备测试门禁之后，按周计划与
手动触发运行 `nox -s benchmark_cuda`。它使用打了标签的 CUDA 12.2 RTX 4090 runner、
独立的结果缓存，上传独立的 CUDA JSON/HTML 产物。这个节奏让加速器回归可见，又不必为
每次源码推送占用专用硬件。

## 初始基准集

| ASV 模块 | 覆盖 | 参数 |
| --- | --- | --- |
| `operators` | matmul、softmax、LayerNorm、GELU | Jittor / 可选 torch；CPU/CUDA |
| `tiny_llama` | LlamaModel 前向、前向+反向 | Jittor / 可选 torch；CUDA |
| `optimizer_step` | 就绪梯度下 SGD 与 AdamW 步的规模伸缩 | 32/128/512 个张量；CPU/CUDA |

Tiny Llama 沿用既定配置：2 层、hidden 256、intermediate 768、8 个注意力头 / 4 个 KV
头、batch 2、序列 128。它的 setup 会检查输出和每一个被请求的梯度是否有限且非零。
优化器 setup 把总元素数固定为 262,144，因此**张量数量这个轴暴露的是每张量的建图与
发射开销**，而不是更大的计算量。CUDA 对比使用 float32 并在两个后端上都开启 TF32。
算子基准使用推理/无梯度语义；Tiny Llama 只在它的"前向+反向"参数上开启梯度。

每个内存结果都是单位为 `bytes` 的 ASV `track_*` 基准。CUDA 上，
`track_working_set_bytes` 是一次操作/模型/步之后后端分配器**同步过的活跃工作集**；
CPU 上则是操作系统报告的进程峰值 RSS。它们在**同一后端、同一机器内**是稳定的回归
信号；它们不是 NVML 峰值，**不应跨设备比较**。
