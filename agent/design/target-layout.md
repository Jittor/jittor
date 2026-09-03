# 目标目录布局

2026-09-02。回答一个问题：仓库应该长成什么样。配套的搬动顺序在
[整改计划](refactor-plan.md) 各阶段末尾的「布局收尾」行里；本文只定终点和每一处搬动的理由。
它取代 `docs/architecture/repository-layout.md` 里原来那棵树——那棵树描述的是 2.0
领域包迁移之后的**现状**，并用结构测试把现状冻成了目标。

## 1. 现状乱在哪

不是目录名的问题，是三件事。

**C++ 内核住在 Python 包里，构建系统与用户 API 混在同一层。** `python/jittor/src` 46.9k 行、
`python/jittor/extern` 27.6k 行，由 `MANIFEST.in` 整目录塞进 wheel，`import jittor` 时现场编译。
包根下 7k 行散文件里，`compiler.py`、`compile_extern.py`、`pyjt_compiler.py`（合计 3.4k 行，是构建
系统）与 `linalg.py`、`distributions.py`、`init.py`（是用户 API）并排；`python/jittor_utils` 作为第二个
顶层包又反过来 import jittor。

**同一个东西在多处。** CUDA kernel 在四个地方：`extern/cuda/*/ops`（31 个 .cc）、`nn/backends/` 10 个
`*_cuda.py`、`nn/` 根下 6 个 `*_cuda.py`、`src/ops` 里的 `#ifdef HAS_CUDA`，`compat/shim/backends/flash_attention.py`
里还有 2k 行内嵌串。池化在 `pool/` 与 `nn/modules/pooling.py` 各一份且数值不同。`misc/tensor_ops.py`
2874 行是万物抽屉；`_runtime/core_api.py` 2614 行同时装着 Var、Module、Function、flags、hook。`src/` 根下
39 个散文件（executor、var、op、node、grad、fused_op）是最重要的代码却没有目录；`src/misc` 12.1k 行里
9.2k 是 vendored miniz，类型系统一半在 `misc/nano_*` 一半在 `type/`。

**仓库根两套文档加杂物。** `agent/`（23 份带日期的 results、baselines、skills）与 `docs/` 并列，真正的
设计文档在 `agent/design`，`docs/architecture` 里躺着那棵冻结的树；`tools/services/legacy`、两份
AWESOME 列表、asv 配置在根目录。

## 2. 目标树

```text
.
├── pyproject.toml  README.md  LICENSE.txt  AGENTS.md  CONTRIBUTING.md  noxfile.py
├── src/                      # C++ 核心。搬出 Python 包，不再是 package data
│   ├── core/                 # node var op graph grad executor fused_op  ← 今天 src/ 根下的 39 个文件
│   ├── type/                 # nano_string nano_vector dtype 表  ← misc/ 里的类型系统并进来
│   ├── mem/                  # 分配器
│   ├── codegen/              # opt/ 全部 + op_compiler + jit_key + jit_compiler + kernel_ir
│   ├── ops/                  # 七个元算子与少数基础算子；composite 算子另立子目录
│   ├── runtime/              # init、flags、profiler、lock
│   ├── bindings/             # pyjt + pybind
│   ├── third_party/          # miniz 及其它 vendored 源码
│   └── tests/                # C++ 单元测试
├── backends/                 # 每个后端同一形状：build 片段 + kernels/ + 注册项
│   ├── cpu/                  # mkl / oneDNN
│   ├── cuda/                 # cublas cudnn cufft curand cusparse cutt cub
│   │                         #   + nn/ 与 nn/backends/ 的 16 个 *_cuda.py 内核
│   │                         #   + flash_attention 的 CUDA 串
│   ├── acl/  rocm/  corex/
│   └── comm/                 # mpi nccl hccl：通信也是后端
├── python/jittor/            # 纯 Python
│   ├── __init__.py  __init__.pyi  selftest.py
│   ├── _core/                # core_api.py 拆成 var.py module.py function.py flags.py hooks.py
│   ├── build/                # compiler compile_extern pyjt_compiler cuda_wheel install_cuda
│   │                         #   + jittor_utils 并入（它不再 import jittor）
│   ├── ops/                  # 今天的 misc/：indexing reductions shape_* + 拆开的 tensor_ops
│   ├── nn/                   # modules/ functional/ utils/ attention.py；根下不再有 *_cuda.py
│   ├── optim/  autograd/  fft/  sparse/  dataset/  transform/  models/
│   ├── linalg/  distributions/  init/     # 三个 700–1600 行的单文件各自成包
│   ├── distributed/          # 启动器、process group、多机 rendezvous 的 Python 面
│   ├── contrib/              # ccl loss3d math_util einops：领域算法与 vendored 库，或直接分出去
│   └── tools/                # nvtx jtune tracer：真正给用户的工具
├── compat/                   # 独立发行物 jittor-torch：torch shim fsdp2 vllm triton
├── tools/                    # 仓库工具：gen_pyi local_doc_builder dumpdef release benchmarks
├── tests/                    # 目录镜像源码：tests/core ↔ src/core，tests/backends/<name> ↔ backends/<name>
├── docs/                     # 唯一的文档树；agent/design 并入 docs/architecture
├── examples/  benchmarks/
└── agent/                    # 只留 manuals/ skills/ scripts/：给 agent 的操作手册，不放设计与结果
```

## 3. 每处搬动的理由

| 来源 | 去向 | 为什么 | 前置任务 |
| --- | --- | --- | --- |
| `python/jittor/src/` | `src/` | 内核不是 package data；搬出后 wheel 的 `MANIFEST.in` 与 `compiler.py` 找 `jittor_path` 的逻辑一起改，与「import 不再编译」是同一件事 | 9.01 |
| `src/` 根下 39 个文件 | `src/core/` | 最重要的代码应当有名字 | 1.01（还原的五个文件一起放进去） |
| `src/misc/nano_*`、`src/type/` | `src/type/` | 类型系统只有一处 | 2.15 |
| `src/misc/miniz*` | `src/third_party/` | vendored 代码不与核心混排 | — |
| `src/misc/cuda_flags.cc`、`init.cc`、`profiler/` | `src/runtime/` | 全局状态归 Runtime 对象所在目录 | 2.13 |
| `src/opt/`、`op_compiler.*`、`jit_key.*`、`jit_compiler.*` | `src/codegen/` | 代码生成是一个子系统 | 3.08 |
| `src/ops/` 里非元算子（getitem/setitem/argsort/candidate/where/fused_adamw…） | `src/ops/composite/` | 「元算子」成为代码里的边界而非宣传语 | 4.04 |
| `src/pyjt/`、`src/pybind/` | `src/bindings/` | 绑定是一层 | 2.08 |
| `python/jittor/extern/cuda/*` | `backends/cuda/` | 后端有共同形状才能有共同契约 | 4.03 |
| `nn/backends/*_cuda.py`、`nn/*_cuda.py`、`flash_attention.py` 的 CUDA 串 | `backends/cuda/kernels/` | 四处变一处；Python 层只留调用 | 4.04、4.10 |
| `extern/acl`、`extern/rocm`、`extern/corex` | `backends/<name>/` | 同上；ROCm 不再吃改写产物 | 4.11、4.12 |
| `extern/mpi`、`extern/cuda/nccl`、`extern/acl/hccl` | `backends/comm/` | 通信器是后端；三份 dtype 表合一 | 6.B01、8.08 |
| `compiler.py`、`compile_extern.py`、`pyjt_compiler.py`、`cuda_wheel.py`、`install_cuda.py`、`python/jittor_utils/` | `python/jittor/build/` | 构建系统与用户 API 分层；`jittor_utils ⇄ compiler` 的环随之消失 | 4.07 |
| `_runtime/core_api.py` | `_core/{var,module,function,flags,hooks}.py` | 2614 行是五个域 | 5.04、5.06、5.07、5.08 |
| `misc/` | `ops/` | 名字说明内容；`tensor_ops.py` 按 indexing/reduction/shape/sort/scan 拆 | 5.13、5.14 |
| `pool/` | `nn/modules/pooling.py`、`nn/functional/pooling.py` | 两份 AvgPool 合一 | 5.17 |
| `linalg.py`、`distributions.py`、`init.py` | 同名包 | 1602/921/738 行的单文件 | 5.18 |
| `ccl/`、`loss3d/`、`math_util/`、`einops/` | `contrib/` 或独立仓库 | 领域算法与 vendored 库不是框架核心；公开 import 路径用转发保留 | — |
| `utils/` | 拆散：`build/`、`tools/`、`python/jittor/tools/`、compat | 杂物间 | 5.25 |
| `compat/` | 顶层 `compat/`，独立 distribution | torch 不是 jittor 的别名；vLLM 适配器本就计划分出去 | 7.12 |
| `agent/design/`、`agent/results/` | `docs/architecture/`、`docs/results/` | 一处权威 | 11.02 |
| `tools/services/legacy`、`tests/system/`、`src/utils/flags.cc` | 删除 | 死目录 | 9.17 |

## 4. 保留不动的

- `nn/modules` 与 `nn/functional` 的拆分，以及 `optim/`、`autograd/`、`dataset/`、`transform/`、`models/`
  这些领域包：名字级检查证实它们没有重复实现，是按域拆分的正确骨架。
- 1.x 的公开 import 路径（`jittor.ccl.ccl_2d`、`jittor.pool.AvgPool2d`、`jittor.misc.*`）：搬动时留转发
  模块并标记 deprecated，一个大版本后删除。
- `tests/` 的顶层分类；只把内部结构改成镜像源码目录。

## 5. 打包与 import 的耦合

今天 wheel 是「源码加 JIT」：`MANIFEST.in` 把 `python/jittor/src`、`math_util/src`、
`compat/shim/cpp_extension/src` 整目录带上，用户机器上编译。`src/` 搬出包后有两种做法：
`pyproject` 用 `package-data` 或 `force-include` 把顶层 `src/` 与 `backends/` 打进 wheel（保持源码加
JIT），或者预编译核心只 JIT 算子。前者是一次机械改动，后者是另一个项目。**本文只要求前者**，
并要求它与 9.01（import 不编译不下载）同一提交：`jittor_path` 的解析、`cache_compile` 的依赖跟踪
与 `jittor_utils` 的路径推断都要在那次一起改。

## 6. 顺序

布局是每个阶段的**收尾**，不是开头。把四份 CUDA kernel 挪进一个目录改不了「有四份」；挪混淆
文件没有意义；每一步都会撞结构测试。因此：

1. 先把结构测试从「精确清单」改成「规则」（计划 0.19），否则每挪一个文件门禁就红。
2. `agent/` 与 `docs/` 定权威（11.02）与 `build/` 包（4.07 之后）可以早做，与其它分区不冲突。
3. `src/` 内部分目录在 `data.gz` 还原（1.01）之后；`src/` 搬出包与 9.01 同一提交。
4. `backends/` 在 OpRegistry（4.04）之后；`comm/` 在 ProcessGroup（8.08）之后。
5. Python 包重排在阶段 5 各任务之后；`compat/` 分出去是最后一步（7.12）。

每次搬动一个提交，只含 `git mv`、转发模块与结构测试的同步修改，不夹带行为改动。
