# 热缓存 `import jittor` 的耗时归因

- Status: Accepted
- Date: 2026-09-04
- Baseline: `bf702127`（`2.0-refactor`），任务 `9.01` 的第 3 个提交之前
- Owner: 整改 `build` 分区
- Review when: 改动 `compiler.py` 模块体、`compile_extern.py` 的 setup 链、
  `cache_compile` 的缓存键、或 `jittor_utils.run_cmds` 的并行策略
- 复现方法: `agent/skills/jittor-build-change-verification/SKILL.md` §2.5

## 为什么先归因

任务 `9.01` 的验收是「热缓存 import < 1 s」。此前两个提交（`361d59b2`、`c4b21762`）
把 Torch 探测与 NCCL/cuTT/MKL setup 挪出了默认 import 路径，热缓存 import 从
更早的数字降到 **1.332 s**，还差 0.33 s，而看板只记了这个总数。总数不能指导下一步：
0.33 s 可能来自十个各 33 ms 的地方（那就没有单点可改），也可能来自一个 900 ms 的
地方（那就有）。本文把这 1.332 s 拆开。

结论先说：**热缓存 import 的最大一项不是探测，也不是 dlopen，而是「核心编译」这一步
在无事可做时的固定开销**——它占 CPU-only 配置 1.325 s 中的 **0.906 s（68%）**。

## 一个前提：1.332 s 是 CPU-only 配置的数字

看板的 1.332 s 与「冷配置 174 个核心 TU、40.015 s」是同一件事的两面，而两者都取自
**`nvcc_path=""` 的 CPU-only 配置**。本机有 CUDA（见
`2026-09-04-cuda-availability-verification.md`），所以同一棵树有两个热缓存数字：

| 配置 | 热缓存 import | 核心 TU 数 | 说明 |
| --- | --- | --- | --- |
| `nvcc_path=""`（CPU-only） | **1.325 / 1.335 s** | 176 | 复现了看板的 1.332 s |
| `nvcc_path=<nvcc>`（CUDA，sm_89） | **2.457 s** | 177 + 6 个 extern op 模块 | 看板未记过 |

「40.015 s 冷编译」也复现到了 **39.96 s**，但它的触发条件值得写清楚：那不是空缓存，
而是**在 CUDA 配置已热的同一个 `JITTOR_HOME` 里改用 CPU-only 配置**。两套配置的
`cfg*` 指纹不同（`cfg08ac6be4` 对 `cfg1d275ac1`），各自要一份完整的核心。也就是说
在三套门禁之间来回切，每切一次就付一次 40 s——这条与审计 §缓存键与缓存布局
「不同配置的进程在同一目录互相重编」是同一个根因的另一半。

## 归因表

三个层次的量法互补，都是 `perf_counter` 墙钟，没有 profiler 开销（cProfile 在这条
路径上加 40% 并会改变各项排序）：

- **level 1** 模块体：子进程 `-X importtime`。精确但粒度是「一个模块」。
- **level 2** 构建扇出：在 `import jittor` **之前**替换 `jittor_utils.run_cmds`。
  必须从外面打，因为要量的东西在 `compiler.py` 的模块体里，等到 `jittor` 可导入时
  它已经跑完了；`compiler.py` 通过属性查找调用 `jittor_utils`，所以补丁能生效。
- **level 3** 生成器：import 之后再调一次。它们是幂等的（同样的输入写同样的文件），
  第二次的耗时就是第一次的价格。

### CPU-only 配置：1.325 s

| 项 | 耗时 | 占比 | 属于 | 这一步在热缓存下实际做了什么 |
| --- | --- | --- | --- | --- |
| `run_cmds`「Compiling jittor_core」176 条命令 | **0.542 s** | 41% | 核心编译 | 建一个 16 进程 `Pool`（`pool_size=min(16, mem/3GB)`），把 176 条编译命令发进去，每个 worker 调 `cc.cache_compile` 读源文件与全部依赖头、算内容哈希、比 `.key`，一致就返回 False。**产物一个字节都没变。** |
| `gen_jit_flags()` | **0.212 s** | 16% | 核心编译 | glob `src/**/*.cc`（176 个），逐个读入、用纯 Python 逐字符剥注释（`strip_cxx_comments`，2.3 M 次 `str.startswith`）、正则抓 `DEFINE_FLAG`，写出与上次**逐字节相同**的 `gen/jit_flags.h` |
| `gen_pyjt`（`pyjt_compiler.compile`） | **0.104 s** | 8% | 核心编译 | 扫 177 个头/源，重写 `gen/*.cc` |
| `run_cmds`「Compiling jittor_mpi_core」7 条 | **0.044 s** | 3% | extern | 同上，7 条命令的空转缓存校验 |
| `jittor.compat.triton` 模块体 | 0.091 s | 7% | 兼容层 | `install()`，导入 `triton.runtime`（0.051 s） |
| `jittor.extern.acl.acl_compiler` 模块体 | 0.044 s | 3% | 后端探测 | Ascend 后端注册（本机无 CANN） |
| `numpy` | 0.043 s | 3% | 依赖 | — |
| `jittor_mpi_core` dlopen | 0.052 s | 4% | extern | — |
| `jittor_core` dlopen | 0.003 s | 0% | 核心 | CPU 版核心很小 |
| `gen_jit_tests()` | 0.004 s | 0% | 核心编译 | — |
| 其余（Python 层各模块体、探测、`probe.json`） | ≈0.19 s | 14% | — | `probe.json` 单次读取 <0.2 ms，可忽略；0.09 落地之后探测已不再是热路径的成本项 |

**「核心编译」四项合计 0.906 s = 1.325 s 的 68%。** 这四项在热缓存下的净效果是零：
生成的头文件逐字节相同，183 条编译命令全部判定为最新。

### CUDA 配置：2.457 s

CPU-only 的每一项都在，另加：

| 项 | 耗时 | 这一步做了什么 |
| --- | --- | --- |
| `run_cmds`「Compiling jittor_core」177 条（含建 Pool） | 0.608 s | 同上 |
| `run_cmds` × 7：`libcuda_extern` 2 条、cub 6、cublas 9、cudnn 16、curand 5、cufft 5、cusparse 6 | **0.351 s** | `compile_extern.setup_cuda_lib` → `compile_custom_ops` → 同样的空转缓存校验，49 条命令 |
| `jittor.init_cupy` → `import cupy` | **0.369 s** | 模块体无条件 `import cupy`，其中 `cupy._environment._detect_duplicate_installation` 扫 289 个 dist-info 花 0.199 s。`numpy2cupy` 只被 `jt.numpy_cupy` 自定义算子路径用到 |
| `jittor_core` dlopen | 0.204 s | CUDA 版核心大得多 |
| `jittor.compile_extern` 模块体 self | 0.587 s | 主要就是上面那 7 次 `run_cmds` 与各 `ctypes.CDLL` |

CUDA 配置下「核心编译」四项 0.916 s + extern 空转校验 0.351 s + `cupy` 0.369 s
= **1.636 s，占 2.457 s 的 67%**。

## 结论对下一步的指向

1. **核心编译这一步必须有一条「已经最新」的快路**。现在它每次 import 都重新
   生成一遍头文件、再用 16 个进程把每个 TU 的依赖闭包读一遍来确认无事可做。
   这既是 `9.01`「核心编译移到显式 bootstrap 或首次算子调用」的落点，也是热缓存
   0.9 s 的来源——两个缺口是同一处代码。
2. **判据不能只看总时间**。`run_cmds` 的空转校验成本与 TU 数成正比
   （每条命令约 3 ms，16 路并行摊到约 6-8 ms/条的墙钟），所以任何增加核心 TU 数的
   改动都会等比推高热缓存 import。报「+0.1 s」不如报「+30 个 TU」。
3. **`cupy` 与 `compat.triton` 是无条件导入的第三方依赖**，合计 0.46 s（CUDA 配置）。
   它们不在 `9.01` 的文字范围内（计划点名的是核心编译、`setup_nccl/cutt/mkl`、
   `FIX_TORCH_ERROR`），但要让 CUDA 配置的热缓存 import 进 1 s 就绕不开 `cupy`。

## 复现

```bash
EXPECT_JITTOR_SRC=<worktree>/python PYTHONPATH=<worktree>/python \
JITTOR_HOME=<你的 JITTOR_HOME> TMPDIR=<你的 TMPDIR> \
nvcc_path=/usr/local/cuda/bin/nvcc \
python agent/skills/jittor-build-change-verification/measure_import_cost.py \
    --json before.json
```

`nvcc_path=""` 换成 CPU-only 配置。两次运行之间**不要**清缓存——本文全部数字是
热缓存的。冷缓存怎么可复现地造、以及为什么「换个 flag 再 import」也是一次冷编译，
见 skill §2.5。
