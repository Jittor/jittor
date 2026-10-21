# 热缓存 `import jittor` 的耗时归因

- 状态：归因接受。它指向的热缓存缺口已由核心与自带库的构建戳补上（热缓存 import
  两种配置都已低于 1 s）；**冷缓存与切换配置时 import 仍编译整个核心**，这一半仍开放
- 日期：2026-09-04
- 基线提交：`bf702127`（归因）；构建戳落地后的数字取自 `d23f9bba`/`d41106d5a`
- 验证范围：x86_64 + CUDA 12 开发机（sm_89），同一 `JITTOR_HOME` 的热缓存，CPU-only
  （`nvcc_path=""`）与 CUDA 两种配置；全部是 `perf_counter` 墙钟，不用 profiler。
  未覆盖：ACL/ROCm 构建、冷缓存的分项
- 维护者：构建系统维护者
- 复查条件：改动 `python/jittor/build/compiler.py` 的模块体、`compile_extern.py` 的
  setup 链、`cache_compile` 的缓存键、`run_cmds` 的并行策略，或核心翻译单元数量明显变化

## 结论

**热缓存 import 的最大一项不是探测，也不是 dlopen，而是「核心编译」这一步在无事可做
时的固定开销**：CPU-only 配置 1.325 s 里占 0.906 s（68%），CUDA 配置 2.457 s 里与自带
CUDA 库的空转校验、`cupy` 合计 1.636 s（67%）。热缓存下这些步骤的净效果是零——生成的
头文件逐字节相同，所有编译命令都判定为最新——但每次 import 都付一遍。

## 量法

三层互补，都在子进程里用 `perf_counter` 墙钟量（cProfile 在这条路径上加 40% 并改变
排序）：

1. 模块体：`-X importtime`，精确但粒度是一个模块；
2. 构建扇出：在 `import jittor` **之前**替换 `jittor_utils.run_cmds`，记下每次扇出的
   命令条数与耗时（要量的东西在编译器模块体里，导入之后已经跑完）；
3. 生成器：import 之后再调一次 `gen_jit_flags` / `pyjt_compiler.compile`，它们幂等，
   第二次的耗时就是第一次的价格。

## 归因表（CPU-only，1.325 s）

| 项 | 耗时 | 占比 | 热缓存下实际做了什么 |
| --- | ---: | ---: | --- |
| `run_cmds`「Compiling jittor_core」176 条 | 0.542 s | 41% | 起 16 进程池，逐条读源文件与全部依赖头、算内容哈希、比 `.key`，一致即返回；产物不变 |
| `gen_jit_flags()` | 0.212 s | 16% | glob 176 个 `.cc`，纯 Python 逐字符剥注释、正则抓 `DEFINE_FLAG`，写出逐字节相同的 `gen/jit_flags.h` |
| `gen_pyjt` | 0.104 s | 8% | 扫 177 个头/源，重写 `gen/*.cc` |
| `run_cmds`「Compiling jittor_mpi_core」7 条 | 0.044 s | 3% | 同样的空转校验 |
| `jittor.compat.triton` 模块体 | 0.091 s | 7% | 导入 `triton.runtime` |
| 其余（numpy、dlopen、各模块体、`probe.json`） | ≈0.33 s | 25% | `probe.json` 单次读取 <0.2 ms，探测已不是热路径成本 |

CUDA 配置另加：自带 CUDA 库（cub/cublas/cudnn/curand/cufft/cusparse 与
`libcuda_extern`）49 条命令的空转校验 0.351 s，无条件 `import cupy` 0.369 s（其中扫
289 个 dist-info 0.199 s），以及更大的 CUDA 核心 dlopen 0.204 s。

另一个常被误读的数字：「冷编译 40 s」并不只发生在空缓存上。同一个 `JITTOR_HOME` 里从
CUDA 配置换到 CPU-only 配置，两套配置指纹不同，各要一份完整核心——在门禁之间来回切，
每切一次就付一次。

## 由此得出、并已落地的部分

1. **核心编译需要一条「已经最新」的快路。** 核心与自带库现在都写构建戳
   （`python/jittor/build/compilation.py` 的 `product_build_is_current` 一族）：戳记录
   全部源文件、配对头文件与生成源的编译器模块的哈希，命中则整步跳过。落地后热缓存
   import CUDA 配置 1.28 → 0.80 s、CPU-only 0.52 → 0.38 s，热 import 的编译扇出从 60 条
   降到 0 条。`e3c369acb` 上同类机器（RTX 4090，CUDA 配置）复测三次为 0.86–0.89 s。
2. **第三方依赖不再无条件导入。** `cupy` 只在首次需要 numpy→cupy 转换时惰性导入
   （`python/jittor/build/init_cupy.py`），普通 `import jittor` 也不再导入
   `jittor.compat` 下的模块。
3. **`JITTOR_NO_BUILD=1`** 让「这次 import 不许编译」变成可断言的：任何要编译的步骤都
   报错而不是悄悄编译，用来保证量的确实是热缓存。

判错的代价不对称：戳把「该重建」判成「最新」会静默算错，判成「过期」只损失时间，所以
戳宁可多记。改了编译器模块（哪怕只加注释）会让核心构建戳失效，跑一次
`python -m jittor_utils.bootstrap` 即可，不是 bug。

## 仍开放

- 冷缓存与切换配置时，`import jittor` 照旧编译整个核心；把核心编译移到显式 bootstrap
  或首次算子调用的工作尚未完成；
- 空转校验成本与核心翻译单元数成正比（约 6–8 ms/条墙钟，16 路并行），所以增加 TU 的
  改动应报「+N 个 TU」而不只是「+0.1 s」。

## 复现

```bash
JITTOR_HOME=<isolated JITTOR_HOME> TMPDIR=<isolated TMPDIR> PYTHONPATH=<worktree>/python \
EXPECT_JITTOR_SRC=<worktree>/python nvcc_path=<path to nvcc> \
python agent/skills/jittor-build-change-verification/measure_import_cost.py --json before.json
```

`nvcc_path=""` 换成 CPU-only 配置（第一次会付一次该配置的冷编译）。两次运行之间不要
清缓存；连续跑三次取后两次。造冷缓存的可复现办法与构建戳的设计规则见同一 skill 的
§2.5–§2.7。
