---
name: cuda-negative-path-verification
description: 在有 CUDA 的开发机上真正跑 CUDA 负向用例（错误路径），并判断一次绿色运行是否算数。适用于：验收 USER_CHECK / ASSERT 迁移、给某个后端边界补负向用例、或看到交接文档写着「本机无 CUDA，未运行负向」而需要核实时。也用于诊断「pytest 跑到一半无声消失」「整目录只跑了 36% 就结束」这类现象。
---

# 先证明这台机器有 CUDA

**交接文档里的「本机无 CUDA」几乎总是环境变量造成的假象。** 最常见的成因是照搬
了 CPU-only 的加速写法（`JITTOR_TEST_DEVICES=cpu nvcc_path=""`）之后再去读
`jt.has_cuda`——`nvcc_path=""` 会让 jittor 认为没有 nvcc，于是 `has_cuda` 为 0，
而这与硬件无关。

自检必须四项一起看，缺一项结论都不成立：

```bash
PYTHONPATH=$WORKTREE/python JITTOR_HOME=$MYCACHE TMPDIR=$MYTMP \
CUDA_VISIBLE_DEVICES=$MYGPU nvcc_path=/usr/local/cuda/bin/nvcc \
PATH=/usr/local/cuda/bin:$PATH \
python -c "
import os, jittor as jt
print('tree     ', os.path.dirname(jt.__file__))   # 必须是你的 worktree
print('nvcc     ', jt.compiler.nvcc_path)          # 必须非空
jt.flags.use_cuda = 1
print('has_cuda ', jt.has_cuda)                    # 必须是 1
import numpy as np
a = jt.array(np.random.rand(64, 64).astype('float32'))
print('matmul   ', float(a.matmul(a).sum().item()) != 0)
"
```

判据与陷阱：

- **`nvcc_path` 必须显式给。** 不给（或给空串）时 `has_cuda` 为 0，而这不是「没有卡」。
- **`PYTHONPATH` 必须显式给**（手写 `python -c` 时）。不给就是在测主树，失败是静默的。
  `pytest` 不需要，`pyproject.toml` 的 `pythonpath` 已经指到本 checkout。
- **`jit_utils was rebuilt and cannot be reloaded in this process`** 不是错误：原样重跑
  同一条命令即可，可能要连跑两次。
- 冷编译 CUDA 核心约 **45 秒**（不是十分钟）。第一次慢是预期。
- 打印出来的 `Found cuda archs: [..]` 才是硬件事实；`sm_89` 是 RTX 4090。

# 跑整个 CUDA 目录

```bash
JITTOR_HOME=$MYCACHE TMPDIR=$MYTMP \
CUDA_VISIBLE_DEVICES=$MYGPU nvcc_path=/usr/local/cuda/bin/nvcc \
PATH=/usr/local/cuda/bin:$PATH \
taskset -c $MYCORES python -m pytest tests/backends/cuda -v -rs -p no:cacheprovider
```

- **不要加 `-x`。** 你要的是全部结果，不是第一个失败。
- **`-v` 而不是 `-q`。** 目录里一旦有用例让进程 abort，`-q` 的进度条只会给你一串
  `.F.` 加一段 `Fatal Python error`，**看不出是哪一个**。`-v` 会在每个用例开跑前打印
  它的名字，abort 之后日志最后一行就是罪魁。
- **`-rs` 打印 skip 原因**，下一节要用。

## 退出码 134 = SIGABRT = 有东西调用了 std::terminate

这不是「一条用例失败」，这是**整个进程死了，后面的文件一个都没跑**。表现是
「23 个文件里只跑到第 12 个就结束了」，而 pytest 的汇总行根本不会出现。

定位方法（缓存已热时几秒）：

```bash
CUDA_VISIBLE_DEVICES=$MYGPU nvcc_path=/usr/local/cuda/bin/nvcc gdb_path= \
PYTHONPATH=$WORKTREE/python JITTOR_HOME=$MYCACHE TMPDIR=$MYTMP \
gdb -batch -ex "set pagination off" -ex run -ex "bt 45" \
    --args python repro.py
```

栈里出现 `__cxa_call_terminate` + `_Unwind_Resume` + 某个 `~Foo()`，就是
**析构里抛了异常**。见下一节。

# 析构里的错误：为什么「外面包一层 try」不管用

C++11 起**析构函数隐式 `noexcept`**。异常离开析构函数的那一刻就是
`std::terminate`，**terminate 发生在析构函数自己的栈帧上**，调用方的 `catch`
在它下面，永远轮不到。

所以这种写法是无效的，而且它看起来完全正确：

```cpp
// 生成的 tp_dealloc —— 这个 catch 抓不到 ~T() 抛的东西
try { ~T(); tp_free(self); return; } catch (...) { /* 永远不会到这里 */ }
```

**正确的位置是析构函数自己：**

```cpp
Foo::~Foo() {
    try {
        may_throw();
    } catch (const std::exception& e) {
        LOGe << "... ignored during teardown:" << e.what();   // LOGe 不抛
    }
}
```

**结构测试查不出这一类。** `tests/structure/core/test_destructor_and_handler_contract.py`
扫的是析构体里**字面出现**的 `ASSERT` / `CHECK` / `LOGf`。经由一次函数调用抛出来的
（`~VarHolder` → `release_both_liveness` → `ASSERT`）它看不见，于是**门禁是绿的而进程
会 abort**。判据：**析构里只要调用了非 `noexcept` 的东西，静态扫描就已经不作数了，
必须有一条运行时用例。**

**扫描的根目录也要核。** 这类扫描按目录列举源码，而目录会搬。后端搬进 `backends/` 之后，
只写 `python/jittor/{src,extern}` 的扫描对 CUDA/ACL/ROCm 的析构**一条都没看**，
而总数看上去仍然健康。所以断言要**逐个根目录非空**，不是断言总数；后缀也要带
`.cu`/`.cuh`。同样的道理适用于任何「按路径清单扫全树」的门禁。

顺带：一旦让错误能从这些路径传出去，**被打断的全局状态要用 RAII 复位**（jittor 的
liveness 队列就是一例：抛在半路会留下 `front` 下标，下一次 drain 从陈旧位置继续）。

# 写一条算数的负向用例

四条缺一不可：

1. **构造真的违反该边界的输入**，并且**直接调用那个算子**——
   `jt.compile_extern.cudnn_ops.cudnn_conv(...)`，不要走 `jt.nn.conv2d`。
   Python 包装层自己先校验，走包装层测到的是包装层，C++ 边界一行都没碰到。
2. **`expect_error(..., exc_type=RuntimeError, match=r"...")` 两个参数都要给。**
   只断言「抛了点什么」，一个无关的 setup 报错也能让它变绿。
3. **必须 `.sync()`**：Jittor 是惰性的，很多 `USER_CHECK` 在 `infer_shape` 或
   `jit_prepare` 里，不 sync 就不会执行。
4. **抛完之后再算一次真东西**，断言运行时还活着：

   ```python
   def rejects(self, make, match):
       error = expect_error(lambda: make().sync(), exc_type=RuntimeError, match=match)
       with jt.flag_scope(use_cuda=1):
           self.assertEqual(float((jt.ones((4, 4)) * 2).sum().item()), 32.0)
       return error
   ```

   「抛出来了」和「抛出来之后还能用」是两件事，而 2.19 分的正是这两档。

**预期会 abort 的子进程一律 `crash_isolated=True`**（`_helpers.child_process`）。
jittor 装了进程级 SIGCHLD 处理器：子进程被信号杀死会让 pytest **无声消失、零输出**，
读起来像「跑挂了」而不是「测试失败」。

**`jit_prepare` 里的 USER_CHECK 会丢类型。** 它跑在并行编译线程上，
`parallel_compiler` 把它重新包成自己的 `RuntimeError`，`UserError` 这个类没了，
消息保留在 `Reason:` 后面。所以这类边界只能断言文本，不能断言异常类。

# skip 不等于通过——逐条问为什么

`N passed, M skipped` 里的每一条 skip 都要有答案。真实分类：

| skip 原因 | 是不是真的 | 怎么办 |
| --- | --- | --- |
| `this machine has 1 visible CUDA device(s), the test needs 2` | 真（你被分了一张卡） | 记下来，不要去抢别人的卡 |
| `Only test without CUDA` | 真（这条就是测无 CUDA 分支的） | 正常 |
| `Not use cutt, Skip` | **曾经是假绿**，2026-09-06 已修 | 见下，形态会重现 |

**「后端库没人装载」这个形态会重复出现，判据是「谁调用 setup_X」。** cuTT 是它的样本：
9.01 把 `setup_nccl`/`setup_cutt`/`setup_mkl` 三个 import 期调用改成惰性时，给 NCCL 和 MKL
补了惰性调用点（`compat/collectives.py`、`nn/functional/matrix.py`），**漏了 `setup_cutt`**。
后果不是「慢路径」而是「后端不可达」：`cutt_ops` 恒为 `None`，`tests/backends/cuda/test_cutt*.py`
六条用例从来没跑过一次。**恒 skip 的条目等于没有条目**，任何以它为依据的验收都不成立。

查一个后端库是不是活的（**注意 `load=True`**：惰性库直接读属性一定是 `None`，
这正是那六条用例给出假理由的原因）：

```bash
python -c "
import jittor as jt; jt.flags.use_cuda = 1
from jittor._runtime.backend_libraries import get_library_ops
for n in ('cub','cudnn','curand','cublas','cufft','cusparse','cutt','mkl'):
    print(n, get_library_ops(n, load=True))
"
```

配套的两条判据：

- **`grep -rn 'setup_<lib>' | grep -v 'def setup_'` 为空 = 该后端不可达**，与硬件无关。
- **测试文件不要在 import 期读后端属性。** 写成 `setUpClass` 里 `load=True` 取一次、
  取不到就 `raise unittest.SkipTest`；在模块作用域取会让「收集一个文件」变成「编译一个后端」。

**打开一个长期恒 skip 的后端，要预期它带着积攒的坏账。** cuTT 这次一次暴露了三层：
`setup_cutt` 的编译命令缺 `cuda_sdk_flags` 与后端 include（与 4.12 报告里第 2、3 层同因，
只是没人编过所以没人看见）；`cutt_transpose` 的构造函数把「axes 从 0 递增」当成恒等置换而
不比较秩，`[0]` 作用在二维输入上**静默返回原张量**，`infer_shape` 里那两条 `USER_CHECK`
根本不可达（`transpose_op.cc`、`fuse_transpose_op.cc` 同一处写法，同样静默）；测试文件里
`test_matmul_grad` 定义了两次，后一个把前一个盖掉。**恒 skip 的目录第一次跑绿之前，
不要把里面任何一条当成证据。**

# 这台机器上真正缺的硬件

只有这些：**Ascend / CANN / NPU、ROCm、Corex、多机（要两台）**。
CUDA、cuDNN、cuBLAS、cuFFT、cuSPARSE、CUB、NCCL(单机多卡) 全都在。
**不要再写「本机无 CUDA」。**
