# 上机验证：删除 HCCL 集合通信的 4 次同步（8.02 待实机部分）

审计 [06-backends.md §分布式](../design/codebase-audit/06-backends.md) 里严重度为
**关键**的一条：四个 HCCL 集合通信算子各自在调用前后都做
`aclrtSynchronizeDevice()` + `aclrtSynchronizeStream(aclstream)`，**每次通信 4 次全设备
或全流同步**，NPU 多卡训练的流水被彻底打断。

这四次同步**按读代码是多余的**：`aclstream` 是全局唯一的（`extern/acl/acl_jittor.h`），
所有 ACL 算子都往它上面排队，所以集合通信天然排在「算出输入的那些算子」之后、
「消费输出的那些算子」之前。

**但"按读代码"不是"验证过"。** 本仓库所在机器没有昇腾硬件，这段代码连编译都做不到，
更不可能跑一次。所以 8.02 **没有直接删掉**这四次同步，而是把它们收进一个开关：

- `JT_HCCL_COLLECTIVE_SYNC=full`（**默认**）——保持历史行为，与改动前逐字等价。
- `JT_HCCL_COLLECTIVE_SYNC=stream-order`——不做那四次同步，依赖流序。

代码位置：`extern/acl/hccl/inc/hccl_wrapper.h` 的 `hccl_collective_begin()` /
`hccl_collective_end()`，开关读取在 `extern/acl/hccl/src/hccl_wrapper.cc` 的
`hccl_collective_full_sync()`。四个算子（all_reduce / all_gather / reduce / broadcast）
都只调这两个函数，不再各写一遍同步。

**本文档的清单在真机上全绿之后，才可以把默认值改成 `stream-order`，并把开关和它的注释
一起删掉。** 在那之前看板 8.02 的 HCCL 部分保持「待实机」。

## 需要的机器与软件

| 项 | 要求 |
| --- | --- |
| 机型 | 昇腾 **Ascend 910B3**，**至少 2 卡**（集合通信最少两张） |
| SDK | CANN（`aclrtSynchronizeDevice` / `HcclAllReduce` 等来自它），`hccl.h` 与 `acl/acl.h` 必须在 include 路径里 |
| 启动方式 | **env/file rendezvous**（`JT_HCCL_WORLD_SIZE` 等），不要用 mpirun——审计已记录昇腾上 conda-OpenMPI + CANN 会崩 |

环境变量按 rank 取值，**全部从环境读，不要写进脚本**：

    JT_HCCL_WORLD_SIZE=<卡数>
    JT_HCCL_RANK=<本 rank>
    JT_HCCL_LOCAL_RANK=<本 rank 在本机的序号>
    JT_HCCL_ROOTINFO_FILE=$TMPDIR/hccl_root.bin

## 第 0 步：先确认真的在 NPU 上跑，没有静默回落到 CPU

**这一步不能跳，也不能只看"跑绿了"。** 集合通信被跳过、或者算子走了 CPU 分支，
表现都是"一切正常"：不报错、loss 曲线好看，只有模型是错的
（审计 §分布式「分布式初始化失败静默退化为单卡」，严重度关键）。

```bash
PYTHONPATH=<worktree>/python JITTOR_HOME=... TMPDIR=... \
JT_HCCL_WORLD_SIZE=1 JT_HCCL_RANK=0 JT_HCCL_LOCAL_RANK=0 \
JT_HCCL_ROOTINFO_FILE=$TMPDIR/hccl_root.bin \
python -c '
import os, jittor as jt
print("SELFCHECK", os.path.dirname(jt.__file__))     # 必须是你的 worktree
assert jt.compile_extern.hccl_ops is not None, "HCCL 没编出来"
jt.flags.use_acl = 1
print("device count", jt.get_device_count())
'
```

**判据**（四条全部满足才算在 NPU 上）：

1. `SELFCHECK` 打印的是你的 worktree，不是 site-packages 里的 editable 安装目标。
2. `jt.compile_extern.hccl_ops is not None`。为 None 说明 HCCL 没编，后面所有集合通信
   测试会**整文件 skip 而以绿色退出**。
3. `jt.flags.use_acl` 为 1。**不设它，算子走 CPU 分支**，你测的不是 HCCL。
4. **反向确认**：故意把 `JT_HCCL_ROOTINFO_FILE` 指到一个不存在的目录，进程必须**报错退出**，
   不允许打一条警告然后按单卡继续。若它继续了，先修那条静默退化，再谈同步。

`skipped` 不是 `passed`：每次跑完数清 `N passed` 里的 N 是不是你以为的那些用例。

## 第 1 步：A/B 数值对拍（默认 vs 去同步）

同一个用例、同一份输入，只改 `JT_HCCL_COLLECTIVE_SYNC`，跑两遍：

```bash
for MODE in full stream-order; do
  JT_HCCL_COLLECTIVE_SYNC=$MODE \
  PYTHONPATH=<worktree>/python JITTOR_HOME=... TMPDIR=... \
  JT_HCCL_WORLD_SIZE=$N JT_HCCL_RANK=$R JT_HCCL_LOCAL_RANK=$R \
  JT_HCCL_ROOTINFO_FILE=$TMPDIR/hccl_root.bin \
  python -m pytest -q <worktree>/tests/distributed/test_hccl_ops_on_device.py
done
```

（`test_hccl_ops_on_device.py` 还不存在——写它是上机那次的第一件事。口径照
`tests/distributed/test_nccl_comm_stream.py`：四个集合通信各一次 rank 相关数值对拍，
期望值从 rank 推出来，非 root 初值填 `-1`，`reduce` 的非 root 断言「全尺寸 + 清零」。）

**判据**：两种模式**逐元素完全相同**（`assert_array_equal`，不是 `allclose`），
且都等于从 rank 推出来的期望值。

## 第 2 步：压竞态（这是真正要验的东西）

去掉同步之后最容易出的错是**竞态**，而竞态在小张量、单次运行下大概率不复现。
照 `agent/skills/jittor-distributed-verification/SKILL.md` 里
「把集合通信挪到侧流之后：跑绿一次不算数」那一节的第 3 条做，把 NCCL 换成 HCCL：

- 输入**现算**（默认流上一串算子的结果），不要用提前造好的常量；
- 输出**立刻被下游消费**；
- **中途不读回**，每轮只留一个标量误差，最后一起查；
- 期望值同时依赖 rank 和迭代号；
- 至少 200 轮。

**判据**：`stream-order` 模式下 200 轮的最大误差为 `0.0`。跑一次绿不算数，
至少连跑三遍。

## 第 3 步：证明去同步真的改变了时序

只对拍数值证明不了同步被删掉了——**同步还在**的话数值也是对的。要么：

- 用 CANN 的 profiler 取 timeline，确认 `stream-order` 下集合通信与计算的
  区间相交，而 `full` 下不相交（方法与判据同 skill 里的 nsys 那一节，
  「立刻同步那一档并发数必须是 0」这条对应 `full`）；
- 或至少给出端到端计时：`stream-order` 必须显著快于 `full`。删掉 4 次全设备同步
  如果**一点不变快**，说明开关没生效（先检查是不是拼错了值——拼错会 LOGf，
  不会静默按 `full` 走）。

## 还剩一处没动

`hccl_broadcast_op.cc` 里 root 分支在 `aclrtMemcpy`（同步版本）之后还有一次单独的
`aclrtSynchronizeDevice()`。`aclrtMemcpy` 本身就是同步的，所以它也是多余的，但它不属于
「每次通信 4 次同步」那一条，也同样没法在本机验证。上机时一并测掉，或留成新任务。
